import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import warnings
import datetime
import hashlib
import json
import pathlib
from functools import partial
from typing import Callable

import ase
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase import units
from ase.md.verlet import VelocityVerlet
from mace.calculators import mace_mp
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.usage_tracking.levels import LEVEL_1
from parsl.concurrent import ParslPoolExecutor
from academy.logging.recommended import recommended_logging
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from cascade.agents.agents import (
    DatabaseMonitor,
    DynamicsRunner,
    Auditor,
    Sampler,
    Labeler,
    Trainer,
    Controller,
)
from cascade.agents.config import (
    DatabaseMonitorConfig,
    DynamicsRunnerConfig,
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    TrainerConfig,
    ControllerConfig,
)
from cascade.model import AdvanceSpec, AuditResult, TrainingFrame
from cascade.learning.mace import MACEInterface
from cascade.learning.finetuning import MultiHeadConfig
from cascade.agents.db_orm import TrajectoryDB
from cascade.calculator import get_calc_factory
from cascade.agents.task import (
    random_audit,
    uq_threshold_audit,
    advance_dynamics,
    random_sample,
    max_uq_sample,
    boundary_uq_sample,
    audit_reason_sample,
    label_frame,
    train,
    ensemble_force_deviation_uq,
    max_force_error
)
from cascade.traj_config import (
    InitialTrajConfig,
    load_initial_configs,
    get_dynamics_cls,
    resolve_dyn_kws,
    prepare_atoms_for_dynamics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Max workers in the executor pool'
    )
    parser.add_argument(
        '--init-config-json',
        type=str,
        required=True,
        help='Path to a JSON file describing the initial configuration for each trajectory '
             '(structure path, optional temperature, dynamics integrator + its kwargs)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10,
        help='Initial chunk size'
    )
    parser.add_argument(
        '--gpu-flush-interval',
        type=int,
        default=10,
        help='How often to release the CUDA caching allocator during dynamics'
    )
    parser.add_argument(
        '--target-length',
        type=int,
        default=10,
        help='Target length of the dynamics'
    )
    parser.add_argument(
        '--retrain-len',
        type=int,
        default=100000,
        help='Retrain length'
    )
    parser.add_argument(
        '--retrain-fraction',
        type=float,
        default=0.5,
        help='Fraction of active trajectories that must be sampled from to trigger retraining'
    )
    parser.add_argument(
        '--retrain-min-frames',
        type=int,
        default=10,
        help='Minimum number of frames before fraction-based retraining can trigger'
    ) # todo: can we clarify why this exsits along with retrain-len?
    parser.add_argument('--n-ensemble',
        type=int,
        default=1,
        help='Number of ensemble members for MLFF'
    )
    parser.add_argument(
        '--n-sample-frames',
        type=int,
        default=1,
        help='Number of sample frames'
    )
    parser.add_argument(
        '--burn-in-rounds',
        type=int,
        default=0,
        help='Force-fail (and sample) chunks with model_version below this count, '
             'so the ensemble gets some real disagreement before the audit is load-bearing'
    )
    parser.add_argument(
        '--burn-in-n-frames',
        type=int,
        default=None,
        help='Frames to sample per chunk while still in burn-in (defaults to --n-sample-frames if unset)'
    )
    parser.add_argument(
        '--max-audit-retries',
        type=int,
        default=None,
        help='Max consecutive audit failures a single chunk may accumulate before its trajectory '
             'is marked FAILED and given up on. Unset means retry indefinitely.'
    )
    parser.add_argument(
        '--accept-rate',
        type=float,
        default=1.0,
        help='Accept rate (only used by the "random" audit strategy)'
    )
    parser.add_argument(
        '--audit-task',
        type=str,
        default='random',
        choices=['random', 'uq_threshold'],
        help='Audit strategy: "random" accepts/rejects chunks randomly (--accept-rate); '
             '"uq_threshold" fails chunks whose ensemble force-disagreement exceeds --audit-threshold'
    )
    parser.add_argument(
        '--audit-threshold',
        type=float,
        default=0.1,
        help='Used by the uq_threshold audit strategy if the controller is not being used'
    )
    parser.add_argument(
        '--target-ferr',
        type=float,
        default=None,
        help='Target observed force error for adaptive threshold calibration (Controller agent). '
             'Only used with --audit-task uq_threshold; if unset, the threshold stays fixed '
             'at --audit-threshold for the whole run.'
    )
    parser.add_argument(
        '--calibration-history-length',
        type=int,
        default=8,
        help='Number of same-model-version labeled frames required before (re)calibrating the threshold'
    )
    parser.add_argument(
        '--per-trajectory-threshold',
        type=int,
        default=1,
        help='Calibrate each trajectory\'s UQ threshold independently from only its own '
             'labeled-frame history, instead of pooling all trajectories into one shared threshold. '
             'Only used with --audit-task uq_threshold and --target-ferr set.'
    )
    parser.add_argument(
        '--audit-random-fail-rate',
        type=float,
        default=0.0,
        help='Frequency at which a chunk that would otherwise pass audit is randomly failed anyway, '
             'independent of the active audit strategy (forces continued sampling/exploration)'
    )
    parser.add_argument(
        '--sample-task',
        type=str,
        default='random',
        choices=['random', 'max_uq', 'boundary', 'audit_reason'],
        help='Sampling strategy for picking training frames out of a failed chunk: "random" (current '
             'default), "max_uq" (highest per-frame UQ), "boundary" (frames around the first frame '
             'crossing threshold), or "audit_reason" (routes to one of the three per why the chunk '
             'failed audit -- see --burn-in-sample-task/--threshold-sample-task/--random-fail-sample-task)'
    )
    parser.add_argument(
        '--burn-in-sample-task',
        type=str,
        default='random',
        choices=['random', 'max_uq', 'boundary'],
        help='Only used with --sample-task audit_reason: strategy for chunks failed by the burn_in '
             'model-version gate, where UQ may not be calibrated yet'
    )
    parser.add_argument(
        '--threshold-sample-task',
        type=str,
        default='boundary',
        choices=['random', 'max_uq', 'boundary'],
        help='Only used with --sample-task audit_reason: strategy for chunks failed by a genuine '
             'UQ threshold crossing'
    )
    parser.add_argument(
        '--random-fail-sample-task',
        type=str,
        default='max_uq',
        choices=['random', 'max_uq', 'boundary'],
        help='Only used with --sample-task audit_reason: strategy for chunks failed by '
             '--audit-random-fail-rate, which have no real crossing to anchor on'
    )
    parser.add_argument(
        '--learner',
        type=str,
        default='mace',
        help='Learner to use'
    )
    parser.add_argument(
        '--calc-type',
        type=str,
        choices=['mace', 'fairchem'],
        default='mace',
        help='Which reference calculator family the Labeler uses to compute ground-truth '
             'energies/forces/stress for sampled frames'
    )
    parser.add_argument(
        '--calc-model',
        type=str,
        default='medium',
        help='For --calc-type=mace, a MACE-MP model size (e.g. "medium") or path to a MACE '
             'checkpoint. For --calc-type=fairchem, the path to a FairChem .pt checkpoint.'
    )
    parser.add_argument(
        '--calc-task',
        type=str,
        default=None,
        help='FairChem task name selecting the model head (e.g. "omol", "omat", "oc20", '
             '"odac", "omc"), ignored for --calc-type=mace. Only needed for --calc-type=fairchem '
             'if the checkpoint supports more than one task; single-task checkpoints infer it '
             'automatically.'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.environ.get('CASCADE_DB_URL'),
        help='Database URL, e.g. postgresql://ase:pw@<host>:5432/cascade '
             '(defaults to the CASCADE_DB_URL env var)'
    )
    parser.add_argument(
        '--device-dyn',
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--device-label',
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--device-train',
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Number of epochs per training round',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for training',
    )
    parser.add_argument('--replay-dataset', default=None, help='Path to an ASE database containing data to replay during finetuning')
    parser.add_argument('--replay-downselect', default=None, type=int, help='Max number of entries to use from replay dataset')
    parser.add_argument('--replay-frequency', default=1, type=int, help='How often to replay')
    parser.add_argument('--replay-lr-reduction', default=1, type=float, help='Factor by which to reduce LR during replay')
    parser.add_argument('--replay-batch-size', default=None, type=int, help='Batch size used during replay')
    args = parser.parse_args()

    args.per_trajectory_threshold = bool(args.per_trajectory_threshold)

    return args


def get_learner(learner_name: str) -> type[ase.calculators.calculator.Calculator]:
    if learner_name == 'mace':
        return MACEInterface()
    else:
        raise ValueError(f'Unknown learner: {learner_name}')


def get_audit_task(audit_task_name: str) -> Callable[..., AuditResult]:
    if audit_task_name == 'random':
        return random_audit
    elif audit_task_name == 'uq_threshold':
        return uq_threshold_audit
    else:
        raise ValueError(f'Unknown audit task: {audit_task_name}')


def get_sample_task(sample_task_name: str) -> Callable[..., list[TrainingFrame]]:
    if sample_task_name == 'random':
        return random_sample
    elif sample_task_name == 'max_uq':
        return max_uq_sample
    elif sample_task_name == 'boundary':
        return boundary_uq_sample
    else:
        raise ValueError(f'Unknown sample task: {sample_task_name}')


async def main():
    # parse arguments
    args = parse_args()

    # Set up run directory, 
    params = args.__dict__.copy()
    start_time = datetime.datetime.utcnow().strftime("%Y.%m.%d-%H:%M:%S")
    params_hash = hashlib.sha256(json.dumps(params).encode()).hexdigest()[:6]
    run_id = f"{start_time}-{params_hash}"
    run_dir = pathlib.Path("run") / (
        f"run-{run_id}"
    )
    run_dir.mkdir(parents=True)

    # Save the run parameters to disk
    (run_dir / "params.json").write_text(json.dumps(params))
    logfile = run_dir / "runtime.log"

    # read in initial model
    learner = get_learner(args.learner)
    init_weights = learner.serialize_model(learner.get_model(mace_mp('small').models[0]))
    init_ensemble_weights = [init_weights] * args.n_ensemble

    # initialize database
    traj_db = TrajectoryDB(args.db_url)
    traj_db.create_tables()

    # set up multi-head replay, if requested
    if args.replay_dataset is not None:
        replay = MultiHeadConfig(
            original_dataset=read(args.replay_dataset, slice(None)),
            num_downselect=args.replay_downselect,
            epoch_frequency=args.replay_frequency,
            lr_reduction=args.replay_lr_reduction,
            batch_size=args.replay_batch_size,
        )
    else:
        replay = None

    # read initial configuration for each trajectory
    init_configs = load_initial_configs(args.init_config_json)
    initial_specs = []
    for i, cfg in enumerate(init_configs):
        a = read(cfg.path, index=-1)
        #logger.info(f"Initializing traj {i} with {len(a)} atoms")

        a = prepare_atoms_for_dynamics(a, cfg)

        if cfg.temperature_K is not None:
            MaxwellBoltzmannDistribution(a, temperature_K=cfg.temperature_K)

        # create trajectory entry in the database
        traj_db.initialize_trajectory(
            run_id=run_id,
            traj_id=i,
            target_length=args.target_length,
            init_atoms=a
        )
        # create advance specification for dynamics engine
        initial_specs.append(
            AdvanceSpec(
                atoms=a,
                run_id=run_id,
                traj_id=i,
                chunk_id=0,
                attempt_index=0,
                steps=args.chunk_size
            )
        )

    # set up parsl
    # a chunk can only be in one worker at a time + training happens concurrently
    # note that this is really too many workers since at least one agent is waiting for
    # a new model while training is happening. can possibly do some math based on the retrain
    # logic to figure out the real max number of used workers
    # but this may not make as much sense once we distribute the workflow, so no worries for now
    n_parsl_workers = args.max_workers or len(initial_specs) + args.n_ensemble
    # only meaningful alongside the uq_threshold audit strategy, which is the
    # only audit_task that reads a 'threshold' kwarg
    use_controller = args.audit_task == 'uq_threshold' and args.target_ferr is not None
    n_agents = len(initial_specs) + 5 + (1 if use_controller else 0)  # one dynamics runner per traj and one of each other agent
    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex_local",
                max_workers_per_node=n_parsl_workers,
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
        usage_tracking=LEVEL_1,
    )

    with ParslPoolExecutor(config=config) as pool:
        async with await Manager.from_exchange_factory(
            factory=LocalExchangeFactory(),
            executors=ThreadPoolExecutor(max_workers=n_agents),
            log_config=recommended_logging(level=args.log_level, logfile=logfile)
        ) as manager:

            # register all agents with manager
            db_reg = await manager.register_agent(DatabaseMonitor)
            trainer_reg = await manager.register_agent(Trainer)
            labeler_reg = await manager.register_agent(Labeler)
            sampler_reg = await manager.register_agent(Sampler)
            auditor_reg = await manager.register_agent(Auditor)
            controller_reg = await manager.register_agent(Controller) if use_controller else None

            # get handles to all agents
            db_handle = manager.get_handle(db_reg)
            trainer_handle = manager.get_handle(trainer_reg)
            labeler_handle = manager.get_handle(labeler_reg)
            sampler_handle = manager.get_handle(sampler_reg)
            auditor_handle = manager.get_handle(auditor_reg)
            controller_handle = manager.get_handle(controller_reg) if controller_reg is not None else None

            # these are used for cleanup
            handles = [
                db_handle,
                trainer_handle,
                labeler_handle,
                sampler_handle,
                auditor_handle,
            ]
            if controller_handle is not None:
                handles.append(controller_handle)

            # set up agent configs
            db_monitor_config = DatabaseMonitorConfig(
                run_id=run_id,
                db_url=args.db_url,
                retrain_len=args.retrain_len,
                retrain_fraction=args.retrain_fraction,
            )
            if args.audit_task == 'random':
                audit_kws = dict(accept_prob=args.accept_rate)
            else:
                audit_kws = dict(threshold=args.audit_threshold, burn_in_model_versions=args.burn_in_rounds)
            auditor_config = AuditorConfig(
                audit_task=get_audit_task(args.audit_task),
                executor=pool,
                run_id=run_id,
                db_url=args.db_url,
                audit_kws=audit_kws,
                random_fail_rate=args.audit_random_fail_rate,
            )
            if args.sample_task == 'audit_reason':
                sample_task = partial(
                    audit_reason_sample,
                    burn_in_sampler=get_sample_task(args.burn_in_sample_task),
                    threshold_sampler=get_sample_task(args.threshold_sample_task),
                    random_sampler=get_sample_task(args.random_fail_sample_task),
                )
            else:
                sample_task = get_sample_task(args.sample_task)
            sampler_config = SamplerConfig(
                run_id=run_id,
                db_url=args.db_url,
                n_frames=args.n_sample_frames,
                executor=pool,
                sample_task=sample_task,
                burn_in_model_versions=args.burn_in_rounds,
                burn_in_n_frames=args.burn_in_n_frames,
            )
            labeler_config = LabelerConfig(
                run_id=run_id,
                db_url=args.db_url,
                executor=pool,
                label_task=label_frame,
                calc_factory=get_calc_factory(args.calc_type, args.calc_model, args.device_label, args.calc_task),
                error_fn=max_force_error,
                )
            if use_controller:
                controller_config = ControllerConfig(
                    run_id=run_id,
                    db_url=args.db_url,
                    target_ferr=args.target_ferr,
                    history_length=args.calibration_history_length,
                    per_trajectory_threshold=args.per_trajectory_threshold,
                    burn_in_model_versions=args.burn_in_rounds,
                )
            trainer_config = TrainerConfig(
                run_id=run_id,
                db_url=args.db_url,
                weights=init_ensemble_weights,
                executor=pool,
                training_task=train,
                training_args=(),
                training_kws=dict(
                    num_epochs=args.num_epochs,
                    device=args.device_train,
                    batch_size=args.batch_size,
                ),
                learner=learner,
                replay=replay,
            )

            # launch all agents
            await manager.launch(
                Auditor,
                kwargs=dict(
                    config=auditor_config,
                    sampler=sampler_handle,
                ),
                registration=auditor_reg,
            )
            await manager.launch(
                Sampler,
                kwargs=dict(
                    config=sampler_config,
                    labeler=labeler_handle,
                ),
                registration=sampler_reg,
            )
            if use_controller:
                await manager.launch(
                    Controller,
                    kwargs=dict(
                        config=controller_config,
                        auditor=auditor_handle,
                    ),
                    registration=controller_reg,
                )
            await manager.launch(
                Labeler,
                kwargs=dict(config=labeler_config, controller=controller_handle),
                registration=labeler_reg,
            )
            await manager.launch(
                Trainer,
                kwargs=dict(config=trainer_config),
                registration=trainer_reg
            )

            # launch one DynamicsRunner per trajectory, accumulating the handles
            dyn_handles = []
            for spec, cfg in zip(initial_specs, init_configs):
                reg = await manager.register_agent(DynamicsRunner)
                handle = manager.get_handle(reg)
                handles.append(handle)
                dyn_handles.append(handle)
                dyn_config = DynamicsRunnerConfig(
                        atoms=spec.atoms,
                        run_id=run_id,
                        db_url=args.db_url,
                        traj_id=spec.traj_id,
                        chunk_size=args.chunk_size,
                        n_steps=args.target_length,
                        executor=pool,
                        advance_dynamics_task=advance_dynamics,
                        learner=learner,
                        run_dir=run_dir,
                        weights=init_ensemble_weights,
                        device=args.device_dyn,
                        dyn_cls=get_dynamics_cls(cfg.dyn_cls),
                        dyn_kws=resolve_dyn_kws(cfg),
                        run_kws=cfg.run_kws,
                        model_version=0,
                        uq_hook=ensemble_force_deviation_uq,
                        gpu_flush_interval=args.gpu_flush_interval,
                        max_audit_retries=args.max_audit_retries,
                )
                await manager.launch(
                    DynamicsRunner,
                    kwargs=dict(
                        config=dyn_config,
                        auditor=auditor_handle
                    ),
                    registration=reg
                )

            # launch the DatabaseMonitor (needs dynamics runners)
            await manager.launch(
                DatabaseMonitor,
                kwargs=dict(
                    config=db_monitor_config,
                    trainer=trainer_handle,
                    dynamics_runners=dyn_handles
                ),
                registration=db_reg,
            )
            # wait for the run to finish!
            try:
                await manager.wait([db_handle])
            except KeyboardInterrupt:
                # attempt graceful shutdown on keyboard interrupt
                for handle in handles:
                    await manager.shutdown(handle, blocking=False)

if __name__ == '__main__':
    asyncio.run(main())
