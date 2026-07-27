import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
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
from mace.calculators import mace_mp
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.usage_tracking.levels import LEVEL_1
from parsl.concurrent import ParslPoolExecutor
from academy.logging import init_logging
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from cascade.agents.agents import (
    DatabaseMonitor,
    DynamicsRunner,
    Auditor,
    Sampler,
    Labeler,
    Trainer
)
from cascade.agents.config import (
    DatabaseMonitorConfig,
    DynamicsRunnerConfig,
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    TrainerConfig
)
from cascade.model import AdvanceSpec, AuditResult
from cascade.learning.mace import MACEInterface
from cascade.learning.finetuning import MultiHeadConfig
from cascade.agents.db_orm import TrajectoryDB
from cascade.agents.task import (
    random_audit,
    uq_threshold_audit,
    advance_dynamics,
    random_sample,
    label_frame,
    train,
    ensemble_force_deviation_uq
)
from cascade.traj_config import (
    InitialTrajConfig,
    load_initial_configs,
    get_dynamics_cls,
    resolve_dyn_kws,
    prepare_atoms_for_dynamics,
)


# Suppress FutureWarning about torch.load weights_only parameter from MACE
warnings.filterwarnings("ignore", category=FutureWarning, module="mace.calculators")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging levl'
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
        help='UQ threshold for the "uq_threshold" audit strategy'
    )
    parser.add_argument(
        '--learner',
        type=str,
        default='mace',
        help='Learner to use'
    )
    parser.add_argument(
        '--calc',
        type=str,
        default='mace',
        help='Calculator to use'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default='postgresql://ase:pw@localhost:5432/cascade',
        help='Database URL'
    )
    parser.add_argument('--replay-dataset', default=None, help='Path to an ASE database containing data to replay during finetuning')
    parser.add_argument('--replay-downselect', default=None, type=int, help='Max number of entries to use from replay dataset')
    parser.add_argument('--replay-frequency', default=1, type=int, help='How often to replay')
    parser.add_argument('--replay-lr-reduction', default=1, type=float, help='Factor by which to reduce LR during replay')
    parser.add_argument('--replay-batch-size', default=None, type=int, help='Batch size used during replay')
    args = parser.parse_args()

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

    # set up logging
    logger = init_logging(level=args.log_level, logfile=logfile)
    logger.setLevel(logging.DEBUG)
    logger.info("Loaded run params")
    logger.info(f'Running job in {run_dir}')
    # separate parsle logging
    parsl_logger = logging.getLogger('parsl')
    for handler in parsl_logger.handlers[:]:  # Iterate over a copy of the list
        parsl_logger.removeHandler(handler)
    parsl_logger.addHandler(logging.FileHandler(run_dir / 'parsl.log'))

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
        logger.info(f"Initializing traj {i} with {len(a)} atoms")

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
    n_parsl_workers = len(initial_specs) + args.n_ensemble
    n_agents = len(initial_specs) + 5 # one dynamics runner per traj and one of each other agent
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
        ) as manager:

            # register all agents with manager
            db_reg = await manager.register_agent(DatabaseMonitor)
            trainer_reg = await manager.register_agent(Trainer)
            labeler_reg = await manager.register_agent(Labeler)
            sampler_reg = await manager.register_agent(Sampler)
            auditor_reg = await manager.register_agent(Auditor)

            # get handles to all agents
            db_handle = manager.get_handle(db_reg)
            trainer_handle = manager.get_handle(trainer_reg)
            labeler_handle = manager.get_handle(labeler_reg)
            sampler_handle = manager.get_handle(sampler_reg)
            auditor_handle = manager.get_handle(auditor_reg)

            # these are used for cleanup
            handles = [
                db_handle,
                trainer_handle,
                labeler_handle,
                sampler_handle,
                auditor_handle,
            ]

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
            )
            sampler_config = SamplerConfig(
                run_id=run_id,
                db_url=args.db_url,
                n_frames=args.n_sample_frames,
                executor=pool,
                sample_task=random_sample,
                burn_in_model_versions=args.burn_in_rounds,
                burn_in_n_frames=args.burn_in_n_frames,
            )
            labeler_config = LabelerConfig(
                run_id=run_id,
                db_url=args.db_url,
                executor=pool,
                label_task=label_frame,
                calc_factory=partial(mace_mp, model='medium', device='cpu', default_dtype="float32"),
                )
            trainer_config = TrainerConfig(
                run_id=run_id,
                db_url=args.db_url,
                weights=init_ensemble_weights,
                executor=pool,
                training_task=train,
                training_args=(),
                training_kws=dict(
                    num_epochs=10,
                    device='cpu',
                    batch_size=2,
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
            await manager.launch(
                Labeler,
                kwargs=dict(config=labeler_config),
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
                        dyn_cls=get_dynamics_cls(cfg.dyn_cls),
                        dyn_kws=resolve_dyn_kws(cfg),
                        run_kws=cfg.run_kws,
                        device='cpu',
                        model_version=0,
                        uq_hook=ensemble_force_deviation_uq,
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
