import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import warnings
import datetime
import hashlib
import json
import pathlib

import ase
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet

# Suppress FutureWarning about torch.load weights_only parameter from MACE
warnings.filterwarnings("ignore", category=FutureWarning, module="mace.calculators")

from mace.calculators import mace_mp
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from academy.logging import init_logging

from cascade.agents.agents import (
    DatabaseMonitor,
    DynamicsRunner,
    Auditor,
    Sampler,
    DummyLabeler,
    DummyTrainer
)
from cascade.agents.config import (
    DatabaseConfig,
    DatabaseMonitorConfig,
    DynamicsRunnerConfig,
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    TrainerConfig
)
from cascade.model import AdvanceSpec
from cascade.learning.mace import MACEInterface
from cascade.agents.db_orm import TrajectoryDB
from cascade.agents.task import random_audit, advance_dynamics, random_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging levl'
    )
    parser.add_argument(
        '--initial-structures',
        nargs='+',
        help='Initial structures to start dynamics'
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
    parser.add_argument(
        '--n-sample-frames',
        type=int,
        default=1,
        help='Number of sample frames'
    )
    parser.add_argument(
        '--accept-rate',
        type=float,
        default=1.0,
        help='Accept rate'
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
        '--dyn-cls',
        type=str,
        default='velocity-verlet',
        help='Dynamics class to use'
    )
    parser.add_argument(
        '--dt_fs',
        type=float,
        default=1.0,
        help='Time step in fs'
    )
    parser.add_argument(
        '--loginterval',
        type=int,
        default=1,
        help='Log interval in steps'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default='postgresql://ase:pw@localhost:5432/cascade',
        help='Database URL'
    )
    args = parser.parse_args()

    return args

def get_learner(learner_name: str) -> type[ase.calculators.calculator.Calculator]:
    if learner_name == 'mace':
        return MACEInterface()
    else:
        raise ValueError(f'Unknown learner: {learner_name}')

def get_dynamics_cls(cls_name: str) -> type[ase.md.md.MolecularDynamics]:
    if cls_name == 'velocity-verlet':
        return VelocityVerlet
    else:
        raise ValueError(f'Unknown dynamics class: {cls_name}')


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

    logger = init_logging(level=args.log_level, logfile=logfile)
    logger.info("Loaded run params")
    logger.info("Created run directory: %s", run_dir)
    
    init_strc = args.initial_structures
    learner = get_learner(args.learner)
    init_weights = learner.serialize_model(learner.get_model(mace_mp('small').models[0]))

    # Initialize trajectories in the database
    traj_db = TrajectoryDB(args.db_url)
    traj_db.create_tables()
    for i, s in enumerate(init_strc):
        a = read(s, index=-1)
        logger.info(f"Initializing traj {i} with {len(a)} atoms")
        success = traj_db.initialize_trajectory(
            run_id=run_id,
            traj_id=i,
            target_length=args.target_length,
            init_atoms=a
        )
        if not success:
            logger.error(f"Failed to initialize traj {i} in database")

    # intial conditions, trajectories
    initial_specs = []
    for i, s in enumerate(init_strc):
        a = read(s, index=-1)
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

    # initialize manager, exchange
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=10),
    ) as manager:

        # register all agents with manager
        db_reg = await manager.register_agent(DatabaseMonitor)
        trainer_reg = await manager.register_agent(DummyTrainer)
        labeler_reg = await manager.register_agent(DummyLabeler)
        sampler_reg = await manager.register_agent(Sampler)
        auditor_reg = await manager.register_agent(Auditor)
        dynamics_reg = await manager.register_agent(DynamicsRunner)

        # get handles to all agents
        db_handle = manager.get_handle(db_reg)
        trainer_handle = manager.get_handle(trainer_reg)
        labeler_handle = manager.get_handle(labeler_reg)
        sampler_handle = manager.get_handle(sampler_reg)
        auditor_handle = manager.get_handle(auditor_reg)
        dynamics_handle = manager.get_handle(dynamics_reg)

        handles = [
            db_handle,
            trainer_handle,
            labeler_handle,
            sampler_handle,
            auditor_handle,
            dynamics_handle
        ]

        sampler_config = SamplerConfig(
            run_id=run_id,
            db_url=args.db_url,
            n_frames=args.n_sample_frames
        )
        labeler_config = LabelerConfig(run_id=run_id, db_url=args.db_url)
        trainer_config = TrainerConfig(run_id=run_id, db_url=args.db_url, learner=learner)

        # launch all agents
        await manager.launch(
            DatabaseMonitor,
            kwargs=dict(
                run_id=run_id,
                db_url=args.db_url,
                retrain_len=args.retrain_len,
                target_length=args.target_length,
                chunk_size=args.chunk_size,
                retrain_fraction=args.retrain_fraction,
                retrain_min_frames=args.retrain_min_frames,
                trainer=trainer_handle,
                dynamics_engine=dynamics_handle,
            ),
            registration=db_reg,
        )
        await manager.launch(
            DynamicsRunner,
            kwargs=dict(
                atoms=atoms,
                run_id=run_id,
                traj_id=0,
                chunk_size=chunk_size,
                n_steps=target_length,
                auditor=aud_handle,
                executor=ProcessPoolExecutor(max_workers=10),
                advance_dynamics_task=advance_dynamics,
                learner=learner,
                weights=init_weights,
                dyn_cls=VelocityVerlet,
                dyn_kws={'timestep': 1 * units.fs},
                run_kws={},
                device='cpu',
                model_version=0
            ),
            registration=dyn_reg
        )
        await manager.launch(
            Auditor,
            kwargs=dict(
                sampler=sampler_handle,
                dynamics=dynamics_handle,
                audit_task=random_audit,
                executor=ProcessPoolExecutor(max_workers=10),
                run_id=run_id,
                db_url=args.db_url,
                audit_kwargs=dict(accept_rate=args.accept_rate,),
                chunk_size=args.chunk_size,
            ),
            registrations=auditor_reg,
        )
        await manager.launch(
            Sampler,
            kwargs=dict(
                run_id=run_id,
                db_url=db_url,
                n_frames=n_frames,
                labeler=labeler_handle,
                executor=ProcessPoolExecutor(max_workers=10),
                sample_task=random_sample,
            ),
            registration=sampler_reg,
        )
        await manager.launch(
            DummyLabeler,
            kwargs=dict(run_id=run_id, db_url=args.db_url),
            registration=labeler_reg,
        )
        await manager.launch(
            DummyTrainer,
            kwargs=dict(run_id=run_id, db_url=args.db_url, learner=learner),
            registration=trainer_reg
        )        

        # submit initial specs to dynamics engine
        for spec in initial_specs:
            logger.info(f"Submitting initial spec to dynamics engine: {spec}")
            await dynamics_handle.submit(spec)
        try:
            await manager.wait([db_handle])
        except KeyboardInterrupt:
            for handle in handles:
                await manager.shutdown(handle, blocking=False)

if __name__ == '__main__':
    asyncio.run(main())
