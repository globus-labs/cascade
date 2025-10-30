import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging

import ase
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from mace.calculators import mace_mp
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from academy.logging import init_logging

from cascade.agents.dummy import (
    DummyDatabase,
    DynamicsEngine,
    DummyAuditor,
    DummySampler,
    DummyLabeler,
    DummyTrainer
)
from cascade.model import Trajectory, AdvanceSpec
from cascade.learning.mace import MACEInterface


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

    init_logging(logging.DEBUG)

    # args
    args = parse_args()
    init_strc = args.initial_structures
    learner = get_learner(args.learner)
    init_weights = learner.serialize_model(learner.get_model(mace_mp('small').models[0]))

    # intial conditions, trajectories
    initial_specs = []
    trajectories = []
    for i, s in enumerate(init_strc):
        a = read(s, index=-1)
        trajectories.append(
            Trajectory(
                init_atoms=a,
                chunks=[],
                target_length=args.target_length,
                id=i,
            )
        )
        initial_specs.append(
            AdvanceSpec(
                a,
                traj_id=i,
                chunk_id=0,
                steps=args.chunk_size
            )
        )

    # initialize manager, exhcange
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=10),
    ) as manager:

        # register all agents with manager
        db_reg = await manager.register_agent(DummyDatabase)
        trainer_reg = await manager.register_agent(DummyTrainer)
        labeler_reg = await manager.register_agent(DummyLabeler)
        sampler_reg = await manager.register_agent(DummySampler)
        auditor_reg = await manager.register_agent(DummyAuditor)
        dynamics_reg = await manager.register_agent(DynamicsEngine)

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

        # launch all agents
        await manager.launch(
            DummyDatabase,
            args=(
                trajectories,
                args.chunk_size,
                args.retrain_len,
                trainer_handle,
                dynamics_handle,
            ),
            registration=db_reg
        )
        await manager.launch(
            DynamicsEngine,
            args=(
                initial_specs,
                auditor_handle,
                learner,
                init_weights,
                get_dynamics_cls(args.dyn_cls),
                {'timestep': args.dt_fs * units.fs, 'loginterval': args.loginterval},
                {},
            ),
            registration=dynamics_reg
        )
        await manager.launch(
            DummyAuditor,
            args=(
                args.accept_rate,
                sampler_handle,
                db_handle
            ),
            registration=auditor_reg,
        )
        await manager.launch(
            DummySampler,
            args=(
                args.n_sample_frames,
                labeler_handle
            ),
            registration=sampler_reg,
        )
        await manager.launch(
            DummyLabeler,
            args=(db_handle,),
            registration=labeler_reg,
        )
        await manager.launch(
            DummyTrainer,
            registration=trainer_reg
        )        

        try:
            await manager.wait([db_handle])
        except KeyboardInterrupt:
            for handle in handles:
                await manager.shutdown(handle, blocking=False)

if __name__ == '__main__':
    asyncio.run(main())
