import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

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


async def main():

    init_logging(logging.DEBUG)

    # args
    init_strc = ['../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp'] * 2

    target_length = 10
    retrain_len = 100000
    n_sample_frames = 1
    accept_rate = 1.
    chunk_size = 10
    learner = MACEInterface()
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    init_weights = learner.serialize_model(model)
    dyn_cls = VelocityVerlet
    dyn_kws = {'timestep': 1*units.fs}
    run_kws = {}

    # intial conditions, trajectories
    initial_specs = []
    trajectories = []
    for i, s in enumerate(init_strc):
        a = read(s, index=-1)
        trajectories.append(
            Trajectory(
                init_atoms=a,
                chunks=[],
                target_length=target_length,
                id=i,
            )
        )
        initial_specs.append(
            AdvanceSpec(
                a,
                traj_id=i,
                chunk_id=0,
                steps=chunk_size
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
                chunk_size,
                retrain_len,
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
                dyn_cls,
                dyn_kws,
                run_kws
            ),
            registration=dynamics_reg
        )
        await manager.launch(
            DummyTrainer,
            registration=trainer_reg
        )
        await manager.launch(
            DummyLabeler,
            args=(db_handle,),
            registration=labeler_reg,
        )
        await manager.launch(
            DummySampler,
            args=(
                n_sample_frames,
                labeler_handle
            ),
            registration=sampler_reg,
        )
        await manager.launch(
            DummyAuditor,
            args=(
                accept_rate,
                sampler_handle,
                db_handle
            ),
            registration=auditor_reg,
        )

        try:
            await manager.wait([db_handle])
        except KeyboardInterrupt:
            for handle in handles:
                await manager.shutdown(handle, blocking=False)

if __name__ == '__main__':
    asyncio.run(main())
