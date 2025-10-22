import asyncio
from concurrent.futures import ThreadPoolExecutor

from ase.io import read
from ase import units
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from mace.calculators import mace_mp

from agents import (
    DummyDatabase,
    DynamicsEngine,
    DummyAuditor,
    DummySampler,
    DummyLabeler,
    DummyTrainer
)
from model import Trajectory, AdvanceSpec
from cascade.learning.mace import MACEInterface
from ase.md.verlet import VelocityVerlet


async def main():

    # args
    init_strc = ['../../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp'] * 2

    target_length = 100
    retrain_len = 10
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
        executors=ThreadPoolExecutor(),
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

        # launch all agents
        await manager.launch(
            DummyDatabase,
            args=(
                trajectories,
                retrain_len,
                trainer_handle,
                dynamics_handle
            )
        )
        await manager.launch(
            DynamicsEngine,
            args=(
                auditor_handle,
                learner,
                init_weights,
                dyn_cls,
                dyn_kws,
                run_kws
            )
        )
        await manager.launch(
            DummyTrainer
        )
        await manager.launch(
            DummyLabeler,
            args=(db_handle,)
        )
        await manager.launch(
            DummySampler,
            args=(
                n_sample_frames,
                labeler_handle
            )
        )
        await manager.launch(
            DummyAuditor,
            args=(
                accept_rate,
                sampler_handle,
                db_handle
            )
        )

        # seed with initial conditions
        for spec in initial_specs:
            await dynamics_handle.submit(spec)

        manager.wait(db_reg)

if __name__ == '__main__':
    asyncio.run(main())

