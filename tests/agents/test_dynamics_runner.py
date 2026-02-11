import pathlib
import pytest
from pytest import fixture
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import datetime

from mace.calculators import mace_mp
from ase import Atoms
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from academy.logging import init_logging

from cascade.agents.agents import DynamicsRunner, CascadeAgent
from cascade.learning.mace import MACEInterface
from cascade.agents.db_orm import TrajectoryDB
from cascade.model import AuditStatus, AdvanceSpec
from cascade.agents.task import advance_dynamics


class DeterministicAuditor(CascadeAgent):

    def __init__(self,
                 sequence=None
                 ):
        self.sequence = sequence

    async def audit(self):
        return self.sequence.pop(0)


@fixture
def atoms() -> Atoms:
    return read('../../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp')


@pytest.mark.asyncio
async def test_dynamics_runner(atoms: Atoms):
    start_time = datetime.datetime.utcnow().strftime("%Y.%m.%d-%H:%M:%S")
    run_id = f'test-{start_time}'
    target_length = 10
    chunk_size = 5
    db_url = 'postgresql://ase:pw@localhost:5432/cascade'

    run_dir = pathlib.Path("run") / run_id
    run_dir.mkdir(parents=True)
    learner = MACEInterface()
    init_weights = learner.serialize_model(learner.get_model(mace_mp('small').models[0]))
    logfile = run_dir / "runtime.log"
    logger = init_logging(level='INFO', logfile=logfile)

    # Initialize trajectories in the database
    traj_db = TrajectoryDB(db_url)
    traj_db.create_tables()
    traj_db.initialize_trajectory(
        run_id=run_id,
        traj_id=0,
        target_length=target_length,
        init_atoms=atoms
    )

    # intial conditions, trajectories
    initial_specs = []
    initial_specs.append(
        AdvanceSpec(
            atoms=atoms,
            run_id=run_id,
            traj_id=0,
            chunk_id=0,
            attempt_index=0,
            steps=chunk_size
        )
    )

    # initialize manager, exchange
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=10),
    ) as manager:

    dyn_reg = await manager.register_agent(DynamicsRunner)
    aud_reg = await manager.register_agent(DeterministicAuditor)
    dyn_handle = manager.get_handle(dyn_reg)
    aud_handle = manager.get_handle(aud_reg)

    await manager.launch(
        DynamicsRunner,
        kwargs=dict(
            atoms=atoms,
            run_id=run_id,
            traj_id=0,
            chunk_size=chunk_size,
            n_steps=target_length,
            auditor=aud_handle,
            executor=ProcessPoolExecutor(1),
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
        DeterministicAuditor,
        sequence=[AuditStatus.PASSED, AuditStatus.FAILED, AuditStatus.PASSED],
    )

    await manager.wait([dyn_handle]) # this should wait until it shuts itself down

    # todo: use caplog to assert things happen as expected