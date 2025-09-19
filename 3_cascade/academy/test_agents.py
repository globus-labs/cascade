import asyncio
import pytest

from ase.io import read
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from mace.calculators import mace_mp
import numpy as np
from ase.db import connect
from ase import units

from agents import (
    DynamicsEngine,
    DummyAuditor,
    DummySampler,
    DummyTrainer,
    Writer
)

from pytest import fixture

from cascade.learning.mace import MACEInterface

@fixture()
def atoms() -> Atoms:
    return read('../../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp', index=-1)


learner = MACEInterface()

@fixture
def model_msg():
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)
    return model_msg


@pytest.mark.asyncio
async def test_dynamics_engine(
    atoms,
    model_msg
):
    engine = DynamicsEngine()

    traj = await engine.advance_dynamics(
        atoms,
        learner=learner,
        model_msg=model_msg,
        steps=10,
        calc_factory=mace_mp,
        dyn_cls=VelocityVerlet,
        dyn_kws={'timestep': 1*units.fs},
    )

    assert len(traj) == 11, "Didn't run correct number of steps"
    for a in traj[1:]:
        assert isinstance(a, Atoms)
        mad = np.sum(np.abs(a.get_positions() - traj[0].get_positions()))
        assert mad > 0, "Positions didn't change at all"


@pytest.mark.asyncio
async def test_writer(
    atoms,
    model_msg,
    tmp_path
):

    # create temporary db
    db_path = tmp_path / 'test.db'
    db_str = str(db_path)

    # create a traj to write
    engine = DynamicsEngine()
    traj = await engine.advance_dynamics(
        atoms,
        learner=learner,
        model_msg=model_msg,
        steps=10,
        calc_factory=mace_mp,
        dyn_cls=VelocityVerlet,
        dyn_kws={'timestep': 1*units.fs},
    )

    # write and assure we have the correct results
    writer = Writer(db_path=db_str)
    await writer.write(traj, traj_i=0, chunk_i=0)
    with connect(db_str) as db:
        assert db.count() == 11

@pytest.mark.asyncio
async def test_dummy_auditor(
    atoms,
):
    traj = [atoms.copy() for i in range(10)]
    auditor = DummyAuditor()
    good, score, frame = await auditor.audit_chunk(traj)
    assert isinstance(good, bool)
    assert isinstance(score, float)
    assert isinstance(atoms, Atoms)

