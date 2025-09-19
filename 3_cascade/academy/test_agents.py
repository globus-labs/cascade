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
    model_msg
):

    # first create a traj to write
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

    writer = Writer(db_path='foo.db')
    await writer.write(traj, traj_i=0, chunk_i=0)
    with connect('foo.db') as db:
        assert db.count() == 11
