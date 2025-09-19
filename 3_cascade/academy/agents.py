import asyncio
from typing import Callable

import numpy as np
from numpy.random import sample
from ase import Atoms
from ase.db import connect
from academy.handle import Handle
from academy.agent import Agent, action
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Dynamics

from cascade.learning.base import BaseLearnableForcefield
from cascade.utils import canonicalize

from base_agents import (
    Auditor, TrainingSampler, TrainingLabeler, ModelTrainer
)

# class CascadseCoordinator(Agent):

#     def __init__(
#         self,
#         dynamics_engine: Handle[DynamicsEngine],
#         auditor: Handle[Auditor],
#         training_sampler: Handle[TrainingSampler],
#     ):
#         pass


class Writer(Agent):
    """some design questions:
    
    * should this thing maintain a connection for its lifetime?
    * how parallel (if at all) should this be?
    * should the chunk index just be in a kwargs? or some dataclass?
    * should it figure out the pass index?
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    @action
    async def write(
        self,
        atoms: list[Atoms],
        db_path: str = None,
        **kwargs
    ):
        """Write chunk of trajectory to a database
        atoms: trajectory to write
        db_path: overrides that passed to initializer
        kwargs: passed to write for each frame
        """

        # handle defaults
        db_path = db_path if db_path is not None else self.db_path
        print('connecting to db')
        with connect(db_path) as db:
            for a in atoms:
                db.write(a, **kwargs)


class DynamicsEngine(Agent):

    def __init__(
        self
    ):
        pass

    @action
    async def advance_dynamics(
        self,
        atoms: Atoms,
        learner: BaseLearnableForcefield,
        model_msg: bytes,
        steps: int,
        calc_factory: Callable[[], Calculator],
        dyn_cls: type[Dynamics],
        dyn_kws: dict[str, object],
        run_kws: dict[str, object] = None
    ) -> list[Atoms]:

        if run_kws is None:
            run_kws = {}
            
        print('making learner')
        calc = learner.make_calculator(model_msg, device='cpu')
        print('done.')
        atoms.calc = calc

        traj = []

        # set up writer
        def write_to_list():
            # needs to be 64 bit for db read
            f = atoms.calc.results['forces']
            atoms.calc.results['forces'] = f.astype(np.float64)
            traj.append(canonicalize(atoms))

        # set up dynamics
        dyn = dyn_cls(
            atoms,
            **dyn_kws
        )
        dyn.attach(write_to_list)
        # run dynamics
        print('running dynamics')
        dyn.run(steps, **run_kws)
        print('done.')
        return traj


class DummyAuditor(Auditor):

    accept_rate = 0.75

    @action
    async def audit_chunk(
        self,
        traj: list[Atoms]
    ) -> tuple[bool, float, Atoms]:
        """a stub of a real audit"""
        good = np.random.random() < self.accept_rate
        score = float(good)
        atoms = traj[-1] if good else traj[0]
        return good, score, atoms


class DummySampler(TrainingSampler):

    @action
    async def sample_frames(
        self,
        atoms: list[Atoms],
        n_frames: int
    ) -> list[Atoms]:
        return sample(atoms, n_frames)


class DummyTrainer(ModelTrainer):

    @action
    async def train_model(
        self, 
        learner: BaseLearnableForcefield
    ) -> bytes:
        calc = mace_mp('small', device='cpu', default_dtype="float32")
        model = calc.models[0]
        model_msg = learner.serialize_model(model)
        return model_msg