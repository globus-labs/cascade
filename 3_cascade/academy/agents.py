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


class DynamicsEngine(Agent):

    def __init__(
        self
    ):
        pass

    @action
    async def advance_dynamics(
        self,
        atoms: Atoms,
        db_path: str,
        traj_i: int,
        learner: BaseLearnableForcefield,
        model_msg: bytes,
        steps: int,
        chunk_i: int,
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

        print('connecting to db')
        with connect(db_path) as db:
            print('done.')

            # delete existing rows for this chunk (we are rerunning)
            print('reading from db')
            rows = db.select(chunk=chunk_i, traj=traj_i)
            print('done.')
            ids = [r.id for r in rows]
            if ids is not None:
                print('deleting old db entries')
                db.delete(ids)
                print('done.')

            # set up writer
            def write_to_db():
                # needs to be 64 bit for db read
                f = atoms.calc.results['forces']
                atoms.calc.results['forces'] = f.astype(np.float64)
                db.write(
                    atoms=atoms,
                    traj=traj_i,
                    chunk=chunk_i,
                )

            # set up dynamics
            dyn = dyn_cls(
                atoms,
                **dyn_kws
            )
            dyn.attach(write_to_db)
            # run dynamics
            print('running dynamics')
            dyn.run(steps, **run_kws)
            print('done.')

            print('reading dynamics results.')
            traj = db.select(traj=traj_i, chunk=chunk_i)
            print('done.')
            traj = [row.toatoms() for row in traj]
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