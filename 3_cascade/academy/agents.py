"""WIP agentic implementation of cascasde with academy

Everything here is a stub

In addition to de-stubbing, we have the following todos:
* thread/process safety for queues, model updates
* rewrite for concurrent, high performance DB backend
* data model improvements
* add parallelism within agents with parsl
* make configuration of agents flexible, updatable (i.e., control parameters)
* logging
"""

import asyncio
from typing import Callable
from collections import defaultdict
from asyncio import Queue

import numpy as np
from numpy.random import sample
from ase import Atoms
from ase.db import connect
from academy.handle import Handle
from academy.agent import Agent, action, loop
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Dynamics
from mace.calculators import mace_mp

from cascade.learning.base import BaseLearnableForcefield
from cascade.utils import canonicalize
from model import AuditStatus

from base_agents import (
    Auditor, TrainingSampler, TrainingLabeler, ModelTrainer, Database
)


class DummyDatabase(Database):
    """Not a real database, just something that will work for now"""

    def __init__(
        self, 
        trajectories: list[Trajectories],
        retrain_len: int,
        trainer: Handle[ModelTrainer],
        dynamics_engine: Handle[DynamicsEngine]
    ):

        self.trajectories = trajectories
        self.training_frames = []
        self.sampled_trajectories = set()
        self.retrain_len = retrain_len
        self.trainer = trainer
        self.dynamics_engine = dynamics_engine

    @property
    def all_done(self):
        return all(t.done for t in self.trajectories)

    @action
    async def write_chunk(self, chunk: TrajectoryChunk):
        id = chunk.traj_id
        self.trajectories[traj_id].add_chunk(chunk)

    @action
    async def write_training_frame(self, frame: Atoms):
        # check here if we have exceeded the retrain length, then set event
        self.training_frames.append(frame)
    
    @action
    async def mark_sampled(self, traj_id: int):
        self.sampled_trajectories.add(traj_id)

    @loop
    async def periodic_retrain(self, shutdown: asyncio.Event):
        # run this thing on event (see the sc25 example)
        """Trigger a retrain event and add the right atoms back to the dynamics queue"""
        while not shutdown.is_set():
            if len(training_frames) <= self.retrain_len:
                continue # sleep?
            
            # retrain model and update weights in dynamics engine
            weights = await self.trainer.retrain()
            await self.dynamics_engine.receive_weights(weights)

            # put the sampled trajectories back in the dynamics queue
            for traj_id in self.sampled_trajectories:
                atoms = self.trajecotires[traj_id].chunks[-1].atoms[-1]
                await self.dymamics_engine.submit(atoms)

            # clear training data
            self.training_frames = []
            self.sampled_frames = set()
                

class DynamicsEngine(Agent):

    def __init__(
        self,
        auditor: Handle["Auditor"],
        learner: BaseLearnableForcefield,
        weights: bytes,
        dyn_cls: type[Dynamics],
        dyn_kws: dict[str, object],
        run_kws: dict[str, object],
        device: str = 'cpu'
    ):

        self.learner = learner
        self.weights = weights
        self.dyn_cls = dyn_cls
        self.device = device
        self.dyn_kws = dyn_kws
        self.run_kws = run_kws

        self.model_version = 0  #todo: mt.2025.10.20 probably this should be persisted somewhere else

        self.queue = Queue()


    @action
    async def receive_weights(self, weights:bytes) -> None:
        self.weights = weights
        self.model_version += 1
    
    @action
    async def submit(chunk: TrajectoryChunk):
        self.queue.put(chunk)

    @loop
    async def advance_dynamics(
        self,
        shutdown: asyncio.Event
    ) -> None:
    """Advance dynamics while there are trajectoires to advance


    questions:
        * should we have an update model method that locks?
    """

        while not shutdown.is_set():

            try:
                old_chunk = self.queue.get()
            except Queue.Empty:
                continue

            atoms = old_chunk.atoms[-1]
 
            print('making learner')
            calc = learner.make_calculator(self.weights, device=self.device)
            print('done.')
            atoms.calc = calc


            chunk = TrajectoryChunk(
                atoms=[],
                model_version = self.model_version,
                traj_id=old_chunk.traj_id,
                chunk_id=old_chunk.chunk_id+1
            )

            # set up dynamics
            dyn_kws = self.dyn_kws or {}
            run_kws = self.run_kws or {}
            dyn = self.dyn_cls(
                atoms,
                **dyn_kws
            )

            # set up writer
            def write_to_list():
                # needs to be 64 bit for db read
                f = atoms.calc.results['forces']
                atoms.calc.results['forces'] = f.astype(np.float64)
                chunk.atoms.append(canonicalize(atoms))
            dyn.attach(write_to_list)
            
            # run dynamics
            print('running dynamics')
            dyn.run(steps, **run_kws)
            print('done.')

            # submit to auditor
            await auditor.submit(chunk)


class DummyAuditor(Auditor):
    
    def __init__(
            self,
            accept_rate: float = 1.,
            sampler: Handle["TrainingSampler"],
            dynamics_engine: Handle["DynamicsEngine"],
            database: Handle["Database"]
        ):

        self.accept_rate = accept_rate
        self.sampler = sampler
        self.dynamics_engine = dynamics_engine
        self.queue = Queue()


    @action
    def submit(self, TrajectoryChunk):
        self.queue.put(TrajectoryChunk)

    @loop
    async def audit(
        self,
        shutdown: asyncio.Event
    ) -> None:
        """a stub of a real audit"""
        
        while not shutdown.is_set():
            try:
                chunk = self.queue.get()
            except Queue.Empty:
                continue
            
            good = np.random.random() < self.accept_rate
            score = float(good)
            if good:
                chunk.audit_status = AuditStatus.PASSED
                # todo: check if done
                await self.dynamics_engine.submit(chunk)
            else:
                chunk.audit_status = AuditStatus.FAILED
                await self.database.mark_sampled(chunk.traj_id)
                await self.training_sampler.submit(chunk)    

class DummySampler(TrainingSampler):

    def __init__(
        self,
        n_frames: int,
        labeler: Handle[TrainingLabeler]
        rng: np.random.Generator = None,
    ):
        self.rng = rng if rng else np.random.default_rng()
        self.queue = Queue()
        self.labeler = labeler
        self.n_frames = n_frames

    @loop
    async def sample(
        self,
        shudown: asyncio.Event
    ) -> None:
        while not shutdown.is_set():
            
            try:
                chunk = self.queue.get()
            except Queue.Empty:
                continue
            
            atoms = chunk.atoms
            frames = self.rng.choice(atoms, self.n_frames, replace=False)
            for frame in frames:
                await labeler.submit_frame(frame)

class DummyLabeler(TrainingLabeler):

    def __init__(
        self,
        databse: Handle[Database]
    ):
        self.database = database
        self.queue = Queue()

    @action
    async def submit_frame(self, frame: Atoms) -> None:

        self.queue.put(frame)

    @loop
    async def label_data(self, shutdown: asyncio.Event) -> None:
        try:
            frame = self.queue.get()
        except Queue.Empty:
            continue
        
        await database.write_training_frame(frame)


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
