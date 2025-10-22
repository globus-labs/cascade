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
from __future__ import annotations

import asyncio
from typing import Callable
from collections import defaultdict
#from asyncio import Queue
from queue import Queue
from threading import Event
from time import sleep
import logging

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

from model import AdvanceSpec, TrajectoryChunk, Trajectory


class DummyDatabase(Agent):
    """Not a real database, just a stub with in-memory storage"""

    def __init__(
        self,
        trajectories: list[Trajectory],
        retrain_len: int,
        trainer: Handle[DummyTrainer],
        dynamics_engine: Handle[DynamicsEngine]
    ):

        self.trajectories = trajectories
        self.training_frames = []
        self.sampled_trajectories = set()
        self.retrain_len = retrain_len
        self.trainer = trainer
        self.dynamics_engine = dynamics_engine
        self.retrain = Event()
        self.logger = logging.getLogger('Database')

    @property
    def all_done(self):
        return all(t.done for t in self.trajectories)

    @action
    async def write_chunk(self, chunk: TrajectoryChunk):
        """Write chunks that have passed auditing and pass for advancement if not complete"""
        id = chunk.traj_id
        self.logger.info(f'Writing chunk of len {len(chunk)} for trajectory {id}')
        traj = self.trajectories[id]
        traj.add_chunk(chunk)
        atoms = chunk.atoms[-1]
        if not traj.is_done():
            advance_spec = AdvanceSpec(
                atoms,
                traj.id,
                chunk.chunk_id+1
            )
            await self.dynamics_engine.submit(advance_spec)

    @action
    async def write_training_frame(self, frame: Atoms):
        # check here if we have exceeded the retrain length, then set event
        self.training_frames.append(frame)
        self.logger.info("Adding training frame")
        if len(self.training_frames) >= self.retrain_len:
            self.logger.info('Setting training event')
            self.retrain.set()

    @action
    async def mark_sampled(self, traj_id: int):
        self.logger.info(f"Adding {traj_id} to sampled trajectories")
        self.sampled_trajectories.add(traj_id)

    @loop
    async def monitor_completion(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            if self.all_done:
                self.logger.info("All trajectories done, setting shutdown")
                shutdown.set()
            else:
                sleep(1)

    @loop
    async def periodic_retrain(self, shutdown: asyncio.Event) -> None:
        """Trigger a retrain event and add the right atoms back to the dynamics queue"""
        while not shutdown.is_set():
            if not self.retrain.wait(timeout=1):
                sleep(1)
                continue
            self.logger.info("Starting Retraining")
            # retrain model and update weights in dynamics engine
            weights = await self.trainer.retrain()
            await self.dynamics_engine.receive_weights(weights)

            # put the sampled trajectories back in the dynamics queue
            self.logger.info("Clearing training set, submitting trajectories to advance")
            for traj_id in self.sampled_trajectories:
                atoms = self.trajecotires[traj_id].chunks[-1].atoms[-1]
                await self.dymamics_engine.submit(atoms)

            # clear training data
            self.training_frames = []
            self.sampled_frames = set()


class DynamicsEngine(Agent):

    def __init__(
        self,
        init_specs: list[AdvanceSpec],
        auditor: Handle[DummyAuditor],
        learner: BaseLearnableForcefield,
        weights: bytes,
        dyn_cls: type[Dynamics],
        dyn_kws: dict[str, object],
        run_kws: dict[str, object],
        device: str = 'cpu'
    ):

        self.learner = learner
        self.weights = weights
        self.auditor = auditor
        self.dyn_cls = dyn_cls
        self.device = device
        self.dyn_kws = dyn_kws
        self.run_kws = run_kws

        self.model_version = 0  # todo: mt.2025.10.20 probably this should be persisted somewhere else

        self.queue = Queue()
        for spec in init_specs:
            self.queue.put(spec)
        self.logger = logging.getLogger("DynamicsEnginie")

    @action
    async def receive_weights(self, weights: bytes) -> None:
        self.weights = weights
        self.model_version += 1
        self.logger.info(f"Received new weights, now on model version {self.model_version}")

    @action
    async def submit(self, spec: AdvanceSpec):
        self.logger.debug("Received advance spec")
        self.queue.put(spec)

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
                spec = self.queue.get()
            except Queue.Empty:
                sleep(1)
                continue

            atoms = spec.atoms
            calc = self.learner.make_calculator(self.weights, device=self.device)
            atoms.calc = calc

            chunk = TrajectoryChunk(
                atoms=[],
                model_version=self.model_version,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id
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
            self.logger.info(f"Running dynamics for chunk {chunk.chunk_id} of traj {chunk.traj_id}.")
            dyn.run(spec.steps, **run_kws)
            del chunk.atoms[0]  # we have the initial conditions saved elsewhere

            # submit to auditor
            self.logger.info(f"Submitting audit for chunk {chunk.chunk_id} of traj {chunk.traj_id}.")
            await self.auditor.submit(chunk)
            self.logger.info('Done.')


class DummyAuditor(Agent):

    def __init__(
            self,
            accept_rate: float,
            sampler: Handle[DummySampler],
            database: Handle[DummyDatabase]
    ):

        self.accept_rate = accept_rate
        self.sampler = sampler
        #self.dynamics_engine = dynamics_engine
        self.queue = Queue()
        self.logger = logging.getLogger("Auditor")

    @action
    async def submit(self, chunk: TrajectoryChunk):
        self.queue.put(chunk)

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
                sleep(1)
                continue

            self.logger.info(f'Auditing chunk {chunk.chunk_id} of traj {chunk.traj_id}')
            good = np.random.random() < self.accept_rate
            #  score = float(good)
            if good:
                chunk.audit_status = AuditStatus.PASSED
                self.logger.info(f'Audit passed for chunk {chunk.chunk_id} of traj {chunk.traj_id}')
                await self.database.submit(chunk)
            else:
                chunk.audit_status = AuditStatus.FAILED
                self.logger.info(f'Audit failed for chunk {chunk.chunk_id} of traj {chunk.traj_id}')
                await self.database.mark_sampled(chunk.traj_id)
                await self.training_sampler.submit(chunk)


class DummySampler(Agent):

    def __init__(
        self,
        n_frames: int,
        labeler: Handle[DummyLabeler],
        rng: np.random.Generator = None,
    ):
        self.rng = rng if rng else np.random.default_rng()
        self.queue = Queue()
        self.labeler = labeler
        self.n_frames = n_frames
        self.logger = logging.getLogger('Sampler')

    @action
    async def submit(self, chunk: TrajectoryChunk):
        self.queue.put(chunk)

    @loop
    async def sample_frames(
        self,
        shutdown: asyncio.Event
    ) -> None:
        while not shutdown.is_set():

            try:
                chunk = self.queue.get()
            except Queue.Empty:
                sleep(1)
                continue

            self.logger.info(f'Sampling frames from chunk {chunk.chunk_id} of traj {chunk.traj_id}')
            atoms = chunk.atoms
            frames = self.rng.choice(atoms, self.n_frames, replace=False)
            for frame in frames:
                await self.labeler.submit(frame)


class DummyLabeler(Agent):

    def __init__(
        self,
        database: Handle[DummyDatabase]
    ):
        self.database = database
        self.queue = Queue()

    @action
    async def submit(self, frame: Atoms) -> None:
        self.queue.put(frame)

    @loop
    async def label_data(self, shutdown: asyncio.Event) -> None:

        while not shutdown.is_set():
            try:
                frame = self.queue.get()
            except Queue.Empty:
                sleep(1)
                continue
            await self.database.write_training_frame(frame)


class DummyTrainer(Agent):

    @action
    async def train_model(
        self,
        learner: BaseLearnableForcefield
    ) -> bytes:
        calc = mace_mp('small', device='cpu', default_dtype="float32")
        model = calc.models[0]
        model_msg = learner.serialize_model(model)
        return model_msg
