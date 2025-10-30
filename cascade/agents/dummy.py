"""WIP agentic implementation of cascasde with academy

Everything here is a stub

In addition to de-stubbing, we have the following todos:
* thread/process safety for queues, model updates
* make configuration of agents flexible, updatable (i.e., control parameters)
* logging
"""
from __future__ import annotations

import asyncio
from asyncio import Queue
from threading import Event
import logging

import numpy as np
import ase
from ase import Atoms
from academy.handle import Handle
from academy.agent import Agent, action, loop
from ase.optimize.optimize import Dynamics
from mace.calculators import mace_mp

from cascade.learning.base import BaseLearnableForcefield
from cascade.utils import canonicalize
from cascade.model import AuditStatus, AdvanceSpec, TrajectoryChunk, Trajectory
from cascade.agents.config import (
    DatabaseConfig,
    DynamicsEngineConfig,
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    TrainerConfig
)


class DummyDatabase(Agent):
    """Not a real database, just a stub with in-memory storage"""

    def __init__(
        self,
        config: DatabaseConfig,
        trainer: Handle[DummyTrainer],
        dynamics_engine: Handle[DynamicsEngine]
    ):

        self.config = config
        self.training_frames = []
        self.sampled_trajectories = set()
        self.trainer = trainer
        self.dynamics_engine = dynamics_engine
        self.retrain = Event()
        self.logger = logging.getLogger('Database')

    @property
    def all_done(self):
        return all(t.done for t in self.config.trajectories)

    @action
    async def write_chunk(self, chunk: TrajectoryChunk):
        """Write chunks that have passed auditing and pass for advancement if not complete"""
        id = chunk.traj_id
        self.logger.info(f'Writing chunk of len {len(chunk)} for trajectory {id}')
        traj = self.config.trajectories[id]
        traj.add_chunk(chunk)
        atoms = chunk.atoms[-1]
        if not traj.done:
            advance_spec = AdvanceSpec(
                atoms,
                traj.id,
                chunk.chunk_id+1,
                self.config.chunk_size
            )
            await self.dynamics_engine.submit(advance_spec)

    @action
    async def write_training_frame(self, frame: Atoms):
        # check here if we have exceeded the retrain length, then set event
        self.training_frames.append(frame)
        self.logger.info("Adding training frame")
        if len(self.training_frames) >= self.config.retrain_len:
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
                await asyncio.sleep(1)

    @loop
    async def periodic_retrain(self, shutdown: asyncio.Event) -> None:
        """Trigger a retrain event and add the right atoms back to the dynamics queue"""
        while not shutdown.is_set():
            if not self.retrain.wait(timeout=1):
                await asyncio.sleep(1)
                continue
            self.logger.info("Starting Retraining")
            # retrain model and update weights in dynamics engine
            weights = await self.trainer.retrain()
            await self.dynamics_engine.receive_weights(weights)

            # put the sampled trajectories back in the dynamics queue
            self.logger.info("Clearing training set, submitting trajectories to advance")
            for traj_id in self.sampled_trajectories:
                atoms = self.config.trajectories[traj_id].chunks[-1].atoms[-1]
                await self.dynamics_engine.submit(atoms)

            # clear training data
            self.training_frames = []
            self.sampled_trajectories = set()


class DynamicsEngine(Agent):

    def __init__(
        self,
        config: DynamicsEngineConfig,
        auditor: Handle[DummyAuditor]
    ):

        self.config = config
        self.auditor = auditor
        self.weights = config.weights  # This needs to be mutable
        self.model_version = 0  # todo: mt.2025.10.20 probably this should be persisted somewhere else
        self.queue = Queue()

        self.logger = logging.getLogger("DynamicsEnginie")
        super().__init__()

    async def agent_on_startup(self):
        for spec in self.config.init_specs:
            await self.queue.put(spec)

    @action
    async def receive_weights(self, weights: bytes) -> None:
        self.weights = weights
        self.model_version += 1
        self.logger.info(f"Received new weights, now on model version {self.model_version}")

    @action
    async def submit(self, spec: AdvanceSpec):
        self.logger.debug("Received advance spec")
        await self.queue.put(spec)

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

            spec = await self.queue.get()

            atoms = spec.atoms
            calc = self.config.learner.make_calculator(self.weights, device=self.config.device)
            atoms.calc = calc

            chunk = TrajectoryChunk(
                atoms=[],
                model_version=self.model_version,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id
            )

            # set up dynamics
            dyn_kws = self.config.dyn_kws or {}
            run_kws = self.config.run_kws or {}
            dyn = self.config.dyn_cls(
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
            config: AuditorConfig,
            sampler: Handle[DummySampler],
            database: Handle[DummyDatabase]
    ):

        self.config = config
        self.sampler = sampler
        self.database = database
        self.queue = Queue()
        self.logger = logging.getLogger("Auditor")

    @action
    async def submit(self, chunk: TrajectoryChunk):
        self.logger.debug(f'Receieved chunk {chunk.chunk_id} from traj {chunk.traj_id}')
        await self.queue.put(chunk)

    @loop
    async def audit(
        self,
        shutdown: asyncio.Event
    ) -> None:
        """a stub of a real audit"""

        while not shutdown.is_set():
            chunk = await self.queue.get()
            self.logger.info(f'Auditing chunk {chunk.chunk_id} of traj {chunk.traj_id}')
            good = np.random.random() < self.config.accept_rate
            #  score = float(good)
            if good:
                chunk.audit_status = AuditStatus.PASSED
                self.logger.info(f'Audit passed for chunk {chunk.chunk_id} of traj {chunk.traj_id}')
                await self.database.write_chunk(chunk)
            else:
                chunk.audit_status = AuditStatus.FAILED
                self.logger.info(f'Audit failed for chunk {chunk.chunk_id} of traj {chunk.traj_id}')
                await self.database.mark_sampled(chunk.traj_id)
                await self.sampler.submit(chunk)


class DummySampler(Agent):

    def __init__(
        self,
        config: SamplerConfig,
        labeler: Handle[DummyLabeler]
    ):
        self.config = config
        self.rng = config.rng if config.rng else np.random.default_rng()
        self.queue = Queue()
        self.labeler = labeler
        self.logger = logging.getLogger('Sampler')

    @action
    async def submit(self, chunk: TrajectoryChunk):
        await self.queue.put(chunk)

    @loop
    async def sample_frames(
        self,
        shutdown: asyncio.Event
    ) -> None:
        while not shutdown.is_set():

            chunk = await self.queue.get()

            self.logger.info(f'Sampling frames from chunk {chunk.chunk_id} of traj {chunk.traj_id}')
            atoms = chunk.atoms
            frames = self.rng.choice(atoms, self.config.n_frames, replace=False)
            for frame in frames:
                await self.labeler.submit(frame)


class DummyLabeler(Agent):

    def __init__(
        self,
        config: LabelerConfig,
        database: Handle[DummyDatabase]
    ):
        self.config = config
        self.database = database
        self.queue = Queue()

    @action
    async def submit(self, frame: Atoms) -> None:
        await self.queue.put(frame)

    @loop
    async def label_data(self, shutdown: asyncio.Event) -> None:

        while not shutdown.is_set():
            frame = await self.queue.get()
            await self.database.write_training_frame(frame)


class DummyTrainer(Agent):

    def __init__(
        self,
        config: TrainerConfig
    ):
        self.config = config

    @action
    async def train_model(
        self,
        learner: BaseLearnableForcefield
    ) -> bytes:
        calc = mace_mp('small', device='cpu', default_dtype="float32")
        model = calc.models[0]
        model_msg = learner.serialize_model(model)
        return model_msg

    @action
    async def retrain(self) -> bytes:
        """Alias for train_model"""
        calc = mace_mp('small', device='cpu', default_dtype="float32")
        model = calc.models[0]
        # Return dummy weights as bytes
        import pickle
        return pickle.dumps(model)
