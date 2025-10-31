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
from functools import cached_property
from typing import NamedTuple
from collections import namedtuple

import numpy as np
import ase
from ase.db import connect
from ase import Atoms
from academy.handle import Handle
from academy.agent import Agent, action, loop
from ase.optimize.optimize import Dynamics
from mace.calculators import mace_mp

from cascade.learning.base import BaseLearnableForcefield
from cascade.utils import canonicalize
from cascade.model import AuditStatus, AdvanceSpec, TrajectoryChunk, Trajectory
from cascade.agents.config import (
    CascadeAgentConfig,
    DatabaseConfig,
    DynamicsEngineConfig,
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    TrainerConfig,
    DatabaseMonitorConfig
)
from cascade.agents.db_orm import TrajectoryDB


# class DummyDatabase(Agent):
#     """Not a real database, just a stub with in-memory storage"""

#     def __init__(
#         self,
#         config: DatabaseConfig,
#         trainer: Handle[DummyTrainer],
#         dynamics_engine: Handle[DynamicsEngine]
#     ):

#         self.config = config
#         self.training_frames = []
#         self.sampled_trajectories = set()
#         self.trainer = trainer
#         self.dynamics_engine = dynamics_engine
#         self.retrain = Event()
#         self.logger = logging.getLogger('Database')

#     @property
#     def all_done(self):
#         return all(t.done for t in self.config.trajectories)

#     @action
#     async def write_chunk(self, chunk: TrajectoryChunk):
#         """Write chunks that have passed auditing and pass for advancement if not complete"""
#         id = chunk.traj_id
#         self.logger.info(f'Writing chunk of len {len(chunk)} for trajectory {id}')
#         traj = self.config.trajectories[id]
#         traj.add_chunk(chunk)
#         atoms = chunk.atoms[-1]
#         if not traj.done:
#             advance_spec = AdvanceSpec(
#                 atoms,
#                 traj.id,
#                 chunk.chunk_id+1,
#                 self.config.chunk_size
#             )
#             await self.dynamics_engine.submit(advance_spec)

#     @action
#     async def write_training_frame(self, frame: Atoms):
#         # check here if we have exceeded the retrain length, then set event
#         self.training_frames.append(frame)
#         self.logger.info("Adding training frame")
#         if len(self.training_frames) >= self.config.retrain_len:
#             self.logger.info('Setting training event')
#             self.retrain.set()

#     @action
#     async def mark_sampled(self, traj_id: int):
#         self.logger.info(f"Adding {traj_id} to sampled trajectories")
#         self.sampled_trajectories.add(traj_id)

#     @loop
#     async def monitor_completion(self, shutdown: asyncio.Event) -> None:
#         while not shutdown.is_set():
#             if self.all_done:
#                 self.logger.info("All trajectories done, setting shutdown")
#                 shutdown.set()
#             else:
#                 await asyncio.sleep(1)

#     @loop
#     async def periodic_retrain(self, shutdown: asyncio.Event) -> None:
#         """Trigger a retrain event and add the right atoms back to the dynamics queue"""
#         while not shutdown.is_set():
#             if not self.retrain.wait(timeout=1):
#                 await asyncio.sleep(1)
#                 continue
#             self.logger.info("Starting Retraining")
#             # retrain model and update weights in dynamics engine
#             weights = await self.trainer.retrain()
#             await self.dynamics_engine.receive_weights(weights)

#             # put the sampled trajectories back in the dynamics queue
#             self.logger.info("Clearing training set, submitting trajectories to advance")
#             for traj_id in self.sampled_trajectories:
#                 atoms = self.config.trajectories[traj_id].chunks[-1].atoms[-1]
#                 await self.dynamics_engine.submit(atoms)

#             # clear training data
#             self.training_frames = []
#             self.sampled_trajectories = set()

ChunkSpec = namedtuple('ChunkSpec', ['traj_id', 'chunk_id'])
TrainingFrame = namedtuple('TrainingFrame', ['atoms', 'model_version'])

class CascadeAgent(Agent):
    """Base class for all cascade agents"""

    def __init__(
        self,
        config: CascadeAgentConfig,
    ):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @cached_property
    def _db(self):
        """An ASE database object for writing individual frames"""
        return connect(self.config.db_url)
    
    @cached_property
    def _traj_db(self) -> TrajectoryDB:
        """A TrajectoryDB object for managing trajectory and chunk metadata"""
        traj_db = TrajectoryDB(self.config.db_url)
        traj_db.create_tables()  # Ensure tables exist
        return traj_db

class DynamicsEngine(CascadeAgent):

    def __init__(
        self,
        config: DynamicsEngineConfig,
        auditor: Handle[DummyAuditor]
    ):
        super().__init__(config)
        self.weights = config.weights  # This needs to be mutable
        self.model_version = 0  # todo: mt.2025.10.20 probably this should be persisted somewhere else
        self.queue = Queue()
        self.auditor = auditor

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

            # Get attempt index before writing frames
            attempt_index = self._traj_db.get_next_attempt_index(
                run_id=self.config.run_id,
                traj_id=chunk.traj_id,
                chunk_id=chunk.chunk_id
            )

            # set up dynamics
            dyn_kws = self.config.dyn_kws or {}
            run_kws = self.config.run_kws or {}
            dyn = self.config.dyn_cls(
                atoms,
                **dyn_kws
            )

            # set up writer
            def write_to_db():
                # needs to be 64 bit for db read
                f = atoms.calc.results['forces']
                atoms.calc.results['forces'] = f.astype(np.float64)
                canonical_atoms = canonicalize(atoms)
                chunk.atoms.append(canonical_atoms)
                self._db.write(canonical_atoms, chunk_id=chunk.chunk_id, traj_id=chunk.traj_id, run_id=self.config.run_id, attempt_index=attempt_index)
            dyn.attach(write_to_db)

            # run dynamics
            self.logger.info(f"Running dynamics for chunk {chunk.chunk_id} of traj {chunk.traj_id}.")
            dyn.run(spec.steps, **run_kws)
            del chunk.atoms[0]  # we have the initial conditions saved elsewhere

            # Calculate number of frames from steps and loginterval
            # Frames are written at: 0, loginterval, 2*loginterval, ...
            # After removing initial frame: n_frames = steps // loginterval
            loginterval = dyn_kws.get('loginterval', 1)
            n_frames = spec.steps // loginterval

            # Record chunk metadata in ORM
            db_chunk = self._traj_db.add_chunk_attempt(
                run_id=self.config.run_id,
                traj_id=chunk.traj_id,
                chunk_id=chunk.chunk_id,
                model_version=chunk.model_version,
                n_frames=n_frames,
                audit_status=AuditStatus.PENDING,
                attempt_index=attempt_index
            )
            self.logger.info(f"Recorded chunk {chunk.chunk_id} of traj {chunk.traj_id} in database (attempt {attempt_index}, {n_frames} frames)")

            # submit to auditor
            self.logger.info(f"Submitting audit for chunk {chunk.chunk_id} of traj {chunk.traj_id}.")
            await self.auditor.submit(ChunkSpec(traj_id=chunk.traj_id, chunk_id=chunk.chunk_id))
            self.logger.info('Done.')


class DummyAuditor(CascadeAgent):

    def __init__(
            self,
            config: AuditorConfig,
            sampler: Handle[DummySampler],
            dynamics_engine: Handle[DynamicsEngine],
    ):
        super().__init__(config)
        self.sampler = sampler
        self.dynamics_engine = dynamics_engine
        self.queue = Queue()

    @action
    async def submit(self, chunk_spec: ChunkSpec):
        self.logger.debug(f'Receieved chunk {chunk_spec.chunk_id} from traj {chunk_spec.traj_id}')
        await self.queue.put(chunk_spec)

    @loop
    async def audit(
        self,
        shutdown: asyncio.Event
    ) -> None:
        """a stub of a real audit"""

        while not shutdown.is_set():
            chunk_spec = await self.queue.get()
            self.logger.info(f'Auditing chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}')
            
            # Get the latest attempt for this chunk from ORM
            db_chunk = self._traj_db.get_latest_chunk_attempt(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id
            )
            
            if not db_chunk:
                self.logger.warning(f"No chunk attempt found for traj {chunk_spec.traj_id}, chunk {chunk_spec.chunk_id}")
                continue
            
            # Perform audit
            good = np.random.random() < self.config.accept_rate
            new_status = AuditStatus.PASSED if good else AuditStatus.FAILED
            
            # Update audit status in ORM
            self._traj_db.update_chunk_audit_status(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=db_chunk.attempt_index,
                audit_status=new_status
            )
            
            if good:
                self.logger.info(f'Audit passed for chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}')
                
                # Check if trajectory is done using the data model
                if not self._traj_db.is_trajectory_done(
                    run_id=self.config.run_id,
                    traj_id=chunk_spec.traj_id
                ):
                    # Trajectory not done - submit next chunk
                    # Get the last frame from the current chunk to use as starting point
                    last_frame = self._traj_db.get_last_frame_from_chunk(
                        run_id=self.config.run_id,
                        traj_id=chunk_spec.traj_id,
                        chunk_id=chunk_spec.chunk_id,
                        attempt_index=db_chunk.attempt_index,
                        ase_db=self._db
                    )
                    
                    if last_frame:
                        # Create and submit next advance spec
                        next_spec = AdvanceSpec(
                            atoms=last_frame,
                            traj_id=chunk_spec.traj_id,
                            chunk_id=chunk_spec.chunk_id + 1,
                            steps=self.config.chunk_size
                        )
                        await self.dynamics_engine.submit(next_spec)
                        self.logger.info(f"Submitted next chunk {chunk_spec.chunk_id + 1} for traj {chunk_spec.traj_id}")
                    else:
                        self.logger.warning(f"No frames found for chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}")
                else:
                    self.logger.info(f"Trajectory {chunk_spec.traj_id} is complete")
            else:
                self.logger.info(f'Audit failed for chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}')
                # Submit failed chunk to sampler
                await self.sampler.submit(chunk_spec)


class DummySampler(CascadeAgent):

    def __init__(
        self,
        config: SamplerConfig,
        labeler: Handle[DummyLabeler]
    ):
        super().__init__(config)
        self.rng = config.rng if config.rng else np.random.default_rng()
        self.queue = Queue()
        self.labeler = labeler

    @action
    async def submit(self, chunk_spec: ChunkSpec):
        await self.queue.put(chunk_spec)

    @loop
    async def sample_frames(
        self,
        shutdown: asyncio.Event
    ) -> None:
        while not shutdown.is_set():

            chunk_spec = await self.queue.get()

            self.logger.info(f'Sampling frames from chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}')
            
            # Get the latest attempt for this chunk to get model version and attempt index
            db_chunk = self._traj_db.get_latest_chunk_attempt(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id
            )
            
            if not db_chunk:
                self.logger.warning(f"No chunk attempt found for chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}")
                continue
            
            # Get frames from ASE database for this chunk and attempt
            frames = list(self._db.select(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=db_chunk.attempt_index
            ))
            
            if not frames:
                self.logger.warning(f"No frames found for chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}")
                continue
            
            # Convert to Atoms objects
            atoms_list = [row.toatoms() for row in frames]
            
            # Sample frames
            n_sample = min(self.config.n_frames, len(atoms_list))
            indices = self.rng.choice(len(atoms_list), size=n_sample, replace=False)
            sampled_frames = [atoms_list[i] for i in indices]
            sampled_frame_ids = [frames[i].id for i in indices]
            
            # Submit frames with their model version and ASE DB IDs
            for frame, ase_db_id in zip(sampled_frames, sampled_frame_ids):
                training_frame = TrainingFrame(
                    atoms=frame,
                    model_version=db_chunk.model_version
                )
                await self.labeler.submit(training_frame, ase_db_id)


class DummyLabeler(CascadeAgent):

    def __init__(
        self,
        config: LabelerConfig,
    ):
        super().__init__(config)
        self.queue = Queue()

    @action
    async def submit(self, training_frame: TrainingFrame, ase_db_id: int) -> None:
        await self.queue.put((training_frame, ase_db_id))

    @loop
    async def label_data(self, shutdown: asyncio.Event) -> None:

        while not shutdown.is_set():
            training_frame, ase_db_id = await self.queue.get()
            
            # Write the training frame to the ORM database
            self._traj_db.add_training_frame(
                run_id=self.config.run_id,
                ase_db_id=ase_db_id,
                model_version_sampled_from=training_frame.model_version
            )
            self.logger.info(f"Added training frame with model version {training_frame.model_version} to database")


class DummyTrainer(CascadeAgent):

    @action
    async def train_model(
        self,
    ) -> bytes:
        calc = mace_mp('small', device='cpu', default_dtype="float32")
        model = calc.models[0]
        model_msg = self.config.learner.serialize_model(model)
        return model_msg


class DatabaseMonitor(CascadeAgent):
    """Monitors the database for training triggers and completion"""

    def __init__(
        self,
        config: DatabaseMonitorConfig,
        trainer: Handle[DummyTrainer],
        dynamics_engine: Handle[DynamicsEngine]
    ):
        super().__init__(config)
        self.trainer = trainer
        self.dynamics_engine = dynamics_engine
        self.last_train_count = 0

    @loop
    async def monitor_completion(self, shutdown: asyncio.Event) -> None:
        """Monitor if all trajectories are done and set shutdown"""
        while not shutdown.is_set():
            # Check if all trajectories are complete
            all_done = True
            # We need to query all trajectories for this run and check if they're done
            with self._traj_db.session() as sess:
                from cascade.agents.db_orm import DBTrajectory
                trajectories = sess.query(DBTrajectory).filter_by(
                    run_id=self.config.run_id
                ).all()
                
                for traj in trajectories:
                    if not self._traj_db.is_trajectory_done(
                        run_id=self.config.run_id,
                        traj_id=traj.traj_id
                    ):
                        all_done = False
                        break
            
            if all_done and len(trajectories) > 0:
                self.logger.info("All trajectories done, setting shutdown")
                shutdown.set()
                return
            else:
                await asyncio.sleep(1)

    @loop
    async def periodic_retrain(self, shutdown: asyncio.Event) -> None:
        """Monitor for enough training frames and trigger retraining"""
        while not shutdown.is_set():
            # Check if we have enough new training frames
            current_count = self._traj_db.count_training_frames(self.config.run_id)
            new_frames = current_count - self.last_train_count
            
            if new_frames >= self.config.retrain_len:
                self.logger.info(f"Starting retraining with {new_frames} new frames (threshold: {self.config.retrain_len})")

                await self.trainer.train_model()
                
                self.last_train_count = current_count
                self.logger.info(f"Retraining complete, resetting frame count to {self.last_train_count}")
            else:
                await asyncio.sleep(5)
