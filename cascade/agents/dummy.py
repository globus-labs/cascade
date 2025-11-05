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
from cascade.agents.db_orm import TrajectoryDB, atoms_to_dict, dict_to_atoms, dict_to_calc_results, DBFrame
from ase.calculators.singlepoint import SinglePointCalculator



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

            # Track frame index
            frame_index = 0

            # set up dynamics
            dyn_kws = self.config.dyn_kws or {}
            run_kws = self.config.run_kws or {}
            dyn = self.config.dyn_cls(
                atoms,
                **dyn_kws
            )

            # set up writer
            def write_to_db():
                nonlocal frame_index
                # needs to be 64 bit for db read
                f = atoms.calc.results['forces']
                atoms.calc.results['forces'] = f.astype(np.float64)
                canonical_atoms = canonicalize(atoms)
                chunk.atoms.append(canonical_atoms)
                
                # Write frame to SQLAlchemy database
                self._traj_db.write_frame(
                    run_id=self.config.run_id,
                    traj_id=chunk.traj_id,
                    chunk_id=chunk.chunk_id,
                    attempt_index=attempt_index,
                    frame_index=frame_index,
                    atoms=canonical_atoms
                )
                frame_index += 1
            dyn.attach(write_to_db)

            # run dynamics
            self.logger.info(f"Running dynamics for chunk {chunk.chunk_id} of traj {chunk.traj_id}.")
            dyn.run(spec.steps, **run_kws)
            del chunk.atoms[0]  # we have the initial conditions saved elsewhere

            # Calculate number of frames from steps and loginterval
            loginterval = dyn_kws.get('loginterval') # None default -> 1
            n_frames = spec.steps // loginterval

            # Record chunk metadata in ORM
            success = self._traj_db.add_chunk_attempt(
                run_id=self.config.run_id,
                traj_id=chunk.traj_id,
                chunk_id=chunk.chunk_id,
                model_version=chunk.model_version,
                n_frames=n_frames,
                audit_status=AuditStatus.PENDING,
                attempt_index=attempt_index
            )
            if success:
                self.logger.info(f"Recorded chunk {chunk.chunk_id} of traj {chunk.traj_id} in database (attempt {attempt_index}, {n_frames} frames)")
            else:
                self.logger.error(f"Failed to record chunk {chunk.chunk_id} of traj {chunk.traj_id} in database")

            # submit to auditor
            self.logger.info(f"Submitting audit for chunk {chunk.chunk_id} of traj {chunk.traj_id}.")
            await self.auditor.submit(ChunkSpec(traj_id=chunk.traj_id, chunk_id=chunk.chunk_id))


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
                attempt_index=db_chunk['attempt_index'],
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
                        attempt_index=db_chunk['attempt_index']
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
                    self.logger.info(f"Traj {chunk_spec.traj_id} is complete")
                #todo: mt.2025.11.04 flatten this logic
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
            
            # Get frames from SQLAlchemy database for this chunk and attempt
            frames_data = self._traj_db.get_frames(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=db_chunk['attempt_index']
            )
            
            if not frames_data:
                self.logger.warning(f"No frames found for chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}")
                continue
            
            # Convert to Atoms objects
            atoms_list = []
            for frame_data in frames_data:
                atoms = dict_to_atoms(frame_data['atoms_data'])
                if frame_data['calc_results']:
                    results = dict_to_calc_results(frame_data['calc_results'])
                    atoms.calc = SinglePointCalculator(atoms, **results)
                atoms_list.append(atoms)
            
            # Sample frames
            n_sample = min(self.config.n_frames, len(atoms_list))
            indices = self.rng.choice(len(atoms_list), size=n_sample, replace=False)
            sampled_frames = [atoms_list[i] for i in indices]
            sampled_frame_ids = [frames_data[i]['id'] for i in indices]
            
            # Submit frames with their model version and frame IDs
            for frame, frame_id in zip(sampled_frames, sampled_frame_ids):
                training_frame = TrainingFrame(
                    atoms=frame,
                    model_version=db_chunk['model_version']
                )
                await self.labeler.submit(training_frame, frame_id)


class DummyLabeler(CascadeAgent):

    def __init__(
        self,
        config: LabelerConfig,
    ):
        super().__init__(config)
        self.queue = Queue()

    @action
    async def submit(self, training_frame: TrainingFrame, frame_id: int) -> None:
        await self.queue.put((training_frame, frame_id))

    @loop
    async def label_data(self, shutdown: asyncio.Event) -> None:

        while not shutdown.is_set():
            training_frame, frame_id = await self.queue.get()
            
            # Get trajectory and chunk info from frame
            try:
                frame = self._traj_db.get_frame_by_id(frame_id)
                if frame:
                    # Get frame metadata from database
                    with self._traj_db.session() as sess:
                        from cascade.agents.db_orm import DBFrame
                        db_frame = sess.query(DBFrame).filter_by(id=frame_id).first()
                        if db_frame:
                            traj_id = db_frame.traj_id
                            chunk_id = db_frame.chunk_id
                            attempt_index = db_frame.attempt_index
                        else:
                            traj_id = '?'
                            chunk_id = '?'
                            attempt_index = '?'
                else:
                    traj_id = '?'
                    chunk_id = '?'
                    attempt_index = '?'
            except Exception as e:
                self.logger.warning(f"Could not get traj/chunk info for frame ID {frame_id}: {e}")
                traj_id = '?'
                chunk_id = '?'
                attempt_index = '?'
            
            # Write the training frame to the ORM database
            success = self._traj_db.add_training_frame(
                run_id=self.config.run_id,
                frame_id=frame_id,
                model_version_sampled_from=training_frame.model_version
            )
            if success:
                self.logger.info(
                    f"Added training frame to database: traj={traj_id}, chunk={chunk_id}, "
                    f"attempt={attempt_index}, model_version={training_frame.model_version}"
                )
            else:
                self.logger.error(
                    f"Failed to add training frame to database: traj={traj_id}, chunk={chunk_id}, "
                    f"attempt={attempt_index}, model_version={training_frame.model_version}"
                )


class DummyTrainer(CascadeAgent):

    @action
    async def train_model(
        self,
    ) -> bytes:
        calc = mace_mp('small', device='cpu', default_dtype="float32") #todo: mt.2025.11.04 this should be configurable
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
            trajectories = self._traj_db.list_trajectories_in_run(self.config.run_id)
            
            if len(trajectories) == 0:
                await asyncio.sleep(1)
                continue
            
            all_done = all(traj['done'] for traj in trajectories)
            
            if all_done:
                self.logger.info("All trajs done, setting shutdown")
                shutdown.set()
                return
            else:
                await asyncio.sleep(1)

    @loop
    async def periodic_retrain(self, shutdown: asyncio.Event) -> None:
        """Monitor for enough training frames and trigger retraining"""
        self.logger.info("periodic_retrain loop started")
        while not shutdown.is_set():
            # Check if we have enough new training frames
            current_count = self._traj_db.count_training_frames(self.config.run_id)
            new_frames = current_count - self.last_train_count
            
            # Check fraction-based condition
            try:
                total_active, active_with_samples = self._traj_db.count_active_trajs_with_samples(
                    run_id=self.config.run_id
                )
                sampled_fraction = active_with_samples / total_active if total_active > 0 else 0.0
                
                self.logger.info(
                    f"Retrain check: new={new_frames}, active={total_active}, sampled={active_with_samples}, "
                    f"fraction={sampled_fraction:.2%}"
                )
            except Exception as e:
                self.logger.error(f"Error in count_active_trajs_with_samples: {e}", exc_info=True)
                total_active = 0
                active_with_samples = 0
                sampled_fraction = 0.0
            
            # Determine which condition triggered retraining
            absolute_condition = new_frames >= self.config.retrain_len
            fraction_condition = sampled_fraction >= self.config.retrain_fraction
            should_retrain = absolute_condition or fraction_condition
            
            if should_retrain:
                trigger_reason = []
                if absolute_condition:
                    trigger_reason.append(f"absolute threshold ({new_frames} >= {self.config.retrain_len})")
                if fraction_condition:
                    trigger_reason.append(f"fraction threshold ({sampled_fraction:.2%} >= {self.config.retrain_fraction:.2%})")
                
                self.logger.info(
                    f"Training frame count: current={current_count}, last_train={self.last_train_count}, "
                    f"new={new_frames}, active_trajs={total_active}, sampled_trajs={active_with_samples}, "
                    f"fraction={sampled_fraction:.2%}"
                )
                self.logger.info(f"Starting retraining triggered by: {', '.join(trigger_reason)}")

                # Train model and update weights in dynamics engine
                weights = await self.trainer.train_model()
                await self.dynamics_engine.receive_weights(weights)
                
                # Get trajectories we sampled frames from and resubmit them
                sampled_traj_ids = self._traj_db.get_sampled_traj_ids(
                    run_id=self.config.run_id
                )
                self.logger.info(f"Found {len(sampled_traj_ids)} sampled traj IDs: {sampled_traj_ids}")
                
                # For each trajectory we sampled from, resubmit from last passed chunk
                for traj_id in sampled_traj_ids:
                    # Get the latest passed chunk to resume from
                    latest_passed = self._traj_db.get_latest_passed_chunk(
                        run_id=self.config.run_id,
                        traj_id=traj_id
                    )
                    self.logger.info(f"Traj {traj_id}: latest_passed = {latest_passed}")
                    
                    if latest_passed:
                        # Get the last frame from this passed chunk
                        last_frame = self._traj_db.get_last_frame_from_chunk(
                            run_id=self.config.run_id,
                            traj_id=traj_id,
                            chunk_id=latest_passed['chunk_id'],
                            attempt_index=latest_passed['attempt_index']
                        )
                        self.logger.info(f"Traj {traj_id}: last_frame = {last_frame is not None}")
                        
                        if last_frame:
                            # Submit to retry the latest chunk (which failed)
                            next_spec = AdvanceSpec(
                                atoms=last_frame,
                                traj_id=traj_id,
                                chunk_id=latest_passed['chunk_id'] + 1,
                                steps=self.config.chunk_size
                            )
                            await self.dynamics_engine.submit(next_spec)
                            self.logger.info(f"Resubmitted traj {traj_id}, chunk {latest_passed['chunk_id'] + 1}")
                        else:
                            self.logger.warning(f"Traj {traj_id}: No last frame found for chunk {latest_passed['chunk_id']}")
                    else:
                        # No passed chunks - start from initial conditions
                        self.logger.info(f"Traj {traj_id}: No latest passed chunk, using initial conditions")
                        init_atoms = self._traj_db.get_initial_atoms(
                            run_id=self.config.run_id,
                            traj_id=traj_id
                        )
                        if init_atoms:
                            next_spec = AdvanceSpec(
                                atoms=init_atoms,
                                traj_id=traj_id,
                                chunk_id=0,
                                steps=self.config.chunk_size
                            )
                            await self.dynamics_engine.submit(next_spec)
                            self.logger.info(f"Resubmitted traj {traj_id} from initial conditions, chunk 0")
                        else:
                            self.logger.warning(f"Traj {traj_id}: No initial atoms found")
                
                self.last_train_count = current_count
                self.logger.info(f"Retraining complete, resetting frame count to {self.last_train_count}")
            else:
                self.logger.debug(f"Retraining not triggered, sleeping for 5 seconds")
                await asyncio.sleep(5)
