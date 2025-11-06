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
from cascade.model import AuditStatus, AdvanceSpec
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
        
        # Try to load latest model version from database on startup
        latest = self._traj_db.get_latest_model_version(self.config.run_id)
        if latest:
            try:
                with open(latest['file_path'], 'rb') as f:
                    self.weights = f.read()
                self.model_version = latest['version']
                self.logger.info(f"Loaded model weights version {self.model_version} on startup")
            except Exception as e:
                self.logger.warning(f"Failed to load latest model weights on startup: {e}, using initial weights")

    @action
    async def submit(self, spec: AdvanceSpec):
        self.logger.debug("Received advance spec")
        await self.queue.put(spec)

    def _check_and_load_new_weights(self):
        """Check for new model version and load if available"""
        latest = self._traj_db.get_latest_model_version(self.config.run_id)
        
        if latest and latest['version'] > self.model_version:
            try:
                with open(latest['file_path'], 'rb') as f:
                    new_weights = f.read()
                # Verify file is not empty
                if len(new_weights) > 0:
                    self.weights = new_weights
                    self.model_version = latest['version']
                    self.logger.info(f"Loaded new model weights: version {self.model_version}")
                else:
                    self.logger.warning(f"Model weights file is empty for version {latest['version']}, skipping")
            except Exception as e:
                # This should be rare since DB guarantees file exists
                self.logger.error(f"Failed to load weights file: {e}, continuing with old version")

    @loop
    async def advance_dynamics(
        self,
        shutdown: asyncio.Event
    ) -> None:
        """Advance dynamics while there are trajectoires to advance"""
        while not shutdown.is_set():
            # Check for new model version before processing each chunk
            self._check_and_load_new_weights()

            spec = await self.queue.get()

            atoms = spec.atoms
            calc = self.config.learner.make_calculator(self.weights, device=self.config.device)
            atoms.calc = calc


            # Get attempt index before writing frames
            attempt_index = self._traj_db.get_next_attempt_index(
                run_id=self.config.run_id,
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
            def write_to_db():
                # needs to be 64 bit for db read
                f = atoms.calc.results['forces']
                atoms.calc.results['forces'] = f.astype(np.float64)
                canonical_atoms = canonicalize(atoms)
                self._db.write(
                    canonical_atoms, 
                    chunk_id=spec.chunk_id, 
                    traj_id=spec.traj_id,
                    run_id=self.config.run_id,
                    attempt_index=attempt_index
                )
            
            dyn.attach(write_to_db)
            # run dynamics
            self.logger.info(f"Running dynamics for chunk {spec.chunk_id} of traj {spec.traj_id}.")
            dyn.run(spec.steps, **run_kws)

            # Calculate number of frames from steps and loginterval
            loginterval = dyn_kws.get('loginterval', 1)
            n_frames = spec.steps // loginterval

            # Record chunk metadata in ORM
            success = self._traj_db.add_chunk_attempt(
                run_id=self.config.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                model_version=self.model_version,
                n_frames=n_frames,
                audit_status=AuditStatus.PENDING,
                attempt_index=attempt_index
            )
            if success:
                self.logger.info(f"Recorded chunk {spec.chunk_id} of traj {spec.traj_id} in database (attempt {attempt_index}, {n_frames} frames)")
            else:
                self.logger.error(f"Failed to record chunk {spec.chunk_id} of traj {spec.traj_id} in database")

            # submit to auditor
            self.logger.info(f"Submitting audit for chunk {spec.chunk_id} of traj {spec.traj_id}.")
            await self.auditor.submit(ChunkSpec(traj_id=spec.traj_id, chunk_id=spec.chunk_id))


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
                        attempt_index=db_chunk['attempt_index'],
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
            
            # Get frames from ASE database for this chunk and attempt
            frames = list(self._db.select(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=db_chunk['attempt_index']
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
                    model_version=db_chunk['model_version']
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
            
            # Get trajectory and chunk info from ASE DB
            try:
                row = self._db.get(ase_db_id)
                traj_id = row.get('traj_id', '?')
                chunk_id = row.get('chunk_id', '?')
                attempt_index = row.get('attempt_index', '?')
            except Exception as e:
                self.logger.warning(f"Could not get traj/chunk info for ASE DB ID {ase_db_id}: {e}")
                traj_id = '?'
                chunk_id = '?'
                attempt_index = '?'
            
            # Write the training frame to the ORM database
            # Extract traj_id, chunk_id, attempt_index from ASE DB row
            try:
                traj_id_int = int(traj_id) if traj_id != '?' else None
                chunk_id_int = int(chunk_id) if chunk_id != '?' else None
                attempt_index_int = int(attempt_index) if attempt_index != '?' else None
                
                if traj_id_int is None or chunk_id_int is None or attempt_index_int is None:
                    self.logger.warning(f"Could not parse chunk info for ASE DB ID {ase_db_id}, skipping training frame")
                    continue
                
                self._traj_db.add_training_frame(
                    run_id=self.config.run_id,
                    ase_db_id=ase_db_id,
                    model_version_sampled_from=training_frame.model_version,
                    traj_id=traj_id_int,
                    chunk_id=chunk_id_int,
                    attempt_index=attempt_index_int
                )
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not parse chunk info for ASE DB ID {ase_db_id}: {e}, skipping training frame")
                continue
            self.logger.info(
                f"Added training frame to database: traj={traj_id}, chunk={chunk_id}, "
                f"attempt={attempt_index}, model_version={training_frame.model_version}"
            )


class DummyTrainer(CascadeAgent):

    @action
    async def train_model(
        self,
    ) -> int:
        """Train model, write weights to disk, and update database
        
        Returns:
            Model version number if successful, raises exception on failure
        """
        calc = mace_mp('small', device='cpu', default_dtype="float32") #todo: mt.2025.11.04 this should be configurable
        model = calc.models[0]
        weights = self.config.learner.serialize_model(model)
        
        # Write to disk and update database
        version = self._traj_db.save_model_weights(
            run_id=self.config.run_id,
            weights=weights,
            weights_dir=self.config.weights_dir
        )
        
        self.logger.info(f"Trained and saved model version {version}")
        return version


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
        self.current_training_round = 0

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
                    run_id=self.config.run_id,
                    ase_db=self._db
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
                
                # Increment training round
                self.current_training_round += 1
                
                self.logger.info(
                    f"Starting retraining (round {self.current_training_round}) triggered by: {', '.join(trigger_reason)}\n"
                    f"Training frame count: current={current_count}, last_train={self.last_train_count}, "
                    f"new={new_frames}, active_trajs={total_active}, sampled_trajs={active_with_samples}, "
                    f"fraction={sampled_fraction:.2%}"
                )

                # Mark all unmarked training frames with the current training round
                frames_marked = self._traj_db.mark_training_frames_for_round(
                    run_id=self.config.run_id,
                    training_round=self.current_training_round
                )
                self.logger.info(f"Marked {frames_marked} training frames for round {self.current_training_round}")

                # Train model (writes to disk and updates database)
                version = await self.trainer.train_model()
                self.logger.info(f"Training complete, model version {version} is now available")
                
                # Get unique chunks that generated frames in this training round
                chunks_to_resubmit = self._traj_db.get_chunks_from_training_round(
                    run_id=self.config.run_id,
                    training_round=self.current_training_round
                )
                
                self.logger.info(f"Found {len(chunks_to_resubmit)} unique chunks to resubmit from training round {self.current_training_round}")
                
                # Resubmit only the chunks that were used in this training round
                # These should all be FAILED chunks - resubmit the SAME chunk_id (will create new attempt)
                for chunk_info in chunks_to_resubmit:
                    traj_id = chunk_info['traj_id']
                    chunk_id = chunk_info['chunk_id']
                    attempt_index = chunk_info['attempt_index']
                    
                    # Verify the chunk status (should be FAILED, but warn if not)
                    chunk_attempt = self._traj_db.get_chunk_attempt(
                        run_id=self.config.run_id,
                        traj_id=traj_id,
                        chunk_id=chunk_id,
                        attempt_index=attempt_index
                    )
                    
                    if chunk_attempt and chunk_attempt['audit_status'] != AuditStatus.FAILED:
                        self.logger.warning(
                            f"Traj {traj_id}: Chunk {chunk_id}, attempt {attempt_index} has status "
                            f"{chunk_attempt['audit_status']}, expected FAILED. Resubmitting anyway."
                        )
                    
                    # Check if a newer attempt for this chunk already exists
                    latest_attempt = self._traj_db.get_latest_chunk_attempt(
                        run_id=self.config.run_id,
                        traj_id=traj_id,
                        chunk_id=chunk_id
                    )
                    
                    if latest_attempt and latest_attempt['attempt_index'] > attempt_index:
                        self.logger.info(
                            f"Traj {traj_id}: Chunk {chunk_id} already has a newer attempt "
                            f"(attempt {latest_attempt['attempt_index']} > {attempt_index}), skipping resubmission"
                        )
                        continue
                    
                    # Get the starting frame for resubmission:
                    # - If chunk_id > 0: use first frame of previous chunk (chunk_id - 1)
                    # - If chunk_id == 0: use initial frame from trajectory
                    if chunk_id > 0:
                        # Get the latest attempt of the previous chunk
                        # Since chunk N exists, chunk N-1 must have passed, so the latest attempt should be the passed one
                        prev_chunk_attempt = self._traj_db.get_latest_chunk_attempt(
                            run_id=self.config.run_id,
                            traj_id=traj_id,
                            chunk_id=chunk_id - 1
                        )
                        
                        if not prev_chunk_attempt:
                            self.logger.warning(
                                f"Traj {traj_id}: No previous chunk {chunk_id - 1} found, skipping resubmission"
                            )
                            continue
                        
                        # Get last frame of previous chunk
                        start_frame = self._traj_db.get_last_frame_from_chunk(
                            run_id=self.config.run_id,
                            traj_id=traj_id,
                            chunk_id=chunk_id - 1,
                            attempt_index=prev_chunk_attempt['attempt_index'],
                            ase_db=self._db
                        )
                    else:
                        # chunk_id == 0: use initial trajectory frame
                        start_frame = self._traj_db.get_initial_trajectory_frame(
                            run_id=self.config.run_id,
                            traj_id=traj_id
                        )
                    
                    if start_frame:
                        # Resubmit the SAME chunk_id (dynamics engine will create a new attempt)
                        retry_spec = AdvanceSpec(
                            atoms=start_frame,
                            traj_id=traj_id,
                            chunk_id=chunk_id,  # Same chunk_id, not chunk_id + 1
                            steps=self.config.chunk_size
                        )
                        await self.dynamics_engine.submit(retry_spec)
                        self.logger.info(f"Resubmitted traj {traj_id}, chunk {chunk_id} for retry (from attempt {attempt_index})")
                    else:
                        self.logger.warning(
                            f"Traj {traj_id}: No starting frame found for chunk {chunk_id} "
                            f"({'initial trajectory' if chunk_id == 0 else f'previous chunk {chunk_id - 1}'})"
                        )
                
                self.last_train_count = current_count
                self.logger.info(f"Retraining complete, setting frame count to {self.last_train_count}")
            else:
                self.logger.debug(f"Retraining not triggered, sleeping for 5 seconds")
                await asyncio.sleep(5)
