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
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Union
from collections import namedtuple
from asyncio import wrap_future  
from concurrent.futures import Executor, Future

import numpy as np
import ase
from ase.db import connect
from ase import Atoms
from academy.handle import Handle
from academy.agent import Agent, action, loop
from ase.optimize.optimize import Dynamics
from mace.calculators import mace_mp
from parsl.concurrent import ParslPoolExecutor
from parsl.config import Config

from cascade.learning.base import BaseLearnableForcefield
from cascade.utils import canonicalize
from cascade.model import AuditStatus, AdvanceSpec, AuditResult
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
from cascade.model import ChunkSpec, TrainingFrame


ExecutorFuture = Union[Future[Any], asyncio.Future[Any]]


class CascadeAgent(Agent):
    """Base class for all cascade agents"""

    def __init__(
        self,
        config: CascadeAgentConfig,
        executor: Optional[Executor] = None
    ):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = executor
    
    def schedule_future_callback(
        self,
        future: ExecutorFuture,
        callback: Callable[[ExecutorFuture], Awaitable[None]],
        *,
        description: Optional[str] = None
    ) -> None:
        """Schedule a coroutine callback after a future completes.

        Args:
            future: Asyncio or concurrent.futures future to await.
            callback: Coroutine accepting the completed future.
            description: Optional identifier for logging.
        """

        async def _await_and_dispatch() -> None:
            try:
                if isinstance(future, asyncio.Future):
                    await future
                else:
                    await wrap_future(future)
            except Exception:
                self.logger.exception(
                    'Executor task %s raised before callback',
                    description or repr(future),
                )
                return

            try:
                await callback(future)
            except Exception:
                self.logger.exception(
                    'Executor callback %s raised an exception',
                    getattr(callback, '__qualname__', repr(callback)),
                )

        asyncio.create_task(_await_and_dispatch())
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
        auditor: Handle[Auditor],
        executor: Executor
    ):
        super().__init__(config, executor)
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
                    attempt_index=attempt_index)
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
            await self.auditor.submit(spec)


class Auditor(CascadeAgent):

    def __init__(
            self,
            config: AuditorConfig,
            sampler: Handle[DummySampler],
            dynamics_engine: Handle[DynamicsEngine],
            audit_task: Callable[[ChunkSpec], AuditResult],
            executor: Executor
    ):
        super().__init__(config, executor)
        self.sampler = sampler
        self.dynamics_engine = dynamics_engine
        self.queue = Queue()
        self.audit_task = audit_task

    @action
    async def submit(self, chunk_spec: ChunkSpec):
        """Submit a chunk for audit"""
        self.logger.debug(f'Received chunk {chunk_spec.chunk_id} from traj {chunk_spec.traj_id}')

        latest_attempt = self._traj_db.get_latest_chunk_attempt(
            run_id=self.config.run_id,
            traj_id=chunk_spec.traj_id,
            chunk_id=chunk_spec.chunk_id
        )
        if not latest_attempt:
            self.logger.warning(
                'No attempt metadata found for traj %s chunk %s; skipping audit',
                chunk_spec.traj_id,
                chunk_spec.chunk_id,
            )
            return
        chunk_atoms = self._traj_db.get_latest_chunk_attempt_atoms(
            self.config.run_id,
            chunk_spec.traj_id,
            chunk_spec.chunk_id,
            self._db
        )
        self.logger.info(f'Submitting audit of chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id} to executor')
        parsl_future = self.executor.submit(
            self.audit_task,
            chunk_atoms=chunk_atoms,
            chunk_spec=chunk_spec,
            attempt_index=latest_attempt['attempt_index']
        )
        self.schedule_future_callback(
            parsl_future,
            self._audit_callback,
            description=f'audit traj {chunk_spec.traj_id} chunk {chunk_spec.chunk_id}',
        )
    async def _audit_callback(self, future: Future) -> None:
        """Handle the results of an audit
    
        If the audit fails, the chunk is submitted to the sampler.
        If the audit passes and the trajectory is done, this is recorded in the database.
        If the audit passes and the trajectory is NOT done, the next chunk is submitted to the dynamics engine.

        Args:
            future: The future that contains the result of the audit

        Returns:
            None
        """
        self.logger.info('Audit callback started')
        if future.exception():
            self.logger.error('Audit failed: %s', future.exception())
            return

        self.logger.info('Getting future result')
        result = future.result()
        status = getattr(
            result,
            'status',
            AuditStatus.PASSED if getattr(result, 'passed', False) else AuditStatus.FAILED
        )
        self.logger.info('Audit result status: %s', status)

        if status not in [AuditStatus.PASSED, AuditStatus.FAILED]:
            self.logger.error('Audit result is not PASSED or FAILED: %s', status)
            return

        self._traj_db.update_chunk_audit_status(
            run_id=self.config.run_id,
            traj_id=result.traj_id,
            chunk_id=result.chunk_id,
            attempt_index=result.attempt_index,
            audit_status=status
        )
        
        if getattr(result, 'passed', False):
            self.logger.info(f'Audit passed for chunk {result.chunk_id} of traj {result.traj_id}')
            
            # Check if trajectory is done using the data model
            done = self._traj_db.is_trajectory_done(
                run_id=self.config.run_id,
                traj_id=result.traj_id
            )
            if done:
                # trajectory done, already marked in db 
                self.logger.info(f"Traj {result.traj_id} is complete")
            else:
                # Trajectory not done - submit next chunk
                # Get the last frame from the current chunk to use as starting point
                last_frame = self._traj_db.get_last_frame_from_chunk(
                    run_id=self.config.run_id,
                    traj_id=result.traj_id,
                    chunk_id=result.chunk_id,
                    attempt_index=result.attempt_index,
                    ase_db=self._db
                )
                # Create and submit next advance spec
                next_spec = AdvanceSpec(
                    atoms=last_frame,
                    traj_id=result.traj_id,
                    chunk_id=result.chunk_id + 1,
                    steps=self.config.chunk_size
                )
                await self.dynamics_engine.submit(next_spec)
                self.logger.info(f"Submitted next chunk {result.chunk_id + 1} for traj {result.traj_id}")
        else:
            # audit failed, submit to sampler
            self.logger.info(f'Audit failed for chunk {result.chunk_id} of traj {result.traj_id}')
            spec = ChunkSpec(
                traj_id=result.traj_id,
                chunk_id=result.chunk_id
            )
            await self.sampler.submit(spec)

    
            

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

                # Train model and update weights in dynamics engine
                weights = await self.trainer.train_model()
                await self.dynamics_engine.receive_weights(weights)
                
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
