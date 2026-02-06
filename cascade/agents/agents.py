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
import gc
import logging
from functools import cached_property
from typing import Any, Awaitable, Callable, NamedTuple, Optional, cast
from concurrent.futures import Executor, Future as ConcurrentFuture

import os
import numpy as np
from ase import Atoms
from academy.handle import Handle
from academy.agent import Agent, action, loop
from ase.optimize.optimize import Dynamics
from mace.calculators import mace_mp
from parsl.concurrent import ParslPoolExecutor
from parsl.config import Config

try:
    import psutil
    PSUTIL_AVAILABLE = True
    _PSUTIL_WARNING_LOGGED = False
except ImportError:
    PSUTIL_AVAILABLE = False
    _PSUTIL_WARNING_LOGGED = False

from cascade.learning.base import BaseLearnableForcefield
from cascade.utils import canonicalize
from cascade.model import AuditStatus, AdvanceSpec, AuditResult, TrajectoryStatus, ChunkEventType
from cascade.agents.config import (
    CascadeAgentConfig,
    DatabaseConfig,
    DynamicsRunnerConfig,
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    TrainerConfig,
    DatabaseMonitorConfig
)
from cascade.agents.db_orm import TrajectoryDB
from cascade.model import ChunkSpec, TrainingFrame, TrainingFrameSpec

class CascadeAgent(Agent):
    """Base class for all cascade agents"""

    def __init__(
        self,
        executor: Optional[Executor] = None
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = executor


    def schedule_future_callback(
        self,
        future: ConcurrentFuture[Any],
        callback: Callable[[ConcurrentFuture[Any]], Awaitable[None]],
    ) -> None:
        """Schedule a callback to run after a future completes.

        Args:
            future: Future from executor.submit() (concurrent.futures.Future)
            callback: Coroutine accepting the completed future.
        """
        async def _run_callback() -> None:
            await asyncio.wrap_future(future)  # make concurrent future awatable 
            await callback(future)

        asyncio.create_task(_run_callback())
    
    @cached_property
    def _traj_db(self) -> TrajectoryDB:
        """A TrajectoryDB object for managing trajectory and chunk metadata"""
        CascadeAgent._resource_counts['traj_db_instances_created'] += 1
        self.logger.info(
            f"Creating TrajectoryDB instance #{CascadeAgent._resource_counts['traj_db_instances_created']} "
            f"for {self.__class__.__name__}"
        )
        traj_db = TrajectoryDB(self.config.db_url, logger=self.logger)
        traj_db.create_tables()  # Ensure tables exist
        return traj_db

class DynamicsRunner(CascadeAgent):

    def __init__(
        self,
        atoms: Atoms,
        run_id: str,
        traj_id: int,
        chunk_size: int,
        n_steps: int,
        auditor: Handle[Auditor],
        executor: Executor,
        advance_dynamics_task: Callable[[AdvanceSpec], None],
        learner: BaseLearnableForcefield,
        weights: bytes,
        dyn_cls: type[Dynamics],
        dyn_kws: dict[str, object] | None,
        run_kws: dict[str, object] | None,
        device: str = 'cpu',
    ):
        """Runs dynamics in a loop until done, or a shutdown message is received

        Arguments:
            atoms: initial conditions
            run_id: which run this is (for logging) # todo surely we dont have to pass this to every agent like this
            chunk_size: how many steps to advance at a time
            n_steps: total number of timesteps to run
            auditor: handle to auditor class
            executor: where the dynamics is executed
            advance_dynamics_task: task run on the executor
            learner: cascade learner class
            weights: weights for the learner
            dyn_cls: ase dynamics class
            dyn_kws: arguments to the dynamics constructor
            run_kws: arguments to the dynamics run method
            device: for torch execution
        """
        super().__init__(executor)
        self.atoms = atoms
        self.run_id = run_id
        self.traj_id = traj_id
        self.chunk_size = chunk_size
        self.n_steps = n_steps
        self.dyn_cls = dyn_cls
        self.weights = weights
        self.learner = learner
        self.weights = weights
        self.auditor = auditor
        self.advance_dynamics_task = advance_dynamics_task
        self.dyn_kws = dyn_kws or {}
        self.run_kws = run_kws or {}
        self.device = device

        self.timestep = 0
        self.chunk = 0
        self.attempt = 0
        self.done = False


    @action
    async def receive_weights(self, weights: bytes) -> None:
        self.weights = weights
        self.model_version += 1
        self.logger.info(f"Received new weights, now on model version {self.model_version}")

    @loop
    async def run(
            self,
            shutdown: asyncio.Event,
    ) -> None:
        """Run dynamics until done, or a shutdown message is received"""
        while not (shutdown.is_set() or self.done):

            # submit dynamics for evaluation
            spec = AdvanceSpec(
                atoms=self.atoms,
                steps=self.n_steps,
                run_id=self.run_id,
                traj_id=self.traj_id,
                chunk_id=self.chunk,
                attempt_index=self.attempt,
            )

            # save started dynamics event
            self._traj_db.record_chunk_event(
                run_id=self.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                attempt_index=spec.attempt_index,
                event_type=ChunkEventType.STARTED_DYNAMICS
            )
            self.logger.info(f"Running dynamics for {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")

            atoms_future = self.executor.submit(
                self.advance_dynamics_task,
                spec=spec,
                learner=self.learner,
                weights=self.weights,
                db_url=self.db_url,
                device=self.device,
                dyn_cls=self.dyn_cls,
                dyn_kws=self.dyn_kws
            )

            # save finished dynamics event
            self._traj_db.record_chunk_event(
                run_id=self.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                attempt_index=spec.attempt_index,
                event_type=ChunkEventType.FINISHED_DYNAMICS
            )
            self.logger.info(f"Finished dynamics for {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")

            # get future result
            atoms = atoms_future.result()

            # submit to auditor
            chunk_spec = ChunkSpec(traj_id=self.traj_id, chunk_id=self.chunk)
            self.logger.info(f"Submitting audit for {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")
            audit_status = await self.auditor.audit(chunk_spec)

            # handle audit result
            if audit_status == AuditStatus.PASSED:
                self.timestep += self.chunk_size
                self.done = self.timestep >= self.n_steps

                if self.done:
                    self._traj_db.mark_trajectory_completed(run_id=self.run_id, traj_id=self.traj_id)
                else:
                    # audit passed but not done, use the new atoms to run a new chunk
                    self.atoms = atoms
                    self.chunk += 1
                    self.timestep += self.chunk_size
                    self.attempt = 0
            else:
                # audit failed, try a new attempt once new model is received
                self.attempt += 1
                # wait for new weights
                # not sure how best to do this
                # await_event_async from academy?

    async def _advance_dynamics_callback(
        self,
        future: ConcurrentFuture[Any]
    ) -> None:
        """Handle the results of an advance dynamics call
        
        Args:
            future: The future that contains the result of the advance dynamics call

        Returns:
            None

        Track the chunk metadata in the database and submit to auditor.
        """
        self.logger.debug('Advance dynamics callback started')
        self.logger.info('Getting future result')
        spec = future.result()
        if isinstance(spec, dict):
            spec = AdvanceSpec(**spec)

        run_id = spec.run_id
        traj_id = spec.traj_id
        chunk_id = spec.chunk_id
        attempt_index = spec.attempt_index
        steps = spec.steps

        # Calculate number of frames from steps and loginterval
        loginterval = self.config.dyn_kws.get('loginterval', 1)
        n_frames = steps // loginterval

        # Record chunk metadata in ORM
        success = self._traj_db.add_chunk_attempt(
            run_id=run_id,
            traj_id=traj_id,
            chunk_id=chunk_id,
            model_version=self.model_version,
            n_frames=n_frames,
            audit_status=AuditStatus.PENDING,
            attempt_index=attempt_index
        )
        if success:
            self.logger.info(f"Recorded chunk {chunk_id} of traj {traj_id} in database (attempt {attempt_index}, {n_frames} frames)")
            self._traj_db.record_chunk_event(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                event_type=ChunkEventType.FINISHED_DYNAMICS
            )
        else:
            self.logger.error(f"Failed to record chunk {chunk_id} of traj {traj_id} in database (attempt {attempt_index})")

        # submit to auditor
        self.logger.info(f"Submitting audit for chunk {chunk_id} of traj {traj_id} attempt {attempt_index}.")
        await self.auditor.submit(ChunkSpec(traj_id=traj_id, chunk_id=chunk_id))


class Auditor(CascadeAgent):

    def __init__(
            self,
            config: AuditorConfig,
            sampler: Handle[Sampler],
            dynamics_engine: Handle[DynamicsRunner],
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
        self.logger.info(f'Received chunk {chunk_spec.chunk_id} from traj {chunk_spec.traj_id}')

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
        attempt_index = latest_attempt['attempt_index']
        self._traj_db.record_chunk_event(
            run_id=self.config.run_id,
            traj_id=chunk_spec.traj_id,
            chunk_id=chunk_spec.chunk_id,
            attempt_index=attempt_index,
            event_type=ChunkEventType.STARTED_AUDIT
        )
        chunk_atoms = self._traj_db.get_latest_chunk_attempt_atoms(
            self.config.run_id,
            chunk_spec.traj_id,
            chunk_spec.chunk_id
        )

        self.logger.info(f'Submitting audit of chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id} to executor')

        initial_future = self.executor.submit(
            self.audit_task,
            chunk_atoms=chunk_atoms,
            chunk_spec=chunk_spec,
            attempt_index=latest_attempt['attempt_index'],
        )

        self.schedule_future_callback(
            initial_future,
            self._audit_callback,
        )
    async def _audit_callback(self, future: ConcurrentFuture[Any]) -> None:
        """Handle the results of an audit
    
        If the audit fails, the chunk is submitted to the sampler.
        If the audit passes and the trajectory is done, this is recorded in the database.
        If the audit passes and the trajectory is NOT done, the next chunk is submitted to the dynamics engine.

        Args:
            future: The future that contains the result of the audit

        Returns:
            None
        """
        self.logger.debug('Audit callback started')
        self.logger.info('Getting future result')
        result = future.result()
        status = result.status
        self.logger.info(
            'Audit result status: %s (traj_id=%d, chunk_id=%d, attempt_index=%d)',
            status, result.traj_id, result.chunk_id, result.attempt_index
        )

        if status not in [AuditStatus.PASSED, AuditStatus.FAILED]:
            self.logger.error(
                'Audit result is not PASSED or FAILED: %s (traj_id=%d, chunk_id=%d, attempt_index=%d)',
                status, result.traj_id, result.chunk_id, result.attempt_index
            )
            return

        self._traj_db.update_chunk_audit_done_status(
            run_id=self.config.run_id,
            traj_id=result.traj_id,
            chunk_id=result.chunk_id,
            attempt_index=result.attempt_index,
            audit_status=status
        )
        
        # Record audit result event
        event_type = ChunkEventType.AUDIT_PASSED if status == AuditStatus.PASSED else ChunkEventType.AUDIT_FAILED
        self._traj_db.record_chunk_event(
            run_id=self.config.run_id,
            traj_id=result.traj_id,
            chunk_id=result.chunk_id,
            attempt_index=result.attempt_index,
            event_type=event_type
        )
        
        if status == AuditStatus.PASSED:
            self.logger.info(
                f'Audit passed for traj {result.traj_id} chunk {result.chunk_id} attempt {result.attempt_index}'
            )
            
            # Check if trajectory is done using the data model
            done = self._traj_db.is_trajectory_done(
                run_id=self.config.run_id,
                traj_id=result.traj_id
            )
            if done:
                self.logger.info(f"Traj {result.traj_id} is complete")
                self._traj_db.record_chunk_event(
                    run_id=self.config.run_id,
                    traj_id=result.traj_id,
                    chunk_id=result.chunk_id,
                    attempt_index=result.attempt_index,
                    event_type=ChunkEventType.TRAJECTORY_COMPLETED
                )
            else:
                # Trajectory not done - submit next chunk
                # Get the last frame from the current chunk to use as starting point
                last_frame = self._traj_db.get_last_frame_from_chunk(
                    run_id=self.config.run_id,
                    traj_id=result.traj_id,
                    chunk_id=result.chunk_id,
                    attempt_index=result.attempt_index
                ) # todo maybe we want to just query once
                # Force garbage collection after retrieving atoms
                gc.collect()
                
                # Create and submit next advance spec
                next_chunk_id = result.chunk_id + 1
                next_attempt_index = 0
                next_spec = AdvanceSpec(
                    atoms=last_frame,
                    run_id=self.config.run_id,
                    traj_id=result.traj_id,
                    chunk_id=next_chunk_id,
                    attempt_index=next_attempt_index,
                    steps=self.config.chunk_size
                )
                await self.dynamics_engine.submit(next_spec)
                self.logger.info(f"Submitted next chunk {result.chunk_id + 1} for traj {result.traj_id} (attempt {next_attempt_index}, after chunk {result.chunk_id} attempt {result.attempt_index} passed)")
        else:
            # audit failed, submit to sampler
            self.logger.info(
                f'Audit failed for traj {result.traj_id} chunk {result.chunk_id} attempt {result.attempt_index}'
            )
            spec = ChunkSpec(
                traj_id=result.traj_id,
                chunk_id=result.chunk_id,
                attempt_index=result.attempt_index
            )
            self.logger.info(f'Submitting failed chunk {result.chunk_id} of traj {result.traj_id} to sampler')
            await self.sampler.submit(spec)

    
class Sampler(CascadeAgent):

    def __init__(
        self,
        config: SamplerConfig,
        labeler: Handle[DummyLabeler],
        executor: Executor,
        sample_task: Callable[..., list[TrainingFrameSpec]],
    ):
        super().__init__(config, executor)
        self.queue = Queue()
        self.labeler = labeler
        self.sample_task = sample_task

    @action
    async def submit(self, chunk_spec: ChunkSpec):
        await self.queue.put(chunk_spec)

    def _make_sample_callback(self, chunk_spec: ChunkSpec):
        async def _sample_frames_callback(future: ConcurrentFuture[Any]) -> None:
            specs = future.result()
            if len(specs) != self.config.n_frames:
                self.logger.warning(
                    "Sampling returned %d frames for traj %s chunk %s (attempt %s), "
                    "expected n_frames=%d",
                    len(specs),
                    chunk_spec.traj_id,
                    chunk_spec.chunk_id,
                    chunk_spec.attempt_index,
                    self.config.n_frames,
                )
            for spec in specs:
                self.logger.info(
                    f'Submitting training frame from traj {chunk_spec.traj_id} '
                    f'chunk {chunk_spec.chunk_id} to labeler'
                )
                await self.labeler.submit(spec)
            self._traj_db.record_chunk_event(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=chunk_spec.attempt_index,
                event_type=ChunkEventType.FINISHED_SAMPLING,
            )
        return _sample_frames_callback

    @loop
    async def sample_frames(
        self,
        shutdown: asyncio.Event
    ) -> None:
        while not shutdown.is_set():
            chunk_spec = await self.queue.get()

            self.logger.info(
                f'Sampling frames from chunk {chunk_spec.chunk_id} of '
                f'traj {chunk_spec.traj_id}'
            )

            # Resolve model_version and attempt_index (auditor submits ChunkSpec
            # without model_version; we store it in chunk metadata)
            model_version = chunk_spec.model_version
            attempt_index = chunk_spec.attempt_index
            if model_version is None or attempt_index is None:
                latest = self._traj_db.get_latest_chunk_attempt(
                    self.config.run_id,
                    chunk_spec.traj_id,
                    chunk_spec.chunk_id,
                )
                if latest is not None:
                    if model_version is None:
                        model_version = latest.get('model_version')
                    if attempt_index is None:
                        attempt_index = latest['attempt_index']
            if model_version is None:
                self.logger.warning(
                    "Cannot determine model_version for traj %s chunk %s "
                    "(attempt %s), skipping",
                    chunk_spec.traj_id,
                    chunk_spec.chunk_id,
                    attempt_index,
                )
                continue

            # Ensure we have attempt_index for downstream use
            resolved_spec = ChunkSpec(
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=attempt_index,
                model_version=model_version,
            )

            # Get frames from trajectory_frames table for this chunk
            atoms_list = self._traj_db.get_latest_chunk_attempt_atoms(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
            )

            if not atoms_list:
                self.logger.warning(
                    f"No frames found for chunk {chunk_spec.chunk_id} of "
                    f"traj {chunk_spec.traj_id}"
                )
                continue

            # Get frame IDs for the sampled frames
            frame_ids = self._traj_db.get_chunk_frame_ids(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=attempt_index,
            )

            # Record STARTED_SAMPLING (only when we have atoms to process)
            self._traj_db.record_chunk_event(
                run_id=self.config.run_id,
                traj_id=resolved_spec.traj_id,
                chunk_id=resolved_spec.chunk_id,
                attempt_index=resolved_spec.attempt_index,
                event_type=ChunkEventType.STARTED_SAMPLING,
            )

            future = self.executor.submit(
                self.sample_task,
                atoms_list=atoms_list,
                frame_ids=frame_ids,
                chunk_spec=resolved_spec,
                model_version=model_version,
                n_frames=self.config.n_frames,
            )
            self.schedule_future_callback(
                future, self._make_sample_callback(resolved_spec)
            )


class DummyLabeler(CascadeAgent):

    def __init__(
        self,
        config: LabelerConfig,
    ):
        super().__init__(config)
        self.queue = Queue()

    @action
    async def submit(self, training_frame_spec: TrainingFrameSpec) -> None:
        self.logger.info(f'Received training frame (trajectory_frame_id={training_frame_spec.trajectory_frame_id})')
        await self.queue.put(training_frame_spec)

    @loop
    async def label_data(self, shutdown: asyncio.Event) -> None:

        while not shutdown.is_set():
            training_frame_spec = await self.queue.get()
            
            # Check if STARTED_LABELING exists for this chunk (idempotent)
            if not self._traj_db.has_chunk_event(
                run_id=self.config.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index,
                event_type=ChunkEventType.STARTED_LABELING
            ):
                self._traj_db.record_chunk_event(
                    run_id=self.config.run_id,
                    traj_id=training_frame_spec.traj_id,
                    chunk_id=training_frame_spec.chunk_id,
                    attempt_index=training_frame_spec.attempt_index,
                    event_type=ChunkEventType.STARTED_LABELING
                )
            
            # Record STARTED_LABELING_FRAME
            self._traj_db.record_chunk_event(
                run_id=self.config.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index,
                event_type=ChunkEventType.STARTED_LABELING_FRAME,
                frame_id=training_frame_spec.trajectory_frame_id
            )
            
            try:
                self._traj_db.add_training_frame(
                    run_id=self.config.run_id,
                    trajectory_frame_id=training_frame_spec.trajectory_frame_id,
                    model_version_sampled_from=training_frame_spec.training_frame.model_version,
                    traj_id=training_frame_spec.traj_id,
                    chunk_id=training_frame_spec.chunk_id,
                    attempt_index=training_frame_spec.attempt_index
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not add training frame for trajectory_frame_id "
                    f"{training_frame_spec.trajectory_frame_id}: {e}, skipping training frame"
                )
                continue
            
            # Record FINISHED_LABELING_FRAME after successfully adding training frame
            self._traj_db.record_chunk_event(
                run_id=self.config.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index,
                event_type=ChunkEventType.FINISHED_LABELING_FRAME,
                frame_id=training_frame_spec.trajectory_frame_id
            )
            
            # Check if all frames for chunk are done
            labeled_count = self._traj_db.count_labeled_frames_for_chunk(
                run_id=self.config.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index
            )
            
            if labeled_count >= training_frame_spec.total_frames_in_chunk:
                # All frames labeled, record FINISHED_LABELING
                self._traj_db.record_chunk_event(
                    run_id=self.config.run_id,
                    traj_id=training_frame_spec.traj_id,
                    chunk_id=training_frame_spec.chunk_id,
                    attempt_index=training_frame_spec.attempt_index,
                    event_type=ChunkEventType.FINISHED_LABELING
                )
            
            self.logger.info(
                f"Added training frame to database: traj={training_frame_spec.traj_id}, "
                f"chunk={training_frame_spec.chunk_id}, attempt={training_frame_spec.attempt_index}, "
                f"model_version={training_frame_spec.training_frame.model_version}"
            )


class DummyTrainer(CascadeAgent):

    @action
    async def train_model(
        self,
        training_round: int
    ) -> bytes:
        # Record STARTED_TRAINING event
        self._traj_db.record_training_event(
            run_id=self.config.run_id,
            event_type=ChunkEventType.STARTED_TRAINING,
            training_round=training_round
        )
        
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
        dynamics_engine: Handle[DynamicsRunner]
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
            
            all_finished = all(
                traj['status'] in (TrajectoryStatus.COMPLETED, TrajectoryStatus.FAILED)
                for traj in trajectories
            )
            
            if all_finished:
                self.logger.info("All trajs done, setting shutdown")
                shutdown.set()
                return
            else:
                await asyncio.sleep(1)

    def _log_and_check_resource_counts(self) -> None:
        """Log resource creation counts and alert if counts are unexpectedly high"""
        # Log resource creation counts
        self.logger.info(
            f"Resource counts: agents={CascadeAgent._resource_counts['agents_created']}, "
            f"traj_db_instances={CascadeAgent._resource_counts['traj_db_instances_created']}"
        )
        
        # Alert if resource counts are unexpectedly high
        if CascadeAgent._resource_counts['traj_db_instances_created'] > CascadeAgent._resource_counts['agents_created'] * 2:
            self.logger.warning(
                f"More TrajectoryDB instances ({CascadeAgent._resource_counts['traj_db_instances_created']}) "
                f"than expected for {CascadeAgent._resource_counts['agents_created']} agents. "
                "Instances may be recreated unnecessarily."
            )

    @loop
    async def periodic_retrain(self, shutdown: asyncio.Event) -> None:
        """Monitor for enough training frames and trigger retraining.
        
        Retraining is triggered when either condition is met:
        - Absolute threshold: number of new training frames >= retrain_len
        - Fraction threshold: fraction of active trajectories with samples >= retrain_fraction
        
        The absolute condition ensures retraining happens after accumulating a minimum
        number of frames, while the fraction condition ensures retraining occurs when
        a sufficient proportion of active trajectories have been sampled, even if the
        absolute count is low.
        
        After retraining, frames are resubmitted for execution with the new model.
        """
        self.logger.info("periodic_retrain loop started")
        while not shutdown.is_set():
            self._log_and_check_resource_counts()
            
            # Check if we have enough new training frames
            current_count = self._traj_db.count_training_frames(self.config.run_id)
            new_frames = current_count - self.last_train_count
            
            # Check fraction-based condition
            total_active, active_with_labeling = self._traj_db.count_active_trajs_with_labeling(
                run_id=self.config.run_id
            )
            sampled_fraction = active_with_labeling / total_active if total_active > 0 else 0.
            
            self.logger.info(
                f"Retrain check: new={new_frames}, active={total_active}, labeled={active_with_labeling}, "
                f"fraction={sampled_fraction:.2%}"
            )
            
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
                
                # Get the training round for frames that will be used in this retraining
                # (frames created before this retraining will have the current max training_round)
                training_round_for_retrain = self._traj_db.get_current_training_round(self.config.run_id)
                
                # Increment training round - new frames created after this will use the new round
                self.current_training_round = training_round_for_retrain + 1
                
                self.logger.info(
                    f"Starting retraining (round {self.current_training_round}) triggered by: {', '.join(trigger_reason)}\n"
                    f"Training frame count: current={current_count}, last_train={self.last_train_count}, "
                    f"new={new_frames}, active_trajs={total_active}, labeled_trajs={active_with_labeling}, "
                    f"fraction={sampled_fraction:.2%}"
                )

                # Train model and update weights in dynamics engine
                weights = await self.trainer.train_model(self.current_training_round)
                
                # Record FINISHED_TRAINING event after training completes
                self._traj_db.record_training_event(
                    run_id=self.config.run_id,
                    event_type=ChunkEventType.FINISHED_TRAINING,
                    training_round=self.current_training_round
                )
                
                await self.dynamics_engine.receive_weights(weights)
                
                # Get chunks where latest event is FINISHED_LABELING (these need resubmission)
                chunks_to_resubmit = self._traj_db.get_trajs_with_latest_event(
                    run_id=self.config.run_id,
                    event_type=ChunkEventType.FINISHED_SAMPLING
                )
                self.logger.info(f"Found {len(chunks_to_resubmit)} trajs to resubmit (latest event FINISHED_LABELING)")

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
                            attempt_index=prev_chunk_attempt['attempt_index']
                        )
                    else:
                        # chunk_id == 0: use initial trajectory frame
                        start_frame = self._traj_db.get_initial_trajectory_frame(
                            run_id=self.config.run_id,
                            traj_id=traj_id
                        )

                    # Resubmit the SAME chunk_id (dynamics engine will create a new attempt)
                    next_attempt_index = self._traj_db.get_next_attempt_index(
                        run_id=self.config.run_id,
                        traj_id=traj_id,
                        chunk_id=chunk_id
                    )
                    
                    if start_frame:
                        retry_spec = AdvanceSpec(
                            atoms=start_frame,
                            run_id=self.config.run_id,
                            traj_id=traj_id,
                            chunk_id=chunk_id,  # Same chunk_id, not chunk_id + 1
                            attempt_index=next_attempt_index,
                            steps=self.config.chunk_size
                        )
                        await self.dynamics_engine.submit(retry_spec)
                        self.logger.info(
                            f"Resubmitted traj {traj_id}, chunk {chunk_id} for retry "
                            f"(from attempt {attempt_index} to attempt {next_attempt_index})"
                        )
                    else:
                        self.logger.warning(
                            f"Traj {traj_id}: No starting frame found for chunk {chunk_id} attempt {next_attempt_index} "
                            f"({'initial trajectory' if chunk_id == 0 else f'previous chunk {chunk_id - 1}'})"
                        )
                
                self.last_train_count = current_count
                self.logger.info(f"Retraining complete, setting frame count to {current_count}")
            else:
                self.logger.debug(f"Retraining not triggered, sleeping for 5 seconds")
                await asyncio.sleep(5)
