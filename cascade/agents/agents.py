"""WIP agentic implementation of cascasde with academy

Everything here is a stub

In addition to de-stubbing, we have the following todos:
* thread/process safety for queues, model updates
* make configuration of agents flexible, updatable (i.e., control parameters)
* logging
"""
from __future__ import annotations

import asyncio
from asyncio import Queue, Event, Lock, wrap_future
import gc
import logging
from functools import cached_property
from typing import Any, Awaitable, Callable, NamedTuple, Optional, cast
from concurrent.futures import Executor, Future as ConcurrentFuture

from ase import Atoms
from academy.handle import Handle
from academy.agent import Agent, action, loop
from academy.exception import AgentTerminatedError
from ase.optimize.optimize import Dynamics
from mace.calculators import mace_mp

try:
    import psutil
    PSUTIL_AVAILABLE = True
    _PSUTIL_WARNING_LOGGED = False
except ImportError:
    PSUTIL_AVAILABLE = False
    _PSUTIL_WARNING_LOGGED = False

from cascade.learning.base import BaseLearnableForcefield
from cascade.model import AuditStatus, AdvanceSpec, AuditResult, TrajectoryStatus, ChunkEventType
from cascade.agents.config import (
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    DatabaseMonitorConfig
)
from cascade.agents.db_orm import TrajectoryDB
from cascade.model import ChunkSpec, TrainingFrameSpec

class CascadeAgent(Agent):
    """Base class for all cascade agents"""

    async def agent_on_startup(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._traj_db = TrajectoryDB(self.db_url, logger=self.logger)
        self._traj_db.create_tables()

class DynamicsRunner(CascadeAgent):

    def __init__(
        self,
        atoms: Atoms,
        run_id: str,
        db_url: str,
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
        model_version: int = 0
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
            model_version: index of current model version
        """
        super().__init__()
        self.atoms = atoms
        self.db_url = db_url
        self.executor = executor
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
        self.model_version = model_version

        self.received_weights = Event()
        self.new_model_lock = Lock()
        self.new_model: tuple[bytes, int] | None = None  # weights, version


    @loop
    async def run(
            self,
            shutdown: asyncio.Event,
    ) -> None:
        """Run dynamics until done, or a shutdown message is received"""
        while not (shutdown.is_set() or self.done):

            # there are two conditions to release this lock: we finish, or we are waiting for new weights
            # await self.weights_lock.acquire()
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
            self.logger.info(f"Running dynamics for traj {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")

            async with self.new_model_lock:

                if self.new_model:
                    self.weights, self.model_version = self.new_model
                    self.new_model = None

                atoms_future = self.executor.submit(
                    self.advance_dynamics_task,
                    spec=spec,
                    learner=self.learner,
                    weights=self.weights,
                    db_url=self.db_url,
                    device=self.device,
                    dyn_cls=self.dyn_cls,
                    dyn_kws=self.dyn_kws,
                    run_kws=self.run_kws
                )

            self._traj_db.add_chunk_attempt(
                run_id=self.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                model_version=self.model_version,
                n_frames=self.chunk_size,
                audit_status=AuditStatus.PENDING,
                attempt_index=self.attempt
            )

            # get future result
            wrapped_future = wrap_future(atoms_future)
            await wrapped_future
            atoms = wrapped_future.result()

            self.logger.info(f"Finished dynamics for {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")
            # save finished dynamics event
            self._traj_db.record_chunk_event(
                run_id=self.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                attempt_index=spec.attempt_index,
                event_type=ChunkEventType.FINISHED_DYNAMICS
            )

            # submit to auditor
            chunk_spec = ChunkSpec(traj_id=self.traj_id, chunk_id=self.chunk)
            self.logger.info(f"Submitting audit for traj {self.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")
            audit_status = await self.auditor.audit(chunk_spec)

            # handle audit result
            if audit_status == AuditStatus.PASSED:

                self.logger.info(f"Audit passed for traj {self.traj_id} chunk {self.chunk} attempt {self.attempt}")
                self.timestep += self.chunk_size
                self.logger.info(f"On timestep {self.timestep} of {self.n_steps}")
                self.done = self.timestep >= self.n_steps
                if self.done:
                    self.logger.info(f"Finished dynamics for traj {self.traj_id} on {self.chunk} attempt {self.attempt}, shutting down")
                    self._traj_db.mark_trajectory_completed(run_id=self.run_id, traj_id=self.traj_id)
                    self.agent_shutdown()
                else:
                    # audit passed but not done, use the new atoms to run a new chunk
                    self.atoms = atoms
                    self.chunk += 1
                    self.attempt = 0
                    self.logger.info(f"Updating traj {self.traj_id} to chunk {self.chunk} attempt {self.attempt}")
            else:
                # audit failed, try a new attempt once new model is received
                self.logger.info(f'Audit failed for traj {self.traj_id} chunk {self.chunk} attempt {self.attempt}, waiting for new weights...')
                self.attempt += 1
                self.received_weights.clear()
                await self.received_weights.wait()
                self.logger.info('Received new weights')

    @action
    async def receive_weights(self, weights: bytes, model_version: int) -> None:
        async with self.new_model_lock:
            self.new_model = (weights, model_version)
        self.logger.info(f"Received weights for model version {self.model_version}")
        self.received_weights.set()


class Auditor(CascadeAgent):

    def __init__(
            self,
            run_id: int, # todo should every agent have this? for DB reasons?
            sampler: Handle[Sampler],
            audit_task: Callable[[ChunkSpec], AuditResult],
            executor: Executor,
            db_url: str,
            chunk_size: int,
            audit_kwargs: dict = None,
    ):
        super().__init__()
        self.run_id = run_id
        self.sampler = sampler
        self.queue = Queue()
        self.audit_task = audit_task
        self.audit_kwargs = audit_kwargs or {}
        self.db_url = db_url
        self.chunk_size = chunk_size
        self.executor = executor

    @action
    async def audit(self, chunk_spec: ChunkSpec) -> AuditResult:
        """Submit a chunk for audit"""
        self.logger.info(f'Received chunk {chunk_spec.chunk_id} from traj {chunk_spec.traj_id}')

        latest_attempt = self._traj_db.get_latest_chunk_attempt(
            run_id=self.run_id,
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
            run_id=self.run_id,
            traj_id=chunk_spec.traj_id,
            chunk_id=chunk_spec.chunk_id,
            attempt_index=attempt_index,
            event_type=ChunkEventType.STARTED_AUDIT
        )
        chunk_atoms = self._traj_db.get_latest_chunk_attempt_atoms(
            self.run_id,
            chunk_spec.traj_id,
            chunk_spec.chunk_id
        )

        self.logger.info(f'Submitting audit of chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id} to executor')

        future = self.executor.submit(
            self.audit_task,
            chunk_atoms=chunk_atoms,
            chunk_spec=chunk_spec,
            attempt_index=latest_attempt['attempt_index'],
            **self.audit_kwargs
        )
        wrapped_future = wrap_future(future)
        await wrapped_future
        result = wrapped_future.result()
        status = result.status

        # todo: should we keep this?
        if status not in [AuditStatus.PASSED, AuditStatus.FAILED]:
            self.logger.error(
                'Audit result is not PASSED or FAILED: %s (traj_id=%d, chunk_id=%d, attempt_index=%d)',
                status, result.traj_id, result.chunk_id, result.attempt_index
            )
            return

        self._traj_db.update_chunk_audit_done_status(
            run_id=self.run_id,
            traj_id=result.traj_id,
            chunk_id=result.chunk_id,
            attempt_index=result.attempt_index,
            audit_status=status
        )
        
        # Record audit result event
        event_type = ChunkEventType.AUDIT_PASSED if status == AuditStatus.PASSED else ChunkEventType.AUDIT_FAILED
        self._traj_db.record_chunk_event(
            run_id=self.run_id,
            traj_id=result.traj_id,
            chunk_id=result.chunk_id,
            attempt_index=result.attempt_index,
            event_type=event_type
        )
        if status == AuditStatus.PASSED:
            self.logger.info(
                f'Audit passed for traj {result.traj_id} chunk {result.chunk_id} attempt {result.attempt_index}'
            )
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
        return status 


class Sampler(CascadeAgent):

    def __init__(
        self,
        run_id: int,
        db_url: str,
        n_frames: int,
        labeler: Handle[DummyLabeler],
        executor: Executor,
        sample_task: Callable[..., list[TrainingFrameSpec]],
    ):
        super().__init__()
        self.run_id = run_id
        self.db_url = db_url
        self.executor = executor
        self.n_frames = n_frames
        self.queue = Queue()
        self.labeler = labeler
        self.sample_task = sample_task

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
                    self.run_id,
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
                run_id=self.run_id,
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
                run_id=self.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=attempt_index,
            )

            # Record STARTED_SAMPLING (only when we have atoms to process)
            self._traj_db.record_chunk_event(
                run_id=self.run_id,
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
                n_frames=self.n_frames,
            )
            wrapped_future = wrap_future(future)
            await wrapped_future
            specs = wrapped_future.result()
            
            if len(specs) != self.n_frames:
                self.logger.warning(
                    "Sampling returned %d frames for traj %s chunk %s (attempt %s), "
                    "expected n_frames=%d",
                    len(specs),
                    chunk_spec.traj_id,
                    chunk_spec.chunk_id,
                    chunk_spec.attempt_index,
                    self.n_frames,
                )
            for spec in specs:
                self.logger.info(
                    f'Submitting training frame from traj {chunk_spec.traj_id} '
                    f'chunk {chunk_spec.chunk_id} to labeler'
                )
                await self.labeler.submit(spec)
            
            self._traj_db.record_chunk_event(
                run_id=self.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id,
                attempt_index=chunk_spec.attempt_index,
                event_type=ChunkEventType.FINISHED_SAMPLING,
            )


class DummyLabeler(CascadeAgent):

    def __init__(
        self,
        run_id: int,
        db_url: str,
    ):
        super().__init__()
        self.run_id = run_id
        self.db_url = db_url
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
                run_id=self.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index,
                event_type=ChunkEventType.STARTED_LABELING
            ):
                self._traj_db.record_chunk_event(
                    run_id=self.run_id,
                    traj_id=training_frame_spec.traj_id,
                    chunk_id=training_frame_spec.chunk_id,
                    attempt_index=training_frame_spec.attempt_index,
                    event_type=ChunkEventType.STARTED_LABELING
                )
            
            # Record STARTED_LABELING_FRAME
            self._traj_db.record_chunk_event(
                run_id=self.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index,
                event_type=ChunkEventType.STARTED_LABELING_FRAME,
                frame_id=training_frame_spec.trajectory_frame_id
            )
            
            try:
                self._traj_db.add_training_frame(
                    run_id=self.run_id,
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
                run_id=self.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index,
                event_type=ChunkEventType.FINISHED_LABELING_FRAME,
                frame_id=training_frame_spec.trajectory_frame_id
            )
            
            # Check if all frames for chunk are done
            labeled_count = self._traj_db.count_labeled_frames_for_chunk(
                run_id=self.run_id,
                traj_id=training_frame_spec.traj_id,
                chunk_id=training_frame_spec.chunk_id,
                attempt_index=training_frame_spec.attempt_index
            )
            
            if labeled_count >= training_frame_spec.total_frames_in_chunk:
                # All frames labeled, record FINISHED_LABELING
                self._traj_db.record_chunk_event(
                    run_id=self.run_id,
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

    def __init__(self,
                 run_id: int,
                 db_url: str,
                 learner: BaseLearnableForcefield
                 ):
        self.run_id = run_id
        self.db_url = db_url
        self.learner = learner

    @action
    async def train_model(
        self,
        training_round: int,
    ) -> bytes:
        # Record STARTED_TRAINING event
        self._traj_db.record_training_event(
            run_id=self.run_id,
            event_type=ChunkEventType.STARTED_TRAINING,
            training_round=training_round
        )
        
        calc = mace_mp('small', device='cpu', default_dtype="float32") #todo: mt.2025.11.04 this should be configurable
        model = calc.models[0]
        model_msg = self.learner.serialize_model(model)
        return model_msg


class DatabaseMonitor(CascadeAgent):
    """Monitors the database for training triggers and completion"""

    def __init__(
        self,
        run_id: str,
        db_url: str,
        retrain_len: int,
        target_length: int,
        chunk_size: int,
        retrain_fraction: float,
        retrain_min_frames: int,
        trainer: Handle[DummyTrainer],
        dynamics_runners: list[Handle[DynamicsRunner]]
    ):
        super().__init__()
        self.trainer = trainer
        self.dynamics_runners = dynamics_runners
        self.last_train_count = 0
        self.current_training_round = 0
        self.model_version = 0
        self.db_url = db_url
        self.run_id = run_id
        self.retrain_len = retrain_len
        self.target_length = target_length
        self.chunk_size = chunk_size
        self.retrain_fraction = retrain_fraction
        self.retrain_min_frames = retrain_min_frames

    @loop
    async def monitor_completion(self, shutdown: asyncio.Event) -> None:
        """Monitor if all trajectories are done and set shutdown"""
        while not shutdown.is_set():
            # Check if all trajectories are complete
            trajectories = self._traj_db.list_trajectories_in_run(self.run_id)
            
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
            await asyncio.sleep(5)
            # Check if we have enough new training frames
            current_count = self._traj_db.count_training_frames(self.run_id)
            new_frames = current_count - self.last_train_count
            
            # Check fraction-based condition
            total_active, active_with_labeling = self._traj_db.count_active_trajs_with_labeling(
                run_id=self.run_id
            )
            sampled_fraction = active_with_labeling / total_active if total_active > 0 else 0.
            
            self.logger.info(
                f"Retrain check: new={new_frames}, active={total_active}, labeled={active_with_labeling}, "
                f"fraction={sampled_fraction:.2%}"
            )
            
            # Determine which condition triggered retraining
            absolute_condition = new_frames >= self.retrain_len
            fraction_condition = sampled_fraction >= self.retrain_fraction
            should_retrain = absolute_condition or fraction_condition
            
            if should_retrain:
                trigger_reason = []
                if absolute_condition:
                    trigger_reason.append(f"absolute threshold ({new_frames} >= {self.retrain_len})")
                if fraction_condition:
                    trigger_reason.append(f"fraction threshold ({sampled_fraction:.2%} >= {self.retrain_fraction:.2%})")
                
                # Get the training round for frames that will be used in this retraining
                # (frames created before this retraining will have the current max training_round)
                training_round_for_retrain = self._traj_db.get_current_training_round(self.run_id)
                
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
                    run_id=self.run_id,
                    event_type=ChunkEventType.FINISHED_TRAINING,
                    training_round=self.current_training_round
                )

                self.model_version += 1

                for runner in self.dynamics_runners:
                    try:
                        await runner.receive_weights(weights, self.model_version)
                    except AgentTerminatedError:
                        pass
