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
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Union, cast
from collections import namedtuple
from asyncio import wrap_future  
from concurrent.futures import Executor, Future

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
from cascade.model import AuditStatus, AdvanceSpec, AuditResult, TrajectoryStatus
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


def count_file_descriptors_by_type() -> dict:
    """Count open file descriptors by type for the current process
    
    Returns:
        Dict with counts by FD type (REG, PIPE, IPv4, IPv6, etc.)
    """
    import os
    import subprocess
    
    pid = os.getpid()
    type_counts = {}
    
    # First, get total count from /proc (fast and reliable)
    try:
        fd_dir = f'/proc/{pid}/fd'
        if os.path.exists(fd_dir):
            total_fds = len(os.listdir(fd_dir))
            type_counts['total'] = total_fds
    except (OSError, PermissionError):
        pass
    
    # Try to get detailed breakdown from lsof, but don't block if it's slow
    try:
        result = subprocess.run(
            ['lsof', '-p', str(pid), '-F', 't'],
            capture_output=True,
            text=True,
            timeout=1  # Reduced timeout to 1 second
        )
        
        if result.returncode == 0:
            # Count by type (lines starting with 't' are type indicators)
            for line in result.stdout.split('\n'):
                if line.startswith('t'):
                    fd_type = line[1:]  # Remove 't' prefix
                    type_counts[fd_type] = type_counts.get(fd_type, 0) + 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # If lsof fails or times out, we still have the total count
        if 'total' not in type_counts:
            return {'error': 'lsof unavailable and /proc access failed'}
    except Exception as e:
        # Other errors - still return what we have
        if 'total' not in type_counts:
            return {'error': f'Exception counting FDs: {e}'}
    
    # Return what we have (at minimum, the total count)
    return type_counts if type_counts else {'error': 'Could not count FDs'}


def check_executor_workers(executor: Executor) -> dict:
    """Check health of ProcessPoolExecutor worker processes
    
    Args:
        executor: ProcessPoolExecutor instance (or any Executor)
        
    Returns:
        Dict with worker process health info, or error dict if unavailable
    """
    global _PSUTIL_WARNING_LOGGED
    if not PSUTIL_AVAILABLE:
        if not _PSUTIL_WARNING_LOGGED:
            logger = logging.getLogger(__name__)
            logger.warning(
                "psutil not available - cannot monitor executor worker health. "
                "Install psutil to enable worker process monitoring."
            )
            _PSUTIL_WARNING_LOGGED = True
        return {'error': 'psutil not available'}
    
    # Only check ProcessPoolExecutor, skip others
    from concurrent.futures import ProcessPoolExecutor
    if not isinstance(executor, ProcessPoolExecutor):
        return {'skipped': 'Not a ProcessPoolExecutor'}
    
    try:
        if not hasattr(executor, '_processes'):
            return {'error': 'Cannot access executor processes'}
        
        alive = 0
        dead = 0
        zombies = 0
        worker_pids = []
        
        for proc in executor._processes.values():
            if proc is None:
                continue
            
            try:
                pid = proc.pid
                worker_pids.append(pid)
                p = psutil.Process(pid)
                status = p.status()
                
                if status == psutil.STATUS_ZOMBIE:
                    zombies += 1
                elif p.is_running():
                    alive += 1
                else:
                    dead += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                dead += 1
            except Exception as e:
                # Catch any other unexpected exceptions from psutil
                logging.getLogger(__name__).debug(f"Error checking process {proc.pid if hasattr(proc, 'pid') else 'unknown'}: {e}")
                dead += 1
        
        return {
            'alive': alive,
            'dead': dead,
            'zombies': zombies,
            'worker_pids': worker_pids,
            'total_workers': len(executor._processes)
        }
    except Exception as e:
        # Catch any unexpected exceptions (e.g., AttributeError accessing _processes)
        logging.getLogger(__name__).debug(f"Error in check_executor_workers: {e}")
        return {'error': f'Exception checking workers: {e}'}


class CascadeAgent(Agent):
    """Base class for all cascade agents"""
    
    # Track resource creation across all agents
    _resource_counts = {
        'agents_created': 0,
        'traj_db_instances_created': 0,
    }

    def __init__(
        self,
        config: CascadeAgentConfig,
        executor: Optional[Executor] = None
    ):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = executor
        CascadeAgent._resource_counts['agents_created'] += 1
        self.logger.debug(
            f"Agent created. Total agents: {CascadeAgent._resource_counts['agents_created']}"
        )
    
    def schedule_future_callback(
        self,
        future: ExecutorFuture,
        callback: Callable[[ExecutorFuture], Awaitable[None]],
        *,
        description: Optional[str] = None,
    ) -> None:
        """Schedule a coroutine callback after a future completes.

        Args:
            future: Future from executor.submit() (concurrent.futures.Future)
            callback: Coroutine accepting the completed future.
            description: Optional identifier for logging (currently unused).
        """
        async def _run_callback() -> None:
            await wrap_future(future)
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

    @loop
    async def monitor_executor_health(self, shutdown: asyncio.Event) -> None:
        """Periodically check executor worker process health"""
        self.logger.info(f"Monitoring executor health for {self.config.run_id}")
        while not shutdown.is_set():
            try:
                # Check if executor exists
                if self.executor is None:
                    self.logger.debug("Executor is None, skipping health check")
                    await asyncio.sleep(30)
                    continue
                
                # Get worker health status
                worker_health = check_executor_workers(self.executor)
                self.logger.info(f"check result: {worker_health}")
                
                # Handle errors or skipped checks
                if 'error' in worker_health:
                    self.logger.debug(f"Executor health check error: {worker_health['error']}")
                    await asyncio.sleep(30)
                    continue
                
                if 'skipped' in worker_health:
                    self.logger.debug(f"Executor health check skipped: {worker_health['skipped']}")
                    await asyncio.sleep(30)
                    continue
                
                # Check for zombie processes
                zombies = worker_health.get('zombies', 0)
                if zombies > 0:
                    self.logger.warning(
                        f"Found {zombies} zombie worker processes: {worker_health}"
                    )
                    await asyncio.sleep(30)
                    continue
                
                # Check for many dead workers
                dead = worker_health.get('dead', 0)
                alive = worker_health.get('alive', 0)
                if dead > alive:
                    self.logger.warning(
                        f"Many dead workers detected: {worker_health}. Consider recreating executor."
                    )
                    await asyncio.sleep(30)
                    continue
                
                # All good - log normal health status
                self.logger.info(f"Executor worker health: {worker_health}")
                
            except Exception as e:
                self.logger.exception(f"Error in executor health monitoring: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds

class DynamicsEngine(CascadeAgent):

    def __init__(
        self,
        config: DynamicsEngineConfig,
        auditor: Handle[Auditor],
        executor: Executor,
        advance_dynamics_task: Callable[[AdvanceSpec], None]
    ):
        super().__init__(config, executor)
        self.weights = config.weights  # This needs to be mutable
        self.model_version = 0  # todo: mt.2025.10.20 probably this should be persisted somewhere else
        self.queue = Queue()
        self.auditor = auditor
        self.advance_dynamics_task = advance_dynamics_task

    # async def agent_on_startup(self):
    #     for spec in self.config.init_specs:
    #         await self.queue.put(spec)

    @action
    async def receive_weights(self, weights: bytes) -> None:
        self.weights = weights
        self.model_version += 1
        self.logger.info(f"Received new weights, now on model version {self.model_version}")

    @action
    async def submit(self, spec: AdvanceSpec):
        if isinstance(spec, dict):
            spec = AdvanceSpec(**spec)
        current_spec = spec
        self._traj_db.mark_trajectory_running(self.config.run_id, current_spec.traj_id)
        self.logger.info(f"Received advance spec for traj {current_spec.traj_id} chunk {current_spec.chunk_id}")
        description = f'advance dynamics traj {current_spec.traj_id} chunk {current_spec.chunk_id}'

        initial_future = self.executor.submit(
            self.advance_dynamics_task,
            spec=current_spec,
            learner=self.config.learner,
            weights=self.weights,
            db_url=self.config.db_url,
            device=self.config.device,
            dyn_cls=self.config.dyn_cls,
            dyn_kws=self.config.dyn_kws
        )

        self.schedule_future_callback(
            initial_future,
            self._advance_dynamics_callback,
            description=description,
        )
    async def _advance_dynamics_callback(
        self,
        future: Future
    ) -> None:
        """Handle the results of an advance dynamics call"""
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
        else:
            self.logger.error(f"Failed to record chunk {chunk_id} of traj {traj_id} in database")

        # submit to auditor
        self.logger.info(f"Submitting audit for chunk {chunk_id} of traj {traj_id}.")
        await self.auditor.submit(ChunkSpec(traj_id=traj_id, chunk_id=chunk_id))


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
        chunk_atoms = self._traj_db.get_latest_chunk_attempt_atoms(
            self.config.run_id,
            chunk_spec.traj_id,
            chunk_spec.chunk_id
        )
        # Force garbage collection after retrieving atoms
        gc.collect()
        self.logger.info(f'Submitting audit of chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id} to executor')
        description = f'audit traj {chunk_spec.traj_id} chunk {chunk_spec.chunk_id}'

        initial_future = self.executor.submit(
            self.audit_task,
            chunk_atoms=chunk_atoms,
            chunk_spec=chunk_spec,
            attempt_index=latest_attempt['attempt_index'],
        )

        self.schedule_future_callback(
            initial_future,
            self._audit_callback,
            description=description,
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

        self._traj_db.update_chunk_audit_status(
            run_id=self.config.run_id,
            traj_id=result.traj_id,
            chunk_id=result.chunk_id,
            attempt_index=result.attempt_index,
            audit_status=status
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
                # trajectory done, already marked in db 
                self.logger.info(f"Traj {result.traj_id} is complete")
            else:
                # Trajectory not done - submit next chunk
                # Get the last frame from the current chunk to use as starting point
                # Track file FDs before ASE DB operation
                fd_counts_before = count_file_descriptors_by_type()
                file_fds_before = fd_counts_before.get('REG', 0)
                
                last_frame = self._traj_db.get_last_frame_from_chunk(
                    run_id=self.config.run_id,
                    traj_id=result.traj_id,
                    chunk_id=result.chunk_id,
                    attempt_index=result.attempt_index
                )
                # Force garbage collection after retrieving atoms
                gc.collect()
                
                # Track file FDs after ASE DB operation
                fd_counts_after = count_file_descriptors_by_type()
                file_fds_after = fd_counts_after.get('REG', 0)
                
                if file_fds_after > file_fds_before:
                    self.logger.warning(
                        f"File FD increase after get_last_frame_from_chunk (traj {result.traj_id}, chunk {result.chunk_id}): "
                        f"{file_fds_before} -> {file_fds_after} (delta: +{file_fds_after - file_fds_before})"
                    )
                # Create and submit next advance spec
                next_chunk_id = result.chunk_id + 1
                next_attempt_index = self._traj_db.get_next_attempt_index(
                    run_id=self.config.run_id,
                    traj_id=result.traj_id,
                    chunk_id=next_chunk_id
                )
                next_spec = AdvanceSpec(
                    atoms=last_frame,
                    run_id=self.config.run_id,
                    traj_id=result.traj_id,
                    chunk_id=next_chunk_id,
                    attempt_index=next_attempt_index,
                    steps=self.config.chunk_size
                )
                await self.dynamics_engine.submit(next_spec)
                self.logger.info(f"Submitted next chunk {result.chunk_id + 1} for traj {result.traj_id}")
        else:
            # audit failed, submit to sampler
            self.logger.info(
                f'Audit failed for traj {result.traj_id} chunk {result.chunk_id} attempt {result.attempt_index}'
            )
            spec = ChunkSpec(
                traj_id=result.traj_id,
                chunk_id=result.chunk_id
            )
            self.logger.info(f'Submitting failed chunk {result.chunk_id} of traj {result.traj_id} to sampler')
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
            
            # Get frames from trajectory_frames table for this chunk and attempt
            atoms_list = self._traj_db.get_latest_chunk_attempt_atoms(
                run_id=self.config.run_id,
                traj_id=chunk_spec.traj_id,
                chunk_id=chunk_spec.chunk_id
            )
            # Force garbage collection after retrieving atoms
            gc.collect()
            
            if not atoms_list:
                self.logger.warning(f"No frames found for chunk {chunk_spec.chunk_id} of traj {chunk_spec.traj_id}")
                continue
            
            # Get frame IDs for the sampled frames
            # Use with_entities to only select the ID column, avoiding loading full ORM objects
            with self._traj_db.session() as sess:
                from cascade.agents.db_orm import DBTrajectoryFrame
                # Query only the ID and frame_index columns, then extract IDs in order
                frame_rows = sess.query(
                    DBTrajectoryFrame.id,
                    DBTrajectoryFrame.frame_index
                ).filter_by(
                    run_id=self.config.run_id,
                    traj_id=chunk_spec.traj_id,
                    chunk_id=chunk_spec.chunk_id,
                    attempt_index=db_chunk['attempt_index']
                ).order_by(DBTrajectoryFrame.frame_index).all()
                frame_ids = [row[0] for row in frame_rows]
            
            # Sample frames
            n_sample = min(self.config.n_frames, len(atoms_list))
            indices = self.rng.choice(len(atoms_list), size=n_sample, replace=False)
            sampled_frames = [atoms_list[i] for i in indices]
            sampled_frame_ids = [frame_ids[i] for i in indices]
            
            # Submit frames with their model version and trajectory frame IDs
            for frame, trajectory_frame_id in zip(sampled_frames, sampled_frame_ids):
                training_frame = TrainingFrame(
                    atoms=frame,
                    model_version=db_chunk['model_version']
                )
                self.logger.info(f'Submitting training frame from traj {chunk_spec.traj_id} chunk {chunk_spec.chunk_id} to labeler')
                await self.labeler.submit(training_frame, trajectory_frame_id)


class DummyLabeler(CascadeAgent):

    def __init__(
        self,
        config: LabelerConfig,
    ):
        super().__init__(config)
        self.queue = Queue()

    @action
    async def submit(self, training_frame: TrainingFrame, trajectory_frame_id: int) -> None:
        self.logger.info(f'Received training frame (trajectory_frame_id={trajectory_frame_id})')
        await self.queue.put((training_frame, trajectory_frame_id))

    @loop
    async def label_data(self, shutdown: asyncio.Event) -> None:

        while not shutdown.is_set():
            training_frame, trajectory_frame_id = await self.queue.get()
            
            # Get trajectory and chunk info from trajectory_frames table
            # Extract scalar values immediately to avoid holding ORM objects
            try:
                traj_id = None
                chunk_id = None
                attempt_index = None
                with self._traj_db.session() as sess:
                    from cascade.agents.db_orm import DBTrajectoryFrame
                    frame = sess.query(DBTrajectoryFrame).filter_by(id=trajectory_frame_id).first()
                    if not frame:
                        self.logger.warning(f"Could not find trajectory frame ID {trajectory_frame_id}")
                        continue
                    # Extract scalar values immediately while session is active
                    traj_id = frame.traj_id
                    chunk_id = frame.chunk_id
                    attempt_index = frame.attempt_index
                
                self._traj_db.add_training_frame(
                    run_id=self.config.run_id,
                    trajectory_frame_id=trajectory_frame_id,
                    model_version_sampled_from=training_frame.model_version,
                    traj_id=traj_id,
                    chunk_id=chunk_id,
                    attempt_index=attempt_index
                )
            except Exception as e:
                self.logger.warning(f"Could not add training frame for trajectory_frame_id {trajectory_frame_id}: {e}, skipping training frame")
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
        """Monitor for enough training frames and trigger retraining"""
        self.logger.info("periodic_retrain loop started")
        while not shutdown.is_set():
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
            
            # Monitor file descriptor counts by type for main process
            import os
            try:
                pid = os.getpid()
                fd_counts = count_file_descriptors_by_type()
                
                if 'error' not in fd_counts:
                    total = fd_counts.get('total', 0)
                    
                    # Always log total count (from /proc, fast and reliable)
                    self.logger.info(f"FD total count: {total}")
                    
                    # Log detailed breakdown if available (from lsof)
                    pipes = fd_counts.get('PIPE', 0)
                    files = fd_counts.get('REG', 0)
                    sockets = fd_counts.get('IPv4', 0) + fd_counts.get('IPv6', 0)
                    
                    if pipes > 0 or files > 0 or sockets > 0:
                        self.logger.info(
                            f"FD breakdown: pipes={pipes}, files={files}, sockets={sockets}"
                        )
                    
                    # Alert on high FD counts (separate warning lines)
                    if total > 500:
                        self.logger.warning(
                            f"FD WARNING: High total FD count: {total}. Possible FD leak detected!"
                        )
                    if pipes > 100:
                        self.logger.warning(
                            f"FD WARNING: High pipe count: {pipes}. ProcessPoolExecutor may have issues."
                        )
                else:
                    # Only log at debug level if FD counting fails - don't spam logs
                    self.logger.debug(f"FD counting unavailable: {fd_counts.get('error', 'unknown')}")
            except Exception as e:
                # Don't let FD counting failures break the retrain loop
                self.logger.debug(f"FD monitoring error: {e}")
            
            # Check executor worker health if executor is available
            if self.executor is not None:
                worker_health = check_executor_workers(self.executor)
                if 'error' not in worker_health and 'skipped' not in worker_health:
                    if worker_health.get('zombies', 0) > 0:
                        self.logger.warning(
                            f"Found {worker_health['zombies']} zombie worker processes: {worker_health}"
                        )
                    elif worker_health.get('dead', 0) > worker_health.get('alive', 0):
                        self.logger.warning(
                            f"Many dead workers detected: {worker_health}. Consider recreating executor."
                        )
                    else:
                        self.logger.debug(f"Executor worker health: {worker_health}")
            
            # Log connection pool statistics for debugging FD leaks
            self._traj_db.log_pool_stats(self.logger)
            
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
                            attempt_index=prev_chunk_attempt['attempt_index']
                        )
                    else:
                        # chunk_id == 0: use initial trajectory frame
                        start_frame = self._traj_db.get_initial_trajectory_frame(
                            run_id=self.config.run_id,
                            traj_id=traj_id
                        )
                    # Force garbage collection after retrieving atoms
                    gc.collect()
                    
                    if start_frame:
                        # Resubmit the SAME chunk_id (dynamics engine will create a new attempt)
                        next_attempt_index = self._traj_db.get_next_attempt_index(
                            run_id=self.config.run_id,
                            traj_id=traj_id,
                            chunk_id=chunk_id
                        )
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
                            f"Traj {traj_id}: No starting frame found for chunk {chunk_id} "
                            f"({'initial trajectory' if chunk_id == 0 else f'previous chunk {chunk_id - 1}'})"
                        )
                
                self.last_train_count = current_count
                self.logger.info(f"Retraining complete, setting frame count to {current_count}")
            else:
                self.logger.debug(f"Retraining not triggered, sleeping for 5 seconds")
                await asyncio.sleep(5)
