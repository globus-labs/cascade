"""Academy agents that implement Cascade

These implement the communication patterns that coordiante a cascade run.
All agents have an Executor where work is done, no work is done in the agents themselves.
All agents are passed a task in their configurations, which is run on the Executor.
"""
from __future__ import annotations

import asyncio
from asyncio import Event, Lock, wrap_future
from functools import partial
import logging
from copy import deepcopy

import numpy as np

from academy.handle import Handle
from academy.agent import Agent, action, loop
from academy.exception import AgentTerminatedError

from cascade.model import AuditStatus, AdvanceSpec, AuditResult, TrajectoryStatus, ChunkEventType
from cascade.agents.config import (
    AuditorConfig,
    SamplerConfig,
    LabelerConfig,
    DatabaseMonitorConfig,
    DynamicsRunnerConfig,
    ControllerConfig,
)
from cascade.agents.db_orm import TrajectoryDB
from cascade.agents.task import audit_with_random_failure
from cascade.model import Chunk, TrainingFrame


class CascadeAgent(Agent):
    """Base class for all cascade agents"""

    async def agent_on_startup(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._traj_db = TrajectoryDB(self.db_url, logger=self.logger)
        self._traj_db.create_tables()


class DynamicsRunner(CascadeAgent):
    """Runs a dynamics task in a loop until done, or a shutdown message is received

    There is one DynamicsRunner per trajectory.
    The trajectory is advanced in chunks of time, which have configurable length.
    Chunks are passed to an Auditor agent. If the audit fails, the DynamicsRunner waits for new model weights
    before trying to run the chunk again.
    """
    def __init__(
        self,
        auditor: Handle[Auditor],
        config: DynamicsRunnerConfig
    ):
        self.db_url = config.db_url
        super().__init__()
        self.config = config
        self.auditor = auditor

        # pull out variables that may change from config
        self.atoms = config.atoms.copy()
        self.init_chunk_size = config.chunk_size
        self.chunk_size = config.chunk_size
        self.model_version = config.model_version
        self.weights = config.weights

        # track progress
        self.timestep = 0
        self.chunk_ix = 0
        self.attempt = 0
        self.done = False

        # for handling weight updates
        self.received_weights = Event()
        self.new_model_lock = Lock()
        self.new_model: tuple[list[bytes], int] | None = None  # weights, version

    @loop
    async def run(
            self,
            shutdown: asyncio.Event,
    ) -> None:
        """Run dynamics until done, or a shutdown message is received"""
        while not (shutdown.is_set() or self.done):

            spec = AdvanceSpec(
                atoms=self.atoms,
                steps=self.chunk_size,
                run_id=self.config.run_id,
                traj_id=self.config.traj_id,
                chunk_id=self.chunk_ix,
                attempt_index=self.attempt,
            )

            self.logger.info(f"Running dynamics for traj {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index} with {spec.steps} steps")

            # there are two conditions to release this lock: 1. we finish this pass over the trajecotyr chunk
            async with self.new_model_lock:

                if self.new_model:
                    self.weights, self.model_version = self.new_model
                    self.new_model = None
                self.logger.debug(
                    f"Submitting dynamics to executor dynamics for traj {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index} with {spec.steps} steps")
                self._traj_db.record_chunk_event(
                    run_id=self.config.run_id,
                    traj_id=spec.traj_id,
                    chunk_id=spec.chunk_id,
                    attempt_index=spec.attempt_index,
                    event_type=ChunkEventType.STARTED_DYNAMICS,
                )
                # submit dynamics for evaluation
                chunk_future = self.config.executor.submit(
                    self.config.advance_dynamics_task,
                    spec=spec,
                    learner=self.config.learner,
                    weights=self.weights,
                    device=self.config.device,
                    dyn_cls=self.config.dyn_cls,
                    dyn_kws=self.config.dyn_kws,
                    run_kws=self.config.run_kws,
                    run_dir=str(self.config.run_dir),
                    uq_hook=self.config.uq_hook,
                    uq_kws=self.config.uq_kws,
                    gpu_flush_interval=self.config.gpu_flush_interval,
                )

            #todo mt.2026.04.27 does this still need to be logged?
            self._traj_db.add_chunk_attempt(
                run_id=self.config.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                model_version=self.model_version,
                n_frames=self.chunk_size,
                audit_status=AuditStatus.PENDING,
                attempt_index=self.attempt
            )

            # get future result
            wrapped_future = wrap_future(chunk_future)
            try:
                await wrapped_future
                chunk_atoms = wrapped_future.result()
            except Exception as exc:
                reason = f"dynamics failed for chunk {spec.chunk_id} attempt {spec.attempt_index}: {exc!r}"
                self.logger.error(f"Traj {self.config.traj_id} {reason}")
                self._traj_db.mark_trajectory_failed(run_id=self.config.run_id, traj_id=self.config.traj_id, reason=reason)
                self.done = True
                self.agent_shutdown()
                continue

            # write atoms # todo wrap this up
            frame_ids = []
            for frame_index, _atoms in enumerate(chunk_atoms):
                frame_index += self.timestep
                _id = self._traj_db.write_frame(
                    run_id=spec.run_id,
                    traj_id=spec.traj_id,
                    chunk_id=spec.chunk_id,
                    attempt_index=spec.attempt_index,
                    frame_index=frame_index,
                    atoms=_atoms
                )
                frame_ids.append(_id)
            self.logger.info(f"Finished dynamics for traj {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")
            self._traj_db.record_chunk_event(
                run_id=self.config.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                attempt_index=spec.attempt_index,
                event_type=ChunkEventType.FINISHED_DYNAMICS,
            )

            # submit to auditor
            chunk = Chunk(
                atoms=chunk_atoms,
                frame_ids=frame_ids,
                traj_id=self.config.traj_id,
                chunk_id=self.chunk_ix,
                attempt_ix=self.attempt,
                model_version=self.model_version
            )
            self.logger.info(f"Submitting audit for traj {self.config.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}")
            self._traj_db.record_chunk_event(
                run_id=self.config.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                attempt_index=spec.attempt_index,
                event_type=ChunkEventType.STARTED_AUDIT,
            )
            audit_result = await self.auditor.audit(chunk)
            self._traj_db.record_chunk_event(
                run_id=self.config.run_id,
                traj_id=spec.traj_id,
                chunk_id=spec.chunk_id,
                attempt_index=spec.attempt_index,
                event_type=ChunkEventType.AUDIT_PASSED if audit_result.status == AuditStatus.PASSED else ChunkEventType.AUDIT_FAILED,
            )

            # handle audit result
            if audit_result.status == AuditStatus.PASSED:

                self.logger.info(f"Audit status passed for traj {self.config.traj_id} chunk {self.chunk_ix} attempt {self.attempt}")
                self.timestep += self.chunk_size
                self.logger.info(f"On timestep {self.timestep} of {self.config.n_steps}")
                self.done = self.timestep >= self.config.n_steps
                if self.done:
                    # audit passed and trajectory is complete: shutdown
                    self.logger.info(f"Finished dynamics for traj {self.config.traj_id} chunk {self.chunk_ix} attempt {self.attempt}, shutting down")
                    self._traj_db.mark_trajectory_completed(run_id=self.config.run_id, traj_id=self.config.traj_id)
                    self.agent_shutdown()
                else:
                    # audit passed but not done: use the new atoms to run a new chunk in next pass of while loop
                    self.atoms = chunk_atoms[-1]
                    self.chunk_ix += 1
                    self.attempt = 0
                    self.logger.info(f"Updating traj {self.config.traj_id} to chunk {self.chunk_ix} attempt {self.attempt}")
            else:
                self.attempt += 1
                if self.config.max_audit_retries is not None and self.attempt > self.config.max_audit_retries:
                    reason = (
                        f"chunk {self.chunk_ix} failed audit {self.attempt} times "
                        f"(max_audit_retries={self.config.max_audit_retries}); "
                        f"last audit reason={audit_result.reason}"
                    )
                    self.logger.error(f"Traj {self.config.traj_id} exceeded max_audit_retries: {reason}")
                    self._traj_db.mark_trajectory_failed(run_id=self.config.run_id, traj_id=self.config.traj_id, reason=reason)
                    self.done = True
                    self.agent_shutdown()
                else:
                    # audit failed: wait for new weights
                    self.logger.info(f'Audit status failed for traj {self.config.traj_id} chunk {self.chunk_ix} attempt {self.attempt}, waiting for new weights...')
                    self.received_weights.clear()
                    await self.received_weights.wait()
                    self.logger.info('Received new weights')

    @action
    async def receive_weights(self, weights: list[bytes], model_version: int) -> None:
        async with self.new_model_lock: # todo mt.2026.07.07: do we need this lock if we only call receive weights from a safe spot in the loop in this agent?
            self.new_model = (weights, model_version)
        self.logger.info(f"Received weights for model version {model_version}")
        self.received_weights.set()


class Auditor(CascadeAgent):
    """Accepts or rejects a chunk based on an audit_task
    If the chunk is rejected, it is passed to the sampler to generate training frames.
    """

    def __init__(
            self,
            sampler: Handle[Sampler],
            config=AuditorConfig
    ):
        self.db_url = config.db_url
        super().__init__()
        self.config = config
        self.sampler = sampler
        self.audit_task = config.audit_task
        if config.random_fail_rate > 0:
            self.audit_task = partial(audit_with_random_failure, audit_task=self.audit_task, fail_rate=config.random_fail_rate)
        self.default_threshold = config.audit_kws.get('threshold')
        self.thresholds: dict[int, float] = {}

    @action
    async def receive_threshold(self, threshold: float, traj_id: int | None = None) -> None:
        """Pushed by Controller after each recalibration.

        traj_id=None updates the shared default used by any trajectory without its
        own calibrated value; otherwise updates only that trajectory's threshold.
        """
        if traj_id is None:
            self.default_threshold = threshold
            self.logger.info(f"Received new default audit threshold {threshold}")
        else:
            self.thresholds[traj_id] = threshold
            self.logger.info(f"Received new audit threshold {threshold} for traj {traj_id}")

    @action
    async def audit(self, chunk: Chunk) -> AuditResult:
        """Submit a chunk for audit"""
        self.logger.info(f'Submitting audit of traj {chunk.traj_id} chunk {chunk.chunk_id} attempt {chunk.attempt_ix} to executor')

        audit_kws = {**self.config.audit_kws}
        if 'threshold' in self.config.audit_kws:
            audit_kws['threshold'] = self.thresholds.get(chunk.traj_id, self.default_threshold)

        future = self.config.executor.submit(
            self.audit_task,
            chunk,
            **audit_kws
        )
        wrapped_future = wrap_future(future)
        await wrapped_future
        result = wrapped_future.result()
        status = result.status

        self._traj_db.update_chunk_audit_done_status(
            run_id=self.config.run_id,
            traj_id=chunk.traj_id,
            chunk_id=chunk.chunk_id,
            attempt_index=chunk.attempt_ix,
            audit_status=status,
            audit_reason=result.reason,
        )
        if status == AuditStatus.PASSED:
            self.logger.info(
                f'Audit passed for traj {chunk.traj_id} chunk {chunk.chunk_id} attempt {chunk.attempt_ix}'
            )
        else:
            # audit failed, submit to sampler
            self.logger.info(
                f'Audit failed for traj {chunk.traj_id} chunk {chunk.chunk_id} attempt {chunk.attempt_ix}'
            )
            self.logger.info(f'Submitting failed chunk {chunk.chunk_id} of traj {chunk.traj_id} to sampler')
            asyncio.create_task(self.sampler.sample_frames(
                chunk,
                audit_reason=result.reason,
                audit_threshold=audit_kws.get('threshold'),
            ))
        return result


class Controller(CascadeAgent):
    """Updates the Auditor's UQ threshold against observed labeling error.

    Based on the alpha and threshold updates from cascade.proxima.SerialLearningCalculator.

    When config.per_trajectory_threshold is set, each trajectory gets its own
    independently calibrated alpha/threshold, otherwise they are shared.
    """

    def __init__(
        self,
        config: ControllerConfig,
        auditor: Handle[Auditor],
    ):
        self.db_url = config.db_url
        super().__init__()
        self.config = config
        self.auditor = auditor
        self.threshold: dict[int | None, float] = {}
        self.alpha: dict[int | None, float] = {}

    @action
    async def update_threshold(self, traj_id: int) -> None:
        """Recalibrate threshold/alpha from labeled frames, if enough observations exist.

        Called by: Labeler, on every newly labeled frame
        Invokes: Auditor (to update threshold)
        """
        key = traj_id if self.config.per_trajectory_threshold else None

        model_version, observations = self._traj_db.get_controller_observations(
            run_id=self.config.run_id,
            burn_in_model_versions=self.config.burn_in_model_versions,
            limit=self.config.history_length,
            traj_id=key,
        )
        if len(observations) < self.config.history_length:
            self.logger.info(
                f"Too few calibration observations ({len(observations)} < "
                f"{self.config.history_length}) for "
                f"{'traj ' + str(key) if key is not None else 'shared'} threshold; "
                f"stays at {self.threshold.get(key)}"
            )
            return

        uncert_metrics, obs_errors = zip(*observations)

        if np.allclose(uncert_metrics, 0.):
            # Happens e.g. when all ensemble members still share the same weights
            self.logger.info('All calibration UQ metrics are zero; setting threshold to zero')
            self.threshold[key] = 0.
        else:
            many_alphas = np.true_divide(obs_errors, np.clip(uncert_metrics, 1e-6, a_max=np.inf))
            alpha = float(np.mean(many_alphas))
            assert alpha >= 0
            self.alpha[key] = alpha

            if key not in self.threshold:
                # initial, conservative estimate (todo: make this tuneable?)
                self.threshold[key] = self.config.target_ferr / alpha / 2
            else:
                current_err = float(np.mean(obs_errors))
                self.threshold[key] -= (current_err - self.config.target_ferr) / alpha
                self.threshold[key] = max(self.threshold[key], 0.)

        mean_error = float(np.mean(obs_errors))
        self._traj_db.write_controller_log(
            run_id=self.config.run_id,
            model_version=model_version,
            threshold=self.threshold[key],
            alpha=self.alpha.get(key, 0.),
            mean_error=mean_error,
            n_observations=len(observations),
            traj_id=key,
        )
        self.logger.info(
            f"Calibrated threshold={self.threshold[key]:.4g}, alpha={self.alpha.get(key)}, "
            f"mean_error={mean_error:.4g}, n_observations={len(observations)}"
        )
        await self.auditor.receive_threshold(self.threshold[key], traj_id=key)


class Sampler(CascadeAgent):
    """Generates training frames based on a trajectory chunk"""
    def __init__(
        self,
        config: SamplerConfig,
        labeler: Handle[Labeler],
    ):
        self.db_url = config.db_url
        super().__init__()
        self.config = config
        self.labeler = labeler
        self.n_frames = config.n_frames

    @action
    async def sample_frames(
        self,
        chunk: Chunk,
        audit_reason: str | None = None,
        audit_threshold: float | None = None,
    ) -> None:

        self.logger.info(
            f'Sampling frames from traj {chunk.traj_id} '
            f'chunk {chunk.chunk_id} '
            f'attempt {chunk.attempt_ix}'
        )
        n_frames = self.config.n_frames
        if chunk.model_version < self.config.burn_in_model_versions and self.config.burn_in_n_frames is not None:
            n_frames = self.config.burn_in_n_frames

        chunk_kws = dict(
            run_id=self.config.run_id,
            traj_id=chunk.traj_id,
            chunk_id=chunk.chunk_id,
            attempt_index=chunk.attempt_ix,
        )
        self._traj_db.record_chunk_event(**chunk_kws, event_type=ChunkEventType.STARTED_SAMPLING)
        sample_kws = dict(n_frames=n_frames, reason=audit_reason)
        if audit_threshold is not None:
            # omit: some strategies have defaults we dont want to override with None
            sample_kws['threshold'] = audit_threshold
        future = self.config.executor.submit(
            self.config.sample_task,
            chunk,
            **sample_kws,
        )
        wrapped_future = wrap_future(future)
        await wrapped_future
        training_frames = wrapped_future.result()
        self._traj_db.record_chunk_event(**chunk_kws, event_type=ChunkEventType.FINISHED_SAMPLING)

        if len(training_frames) != n_frames:
            self.logger.warning(
                "Sampling returned %d frames for traj %s chunk %s (attempt %s), "
                "expected n_frames=%d",
                len(training_frames),
                chunk.traj_id,
                chunk.chunk_id,
                chunk.attempt_ix,
                n_frames,
            )
        for frame in training_frames:
            self.logger.info(
                f'Submitting training frame from traj {chunk.traj_id} '
                f'chunk {chunk.chunk_id} attempt {chunk.attempt_ix} to labeler'
            )
            asyncio.create_task(self.labeler.label_data(frame))


class Labeler(CascadeAgent):
    """Labels training frames"""

    def __init__(
        self,
        config: LabelerConfig,
        controller: Handle[Controller] | None = None,
    ):
        self.db_url = config.db_url
        super().__init__()
        self.config = config
        self.controller = controller
        self.error_fn = config.error_fn

    def _record_labeling_started(self, frame: TrainingFrame) -> None:
        # todo: discuss with will. wouldnt a pub/sub be better than DB for communicating this information. this is essentially a pub/sub spoof
        chunk_kws = dict(
            run_id=self.config.run_id,
            traj_id=frame.traj_id,
            chunk_id=frame.chunk_id,
            attempt_index=frame.attempt_index,
        )
        if not self._traj_db.has_chunk_event(**chunk_kws, event_type=ChunkEventType.STARTED_LABELING):
            self._traj_db.record_chunk_event(**chunk_kws, event_type=ChunkEventType.STARTED_LABELING)
        self._traj_db.record_chunk_event(**chunk_kws, event_type=ChunkEventType.STARTED_LABELING_FRAME, frame_id=frame.frame_id)

    def _record_labeling_finished(self, frame: TrainingFrame) -> None:
        chunk_kws = dict(
            run_id=self.config.run_id,
            traj_id=frame.traj_id,
            chunk_id=frame.chunk_id,
            attempt_index=frame.attempt_index,
        )
        self._traj_db.add_training_frame(
            **chunk_kws,
            trajectory_frame_id=frame.frame_id,
            model_version_sampled_from=frame.model_version,
            atoms_labeled=frame.atoms_labeled,
            calibration_uq=frame.atoms.info.get(self.config.uq_field),
            calibration_error=self.error_fn(frame.atoms, frame.atoms_labeled),
        )
        self._traj_db.record_chunk_event(**chunk_kws, event_type=ChunkEventType.FINISHED_LABELING_FRAME, frame_id=frame.frame_id)

        labeled_count = self._traj_db.count_labeled_frames_for_chunk(**chunk_kws)
        self.logger.info(
            f"Finished labeleing traj={frame.traj_id}, "
            f"chunk={frame.chunk_id}, attempt={frame.attempt_index};"
            f"labled from chunk={labeled_count}, sampled from chunk:{frame.n_sampled_frames}"
        )
        if labeled_count == frame.n_sampled_frames-1:  # recall the chunk stores an initial frame which wont get labeled
            self._traj_db.record_chunk_event(**chunk_kws, event_type=ChunkEventType.FINISHED_LABELING)
        self.logger.info(
            f"Added training frame to database: traj={frame.traj_id}, "
            f"chunk={frame.chunk_id}, attempt={frame.attempt_index}, "
            f"model_version={frame.model_version}"
        )

    @action
    async def label_data(self, frame: TrainingFrame) -> None:
        self._record_labeling_started(frame)

        frame_future = self.config.executor.submit(
            self.config.label_task,
            frame,
            self.config.calc_factory
        )
        wrapped_future = wrap_future(frame_future)
        await wrapped_future
        frame = wrapped_future.result()

        self._record_labeling_finished(frame)
        if self.controller is not None:
            asyncio.create_task(self.controller.update_threshold(traj_id=frame.traj_id))


class Trainer(CascadeAgent):
    """Produces new model weights"""

    def __init__(self, config):
        self.db_url = config.db_url
        super().__init__()
        self.config = config
        self.weights = deepcopy(config.weights)

    def _filter_unphysical_frames(self, frames):
        """Drop frames whose minimum interatomic distance is below a physically-plausible
        threshold (e.g. from a collapsing/unstable cell), so training never sees collided
        structures."""
        kept = []
        for atoms in frames:
            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)
            if dists.min() >= self.config.min_interatomic_distance:
                kept.append(atoms)
        n_dropped = len(frames) - len(kept)
        if n_dropped:
            self.logger.warning(
                f'Dropped {n_dropped}/{len(frames)} training frames with min interatomic '
                f'distance < {self.config.min_interatomic_distance} A'
            )
        return kept

    @staticmethod
    def _training_diverged(log) -> bool:
        """Whether a member's returned training log shows the run diverged (non-finite loss)."""
        loss_col = 'total_loss_valid' if 'total_loss_valid' in log.columns else 'total_loss_train'
        return not np.isfinite(log[loss_col].iloc[-1])

    async def _train_member(self, member_index, member_weights, train_data, valid_data, n_sample, rng):
        """Train one ensemble member, retrying with a fresh bootstrap draw if it diverges.
        Falls back to this member's current weights (logging the diverged run for
        observability) if it's still diverged after max_training_retries."""
        for attempt in range(self.config.max_training_retries + 1):
            boot_idx = rng.integers(0, len(train_data), size=n_sample)
            boot_data = [train_data[i] for i in boot_idx]
            future = self.config.executor.submit(
                self.config.training_task,
                learner=self.config.learner,
                weights=member_weights,
                train_data=boot_data,
                valid_data=valid_data,
                train_kws=self.config.training_kws,
                replay=self.config.replay,
            )
            new_weights, log = await wrap_future(future)
            if not self._training_diverged(log):
                return new_weights, log
            remaining = self.config.max_training_retries - attempt
            self.logger.warning(
                f'Member {member_index} training diverged on attempt {attempt + 1}/'
                f'{self.config.max_training_retries + 1}'
                + (f'; retrying ({remaining} attempt(s) left)' if remaining > 0 else '; keeping previous weights')
            )
        return member_weights, log

    @action
    async def train_model(
        self,
        training_round: int,
    ) -> list[bytes]:
        from sklearn.model_selection import train_test_split

        self.logger.info(f'Fetching training data for training round {training_round}')
        train_data = self._traj_db.get_training_frames(
            self.config.run_id,
            training_round=training_round,
        )
        if not train_data:
            self.logger.warning(f'No training frames found for round {training_round}, skipping training')
            return self.weights
        self.logger.info(f'Got {len(train_data)} training frames')
        train_data = self._filter_unphysical_frames(train_data)
        if not train_data:
            self.logger.warning(f'All training frames for round {training_round} were filtered as unphysical, skipping training')
            return self.weights
        train_data, valid_data = train_test_split(train_data, test_size=0.2)
        self.logger.info(f'Train size: {len(train_data)}, val size: {len(valid_data)}')

        rng = np.random.default_rng()
        n_sample = int(len(train_data) * self.config.bootstrap_fraction)

        self.logger.info(f'Submitting {len(self.weights)} bootstrapped training tasks')
        futures = [
            self._train_member(member_index, member_weights, train_data, valid_data, n_sample, rng)
            for member_index, member_weights in enumerate(self.weights)
        ]
        results = await asyncio.gather(*futures)

        self.logger.info('Retrieving new weights')
        new_weights = [w for w, _ in results]
        for member_index, (_, log) in enumerate(results):
            self._traj_db.write_training_log(self.config.run_id, training_round, log, member_index=member_index)
        self.weights = new_weights
        return new_weights


class DatabaseMonitor(CascadeAgent):
    """Monitors the database for training triggers and completion"""

    def __init__(
        self,
        config: DatabaseMonitorConfig,
        trainer: Handle[Trainer],
        dynamics_runners: list[Handle[DynamicsRunner]]
    ):
        self.db_url = config.db_url
        self.config = config
        super().__init__()
        self.trainer = trainer
        self.dynamics_runners = dynamics_runners
        self.last_train_count = 0
        self.current_training_round = 0
        self.model_version = 0

        self.retrain_len = self.config.retrain_len
        self.retrain_fraction = self.config.retrain_fraction

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
            current_count = self._traj_db.count_training_frames(self.config.run_id)
            new_frames = current_count - self.last_train_count

            # Check fraction-based condition
            total_active, active_with_labeling = self._traj_db.count_active_trajs_with_labeling(
                run_id=self.config.run_id
            )
            # todo: clarify names here, sampling and labling are confused
            sampled_fraction = active_with_labeling / total_active if total_active > 0 else 0.

            # Determine which condition triggered retraining
            absolute_condition = new_frames >= self.retrain_len
            fraction_condition = sampled_fraction >= self.retrain_fraction
            should_retrain = absolute_condition or fraction_condition

            self.logger.info(
                f"Retrain check: new training frames={new_frames}, active trajectories={total_active}, trajectories with labeled frames={active_with_labeling}, "
                f"fraction={sampled_fraction:.2%}, should_retrain={should_retrain}"
            )

            if should_retrain:
                trigger_reason = []
                if absolute_condition:
                    trigger_reason.append(f"absolute threshold ({new_frames} >= {self.retrain_len})")
                if fraction_condition:
                    trigger_reason.append(f"fraction threshold ({sampled_fraction:.2%} >= {self.retrain_fraction:.2%})")

                # Get the training round for frames that will be used in this retraining
                # (frames created before this retraining will have the current max training_round)
                self.logger.info(
                    f"Starting retraining (round {self.current_training_round}) triggered by: {', '.join(trigger_reason)}\n"
                    f"Training frame count: current={current_count}, last_train={self.last_train_count}, "
                    f"new={new_frames}, active_trajs={total_active}, labeled_trajs={active_with_labeling}, "
                    f"fraction={sampled_fraction:.2%}"
                )
                # Stamp all unlabeled frames with the current round before training
                self._traj_db.mark_training_frames_for_round(
                    self.config.run_id,
                    training_round=self.current_training_round,
                )
                # Train model and update weights in dynamics engine
                self._traj_db.record_training_event(
                    run_id=self.config.run_id,
                    event_type=ChunkEventType.STARTED_TRAINING,
                    training_round=self.current_training_round
                )
                weights = await self.trainer.train_model(self.current_training_round)
                # Record FINISHED_TRAINING event after training completes
                self._traj_db.record_training_event(
                    run_id=self.config.run_id,
                    event_type=ChunkEventType.FINISHED_TRAINING,
                    training_round=self.current_training_round
                )
                self.current_training_round += 1
                self.last_train_count = current_count

                self.model_version += 1

                for runner in self.dynamics_runners:
                    try:
                        await runner.receive_weights(weights, self.model_version)
                    except AgentTerminatedError:
                        pass
