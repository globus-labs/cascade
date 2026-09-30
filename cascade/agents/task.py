from __future__ import annotations

import time
from pathlib import Path
import logging
import os

import numpy as np
from ase.optimize.optimize import Dynamics
from mace.calculators import mace_mp
import torch

from cascade.model import AuditResult, AuditStatus, TrajectoryDiverged
from cascade.utils import canonicalize
from cascade.traj_config import extract_dyn_state, restore_dyn_state

_module_logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    from cascade.model import AuditResult, Chunk
    from cascade.model import AdvanceSpec, TrainingFrame
    from cascade.learning.base import BaseLearnableForcefield
    from cascade.calculator import Calculator
    from cascade.learning.finetuning import MultiHeadConfig
    from ase import Atoms
    from pathlib import Path
    import numpy as np
    import pandas as pd


# can make this a classmethod on some audittask class
# to get some shared informaiton and inheritance
def random_audit(
    chunk: Chunk,
    accept_prob: float = 0.5,
    sleep_time: float = 0.,
) -> AuditResult:
    """Random audit of a chunk of a trajectory

    Intended to be used as a stub for a real audit function.
    """

    time.sleep(sleep_time)
    # Create a new random generator seeded with OS entropy to ensure
    # each worker process gets a unique random state
    rng = np.random.default_rng(seed=None)
    passed = rng.random() < accept_prob
    score = rng.random() if passed else 0.0
    status = AuditStatus.PASSED if passed else AuditStatus.FAILED
    return AuditResult(status=status, score=score, reason='random_accept')


def _frames_from_indices(chunk: Chunk, indices, n_sample: int) -> list[TrainingFrame]:
    """Build one TrainingFrame per index into chunk.atoms/frame_ids.

    Used by sampling tasks to create training frames for labeler
    """
    from cascade.model import TrainingFrame

    return [
        TrainingFrame(
            atoms=chunk.atoms[i],
            frame_id=chunk.frame_ids[i],
            model_version=chunk.model_version,
            traj_id=chunk.traj_id,
            chunk_id=chunk.chunk_id,
            attempt_index=chunk.attempt_ix,
            n_sampled_frames=n_sample,
        )
        for i in indices
    ]


def random_sample(
    chunk: Chunk,
    n_frames: int,
    sleep_time: float = 0.,
    **kwargs
) -> list[TrainingFrame]:
    """Random sample of frames from a chunk.

    Intended to be used as a stub for a real sampling function.
    """
    import time

    time.sleep(sleep_time)
    # Create a new random generator seeded with OS entropy to ensure
    # each worker process gets a unique random state
    rng = np.random.default_rng(seed=None)
    n_sample = min(n_frames, len(chunk.atoms))
    indices = rng.choice(len(chunk.atoms), size=n_sample, replace=False)
    return _frames_from_indices(chunk, indices, n_sample)


def max_uq_sample(
    chunk: Chunk,
    n_frames: int,
    *,
    field: str = 'uq_force_std_max',
    **kwargs
) -> list[TrainingFrame]:
    """The n_frames frames with the highest per-frame UQ score.

    Requires the chunk's frames to have been produced with a uq_hook (e.g.
    ensemble_force_deviation_uq) that populates atoms.info[field].
    """
    values = np.array([a.info[field] for a in chunk.atoms])
    n_sample = min(n_frames, len(chunk.atoms))
    indices = np.argsort(-values)[:n_sample]
    return _frames_from_indices(chunk, indices, n_sample)


def boundary_uq_sample(
    chunk: Chunk,
    n_frames: int,
    threshold: float = 0.1,
    field: str = 'uq_force_std_max',
    **kwargs
) -> list[TrainingFrame]:
    """Frames leading up to and including the first frame that crossed threshold.

    Never selects frames after the crossing -- anything past the point divergence
    started is likely nonsense and not useful training data.
    """
    values = np.array([a.info[field] for a in chunk.atoms])
    crossings = np.flatnonzero(values >= threshold)
    n_sample = min(n_frames, len(chunk.atoms))
    crossing_idx = int(crossings[0])
    start = max(0, crossing_idx - n_sample + 1)
    indices = list(range(start, crossing_idx + 1))
    return _frames_from_indices(chunk, indices, n_sample)


def audit_reason_sample(
    chunk: Chunk,
    n_frames: int,
    *,
    reason: str | None = None,
    threshold: float = 0.1,
    field: str = 'uq_force_std_max',
    burn_in_sampler: Callable[..., list[TrainingFrame]] = random_sample,
    threshold_sampler: Callable[..., list[TrainingFrame]] = boundary_uq_sample,
    random_sampler: Callable[..., list[TrainingFrame]] = max_uq_sample,
) -> list[TrainingFrame]:
    """Dispatches to different smapling methods based on audit failure reason

    burn_in_sampler: when the audit failure reasion is "burn_in"
    threshold_sampler: when the audit failure reason is "threshold"
    random_sampler: when the audit failure reason is "random_fail"
    """
    strategy = {
        'burn_in': burn_in_sampler,
        'threshold': threshold_sampler,
        'random_fail': random_sampler,
    }.get(reason)
    return strategy(chunk, n_frames, threshold=threshold, field=field)


def advance_dynamics(
    spec: AdvanceSpec,
    learner: BaseLearnableForcefield,
    weights: list[bytes],
    device: str,
    run_dir: str,
    dyn_cls: type[Dynamics],
    dyn_kws: dict[str, object],
    run_kws: dict[str, object],
    uq_hook: Callable[[Atoms], tuple[dict, dict]] | None = None,
    uq_kws: dict[str, object] | None = None,
    gpu_flush_interval: int = 10,
    early_stop_threshold: float | None = None,
    uq_field: str = 'uq_force_std_max',
    catch_crashes: bool = True,
) -> tuple[list[Atoms], dict | None]:
    """Advance dynamics of a chunk of a trajectory

    Arguments:
        spec: contains atoms and metadata about trajectory
        learner: used to make the calculator
        weights: weights for the calculator, one entry per ensemble member. A
            single-element list uses a plain calculator; more than one builds an
            ensemble calculator (see BaseLearnableForcefield.make_ensemble_calculator).
        db_url: url to write frames to
        device: for torch
        dyn_cls: ASE dynamics class
        dyn_kws: kws to the dynamics constructor
        run_kws: kws to the dynamics run method
        uq_hook: optional callable invoked on each frame's Atoms after force evaluation.
            Returns (per_atom, per_frame) dicts of named UQ quantities, stored into
            atoms.arrays / atoms.info respectively. Expects an ensemble-producing
            calculator (e.g. one populating atoms.calc.results['forces_ens']).
        uq_kws: keyword arguments passed to uq_hook
        gpu_flush_interval: how often to release PyTorch's CUDA caching
            allocator back to the driver. Without this NPT dynamics will cause
            memory leaks through neighbor list size changes
        early_stop_threshold: if set, stop dynamics the first step whose
            atoms.info[uq_field] >= this value, returning the frames captured so far
            instead of running the full spec.steps. None disables early stopping.
        uq_field: atoms.info key checked against early_stop_threshold each step
        catch_crashes: if True (default), a hard crash (e.g. LinAlgError from the NPT
            barostat) is also caught and treated like a controlled early stop. If
            False, only a TrajectoryDiverged (UQ-threshold) stop is caught; a hard
            crash propagates and fails the trajectory, same as before this feature
            existed. Set False to test whether early_stop_threshold alone is
            sufficient to prevent a crash, without the safety net masking it.

        Returns:
            (traj, integrator_state)
    """

    uq_kws = uq_kws or {}

    # todo: stop this from writing to the screen
    logfile = str(Path(run_dir) / f'traj-{spec.traj_id}_chunk-{spec.chunk_id}_att-{spec.attempt_index}_md.log')
    logger = logging.getLogger(logfile)
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    is_cuda = torch.cuda.is_available() and 'cuda' in device

    atoms = spec.atoms
    logger.info('Creating calculator')
    if len(weights) == 1:
        calc = learner.make_calculator(weights[0], device=device)
    else:
        calc = learner.make_ensemble_calculator(weights, device=device)
    atoms.calc = calc

    logger.info('Creating dynamics class')
    dyn = dyn_cls(atoms, **dyn_kws)
    restore_dyn_state(dyn, spec.dyn_state)

    frames = []

    def flush_gpu_memory():
        if is_cuda:
            torch.cuda.empty_cache()

    def write_frame():
        logger.info('getting results from calc')
        f = atoms.calc.results['forces']
        atoms.calc.results['forces'] = f.astype(np.float64)

        if uq_hook is not None:
            logger.info('computing UQ')
            per_atom, per_frame = uq_hook(atoms, **uq_kws)
            for name, values in per_atom.items():
                atoms.set_array(name, np.asarray(values))
            for name, value in per_frame.items():
                atoms.info[name] = value

        canonical_atoms = canonicalize(atoms)

        logger.info('writing frame to db')
        frames.append(canonical_atoms)

        if early_stop_threshold is not None and uq_field in atoms.info:
            uq_value = atoms.info[uq_field]
            if uq_value >= early_stop_threshold:
                logger.warning(
                    f'{uq_field}={uq_value:.4g} >= early-stop threshold '
                    f'{early_stop_threshold:.4g} at frame {len(frames) - 1}; stopping chunk early'
                )
                raise TrajectoryDiverged(f'{uq_field}={uq_value:.4g} >= threshold {early_stop_threshold:.4g}')

    dyn.attach(write_frame)
    dyn.attach(flush_gpu_memory, interval=gpu_flush_interval)

    logger.info('Starting dynamics')
    try:
        dyn.run(spec.steps, **run_kws)
    except TrajectoryDiverged as exc:
        # A controlled early stop: always caught. Returning the partial frames lets
        # DynamicsRunner treat this as a short chunk through the normal pipeline
        # instead of failing the whole trajectory.
        msg = f'traj {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}: dynamics stopped early (TrajectoryDiverged): {exc}'
        logger.warning(msg)
        _module_logger.warning(msg)  # per-attempt logfile below is deleted; this one isn't
    except Exception as exc:
        if not catch_crashes:
            raise
        # Hard crash (e.g. LinAlgError from the NPT barostat), caught only because
        # catch_crashes=True -- treated the same as a controlled early stop.
        msg = f'traj {spec.traj_id} chunk {spec.chunk_id} attempt {spec.attempt_index}: dynamics stopped early ({type(exc).__name__}): {exc}'
        logger.warning(msg)
        _module_logger.warning(msg)
    flush_gpu_memory()
    os.remove(logfile)

    new_dyn_state = extract_dyn_state(dyn)
    return frames, new_dyn_state


def ensemble_force_deviation_uq(atoms: Atoms) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """UQ hook for use with an ensemble calculator (see EnsembleCalculator)

    Reduces the ensemble-member axis of the per-atom force disagreement, leaving
    per-atom and per-frame quantities for write_frame to attach to the Atoms.

    Requires atoms.calc.results to contain 'forces_ens', i.e. an ensemble calculator
    must have been used to produce this frame.
    """
    import numpy as np

    ens = atoms.calc.results['forces_ens']       # (n_models, n_atoms, 3)
    f = atoms.calc.results['forces']              # (n_atoms, 3)
    dev = np.linalg.norm(ens - f[None], axis=-1).mean(axis=0)  # (n_atoms,)

    per_atom = {'uq_force_std': dev}
    per_frame = {'uq_force_std_max': float(dev.max())}
    return per_atom, per_frame


def uq_threshold_audit(
    chunk: Chunk,
    field: str = 'uq_force_std_max',
    threshold: float = 0.1,
    burn_in_model_versions: int = 0,
) -> AuditResult:
    """Audit a chunk by thresholding a per-frame UQ scalar stored in atoms.info

    Requires the chunk's frames to have been produced with a uq_hook (e.g.
    ensemble_force_deviation_uq) that populates `field`.
    """
    from cascade.model import AuditResult, AuditStatus
    import numpy as np

    if chunk.model_version < burn_in_model_versions:
        return AuditResult(status=AuditStatus.FAILED, score=float('inf'), reason='burn_in')

    values = np.array([a.info[field] for a in chunk.atoms])
    score = float(values.max())
    status = AuditStatus.PASSED if score < threshold else AuditStatus.FAILED
    return AuditResult(status=status, score=score, reason='threshold')


def audit_with_random_failure(
    chunk: Chunk,
    audit_task: Callable[..., AuditResult],
    fail_rate: float = 0.0,
    **audit_kws,
) -> AuditResult:
    """Wraps another audit_task with random failures. Will only trip on a successful audit.
    """
    from cascade.model import AuditResult, AuditStatus
    import numpy as np

    result = audit_task(chunk, **audit_kws)
    if result.status == AuditStatus.PASSED and fail_rate > 0:
        rng = np.random.default_rng(seed=None)
        if rng.random() < fail_rate:
            return AuditResult(status=AuditStatus.FAILED, score=result.score, reason='random_fail')
    return result

def max_force_error(atoms_predicted: Atoms, atoms_labeled: Atoms) -> float:
    """Get the maximum error in the forces between the predicted and labeled forces on the atoms
    """
    f_pred = atoms_predicted.calc.results['forces']
    f_true = atoms_labeled.calc.results['forces']
    return float(np.linalg.norm(f_pred - f_true, axis=-1).max())


def label_noop(spec: TrainingFrame, calc_factory: Callable[..., Calculator]) -> TrainingFrame:
    """Returns forces from the training frame spec unmodified"""
    return spec

def label_frame(frame: TrainingFrame, calc_factory: Callable[..., Calculator]) -> TrainingFrame:
    """runs the specified calculator on the atoms"""
    from ase.calculators.calculator import all_changes # required by some calcs (e.g. FairChemCalculator)
    from cascade.utils import canonicalize
    calc = calc_factory()
    atoms_labeled = frame.atoms.copy()
    atoms_labeled.calc = calc
    calc.calculate(atoms_labeled, properties=calc.implemented_properties, system_changes=all_changes)
    frame.atoms_labeled = canonicalize(atoms_labeled)
    return frame

# todo: this should be configurable, or at least not hard code magic knowledge
def training_noop(learner: BaseLearnableForcefield, device) -> bytes:
    """just return a model"""
    calc = mace_mp('small', device=device, default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)
    return model_msg

def train(learner: BaseLearnableForcefield,
          weights: bytes,
          train_data: list[Atoms],
          valid_data: list[Atoms],
          train_kws: dict[str, object],
          replay: MultiHeadConfig | None = None,
          ) -> tuple[bytes, pd.DataFrame]:
    weights, results = learner.train(weights, train_data, valid_data, replay=replay, **train_kws)
    return weights, results
