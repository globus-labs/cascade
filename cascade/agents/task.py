from __future__ import annotations

from typing import TYPE_CHECKING

from matscipy.calculators.polydisperse import calculator

if TYPE_CHECKING:
    from typing import Callable
    from cascade.model import AuditResult, Chunk
    from cascade.model import AdvanceSpec, TrainingFrame
    from cascade.learning.base import BaseLearnableForcefield
    from cascade.calculator import Calculator
    from ase import Atoms
    from pathlib import Path
    import numpy as np
    import pandas as pd
from ase.optimize.optimize import Dynamics


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
    from cascade.model import AuditResult, AuditStatus
    import time
    import numpy as np

    time.sleep(sleep_time)
    # Create a new random generator seeded with OS entropy to ensure
    # each worker process gets a unique random state
    rng = np.random.default_rng(seed=None)
    passed = rng.random() < accept_prob
    score = rng.random() if passed else 0.0
    status = AuditStatus.PASSED if passed else AuditStatus.FAILED
    return AuditResult(status=status, score=score)


def random_sample(
    chunk: Chunk,
    n_frames: int,
    sleep_time: float = 0.,
) -> list[TrainingFrame]:
    """Random sample of frames from a chunk.

    Intended to be used as a stub for a real sampling function.
    """
    from cascade.model import TrainingFrame
    import time
    import numpy as np

    time.sleep(sleep_time)
    # Create a new random generator seeded with OS entropy to ensure
    # each worker process gets a unique random state
    rng = np.random.default_rng(seed=None)
    n_sample = min(n_frames, len(chunk.atoms))
    indices = rng.choice(len(chunk.atoms), size=n_sample, replace=False)
    result = []
    for i in indices:
        result.append(
            TrainingFrame(
                atoms=chunk.atoms[i],
                frame_id=chunk.frame_ids[i],
                model_version=chunk.model_version,
                traj_id=chunk.traj_id,
                chunk_id=chunk.chunk_id,
                attempt_index=chunk.attempt_ix,
                n_sampled_frames=n_sample
            )
        )
    return result


def advance_dynamics(
    spec: AdvanceSpec,
    learner: BaseLearnableForcefield,
    weights: bytes,
    device: str,
    run_dir: str,
    dyn_cls: type[Dynamics],
    dyn_kws: dict[str, object],
    run_kws: dict[str, object],
    uq_hook: Callable[[Atoms], tuple[dict, dict]] | None = None,
    uq_kws: dict[str, object] | None = None,
) -> list[Atoms]:
    """Advance dynamics of a chunk of a trajectory

    Arguments:
        spec: contains atoms and metadata about trajectory
        learner: used to make the calculator
        weights: weights to add to the calculator
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
    """
    import numpy as np
    from cascade.utils import canonicalize
    from pathlib import Path

    import logging
    import os

    uq_kws = uq_kws or {}

    # todo: stop this from writing to the screen
    logfile = str(Path(run_dir) / f'traj-{spec.traj_id}_chunk-{spec.chunk_id}_att-{spec.attempt_index}_md.log')
    logger = logging.getLogger(logfile)
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    atoms = spec.atoms
    logger.info('Creating calculator')
    calc = learner.make_calculator(weights, device=device)
    atoms.calc = calc

    logger.info('Creating dynamics class')
    dyn = dyn_cls(atoms, **dyn_kws)

    frames = []

    def write_frame():
        logger.info('getting results from calc')
        f = atoms.calc.results['forces']
        atoms.calc.results['forces'] = f.astype(np.float64)

        if uq_hook is not None:
            logger.info('computing UQ')
            per_atom, per_frame = uq_hook(atoms, **uq_kws)
            for name, values in per_atom.items():
                atoms.new_array(name, np.asarray(values))
            for name, value in per_frame.items():
                atoms.info[name] = value

        canonical_atoms = canonicalize(atoms)

        logger.info('writing frame to db')
        frames.append(canonical_atoms)

    dyn.attach(write_frame)

    logger.info('Starting dynamics')
    dyn.run(spec.steps, **run_kws)
    os.remove(logfile)

    return frames


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
) -> AuditResult:
    """Audit a chunk by thresholding a per-frame UQ scalar stored in atoms.info

    Requires the chunk's frames to have been produced with a uq_hook (e.g.
    ensemble_force_deviation_uq) that populates `field`.
    """
    from cascade.model import AuditResult, AuditStatus
    import numpy as np

    values = np.array([a.info[field] for a in chunk.atoms])
    score = float(values.max())
    status = AuditStatus.PASSED if score < threshold else AuditStatus.FAILED
    return AuditResult(status=status, score=score)

def label_noop(spec: TrainingFrame, calc_factory: Callable[..., Calculator]) -> TrainingFrame:
    """Returns forces from the training frame spec unmodified"""
    return spec

def label_frame(frame: TrainingFrame, calc_factory: Callable[..., Calculator]) -> TrainingFrame:
    """runs the specified calculator on the atoms"""
    from cascade.utils import canonicalize
    calc = calc_factory()
    atoms_labeled = frame.atoms.copy()
    atoms_labeled.calc = calc
    calc.calculate(atoms_labeled)
    frame.atoms_labeled = canonicalize(atoms_labeled)
    return frame

# todo: this should be configurable, or at least not hard code magic knowledge
def training_noop(learner: BaseLearnableForcefield) -> bytes:
    """just return a model"""
    from mace.calculators import mace_mp

    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)
    return model_msg

def train(learner: BaseLearnableForcefield,
          weights: bytes,
          train_data: list[Atoms],
          valid_data: list[Atoms],
          train_kws: dict[str, object],
          ) -> tuple[bytes, pd.DataFrame]:
    weights, results = learner.train(weights, train_data, valid_data, **train_kws)
    return weights, results
