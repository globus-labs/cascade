from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cascade.model import ChunkSpec, AuditResult, Chunk
    from cascade.model import AdvanceSpec, TrainingFrameSpec
    from cascade.learning.base import BaseLearnableForcefield
    from ase import Atoms
    from pathlib import Path
    import numpy as np
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
    from cascade.model import TrainingFrame, TrainingFrameSpec
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
                model_version=chunk.model_version,
            )
        )
    return result


def advance_dynamics(
    spec: AdvanceSpec,
    learner: BaseLearnableForcefield,
    weights: bytes,
    db_url: str,
    device: str,
    run_dir: str,
    dyn_cls: type[Dynamics],
    dyn_kws: dict[str, object],
    run_kws: dict[str, object],
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
    """
    import numpy as np
    from cascade.utils import canonicalize
    from pathlib import Path

    import logging
    import os

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
        canonical_atoms = canonicalize(atoms)

        logger.info('writing frame to db')
        frames.append(canonical_atoms)

    dyn.attach(write_frame)

    logger.info('Starting dynamics')
    dyn.run(spec.steps, **run_kws)
    os.remove(logfile)

    return frames


def label_noop(spec: TrainingFrameSpec) -> TrainingFrameSpec:
    """Returns forces from the training frame spec unmodified"""
    return spec


# todo: this should be configurable, or at least not hard code magic knowledge
def training_noop(learner: BaseLearnableForcefield) -> bytes:
    """just return a model"""
    from mace.calculators import mace_mp

    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)
    return model_msg