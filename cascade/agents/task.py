from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cascade.model import ChunkSpec, AuditResult
    from cascade.model import AdvanceSpec, TrainingFrameSpec
    from cascade.learning.base import BaseLearnableForcefield
    from ase import Atoms
from ase.optimize.optimize import Dynamics

# can make this a classmethod on some audittask class
# to get some shared informaiton and inheritance
def random_audit(
    chunk_atoms: list[Atoms],
    chunk_spec: ChunkSpec,
    attempt_index: int,
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
    return AuditResult(status=status, score=score, traj_id=chunk_spec.traj_id, chunk_id=chunk_spec.chunk_id, attempt_index=attempt_index)


def random_sample(
    atoms_list: list[Atoms],
    frame_ids: list[int],
    chunk_spec: ChunkSpec,
    model_version: int,
    n_frames: int,
    sleep_time: float = 0.,
) -> list:
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
    n_sample = min(n_frames, len(atoms_list))
    indices = rng.choice(len(atoms_list), size=n_sample, replace=False)
    sampled_frames = [atoms_list[i] for i in indices]
    sampled_frame_ids = [frame_ids[i] for i in indices]

    result = []
    for frame, trajectory_frame_id in zip(sampled_frames, sampled_frame_ids):
        training_frame = TrainingFrame(atoms=frame, model_version=model_version)
        spec = TrainingFrameSpec(
            training_frame=training_frame,
            trajectory_frame_id=trajectory_frame_id,
            traj_id=chunk_spec.traj_id,
            chunk_id=chunk_spec.chunk_id,
            attempt_index=chunk_spec.attempt_index,
            total_frames_in_chunk=n_sample,
        )
        result.append(spec)
    return result


def advance_dynamics(
    spec: AdvanceSpec,
    learner: BaseLearnableForcefield,
    weights: bytes,
    db_url: str,
    device: str = 'cpu',
    dyn_cls: type[Dynamics] = Dynamics,
    dyn_kws: dict[str, object] = {},
    run_kws: dict[str, object] = {},
) -> None:
    """Advance dynamics of a chunk of a trajectory
    
    Intended to be used as a stub for a real advance dynamics function.
    """
    import numpy as np
    from cascade.utils import canonicalize
    from cascade.agents.db_orm import TrajectoryDB
    
    # Create TrajectoryDB instance in the worker process
    traj_db = TrajectoryDB(db_url)
    
    atoms = spec.atoms
    calc = learner.make_calculator(weights, device=device)
    atoms.calc = calc

    dyn = dyn_cls(atoms, **dyn_kws)
    
    frame_index = 0  # Track frame index within this chunk

    def write_to_db():
        nonlocal frame_index
        f = atoms.calc.results['forces']
        atoms.calc.results['forces'] = f.astype(np.float64)
        canonical_atoms = canonicalize(atoms)
        
        # Write frame to database
        traj_db.write_frame(
            run_id=spec.run_id,
            traj_id=spec.traj_id,
            chunk_id=spec.chunk_id,
            attempt_index=spec.attempt_index,
            frame_index=frame_index,
            atoms=canonical_atoms
        )
        frame_index += 1
    
    dyn.attach(write_to_db)

    dyn.run(spec.steps, **run_kws)
    
    return spec

