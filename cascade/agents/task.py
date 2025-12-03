from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cascade.model import ChunkSpec, AuditResult
    from cascade.model import AdvanceSpec
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
    
    # Create TrajectoryDB instance in the worker process with NullPool (no connection pooling)
    traj_db = TrajectoryDB(db_url, use_null_pool=True)
    
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

