from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cascade.model import ChunkSpec, AuditResult
    from ase.db import connect
    from cascade.model import AdvanceSpec
    from cascade.learning.base import BaseLearnableForcefield
    from ase import Atoms
    from ase.optimize.optimize import Dynamics
    import numpy as np

# can make this a classmethod on some audittask class
# to get some shared informaiton and inheritance
def random_audit(
    chunk_atoms: list[Atoms],
    chunk_spec: ChunkSpec,
    attempt_index: int,
    rng: np.random.RandomState = np.random.default_rng(),
    accept_prob: float = 0.5,
    sleep_time: float = 0.,
) -> AuditResult:
    """Random audit of a chunk of a trajectory
    
    Intended to be used as a stub for a real audit function.
    """
    from cascade.model import AuditResult
    import time

    time.sleep(sleep_time)
    passed = rng.random() < accept_prob
    score = rng.random() if passed else 0.0
    return AuditResult(passed=passed, score=score, traj_id=chunk_spec.traj_id, chunk_id=chunk_spec.chunk_id, attempt_index=attempt_index)

def advance_dynamics(
    spec: AdvanceSpec,
    learner: BaseLearnableForcefield,
    weights: bytes,
    db: connect,
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
    
    atoms = spec.atoms
    calc = learner.make_calculator(weights, device=device)
    atoms.calc = calc

    dyn = dyn_cls(atoms, **dyn_kws)

    def write_to_db():
        # needs to be 64 bit for db read
        f = atoms.calc.results['forces']
        atoms.calc.results['forces'] = f.astype(np.float64)
        canonical_atoms = canonicalize(atoms)
        db.write(
            canonical_atoms, 
            chunk_id=spec.chunk_id,
            traj_id=spec.traj_id,
            run_id=spec.run_id,
            attempt_index=spec.attempt_index) # todo: get attempt index from db
    dyn.attach(write_to_db)

    dyn.run(spec.steps, **run_kws)
    return spec

