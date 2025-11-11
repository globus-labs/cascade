from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cascade.model import ChunkSpec, AuditResult

import numpy as np
from ase import Atoms

# can make this a classmethod on some audittask class
# to get some shared informaiton and inheritance
def random_audit(
    chunk_atoms: list[Atoms],
    rng: np.random.RandomState = np.random.default_rng(),
    accept_prob: float = 0.5,
) -> AuditResult:
    """Random audit of a chunk of a trajectory
    
    Intended to be used as a stub for a real audit function.
    """
    from cascade.model import AuditResult
    
    passed = rng.random() < accept_prob
    score = rng.random() if passed else 0.0
    return AuditResult(passed=passed, score=score)