from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ase import Atoms
from ase.io import write


class AuditStatus(Enum):
    PENDING = auto()
    FAILED = auto()
    PASSED = auto()


@dataclass
class AdvanceSpec:

    atoms: Atoms
    traj_id: int
    chunk_id: int
    steps: int


@dataclass
class TrajectoryChunk:
    """
    Contains the list of atoms generated in a trajectory chunk, along with 
    relevant metadata
    """

    atoms: list[Atoms]
    """List of atoms"""

    model_version: int
    """Model version that was used to generate this chunk"""

    traj_id: int
    """Which trajectory this is a part of"""

    chunk_id: int
    """Which chunk this is"""

    audit_status: AuditStatus = AuditStatus.PENDING
    """Has this been audited, and if so, has it passed"""

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return f'Chunk(traj_id={self.traj_id}, chunk_id={self.chunk_id})'


@dataclass
class Trajectory:

    chunks: list[TrajectoryChunk]

    target_length: int
    """how long the trajectory should be"""

    id: int
    """Trajectory ID"""

    init_atoms: Atoms
    """Initial frame"""

    def add_chunk(self, chunk: list[Atoms]):
        self.chunks.append(chunk)

    def __len__(self):
        return sum(len(chunk) for chunk in self.chunks)

    @property
    def audit_status(self):
        return self.chunks[-1].audit_status

    @property
    def done(self):
        return (
            len(self) >= self.target_length
            and
            self.audit_status == AuditStatus.PASSED
        )

    def write(self):
        for a in self.atoms:
            write(self.path, a)
