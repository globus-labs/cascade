"""Classes mostly used to pass state about trajectories between agents over exchange or through database"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ase import Atoms

@dataclass
class Chunk:
    """A time chunk fo a trajectory"""
    atoms: list[Atoms]
    """The atoms in the chunk"""
    frame_ids: list[int]
    """List of database keys for every frame of """
    traj_id: int
    chunk_id: int
    attempt_ix: int
    model_version: int
    """model version that generated this chunk"""


@dataclass
class AuditResult:
    """The result of an audit"""
    status: AuditStatus
    """Whether the chunk passed audit"""
    score: float
    """How good or bad the chunk was in terms of uncertainty"""

@dataclass
class TrainingFrame:
    atoms: Atoms
    model_version: int
    traj_id: int
    chunk_id: int
    attempt_index: int
    frame_id: int
    n_sampled_frames: int # todo: is this really the way to pass this around
    labeled: bool = False


@dataclass
class AdvanceSpec:
    """Trajectory advancement specification.

    This is bare minimum information to pass for the dynamics engine
    to create a trajectory chunk.
    """
    atoms: Atoms
    """Initial atoms for the trajectory chunk"""
    run_id: str
    """Run identifier"""
    traj_id: int
    """Which trajectory"""
    chunk_id: int
    """Which chunk"""
    attempt_index: int
    """Attempt index for this chunk"""
    steps: int
    """How many steps to run dynamics for"""


class AuditStatus(Enum):
    """Whether a trajectory chunk is awaiting or has passed/failed an audit"""
    PENDING = auto()
    FAILED = auto()
    PASSED = auto()


class TrajectoryStatus(Enum):
    """Lifecycle state for a trajectory."""
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class ChunkEventType(Enum):
    """Event types tracked for trajectory chunks"""
    STARTED_DYNAMICS = auto()
    FINISHED_DYNAMICS = auto()
    STARTED_AUDIT = auto()
    AUDIT_PASSED = auto()
    AUDIT_FAILED = auto()
    STARTED_SAMPLING = auto()
    FINISHED_SAMPLING = auto()
    STARTED_LABELING = auto()
    STARTED_LABELING_FRAME = auto()
    FINISHED_LABELING_FRAME = auto()
    FINISHED_LABELING = auto()
    TRAJECTORY_COMPLETED = auto()
    STARTED_TRAINING = auto()
    FINISHED_TRAINING = auto()

