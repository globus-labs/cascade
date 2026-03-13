from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from collections import namedtuple

from ase import Atoms

@dataclass
class ChunkSpec:
    traj_id: int
    chunk_id: int
    attempt_index: int | None = None
    model_version: int | None = None

@dataclass
class TrainingFrame:
    atoms: Atoms
    model_version: int

@dataclass
class TrajectoryState:
    atoms: Atoms
    timestep: int
    chunk: int
    attempt: int

@dataclass
class TrainingFrameSpec:
    """Training frame specification with all metadata needed for processing.
    
    This encapsulates both the training frame content and its trajectory metadata
    to avoid database lookups when passing frames between agents.
    """
    training_frame: TrainingFrame
    """The training frame with atoms and model version"""
    trajectory_frame_id: int
    """ID of the frame in the trajectory_frames table"""
    traj_id: int
    """Trajectory identifier"""
    chunk_id: int
    """Chunk identifier"""
    attempt_index: int
    """Attempt index for this chunk"""
    total_frames_in_chunk: int
    """Total number of frames that will be labeled for this chunk"""

@dataclass
class AuditResult:
    """The result of an audit"""
    status: AuditStatus
    """Whether the chunk passed audit"""
    score: float
    """The score assigned by the auditor"""
    traj_id: int
    """The trajectory ID"""
    chunk_id: int
    """The chunk ID""" 
    attempt_index: int
    """The attempt index"""

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

@dataclass
class TrajectorySpec:
    """Enough information to initialize a trajectory"""
    run_id: hash
    """The run ID"""
    traj_id: int
    """The trajectory ID"""
    target_length: int
    """The target length of the trajectory"""
    init_atoms: Atoms
    """The initial atoms for the trajectory"""

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
