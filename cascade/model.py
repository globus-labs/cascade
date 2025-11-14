from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from collections import namedtuple

from ase import Atoms

@dataclass
class ChunkSpec:
    traj_id: int
    chunk_id: int

@dataclass
class TrainingFrame:
    atoms: Atoms
    model_version: int

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
