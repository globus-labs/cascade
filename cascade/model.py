from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ase import Atoms
from ase.io import write


class AuditStatus(Enum):
    """Whether a trajectory chunk is awaiting or has passed/failed an audit"""
    PENDING = auto()
    FAILED = auto()
    PASSED = auto()

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
    create a trajectory chunk
    """
    atoms: Atoms
    """Initial atoms for the trajectory chunk"""
    traj_id: int
    """Which trajectory"""
    chunk_id: int
    """Which chunk"""
    steps: int
    """How many steps to run dynamics for"""
