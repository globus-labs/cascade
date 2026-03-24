from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.optimize.optimize import Dynamics
    from cascade.model import Trajectory
    from cascade.learning.base import BaseLearnableForcefield
    import numpy as np
    from concurrent.futures import Executor
    from typing import Callable
    from cascade.model import AdvanceSpec, AuditResult, ChunkSpec


@dataclass
class CascadeAgentConfig:
    """Base configuration for all cascade agents"""
    run_id: str
    """Run ID"""
    db_url: str
    """Database URL"""


@dataclass
class DatabaseConfig(CascadeAgentConfig):
    """Configuration for DummyDatabase agent"""
    trajectories: list[Trajectory]
    chunk_size: int
    retrain_len: int


@dataclass
class DynamicsRunnerConfig(CascadeAgentConfig):
    """Configuration for DynamicsEngine agent"""
    atoms: Atoms
    run_id: str
    db_url: str
    traj_id: int
    chunk_size: int
    n_steps: int
    run_dir: str
    executor: Executor
    advance_dynamics_task: Callable[[AdvanceSpec], None]
    learner: BaseLearnableForcefield
    weights: bytes
    dyn_cls: type[Dynamics]
    dyn_kws: dict[str, object] | None
    run_kws: dict[str, object] | None
    device: str = 'cpu'
    model_version: int = 0


@dataclass
class AuditorConfig(CascadeAgentConfig):
    """Configuration for DummyAuditor agent"""
    run_id: int
    db_url: str
    audit_task: Callable[[ChunkSpec], AuditResult]
    audit_kwargs: dict
    executor: Executor
    chunk_size: int


@dataclass
class SamplerConfig(CascadeAgentConfig):
    """Configuration for Sampler agent"""
    n_frames: int
    rng: np.random.Generator | None = None


@dataclass
class LabelerConfig(CascadeAgentConfig):
    """Configuration for DummyLabeler agent"""


@dataclass
class TrainerConfig(CascadeAgentConfig):
    """Configuration for DummyTrainer agent"""
    learner: BaseLearnableForcefield


@dataclass
class DatabaseMonitorConfig(CascadeAgentConfig):
    """Configuration for DatabaseMonitor agent"""
    run_id: int
    db_url: str
    retrain_len: int
    chunk_size: int
    retrain_fraction: float = 0.5
    retrain_min_frames: int = 10
