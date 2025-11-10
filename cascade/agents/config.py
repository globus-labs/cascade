from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.optimize.optimize import Dynamics
    from cascade.model import Trajectory
    from cascade.learning.base import BaseLearnableForcefield
    import numpy as np


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
class DynamicsEngineConfig(CascadeAgentConfig):
    """Configuration for DynamicsEngine agent"""
    init_specs: list  # list[AdvanceSpec], but avoid circular import
    learner: BaseLearnableForcefield
    weights: bytes
    dyn_cls: type[Dynamics]
    dyn_kws: dict[str, object] | None
    run_kws: dict[str, object] | None
    device: str = 'cpu'


@dataclass
class AuditorConfig(CascadeAgentConfig):
    """Configuration for DummyAuditor agent"""
    accept_rate: float
    chunk_size: int


@dataclass
class SamplerConfig(CascadeAgentConfig):
    """Configuration for DummySampler agent"""
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
    retrain_len: int
    target_length: int
    chunk_size: int
    retrain_fraction: float = 0.5
    retrain_min_frames: int = 10

