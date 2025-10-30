from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.optimize.optimize import Dynamics
    from cascade.model import Trajectory
    from cascade.learning.base import BaseLearnableForcefield
    import numpy as np


@dataclass
class DatabaseConfig:
    """Configuration for DummyDatabase agent"""
    trajectories: list[Trajectory]
    chunk_size: int
    retrain_len: int


@dataclass
class DynamicsEngineConfig:
    """Configuration for DynamicsEngine agent"""
    init_specs: list  # list[AdvanceSpec], but avoid circular import
    learner: BaseLearnableForcefield
    weights: bytes
    dyn_cls: type[Dynamics]
    dyn_kws: dict[str, object] | None
    run_kws: dict[str, object] | None
    device: str = 'cpu'


@dataclass
class AuditorConfig:
    """Configuration for DummyAuditor agent"""
    accept_rate: float


@dataclass
class SamplerConfig:
    """Configuration for DummySampler agent"""
    n_frames: int
    rng: np.random.Generator | None = None


@dataclass
class LabelerConfig:
    """Configuration for DummyLabeler agent"""
    pass  # No configuration parameters


@dataclass
class TrainerConfig:
    """Configuration for DummyTrainer agent"""
    pass  # No configuration parameters

