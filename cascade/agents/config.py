from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.optimize.optimize import Dynamics
    from cascade.model import Trajectory
    from cascade.learning.base import BaseLearnableForcefield
    from concurrent.futures import Executor
    from typing import Callable
    from cascade.model import (
        AdvanceSpec,
        AuditResult,
        TrainingFrame,
        Chunk
    )

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
    """Initial configuration for dynamics"""
    traj_id: int
    """Trajectory ID"""
    chunk_size: int
    """how many steps to run at a time before audit"""
    n_steps: int
    """Total number of steps to run"""
    run_dir: str
    """For MD logging"""
    executor: Executor
    """Where tasks get run"""
    advance_dynamics_task: Callable[[AdvanceSpec], None]
    """Task to run dynamics"""
    learner: BaseLearnableForcefield
    """Learner to be used for dynamics"""
    weights: bytes
    """Initial weights for dynamics"""
    dyn_cls: type[Dynamics]
    """ASE dynamics integrator"""
    dyn_kws: dict[str, object] | None
    """Passed to dynamics constructor"""
    run_kws: dict[str, object] | None
    """Passed to dynamics.run"""
    device: str = 'cpu'
    """Device to run learner for dynamics"""
    model_version: int = 0  # todo: I am not so sure this belongs here


@dataclass
class AuditorConfig(CascadeAgentConfig):
    """Configuration for DummyAuditor agent"""
    audit_task: Callable[[Chunk], AuditResult]
    """Function to audit a chunk"""
    audit_kws: dict
    """Keyword arguments to audit_task"""
    executor: Executor
    """Where to run audit task"""


@dataclass
class SamplerConfig(CascadeAgentConfig):
    """Configuration for Sampler agent"""
    n_frames: int
    """How many frames to sample given a trajectory chunk"""
    executor: Executor
    """Where to run sample_task"""
    sample_task: Callable[..., list[TrainingFrame]]
    """Method that returns unlabled training frames given a trajectory chunk"""


@dataclass
class LabelerConfig(CascadeAgentConfig):
    """Configuration for Labeler agent"""
    executor: Executor
    """Where to run label_task"""
    label_task: Callable[[TrainingFrame], TrainingFrame]
    """Adds labels to training frames"""

@dataclass
class TrainerConfig(CascadeAgentConfig):
    """Configuration for Trainer agent"""
    training_task: Callable[..., bytes]
    """Returns trained model weights"""
    training_args: list | tuple
    """passed to training_task"""
    training_kws: dict
    """Passed to training_task"""
    learner: BaseLearnableForcefield
    executor: Executor
    """Where to run training_task"""


@dataclass
class DatabaseMonitorConfig(CascadeAgentConfig):
    """Configuration for DatabaseMonitor agent"""
    retrain_len: int
    """How many labeled frames to trigger retraining"""
    retrain_fraction: float = 0.5
    """What fraction of trajectories with labeled frames to retrain"""