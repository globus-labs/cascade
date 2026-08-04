from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.calculators.calculator import Calculator

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
    import pandas as pd
    from cascade.learning.finetuning import MultiHeadConfig

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
    weights: list[bytes]
    """Initial weights for dynamics, one entry per ensemble member"""
    dyn_cls: type[Dynamics]
    """ASE dynamics integrator"""
    dyn_kws: dict[str, object] | None
    """Passed to dynamics constructor"""
    run_kws: dict[str, object] | None
    """Passed to dynamics.run"""
    device: str = 'cpu'
    """Device to run learner for dynamics"""
    model_version: int = 0  # todo: I am not so sure this belongs here
    uq_hook: Callable[[Atoms], tuple[dict, dict]] | None = None
    """Optional hook called on each frame to compute UQ from an ensemble calculator's results"""
    uq_kws: dict[str, object] = field(default_factory=dict)
    """Keyword arguments passed to uq_hook"""
    gpu_flush_interval: int = 10
    """How often advance_dynamics releases PyTorch's CUDA caching allocator"""


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
    burn_in_model_versions: int = 0
    """Chunks with model_version below this count use burn_in_n_frames instead of n_frames"""
    burn_in_n_frames: int | None = None
    """Frames to sample per chunk while model_version < burn_in_model_versions (falls back to n_frames if unset)"""


@dataclass
class LabelerConfig(CascadeAgentConfig):
    """Configuration for Labeler agent"""
    executor: Executor
    """Where to run label_task"""
    calc_factory: Callable[..., Calculator]
    """Create the calculator to use for labeling"""
    label_task: Callable[[TrainingFrame, Callable[..., Calculator]], TrainingFrame]
    """Adds labels to training frames"""
    uq_field: str = 'uq_force_std_max'
    """atoms.info key holding the UQ scalar recorded at sample time"""
    error_fn: Callable[[Atoms, Atoms], float] | None = None
    """(predicted_atoms, labeled_atoms) -> observed error, recorded for Controller calibration.
    Defaults to cascade.agents.task.max_force_error (resolved lazily by Labeler to avoid
    importing task.py's heavy dependencies at config-module load time)."""


@dataclass
class ControllerConfig(CascadeAgentConfig):
    """Configuration for Controller agent"""
    target_ferr: float
    """Target observed error (Eq. 1/3 of the proxima paper)"""
    history_length: int = 8
    """Max number of observations pulled per calibration window"""
    recalibrate_every: int = 5
    """New labeled frames required between recalibrations"""
    burn_in_model_versions: int = 0
    """Ignore calibration observations sampled below this model version"""

@dataclass
class TrainerConfig(CascadeAgentConfig):
    """Configuration for Trainer agent"""
    weights: list[bytes]
    """Current weights for each ensemble member"""
    training_task: Callable[..., tuple[bytes, pd.DataFrame]]
    """Returns trained model weights"""
    training_args: list | tuple
    """passed to training_task"""
    training_kws: dict
    """Passed to training_task"""
    learner: BaseLearnableForcefield
    executor: Executor
    """Where to run training_task"""
    bootstrap_fraction: float = 1.0
    """Fraction of available training frames to resample (with replacement) per ensemble member"""
    replay: MultiHeadConfig | None = None
    """Multi-head replay config (see cascade.learning.finetuning.MultiHeadConfig), passed through
    to learner.train to prevent catastrophic forgetting. Only meaningful for learners whose train()
    accepts a `replay` kwarg (currently MACEInterface)."""


@dataclass
class DatabaseMonitorConfig(CascadeAgentConfig):
    """Configuration for DatabaseMonitor agent"""
    retrain_len: int
    """How many labeled frames to trigger retraining"""
    retrain_fraction: float = 0.5
    """What fraction of trajectories with labeled frames to retrain"""