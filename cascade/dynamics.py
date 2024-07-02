"""Interface to run a dynamics protocol using a learned forcefield"""
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from uuid import uuid4

from ase import Atoms
from ase.io import Trajectory, read
from ase.optimize.optimize import Dynamics, Optimizer

from cascade.learning.base import BaseLearnableForcefield, State


# TODO (wardlt): Consider having the state of the `Dynamics` class stored here. Some dynamics classes (e.g., NPT) have state
@dataclass
class Progress:
    """The progress of an atomic state through a dynamics protocol"""

    atoms: Atoms
    """Current atomic structure"""
    name: str = field(default_factory=lambda: str(uuid4()))
    """Name assigned this trajectory. Defaults to a UUID"""
    stage: int = 0
    """Current stage within the overall :class:`DynamicsProtocol`."""
    timestep: int = 0
    """Timestep within the current stage"""

    def update(self, new_atoms: Atoms, steps_completed: int, finished_stage: bool):
        """Update the state of the current progress

        Args:
            new_atoms: Current structure
            steps_completed: Number of steps completed since the last progress update
            finished_step: Whether the structure has finished a step within the overall protocol
        """

        self.atoms = new_atoms.copy()
        if finished_stage:
            self.stage += 1
            self.timestep = 0
        else:
            self.timestep += steps_completed


@dataclass
class DynamicsStage:
    driver: type[Dynamics]
    """Which dynamics to run as an ASE Dynamics class"""
    timesteps: int | None = None
    """Maximum number of timesteps to run.

    Use ``None`` to run until :attr:`driver` reports the dynamics as converged"""
    driver_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments used to create the driver"""
    run_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the driver's run method"""
    post_fun: Optional[Callable[[Atoms], None]] = None
    """Post-processing function applied after run is complete. Modifies the input arguments"""


class DynamicsProtocol:
    """A protocol for running several stages of dynamics calls together

    Args:
        stages: List of dynamics to be run in sequential order
        scratch_dir: Directory in which to write temporary files
    """

    stages: list[DynamicsStage]
    """List of dynamics processes to run sequentially"""

    def __init__(self, stages: list[DynamicsStage], scratch_dir: Path | None = None):
        self.stages = stages.copy()
        self.scratch_dir = scratch_dir

    # TODO (wardlt): We might need to run dynamics with a physics code, which will require changing the interface
    def run_dynamics(self,
                     start: Progress,
                     model_msg: bytes | State,
                     learner: BaseLearnableForcefield,
                     max_timesteps: int,
                     max_frames: int | None = None,
                     device: str = None) -> tuple[bool, list[Atoms]]:
        """Run dynamics for a maximum number of timesteps using a particular forcefield

        Runs dynamics until the end of a process or until the maximum number of timesteps is reached.

        Args:
            start: Starting point of the dynamic trajectory
            model_msg: Serialized form of the forcefield used for training
            learner: Class used to generate the ASE calculator object
            max_timesteps: Maximum number of timesteps to run dynamics
            max_frames: Maximum number of frames from the atomistic simulation to return for auditing
            device: Device to use for evaluating forcefield
        Returns:
            List of frames selected at used for auditing the dynamics
        """

        # Create a temporary directory in which to run the data
        stage = self.stages[start.stage]  # Pick the current process
        with TemporaryDirectory(dir=self.scratch_dir, prefix='cascade-dyn_', suffix=f'_{start.name}') as tmp:
            tmp = Path(tmp)
            dyn = stage.driver(start.atoms, logfile=str(tmp / 'dyn.log'), **stage.driver_kwargs)

            # Attach the calculator
            calc = learner.make_calculator(model_msg, device)
            atoms = start.atoms
            atoms.calc = calc

            # Attach the trajectory writer
            traj_freq = 1 if max_frames is None else max_timesteps // max_frames
            traj_path = str(tmp / 'run.traj')
            with Trajectory(traj_path, mode='w', atoms=start.atoms) as traj:
                dyn.attach(traj, traj_freq)

                # Run dynamics, then check if we have finished
                converged = dyn.run(steps=max_timesteps, **stage.run_kwargs)
                total_timesteps = max_timesteps + start.timestep  # Total progress along this stage

                if converged and isinstance(dyn, Optimizer):  # Optimization is done if convergence is met
                    done = True
                elif isinstance(dyn, Optimizer) and stage.timesteps is not None:  # Optimization is also done if we've run out of timesteps
                    done = total_timesteps >= stage.timesteps
                else:
                    done = total_timesteps >= stage.timesteps

                if done and stage.post_fun is not None:
                    stage.post_fun(atoms)

            # Read in the trajectory then append the current frame to it
            traj_atoms = read(traj_path, ':')
            if traj_atoms[-1] != atoms:
                traj_atoms.append(atoms)

            return done, traj_atoms
