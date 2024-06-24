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


@dataclass
class Progress:
    """The progress of an atomic state through a dynamics protocol"""

    atoms: Atoms
    """Current atomic structure"""
    name: str = field(default_factory=lambda: str(uuid4()))
    """Name assigned this trajectory. Defaults to a UUID"""
    process: int = 0
    """Current step process within the overall :class:`DynamicsProtocol`."""
    timestep: int = 0
    """Timestep within the current process"""

    def update(self, new_atoms: Atoms, steps_completed: int, finished_step: bool):
        """Update the state of the current progress

        Args:
            new_atoms: Current structure
            steps_completed: Number of steps completed since the last progress update
            finished_step: Whether the structure has finished a step within the overall protocol
        """

        self.atoms = new_atoms.copy()
        if finished_step:
            self.process += 1
            self.timestep = 0
        else:
            self.timestep += steps_completed


DynamicsStep = tuple[type[Dynamics], int | None, dict[str, Any], dict[str, Any], Optional[Callable[[Atoms], None]]]
"""Definition of a step within a dynamics protocol:

1. Which dynamics to run. An ASE Dynamics class
2. Maximum number of timesteps to run. Use `None` to run until convergence is reached
3. Options to class, used when instantiating the dynamics
4. Options for the run method
5. Post-processing function applied after run is complete"""


class DynamicsProtocol:
    """A protocol for running several steps of dynamics calls together

    Args:
        processes: List of dynamics to be run in sequential order
        scratch_dir: Directory in which to write temporary files
    """

    processes: list[DynamicsStep]  # TODO (wardlt): Find another name besides "step" to describe this
    """List of steps to run expressed"""

    def __init__(self, processes: list[DynamicsStep], scratch_dir: Path | None = None):
        self.processes = processes.copy()
        self.scratch_dir = scratch_dir

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
        dyn_cls, dyn_steps, dyn_args, run_args, post_func = self.processes[start.process]  # Pick the current process
        with TemporaryDirectory(dir=self.scratch_dir, prefix='cascade-dyn_', suffix=f'_{start.name}') as tmp:
            tmp = Path(tmp)
            dyn = dyn_cls(start.atoms, logfile=str(tmp / 'dyn.log'), **dyn_args)

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
                converged = dyn.run(steps=max_timesteps, **run_args)
                total_timesteps = max_timesteps + start.timestep  # Total progress along this step

                if converged and isinstance(dyn, Optimizer):  # Optimization is done if convergence is met
                    done = True
                elif isinstance(dyn, Optimizer) and dyn_steps is not None:  # Optimization is also done if we've run out of timesteps
                    done = total_timesteps >= dyn_steps
                else:
                    done = total_timesteps >= dyn_steps

                if done and post_func is not None:
                    post_func(atoms)

            # Read in the trajectory then append the current frame to it
            traj_atoms = read(traj_path, ':')
            traj_atoms.append(atoms)

            return done, traj_atoms
