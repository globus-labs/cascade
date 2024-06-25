Running Dynamics
================

.. note:: A picture would be nice.

The :class:`~cascade.dynamics.DynamicsProtocol` class from Cascade propagates a dynamic system
using the learned surrogate and returns copies of the system at many timesteps for use
by the Auditing code.

Create a Dynamics class by defining what happens in each stage of a process as a :class:`~cascade.dynamics.DynamicsStage`:

1. *Which type of dynamics*, as an ASE Dynamics class (e.g., an optimizer or molecular dynamics method)
2. *Maximum number of timesteps*. Use ``None`` to run until convergence is reached
3. *Options to Dynamics*, such as the temperature for molecular dynamics or time per timestep
4. *Options for the run method*, such as the convergence threshold
5. *Post-processing function applied after run is complete*, which modifies the Atoms object as the list step.

A dynamics process could have multiple stages, such as an optimization followed by molecular dynamics
under several different control conditions.

.. note:: We need a better name than process for which type of dynamics it is running.

.. code-block:: python

    # Optimization then constant-energy molecular dynamics
    dyn_steps = [
        DynamicsStage(BFGS, 100, {}, {'fmax': 0.001}, partial(MaxwellBoltzmannDistribution, temperature_K=300)),
        DynamicsStage(VelocityVerlet, 100, {'timestep': 0.5}, {}, None)
    ]
    dyn = DynamicsProtocol(stages=dyn_steps)


The :class:`~cascade.dynamics.Progress` of a single trajectory along the dynamics is defined by which process step it is running
and how many timesteps it has completed along that process.

Run the dynamics by passing the starting state, a surrogate model, and a maximum number of timestep to run.
The dynamics process will return whether the current process was completed and a series of snapshots along the trajectory.

.. code-block:: python

    start = Progress(atoms)
    done, traj = dyn.run_dynamics(start, model_msg, TorchANI(), 100)

Use an auditing strategy to determine whether the surrogate model was reliable during the last call to the dynamics
and, if so, increment the Progress through its :meth:`~cascade.dynamics.Progress.update` method before
continuing to propagate the dynamics.

.. code-block:: python

    if audit(traj):
        start.update(traj[-1], steps_completed=100, finished_step=done)
    done, traj = dyn.run_dynamics(start, example_model, TorchANI(), 5)
