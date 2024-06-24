"""Test class which runs a dynamics protocol"""
from functools import partial

import numpy as np
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from pytest import fixture

from cascade.dynamics import DynamicsProtocol, Progress
from cascade.learning.torchani import ANIModelContents, TorchANI
from cascade.learning.torchani.build import make_aev_computer, make_output_nets
from cascade.learning.utils import estimate_atomic_energies


@fixture()
def example_model(example_data) -> ANIModelContents:
    ref_energies = estimate_atomic_energies(example_data)
    species = list(ref_energies.keys())
    aev = make_aev_computer(species)
    nn = make_output_nets(species, aev)
    return aev, nn, ref_energies


def test_dynamics(example_model, example_data):
    # Prepare the protocol
    dyn_steps = [
        (BFGS, 100, {}, {'fmax': 0.001}, partial(MaxwellBoltzmannDistribution, temperature_K=300)),
        (VelocityVerlet, 100, {'timestep': 0.5}, {}, None)
    ]
    dyn = DynamicsProtocol(processes=dyn_steps)

    # Prepare the starting point
    start = Progress(example_data[0])
    assert start.name is not None

    # Make the ANI model and run the steps
    done, traj = dyn.run_dynamics(start, example_model, TorchANI(), 1)
    assert not done, 'Dynamics converged'
    assert 5 > len(traj) > 1  # Should have the first structure and last with some repeats

    # Run until convergence
    done, traj = dyn.run_dynamics(start, example_model, TorchANI(), 5)
    assert done, 'Dynamics did not converge'
    assert (np.abs(traj[-1].get_forces()) <= 0.001).all()

    # Update the state and then run the next step
    start.update(traj[-1], steps_completed=len(traj), finished_step=True)
    assert start.process == 1  # Should be on to the next step
    assert start.timestep == 0  # Should not have any advanced
    done, traj = dyn.run_dynamics(start, example_model, TorchANI(), 100, 10)

    assert done
    assert len(traj) >= 10
