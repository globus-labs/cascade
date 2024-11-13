from io import BytesIO

import torch
from pytest import fixture
import schnetpack as spk
import numpy as np


from cascade.learning.spk import SchnetPackInterface

@fixture
def schnet():

    # Make the input representation
    cutoff = 5
    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=32,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    # Output layers
    pred_energy = spk.atomistic.Atomwise(n_in=32, output_key='energy')
    pred_forces = spk.atomistic.Forces(calc_stress=True)

    model = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[spk.atomistic.Strain(), pairwise_distance],
        output_modules=[pred_energy, pred_forces],
    )
    return model

def test_inference(schnet, example_data):
    # Delete any previous results from the example data
    for atoms in example_data:
        atoms.calc = None

    mi = SchnetPackInterface()
    energy, forces, stresses = mi.evaluate(schnet, example_data)

    assert energy.shape == (2,)
    for atoms, f in zip(example_data, forces):
        assert f.shape == (len(atoms), 3)
    assert stresses.shape == (2, 3, 3)

    # Test the calculator interface
    calc = mi.make_calculator(schnet, 'cpu')
    atoms = example_data[0]
    atoms.calc = calc
    assert np.isclose(atoms.get_potential_energy(), energy[0], atol=1e-4).all()
    assert np.allclose(atoms.get_forces(), forces[0], atol=1e-3)
    assert np.allclose(atoms.get_stress(voigt=False), stresses[0], atol=1e-3)
