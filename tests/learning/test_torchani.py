import numpy as np

from cascade.learning.torchani.build import make_aev_computer, make_output_nets
from cascade.learning.torchani import TorchANI, adjust_energy_scale, make_data_loader
from cascade.learning.utils import estimate_atomic_energies


def build_model(example_data):
    """Build an example network"""

    ref_energies = estimate_atomic_energies(example_data)
    species = list(ref_energies.keys())
    aev = make_aev_computer(species)
    nn = make_output_nets(species, aev)
    return aev, nn


def test_inference(example_data):
    """Run example network on test data"""

    # Make the model requirements
    ref_energies = estimate_atomic_energies(example_data)
    aev, nn = build_model(example_data)

    # Remove the calculator from the Atoms object (not needed for inference)
    for atoms in example_data:
        atoms.calc = None

    ani = TorchANI()
    batch_energies, batch_forces, batch_stresses = ani.evaluate((aev, nn, ref_energies), example_data)
    assert batch_energies.shape == (2,)
    for atoms, forces in zip(example_data, batch_forces):
        assert forces.shape == (len(atoms), 3)
    assert batch_stresses.shape == (2, 3, 3)

    # Test the calculator interface
    calc = ani.make_calculator((aev, nn, ref_energies), 'cpu')
    assert next(nn.parameters()).requires_grad
    atoms = example_data[0]
    atoms.calc = calc
    assert np.isclose(atoms.get_potential_energy(), batch_energies[0]).all()
    assert np.isclose(atoms.get_forces(), batch_forces[0]).all()
    assert np.isclose(atoms.get_stress(voigt=False), batch_stresses[0]).all()


def test_training(example_data):
    """Run example network on test data"""

    # Make the model requirements
    ref_energies = estimate_atomic_energies(example_data)
    aev, nn = build_model(example_data)

    # Get baseline predictions, train
    ani = TorchANI()
    orig_e, orig_f, orig_s = ani.evaluate((aev, nn, ref_energies), example_data)
    model_msg, log = ani.train((aev, nn, ref_energies), example_data, example_data, 2, batch_size=2)
    assert len(log) == 2

    # Make sure the predictions change
    new_e, new_f, new_s = ani.evaluate((aev, nn, ref_energies), example_data)
    assert not np.isclose(new_e, orig_e).all()
    for new, orig in zip(new_f, orig_f):
        assert not np.isclose(new, orig).all()


def test_scale_energy(example_si_data):
    # Make the model requirements
    ref_energies = estimate_atomic_energies(example_si_data)
    aev, nn = build_model(example_si_data)

    # Get baseline predictions, ensure results don't change if we reset, train, or scale energies
    ani = TorchANI()
    orig_e, _, _ = ani.evaluate((aev, nn, ref_energies), example_si_data)
    loader = make_data_loader(example_si_data, list(ref_energies), batch_size=2, train=False)
    ref_energies_array = np.array(list(ref_energies.values())).astype(np.float32)
    adjust_energy_scale(aev, nn, loader, ref_energies_array)

    scaled_e, _, _ = ani.evaluate((aev, nn, ref_energies), example_si_data)

    assert np.isclose(np.std([e / len(a) for e, a in zip(scaled_e, example_si_data)]),
                      np.std([a.get_potential_energy() / len(a) for a in example_si_data]), atol=0.1)


def test_multi_size_batches(example_si_data):
    """Test that inference with multiple sizes of cell does not change the answer"""

    ref_energies = estimate_atomic_energies(example_si_data)
    aev, nn = build_model(example_si_data)

    ani = TorchANI()
    single_e, single_f, single_s = ani.evaluate((aev, nn, ref_energies), example_si_data, batch_size=1)
    all_e, all_f, all_s = ani.evaluate((aev, nn, ref_energies), example_si_data, batch_size=16)

    assert np.allclose(single_e, all_e)
    assert np.allclose(single_s, all_s)
    for s, a in zip(single_f, all_f):
        assert np.allclose(s, a, atol=1e-3), np.abs(s - a).max()
