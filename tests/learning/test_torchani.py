from cascade.learning.torchani.build import make_aev_computer, make_output_nets
from cascade.learning.torchani import TorchANI
from cascade.learning.utils import estimate_atomic_energies


def test_build(example_data):
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
    aev, nn = test_build(example_data)

    # Remove the claculator from the Atoms object (not needed for inference)
    for atoms in example_data:
        atoms.calc = None

    ani = TorchANI()
    batch_energies, batch_forces = ani.evaluate((aev, nn, ref_energies), example_data)
    assert batch_energies.shape == (2,)
    for atoms, forces in zip(example_data, batch_forces):
        assert forces.shape == (len(atoms), 3)
