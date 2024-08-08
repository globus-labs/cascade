from chgnet.model import CHGNet
from pytest import fixture
import numpy as np

from cascade.learning.chgnet import CHGNetInterface


@fixture()
def chgnet() -> CHGNet:
    return CHGNet.load(use_device='cpu')


def test_inference(chgnet, example_data):
    cg = CHGNetInterface()
    energy, forces, stresses = cg.evaluate(chgnet, example_data)

    assert energy.shape == (2,)
    for atoms, f in zip(example_data, forces):
        assert f.shape == (len(atoms), 3)
    assert stresses.shape == (2, 3, 3)

    # Test the calculator interface
    calc = cg.make_calculator(chgnet, 'cpu')
    atoms = example_data[0]
    atoms.calc = calc
    assert np.isclose(atoms.get_potential_energy(), energy[0]).all()
    assert np.isclose(atoms.get_forces(), forces[0]).all()
    assert np.isclose(atoms.get_stress(voigt=False), stresses[0]).all()
