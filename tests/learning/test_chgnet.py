from contextlib import redirect_stdout
from os import devnull

from chgnet.model import CHGNet
from pytest import fixture, mark
import numpy as np

from cascade.learning.chgnet import CHGNetInterface


@fixture()
def chgnet() -> CHGNet:
    with open(devnull, 'w') as fp, redirect_stdout(fp):
        return CHGNet.load(use_device='cpu', verbose=False)


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


@mark.parametrize('reset_weights', [False, True])
def test_training(example_data, chgnet, reset_weights):
    """Run example network on test data"""

    # Get baseline predictions, train
    cgi = CHGNetInterface()
    orig_e, orig_f, orig_s = cgi.evaluate(chgnet, example_data)
    model_msg, log = cgi.train(chgnet, example_data, example_data, 2, batch_size=2, reset_weights=reset_weights)
    assert len(log) == 2

    # Make sure the predictions change
    new_e, new_f, new_s = cgi.evaluate(model_msg, example_data)
    assert not np.isclose(new_e, orig_e).all()
    for new, orig in zip(new_f, orig_f):
        assert not np.isclose(new, orig).all()
