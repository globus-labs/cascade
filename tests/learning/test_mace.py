from mace.calculators import mace_mp
from pytest import fixture, mark
import numpy as np

from cascade.learning.mace import MACEState, MACEInterface


@fixture()
def mace() -> MACEState:
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    return calc.models[0]


@mark.parametrize('reset_weights', [False, True])
def test_training(example_data, mace, reset_weights):
    """Run example network on test data"""

    # Get baseline predictions, train
    mi = MACEInterface()
    orig_e, orig_f, orig_s = mi.evaluate(mace, example_data)
    model_msg, log = mi.train(mace, example_data, example_data, 2, batch_size=2, reset_weights=reset_weights, patience=1)
    assert len(log) == 2

    # Make sure the predictions change
    new_e, new_f, new_s = mi.evaluate(model_msg, example_data)
    assert not np.isclose(new_e, orig_e).all()
    for new, orig in zip(new_f, orig_f):
        assert not np.isclose(new, orig).all()


def test_inference(mace, example_data):
    # Delete any previous results from the example data
    for atoms in example_data:
        atoms.calc = None

    mi = MACEInterface()
    energy, forces, stresses = mi.evaluate(mace, example_data)

    assert energy.shape == (2,)
    for atoms, f in zip(example_data, forces):
        assert f.shape == (len(atoms), 3)
    assert stresses.shape == (2, 3, 3)

    # Test the calculator interface
    calc = mi.make_calculator(mace, 'cpu')
    atoms = example_data[0]
    atoms.calc = calc
    assert np.isclose(atoms.get_potential_energy(), energy[0]).all()
    assert np.isclose(atoms.get_forces(), forces[0]).all()
    assert np.isclose(atoms.get_stress(voigt=False), stresses[0]).all()


def test_freeze(example_data, mace):
    # Get baseline predictions, train
    mi = MACEInterface()
    model_msg, _ = mi.train(mace, example_data, example_data, 2, batch_size=2, patience=1, num_freeze=2)
    model: MACEState = mi.get_model(model_msg)
    is_trainable = [all(y.requires_grad for y in x.parameters()) for x in model.children()]
    assert not any(is_trainable[:2])
    assert all(is_trainable[4:6])  # Layers >4 include some layers which are not trainable
