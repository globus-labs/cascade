from mace.calculators import mace_mp
from pytest import fixture, mark
import numpy as np

from cascade.learning.mace import MACEState, MACEInterface


@fixture()
def mace() -> MACEState:
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    return calc.models[0], calc.z_table, calc.r_max


@mark.parametrize('reset_weights', [False, True])
def test_training(example_data, mace, reset_weights):
    """Run example network on test data"""

    # Get baseline predictions, train
    mi = MACEInterface()
    orig_e, orig_f, orig_s = mi.evaluate(mace, example_data)
    model_msg, log = mi.train(mace, example_data, example_data, 2, batch_size=2, reset_weights=reset_weights)
    assert len(log) == 2

    # Make sure the predictions change
    new_e, new_f, new_s = mi.evaluate(model_msg, example_data)
    assert not np.isclose(new_e, orig_e).all()
    for new, orig in zip(new_f, orig_f):
        assert not np.isclose(new, orig).all()
