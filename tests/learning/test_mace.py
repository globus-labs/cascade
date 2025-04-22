from mace.calculators import mace_mp
from pytest import fixture, mark
import numpy as np

from cascade.learning.finetuning import MultiHeadConfig
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
    assert not is_trainable[0]
    assert all(is_trainable[4:6])  # Layers >4 include some layers which are not trainable


def test_make_heads(mace):
    mi = MACEInterface()
    new_models = mi.create_extra_heads(mace, 2)
    assert len(new_models) == 2
    assert new_models[0].interactions is new_models[1].interactions
    assert new_models[0].interactions is mace.interactions

    assert new_models[1].readouts is not new_models[0].readouts
    assert new_models[1].readouts is not mace.readouts


@mark.parametrize('epoch_frequency,num_downselect',
                  [(1, None), (2, None), (1, 1)])
def test_replay(example_data, mace, epoch_frequency, num_downselect):
    # Create the replay information
    replay = MultiHeadConfig(
        num_downselect=num_downselect,
        original_dataset=example_data,
        epoch_frequency=epoch_frequency
    )

    # Get baseline predictions, train
    mi = MACEInterface()
    _, log = mi.train(mace, example_data, example_data, 4, batch_size=2, patience=1, replay=replay)
    assert 'total_loss_replay' in log.columns
    assert log['total_loss_replay'].isna().sum() == (4 - 4 // epoch_frequency)  # Ensure reply is skipped occasionally
