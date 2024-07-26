"""Test the proxima calculator"""
import logging
from pathlib import Path
import pickle as pkl

from ase.calculators.cp2k import CP2K
from pytest import fixture
from ase.db import connect
from ase.io import read
from ase import Atoms

from cascade.calculator import make_calculator
from cascade.learning.torchani import ANIModelContents, TorchANI
from cascade.learning.torchani.build import make_aev_computer, make_output_nets
from cascade.proxima import SerialLearningCalculator


@fixture()
def starting_frame(example_cell) -> Atoms:
    return example_cell


@fixture()
def simple_model(starting_frame) -> tuple[list[ANIModelContents], TorchANI]:
    species = list(set(starting_frame.symbols))
    aev_computer = make_aev_computer(species)
    model_msgs = [
        (aev_computer, make_output_nets(species, aev_computer, hidden_layers=1), dict((s, 0) for s in species))
        for _ in range(2)
    ]
    return model_msgs, TorchANI()


@fixture()
def target_calc() -> CP2K:
    return make_calculator('blyp')


@fixture()
def simple_proxima(simple_model, target_calc, tmpdir):
    tmpdir = Path(tmpdir)
    model_msgs, learner = simple_model
    return SerialLearningCalculator(
        target_calc=target_calc,
        learner=learner,
        models=model_msgs,
        train_kwargs={'num_epochs': 4, 'batch_size': 4},
        db_path=tmpdir / 'data.db',
        target_ferr=1e-12,  # Should be unachievable
    )


def test_proxima(starting_frame, simple_proxima):
    # Test a single point calculation
    simple_proxima.get_forces(starting_frame)
    assert not simple_proxima.used_surrogate
    assert len(simple_proxima.error_history) == 1
    assert simple_proxima.parameters['db_path'].is_file()
    assert simple_proxima.target_invocations == simple_proxima.total_invocations == 1

    # Run enough calculations to determine a threshold
    for i in range(simple_proxima.parameters['history_length'] - 1):
        assert simple_proxima.threshold is None
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.02, seed=i)
        simple_proxima.get_forces(new_atoms)

        assert not simple_proxima.used_surrogate

    assert simple_proxima.threshold is not None, len(simple_proxima.error_history)

    # No calculation should have been below the threshold
    assert all(a > simple_proxima.threshold for a, _ in simple_proxima.error_history)

    # Run enough to start training the model
    original_model = simple_proxima.parameters['models'][0]
    for i in range(20):  # Run e
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.2, seed=i + 50)  # Keep seed different from loop above
        simple_proxima.get_forces(new_atoms)
        assert not simple_proxima.used_surrogate

        if len(read(simple_proxima.parameters['db_path'], ':')) > 10:
            break
    else:
        raise ValueError('Surrogate ran too often')

    simple_proxima.retrain_surrogate()
    assert original_model is not simple_proxima.parameters['models'][0], 'Model was not retrained'

    # Make a call where the surrogate will run, so we can test updating the state
    simple_proxima.threshold = 1e10
    simple_proxima.get_forces(starting_frame)
    assert simple_proxima.surrogate_calc is not None

    # Pull the state and make sure it has the models in it as bytestrings
    state = pkl.loads(pkl.dumps(simple_proxima.get_state()))
    assert 'models' in state
    assert isinstance(state['models'][0], bytes)

    simple_proxima.threshold = None
    simple_proxima.set_state(state)
    assert simple_proxima.threshold is not None


def test_max_size(starting_frame, simple_proxima, target_calc, caplog):
    # Insert 16 random calculations into the db
    with connect(simple_proxima.parameters['db_path']) as db:
        for i in range(12):
            new_frame = starting_frame.copy()
            new_frame.calc = target_calc
            new_frame.rattle(0.01, seed=i)
            new_frame.get_forces()
            db.write(new_frame)
        assert len(db) == 12

    # Make the maximum training size 8, then verify that only 8 frames are taken for training
    simple_proxima.parameters['train_max_size'] = 8
    with caplog.at_level(logging.DEBUG, logger='cascade.proxima'):
        simple_proxima.retrain_surrogate()
    assert '8 atoms and validating on 1' in caplog.messages[-2]
    assert simple_proxima.model_version == 1
