"""Test the proxima calculator"""
from collections import deque
from pathlib import Path
from math import isclose
import pickle as pkl
import logging

from ase.calculators.cp2k import CP2K
from pytest import fixture
from ase.db import connect
from ase.io import read
from ase import Atoms

from cascade.calculator import make_calculator
from cascade.learning.torchani import ANIModelContents, TorchANI
from cascade.learning.torchani.build import make_aev_computer, make_output_nets
from cascade.proxima import SerialLearningCalculator
from cascade.utils import canonicalize

_initial_calcs: list[Atoms] = []  # Used if we want the database pre-populated


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
    yield make_calculator('lda')


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


@fixture()
def initialized_db(simple_proxima, target_calc, starting_frame):
    """Initialize the database"""
    # Compute a set of initial calcs if required
    global _initial_calcs
    if len(_initial_calcs) == 0:
        for i in range(12):
            new_frame = starting_frame.copy()
            new_frame.calc = target_calc
            new_frame.rattle(0.01, seed=i)
            new_frame.get_forces()
            _initial_calcs.append(canonicalize(new_frame))

    with connect(simple_proxima.parameters['db_path']) as db:
        for new_frame in _initial_calcs:
            db.write(new_frame)
        assert len(db) == 12


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


def test_max_size(starting_frame, simple_proxima, target_calc, initialized_db, caplog):
    # Make the maximum training size 8, then verify that only 8 frames are taken for training
    simple_proxima.parameters['train_max_size'] = 8
    with caplog.at_level(logging.DEBUG, logger='cascade.proxima'):
        simple_proxima.retrain_surrogate()
    assert '8 atoms and validating on 1' in caplog.messages[-2]
    assert simple_proxima.model_version == 1


def test_pretrained_threshold(starting_frame, simple_proxima, target_calc):
    # Duplicate the model to simulate a pre-trained model
    model = simple_proxima.parameters['models'][0]
    simple_proxima.parameters['models'] = [model] * 2

    # Fake the history
    simple_proxima.error_history = deque([(0, 1) for _ in range(simple_proxima.parameters['history_length'])])

    # Run a single calculation with the starting frame to trigger updating the threshold
    assert simple_proxima.threshold is None and simple_proxima.alpha is None
    simple_proxima.get_forces(starting_frame)
    assert isclose(simple_proxima.threshold, 0., abs_tol=1e-12) and simple_proxima.alpha is None  # It sets to zero if all UQs are the same


def test_blending(starting_frame, simple_model, target_calc, tmpdir):

    # how many steps to blend for
    n_blending_steps = 3

    # create a different proxima instance since we will have different parameters
    tmpdir = Path(tmpdir)
    model_msgs, learner = simple_model
    calc = SerialLearningCalculator(
        target_calc=target_calc,
        learner=learner,
        models=model_msgs,
        train_kwargs={'num_epochs': 4, 'batch_size': 4},
        db_path=tmpdir / 'data.db',
        n_blending_steps=n_blending_steps,
        min_target_fraction=0,
        target_ferr=1e12,  # Should always use the ML
    )

    # Run enough calculations to determine a threshold
    for i in range(calc.parameters['history_length']):
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.02, seed=i)
        calc.get_forces(new_atoms)

    # assert the incrementing happens the way its supposed to
    for i in range(1, n_blending_steps+1):
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.2, seed=i+100)  # don't reuse above seeds
        calc.get_stress(new_atoms)  # test blending of stresses since this caused errors in the past
        assert calc.used_surrogate
        assert calc.blending_step == i
    assert calc.lambda_target == 0

    # and that we dont go out of bounds
    for i in range(2):
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.2, seed=i+500)  # don't reuse above seeds
        calc.get_forces(new_atoms)
        assert calc.used_surrogate
        assert calc.blending_step == n_blending_steps
        assert calc.lambda_target == 0

    # now change the error target such that the surrogate will *never* be called
    calc.parameters['target_ferr'] = 1e-12
    calc.threshold = 0  # force the threshold to be small
    # make sure blending back to target works as expected
    for i in range(n_blending_steps-1, -1, -1):
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.2, seed=i+800)  # don't reuse above seeds
        calc.get_forces(new_atoms)
        assert not calc.used_surrogate
        assert calc.blending_step == i
    assert calc.blending_step == 0
    assert calc.lambda_target == 1

    # again assert we dont go out of bounds and blending stays the same
    for i in range(2):
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.2, seed=i+1200)  # don't reuse above seeds
        calc.get_forces(new_atoms)
        assert not calc.used_surrogate
        assert calc.blending_step == 0
        assert calc.lambda_target == 1
