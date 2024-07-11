"""Test the proxima calculator"""
from pathlib import Path
import pickle as pkl

from ase.calculators.cp2k import CP2K
from pytest import fixture
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


def test_proxima(starting_frame, simple_model, target_calc, tmpdir):
    tmpdir = Path(tmpdir)

    # Make the calculator
    model_msgs, learner = simple_model
    calc = SerialLearningCalculator(
        target_calc=target_calc,
        learner=learner,
        models=model_msgs,
        train_kwargs={'num_epochs': 4, 'batch_size': 4},
        db_path=tmpdir / 'data.db',
        target_ferr=1e-12,  # Should be unachievable
    )

    # Test a single point calculation
    calc.get_forces(starting_frame)
    assert not calc.used_surrogate
    assert len(calc.error_history) == 1
    assert (tmpdir / 'data.db').is_file()

    # Run enough calculations to determine a threshold
    for i in range(calc.parameters['history_length'] - 1):
        assert calc.threshold is None
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.02, seed=i)
        calc.get_forces(new_atoms)

        assert not calc.used_surrogate

    assert calc.threshold is not None, len(calc.error_history)

    # No calculation should have been below the threshold
    assert all(a > calc.threshold for a, _ in calc.error_history)

    # Run enough to start training the model
    original_model = calc.parameters['models'][0]
    for i in range(20):  # Run e
        new_atoms = starting_frame.copy()
        new_atoms.rattle(0.2, seed=i + 50)  # Keep seed different from loop above
        calc.get_forces(new_atoms)
        assert not calc.used_surrogate

        if len(read(calc.parameters['db_path'], ':')) > 10:
            break
    else:
        raise ValueError('Surrogate ran too often')

    calc.retrain_surrogate()
    assert original_model is not calc.parameters['models'][0], 'Model was not retrained'

    # Make a call where the surrogate will run, so we can test updating the state
    calc.threshold = 1e10
    calc.get_forces(starting_frame)
    assert calc.surrogate_calc is not None

    # Pull the state and make sure it has the models in it as bytestrings
    state = pkl.loads(pkl.dumps(calc.get_state()))
    assert 'models' in state
    assert isinstance(state['models'][0], bytes)

    calc.threshold = None
    calc.set_state(state)
    assert calc.threshold is not None
