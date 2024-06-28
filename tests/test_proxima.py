"""Test the proxima calculator"""
from ase.calculators.cp2k import CP2K
from pytest import fixture
from ase import Atoms, build

from cascade.calculator import make_calculator
from cascade.learning.torchani import ANIModelContents, TorchANI
from cascade.learning.torchani.build import make_aev_computer, make_output_nets
from cascade.proxima import SerialLearningCalculator


@fixture()
def starting_frame(example_cell) -> Atoms:
    return example_cell


@fixture()
def simple_model(starting_frame) -> tuple[tuple[ANIModelContents, ...], TorchANI]:
    species = list(set(starting_frame.symbols))
    aev_computer = make_aev_computer(species)
    model_msgs = [
        (aev_computer, make_output_nets(species, aev_computer, hidden_layers=1), dict((s, 0) for s in species))
        for _ in range(2)
    ]
    return tuple(model_msgs), TorchANI()


@fixture()
def target_calc() -> CP2K:
    return make_calculator('blyp')


def test_proxima(starting_frame, simple_model, target_calc):
    model_msgs, learner = simple_model

    calc = SerialLearningCalculator(
        target_calc=target_calc,
        learner=learner,
        models=model_msgs,
        db_path='data.db',
    )

    calc.get_forces(starting_frame)
