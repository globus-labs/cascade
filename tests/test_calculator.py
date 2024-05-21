from ase.build import molecule
from ase import Atoms
from pytest import fixture, mark

from cascade.calculator import make_calculator, create_run_hash


@fixture()
def example_cell() -> Atoms:
    """Single water in a box"""
    water = molecule('H2O')
    water.cell = [4.] * 3
    water.pbc = True
    return water


@mark.parametrize('method', ['blyp', 'pm6', 'b97m'])
def test_make_calculator(method, example_cell, tmpdir):
    calc = make_calculator(method, directory=tmpdir)
    with calc:
        calc.get_potential_energy(example_cell)


def test_hasher(example_cell):
    # Should be the same with the same cell and the same kwargs, regardless of order
    assert create_run_hash(example_cell) == create_run_hash(example_cell)
    assert create_run_hash(example_cell, a=1, b=2) == create_run_hash(example_cell, b=2, a=1)

    # Should be the same if the kwargs or structure change
    assert create_run_hash(example_cell, a=1, b=2) != create_run_hash(example_cell, b=2, a=2)
    different_cell = example_cell.copy()
    different_cell.set_chemical_symbols(['S', 'H', 'H'])
    assert create_run_hash(different_cell) != create_run_hash(example_cell)
