from ase.build import molecule
from ase import Atoms
from pytest import fixture, mark

from cascade.calculator import make_calculator


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
