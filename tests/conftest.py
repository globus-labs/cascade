from ase.build import molecule
from ase import Atoms
from pytest import fixture


@fixture()
def example_cell() -> Atoms:
    """Single water in a box"""
    water = molecule('H2O')
    water.cell = [4.] * 3
    water.pbc = True
    return water
