from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pytest import fixture
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule
from ase import Atoms, io
import numpy as np

_file_dir = Path(__file__).parent / 'files'


@fixture()
def example_cell() -> Atoms:
    """Single water in a box"""
    water = molecule('H2O')
    water.cell = [4.] * 3
    water.pbc = True
    return water


@fixture
def example_data() -> list[Atoms]:
    atoms_1 = Atoms(symbols=['H', 'He'], positions=np.zeros((2, 3)), cell=[5., 5., 5.], pbc=True)
    atoms_2 = Atoms(symbols=['He', 'He'], positions=np.zeros((2, 3)), cell=[5., 5., 5.], pbc=True)

    atoms_1.positions[0, 0] = 3.
    atoms_2.positions[0, 0] = 3.

    atoms_1.calc = SinglePointCalculator(atoms_1, energy=3., forces=np.zeros((2, 3)), stress=np.zeros((3, 3)))
    atoms_2.calc = SinglePointCalculator(atoms_2, energy=4., forces=np.zeros((2, 3)), stress=np.zeros((3, 3)))
    return [atoms_1, atoms_2]


@fixture()
def example_si_data() -> list[Atoms]:
    """Example data where there are 8 pure and 8 Si 2x2x2 supercells with one vacancy"""

    return io.read(_file_dir / 'si-pure-and-vacancy.db', slice(None))
