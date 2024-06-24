from pytest import fixture
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import numpy as np


@fixture
def example_data() -> list[Atoms]:
    atoms_1 = Atoms(symbols=['H', 'He'], positions=np.zeros((2, 3)), cell=[5., 5., 5.], pbc=True)
    atoms_2 = Atoms(symbols=['He', 'He'], positions=np.zeros((2, 3)), cell=[5., 5., 5.], pbc=True)

    atoms_1.positions[0, 0] = 3.
    atoms_2.positions[0, 0] = 3.

    atoms_1.calc = SinglePointCalculator(atoms_1, energy=3., forces=np.zeros((2, 3)))
    atoms_2.calc = SinglePointCalculator(atoms_2, energy=4., forces=np.zeros((2, 3)))
    return [atoms_1, atoms_2]
