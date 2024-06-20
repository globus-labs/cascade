from ase import Atoms
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from cascade.learning.utils import estimate_atomic_energies


def test_fit_reference_energies():
    atoms_1 = Atoms(symbols=['H', 'He'], positions=np.zeros((2, 3)))
    atoms_2 = Atoms(symbols=['He', 'He'], positions=np.zeros((2, 3)))

    atoms_1.calc = SinglePointCalculator(atoms_1, energy=3.)
    atoms_2.calc = SinglePointCalculator(atoms_2, energy=4.)

    ref_energies = estimate_atomic_energies([atoms_1, atoms_2])
    assert len(ref_energies) == 2
    assert np.isclose(ref_energies['H'], 1.)
    assert np.isclose(ref_energies['He'], 2.)
