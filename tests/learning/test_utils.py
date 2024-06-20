import numpy as np

from cascade.learning.utils import estimate_atomic_energies


def test_fit_reference_energies(example_data):

    ref_energies = estimate_atomic_energies(example_data)
    assert len(ref_energies) == 2
    assert np.isclose(ref_energies['H'], 1.)
    assert np.isclose(ref_energies['He'], 2.)
