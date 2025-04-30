import numpy as np

from cascade.learning.finetuning import filter_by_elements
from cascade.learning.utils import estimate_atomic_energies


def test_fit_reference_energies(example_data):
    ref_energies = estimate_atomic_energies(example_data)
    assert len(ref_energies) == 2
    assert np.isclose(ref_energies['H'], 1.)
    assert np.isclose(ref_energies['He'], 2.)


def test_get_relevant_entries(example_data):
    only_he = list(filter_by_elements(example_data, ['He']))
    assert len(only_he) == 1
    assert only_he[0].get_chemical_formula(empirical=True) == 'He'

    h_he = list(filter_by_elements(example_data, ['H', 'He']))
    assert len(h_he) == 2

    h_he_o = list(filter_by_elements(example_data, ['H', 'He', 'O']))
    assert len(h_he_o) == 2
