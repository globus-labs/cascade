from ase.calculators.lj import LennardJones
from pytest import mark

from cascade.calculator import make_calculator, EnsembleCalculator


@mark.parametrize('method', ['blyp', 'pm6', 'b97m'])
def test_make_calculator(method, example_cell, tmpdir):
    calc = make_calculator(method, directory=tmpdir)
    with calc:
        calc.get_potential_energy(example_cell)


def test_ensemble_calculator(example_cell):
    lj1, lj2 = LennardJones(sigma=1, eps=1), LennardJones(sigma=1, epsilon=1.1)
    ens = EnsembleCalculator([lj1, lj2])
    assert set(ens.implemented_properties) == set(lj2.implemented_properties)

    forces = ens.get_forces(example_cell)
    assert forces.shape == (len(example_cell), 3)
    assert ens.results['forces_ens'].shape == (2, len(example_cell), 3)

    stress = ens.get_stress(example_cell)
    assert stress.shape == (6,)
    assert ens.results['stress_ens'].shape == (2, 6)
