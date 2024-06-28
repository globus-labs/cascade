from pytest import mark

from cascade.calculator import make_calculator


@mark.parametrize('method', ['blyp', 'pm6', 'b97m'])
def test_make_calculator(method, example_cell, tmpdir):
    calc = make_calculator(method, directory=tmpdir)
    with calc:
        calc.get_potential_energy(example_cell)
