"""Make a CP2K calculator ready to run a specific environment"""
from pathlib import Path
from string import Template

from ase.calculators.cp2k import CP2K
from ase import units

_file_dir = Path(__file__).parent / 'files'


def make_calculator(
        method: str,
        multiplicity: int = 0,
        command: str | None = None,
        directory: str = 'run'
) -> CP2K:
    """Make a calculator ready to run with different configurations

    Supported methods:
        - `pm6`: A force-matched PM6 shown by `Welborn et al <https://onlinelibrary.wiley.com/doi/10.1002/jcc.23887>`_
           to agree better for properties of liquid water than the original formulation
        - `blyp`: The BLYP GGA potential with D3 vdW corrections, suggested by `Lin et al. <https://pubs.acs.org/doi/10.1021/ct3001848>`_
            to give the best properties of liquid water
        - `b97m`: The `B97M-rV <http://xlink.rsc.org/?DOI=C6SC04711D>`_ metaGGA functional

    Args:
        method: Which method to run
        multiplicity: Multiplicity of the system
        command: Command used to launch CP2K. Defaults to whatever ASE autodetermines or ``cp2k_shell``
        directory: Path in which to write run file
    Returns:
        Calculator configured for target method
    """

    # Get the input file and cutoff energy
    input_file = _file_dir / f'cp2k-{method}-template.inp'
    cutoff = {
        'b97m': 800 * units.Ry,
        'blyp': 500 * units.Ry
    }.get(method, None)
    max_scf = {'b97m': 32}.get(method, 128)

    # Get the basis set and potential type
    basis_set = {
        'blyp': 'DZVP-MOLOPT-SR-GTH',
        'b97m': 'DZVP-MOLOPT-SR-GTH',
    }.get(method)
    potential = {
        'blyp': 'GTH-PBE',
        'b97m': 'GTH-BLYP'
    }.get(method)

    inp = Template(input_file.read_text()).substitute(mult=multiplicity)  # No changes as of it

    cp2k_opts = dict(
        xc=None,
        inp=inp,
        basis_set=basis_set,
        pseudo_potential=potential,
        poisson_solver=None,
    )
    if command is not None:
        cp2k_opts['command'] = command
    return CP2K(directory=directory,
                stress_tensor=True,
                max_scf=max_scf,
                cutoff=cutoff,
                potential_file=None,
                set_pos_file=True,
                **cp2k_opts)
