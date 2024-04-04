"""Make a CP2K calculator ready to run a specific environment"""
from pathlib import Path
from string import Template

from ase.calculators.cp2k import CP2K
from ase import units

_file_dir = Path(__file__).parent / 'files'


def make_calculator(
        method: str,
        multiplicity: int,
        directory: str = 'run'
) -> CP2K:
    """Make a calculator ready to run with different configurations

    Supported methods:
        - `pm6`: A force-matched PM6 shown by `Welborn et al <https://onlinelibrary.wiley.com/doi/10.1002/jcc.23887>`_
           to agree better for proeprties of liquid water than the original formulation
        - `blyp`: The BLYP GGA potential with D3 vdW corrections, suggested by `Lin et al. <https://pubs.acs.org/doi/10.1021/ct3001848>`_
            to give the best properties of liquid water
        - `b97m`: The `B97M-rV <http://xlink.rsc.org/?DOI=C6SC04711D>`_ metaGGA functional

    Args:
        method: Which method to run
        multiplicity: Multiplicity of the system
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

    inp = Template(input_file.read_text()).substitute(
        mult=multiplicity,
    )

    cp2k_opts = dict(
        xc=None,
        inp=inp,
        basis_set_file=None,
        basis_set=None,
        pseudo_potential=None,
        poisson_solver=None,
    )
    return CP2K(directory=directory,
                command='/home/lward/Software/cp2k-2024.1/exe/local_cuda/cp2k_shell.ssmp',
                stress_tensor=True,
                max_scf=max_scf,
                cutoff=cutoff,
                potential_file=None,
                **cp2k_opts)
