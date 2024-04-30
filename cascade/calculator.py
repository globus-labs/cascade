"""Make a CP2K calculator ready to run a specific environment"""
from pathlib import Path
from string import Template

from ase.calculators.cp2k import CP2K
from ase import units
import yaml

_file_dir = Path(__file__).parent / 'files'


def make_calculator(
        method: str,
        multiplicity: int = 0,
        command: str | None = None,
        directory: str = 'run'
) -> CP2K:
    """Make a calculator ready to run with different configurations

    Args:
        method: Which method to run
        multiplicity: Multiplicity of the system
        command: Command used to launch CP2K. Defaults to whatever ASE autodetermines or ``cp2k_shell``
        directory: Path in which to write run file
    Returns:
        Calculator configured for target method
    """

    # Load the presets file
    with (_file_dir / 'presets.yml').open() as fp:
        presets = yaml.safe_load(fp)
    if method not in presets:
        raise ValueError(f'"{method}" not in presets file')
    kwargs = presets[method]
    if kwargs.get('cutoff') is not None:
        kwargs['cutoff'] *= units.Ry
    kwargs.pop('description')

    # Get the input file and replace any templated arguments
    input_file = _file_dir / f'cp2k-{method}-template.inp'
    inp = Template(input_file.read_text()).substitute(mult=multiplicity)

    cp2k_opts = dict(
        xc=None,
        inp=inp,
        poisson_solver=None,
        **kwargs
    )
    if command is not None:
        cp2k_opts['command'] = command
    return CP2K(directory=directory,
                stress_tensor=True,
                potential_file=None,
                set_pos_file=True,
                **cp2k_opts)
