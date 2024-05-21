"""Make a CP2K calculator ready to run a specific environment"""
from pathlib import Path
from string import Template
from hashlib import sha256
import os

from ase.calculators.cp2k import CP2K
from ase import units, Atoms
import yaml

_file_dir = Path(__file__).parent / 'files'


def create_run_hash(atoms: Atoms, **kwargs) -> str:
    """Generate a unique has for a certain simulation

    Args:
        atoms: Atoms describing the start of the dynamics
        kwargs: Any other keyword arguments used to describe the run
    Returns:
        A hash describing the run
    """

    # Update using the structure
    hasher = sha256()
    hasher.update(atoms.get_atomic_numbers().tobytes())
    hasher.update(atoms.positions.tobytes())

    # Add the optional arguments
    options = sorted(kwargs.items())
    for key, value in options:
        hasher.update(key.encode())
        hasher.update(str(value).encode())

    return hasher.hexdigest()[-8:]


def make_calculator(
        method: str,
        multiplicity: int = 0,
        command: str | None = None,
        directory: str = 'run',
        template_dir: str | Path | None = None,
        set_pos_file: bool = True,
        timeout: float | None = None,
        debug: bool = False
) -> CP2K:
    """Make a calculator ready to run with different configurations

    Args:
        method: Which method to run
        multiplicity: Multiplicity of the system
        command: Command used to launch CP2K. Defaults to whatever ASE autodetermines or ``cp2k_shell``
        directory: Path in which to write run file
        template_dir: Path to the directory containing templates.
            Default is to use the value of CASCADE_CP2K_TEMPLATE environment variable
            or the template directory provided with cascade if the environment variable
            has not been set.
        set_pos_file: whether cp2k and ase communicate positions via a file on disk
        timeout: ase/cp2k_shell communication timeout (in seconds)
        debug: wether to run the ase cp2k calculator in debug mode
    Returns:
        Calculator configured for target method
    """

    # Default to the environment variable
    if template_dir is None:
        template_dir = Path(os.environ.get('CASCADE_CP2K_TEMPLATE', _file_dir))

    # Load the presets file
    with (template_dir / 'presets.yml').open() as fp:
        presets = yaml.safe_load(fp)
    if method not in presets:
        raise ValueError(f'"{method}" not in presets file')
    kwargs = presets[method]
    if kwargs.get('cutoff') is not None:
        kwargs['cutoff'] *= units.Ry
    kwargs.pop('description')

    # Get the input file and replace any templated arguments
    input_file = template_dir / f'cp2k-{method}-template.inp'
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
                set_pos_file=set_pos_file,
                timeout=timeout,
                debug=debug,
                **cp2k_opts)
