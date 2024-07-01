"""Utilities for employing ASE calculators"""
from typing import List
from pathlib import Path
from string import Template
from hashlib import sha256
import numpy as np

from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.calculators.cp2k import CP2K
from ase import units, Atoms

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
        set_pos_file: bool = False,
        debug: bool = False
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
        template_dir: Path to the directory containing templates.
            Default is to use the value of CASCADE_CP2K_TEMPLATE environment variable
            or the template directory provided with cascade if the environment variable
            has not been set.
        set_pos_file: whether cp2k and ase communicate positions via a file on disk
        debug: Whether to run the ase cp2k calculator in debug mode
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
                set_pos_file=set_pos_file,
                debug=debug,
                **cp2k_opts)


class EnsembleCalculator(Calculator):
    """A single calculator which combines the results of many


    Stores the mean of all calculators as the standard property names,
    and stores the values for each calculator in the :attr:`results`
    as the name of the property with "_ens" appended (e.g., "forces_ens")

    Args:
        calculators: the calculators to ensemble over
    """

    def __init__(self,
                 calculators: list[Calculator],
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.calculators = calculators
        self.num_calculators = len(calculators)

    @property
    def implemented_properties(self) -> List[str]:
        joint = set(self.calculators[0].implemented_properties)
        for calc in self.calculators[1:]:
            joint.intersection_update(calc.implemented_properties)

        return list(joint)

    def calculate(self,
                  atoms: Atoms = None,
                  properties=all_properties,
                  system_changes=all_changes):
        # Run each of the subcalculators
        for calc in self.calculators:
            calc.calculate(atoms, properties=properties, system_changes=system_changes)

        # Determine the intersection of the properties computed from all calculators
        all_results = set(self.calculators[0].results.keys())
        for calc in self.calculators[1:]:
            all_results.intersection_update(calc.results.keys())

        # Merge their results
        results = {}
        for key in all_results:
            all_results = np.concatenate([np.expand_dims(calc.results[key], axis=0) for calc in self.calculators], axis=0)
            results[f'{key}_ens'] = all_results
            results[key] = all_results.mean(axis=0)

        self.results = results
