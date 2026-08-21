"""Utilities for employing ASE calculators"""
from typing import Callable, List
from functools import partial
from pathlib import Path
from string import Template
from hashlib import sha256
import numpy as np
import yaml
import os

from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.calculators.cp2k import CP2K
from ase import units, Atoms

_file_dir = Path(__file__).parent / 'files'


def get_calc_factory(
        calc_type: str,
        calc_model: str,
        device: str,
        calc_task: str | None = None,
) -> Callable[[], Calculator]:
    """Build a zero-argument factory for a reference MLFF calculator

    Imports for each ``calc_type`` are deferred until that branch runs, so
    selecting one calculator type does not require the packages for the others
    to be installed.

    Args:
        calc_type: Which calculator family to use (``mace`` or ``fairchem``)
        calc_model: For ``mace``, a MACE-MP model size (e.g. ``medium``) or path
            to a MACE checkpoint. For ``fairchem``, the path to a FairChem
            ``.pt`` checkpoint.
        device: Device to run the calculator on (e.g. ``cpu``, ``cuda``)
        calc_task: FairChem task name selecting the model head (e.g. ``omol``,
            ``omat``, ``oc20``, ``odac``, ``omc``), ignored for ``mace``. Only
            required for ``fairchem`` if the checkpoint supports more than one
            task; single-task checkpoints infer it automatically.
    Returns:
        A zero-argument callable that returns a new Calculator instance
    """
    # torch>=2.6 defaults to weights_only=True, which rejects both MACE's and FairChem's
    # checkpoints. See https://github.com/ACEsuit/mace/issues/555.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

    if calc_type == 'mace':
        from mace.calculators import mace_mp
        return partial(mace_mp, model=calc_model, device=device, default_dtype='float32')
    elif calc_type == 'fairchem':
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

        def build_fairchem_calc():
            # FAIRChemCalculator only accepts bare "cpu"/"cuda" (unlike mace_mp), and
            # resolves "cuda" via torch.cuda.current_device() - so an indexed device
            # (e.g. "cuda:1") has to select the device explicitly first, or it'd silently
            # fall back to whichever device is current (usually index 0).
            if device.startswith('cuda:'):
                import torch
                torch.cuda.set_device(int(device.split(':', 1)[1]))
            return FAIRChemCalculator.from_model_checkpoint(
                name_or_path=calc_model,
                task_name=calc_task,
                device='cuda' if device.startswith('cuda') else device,
            )
        return build_fairchem_calc
    else:
        raise ValueError(f'Unknown calc_type: {calc_type!r}')


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
