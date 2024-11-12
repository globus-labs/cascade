from io import StringIO
from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm
import ase
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import get_distances
from joblib import Parallel, delayed


# Taken from ExaMol
# Taken from jitterbug
def write_to_string(atoms: Atoms, fmt: str, **kwargs) -> str:
    """Write an ASE atoms object to string

    Args:
        atoms: Structure to write
        fmt: Target format
        kwargs: Passed to the write function
    Returns:
        Structure written in target format
    """

    out = StringIO()
    atoms.write(out, fmt, **kwargs)
    return out.getvalue()


def read_from_string(atoms_msg: str, fmt: str) -> Atoms:
    """Read an ASE atoms object from a string

    Args:
        atoms_msg: String format of the object to read
        fmt: Format (cannot be autodetected)
    Returns:
        Parsed atoms object
    """

    out = StringIO(str(atoms_msg))  # str() ensures that Proxies are resolved
    return ase.io.read(out, format=fmt)


def canonicalize(atoms: Atoms) -> Atoms:
    """A convenience function to standardize the format of an ase.Atoms object

    The main motivation is to freeze the Atoms.calc attribute into an immutable
    singlepoint calculator

    Args:
        atoms: Structure to write
    Returns:
        Atoms object that has been serialized and deserialized
    """
    # TODO (wardlt): Make it so the single-point calculator can hold unknown properties? (see discussion https://gitlab.com/ase/ase/-/issues/782)
    out_atoms = atoms.copy()
    if atoms.calc is not None:
        old_calc = atoms.calc
        out_atoms.calc = SinglePointCalculator(atoms)
        out_atoms.calc.results = old_calc.results.copy()
    
        for k, v in out_atoms.calc.results.items(): 
            if isinstance(v, np.ndarray): 
                out_atoms.calc.results[k] = v.astype(np.float64)

    return out_atoms


def apply_calculator(
        calc: Calculator,
        traj: list[Atoms]) -> list[Atoms]:
    """Run a calculator on every atoms object in a list, returning a new list

    Args:
        calc: the calculator to be applied
        traj: the list of atoms to have the calculator applied to
    Returns:
        list of atoms that have been canonicalized s.t. their results are stored in a SinglePointCalculator
    """
    traj = deepcopy(traj)
    for i, atoms in enumerate(traj):
        atoms.calc = calc
        atoms.get_forces()
        traj[i] = canonicalize(atoms)
    return traj


def to_voigt(stress: np.ndarray) -> np.ndarray:
    """Converts a (3,3) stress tensor to voigt form, or do nothing if its already in this form

    Args:
        stress: a stess tensor of shape (3,3) or (6,), in which case this function is a no-op
    Returns:
        stress: a (6,) stress tensor in voigt form
    """
    if stress.shape == (3, 3):
        stress = np.array([stress[0, 0], stress[1, 1], stress[2, 2],
                           stress[1, 2], stress[0, 2], stress[0, 1]])
    elif stress.shape != (6, ):
        raise ValueError(f"Stress tensor must either be of shape (3,3) or (6,), got {stress.shape}")
    return stress


def unwrap_trajectory(traj: list[ase.Atoms]) -> np.ndarray:
    """Unwraps a trajectory from periodic boundary conditions
    
    Does so by incrementing finding the minimum image displacement vector between 
    time-adjacent frames and using this displacement to advance positions over time

    Args: 
        traj: a dynamics trajectory of length n_frames run with PBC
    Returns: 
        a numpy array (n_frames, n_atoms, 3) of frames unwrapped from PBC
    """

    out = []
    
    # save initial positions 
    r = traj[0].positions.copy()
    out.append(r)
    
    # iterate over time-adjacent frames
    for i in range(1, len(traj)):

        # (n, n, 3) - dr is the displacement *tensor* 
        # between every pair of atoms across the frames
        dr, _ = get_distances(traj[i-1].positions, 
                              traj[i].positions, 
                              cell=traj[i].cell,  # this is scary what if the cell is changing
                              pbc=True)
        
        # (n, 3) - just take the displacement of atom_i across time
        dr = np.diagonal(dr).T
        
        # update and save positions
        r += dr
        out.append(r.copy())
    
    return np.asarray(out)


def calculate_sqared_disp(traj: np.ndarray, 
                          n_jobs: int = None, 
                          verbose: int = 1,
                          subtract_com_shift: bool = False) -> np.ndarray:
    """Calculate the MSD time correlation function
    
    Calculates the MSD for every length of time accessible in the trajectory, 
    using every eligable timestep pair for that length of time

    Args: 
        traj: a numpy array of positions (n_frames, n_atoms, 3)
              **!important:** make sure these positions are unwrapped from PBC
        n_jobs: passed to joblib.Parallel
        verbose: passed to joblib.Parallel
    Returns: 
        a numpy array of mean squared displacement (n_frames, )
    """
    n_frames, n_atoms, _ = traj.shape
    step_sizes = np.arange(1, n_frames)

    def calc_for_step_size(step_size):
        # this is the number of steps of size step_size in our simulation
        n_steps = n_frames-step_size

        # we'll just store every delta squared
        # this is inneficient, but I know its correct
        disp_sq = np.zeros((n_atoms, n_steps), float)
        # for every possible starting position
        for start_ix in np.arange(n_steps):
            # this is the corresponding stop position
            stop_ix = start_ix + step_size
            # stop - start
            dr = traj[stop_ix, :] - traj[start_ix, :]
            if subtract_com_shift: 
                delta_com = traj[stop_ix, :].mean(0) - traj[start_ix, :].mean(0)
                dr -= delta_com
            disp_sq[:, start_ix] = (dr * dr).sum(1)
        # subtract 1 since its zero indexed
        return disp_sq.mean()
    
    f = delayed(calc_for_step_size)
    p = Parallel(n_jobs=n_jobs, verbose=verbose)
    out = p(f(s) for s in step_sizes)
    return np.asarray(out)


def set_volume(frame: ase.Atoms, vol: float) -> None:
    """Set the volume of a cell and move the atoms accordingly"""
    scalar = (vol / frame.get_volume())**(1/3)
    frame.set_cell(scalar*frame.cell, scale_atoms=True)
