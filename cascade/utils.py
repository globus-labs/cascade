from io import StringIO
from copy import deepcopy

import ase
from ase import Atoms
from ase.calculators.calculator import Calculator


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
    fmt = 'extxyz'  # the ase.io format to write to and read from
    return read_from_string(write_to_string(atoms, fmt), fmt)


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
