from io import StringIO
import ase
from ase import Atoms

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

def canonicalize(atoms: Atoms, fmt: str = 'extxyz') -> Atoms:
    """A convenience function to standardize the format of an ase.Atoms object

    The main motiviation is to freeze the Atoms.calc attribute into an immutable
    singlepoint calculator
    
    Args:
        atoms: Structure to write
        fmt: the ase.io format to write to and read from
    Returns: 
        Atoms object that has been serialied and deserialized
    
    """
    return read_from_string(write_to_string(atoms, fmt), fmt)