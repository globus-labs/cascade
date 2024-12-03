import ase
from ase.io import read, write


class CascadeTrajectory:
    """A class to encasplulate a cascade trajectory

    This is useful for reading and auditing trajectories
    so we know where to start sampling from (e.g., after the last trusted timestep)
    """
    def __init__(self,
                 path: str,
                 starting: ase.Atoms = None):
        self.path = path
        self.starting = starting
        if self.starting is not None:
            write(self.path, self.starting)
        else:
            self.starting = read(self.path)

        self.current = self.trusted = starting
        self.current_timestep = 0
        self.last_trusted_timestep = 0

    def read(self, index=':', *args, **kwargs) -> list[ase.Atoms]:
        """Read the trajectory into an iterable of atoms"""
        return read(self.path, *args, index=index, **kwargs)

    def get_untrusted_segment(self) -> list[ase.Atoms]:
        """Return the part of the trajectory that needs to be audited"""
        return read(self.path, index=f'{self.last_trusted_timestep+1}:')

    def trim_untrusted_segment(self):
        """Remove the part of a trajectory that failed an audit, updating timesteps as appropriate"""
        # todo: is there a way to do this without loading into memory?
        write(self.path, read(self.path, index=f':{self.last_trusted_timestep+1}'))
        self.current_timestep = self.last_trusted_timestep

    def __repr__(self):
        return f"CascadeTrajectory(path={self.path}, current_timestep={self.current_timestep}, last_trusted_timestep={self.last_trusted_timestep})"
