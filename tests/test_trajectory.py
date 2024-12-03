import ase
from ase.io.trajectory import TrajectoryWriter
import numpy as np

from cascade.trajectory import CascadeTrajectory


def test_trajectory():
    atoms = ase.Atoms('N')
    filename = 'test.traj'
    atoms.set_positions([[0, 0, 0]])
    traj = CascadeTrajectory(path=filename, starting=atoms)
    atoms.set_positions([[0, 0, 1]])
    writer = TrajectoryWriter(traj.path, mode='a')
    writer.write(atoms)
    traj.current_timestep = 1

    # just make sure read works
    traj_read = traj.read()
    assert len(traj_read) == 2
    assert traj_read[1] == atoms

    # make sure we get the correct untrusted segment
    untrusted = traj.get_untrusted_segment()
    assert len(untrusted) == 1
    assert (untrusted[0].positions == np.array([0, 0, 1])).all()

    #  make sure deleting the untrsuted segment works correctly
    traj.trim_untrusted_segment()
    traj_read = traj.read()
    assert len(traj_read) == 1
    assert traj.current_timestep == traj.last_trusted_timestep == 0
