from pytest import fixture
from ase import Atoms
from ase.build import molecule
import numpy as np

from cascade.agents.task import label_frame
from cascade.model import TrainingFrame


@fixture
def calc_factory():
    from mace.calculators import mace_mp
    def _factory():
        return mace_mp('small', device='cpu', default_dtype='float32')
    return _factory


@fixture
def water_frame() -> TrainingFrame:
    water = molecule('H2O')
    water.cell = [4.] * 3
    water.pbc = True
    return TrainingFrame(
        atoms=water,
        model_version=0,
        traj_id=0,
        chunk_id=0,
        attempt_index=0,
        frame_id=0,
        n_sampled_frames=1,
    )


def test_label_frame(water_frame, calc_factory):
    result = label_frame(water_frame, calc_factory)

    assert result.atoms_labeled is not None
    assert result.atoms_labeled is not water_frame.atoms  # copy, not the original

    energy = result.atoms_labeled.info["energy"]
    forces = result.atoms_labeled.arrays["forces"]

    assert np.isfinite(energy)
    assert forces.shape == (len(water_frame.atoms), 3)
    assert np.isfinite(forces).all()
