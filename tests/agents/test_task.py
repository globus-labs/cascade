from pytest import fixture
from ase import Atoms
from ase.build import molecule
import numpy as np

from cascade.agents.task import (
    label_frame,
    max_uq_sample,
    boundary_uq_sample,
    audit_reason_sample,
)
from cascade.model import Chunk, TrainingFrame


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


def uq_chunk(values: list[float]) -> Chunk:
    """A Chunk of single-atom placeholders, one per UQ value in `values`."""
    atoms = []
    for v in values:
        a = Atoms('H')
        a.info['uq_force_std_max'] = v
        atoms.append(a)
    return Chunk(
        atoms=atoms,
        frame_ids=list(range(len(atoms))),
        traj_id=0,
        chunk_id=0,
        attempt_ix=0,
        model_version=1,
    )


def test_max_uq_sample_picks_highest_values():
    chunk = uq_chunk([0.01, 0.02, 0.03, 0.5, 0.6, 0.04, 0.02])

    frames = max_uq_sample(chunk, 2)

    assert sorted(f.frame_id for f in frames) == [3, 4]


def test_boundary_uq_sample_anchors_on_first_crossing_not_global_max():
    # global max is at index 4 (0.6); first crossing of threshold=0.1 is index 3
    chunk = uq_chunk([0.01, 0.02, 0.03, 0.5, 0.6, 0.04, 0.02])

    frames = boundary_uq_sample(chunk, 3, threshold=0.1)

    assert sorted(f.frame_id for f in frames) == [2, 3, 4]  # centered on 3, not 4


def test_boundary_uq_sample_falls_back_when_nothing_crosses():
    chunk = uq_chunk([0.01, 0.02, 0.03, 0.04, 0.02])

    frames = boundary_uq_sample(chunk, 2, threshold=0.5)

    assert sorted(f.frame_id for f in frames) == [2, 3]  # same as max_uq_sample(chunk, 2)


def test_audit_reason_sample_routes_by_reason():
    chunk = uq_chunk([0.01, 0.02])
    calls = []

    def marker(name):
        def _sampler(chunk, n_frames, **kws):
            calls.append(name)
            return []
        return _sampler

    kwargs = dict(
        burn_in_sampler=marker('burn_in'),
        threshold_sampler=marker('threshold'),
        random_sampler=marker('random_fail'),
        default_sampler=marker('default'),
    )

    audit_reason_sample(chunk, 1, reason='burn_in', **kwargs)
    audit_reason_sample(chunk, 1, reason='threshold', **kwargs)
    audit_reason_sample(chunk, 1, reason='random_fail', **kwargs)
    audit_reason_sample(chunk, 1, reason='some_future_reason', **kwargs)
    audit_reason_sample(chunk, 1, reason=None, **kwargs)

    assert calls == ['burn_in', 'threshold', 'random_fail', 'default', 'default']
