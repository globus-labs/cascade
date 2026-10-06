from ase.io import write
from pytest import fixture, raises

from cascade.learning.finetuning import ReplaySampler, build_frame_index, count_frames


@fixture()
def replay_dir(tmp_path, example_data):
    """Directory of extxyz files holding 1, 2, and 3 frames"""
    for i in range(3):
        frames = [example_data[j % 2] for j in range(i + 1)]
        write(tmp_path / f'mp-{i}.extxyz', frames)
    return tmp_path


def test_count_frames(replay_dir):
    assert [count_frames(replay_dir / f'mp-{i}.extxyz') for i in range(3)] == [1, 2, 3]

    index = build_frame_index(replay_dir, max_workers=2)
    assert index['file'].tolist() == ['mp-0.extxyz', 'mp-1.extxyz', 'mp-2.extxyz']
    assert index['n_frames'].tolist() == [1, 2, 3]


def test_sample_directory(replay_dir):
    sampler = ReplaySampler(replay_dir, num_samples=4)
    assert (replay_dir / ReplaySampler.index_name).is_file()
    assert len(sampler) == 6

    samples = sampler.sample(1)
    assert len(samples) == 4
    for atoms in samples:
        assert atoms.get_potential_energy() in (3., 4.)
        assert atoms.get_forces().shape == (2, 3)

    # Asking for every frame returns each one exactly once
    sampler.num_samples = 6
    energies = sorted(a.get_potential_energy() for a in sampler.sample())
    assert energies == [3., 3., 3., 3., 4., 4.]

    # Index is reused on the next load
    (replay_dir / 'mp-2.extxyz').unlink()
    assert len(ReplaySampler(replay_dir, num_samples=1)) == 6


def test_sample_directory_requires_num_samples(replay_dir):
    with raises(ValueError):
        ReplaySampler(replay_dir, num_samples=None)


def test_sample_file(tmp_path, example_data):
    path = tmp_path / 'replay.extxyz'
    write(path, example_data * 3)

    assert len(ReplaySampler(path, None).sample()) == 6
    assert len(ReplaySampler(path, 2).sample()) == 2
