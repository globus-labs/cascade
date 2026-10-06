"""Utilities and data models used when finetuning a model"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Collection, Iterable, Any
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

logger = logging.getLogger(__name__)


def filter_by_elements(atoms_gen: Iterable[Atoms], allowed_elems: Collection[str]) -> Iterable[Atoms]:
    """Process a stream of entries to only include those with allowed elements

    Args:
        atoms_gen: Stream of Atoms structures to be filtered
        allowed_elems: List of elements which are allowed in the dataset
    Yields:
        Atoms from the stream which contain only the desired elements
    """

    allowed_elems = set(allowed_elems)
    for atoms in atoms_gen:
        elems = set(atoms.get_chemical_symbols())
        if any(e not in allowed_elems for e in elems):
            continue
        yield atoms


# TODO (wardlt): Build towards more advanced methods, like https://arxiv.org/abs/2404.12526v1
@dataclass
class MultiHeadConfig:
    """Configuration used to define replay training"""

    # Defining the training data
    original_dataset: list[Atoms] = ...
    """Path to dataset containing the original training samples

    Must be in a form readable by ASE.
    """
    num_downselect: int | None = None
    """Number of points from the dataset to use for training each training round"""

    # Defining the training procedure
    epoch_frequency: int = 1
    """How often to retrain using the original dataset"""
    lr_reduction: float = 1
    """Factor by which to reduce the learning rate during replay"""
    batch_size: int | None = None
    """Batch size to use during replay"""

    learner_options: dict[str, Any] = field(default_factory=dict)
    """Options specific to a certain learner"""


def count_frames(path: str | Path) -> int:
    """Count the frames in an extxyz file without parsing them

    Args:
        path: Path to the extxyz file
    Returns:
        Number of frames in the file
    """
    n_frames = 0
    with open(path) as fp:
        for line in fp:
            if not line.strip():
                continue
            # Skip the comment line and one line per atom
            for _ in range(int(line) + 1):
                next(fp)
            n_frames += 1
    return n_frames


def build_frame_index(directory: str | Path, pattern: str = '*.extxyz', max_workers: int = 8) -> pd.DataFrame:
    """Count the frames in each file of a directory of extxyz files

    Args:
        directory: Directory holding the extxyz files
        pattern: Glob pattern used to find the files
        max_workers: Number of threads used to read files
    Returns:
        Dataframe with the file name (relative to ``directory``) and the number of frames in it
    """
    files = sorted(p.name for p in Path(directory).glob(pattern))
    with ThreadPoolExecutor(max_workers) as pool:
        counts = list(pool.map(count_frames, (Path(directory) / f for f in files)))
    return pd.DataFrame({'file': files, 'n_frames': counts})


class ReplaySampler:
    """Draw random subsets of frames from a replay dataset

    The dataset is either a single file readable by ASE, which is held in memory,
    or a directory of extxyz files, from which only the sampled frames are read.
    Frames are sampled uniformly and without replacement.
    A directory requires an index of the number of frames per file,
    which is read from ``index_path`` (default: ``frame_index.csv`` in the directory)
    or built and saved there if it does not exist.

    Args:
        path: Path to a file or directory of extxyz files
        num_samples: Number of frames per sample. ``None`` to return the whole dataset (file only)
        index_path: Path to the frame index for a directory
        max_workers: Number of threads used to read files
    """

    index_name = 'frame_index.csv'

    def __init__(self, path: str | Path, num_samples: int | None, index_path: str | Path | None = None, max_workers: int = 8):
        self.path = Path(path)
        self.num_samples = num_samples
        self.max_workers = max_workers

        if self.path.is_dir():
            if num_samples is None:
                raise ValueError('num_samples is required when replaying from a directory')
            index_path = Path(index_path) if index_path is not None else self.path / self.index_name
            if index_path.is_file():
                index = pd.read_csv(index_path)
            else:
                logger.warning(f'No frame index found at {index_path}. Building one now, which may take several minutes')
                index = build_frame_index(self.path, max_workers=max_workers)
                index.to_csv(index_path, index=False)
            index = index[index['n_frames'] > 0]
            self.files: list[str] | None = index['file'].tolist()
            self.offsets: np.ndarray | None = np.cumsum(index['n_frames'].values)
            self.dataset: list[Atoms] | None = None
        else:
            self.files = self.offsets = None
            self.dataset = read(self.path, ':')

    def __len__(self) -> int:
        """Total number of frames in the dataset"""
        return len(self.dataset) if self.dataset is not None else int(self.offsets[-1])

    def sample(self, rng: np.random.Generator | None = None) -> list[Atoms]:
        """Draw a random subset of frames

        Args:
            rng: Random number generator
        Returns:
            Sampled frames
        """
        rng = np.random.default_rng(rng)
        if self.num_samples is None or self.num_samples >= len(self):
            if self.dataset is not None:
                return list(self.dataset)
            frame_ids = np.arange(len(self))
        else:
            frame_ids = rng.choice(len(self), size=self.num_samples, replace=False)

        if self.dataset is not None:
            return [self.dataset[i] for i in frame_ids]

        # Map global frame ids to (file, frame in file), and read each file once
        file_ids = np.searchsorted(self.offsets, frame_ids, side='right')
        starts = np.concatenate([[0], self.offsets[:-1]])
        local_ids = frame_ids - starts[file_ids]
        to_read: dict[int, list[int]] = {}
        for f, i in zip(file_ids.tolist(), local_ids.tolist()):
            to_read.setdefault(f, []).append(i)

        def _read(item: tuple[int, list[int]]) -> list[Atoms]:
            f, ids = item
            frames = read(self.path / self.files[f], ':')
            return [frames[i] for i in ids]

        with ThreadPoolExecutor(self.max_workers) as pool:
            return [a for chunk in pool.map(_read, to_read.items()) for a in chunk]
