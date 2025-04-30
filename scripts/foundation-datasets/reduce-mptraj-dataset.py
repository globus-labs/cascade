"""Pull relevant training data from MPTraj dataset"""
import zipfile
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
from shutil import copyfileobj
from argparse import ArgumentParser

from ase.db import connect
from tqdm import tqdm
import requests
import ase

from cascade.learning.finetuning import filter_by_elements
from cascade.utils import read_from_string

_url = "https://figshare.com/files/49034296"
_data_path = Path('datasets') / '2024-09-03-mp-trj.extxyz.zip'


def _download_from_figshare():
    """Download the dataset from figshare"""

    reply = requests.get(_url, stream=True)
    _data_path.parent.mkdir(exist_ok=True)
    with _data_path.open('wb') as fo:
        copyfileobj(reply.raw, fo)


def _stream_from_zip(zp: ZipFile) -> Iterable[ase.Atoms]:
    for info in zp.infolist():
        if info.filename.endswith('extxyz'):
            yield from read_from_string(zp.read(info).decode(), 'extxyz', index=':')


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser(description='Gathers all entries from MPTraj which only contain elements from a finetuning set')
    parser.add_argument('finetune_dataset', help='Path to an ASE DB object which contains the finetuning set')
    args = parser.parse_args()

    # Get the list of elements from the finetuning set
    all_elems = set()
    with connect(args.finetune_dataset) as db:
        for row in db.select(''):
            atoms = row.toatoms()
            all_elems.update(atoms.get_chemical_symbols())
        db_size = len(db)
    print(f'Loaded a list of {len(all_elems)} elements from {db_size} records{args.finetune_dataset}.')
    out_path = _data_path.parent / f'mptraj-{"".join(sorted(all_elems))}.db'
    if out_path.exists():
        raise ValueError(f'Dataset already exists at {out_path}')
    print(f'  Writing new frames to {out_path}')

    # Download the data if not yet available
    if not _data_path.is_file():
        _download_from_figshare()

    with connect(out_path) as db, zipfile.ZipFile(_data_path) as zp:
        for atoms in filter_by_elements(tqdm(_stream_from_zip(zp)), all_elems):
            db.write(atoms)
