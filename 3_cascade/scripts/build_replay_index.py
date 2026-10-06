#!/usr/bin/env python3
"""Count the frames in each file of a directory of extxyz files.

Writes ``frame_index.csv`` into the directory, which ``ReplaySampler`` uses to
sample replay frames uniformly. Run once per dataset; the sampler builds the
index itself if it is missing, but that slows down the start of a run.

Example:
    python scripts/build_replay_index.py datasets/mptrj-gga-ggapu --max-workers 8
"""
import argparse
from pathlib import Path

from cascade.learning.finetuning import ReplaySampler, build_frame_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path, help='Directory of extxyz files')
    parser.add_argument('--pattern', default='*.extxyz', help='Glob pattern used to find the files')
    parser.add_argument('--max-workers', type=int, default=8, help='Number of threads used to read files')
    parser.add_argument('--output', type=Path, default=None, help=f'Output path (default: <directory>/{ReplaySampler.index_name})')
    args = parser.parse_args()

    index = build_frame_index(args.directory, pattern=args.pattern, max_workers=args.max_workers)
    output = args.output or args.directory / ReplaySampler.index_name
    index.to_csv(output, index=False)
    print(f'Wrote {output}: {len(index)} files, {index["n_frames"].sum()} frames')
