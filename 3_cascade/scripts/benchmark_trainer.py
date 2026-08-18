#!/usr/bin/env python3
"""Benchmark MACE Trainer batch_size / num_epochs against real labeled data.

Sweeps every combination of candidate batch_size x replay_batch_size (a 1x1
grid when replay is off or only one of each is given), training with a
generous epoch ceiling and early stopping (--patience), and records wall-clock
time, peak GPU memory, and per-epoch train/valid loss for each combination --
so you can pick the largest batch_size that fits/runs well and read off how
many epochs training actually needs before committing to values for a real run.

Writes live: per-epoch loss is flushed to disk as training proceeds, and the
summary is appended to after each combination, so if the process dies (e.g. an
out-of-memory kill) nothing already recorded is lost. Re-running with the same
--out-dir skips combinations already present in summary.csv.

Example:
    python scripts/benchmark_trainer.py --run-id my-reference-run \\
        --batch-sizes 4,8,16,32,64 --num-epochs 200 --patience 10 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from ase.io import read
from mace.calculators import mace_mp
from sklearn.model_selection import train_test_split

from cascade.agents.db_orm import TrajectoryDB
from cascade.learning.finetuning import MultiHeadConfig
from cascade.learning.mace import MACEInterface

PHASE_MEMORY_FIELDS = [
    'peak_gpu_mb_train', 'peak_gpu_mb_valid', 'peak_gpu_mb_replay',
    'reserved_gpu_mb_train', 'reserved_gpu_mb_valid', 'reserved_gpu_mb_replay',
]
SUMMARY_FIELDS = ['batch_size', 'replay_batch_size', 'status', 'epochs_run', 'wall_clock_s', 'sec_per_epoch',
                   'peak_gpu_mb', 'final_valid_loss', 'n_train', 'n_atoms_train'] + PHASE_MEMORY_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.environ.get('CASCADE_DB_URL'),
        help='Database URL (defaults to the CASCADE_DB_URL env var)',
    )
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='Run to pull labeled frames from (a completed cascade or reference-dynamics run)',
    )
    parser.add_argument(
        '--traj-id',
        type=int,
        default=None,
        help='Restrict to one trajectory (defaults to all trajectories in the run)',
    )
    parser.add_argument(
        '--batch-sizes',
        type=str,
        default='4,8,16,32',
        help='Comma-separated candidate batch sizes to sweep',
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=200,
        help='Epoch ceiling per candidate (--patience decides how many actually run)',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Stop a candidate after this many epochs without validation improvement',
    )
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument(
        '--base-model',
        type=str,
        default='small',
        help='Starting mace_mp model size for the benchmark',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Optional cap on how many labeled frames to use, to keep the benchmark itself fast',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='benchmark_trainer_out',
        help='Where summary.csv and per-batch-size epoch CSVs are written',
    )
    parser.add_argument('--replay-dataset', default=None, help='Path to an ASE database containing data to replay during finetuning')
    parser.add_argument('--replay-downselect', default=None, type=int, help='Max number of entries to use from replay dataset')
    parser.add_argument('--replay-frequency', default=1, type=int, help='How often to replay')
    parser.add_argument('--replay-lr-reduction', default=1, type=float, help='Factor by which to reduce LR during replay')
    parser.add_argument(
        '--replay-batch-size',
        default=None,
        type=str,
        help='Comma-separated batch sizes to sweep during replay (swept against every --batch-sizes '
             'candidate, so the total run count is the product of the two lists). Defaults to a single '
             "candidate that falls back to each run's main batch_size.",
    )
    return parser.parse_args()


def _build_replay_variants(args: argparse.Namespace) -> list[tuple[MultiHeadConfig | None, str]]:
    """One (MultiHeadConfig, label) pair per --replay-batch-size candidate.

    Returns [(None, 'none')] if replay isn't enabled at all -- distinct from a
    MultiHeadConfig with batch_size=None, which means "replay enabled, fall
    back to the main batch_size" (label 'default').
    """
    if args.replay_dataset is None:
        return [(None, 'none')]

    original_dataset = read(args.replay_dataset, slice(None))
    replay_batch_sizes = (
        [int(b) for b in args.replay_batch_size.split(',')] if args.replay_batch_size else [None]
    )
    return [
        (
            MultiHeadConfig(
                original_dataset=original_dataset,
                num_downselect=args.replay_downselect,
                epoch_frequency=args.replay_frequency,
                lr_reduction=args.replay_lr_reduction,
                batch_size=rbs,
            ),
            str(rbs) if rbs is not None else 'default',
        )
        for rbs in replay_batch_sizes
    ]


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, getattr(torch.cuda, 'OutOfMemoryError', ())):
        return True
    return 'out of memory' in str(exc).lower()


def _load_data(db: TrajectoryDB, run_id: str, traj_id: int | None, max_frames: int | None) -> list:
    if traj_id is not None:
        traj_ids = [traj_id]
    else:
        traj_ids = [t['traj_id'] for t in db.list_trajectories_in_run(run_id)]

    atoms = []
    for tid in traj_ids:
        atoms.extend(db.get_trajectory_atoms(run_id, tid))
    if max_frames is not None:
        atoms = atoms[:max_frames]
    return atoms


def _read_completed_combos(summary_path: Path) -> set[tuple[int, str]]:
    if not summary_path.exists():
        return set()
    with open(summary_path, newline='') as f:
        return {(int(row['batch_size']), row['replay_batch_size']) for row in csv.DictReader(f)}


def _run_candidate(learner: MACEInterface, weights: bytes, train_data: list, valid_data: list,
                    batch_size: int, args: argparse.Namespace, epoch_path: Path, is_cuda: bool,
                    replay: MultiHeadConfig | None, replay_label: str) -> dict:
    """Train one (batch_size, replay_batch_size) combination, writing its per-epoch loss live to epoch_path."""
    tag = f'batch_size={batch_size} replay_batch_size={replay_label}'
    epoch_writer = None
    with open(epoch_path, 'w', newline='') as ef:

        def on_epoch(row: dict) -> None:
            nonlocal epoch_writer
            if epoch_writer is None:
                epoch_writer = csv.DictWriter(ef, fieldnames=list(row))
                epoch_writer.writeheader()
            epoch_writer.writerow(row)
            ef.flush()

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
        t0 = time.perf_counter()
        status = 'ok'
        log = None
        try:
            _, log = learner.train(
                weights, train_data, valid_data,
                num_epochs=args.num_epochs, batch_size=batch_size,
                device=args.device, patience=args.patience,
                epoch_callback=on_epoch, replay=replay,
            )
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            status = 'oom'
            print(f'{tag}: OOM, recording partial progress and continuing')
        wall_s = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated(args.device) / 1e6 if is_cuda else 0.0

    with open(epoch_path, newline='') as ef2:
        epoch_rows = list(csv.DictReader(ef2))

    if log is not None and len(log):
        epochs_run = int(log['epoch'].max()) + 1
        last_epoch = log[log['epoch'] == log['epoch'].max()]
        valid_col = 'total_loss_valid' if 'total_loss_valid' in log.columns else None
        final_valid_loss = float(last_epoch[valid_col].mean()) if valid_col else ''
    else:
        # OOM (or an otherwise-empty result): fall back to what the live epoch CSV
        # already captured before the crash, rather than reporting nothing.
        epochs_run, final_valid_loss = 0, ''
        if epoch_rows:
            epochs_run = len(epoch_rows)
            final_valid_loss = epoch_rows[-1].get('total_loss_valid', '')

    result = {
        'batch_size': batch_size,
        'replay_batch_size': replay_label,
        'status': status,
        'epochs_run': epochs_run,
        'n_train': len(train_data),
        'n_atoms_train': sum(len(a) for a in train_data),
        'wall_clock_s': round(wall_s, 2),
        'sec_per_epoch': round(wall_s / epochs_run, 3) if epochs_run else '',
        'peak_gpu_mb': round(peak_mb, 1),
        'final_valid_loss': final_valid_loss,
    }

    # Roll up per-phase peak memory (only present when a CUDA run tracked it) so
    # combos can be compared directly from summary.csv, without opening per-epoch CSVs.
    for col in PHASE_MEMORY_FIELDS:
        values = [float(r[col]) for r in epoch_rows if r.get(col)]
        if values:
            result[col] = round(max(values), 1)

    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'summary.csv'

    db = TrajectoryDB(args.db_url)
    train_data_all = _load_data(db, args.run_id, args.traj_id, args.max_frames)
    if len(train_data_all) < 2:
        print(f'Not enough labeled frames found for run_id={args.run_id} (got {len(train_data_all)})')
        return
    train_data, valid_data = train_test_split(train_data_all, test_size=0.2)
    print(f'Loaded {len(train_data)} train / {len(valid_data)} valid frames from run_id={args.run_id}')

    learner = MACEInterface()
    init_weights = learner.serialize_model(learner.get_model(mace_mp(args.base_model).models[0]))
    replay_variants = _build_replay_variants(args)

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    completed = _read_completed_combos(summary_path)
    is_cuda = torch.cuda.is_available() and 'cuda' in args.device
    print(f'Sweeping {len(batch_sizes)} batch_size x {len(replay_variants)} replay_batch_size = '
          f'{len(batch_sizes) * len(replay_variants)} combinations')

    write_header = not summary_path.exists()
    with open(summary_path, 'a', newline='') as sf:
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        if write_header:
            summary_writer.writeheader()
            sf.flush()

        for batch_size in batch_sizes:
            for replay, replay_label in replay_variants:
                combo_key = (batch_size, replay_label)
                if combo_key in completed:
                    print(f'batch_size={batch_size} replay_batch_size={replay_label}: already in {summary_path}, skipping')
                    continue

                print(f'batch_size={batch_size} replay_batch_size={replay_label}: starting')
                epoch_path = out_dir / f'bs{batch_size}_replay{replay_label}_epochs.csv'
                result = _run_candidate(
                    learner, init_weights, train_data, valid_data, batch_size, args,
                    epoch_path, is_cuda, replay, replay_label,
                )

                summary_writer.writerow(result)
                sf.flush()
                print(
                    f"batch_size={batch_size} replay_batch_size={replay_label}: status={result['status']} "
                    f"epochs_run={result['epochs_run']} wall_s={result['wall_clock_s']} peak_gpu_mb={result['peak_gpu_mb']}"
                )

    print(f'Done. Summary at {summary_path}')


if __name__ == '__main__':
    main()
