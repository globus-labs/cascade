#!/usr/bin/env python3
"""Benchmark MACE Trainer batch_size / num_epochs against real labeled data.

Sweeps every combination of candidate batch_size x replay_batch_size (a 1x1
grid when replay is off or only one of each is given). Each candidate trains
an ensemble of --n-ensemble members, routed through cascade.agents.task.train
(the same task cascade.agents.agents.Trainer.train_model submits in
production) via a Parsl executor, bootstrapping each member's training set
the same way Trainer.train_model does.

Records wall-clock time and per-epoch train/valid loss.

Example:
    python scripts/benchmark_trainer.py --run-id my-reference-run \\
        --batch-sizes 4,8,16,32,64 --num-epochs 200 --patience 10 --device cuda \\
        --n-ensemble 4
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import mace_mp
from parsl.concurrent import ParslPoolExecutor
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.usage_tracking.levels import LEVEL_1
from sklearn.model_selection import train_test_split

from cascade.agents.db_orm import TrajectoryDB
from cascade.agents.task import train as training_task
from cascade.learning.finetuning import MultiHeadConfig
from cascade.learning.mace import MACEInterface

SUMMARY_FIELDS = ['batch_size', 'replay_batch_size', 'member_index', 'status', 'epochs_run',
                   'wall_clock_s', 'sec_per_epoch', 'final_valid_loss', 'n_train', 'n_atoms_train']


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
        help='Where summary.csv and per-member epoch CSVs are written',
    )
    parser.add_argument(
        '--n-ensemble',
        type=int,
        default=1,
        help='Ensemble members trained per (batch_size, replay_batch_size) candidate, each on an '
             'independent bootstrap resample -- mirrors the bootstrap/submit/gather loop '
             'cascade.agents.agents.Trainer.train_model uses in production',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Max concurrent Parsl training tasks (defaults to --n-ensemble, i.e. full concurrency, '
             'matching production). Lower this if concurrent ensemble members OOM on a shared GPU.',
    )
    parser.add_argument(
        '--bootstrap-fraction',
        type=float,
        default=1.0,
        help='Fraction of train_data resampled with replacement per ensemble member '
             "(matches TrainerConfig's default of 1.0)",
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
             "candidate, so the total run count is the product of the two lists). Defaults the run's main batch_size.",
    )
    return parser.parse_args()


def _build_replay_variants(args: argparse.Namespace) -> list[tuple[MultiHeadConfig | None, str]]:
    """One (MultiHeadConfig, label) pair per --replay-batch-size candidate.

    Returns [(None, 'none')] if replay isn't enabled
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


def _read_completed_combos(summary_path: Path) -> set[tuple[int, str, int]]:
    """(batch_size, replay_batch_size_label, member_index) triples already recorded.

    Rows from a pre-ensemble summary.csv (no member_index column, e.g. an --out-dir reused
    from before this script trained ensembles) are skipped rather than treated as a completed
    member 0, since they weren't trained on a bootstrap resample and aren't a like-for-like
    match for the new per-member semantics.
    """
    if not summary_path.exists():
        return set()
    completed = set()
    with open(summary_path, newline='') as f:
        for row in csv.DictReader(f):
            if not row.get('member_index'):
                continue
            completed.add((int(row['batch_size']), row['replay_batch_size'], int(row['member_index'])))
    return completed


def _run_ensemble_candidate(pool: ParslPoolExecutor, learner: MACEInterface, init_ensemble_weights: list[bytes],
                             train_data: list, valid_data: list, batch_size: int,
                             replay: MultiHeadConfig | None, replay_label: str, args: argparse.Namespace,
                             out_dir: Path, completed_members: set[int]):
    """Submit one bootstrapped training_task per not-yet-completed ensemble member for this
    (batch_size, replay) candidate -- mirrors Trainer.train_model's bootstrap/submit/gather loop.

    Yields each member's result as soon as it's ready (rather than returning a fully-materialized
    list), so the caller can write summary.csv rows incrementally -- otherwise a crash partway
    through the ensemble would lose the summary row for every already-finished member too, even
    though their epoch CSVs were already safely on disk.
    """
    rng = np.random.default_rng()
    n_sample = int(len(train_data) * args.bootstrap_fraction)
    train_kws = dict(num_epochs=args.num_epochs, batch_size=batch_size,
                      device=args.device, patience=args.patience)

    pending = {}
    for member_index, member_weights in enumerate(init_ensemble_weights):
        if member_index in completed_members:
            continue
        boot_idx = rng.integers(0, len(train_data), size=n_sample)
        boot_data = [train_data[i] for i in boot_idx]
        future = pool.submit(
            training_task,
            learner=learner,
            weights=member_weights,
            train_data=boot_data,
            valid_data=valid_data,
            train_kws=train_kws,
            replay=replay,
        )
        pending[member_index] = {
            'future': future,
            't_submit': time.perf_counter(),
            'n_train': len(boot_data),
            'n_atoms_train': sum(len(a) for a in boot_data),
        }

    for member_index, meta in pending.items():
        yield _collect_member_result(meta, member_index, batch_size, replay_label, out_dir)


def _collect_member_result(meta: dict, member_index: int, batch_size: int, replay_label: str,
                            out_dir: Path) -> dict:
    tag = f'batch_size={batch_size} replay_batch_size={replay_label} member={member_index}'
    status, log = 'ok', None
    try:
        _, log = meta['future'].result()
    except Exception as e:
        if not _is_oom(e):
            raise
        status = 'oom'
        print(f'{tag}: OOM, recording partial result and continuing (other members unaffected)')
    wall_s = time.perf_counter() - meta['t_submit']

    if log is not None and len(log):
        epoch_path = out_dir / f'bs{batch_size}_replay{replay_label}_member{member_index}_epochs.csv'
        log.to_csv(epoch_path, index=False)
        epochs_run = int(log['epoch'].max()) + 1
        last_epoch = log[log['epoch'] == log['epoch'].max()]
        valid_col = 'total_loss_valid' if 'total_loss_valid' in log.columns else None
        final_valid_loss = float(last_epoch[valid_col].mean()) if valid_col else ''
    else:
        epochs_run, final_valid_loss = 0, ''

    return {
        'batch_size': batch_size,
        'replay_batch_size': replay_label,
        'member_index': member_index,
        'status': status,
        'epochs_run': epochs_run,
        'n_train': meta['n_train'],
        'n_atoms_train': meta['n_atoms_train'],
        'wall_clock_s': round(wall_s, 2),
        'sec_per_epoch': round(wall_s / epochs_run, 3) if epochs_run else '',
        'final_valid_loss': final_valid_loss,
    }


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
    init_ensemble_weights = [init_weights] * args.n_ensemble
    replay_variants = _build_replay_variants(args)

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    completed = _read_completed_combos(summary_path)
    max_workers = args.max_workers or args.n_ensemble

    n_total = len(batch_sizes) * len(replay_variants) * args.n_ensemble
    print(f'Sweeping {len(batch_sizes)} batch_size x {len(replay_variants)} replay_batch_size x '
          f'{args.n_ensemble} ensemble members = {n_total} training tasks ({max_workers} concurrent)')

    config = Config(
        executors=[
            HighThroughputExecutor(
                label='htex_local',
                max_workers_per_node=max_workers,
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
        usage_tracking=LEVEL_1,
    )

    write_header = not summary_path.exists()
    with ParslPoolExecutor(config=config) as pool, open(summary_path, 'a', newline='') as sf:
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        if write_header:
            summary_writer.writeheader()
            sf.flush()

        for batch_size in batch_sizes:
            for replay, replay_label in replay_variants:
                completed_members = {
                    m for (bs, rl, m) in completed if bs == batch_size and rl == replay_label
                }
                if len(completed_members) == args.n_ensemble:
                    print(f'batch_size={batch_size} replay_batch_size={replay_label}: all '
                          f'{args.n_ensemble} members already in {summary_path}, skipping')
                    continue

                print(f'batch_size={batch_size} replay_batch_size={replay_label}: starting '
                      f'({args.n_ensemble - len(completed_members)} of {args.n_ensemble} members)')
                results = _run_ensemble_candidate(
                    pool, learner, init_ensemble_weights, train_data, valid_data, batch_size,
                    replay, replay_label, args, out_dir, completed_members,
                )

                for result in results:
                    summary_writer.writerow(result)
                    sf.flush()
                    print(
                        f"batch_size={batch_size} replay_batch_size={replay_label} "
                        f"member={result['member_index']}: status={result['status']} "
                        f"epochs_run={result['epochs_run']} wall_s={result['wall_clock_s']}"
                    )

    print(f'Done. Summary at {summary_path}')


if __name__ == '__main__':
    main()
