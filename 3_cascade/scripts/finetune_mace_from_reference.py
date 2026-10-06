#!/usr/bin/env python3
"""Fine-tune a mace_mp model on frames from the most recent reference-dynamics run.

Pulls trajectories written by run_reference_dynamics.py (run_id prefix "reference-"),
takes the first --first-n-frames of each of --k-trajectories trajectories, optionally
random-samples --sample-m-frames from each trajectory's pool independently, and
fine-tunes with multi-head replay (--replay-dataset). With --n-ensemble > 1, trains a
bootstrap ensemble the same way cascade.agents.agents.Trainer.train_model does, one
checkpoint per member.

Weights are written as learner.serialize_model bytes, loadable directly via
run_cascade_academy.py's --init-weights-paths.

Example:
    python scripts/finetune_mace_from_reference.py \\
        --k-trajectories 3 --first-n-frames 5000 --sample-m-frames 500 \\
        --n-ensemble 4 --num-epochs 200 --patience 15 --device cuda \\
        --replay-dataset ./datasets/mace-mp/sampled_1000.traj
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import mace_mp

from cascade.agents.db_orm import TrajectoryDB
from cascade.agents.task import train as training_task
from cascade.learning.finetuning import MultiHeadConfig
from cascade.learning.mace import MACEInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', type=str, default=os.environ.get('CASCADE_DB_URL'))
    parser.add_argument(
        '--run-id', type=str, default=None,
        help='Reference-dynamics run to pull frames from. Defaults to the most recent '
             'run_id with the "reference-" prefix.'
    )
    parser.add_argument(
        '--traj-ids', type=str, default=None,
        help='Comma-separated trajectory IDs to use. Overrides --k-trajectories.'
    )
    parser.add_argument(
        '--k-trajectories', type=int, default=None,
        help='Use the first K trajectories (by traj_id) in the run. Default: all.'
    )
    parser.add_argument(
        '--first-n-frames', type=int, default=None,
        help='Take only the first N frames of each selected trajectory. Default: all frames.'
    )
    parser.add_argument(
        '--sample-m-frames', type=int, default=None,
        help='Randomly sample M frames from each trajectory\'s first-N-frames pool. '
             'Default: use all of them.'
    )
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--base-model', type=str, default='small')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument(
        '--n-ensemble', type=int, default=1,
        help='Number of bootstrap ensemble members to train.'
    )
    parser.add_argument(
        '--bootstrap-fraction', type=float, default=1.0,
        help='Fraction of train_data resampled with replacement per member.'
    )
    parser.add_argument('--replay-dataset', default=None, help='ASE-readable dataset to replay during finetuning')
    parser.add_argument('--replay-downselect', default=None, type=int)
    parser.add_argument('--replay-frequency', default=1, type=int)
    parser.add_argument('--replay-lr-reduction', default=1, type=float)
    parser.add_argument('--replay-batch-size', default=None, type=int)
    parser.add_argument('--out-dir', type=str, default='finetuned_weights')
    return parser.parse_args()


def resolve_reference_run_id(db: TrajectoryDB) -> str:
    for run in db.list_runs():
        if run['run_id'].startswith('reference-'):
            return run['run_id']
    raise SystemExit('No reference-dynamics runs found (run_id prefix "reference-")')


def resolve_traj_ids(db: TrajectoryDB, run_id: str, args: argparse.Namespace) -> list[int]:
    if args.traj_ids is not None:
        return [int(t) for t in args.traj_ids.split(',')]
    all_ids = sorted(t['traj_id'] for t in db.list_trajectories_in_run(run_id))
    if args.k_trajectories is not None:
        return all_ids[:args.k_trajectories]
    return all_ids


def load_candidate_frames(
    db: TrajectoryDB, run_id: str, traj_ids: list[int], first_n: int | None,
    sample_m: int | None, rng: np.random.Generator,
) -> list:
    """First N frames of each trajectory, each independently sampled down to M frames."""
    frames = []
    for tid in traj_ids:
        atoms = db.get_trajectory_atoms(run_id, tid)
        pool = atoms[:first_n] if first_n is not None else atoms
        if sample_m is not None:
            if sample_m > len(pool):
                print(f'traj {tid}: --sample-m-frames={sample_m} exceeds {len(pool)} candidates; using all')
            else:
                idx = rng.choice(len(pool), size=sample_m, replace=False)
                pool = [pool[i] for i in idx]
        frames.extend(pool)
    return frames


def build_replay(args: argparse.Namespace) -> MultiHeadConfig | None:
    if args.replay_dataset is None:
        return None
    return MultiHeadConfig(
        original_dataset=read(args.replay_dataset, slice(None)),
        num_downselect=args.replay_downselect,
        epoch_frequency=args.replay_frequency,
        lr_reduction=args.replay_lr_reduction,
        batch_size=args.replay_batch_size,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = TrajectoryDB(args.db_url)
    run_id = args.run_id or resolve_reference_run_id(db)
    traj_ids = resolve_traj_ids(db, run_id, args)
    print(f'Using run_id={run_id}, trajectories={traj_ids}')

    rng = np.random.default_rng(args.seed)
    candidates = load_candidate_frames(db, run_id, traj_ids, args.first_n_frames, args.sample_m_frames, rng)
    print(f'{len(candidates)} frames selected for training')

    n_val = max(1, int(len(candidates) * args.val_fraction))
    perm = rng.permutation(len(candidates))
    valid_data = [candidates[i] for i in perm[:n_val]]
    train_data = [candidates[i] for i in perm[n_val:]]
    print(f'{len(train_data)} train / {len(valid_data)} valid frames')

    learner = MACEInterface()
    init_weights = learner.serialize_model(learner.get_model(mace_mp(args.base_model).models[0]))
    replay = build_replay(args)
    train_kws = dict(num_epochs=args.num_epochs, batch_size=args.batch_size,
                      device=args.device, patience=args.patience)

    n_sample = int(len(train_data) * args.bootstrap_fraction)
    weight_paths = []
    for member_index in range(args.n_ensemble):
        boot_idx = rng.integers(0, len(train_data), size=n_sample)
        boot_data = [train_data[i] for i in boot_idx]
        print(f'Training member {member_index} on {len(boot_data)} bootstrapped frames')
        weights, log = training_task(
            learner=learner, weights=init_weights, train_data=boot_data,
            valid_data=valid_data, train_kws=train_kws, replay=replay,
        )
        weights_path = out_dir / f'member{member_index}_weights.pt'
        weights_path.write_bytes(weights)
        log.to_csv(out_dir / f'member{member_index}_log.csv', index=False)
        weight_paths.append(str(weights_path.resolve()))
        print(f'Member {member_index}: wrote {weights_path}')

    manifest_path = out_dir / 'weights_manifest.txt'
    manifest_path.write_text('\n'.join(weight_paths) + '\n')
    print(f'Wrote {manifest_path}')
    print('--init-weights-paths ' + ','.join(weight_paths))


if __name__ == '__main__':
    main()
