"""Score a fine-tuned MACE checkpoint against the true labels of reference-run frames.

Reports force RMSE against the real labels alongside the "signal floor" -- the RMS of
the true forces, which is the score a model gets by predicting zero everywhere. A model
that has actually learned something lands well below the floor; one trained against
dropped (zeroed) labels sits at or just above it.

    python scripts/check_force_accuracy.py \\
        --weights falsification_labelfix/member0_weights.pt \\
        --run-id reference-2026.08.21-19:31:10-43722e
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from cascade.agents.db_orm import TrajectoryDB
from cascade.learning.mace import MACEInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=str, required=True, action='append',
                        help='Checkpoint to score; repeat to compare several')
    parser.add_argument('--run-id', type=str, required=True)
    parser.add_argument('--traj-id', type=int, default=0)
    parser.add_argument('--n-frames', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--db-url', type=str, default=os.environ.get('CASCADE_DB_URL'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = TrajectoryDB(args.db_url)
    frames = db.get_trajectory_atoms(args.run_id, args.traj_id)[:args.n_frames]
    true = np.concatenate([f.get_forces() for f in frames])
    floor = np.sqrt((true ** 2).mean())

    print(f'{len(frames)} frames from {args.run_id} traj {args.traj_id}')
    print(f'true |F| mean={np.abs(true).mean():.4f}  RMS={floor:.4f} eV/A  <- signal floor\n')

    learner = MACEInterface()
    for path in args.weights:
        calc = learner.make_calculator(open(path, 'rb').read(), args.device)
        pred = []
        for frame in frames:
            copy = frame.copy()
            copy.calc = calc
            pred.append(copy.get_forces())
        pred = np.concatenate(pred)

        rmse = np.sqrt(((pred - true) ** 2).mean())
        verdict = 'PASS' if rmse < 0.5 * floor else 'FAIL'
        print(f'{path}')
        print(f'  predicted |F| mean = {np.abs(pred).mean():.4f} eV/A')
        print(f'  force RMSE vs true = {rmse:.4f} eV/A  ({rmse / floor:.2f} x floor)  [{verdict}]')


if __name__ == '__main__':
    main()
