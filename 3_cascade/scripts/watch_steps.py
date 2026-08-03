#!/usr/bin/env python3
"""Print per-trajectory step counts from the TrajectoryDB, for use with `watch`.

Example:
    watch -n 2 python scripts/watch_steps.py
"""
from __future__ import annotations

import argparse

from sqlalchemy import create_engine, text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db-url',
        type=str,
        default='postgresql://ase:pw@localhost:5432/cascade',
        help='Database URL',
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run to report on (defaults to whichever run_id most recently wrote a frame)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = create_engine(args.db_url)
    with engine.connect() as conn:
        run_id = args.run_id
        if run_id is None:
            run_id = conn.execute(text(
                'SELECT run_id FROM trajectory_frames ORDER BY created_at DESC LIMIT 1'
            )).scalar()

        rows = conn.execute(text('''
            SELECT traj_id, chunk_id, count(*) AS n_steps, max(frame_index) AS max_frame_index
            FROM trajectory_frames
            WHERE run_id = :run_id
            GROUP BY traj_id, chunk_id
            ORDER BY traj_id, chunk_id
        '''), {'run_id': run_id}).fetchall()

    print(f'run_id = {run_id}')
    print(f"{'traj_id':>8} {'chunk_id':>9} {'n_steps':>8} {'max_frame_index':>16}")
    for r in rows:
        print(f'{r.traj_id:>8} {r.chunk_id:>9} {r.n_steps:>8} {r.max_frame_index:>16}')


if __name__ == '__main__':
    main()
