#!/usr/bin/env python3
"""Print a live per-trajectory pipeline status for a cascade run, for use with `watch`.

Composes existing TrajectoryDB read methods (no new queries beyond what's
already exposed) into one view: which chunk/attempt/model_version each
trajectory is on, its lifecycle status, where it currently sits in the
dynamics/audit/sampling pipeline, plus run-level training/threshold state.

Example:
    watch -n 2 python scripts/watch_run.py
"""
from __future__ import annotations

import argparse
import os

from sqlalchemy import create_engine, text

from cascade.agents.db_orm import TrajectoryDB
from cascade.model import AuditStatus, ChunkEventType, TrajectoryStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.environ.get('CASCADE_DB_URL'),
        help='Database URL, e.g. postgresql://ase:pw@<host>:5432/cascade '
             '(defaults to the CASCADE_DB_URL env var)',
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run to report on (defaults to whichever run_id most recently wrote a frame)',
    )
    return parser.parse_args()


def resolve_run_id(db_url: str) -> str | None:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        return conn.execute(text(
            'SELECT run_id FROM trajectory_frames ORDER BY created_at DESC LIMIT 1'
        )).scalar()


def pipeline_stage(traj_status: TrajectoryStatus, latest_event: dict | None) -> str:
    """Best-effort label for where a trajectory currently sits in the pipeline.

    Derived entirely from the latest chunk_events row; no dedicated "waiting for
    model" event exists because it doesn't need one — it's exactly the state a
    RUNNING trajectory is in right after its latest event is AUDIT_FAILED (the
    DynamicsRunner records nothing else until it receives new weights and starts
    the next attempt's STARTED_DYNAMICS).
    """
    if traj_status == TrajectoryStatus.COMPLETED:
        return 'COMPLETED'
    if traj_status == TrajectoryStatus.FAILED:
        return 'FAILED'
    if latest_event is None:
        return 'PENDING'
    event_type = latest_event['event_type']
    if event_type == ChunkEventType.AUDIT_FAILED:
        return 'WAITING_FOR_MODEL_UPDATE'
    return event_type.name


def main() -> None:
    args = parse_args()
    db_url = args.db_url
    run_id = args.run_id or resolve_run_id(db_url)
    if run_id is None:
        print('No runs found (no rows in trajectory_frames yet)')
        return

    db = TrajectoryDB(db_url)

    trajectories = db.list_trajectories_in_run(run_id)
    latest_events = {e['traj_id']: e for e in db.get_latest_event_per_trajectory(run_id)}

    training_round = db.get_current_training_round(run_id)
    latest_training_event = db.get_latest_training_event(run_id)
    if latest_training_event and latest_training_event['event_type'] == ChunkEventType.STARTED_TRAINING:
        training_status = f"training round {latest_training_event['training_round']} in progress"
    else:
        training_status = f'idle (current round {training_round})'

    controller_log = db.get_controller_log(run_id)
    thresholds: dict[int | None, float] = {}
    if not controller_log.empty:
        latest_rows = controller_log.groupby('traj_id', dropna=False).tail(1)
        thresholds = dict(zip(latest_rows['traj_id'], latest_rows['threshold']))

    print(f'run_id = {run_id}')
    print(f'training: {training_status}')
    if thresholds:
        if list(thresholds) == [None]:
            print(f'threshold (shared): {thresholds[None]:.4g}')
        else:
            print('thresholds (per-traj): ' + ', '.join(
                f'{tid}={v:.4g}' for tid, v in sorted(thresholds.items(), key=lambda kv: (kv[0] is None, kv[0]))
            ))
    print()

    header = f"{'traj':>5} {'status':>10} {'chunk':>6} {'attempt':>8} {'model_v':>8} {'audit':>10} {'stage':>24} {'progress':>12}"
    print(header)
    for traj in trajectories:
        traj_id = traj['traj_id']
        latest_chunk_id = db.get_latest_chunk_id(run_id, traj_id)
        chunk_attempt = (
            db.get_latest_chunk_attempt(run_id, traj_id, latest_chunk_id)
            if latest_chunk_id is not None else None
        )
        stage = pipeline_stage(traj['status'], latest_events.get(traj_id))

        chunk_str = str(latest_chunk_id) if latest_chunk_id is not None else '-'
        attempt_str = str(chunk_attempt['attempt_index']) if chunk_attempt else '-'
        model_v_str = str(chunk_attempt['model_version']) if chunk_attempt else '-'
        audit_str = chunk_attempt['audit_status'].name if chunk_attempt and isinstance(chunk_attempt['audit_status'], AuditStatus) else '-'
        progress_str = f"{traj['chunks_completed']}/{traj['target_length']}"
        reason_suffix = f" ({traj['failure_reason']})" if traj['status'] == TrajectoryStatus.FAILED and traj['failure_reason'] else ''

        print(
            f"{traj_id:>5} {traj['status'].name:>10} {chunk_str:>6} {attempt_str:>8} "
            f"{model_v_str:>8} {audit_str:>10} {stage:>24} {progress_str:>12}{reason_suffix}"
        )


if __name__ == '__main__':
    main()
