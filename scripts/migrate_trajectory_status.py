#!/usr/bin/env python3
"""
One-off migration to add trajectory lifecycle status tracking.

The script:
* Ensures the legacy boolean `trajectories.done` exists.
* Adds a `trajectories.status` column (enum on Postgres, TEXT elsewhere).
* Backfills status using existing data (`done` -> COMPLETED, else RUNNING).
* Synchronizes the boolean flag with the new enum state.

Usage:
    python scripts/migrate_trajectory_status.py --db-url sqlite:///cascade.db
    python scripts/migrate_trajectory_status.py --db-url postgresql://user:pass@host/dbname

Run with `--dry-run` to inspect planned statements without applying them.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


STATUS_VALUES: Iterable[str] = ("RUNNING", "COMPLETED", "FAILED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add trajectory status support to an existing cascade database.",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help=(
            "SQLAlchemy database URL (e.g. sqlite:///cascade.db or "
            "postgresql://user:pass@host/db)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without executing them.",
    )
    return parser.parse_args()


def column_exists(engine: Engine, table: str, column: str) -> bool:
    return any(col["name"] == column for col in inspect(engine).get_columns(table))


def ensure_done_column(engine: Engine, dry_run: bool) -> None:
    if column_exists(engine, "trajectories", "done"):
        return
    stmt = (
        "ALTER TABLE trajectories ADD COLUMN done BOOLEAN NOT NULL DEFAULT 0"
        if engine.dialect.name == "sqlite"
        else "ALTER TABLE trajectories ADD COLUMN done BOOLEAN NOT NULL DEFAULT false"
    )
    if dry_run:
        print(f"[DRY-RUN] Would execute: {stmt}")
        return
    with engine.begin() as conn:
        conn.execute(text(stmt))
    print("Added `done` column with default false.")


def ensure_status_enum(engine: Engine, dry_run: bool) -> None:
    if engine.dialect.name != "postgresql":
        return
    ddl = f"""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_type WHERE typname = 'trajectorystatus'
        ) THEN
            CREATE TYPE trajectorystatus AS ENUM ({", ".join(f"'{v}'" for v in STATUS_VALUES)});
        END IF;
    END$$;
    """
    if dry_run:
        print("[DRY-RUN] Would ensure postgres enum `trajectorystatus` exists.")
        return
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("Ensured postgres enum `trajectorystatus` exists.")


def ensure_status_column(engine: Engine, dry_run: bool) -> None:
    if column_exists(engine, "trajectories", "status"):
        return

    if engine.dialect.name == "postgresql":
        ensure_status_enum(engine, dry_run)
        stmt = (
            "ALTER TABLE trajectories "
            "ADD COLUMN status trajectorystatus NOT NULL DEFAULT 'RUNNING'"
        )
    else:
        stmt = "ALTER TABLE trajectories ADD COLUMN status TEXT NOT NULL DEFAULT 'RUNNING'"

    if dry_run:
        print(f"[DRY-RUN] Would execute: {stmt}")
        return
    with engine.begin() as conn:
        conn.execute(text(stmt))
    print("Added `status` column with default RUNNING.")


def backfill_status(engine: Engine, dry_run: bool) -> None:
    if engine.dialect.name == "postgresql":
        update_sql = """
        UPDATE trajectories
        SET status = CASE
            WHEN done THEN 'COMPLETED'::trajectorystatus
            ELSE 'RUNNING'::trajectorystatus
        END
        WHERE status IS NULL
           OR status NOT IN (:running, :completed, :failed)
        """
        sync_sql = """
        UPDATE trajectories
        SET done = (status = 'COMPLETED'::trajectorystatus)
        """
    else:
        update_sql = """
        UPDATE trajectories
        SET status = CASE
            WHEN done THEN 'COMPLETED'
            ELSE 'RUNNING'
        END
        WHERE status IS NULL
           OR status NOT IN (:running, :completed, :failed)
        """
        sync_sql = """
        UPDATE trajectories
        SET done = (status = :completed)
        """
    params = {
        "running": "RUNNING",
        "completed": "COMPLETED",
        "failed": "FAILED",
    }
    if dry_run:
        print("[DRY-RUN] Would backfill status from done flag.")
    else:
        with engine.begin() as conn:
            result = conn.execute(text(update_sql), params)
        print(f"Backfilled {result.rowcount} trajectory rows with RUNNING/COMPLETED status.")

    if dry_run:
        print("[DRY-RUN] Would synchronize done flag with status values.")
        return
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text(sync_sql))
        else:
            conn.execute(text(sync_sql), {"completed": "COMPLETED"})
    print("Synchronized `done` boolean with `status` values.")


def main() -> int:
    args = parse_args()
    engine = create_engine(args.db_url)

    inspector = inspect(engine)
    if "trajectories" not in inspector.get_table_names():
        print("No `trajectories` table found; nothing to migrate.", file=sys.stderr)
        return 1

    ensure_done_column(engine, args.dry_run)
    ensure_status_column(engine, args.dry_run)
    backfill_status(engine, args.dry_run)

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

