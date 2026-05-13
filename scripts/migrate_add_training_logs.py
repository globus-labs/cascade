#!/usr/bin/env python3
"""
Migration to add the training_logs table for persisting per-round training metrics.

Fresh runs get this table automatically via create_tables(). This script is only
needed for existing databases created before this table was introduced.

Usage:
    python scripts/migrate_add_training_logs.py --db-url sqlite:///cascade.db
    python scripts/migrate_add_training_logs.py --db-url postgresql://user:pass@host/db

Run with --dry-run to preview actions without applying them.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add training_logs table to an existing cascade database.",
    )
    parser.add_argument("--db-url", required=True, help="SQLAlchemy database URL")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without executing them.")
    return parser.parse_args()


def ensure_training_logs_table(engine: Engine, dry_run: bool) -> None:
    if "training_logs" in inspect(engine).get_table_names():
        print("`training_logs` table already exists, nothing to do.")
        return

    if engine.dialect.name == "postgresql":
        stmt = """
        CREATE TABLE training_logs (
            id               SERIAL PRIMARY KEY,
            run_id           TEXT NOT NULL,
            training_round   INTEGER NOT NULL,
            log_json         JSONB NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_training_log_run_round UNIQUE (run_id, training_round)
        );
        CREATE INDEX ix_training_logs_run_id         ON training_logs (run_id);
        CREATE INDEX ix_training_logs_training_round ON training_logs (training_round);
        """
    else:
        stmt = """
        CREATE TABLE training_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           TEXT NOT NULL,
            training_round   INTEGER NOT NULL,
            log_json         JSON NOT NULL,
            created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (run_id, training_round)
        );
        CREATE INDEX ix_training_logs_run_id         ON training_logs (run_id);
        CREATE INDEX ix_training_logs_training_round ON training_logs (training_round);
        """

    if dry_run:
        print(f"[DRY-RUN] Would execute:\n{stmt}")
        return

    with engine.begin() as conn:
        for statement in stmt.strip().split(';'):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    print("Created `training_logs` table.")


def main() -> int:
    args = parse_args()
    engine = create_engine(args.db_url)
    ensure_training_logs_table(engine, args.dry_run)
    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
