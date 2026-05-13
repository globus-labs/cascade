#!/usr/bin/env python3
"""
Migration to add atoms_labeled_blob column to training_frames table.

Existing rows receive NULL, which get_training_frames already handles by
skipping rows where atoms_labeled_blob is None.

Usage:
    python scripts/migrate_add_atoms_labeled.py --db-url sqlite:///cascade.db
    python scripts/migrate_add_atoms_labeled.py --db-url postgresql://user:pass@host/db

Run with --dry-run to inspect planned statements without applying them.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add atoms_labeled_blob column to training_frames table.",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="SQLAlchemy database URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without executing them.",
    )
    return parser.parse_args()


def column_exists(engine: Engine, table: str, column: str) -> bool:
    return any(col["name"] == column for col in inspect(engine).get_columns(table))


def ensure_atoms_labeled_blob(engine: Engine, dry_run: bool) -> None:
    if column_exists(engine, "training_frames", "atoms_labeled_blob"):
        print("`atoms_labeled_blob` already exists, nothing to do.")
        return

    stmt = (
        "ALTER TABLE training_frames ADD COLUMN atoms_labeled_blob BYTEA"
        if engine.dialect.name == "postgresql"
        else "ALTER TABLE training_frames ADD COLUMN atoms_labeled_blob BLOB"
    )

    if dry_run:
        print(f"[DRY-RUN] Would execute: {stmt}")
        return

    with engine.begin() as conn:
        conn.execute(text(stmt))
    print("Added `atoms_labeled_blob` column to training_frames.")


def main() -> int:
    args = parse_args()
    engine = create_engine(args.db_url)

    if "training_frames" not in inspect(engine).get_table_names():
        print("No `training_frames` table found; nothing to migrate.", file=sys.stderr)
        return 1

    ensure_atoms_labeled_blob(engine, args.dry_run)
    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
