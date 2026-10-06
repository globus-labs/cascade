#!/bin/bash
#PBS -A Diaspora
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=flare:home
#PBS -N pg-test-connect
#PBS -j oe
# Check that a compute node can reach the cascade Postgres server started by pg-serve.sh.
# Submit with: qsub pg/test-connect.sh  (or run it by hand inside `qsub -I`)
set -o pipefail

REPO=/lus/flare/projects/Diaspora/mike/cascade
PGBIN=$REPO/conda-env/bin
PGROOT=${PGROOT:-$(dirname "$REPO")/cascade-pg}
PGPORT=$(cat "$PGROOT/port")
PG_HOST=$(cat "$PGROOT/host")
export CASCADE_DB_URL=$(cat "$PGROOT/db_url")

echo "compute node: $(hostname)"
echo "CASCADE_DB_URL=$CASCADE_DB_URL"

echo "== pg_isready via $PG_HOST"
"$PGBIN/pg_isready" -h "$PG_HOST" -p "$PGPORT" -t 10

echo "== psql (password from ~/.pgpass)"
"$PGBIN/psql" -h "$PG_HOST" -p "$PGPORT" -U cascade -d cascade -w \
    -c "select inet_client_addr() as me, inet_server_addr() as server, version();"

echo "== Python / SQLAlchemy via TrajectoryDB"
module load frameworks
source "$REPO/venv/bin/activate"
python - <<'EOF'
import os
from sqlalchemy import text
from cascade.agents.db_orm import TrajectoryDB

db = TrajectoryDB(os.environ["CASCADE_DB_URL"])
db.create_tables()
with db.engine.connect() as conn:
    tables = conn.execute(text("select tablename from pg_tables where schemaname='public'")).scalars().all()
print("connected; tables:", sorted(tables))
EOF
