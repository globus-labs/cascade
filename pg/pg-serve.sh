#!/bin/bash
# Run the cascade Postgres server in the foreground on this login node.
# Intended to run inside screen; Ctrl-C performs a clean (fast) shutdown.
# Workers should use: export CASCADE_DB_URL=$(cat $PGROOT/db_url)
#
# Environment variables (all optional):
#   PGROOT   where database files live (default: <repo>/../cascade-pg)
#   PG_HOST  address workers use to reach this node (default: short hostname);
#            set to an IP if compute nodes can't resolve the hostname, e.g.
#            PG_HOST=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PGBIN=$REPO/conda-env/bin
PGROOT=${PGROOT:-$(dirname "$REPO")/cascade-pg}
PGDATA=$PGROOT/data
SOCK=/tmp/cascade-pg-$USER
PG_HOST=${PG_HOST:-$(hostname -s)}

if [[ ! -d $PGDATA ]]; then
    echo "$PGDATA not found; run pg-init.sh first" >&2
    exit 1
fi
PGPORT=$(cat "$PGROOT/port")
mkdir -p "$SOCK"
chmod 700 "$SOCK"

# Guard against running on two nodes at once, and clear a postmaster.pid
# left behind by a server that ran on a different node
if [[ -f $PGROOT/host ]]; then
    PREV=$(cat "$PGROOT/host")
    if [[ $PREV != "$PG_HOST" ]]; then
        if "$PGBIN/pg_isready" -q -h "$PREV" -p "$PGPORT"; then
            echo "Postgres is already running on $PREV:$PGPORT" >&2
            exit 1
        fi
        rm -f "$PGDATA/postmaster.pid"
    fi
fi
echo "$PG_HOST" > "$PGROOT/host"

# No password in the URL; libpq reads it from ~/.pgpass
echo "postgresql+psycopg://cascade@$PG_HOST:$PGPORT/cascade" > "$PGROOT/db_url"
echo "CASCADE_DB_URL=$(cat "$PGROOT/db_url")"

exec "$PGBIN/postgres" -D "$PGDATA"
