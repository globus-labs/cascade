#!/bin/bash
# One-time setup for the cascade Postgres server: create the cluster, the
# `cascade` role and database, and store a random password in ~/.pgpass.
# Tables are created by TrajectoryDB.create_tables() at run time.
#
# Environment variables (all optional):
#   PGROOT         where database files live (default: <repo>/../cascade-pg)
#   PGPORT         port to listen on (default: 54329)
#   PG_ALLOW_CIDR  space-separated networks allowed to connect
#                  (default: all private ranges; a password is always required)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PGBIN=$REPO/conda-env/bin
PGROOT=${PGROOT:-$(dirname "$REPO")/cascade-pg}  # outside the repo
PGDATA=$PGROOT/data
PGPORT=${PGPORT:-54329}  # not 5432, to avoid clashing with other users on the login node
PG_ALLOW_CIDR=${PG_ALLOW_CIDR:-10.0.0.0/8 172.16.0.0/12 192.168.0.0/16}
DBUSER=cascade
DBNAME=cascade
SOCK=/tmp/cascade-pg-$USER

if [[ -e $PGDATA ]]; then
    echo "$PGDATA already exists; refusing to re-initialize" >&2
    exit 1
fi
mkdir -p "$PGROOT" "$SOCK"
chmod 700 "$PGROOT" "$SOCK"

# Superuser is the invoking OS user, authenticated by peer on the unix socket
"$PGBIN/initdb" -D "$PGDATA" --auth-local=peer --auth-host=scram-sha-256 -E UTF8

cat >> "$PGDATA/postgresql.conf" <<EOF

# --- cascade settings ---
listen_addresses = '*'
port = $PGPORT
unix_socket_directories = '$SOCK'
max_connections = 500
shared_buffers = 1GB
password_encryption = scram-sha-256
logging_collector = off  # log to stdout (the screen session)
EOF

# Admin via local peer auth; the cascade role by password from allowed networks only
{
    echo "# TYPE  DATABASE  USER     ADDRESS           METHOD"
    echo "local   all       $USER                      peer"
    echo "host    $DBNAME   $DBUSER  127.0.0.1/32      scram-sha-256"
    for cidr in $PG_ALLOW_CIDR; do
        echo "host    $DBNAME   $DBUSER  $cidr    scram-sha-256"
    done
} > "$PGDATA/pg_hba.conf"

echo "$PGPORT" > "$PGROOT/port"

# Start briefly (socket only) to create the role and database
"$PGBIN/pg_ctl" -D "$PGDATA" -w -l "$PGROOT/init.log" -o "-c listen_addresses=''" start
trap '"$PGBIN/pg_ctl" -D "$PGDATA" -w stop' EXIT

# Password is passed as a psql variable: not visible in ps, history, or server logs
PW=$(openssl rand -hex 24)
"$PGBIN/psql" -h "$SOCK" -p "$PGPORT" -d postgres -v ON_ERROR_STOP=1 \
    -v pw="$PW" -v u="$DBUSER" -v db="$DBNAME" <<'SQL'
CREATE ROLE :"u" LOGIN PASSWORD :'pw';
CREATE DATABASE :"db" OWNER :"u";
SQL

# libpq reads ~/.pgpass automatically; $HOME must be shared with compute nodes
touch ~/.pgpass
chmod 600 ~/.pgpass
echo "*:$PGPORT:$DBNAME:$DBUSER:$PW" >> ~/.pgpass

echo "Initialized $PGDATA (port $PGPORT). Password stored in ~/.pgpass."
echo "Start the server with: $REPO/pg/pg-serve.sh"
