from subprocess import run, Popen
from pathlib import Path
import os, time
import psycopg2

db_path = Path('db').absolute()
os.environ['PGDATA'] = str(db_path)

if not (db_path / 'PG_VERSION').exists():
    run(['pg_ctl', 'initdb'])

proc = Popen(['postgres', '-D', str(db_path)])

for _ in range(50):
    try:
        conn = psycopg2.connect(dbname='postgres', host='localhost')
        break
    except psycopg2.OperationalError:
        time.sleep(0.3)

conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_roles WHERE rolname='ase'")
if not cur.fetchone():
    cur.execute("CREATE ROLE ase WITH LOGIN")

cur.execute("SELECT 1 FROM pg_database WHERE datname='cascade'")
if not cur.fetchone():
    cur.execute("CREATE DATABASE cascade OWNER ase")

conn.close()

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    proc.wait()