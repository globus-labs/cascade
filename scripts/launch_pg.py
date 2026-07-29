from subprocess import run, Popen
from pathlib import Path
import os

db_path = Path('db')
os.environ['PGDATA'] = str(db_path.absolute())
run(['pg_ctl', 'initdb'])
psql_proc = Popen(['postgres', '-D', db_path.absolute()])
