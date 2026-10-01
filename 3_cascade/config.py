from __future__ import annotations

import os
from typing import Any

from parsl.addresses import address_by_hostname, address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider, PBSProProvider

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_parsl_config(name: str, run_dir: str, **kwargs: Any) -> Config:
    if name == "local":
        return get_local_config(run_dir, **kwargs)
    elif name == "aurora":
        return get_aurora_config(run_dir, **kwargs)
    else:
        raise AssertionError(f"Unknown Parsl config name: {name}.")


def get_local_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    executor = HighThroughputExecutor(
        label="htex-local",
        max_workers_per_node=workers_per_node,
        address=address_by_hostname(),
        cores_per_worker=1,
        provider=LocalProvider(init_blocks=0, max_blocks=1),
    )
    return Config(
        executors=[executor],
        run_dir=run_dir,
        initialize_logging=False,
        retries=0,
    )


def get_aurora_config(
    run_dir: str,
    account: str = "Diaspora",
    queue: str = "debug",
    walltime: str = "0:30:00",
    nodes_per_job: int = 1,
    max_num_jobs: int = 1,
) -> Config:
    """Run from an Aurora login node; Parsl submits the PBS jobs itself.

    Each worker is pinned to one GPU tile, so tasks should use device='xpu'.
    Based on the ALCF Parsl example for Aurora.
    """
    # The frameworks module sets ZE_FLAT_DEVICE_HIERARCHY=FLAT, which exposes each
    # tile as its own device 0-11; 'gpu.tile' masks only work in COMPOSITE mode
    tile_names = [str(i) for i in range(12)]
    # The config will launch workers from this directory
    execute_dir = os.getcwd()
    return Config(
        executors=[
            HighThroughputExecutor(
                # Ensures connections are made over the slingshot network
                address=address_by_interface('hsn0'),
                # Ensures one worker per GPU tile on each node
                available_accelerators=tile_names,
                max_workers_per_node=12,
                # Distributes threads to workers/tiles in a way optimized for Aurora
                cpu_affinity="list:1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204",
                # Increase if you have many more tasks than workers
                prefetch_capacity=0,
                # Options that specify properties of PBS Jobs
                provider=PBSProProvider(
                    # Project name
                    account=account,
                    # Submission queue
                    queue=queue,
                    # Commands run before workers launched. PBS's per-job TMPDIR is too long
                    # for the unix socket the worker pool's multiprocessing manager creates
                    worker_init=f'''module load frameworks; source {REPO}/venv/bin/activate; cd {execute_dir}; export TMPDIR=/tmp''',
                    # Wall time for batch jobs
                    walltime=walltime,
                    # Change if data/modules located on other filesystem
                    scheduler_options="#PBS -l filesystems=home:flare",
                    # Ensures 1 manager per node; the manager will distribute work to its 12 workers, one per tile
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
                    # options added to #PBS -l select aside from ncpus
                    select_options="",
                    # Number of nodes per PBS job
                    nodes_per_block=nodes_per_job,
                    # Minimum number of concurrent PBS jobs running workflow
                    min_blocks=0,
                    # Maximum number of concurrent PBS jobs running workflow
                    max_blocks=max_num_jobs,
                    # Hardware threads per node
                    cpus_per_node=208,
                ),
            ),
        ],
        run_dir=run_dir,
        # 0 so failures surface during benchmarking; set to 1 for production runs
        # whose tasks may be interrupted by a PBS job ending
        retries=0,
    )
