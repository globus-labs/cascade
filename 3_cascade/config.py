from __future__ import annotations

from typing import Any

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider


def get_parsl_config(name: str, run_dir: str, **kwargs: Any) -> Config:
    if name == "local":
        return get_local_config(run_dir, **kwargs)
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
        app_cache=False,
    )
