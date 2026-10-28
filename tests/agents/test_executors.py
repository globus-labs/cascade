"""Tests for the executor that submits to Parsl with a priority"""
from concurrent.futures import Future
from functools import partial
from pathlib import Path
import pickle
import time

import pytest

from cascade.agents.executors import PriorityExecutor


def add(a, b=0):
    return a + b


class FakePool:
    """Records submissions and runs the function the way a Parsl worker would, with the extra keyword argument"""

    def __init__(self):
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def test_priority_reaches_pool_and_argument_is_dropped():
    pool = FakePool()
    executor = PriorityExecutor(pool, priority=3)
    assert executor.submit(add, 1, b=2).result() == 3
    _, args, kwargs = pool.calls[0]
    assert args == (1,)
    assert kwargs == {'b': 2, 'parsl_resource_specification': {'priority': 3}}


def test_wrapper_cached_per_function_and_keeps_name():
    pool = FakePool()
    executor = PriorityExecutor(pool)
    executor.submit(add, 1)
    executor.submit(add, 2)
    assert pool.calls[0][0] is pool.calls[1][0]  # same object, so Parsl's per-function app cache is hit
    assert pool.calls[0][0].__name__ == 'add'


def test_partial_gets_a_name_and_runs():
    pool = FakePool()
    executor = PriorityExecutor(pool)
    assert executor.submit(partial(add, b=10), 1).result() == 11
    assert pool.calls[0][0].__name__ == 'add'  # Parsl needs a __name__, which a partial lacks


def test_wrapped_function_pickles():
    wrapped = PriorityExecutor(FakePool())._wrap(add)
    assert pickle.loads(pickle.dumps(wrapped))(1, b=1, parsl_resource_specification={'priority': 0}) == 2


def test_shutdown_leaves_pool_alone():
    pool = FakePool()
    PriorityExecutor(pool).shutdown()
    assert pool.calls == []


def sleep_and_report(seconds: float) -> float:
    time.sleep(seconds)
    return time.time()


def test_parsl_runs_prioritized_task_ahead_of_waiting_ones(tmp_path):
    """One worker: a prioritized task submitted last must start before tasks that were waiting first"""
    pytest.importorskip('parsl')
    from parsl.concurrent import ParslPoolExecutor
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider
    from parsl.addresses import address_by_hostname

    config = Config(
        executors=[HighThroughputExecutor(
            label='htex-test', max_workers_per_node=1, address=address_by_hostname(), prefetch_capacity=0,
            # the worker imports this module to run sleep_and_report
            provider=LocalProvider(init_blocks=1, max_blocks=1, worker_init=f'export PYTHONPATH={Path(__file__).parent}:$PYTHONPATH'))],
        run_dir=str(tmp_path / 'parsl'), initialize_logging=False)
    with ParslPoolExecutor(config=config) as pool:
        priority = PriorityExecutor(pool)
        first = pool.submit(sleep_and_report, 3.)  # occupies the only worker
        time.sleep(1.)  # let it start
        waiting = [pool.submit(sleep_and_report, 0.2) for _ in range(3)]  # no priority: served in order, last
        urgent = priority.submit(sleep_and_report, 0.2)  # submitted last
        first.result()
        finished = {'urgent': urgent.result(), 'waiting': [w.result() for w in waiting]}
    assert finished['urgent'] < min(finished['waiting'])
