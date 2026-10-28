"""Executors that sit in front of the shared Parsl pool"""
from __future__ import annotations

import functools
import threading
from concurrent.futures import Executor, Future
from typing import Any, Callable


class _DropResourceSpec:
    """Calls ``fn`` and drops the ``parsl_resource_specification`` argument Parsl passes on to the task function.

    A class instead of a closure so the pickle sent to the workers only holds ``fn`` and a reference to this class.
    Copies ``fn``'s name, which Parsl uses to name the app in its logs.
    """

    def __init__(self, fn: Callable):
        self.fn = fn
        # a functools.partial has no __name__, but Parsl needs one
        self.__name__ = getattr(fn, '__name__', None) or getattr(getattr(fn, 'func', None), '__name__', type(fn).__name__)
        functools.update_wrapper(self, fn)  # also sets __wrapped__, which makes inspect.signature report fn's signature

    def __call__(self, *args: Any, parsl_resource_specification: dict | None = None, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class PriorityExecutor(Executor):
    """Submits tasks to a Parsl pool with a priority, so they run ahead of tasks submitted without one.

    Parsl's high-throughput executor serves waiting tasks lowest priority value first, and tasks submitted
    with no priority last. It never preempts a running task, only reorders the ones still waiting for a worker.

    Args:
        pool: The ``ParslPoolExecutor`` to submit to. Its owner shuts it down, not this wrapper.
        priority: Lower runs first.
    """

    def __init__(self, pool: Executor, priority: int = 0):
        self.pool = pool
        self.priority = priority
        self._wrapped: dict[Callable, _DropResourceSpec] = {}
        self._lock = threading.Lock()

    def _wrap(self, fn: Callable) -> _DropResourceSpec:
        # one wrapper per function, so the pool's per-function app cache doesn't grow with every call
        with self._lock:
            if fn not in self._wrapped:
                self._wrapped[fn] = _DropResourceSpec(fn)
            return self._wrapped[fn]

    def submit(self, fn: Callable, /, *args: Any, **kwargs: Any) -> Future:
        return self.pool.submit(
            self._wrap(fn), *args,
            parsl_resource_specification={'priority': self.priority}, **kwargs
        )

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Does nothing: the pool belongs to whoever created it"""
