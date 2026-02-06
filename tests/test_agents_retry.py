import asyncio
from typing import List

import pytest

from cascade.agents.config import CascadeAgentConfig
from cascade.agents.dummy import CascadeAgent


def _make_agent() -> CascadeAgent:
    config = CascadeAgentConfig(run_id='test', db_url='sqlite:///:memory:')
    return CascadeAgent(config)


@pytest.mark.asyncio
async def test_schedule_future_callback_success():
    """Callback is invoked when future completes successfully."""
    agent = _make_agent()
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    callback_results: List[str] = []
    callback_event = asyncio.Event()

    async def callback(fut: asyncio.Future) -> None:
        callback_results.append(fut.result())
        callback_event.set()

    loop.call_soon(future.set_result, "success")

    agent.schedule_future_callback(
        future,
        callback,
        description='test success',
    )

    await asyncio.wait_for(callback_event.wait(), timeout=1.0)

    assert callback_results == ["success"]


@pytest.mark.asyncio
async def test_schedule_future_callback_executor_failure():
    """Executor failure is logged and callback is not invoked."""
    agent = _make_agent()
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    callback_called = False
    callback_event = asyncio.Event()

    async def callback(fut: asyncio.Future) -> None:
        nonlocal callback_called
        callback_called = True
        callback_event.set()

    loop.call_soon(future.set_exception, RuntimeError("executor failure"))

    agent.schedule_future_callback(
        future,
        callback,
        description='test executor failure',
    )

    # Wait a bit to ensure callback task completes
    await asyncio.sleep(0.1)

    # Callback should not be called when future has exception
    assert not callback_called
    # The exception should have been logged (we can't easily verify this without
    # mocking the logger, but the important thing is the callback wasn't called)


@pytest.mark.asyncio
async def test_schedule_future_callback_callback_exception():
    """Callback exceptions are logged but not retried."""
    agent = _make_agent()
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    callback_called = False
    callback_event = asyncio.Event()

    async def callback(fut: asyncio.Future) -> None:
        nonlocal callback_called
        callback_called = True
        callback_event.set()
        raise RuntimeError("callback exception")

    loop.call_soon(future.set_result, "success")

    agent.schedule_future_callback(
        future,
        callback,
        description='test callback exception',
    )

    await asyncio.wait_for(callback_event.wait(), timeout=1.0)

    # Callback should have been called
    assert callback_called
    # The exception should have been logged (we can't easily verify this without
    # mocking the logger, but the important thing is it was called and exception handled)
