import asyncio
from typing import List

import pytest

from cascade.agents.config import CascadeAgentConfig
from cascade.agents.dummy import CascadeAgent


def _make_agent() -> CascadeAgent:
    config = CascadeAgentConfig(run_id='test', db_url='sqlite:///:memory:')
    return CascadeAgent(config)


@pytest.mark.asyncio
async def test_schedule_future_retries_executor_failure(monkeypatch):
    """Executor failure triggers a single retry that then succeeds.

    The first future raises immediately. We ensure schedule_future_callback:
    - sleeps for the configured delay before retrying,
    - requests a new future via the resubmit hook,
    - and eventually invokes the callback once the retry future resolves.
    """
    agent = _make_agent()
    sleep_delays: List[float] = []

    async def fake_sleep(delay: float, result=None):
        sleep_delays.append(delay)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    loop = asyncio.get_running_loop()
    first_future = loop.create_future()
    retry_future = loop.create_future()

    callback_results: List[str] = []
    callback_event = asyncio.Event()

    async def callback(fut: asyncio.Future) -> None:
        callback_results.append(fut.result())
        callback_event.set()

    resubmit_calls: List[int] = []

    def resubmit(attempt: int):
        resubmit_calls.append(attempt)
        loop.call_soon(retry_future.set_result, "success")
        return retry_future

    loop.call_soon(first_future.set_exception, RuntimeError("boom"))

    agent.schedule_future_callback(
        first_future,
        callback,
        description='test executor failure',
        max_attempts=2,
        retry_delay=0.5,
        resubmit=resubmit,
    )

    await asyncio.wait_for(callback_event.wait(), timeout=0.1)

    assert callback_results == ["success"]
    assert resubmit_calls == [2]
    assert sleep_delays == [0.5]


@pytest.mark.asyncio
async def test_schedule_future_retries_callback_failure(monkeypatch):
    """Callback failure on first attempt retries and succeeds on second.

    The initial future resolves, but the callback raises. We verify:
    - the helper asks for a retry future,
    - the callback runs again with the retry future's result,
    - and overall we only succeed after that retry.
    """
    agent = _make_agent()

    async def fake_sleep(delay: float, result=None):
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    loop = asyncio.get_running_loop()
    initial_future = loop.create_future()
    retry_future = loop.create_future()

    callback_attempts = 0
    callback_results: List[str] = []
    callback_event = asyncio.Event()

    async def callback(fut: asyncio.Future) -> None:
        nonlocal callback_attempts
        callback_attempts += 1
        if callback_attempts == 1:
            raise RuntimeError("callback failure")
        callback_results.append(fut.result())
        callback_event.set()

    resubmit_calls: List[int] = []

    def resubmit(attempt: int):
        resubmit_calls.append(attempt)
        loop.call_soon(retry_future.set_result, "final")
        return retry_future

    loop.call_soon(initial_future.set_result, "ignored")

    agent.schedule_future_callback(
        initial_future,
        callback,
        description='test callback failure',
        max_attempts=2,
        resubmit=resubmit,
    )

    await asyncio.wait_for(callback_event.wait(), timeout=0.1)

    assert callback_attempts == 2
    assert callback_results == ["final"]
    assert resubmit_calls == [2]


@pytest.mark.asyncio
async def test_schedule_future_stops_after_exhaustion(monkeypatch):
    """All attempts fail and the helper stops after hitting the cap.

    Both the initial future and the retry future raise. The test confirms:
    - no callback is invoked,
    - resubmit is called exactly once (for attempt #2),
    - the on_exhausted hook observes the final exception.
    """
    agent = _make_agent()

    # so that the retry loop doesnt sleep
    async def fake_sleep(delay: float, result=None):
        return result
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    loop = asyncio.get_running_loop()
    first_future = loop.create_future()
    second_future = loop.create_future()

    callback_called = False
    exhausted_errors: List[Exception] = []
    exhausted_event = asyncio.Event()

    async def callback(_fut: asyncio.Future) -> None:
        nonlocal callback_called
        callback_called = True

    resubmit_calls: List[int] = []

    def resubmit(attempt: int):
        resubmit_calls.append(attempt)
        loop.call_soon(second_future.set_exception, RuntimeError("boom again"))
        return second_future

    async def on_exhausted(exc: Exception) -> None:
        exhausted_errors.append(exc)
        exhausted_event.set()

    loop.call_soon(first_future.set_exception, RuntimeError("boom"))

    agent.schedule_future_callback(
        first_future,
        callback,
        description='test exhaustion',
        max_attempts=2,
        resubmit=resubmit,
        on_exhausted=on_exhausted,
    )

    await asyncio.wait_for(exhausted_event.wait(), timeout=0.1)

    assert not callback_called
    assert resubmit_calls == [2]
    assert len(exhausted_errors) == 1
    assert str(exhausted_errors[0]) == "boom again"

