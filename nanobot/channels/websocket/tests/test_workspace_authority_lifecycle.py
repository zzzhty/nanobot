"""Keep authority-test startup failures from leaking listeners or session workers."""

import asyncio
import errno
import socket
import sqlite3
from contextlib import nullcontext

import pytest

from nanobot.webui.outbound_delivery import WebUIOutboundDelivery

from . import test_workspace_authority as authority
from .ws_test_client import WsTestClient


@pytest.fixture
async def observed_gateways(monkeypatch):
    channels = []
    start_tasks = []
    create_channel = authority.WebSocketChannel

    def create(*args, **kwargs):
        channel = create_channel(*args, **kwargs)
        start = channel.start

        async def tracked_start():
            task = asyncio.current_task()
            assert task is not None
            start_tasks.append(task)
            await start()

        monkeypatch.setattr(channel, "start", tracked_start)
        channels.append(channel)
        return channel

    monkeypatch.setattr(authority, "WebSocketChannel", create)
    try:
        yield channels, start_tasks
    finally:
        # Also clean up when running these regressions against the old helper.
        for task in start_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*start_tasks, return_exceptions=True)
        for channel in reversed(channels):
            try:
                await channel.stop()
            finally:
                await channel.gateway.session_manager.state.aclose()


async def _assert_storage_closed(sessions, connections=()):
    with pytest.raises(RuntimeError, match="session state is closed"):
        await sessions.state.list_sessions()
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


async def test_startup_timeout_cancels_pending_initialization(
    tmp_path, monkeypatch, observed_gateways,
):
    channels, start_tasks = observed_gateways
    entered = asyncio.Event()
    release = asyncio.Event()
    initialize = WebUIOutboundDelivery.initialize

    async def blocked_initialize(self):
        entered.set()
        await release.wait()
        await initialize(self)

    monkeypatch.setattr(WebUIOutboundDelivery, "initialize", blocked_initialize)
    monkeypatch.setattr(authority, "_GATEWAY_STARTUP_TIMEOUT_S", 0.05, raising=False)

    async def enter_gateway():
        async with authority._running_gateway(tmp_path):
            pytest.fail("an uninitialized gateway became ready")

    driver = asyncio.create_task(enter_gateway())
    try:
        await asyncio.wait_for(entered.wait(), timeout=6)
        # The old helper has a hard-coded 5s deadline, then hangs in teardown.
        # A separate watchdog detects that failure without cancelling our evidence.
        done, _pending = await asyncio.wait({driver}, timeout=6)
        assert driver in done, "gateway setup timed out but its cleanup did not finish"
        with pytest.raises(TimeoutError):
            await driver
        assert start_tasks and all(task.done() for task in start_tasks)
        channel = channels[0]
        await _assert_storage_closed(channel.gateway.session_manager)
        release.set()
        await asyncio.sleep(0)
        assert not channel.is_running
        assert channel._server is None
    finally:
        driver.cancel()
        for task in start_tasks:
            task.cancel()
        release.set()
        await asyncio.gather(driver, *start_tasks, return_exceptions=True)


async def test_initialization_error_keeps_its_cause_and_closes_storage(
    tmp_path, monkeypatch, observed_gateways,
):
    channels, start_tasks = observed_gateways
    initialize = WebUIOutboundDelivery.initialize
    connections = []
    failure = RuntimeError("history initialization failed")

    async def failed_initialize(self):
        await initialize(self)
        connections.extend(channels[0].gateway.session_manager.state._connections)
        raise failure

    monkeypatch.setattr(WebUIOutboundDelivery, "initialize", failed_initialize)
    with pytest.raises(RuntimeError, match="history initialization failed") as caught:
        async with authority._running_gateway(tmp_path):
            pytest.fail("failed initialization became ready")
    assert caught.value is failure
    assert connections
    assert start_tasks and all(task.done() for task in start_tasks)
    channel = channels[0]
    assert channel._server is None
    await _assert_storage_closed(channel.gateway.session_manager, connections)


@pytest.mark.parametrize("body_fails", [False, True], ids=["normal", "body-error"])
async def test_gateway_exit_closes_listener_and_storage(
    tmp_path, observed_gateways, body_fails,
):
    _channels, start_tasks = observed_gateways
    expected = pytest.raises(ValueError, match="test body failed") if body_fails else nullcontext()
    with expected:
        async with authority._running_gateway(tmp_path) as (channel, uri, *_rest):
            server = channel._server
            assert server is not None and server.is_serving()
            token = channel.gateway.tokens.issue_token(60)
            async with WsTestClient(uri, token=token) as client:
                assert (await client.recv_ready()).event == "ready"
            sessions = channel.gateway.session_manager
            connections = tuple(sessions.state._connections)
            assert connections
            if body_fails:
                raise ValueError("test body failed")
    assert not server.is_serving()
    assert channel._server is None
    assert not channel.is_running
    assert start_tasks and all(task.done() for task in start_tasks)
    await _assert_storage_closed(sessions, connections)


async def test_occupied_loopback_port_preserves_bind_error(
    tmp_path, monkeypatch, observed_gateways,
):
    channels, start_tasks = observed_gateways
    create_channel = authority.WebSocketChannel
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()

        def create(*args, **kwargs):
            channel = create_channel(*args, **kwargs)
            channel.config.port = occupied.getsockname()[1]
            return channel

        monkeypatch.setattr(authority, "WebSocketChannel", create)
        with pytest.raises(OSError) as caught:
            async with authority._running_gateway(tmp_path):
                pytest.fail("gateway bound an occupied loopback port")
    # Windows can report access denied for a socket held with exclusive use.
    assert caught.value.errno in {errno.EADDRINUSE, errno.EACCES}
    assert start_tasks and all(task.done() for task in start_tasks)
    assert caught.value is start_tasks[0].exception()
    channel = channels[0]
    assert channel._server is None
    await _assert_storage_closed(channel.gateway.session_manager)
