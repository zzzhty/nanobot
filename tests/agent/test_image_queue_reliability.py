"""Image-queue failure paths, using a fake provider and a finite wait.

These tests fail while a provider call, a delivery, or an unknown row can
hold a session open with no terminal state. They do not contact QQ or a
real image service.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from nanobot.agent.image_queue import ImageQueue
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.image_generation import ImageGenerationTool, ImageRequestRejectedError
from nanobot.agent.tools.image_generation_config import ImageGenerationToolConfig
from nanobot.bus.events import InboundMessage
from tests.agent.test_image_queue import _loop, _queue, _tool


def _message(
    text: str,
    message_id: str,
    *,
    generation: str = "generation-a",
    chat_id: str = "group-a",
    sender_id: str = "user-a",
) -> InboundMessage:
    return InboundMessage(
        channel="qq",
        sender_id=sender_id,
        chat_id=chat_id,
        content=text,
        session_generation=generation,
        metadata={"message_id": message_id, "raw_content": text, "chat_type": "group"},
    )


def _states(root: Path) -> list[tuple[str | None, str]]:
    db = sqlite3.connect(root / "jobs.sqlite3")
    try:
        rows = db.execute(
            "SELECT json_extract(params, '$.prompt'), state FROM jobs ORDER BY rowid"
        ).fetchall()
    finally:
        db.close()
    return [(None if prompt is None else str(prompt), str(state)) for prompt, state in rows]


def _prompts(queue: ImageQueue) -> list[str]:
    return [json.loads(str(row["params"]))["prompt"] for row in queue._candidates() if row["params"]]


class ScriptedImageTool(ImageGenerationTool):
    """Deterministic provider: hang, fail, or return one artifact path."""

    def __init__(
        self,
        workspace: Path,
        *,
        hang: set[str] | None = None,
        fail: set[str] | None = None,
        swallow: set[str] | None = None,
        reject: dict[str, str] | None = None,
        unresolved: set[str] | None = None,
        distinct: bool = False,
    ) -> None:
        super().__init__(
            workspace=workspace,
            config=ImageGenerationToolConfig(
                persistent_queue=True,
                provider="fake",
                model="fake-image",
                enabled=True,
            ),
        )
        self.calls: list[str] = []
        self.release = asyncio.Event()
        self.hang = hang or set()
        self.fail = fail or set()
        self.swallow = swallow or set()
        self.reject = reject or {}
        self.unresolved = unresolved or set()
        self.distinct = distinct
        self.finished = 0

    async def generate_now(
        self,
        prompt: str,
        reference_images: list[str] | None,
        aspect_ratio: str | None,
        image_size: str | None,
        count: int | None,
        *,
        references_are_resolved: bool = False,
        propagate_provider_errors: bool = False,
    ) -> str:
        del reference_images, aspect_ratio, image_size, count, references_are_resolved
        del propagate_provider_errors
        self.calls.append(prompt)
        occurrence = self.calls.count(prompt)
        try:
            if prompt in self.reject and occurrence == 1:
                raise ImageRequestRejectedError(self.reject[prompt])
            if prompt in self.unresolved and occurrence == 1:
                raise OSError("connection reset")
            if prompt in self.swallow and occurrence == 1:
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        continue
            elif prompt in self.hang:
                await self.release.wait()
            if prompt in self.fail:
                raise RuntimeError("provider failed")
            path = f"generated/{len(self.calls)}.png" if self.distinct else "generated/out.png"
            return json.dumps({"artifacts": [{"path": path}]})
        finally:
            self.finished += 1


def _persist(loop: AgentLoop, key: str) -> str:
    session = loop.sessions.get_or_create(key)
    loop.sessions.save(session)
    return session.generation


def _open_queue(loop: AgentLoop, root: Path) -> ImageQueue:
    queue = _queue(loop, root)
    queue.provider_timeout = 0.2
    queue.delivery_timeout = 0.2
    return queue


def _submit(queue: ImageQueue, text: str, message_id: str, tool: ImageGenerationTool, generation: str) -> None:
    message = _message(text, message_id, generation=generation)
    queue.admit(message)
    record = queue.inputs[message.metadata["_local_image_queue_input"]]
    queue.submit(record, generation, tool, {"prompt": text}, "new", None)
    with queue.turn(message):
        queue.finish(message, True)


async def _until(predicate: Callable[[], bool], timeout: float = 1.5) -> None:
    async def _wait() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(_wait(), timeout)


async def _stop(queue: ImageQueue, pump: asyncio.Task[None], release: asyncio.Event | None = None) -> None:
    if release is not None:
        release.set()
    pump.cancel()
    pending: list[asyncio.Task[object]] = [pump, *queue.workers.values()]
    background = getattr(queue, "_background", None)
    if isinstance(background, dict):
        for tasks in background.values():
            pending.extend(tasks)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=1)
    if queue._owner.is_locked:
        queue._owner.release()


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _watched_queue(loop: AgentLoop, root: Path, clock: Clock) -> ImageQueue:
    return ImageQueue(
        loop,
        root,
        clock=clock,
        debounce=0,
        provider_timeout=180,
        watch_extend=300,
        delivery_timeout=0.2,
    )


def _command(text: str, generation: str, *, chat_id: str = "group-a") -> InboundMessage:
    return InboundMessage(
        channel="qq",
        sender_id="user-a",
        chat_id=chat_id,
        content=text,
        session_generation=generation,
        metadata={"message_id": "cmd-" + text, "raw_content": text, "chat_type": "group"},
    )


async def _nudge(queue: ImageQueue, clock: Clock, seconds: float) -> None:
    clock.advance(seconds)
    queue.decision.set()
    await asyncio.sleep(0.05)
    queue.decision.set()


def _rows(root: Path) -> list[dict[str, object]]:
    db = sqlite3.connect(root / "jobs.sqlite3")
    try:
        found = db.execute(
            """SELECT json_extract(params, '$.prompt'), state, detail, extensions, prompted,
                      prompt_error, delivery_error, superseded, paths, watch_until, cancelled
               FROM jobs ORDER BY rowid"""
        ).fetchall()
    finally:
        db.close()
    names = (
        "prompt", "state", "detail", "extensions", "prompted", "prompt_error",
        "delivery_error", "superseded", "paths", "watch_until", "cancelled",
    )
    return [dict(zip(names, row, strict=True)) for row in found]


def _capture(notes: list[dict[str, object]]):
    async def _send(message: object, **_kwargs: object) -> None:
        notes.append({
            "channel": str(getattr(message, "channel", "")),
            "chat": str(getattr(message, "chat_id", "")),
            "content": str(getattr(message, "content", "")),
            "media": [str(item) for item in getattr(message, "media", []) or []],
        })

    return _send


@pytest.mark.asyncio
async def test_soft_timeout_prompts_without_dropping_the_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[tuple[str, str]] = []

    async def _capture(message: object, **_kwargs: object) -> None:
        content = getattr(message, "content", "")
        chat = getattr(message, "chat_id", "")
        notes.append((str(chat), str(content)))

    monkeypatch.setattr("nanobot.agent.image_queue.deliver_to_channel", _capture)
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    loop.tools.register(tool)
    clock = Clock()
    queue = _watched_queue(loop, tmp_path / "image-queue", clock)
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        chat = _message("再等五分钟", "m-chat", generation=generation)
        assert await queue.intercept(chat, "qq:group-a") is False
        almost = _command("/image-wait please", generation)
        assert await queue.intercept(almost, "qq:group-a") is False
        early = _command("/image-wait", generation)
        queue.admit(early)
        assert "_local_image_queue_input" not in early.metadata
        assert await queue.intercept(early, "qq:group-a") is True
        await _until(lambda: any("第一次等待" in text for _chat, text in notes))
        assert tool.calls == ["one"]
        await _nudge(queue, clock, 180)
        await _until(lambda: any("可能还在生成" in text for _chat, text in notes), timeout=3)
        assert tool.calls == ["one"]
        assert _states(queue.root) == [("one", "running")]
        assert queue.status("qq:group-a", generation)["jobs"][0]["extensions"] == 0
        assert notes[0][0] == "group-a"
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_cancel_during_provider_does_not_repeat_or_block_the_next_draft(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        assert queue.cancel("qq:group-a") >= 0
        await _until(lambda: _states(queue.root) == [("one", "cancelled")])
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["one", "two"])
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_provider_that_swallows_cancel_does_not_hold_the_session(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path, swallow={"one"})
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        queue.cancel("qq:group-a")
        await _until(lambda: _states(queue.root)[0][1] == "cancelled")
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["one", "two"])
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_delivery_hang_does_not_call_the_provider_again_or_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    async def _hang(*_args: object, **_kwargs: object) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("nanobot.agent.image_queue.deliver_to_channel", _hang)
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path)
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: started.is_set() and _states(queue.root) == [("one", "done")])
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["one", "two"])
        failed = queue.status("qq:group-a", generation)
        first = next(job for job in failed["jobs"] if job.get("prompt") == "one")
        assert first["paths"] == ["generated/out.png"]
        assert first["delivery_error"] == 1
        assert tool.calls.count("one") == 1
        follow = _message("thanks", "m-3", generation=generation)
        queue.admit(follow)
        with queue.turn(follow):
            block = await queue.planning_context(SimpleNamespace(
                session_generation=generation,
                persist_session=True,
                log_content=True,
                session_key=follow.session_key,
                channel="qq",
                chat_id="group-a",
                sender_id="user-a",
            ))
        assert block is not None
        assert "was not delivered" in block.content
    finally:
        await _stop(queue, pump)


@pytest.mark.asyncio
async def test_provider_exception_does_not_retry_or_block_the_next_draft(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path, fail={"one"})
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, generation)
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["one", "two"])
        assert _states(queue.root)[0][1] == "unknown"
        assert tool.calls.count("one") == 1
    finally:
        await _stop(queue, pump)


@pytest.mark.asyncio
async def test_running_without_a_worker_becomes_unknown_and_is_not_sent_again(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path)
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    _submit(queue, "one", "m-1", tool, generation)
    frozen = queue._freeze(queue._candidates()[0])
    assert frozen is not None
    pump = asyncio.create_task(queue.run())
    try:
        await _until(lambda: _states(queue.root)[0] == ("one", "unknown") and tool.calls == [])
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["two"])
    finally:
        await _stop(queue, pump)


@pytest.mark.asyncio
async def test_restart_does_not_retry_an_unknown_call(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path)
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    _submit(queue, "one", "m-1", tool, generation)
    frozen = queue._freeze(queue._candidates()[0])
    assert frozen is not None
    await queue.close()

    restarted = _open_queue(loop, queue.root)
    pump = asyncio.create_task(restarted.run())
    try:
        await _until(lambda: _states(restarted.root) == [("one", "unknown")] and tool.calls == [])
        _submit(restarted, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["two"])
        assert tool.calls.count("one") == 0
    finally:
        await _stop(restarted, pump)


@pytest.mark.asyncio
async def test_another_session_runs_while_one_provider_is_stuck(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    first = _persist(loop, "qq:group-a")
    second = _persist(loop, "qq:group-b")
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    queue.provider_timeout = 30
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, first)
        await _until(lambda: tool.calls == ["one"])

        message = _message("two", "m-2", generation=second, chat_id="group-b", sender_id="user-b")
        queue.admit(message)
        record = queue.inputs[message.metadata["_local_image_queue_input"]]
        queue.submit(record, second, tool, {"prompt": "two"}, "new", None)
        with queue.turn(message):
            queue.finish(message, True)
        await _until(lambda: "two" in tool.calls)
        assert tool.calls == ["one", "two"]
        assert _states(queue.root)[0] == ("one", "running")
    finally:
        await _stop(queue, pump, tool.release)


@pytest.fixture
def queue(tmp_path: Path):
    image_queue = _queue(_loop(tmp_path), tmp_path / "image-queue")
    try:
        yield image_queue
    finally:
        if image_queue._owner.is_locked:
            image_queue._owner.release()


def test_unknown_row_does_not_block_a_later_draft(queue: ImageQueue, tmp_path: Path) -> None:
    first = _message("one", "m-1")
    queue.admit(first)
    created = queue.inputs[first.metadata["_local_image_queue_input"]]
    queue.submit(created, "generation-a", _tool(tmp_path), {"prompt": "one"}, "new", None)
    with queue.turn(first):
        queue.finish(first, True)
    frozen = queue._freeze(queue._candidates()[0])
    assert frozen is not None
    with queue._tx() as db:
        db.execute("UPDATE jobs SET state='unknown', notice=1 WHERE id=?", (str(frozen["id"]),))

    second = _message("two", "m-2")
    queue.admit(second)
    queued = queue.inputs[second.metadata["_local_image_queue_input"]]
    queue.submit(queued, "generation-a", _tool(tmp_path), {"prompt": "two"}, "new", None)
    with queue.turn(second):
        queue.finish(second, True)

    assert _prompts(queue) == ["two"]


def test_cancel_terminalizes_unknown_and_a_later_draft_can_run(queue: ImageQueue, tmp_path: Path) -> None:
    first = _message("one", "m-1")
    queue.admit(first)
    created = queue.inputs[first.metadata["_local_image_queue_input"]]
    queue.submit(created, "generation-a", _tool(tmp_path), {"prompt": "one"}, "new", None)
    with queue.turn(first):
        queue.finish(first, True)
    frozen = queue._freeze(queue._candidates()[0])
    assert frozen is not None
    with queue._tx() as db:
        db.execute("UPDATE jobs SET state='unknown' WHERE id=?", (str(frozen["id"]),))
    queue.cancel(first.session_key)

    second = _message("two", "m-2")
    queue.admit(second)
    queued = queue.inputs[second.metadata["_local_image_queue_input"]]
    queue.submit(queued, "generation-a", _tool(tmp_path), {"prompt": "two"}, "new", None)
    with queue.turn(second):
        queue.finish(second, True)

    assert _states(queue.root)[0][1] == "cancelled"
    assert _prompts(queue) == ["two"]


def test_public_failure_is_not_a_dropped_connection() -> None:
    from nanobot.agent.tools.image_generation import _transport_gave_up, public_image_reason
    from nanobot.providers.image_generation import ImageGenerationError

    reason = public_image_reason(ImageGenerationError(
        "api_key=sekret https://files.example/secret denied by policy"
    ))
    assert "sekret" not in reason
    assert "http" not in reason
    assert "denied by policy" in reason
    assert _transport_gave_up(ImageGenerationError("timed out") ) is False
    try:
        raise ImageGenerationError("timed out") from TimeoutError("deadline")
    except ImageGenerationError as exc:
        assert _transport_gave_up(exc) is True
    try:
        raise ImageGenerationError("no") from httpx.HTTPStatusError(
            "no",
            request=httpx.Request("POST", "https://image.example/v1"),
            response=httpx.Response(400, request=httpx.Request("POST", "https://image.example/v1")),
        )
    except ImageGenerationError as exc:
        assert _transport_gave_up(exc) is False


def test_queued_http_timeout_can_outlive_the_provider_default() -> None:
    from nanobot.providers.image_generation import (
        OpenRouterImageGenerationClient,
        image_http_timeout,
    )

    client = OpenRouterImageGenerationClient(api_key="fixture")
    assert client._http_client_kwargs()["timeout"] == client.timeout
    token = image_http_timeout.set(780)
    try:
        assert client._http_client_kwargs()["timeout"] == 780
    finally:
        image_http_timeout.reset(token)


def test_same_origin_cannot_submit_a_second_provider_job(queue: ImageQueue, tmp_path: Path) -> None:
    message = _message("one", "m-1")
    queue.admit(message)
    record = queue.inputs[message.metadata["_local_image_queue_input"]]
    tool = _tool(tmp_path)
    first = queue.submit(record, "generation-a", tool, {"prompt": "one"}, "new", None)
    second = queue.submit(record, "generation-a", tool, {"prompt": "again"}, "new", None)
    assert "draft queued" in first
    assert "already accepted" in second
    assert _states(queue.root) == [("one", "draft")]


def test_defer_hold_still_blocks_until_it_is_revised_or_cancelled(queue: ImageQueue, tmp_path: Path) -> None:
    message = _message("one", "m-1")
    queue.admit(message)
    record = queue.inputs[message.metadata["_local_image_queue_input"]]
    queue.submit(record, "generation-a", _tool(tmp_path), {"prompt": "one"}, "merge", None)
    with queue.turn(message):
        queue.finish(message, True)
    question = _message("which color", "m-2")
    queue.admit(question)
    held = queue.inputs[question.metadata["_local_image_queue_input"]]
    assert queue.defer(held, "generation-a").startswith("Draft held")
    with queue.turn(question):
        queue.finish(question, True)
    assert queue._candidates() == []
    assert queue.cancel(message.session_key) == 1
    assert _states(queue.root)[0][1] == "cancelled"
    assert queue._candidates() == []


def _watched(tmp_path: Path, tool: ScriptedImageTool, notes: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nanobot.agent.image_queue.deliver_to_channel", _capture(notes))
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    loop.tools.register(tool)
    clock = Clock()
    queue = _watched_queue(loop, tmp_path / "image-queue", clock)
    pump = asyncio.create_task(queue.run())
    return generation, clock, queue, pump


@pytest.mark.asyncio
async def test_waiting_keeps_the_same_call_until_it_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    generation, clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        await _nudge(queue, clock, 180)
        await _until(lambda: any("可能还在生成" in str(item["content"]) for item in notes), timeout=3)
        assert await queue.intercept(_command("/image-wait", generation), "qq:group-a") is True
        await _until(lambda: any("继续观察" in str(item["content"]) for item in notes), timeout=3)
        deadline = _rows(queue.root)[0]["watch_until"]
        assert _rows(queue.root)[0]["extensions"] == 1
        assert await queue.intercept(_command("/image-wait", generation), "qq:group-a") is True
        await _until(lambda: any("已经在多等" in str(item["content"]) for item in notes), timeout=3)
        assert _rows(queue.root)[0]["watch_until"] == deadline
        assert await queue.intercept(_message("再等五分钟", "m-chat", generation=generation), "qq:group-a") is False
        assert tool.calls == ["one"]
        tool.release.set()
        await _until(
            lambda: any("图画好了" in str(item["content"]) for item in notes) and tool.finished == 1,
            timeout=3,
        )
        pictures = [item for item in notes if "图画好了" in str(item["content"])]
        assert len(pictures) == 1
        assert pictures[0]["chat"] == "group-a"
        assert pictures[0]["media"] == ["generated/out.png"]
        assert tool.calls == ["one"]
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_second_wait_stops_without_another_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    generation, clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        await _nudge(queue, clock, 180)
        await _until(lambda: any("可能还在生成" in str(item["content"]) for item in notes), timeout=3)
        assert await queue.intercept(_command("/image-wait", generation), "qq:group-a") is True
        await _nudge(queue, clock, 300)
        await _until(lambda: _rows(queue.root)[0]["state"] == "unknown", timeout=3)
        assert _rows(queue.root)[0]["detail"] == "extended"
        assert tool.calls == ["one"]
        assert any("又等了" in str(item["content"]) for item in notes)
        assert await queue.intercept(_command("/image-wait", generation), "qq:group-a") is True
        await _until(lambda: any("不在观察中" in str(item["content"]) for item in notes), timeout=3)
        assert tool.calls == ["one"]
        assert all("图画好了" not in str(item["content"]) for item in notes)
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_silence_stops_the_same_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    generation, clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        await _nudge(queue, clock, 180)
        await _until(lambda: _rows(queue.root)[0]["prompted"] == 1, timeout=3)
        await _nudge(queue, clock, 300)
        await _until(lambda: _rows(queue.root)[0]["detail"] == "silence", timeout=3)
        assert tool.calls == ["one"]
        assert any("没有选择" in str(item["content"]) for item in notes)
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["one", "two"], timeout=3)
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_confirmed_failure_asks_for_one_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, reject={"one": "内容审核未通过"}, hang={"one"})
    generation, clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    del clock
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: _rows(queue.root)[0]["state"] == "failed", timeout=3)
        await _until(lambda: any("生图失败" in str(item["content"]) for item in notes), timeout=3)
        failure = next(item for item in notes if "生图失败" in str(item["content"]))
        assert "内容审核未通过" in str(failure["content"])
        assert "/image-retry" in str(failure["content"])
        assert "可能还在生成" not in str(failure["content"])
        assert "不能把它说成失败" not in str(failure["content"])
        assert "没有可重试" in queue.choose_retry("qq:group-a", "other-generation")
        assert tool.calls == ["one"]
        assert await queue.intercept(_command("/image-retry", generation), "qq:group-a") is True
        await _until(lambda: tool.calls == ["one", "one"], timeout=3)
        assert any("可能已经计费" in str(item["content"]) for item in notes)
        assert await queue.intercept(_command("/image-retry", generation), "qq:group-a") is True
        await _until(lambda: any("不用重复发送" in str(item["content"]) for item in notes), timeout=3)
        assert tool.calls == ["one", "one"]
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_unknown_retry_is_a_new_call_and_a_duplicate_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, unresolved={"one"}, hang={"one"})
    generation, _clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: _rows(queue.root)[0]["state"] == "unknown", timeout=3)
        await _until(lambda: any("不能把它说成失败" in str(item["content"]) for item in notes), timeout=3)
        notice = next(item for item in notes if "不能把它说成失败" in str(item["content"]))
        assert "connection reset" not in str(notice["content"])
        assert "生图失败" not in str(notice["content"])
        assert "/image-wait" not in str(notice["content"])
        assert await queue.intercept(_command("/image-retry", generation), "qq:group-a") is True
        await _until(lambda: tool.calls == ["one", "one"], timeout=3)
        assert any("可能已经计费" in str(item["content"]) for item in notes)
        assert await queue.intercept(_command("/image-retry", generation), "qq:group-a") is True
        await asyncio.sleep(0.2)
        assert tool.calls == ["one", "one"]
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_retry_does_not_adopt_a_late_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, swallow={"one"}, distinct=True)
    generation, _clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        assert await queue.intercept(_command("/image-retry", generation), "qq:group-a") is True
        await _until(lambda: any(item["media"] == ["generated/2.png"] for item in notes), timeout=3)
        assert tool.calls == ["one", "one"]
        tool.release.set()
        await _until(lambda: tool.finished >= 2, timeout=3)
        await asyncio.sleep(0.1)
        assert all("generated/1.png" not in item["media"] for item in notes)
        assert [item["media"] for item in notes if item["media"]] == [["generated/2.png"]]
        assert _rows(queue.root)[0]["state"] == "unknown"
        assert _rows(queue.root)[1]["paths"] == '["generated/2.png"]'
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_cancel_drops_a_late_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, swallow={"one"}, distinct=True)
    generation, _clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        queue.cancel("qq:group-a")
        await _until(lambda: _rows(queue.root)[0]["state"] == "cancelled", timeout=3)
        tool.release.set()
        await _until(lambda: tool.finished >= 1, timeout=3)
        await asyncio.sleep(0.1)
        assert all("图画好了" not in str(item["content"]) for item in notes)
        assert _rows(queue.root)[0]["paths"] == "[]"
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["one", "two"], timeout=3)
        await _until(lambda: any(item["media"] == ["generated/2.png"] for item in notes), timeout=3)
        assert all("generated/1.png" not in item["media"] for item in notes)
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_sessions_do_not_share_a_wait_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    monkeypatch.setattr("nanobot.agent.image_queue.deliver_to_channel", _capture(notes))
    loop = _loop(tmp_path)
    first = _persist(loop, "qq:group-a")
    second = _persist(loop, "qq:group-b")
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    loop.tools.register(tool)
    clock = Clock()
    queue = _watched_queue(loop, tmp_path / "image-queue", clock)
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, first)
        await _until(lambda: tool.calls == ["one"])
        message = _message("two", "m-2", generation=second, chat_id="group-b", sender_id="user-b")
        queue.admit(message)
        record = queue.inputs[message.metadata["_local_image_queue_input"]]
        queue.submit(record, second, tool, {"prompt": "two"}, "new", None)
        with queue.turn(message):
            queue.finish(message, True)
        await _until(lambda: any(item["chat"] == "group-b" and "图画好了" in str(item["content"]) for item in notes), timeout=3)
        await _nudge(queue, clock, 180)
        await _until(lambda: any("可能还在生成" in str(item["content"]) for item in notes), timeout=3)
        assert all(item["chat"] == "group-a" for item in notes if "可能还在生成" in str(item["content"]))
        assert tool.calls == ["one", "two"]
        assert _rows(queue.root)[0]["state"] == "running"
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_prompt_delivery_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("channel down")

    monkeypatch.setattr("nanobot.agent.image_queue.deliver_to_channel", _fail)
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    loop.tools.register(tool)
    clock = Clock()
    queue = _watched_queue(loop, tmp_path / "image-queue", clock)
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        await _nudge(queue, clock, 180)
        await _until(lambda: _rows(queue.root)[0]["prompt_error"] == 1, timeout=3)
        assert _rows(queue.root)[0]["state"] == "running"
        assert tool.calls == ["one"]
        assert await queue.intercept(_command("/image-wait", generation), "qq:group-a") is True
        await asyncio.sleep(0.2)
        assert tool.calls == ["one"]
        assert _rows(queue.root)[0]["extensions"] == 1
        assert _rows(queue.root)[0]["prompt_error"] == 1
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_restart_does_not_pretend_waiting_can_recover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    monkeypatch.setattr("nanobot.agent.image_queue.deliver_to_channel", _capture(notes))
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path)
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    _submit(queue, "one", "m-1", tool, generation)
    frozen = queue._freeze(queue._candidates()[0])
    assert frozen is not None
    await queue.close()
    restarted = _open_queue(loop, queue.root)
    pump = asyncio.create_task(restarted.run())
    try:
        await _until(lambda: any("没有查询接口" in str(item["content"]) for item in notes), timeout=3)
        assert tool.calls == []
        assert await restarted.intercept(_command("/image-wait", generation), "qq:group-a") is True
        await _until(lambda: any("不能取回" in str(item["content"]) for item in notes), timeout=3)
        assert tool.calls == []
        assert await restarted.intercept(_command("/image-retry", generation), "qq:group-a") is True
        await _until(lambda: tool.calls == ["one"], timeout=3)
        assert any("可能已经计费" in str(item["content"]) for item in notes)
    finally:
        await _stop(restarted, pump)


@pytest.mark.asyncio
async def test_websocket_choice_is_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    monkeypatch.setattr("nanobot.agent.image_queue.deliver_to_channel", _capture(notes))
    loop = _loop(tmp_path)
    generation = _persist(loop, "websocket:web-a")
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    loop.tools.register(tool)
    clock = Clock()
    queue = _watched_queue(loop, tmp_path / "image-queue", clock)
    pump = asyncio.create_task(queue.run())
    try:
        message = _message("one", "m-1", generation=generation, chat_id="web-a")
        message.channel = "websocket"
        queue.admit(message)
        record = queue.inputs[message.metadata["_local_image_queue_input"]]
        queue.submit(record, generation, tool, {"prompt": "one"}, "new", None)
        with queue.turn(message):
            queue.finish(message, True)
        await _until(lambda: tool.calls == ["one"])
        chat = InboundMessage(
            channel="websocket", sender_id="user-a", chat_id="web-a", content="再等五分钟",
            session_generation=generation,
        )
        assert await queue.intercept(chat, "websocket:web-a") is False
        command = InboundMessage(
            channel="websocket", sender_id="user-a", chat_id="web-a", content="/image-wait",
            session_generation=generation, metadata={"webui_turn_id": "turn-1"},
        )
        assert await queue.intercept(command, "websocket:web-a") is True
        await _until(lambda: any("第一次等待" in str(item["content"]) for item in notes), timeout=3)
        assert notes[-1]["channel"] == "websocket"
        assert notes[-1]["chat"] == "web-a"
        assert tool.calls == ["one"]
        del clock
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_completed_image_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[dict[str, object]] = []
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    generation, _clock, queue, pump = _watched(tmp_path, tool, notes, monkeypatch)
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"])
        tool.release.set()
        await _until(lambda: _rows(queue.root)[0]["state"] == "done", timeout=3)
        assert await queue.intercept(_command("/image-retry", generation), "qq:group-a") is True
        await _until(lambda: any("已经完成" in str(item["content"]) for item in notes), timeout=3)
        assert tool.calls == ["one"]
    finally:
        await _stop(queue, pump, tool.release)


@pytest.mark.asyncio
async def test_generate_now_does_not_call_a_timeout_a_failure(tmp_path: Path) -> None:
    from nanobot.agent.tools.image_generation import ImageRequestUnresolvedError
    from nanobot.providers.image_generation import ImageGenerationError

    tool = ImageGenerationTool(
        workspace=tmp_path,
        config=ImageGenerationToolConfig(enabled=True, provider="fake", model="fake-image"),
    )

    class Stub:
        def __init__(self, exc: BaseException) -> None:
            self.exc = exc

        async def generate(self, **_kwargs: object) -> None:
            raise self.exc

    try:
        raise ImageGenerationError("timed out") from TimeoutError("deadline")
    except ImageGenerationError as exc:
        timed_out = exc
    tool._provider_client = lambda: Stub(timed_out)
    with pytest.raises(ImageRequestUnresolvedError):
        await tool.generate_now("draw", None, None, None, 1, propagate_provider_errors=True)

    rejected = ImageGenerationError("api_key=sekret https://files.example/a denied by policy")
    tool._provider_client = lambda: Stub(rejected)
    with pytest.raises(ImageRequestRejectedError) as caught:
        await tool.generate_now("draw", None, None, None, 1, propagate_provider_errors=True)
    assert "sekret" not in caught.value.reason
    assert "http" not in caught.value.reason
    assert "denied by policy" in caught.value.reason

    plain = await tool.generate_now("draw", None, None, None, 1)
    assert str(plain).startswith("Error:")
