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

import pytest

from nanobot.agent.image_queue import ImageQueue
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.image_generation import ImageGenerationTool
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

    async def generate_now(
        self,
        prompt: str,
        reference_images: list[str] | None,
        aspect_ratio: str | None,
        image_size: str | None,
        count: int | None,
        *,
        references_are_resolved: bool = False,
    ) -> str:
        del reference_images, aspect_ratio, image_size, count, references_are_resolved
        self.calls.append(prompt)
        if prompt in self.swallow:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue
        elif prompt in self.hang:
            await self.release.wait()
        if prompt in self.fail:
            raise RuntimeError("provider failed")
        return json.dumps({"artifacts": [{"path": "generated/out.png"}]})


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


@pytest.mark.asyncio
async def test_hung_provider_reaches_unknown_and_the_next_draft_runs(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    generation = _persist(loop, "qq:group-a")
    tool = ScriptedImageTool(tmp_path, hang={"one"})
    loop.tools.register(tool)
    queue = _open_queue(loop, tmp_path / "image-queue")
    pump = asyncio.create_task(queue.run())
    try:
        _submit(queue, "one", "m-1", tool, generation)
        await _until(lambda: tool.calls == ["one"] and _states(queue.root) == [("one", "unknown")])
        _submit(queue, "two", "m-2", tool, generation)
        await _until(lambda: tool.calls == ["one", "two"])
        assert _states(queue.root)[0][1] == "unknown"
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
