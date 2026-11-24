"""Persistent image queue state machine, without a provider or QQ connection."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.image_queue import TOKEN_KEY, ImageQueue
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.image_generation import ImageGenerationTool
from nanobot.agent.tools.image_generation_config import ImageGenerationToolConfig
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


def _loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )


def _queue(loop: AgentLoop, root: Path) -> ImageQueue:
    return ImageQueue(loop, root, clock=lambda: 1_000.0, debounce=0)


def _message(text: str, message_id: str, *, command: str | None = None) -> InboundMessage:
    raw = command if command is not None else text
    return InboundMessage(
        channel="qq",
        sender_id="user-a",
        chat_id="group-a",
        content=text,
        session_generation="generation-a",
        metadata={"message_id": message_id, "raw_content": raw, "chat_type": "group"},
    )


def _tool(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(provider="fake", model="fake-image", max_images_per_turn=4),
        workspace=tmp_path,
    )


def _states(root: Path) -> list[tuple[str, str]]:
    db = sqlite3.connect(root / "jobs.sqlite3")
    try:
        rows = db.execute(
            "SELECT json_extract(params, '$.prompt'), state FROM jobs ORDER BY rowid"
        ).fetchall()
    finally:
        db.close()
    return [(None if prompt is None else str(prompt), str(state)) for prompt, state in rows]


@pytest.fixture
def queue(tmp_path: Path):
    loop = _loop(tmp_path)
    image_queue = _queue(loop, tmp_path / "image-queue")
    try:
        yield image_queue
    finally:
        if image_queue._owner.is_locked:
            image_queue._owner.release()


def test_queue_is_fifo_and_commands_are_not_admitted(queue: ImageQueue, tmp_path: Path) -> None:
    first = _message("draw one", "m-1")
    second = _message("draw two", "m-2")
    stop = _message("context\n/stop", "m-stop", command="/stop")
    resume = _message("/image-resume", "m-resume", command="/image-resume")
    queue.admit(stop)
    queue.admit(first)
    assert TOKEN_KEY not in stop.metadata
    record = queue.inputs[first.metadata[TOKEN_KEY]]
    queue.submit(record, "generation-a", _tool(tmp_path), {"prompt": "one"}, "new", None)
    with queue.turn(first):
        queue.finish(first, True)
    queue.admit(second)
    second_record = queue.inputs[second.metadata[TOKEN_KEY]]
    queue.submit(second_record, "generation-a", _tool(tmp_path), {"prompt": "two"}, "new", None)
    with queue.turn(second):
        queue.finish(second, True)

    prompts = [json.loads(str(row["params"]))["prompt"] for row in queue._candidates()]
    assert prompts == ["one"]
    assert _states(queue.root) == [("one", "draft"), ("two", "draft")]
    queue.admit(resume)
    assert TOKEN_KEY in resume.metadata
    assert queue._candidates() == []
    with queue.turn(resume):
        queue.finish(resume, True)
    assert [json.loads(str(row["params"]))["prompt"] for row in queue._candidates()] == ["one"]


@pytest.mark.asyncio
async def test_cancel_drops_drafts_and_close_keeps_unknown_without_retry(
    queue: ImageQueue,
    tmp_path: Path,
) -> None:
    message = _message("draw one", "m-1")
    queue.admit(message)
    record = queue.inputs[message.metadata[TOKEN_KEY]]
    queue.submit(record, "generation-a", _tool(tmp_path), {"prompt": "one"}, "new", None)
    with queue.turn(message):
        queue.finish(message, True)
    assert queue.cancel(message.session_key) == 1
    assert _states(queue.root) == [(None, "cancelled")]

    another = _message("draw two", "m-2")
    queue.admit(another)
    queued = queue.inputs[another.metadata[TOKEN_KEY]]
    queue.submit(queued, "generation-a", _tool(tmp_path), {"prompt": "two"}, "new", None)
    with queue.turn(another):
        queue.finish(another, True)
    frozen = queue._freeze(queue._candidates()[0])
    assert frozen is not None
    await queue.close()

    assert _states(queue.root) == [(None, "cancelled"), ("two", "unknown")]
    restarted = _queue(queue.loop, queue.root)
    try:
        assert _states(restarted.root) == [(None, "cancelled"), ("two", "unknown")]
        assert [str(row["state"]) for row in restarted._candidates()] == ["unknown"]
    finally:
        await restarted.close()


def test_queue_file_is_private_and_symlink_root_is_rejected(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    root = tmp_path / "image-queue"
    queue = _queue(loop, root)
    try:
        assert (root / "jobs.sqlite3").stat().st_mode & 0o777 == 0o600
    finally:
        queue._owner.release()
    link = tmp_path / "linked-queue"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        _queue(loop, link)


@pytest.mark.asyncio
async def test_enabled_tool_queues_instead_of_calling_the_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ImageGenerationTool(
        workspace=tmp_path,
        config=ImageGenerationToolConfig(persistent_queue=True, provider="fake", model="fake-image"),
    )
    called = False

    async def _generate(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return "{}"

    tool.generate_now = _generate  # type: ignore[method-assign]
    result = await tool.execute(prompt="draw one")
    disabled = ImageGenerationTool(
        workspace=tmp_path,
        config=ImageGenerationToolConfig(),
    )

    assert called is False
    assert "interactive" in result
    assert "queue_mode" in tool.parameters["properties"]
    assert tool.parameters["required"] == []
    assert disabled.parameters["required"] == ["prompt"]
    monkeypatch.delenv("NANOBOT_IMAGE_QUEUE", raising=False)


@pytest.mark.asyncio
async def test_bus_admit_hook_runs_before_the_inbound_queue() -> None:
    bus = MessageBus()
    seen: list[str] = []
    bus.image_queue_admit = lambda message: seen.append(message.chat_id)
    await bus.publish_inbound(
        InboundMessage(channel="qq", sender_id="user-a", chat_id="group-a", content="hi")
    )
    queued = await bus.consume_inbound()
    assert seen == ["group-a"]
    assert queued.chat_id == "group-a"
