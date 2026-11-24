"""QQ reply dedup, presets, and slow receipts are agent behavior, not a site-packages patch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.image_queue import ImageQueue
from nanobot.agent.loop import AgentLoop
from nanobot.agent.slow_ack import plan_slow_ack
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.message import MessageTool, duplicate_final_should_suppress
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.router import command_text
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.manager import Session
from nanobot.session.model_selection import SessionModelPreset


def _loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")


def test_qq_duplicate_final_requires_an_exact_completed_reply() -> None:
    sends = {("qq", "qq:group-a"): ["  hello  "]}
    assert duplicate_final_should_suppress(
        sends,
        channel="qq",
        chat_id="group-a",
        final_content="hello",
        stop_reason="completed",
        had_injections=False,
    )
    assert not duplicate_final_should_suppress(
        sends,
        channel="qq",
        chat_id="group-a",
        final_content="hello there",
        stop_reason="completed",
        had_injections=False,
    )
    assert not duplicate_final_should_suppress(
        sends,
        channel="qq",
        chat_id="group-a",
        final_content="",
        stop_reason="completed",
        had_injections=False,
    )
    assert not duplicate_final_should_suppress(
        sends,
        channel="qq",
        chat_id="group-a",
        final_content="hello",
        stop_reason="error",
        had_injections=False,
    )
    assert duplicate_final_should_suppress(
        {("feishu", "chat-a"): ["different"]},
        channel="feishu",
        chat_id="chat-a",
        final_content=None,
        stop_reason="empty_final_response",
        had_injections=True,
    )


@pytest.mark.asyncio
async def test_message_tool_strips_qq_prefix_and_records_cross_chat(tmp_path: Path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(message: OutboundMessage) -> None:
        sent.append(message)

    tool = MessageTool(send_callback=_send, workspace=tmp_path)
    with request_context(RequestContext(channel="qq", chat_id="group-a")):
        await tool.execute(content="same", channel="qq", chat_id="qq:group-a")
        await tool.execute(content="other", channel="qq", chat_id="qq:group-b")

    assert sent[0].chat_id == "group-a"
    assert "_record_channel_delivery" not in sent[0].metadata
    assert sent[1].chat_id == "group-b"
    assert sent[1].metadata["_record_channel_delivery"] is True


@pytest.mark.asyncio
async def test_qq_exact_final_is_suppressed_and_a_different_final_is_kept(tmp_path: Path) -> None:
    loop = _loop(tmp_path)

    async def _run(final: str) -> OutboundMessage | None:
        tool_call = ToolCallRequest(
            id="call-1",
            name="message",
            arguments={"content": "hello", "channel": "qq", "chat_id": "qq:group-a"},
        )
        loop.provider.chat_stream_with_retry = AsyncMock(
            side_effect=[
                LLMResponse(content="", tool_calls=[tool_call]),
                LLMResponse(content=final, tool_calls=[]),
            ]
        )
        loop.tools.get_definitions = MagicMock(return_value=[])
        return await loop._process_message(
            InboundMessage(
                channel="qq",
                sender_id="user-a",
                chat_id="group-a",
                content="hi",
                metadata={"chat_type": "group", "raw_content": "hi"},
            )
        )

    suppressed = await _run("hello")
    kept = await _run("sent it")
    assert suppressed is None
    assert kept is not None
    assert "sent it" in kept.content


def test_slow_ack_is_group_only_and_queued_images_are_not_injected(tmp_path: Path) -> None:
    group = InboundMessage(
        channel="qq",
        sender_id="user-a",
        chat_id="group-a",
        content="hi",
        metadata={"chat_type": "group", "slow_ack_seconds": 2, "slow_ack_texts": ["收到"]},
    )
    direct = InboundMessage(
        channel="qq",
        sender_id="user-a",
        chat_id="user-a",
        content="hi",
        metadata={"chat_type": "c2c", "slow_ack_seconds": 2, "slow_ack_texts": ["收到"]},
    )
    assert plan_slow_ack(group, choose=lambda pool: pool[0]) == (2.0, "收到")
    assert plan_slow_ack(direct) is None

    loop = _loop(tmp_path)
    image = InboundMessage(
        channel="qq",
        sender_id="user-a",
        chat_id="group-a",
        content="see",
        media=[str(tmp_path / "picture.png")],
        metadata={"message_id": "m-image", "raw_content": "see", "chat_type": "group"},
    )
    text = InboundMessage(channel="qq", sender_id="user-a", chat_id="group-a", content="next")
    assert loop._can_inject_message(image) is True
    queue = ImageQueue(loop, tmp_path / "image-queue", clock=lambda: 1_000.0, debounce=0)
    loop._image_queue = queue
    try:
        queue.admit(image)
        assert queue.tracked(image) is True
        assert loop._can_inject_message(image) is False
        assert loop._can_inject_message(text) is True
    finally:
        if queue._owner.is_locked:
            queue._owner.release()


def test_qq_commands_use_raw_content() -> None:
    message = InboundMessage(
        channel="qq",
        sender_id="user-a",
        chat_id="group-a",
        content="[最近群消息]\n/stop",
        metadata={"raw_content": "/stop"},
    )
    assert command_text(message) == "/stop"


@pytest.mark.asyncio
async def test_new_qq_session_gets_the_configured_preset_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    monkeypatch.setenv("QQ_PRESET_GROUP", "group-preset")
    calls: list[str] = []

    def _resolve(name: str) -> SimpleNamespace:
        calls.append(name)
        return SimpleNamespace(model_preset=name)

    saved: list[dict[str, object]] = []

    async def _save(key: str, updates: dict[str, object], **_kwargs: object) -> None:
        assert key == "qq:group-a"
        saved.append(dict(updates))

    loop.runtime_resolver.resolve_preset = _resolve  # type: ignore[method-assign]
    loop.sessions.state.update_metadata = _save  # type: ignore[method-assign]
    session = Session(key="qq:group-a")
    message = InboundMessage(
        channel="qq",
        sender_id="user-a",
        chat_id="group-a",
        content="hi",
        metadata={"chat_type": "group"},
    )
    await loop._apply_qq_default_session_preset(session, message)
    await loop._apply_qq_default_session_preset(session, message)

    assert calls == ["group-preset"]
    assert SessionModelPreset.from_metadata(session.metadata).base_preset == "group-preset"
    assert saved

    system = InboundMessage(
        channel="qq",
        sender_id="system",
        chat_id="group-a",
        content="hi",
        metadata={"chat_type": "group"},
    )
    fresh = Session(key="qq:group-b")
    await loop._apply_qq_default_session_preset(fresh, system)
    assert fresh.metadata == {}
