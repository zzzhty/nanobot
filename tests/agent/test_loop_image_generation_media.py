from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nanobot.agent.image_queue import ImageQueue
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.image_generation import register_image_generation_tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import set_config_path
from nanobot.config.schema import Config
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.providers.image_generation import GeneratedImageResponse, ImageGenerationError

PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeImageClient:
    def __init__(self, **kwargs: Any) -> None:
        pass

    async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
        return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})


@pytest.mark.asyncio
async def test_generate_image_does_not_attach_media_to_final_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_config_path(tmp_path / "config.json")
    monkeypatch.setattr(
        "nanobot.agent.tools.image_generation.get_image_gen_provider",
        lambda name: FakeImageClient if name == "openrouter" else None,
    )
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 4096
    provider.chat_stream_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id="call_img",
                        name="generate_image",
                        arguments={"prompt": "draw a tiny icon"},
                    )
                ],
            ),
            LLMResponse(content="Done", finish_reason="stop"),
        ]
    )
    config = Config()
    config.tools.image_generation.enabled = True
    config.providers.openrouter.api_key = "sk-or-test"
    tools = ToolRegistry()
    register_image_generation_tool(config, tools, workspace=tmp_path)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=config.tools,
        tool_registry=tools,
    )

    result = await loop._process_message(
        InboundMessage(
            channel="websocket",
            sender_id="user",
            chat_id="chat-image",
            content="draw an icon",
        )
    )

    assert result is not None
    assert result.content == "Done"
    assert result.media == []


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content or "")


@pytest.mark.asyncio
async def test_wait_mode_without_a_turn_does_not_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class CountingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
            calls.append(str(kwargs.get("prompt") or ""))
            return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})

    monkeypatch.setattr(
        "nanobot.agent.tools.image_generation.get_image_gen_provider",
        lambda name, connection=None: CountingClient if name == "openrouter" else None,
    )
    config = Config()
    config.tools.image_generation.enabled = True
    config.tools.image_generation.persistent_queue = True
    config.providers.openrouter.api_key = "sk-or-test"
    tools = ToolRegistry()
    register_image_generation_tool(config, tools, workspace=tmp_path)
    tool = tools.get("generate_image")
    assert tool is not None

    for mode in ("wait", "retry"):
        result = await tool.execute(queue_mode=mode)
        assert calls == []
        assert "interactive" in result


class _Conversation:
    def __init__(self, loop: AgentLoop, bus: MessageBus, notes: list[tuple[str, tuple[str, ...]]]) -> None:
        self.loop = loop
        self.bus = bus
        self.notes = notes

    async def until(self, check: Callable[[], bool], detail: Callable[[], str], timeout: float = 10) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if check():
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"timed out; notes={self.notes!r}; {detail()}")
            await asyncio.sleep(0.05)

    async def say(self, content: str) -> None:
        await self.bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="user-a",
            chat_id="web-a",
            content=content,
        ))

    def status(self) -> dict[str, Any] | None:
        queue = self.loop._image_queue
        header = self.loop.sessions.state.peek_header("websocket:web-a")
        if queue is None or header is None:
            return None
        return queue.status("websocket:web-a", header.generation)

    def jobs(self) -> list[sqlite3.Row]:
        queue = self.loop._image_queue
        assert queue is not None
        db = sqlite3.connect(queue.root / "jobs.sqlite3")
        db.row_factory = sqlite3.Row
        try:
            return list(db.execute(
                "SELECT state, paths, superseded, detail FROM jobs ORDER BY rowid"
            ))
        finally:
            db.close()


def _open_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client_cls: type,
    chat: Callable[..., Awaitable[LLMResponse]],
) -> tuple[_Conversation, asyncio.Task[None]]:
    set_config_path(tmp_path / "config.json")
    monkeypatch.setattr(
        "nanobot.agent.tools.image_generation.get_image_gen_provider",
        lambda name, connection=None: client_cls if name == "openrouter" else None,
    )
    real_init = ImageQueue.__init__

    def fast_init(self: ImageQueue, loop: AgentLoop, root: Path | None = None, **kwargs: Any) -> None:
        kwargs["debounce"] = 0
        kwargs["provider_timeout"] = 0.2
        real_init(self, loop, root, **kwargs)

    monkeypatch.setattr(ImageQueue, "__init__", fast_init)
    provider = MagicMock()
    provider.aclose = AsyncMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 4096
    provider.chat_stream_with_retry = chat
    config = Config()
    config.tools.image_generation.enabled = True
    config.tools.image_generation.persistent_queue = True
    config.providers.openrouter.api_key = "sk-or-test"
    tools = ToolRegistry()
    register_image_generation_tool(config, tools, workspace=tmp_path)
    notes: list[tuple[str, tuple[str, ...]]] = []
    bus = MessageBus()
    bus.outbound_recorder = lambda msg: notes.append((str(msg.content or ""), tuple(msg.media or ())))
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=config.tools,
        tool_registry=tools,
    )
    return _Conversation(loop, bus, notes), asyncio.create_task(loop.run())


def _last_user(messages: list[dict[str, Any]]) -> str:
    text = ""
    for item in messages:
        if item.get("role") == "user":
            text = _message_text(item.get("content"))
    return text


def _tool_call(call_id: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[ToolCallRequest(id=call_id, name="generate_image", arguments=arguments)],
    )


async def _stop_conversation(conversation: _Conversation, run_task: asyncio.Task[None]) -> None:
    conversation.loop.stop()
    await asyncio.wait_for(run_task, timeout=5)


@pytest.mark.asyncio
async def test_agent_wait_keeps_the_same_image_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal reply reaches the model, which calls queue_mode=wait on the same provider call."""

    class HangingImageClient:
        calls = 0
        prompts: list[str] = []
        release = asyncio.Event()

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
            type(self).calls += 1
            type(self).prompts.append(str(kwargs.get("prompt") or ""))
            await type(self).release.wait()
            return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})

    issued: list[dict[str, Any]] = []
    wait_requests: list[str] = []

    async def chat_stream_with_retry(*, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        last_user = _last_user(messages)
        last = messages[-1] if messages else {}
        if "继续等" in last_user:
            wait_requests.append(last_user)
            if last.get("role") == "tool":
                body = _message_text(last.get("content"))
                if "已经在多等" in body:
                    return LLMResponse(content="还是这一次，没有重新生图。", finish_reason="stop")
                if "继续观察" in body:
                    return LLMResponse(content="好，我继续等这一次。", finish_reason="stop")
                return LLMResponse(content=body, finish_reason="stop")
            arguments = {"queue_mode": "wait"}
            issued.append(arguments)
            return _tool_call(f"wait-{len(issued)}", arguments)
        if last.get("role") == "tool":
            return LLMResponse(content="已经排上了。", finish_reason="stop")
        arguments = {"prompt": "a small icon"}
        issued.append(arguments)
        return _tool_call("draw-1", arguments)

    conversation, run_task = _open_conversation(tmp_path, monkeypatch, HangingImageClient, chat_stream_with_retry)
    try:
        await conversation.until(lambda: conversation.loop._image_queue is not None, lambda: "")
        await conversation.say("画一张小图")
        await conversation.until(
            lambda: any("可能还在生成" in text for text, _media in conversation.notes),
            lambda: f"calls={HangingImageClient.calls}",
        )
        status = conversation.status()
        assert status is not None
        assert status["can_wait"] is True
        assert HangingImageClient.calls == 1
        assert HangingImageClient.prompts == ["a small icon"]

        await conversation.say("继续等一下")
        await conversation.until(
            lambda: any(text == "好，我继续等这一次。" for text, _media in conversation.notes),
            lambda: f"calls={HangingImageClient.calls}; issued={issued!r}",
        )
        status = conversation.status()
        assert status is not None
        assert status["can_wait"] is True
        running = [job for job in status["jobs"] if job["state"] == "running"]
        assert len(running) == 1
        assert running[0]["extensions"] == 1
        assert HangingImageClient.calls == 1

        await conversation.say("还是继续等，不要重新画")
        await conversation.until(
            lambda: any(text == "还是这一次，没有重新生图。" for text, _media in conversation.notes),
            lambda: f"issued={issued!r}",
        )
        assert HangingImageClient.calls == 1
        assert issued == [
            {"prompt": "a small icon"},
            {"queue_mode": "wait"},
            {"queue_mode": "wait"},
        ]
        assert wait_requests
        assert "继续等一下" in wait_requests[0]
        assert "queue_mode=wait" in wait_requests[0]
        assert "running" in wait_requests[0]
        assert not wait_requests[0].lstrip().startswith("/")

        HangingImageClient.release.set()
        await conversation.until(
            lambda: sum("图画好了" in text for text, _media in conversation.notes) == 1,
            lambda: f"calls={HangingImageClient.calls}",
        )
        pictures = [(text, media) for text, media in conversation.notes if "图画好了" in text]
        assert pictures == [("图画好了。", pictures[0][1])]
        assert pictures[0][1]
        assert HangingImageClient.calls == 1
    finally:
        HangingImageClient.release.set()
        await _stop_conversation(conversation, run_task)


@pytest.mark.asyncio
async def test_agent_retry_after_a_confirmed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected image is explained, then a normal reply starts one new attempt."""

    class RejectingClient:
        calls = 0

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
            type(self).calls += 1
            if type(self).calls == 1:
                request = httpx.Request("POST", "https://images.example/generate")
                response = httpx.Response(400, request=request)
                cause = httpx.HTTPStatusError("policy rejected this prompt", request=request, response=response)
                raise ImageGenerationError("policy rejected this prompt") from cause
            return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})

    issued: list[dict[str, Any]] = []
    retry_requests: list[str] = []

    async def chat_stream_with_retry(*, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        last_user = _last_user(messages)
        last = messages[-1] if messages else {}
        if "再试一次" in last_user:
            retry_requests.append(last_user)
            if last.get("role") == "tool":
                return LLMResponse(content="好，重新生成一次。", finish_reason="stop")
            arguments = {"queue_mode": "retry"}
            issued.append(arguments)
            return _tool_call(f"retry-{len(issued)}", arguments)
        if last.get("role") == "tool":
            return LLMResponse(content="已经排上了。", finish_reason="stop")
        arguments = {"prompt": "a small icon"}
        issued.append(arguments)
        return _tool_call("draw-1", arguments)

    conversation, run_task = _open_conversation(tmp_path, monkeypatch, RejectingClient, chat_stream_with_retry)
    try:
        await conversation.until(lambda: conversation.loop._image_queue is not None, lambda: "")
        await conversation.say("画一张小图")
        await conversation.until(
            lambda: any("生图失败" in text and "policy rejected this prompt" in text for text, _media in conversation.notes),
            lambda: f"calls={RejectingClient.calls}; notes={conversation.notes!r}",
        )
        failure = next(text for text, _media in conversation.notes if "生图失败" in text)
        assert "不是超时" in failure
        assert "直接说要重试" in failure
        assert "/image-retry" not in failure
        assert RejectingClient.calls == 1

        await conversation.say("再试一次")
        await conversation.until(
            lambda: any(text == "好，重新生成一次。" for text, _media in conversation.notes),
            lambda: f"calls={RejectingClient.calls}; issued={issued!r}",
        )
        await conversation.until(
            lambda: sum("图画好了" in text for text, _media in conversation.notes) == 1,
            lambda: f"calls={RejectingClient.calls}",
        )
        assert RejectingClient.calls == 2
        assert {"queue_mode": "retry"} in issued
        assert retry_requests
        assert "再试一次" in retry_requests[0]
        assert "failed" in retry_requests[0]
        assert "queue_mode=retry" in retry_requests[0]
        assert not retry_requests[0].lstrip().startswith("/")
        pictures = [text for text, _media in conversation.notes if "图画好了" in text]
        assert pictures == ["图画好了。"]
    finally:
        await _stop_conversation(conversation, run_task)


@pytest.mark.asyncio
async def test_agent_retry_of_an_unknown_call_does_not_adopt_the_late_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry explains the billing risk, ignores a duplicate call, and drops the old image."""

    class LateClient:
        calls = 0
        release = asyncio.Event()
        late_done = asyncio.Event()

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
            type(self).calls += 1
            if type(self).calls == 1:
                while not type(self).release.is_set():
                    try:
                        await type(self).release.wait()
                    except asyncio.CancelledError:
                        continue
                type(self).late_done.set()
                return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})
            return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})

    issued: list[dict[str, Any]] = []
    retry_requests: list[str] = []

    async def chat_stream_with_retry(*, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        last_user = _last_user(messages)
        last = messages[-1] if messages else {}
        if "重新画" in last_user:
            retry_requests.append(last_user)
            if last.get("role") == "tool":
                body = _message_text(last.get("content"))
                if "不用重复发送" in body:
                    return LLMResponse(
                        content="已开始一次新的生图。原来的请求可能已经计费，它的结果不会当作这次的图。",
                        finish_reason="stop",
                    )
                arguments = {"queue_mode": "retry"}
                issued.append(arguments)
                return _tool_call(f"retry-{len(issued)}", arguments)
            arguments = {"queue_mode": "retry"}
            issued.append(arguments)
            return _tool_call(f"retry-{len(issued)}", arguments)
        if last.get("role") == "tool":
            return LLMResponse(content="已经排上了。", finish_reason="stop")
        arguments = {"prompt": "a small icon"}
        issued.append(arguments)
        return _tool_call("draw-1", arguments)

    conversation, run_task = _open_conversation(tmp_path, monkeypatch, LateClient, chat_stream_with_retry)
    try:
        await conversation.until(lambda: conversation.loop._image_queue is not None, lambda: "")
        await conversation.say("画一张小图")
        await conversation.until(lambda: LateClient.calls == 1, lambda: f"calls={LateClient.calls}")
        await conversation.say("重新画一张")
        await conversation.until(
            lambda: any("可能已经计费" in text for text, _media in conversation.notes),
            lambda: f"calls={LateClient.calls}; issued={issued!r}",
        )
        await conversation.until(
            lambda: sum("图画好了" in text for text, _media in conversation.notes) == 1,
            lambda: f"calls={LateClient.calls}; jobs={conversation.jobs()!r}",
        )
        assert LateClient.calls == 2
        assert issued.count({"queue_mode": "retry"}) == 2
        assert retry_requests
        assert "重新画一张" in retry_requests[0]
        assert "queue_mode=retry" in retry_requests[0]
        assert not retry_requests[0].lstrip().startswith("/")
        rows = conversation.jobs()
        assert rows[0]["state"] == "unknown"
        assert int(rows[0]["superseded"]) == 1
        assert rows[0]["paths"] == "[]"
        assert rows[1]["state"] == "done"
        delivered = next(media for text, media in conversation.notes if "图画好了" in text)
        assert json_paths(rows[1]["paths"]) == list(delivered)

        LateClient.release.set()
        await asyncio.wait_for(LateClient.late_done.wait(), timeout=3)
        await asyncio.sleep(0.1)
        assert LateClient.calls == 2
        assert sum("图画好了" in text for text, _media in conversation.notes) == 1
        assert conversation.jobs()[0]["paths"] == "[]"
        assert [media for text, media in conversation.notes if "图画好了" in text] == [delivered]
    finally:
        LateClient.release.set()
        await _stop_conversation(conversation, run_task)


@pytest.mark.asyncio
async def test_agent_unknown_notice_then_one_confirmed_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved call is explained first. The user then asks for one new attempt."""

    class UnresolvedClient:
        calls = 0

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
            type(self).calls += 1
            if type(self).calls == 1:
                raise ImageGenerationError("timed out") from TimeoutError("deadline")
            return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})

    issued: list[dict[str, Any]] = []
    retry_requests: list[str] = []

    async def chat_stream_with_retry(*, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        last_user = _last_user(messages)
        last = messages[-1] if messages else {}
        if "重新来" in last_user:
            retry_requests.append(last_user)
            if last.get("role") == "tool":
                body = _message_text(last.get("content"))
                if "已经完成" in body or "不用重复发送" in body:
                    return LLMResponse(content="没有再开一次。", finish_reason="stop")
                return LLMResponse(content="好，按你的意思新开一次。原来的可能已经计费。", finish_reason="stop")
            arguments = {"queue_mode": "retry"}
            issued.append(arguments)
            return _tool_call(f"retry-{len(issued)}", arguments)
        if last.get("role") == "tool":
            return LLMResponse(content="已经排上了。", finish_reason="stop")
        arguments = {"prompt": "a small icon"}
        issued.append(arguments)
        return _tool_call("draw-1", arguments)

    conversation, run_task = _open_conversation(tmp_path, monkeypatch, UnresolvedClient, chat_stream_with_retry)
    try:
        await conversation.until(lambda: conversation.loop._image_queue is not None, lambda: "")
        await conversation.say("画一张小图")
        await conversation.until(
            lambda: any("不能把它说成失败" in text and "可能已经计费" in text for text, _media in conversation.notes),
            lambda: f"calls={UnresolvedClient.calls}",
        )
        notice = next(text for text, _media in conversation.notes if "不能把它说成失败" in text)
        assert "不能取回" in notice
        assert "直接说要重试" in notice
        assert "/image-retry" not in notice
        assert UnresolvedClient.calls == 1

        await conversation.say("那就重新来")
        await conversation.until(
            lambda: any(text == "好，按你的意思新开一次。原来的可能已经计费。" for text, _media in conversation.notes),
            lambda: f"issued={issued!r}",
        )
        await conversation.until(
            lambda: sum("图画好了" in text for text, _media in conversation.notes) == 1,
            lambda: f"calls={UnresolvedClient.calls}",
        )
        assert UnresolvedClient.calls == 2
        assert retry_requests
        assert "unknown" in retry_requests[0]
        assert "queue_mode=retry" in retry_requests[0]
        assert not retry_requests[0].lstrip().startswith("/")

        await conversation.say("再重新来一次")
        await conversation.until(
            lambda: any(text == "没有再开一次。" for text, _media in conversation.notes),
            lambda: f"calls={UnresolvedClient.calls}; issued={issued!r}",
        )
        assert UnresolvedClient.calls == 2
        assert sum("图画好了" in text for text, _media in conversation.notes) == 1
    finally:
        await _stop_conversation(conversation, run_task)


def json_paths(raw: object) -> list[str]:
    parsed = json.loads(str(raw or "[]"))
    return [str(item) for item in parsed]
