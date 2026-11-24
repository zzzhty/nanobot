from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
from nanobot.providers.image_generation import GeneratedImageResponse

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

    result = await tool.execute(queue_mode="wait")

    assert calls == []
    assert "interactive" in result


@pytest.mark.asyncio
async def test_agent_wait_keeps_the_same_image_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal reply reaches the model, which calls queue_mode=wait on the same provider call."""
    set_config_path(tmp_path / "config.json")

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

    monkeypatch.setattr(
        "nanobot.agent.tools.image_generation.get_image_gen_provider",
        lambda name, connection=None: HangingImageClient if name == "openrouter" else None,
    )
    real_init = ImageQueue.__init__

    def fast_init(self: ImageQueue, loop: AgentLoop, root: Path | None = None, **kwargs: Any) -> None:
        kwargs["debounce"] = 0
        kwargs["provider_timeout"] = 0.2
        real_init(self, loop, root, **kwargs)

    monkeypatch.setattr(ImageQueue, "__init__", fast_init)

    issued: list[dict[str, Any]] = []
    wait_requests: list[str] = []

    async def chat_stream_with_retry(*, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        last_user = ""
        for item in messages:
            if item.get("role") == "user":
                last_user = _message_text(item.get("content"))
        last = messages[-1] if messages else {}
        if "再等" in last_user:
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
            return LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(id=f"wait-{len(issued)}", name="generate_image", arguments=arguments)],
            )
        if last.get("role") == "tool":
            return LLMResponse(content="已经排上了。", finish_reason="stop")
        arguments = {"prompt": "a small icon"}
        issued.append(arguments)
        return LLMResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[ToolCallRequest(id="draw-1", name="generate_image", arguments=arguments)],
        )

    provider = MagicMock()
    provider.aclose = AsyncMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 4096
    provider.chat_stream_with_retry = chat_stream_with_retry
    config = Config()
    config.tools.image_generation.enabled = True
    config.tools.image_generation.persistent_queue = True
    config.providers.openrouter.api_key = "sk-or-test"
    tools = ToolRegistry()
    register_image_generation_tool(config, tools, workspace=tmp_path)
    notes: list[tuple[str, bool]] = []
    bus = MessageBus()
    bus.outbound_recorder = lambda msg: notes.append((str(msg.content or ""), bool(msg.media)))
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=config.tools,
        tool_registry=tools,
    )

    async def until(check: Any, timeout: float = 10) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if check():
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(
                    f"timed out; notes={notes!r}; calls={HangingImageClient.calls}; issued={issued!r}"
                )
            await asyncio.sleep(0.05)

    def queue_status() -> dict[str, Any] | None:
        queue = loop._image_queue
        header = loop.sessions.state.peek_header("websocket:web-a")
        if queue is None or header is None:
            return None
        return queue.status("websocket:web-a", header.generation)

    run_task = asyncio.create_task(loop.run())
    try:
        await until(lambda: loop._image_queue is not None)
        await bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="user-a",
            chat_id="web-a",
            content="画一张小图",
        ))
        await until(lambda: any("可能还在生成" in text for text, _media in notes))
        status = queue_status()
        assert status is not None
        assert status["can_wait"] is True
        assert HangingImageClient.calls == 1
        assert HangingImageClient.prompts == ["a small icon"]

        await bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="user-a",
            chat_id="web-a",
            content="再等五分钟",
        ))
        await until(lambda: any(text == "好，我继续等这一次。" for text, _media in notes))
        status = queue_status()
        assert status is not None
        assert status["can_wait"] is True
        running = [job for job in status["jobs"] if job["state"] == "running"]
        assert len(running) == 1
        assert running[0]["extensions"] == 1
        assert HangingImageClient.calls == 1

        await bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="user-a",
            chat_id="web-a",
            content="再等一下，不要重新画",
        ))
        await until(lambda: any(text == "还是这一次，没有重新生图。" for text, _media in notes))
        assert HangingImageClient.calls == 1
        assert issued == [
            {"prompt": "a small icon"},
            {"queue_mode": "wait"},
            {"queue_mode": "wait"},
        ]
        assert wait_requests
        assert "再等五分钟" in wait_requests[0]
        assert "queue_mode=wait" in wait_requests[0]
        assert "running" in wait_requests[0]
        assert not wait_requests[0].lstrip().startswith("/")

        HangingImageClient.release.set()
        await until(lambda: sum("图画好了" in text for text, _media in notes) == 1)
        pictures = [(text, media) for text, media in notes if "图画好了" in text]
        assert pictures == [("图画好了。", True)]
        assert HangingImageClient.calls == 1
    finally:
        HangingImageClient.release.set()
        loop.stop()
        await asyncio.wait_for(run_task, timeout=5)
