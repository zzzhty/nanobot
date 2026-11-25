"""Unknown QQ events log a redacted payload without editing qq-botpy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from loguru import logger

from nanobot.channels.qq.group_events import install_unknown_event_logger


class _Socket:
    WS_DISPATCH_EVENT = 0

    def __init__(self) -> None:
        self._parser = {"ready": lambda payload: None}
        self.calls: list[object] = []

    async def on_message(self, ws: object, message: object) -> str:
        self.calls.append(message)
        return "original"


def _payload(**extra: object) -> str:
    body = {
        "id": "m1",
        "group_openid": "group-a",
        "content": "hello",
        "author": {"member_openid": "user-a", "secret": "author-secret"},
        "token": "super-secret-token-value",
    }
    body.update(extra)
    return json.dumps({"t": "UNKNOWN_THING", "op": 0, "s": 1, "d": body})


@pytest.mark.asyncio
async def test_unknown_event_log_is_whitelisted_and_idempotent() -> None:
    install_unknown_event_logger(_Socket)
    install_unknown_event_logger(_Socket)
    wrapped = _Socket.on_message
    assert getattr(wrapped, "_nanobot_redacted_event_log", False) is True
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        owner = _Socket()
        assert await owner.on_message(None, _payload()) == "original"
        assert await owner.on_message(None, _payload(content="测" * 2000)) == "original"
        assert await owner.on_message(None, json.dumps({
            "t": "READY", "op": 0, "s": 1, "d": {"token": "super-secret-token-value"},
        })) == "original"
        assert await owner.on_message(None, "not-json") == "original"
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    payloads = [
        line.split("payload (redacted):", 1)[1].strip()
        for line in rendered.splitlines()
        if "payload (redacted):" in line
    ]
    assert len(payloads) == 2
    assert all(len(item) <= 1200 for item in payloads)
    assert "group-a" in payloads[0]
    assert "user-a" in payloads[0]
    assert "super-secret-token-value" not in rendered
    assert "author-secret" not in rendered
    assert owner.calls


def test_install_does_not_edit_botpy_files() -> None:
    try:
        import botpy.gateway as gateway
    except ImportError:
        pytest.skip("qq-botpy is not installed")
    path = Path(gateway.__file__)
    before = path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    install_unknown_event_logger()
    assert path.read_bytes() == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    source = path.read_text(encoding="utf-8")
    # The installer must not have appended itself to the installed file.
    assert source.count("def on_message") == before.decode("utf-8").count("def on_message")
