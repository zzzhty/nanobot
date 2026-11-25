"""QQ group authorization, routing, and context live in the channel source."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.qq.group_context import GroupContextLog
from nanobot.channels.qq.group_events import (
    MentionState,
    consume_plain_group_message,
    decide_group_event,
    install_group_message_parser,
    mark_plain_group_message,
    parser_installed,
)
from nanobot.channels.qq.runtime import QQChannel, QQConfig
from nanobot.command.router import command_text
from nanobot.config import loader as config_loader


@pytest.fixture
def isolated_config(tmp_path: Path):
    previous = config_loader._current_config_path
    config_loader.set_config_path(tmp_path / "config.json")
    yield
    config_loader._current_config_path = previous


def _group(chat_id: str, user_id: str, content: str, message_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        content=content,
        group_openid=chat_id,
        author=SimpleNamespace(member_openid=user_id),
        attachments=[],
    )


def _direct(user_id: str, content: str, message_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        content=content,
        author=SimpleNamespace(user_openid=user_id),
        attachments=[],
    )


def _channel(tmp_path: Path, **overrides: object) -> QQChannel:
    config = {
        "app_id": "app",
        "secret": "secret",
        "ack_message": "",
        "group_context_dir": str(tmp_path / "group-context"),
    }
    config.update(overrides)
    return QQChannel(QQConfig(**config), MessageBus())  # type: ignore[arg-type]


class _Api:
    def __init__(self, *, fail_with_id: bool = False) -> None:
        self.group_calls: list[dict[str, object]] = []
        self.c2c_calls: list[dict[str, object]] = []
        self.fail_with_id = fail_with_id

    async def post_group_message(self, **kwargs: object) -> None:
        self.group_calls.append(kwargs)
        if self.fail_with_id and kwargs.get("msg_id"):
            raise RuntimeError("code 40034031 msgid expired")

    async def post_c2c_message(self, **kwargs: object) -> None:
        self.c2c_calls.append(kwargs)
        if self.fail_with_id and kwargs.get("msg_id"):
            raise RuntimeError("reply window expired")


def _client(api: _Api) -> SimpleNamespace:
    return SimpleNamespace(api=api)


@pytest.mark.asyncio
async def test_group_authorization_is_isolated_from_direct_messages(
    isolated_config: None,
    tmp_path: Path,
) -> None:
    channel = _channel(tmp_path, group_allow_from=["group-a"])
    authorizations: list[str | None] = []
    original = channel._handle_message

    async def _spy(**kwargs: object) -> bool:
        authorizations.append(kwargs.get("authorization_id"))  # type: ignore[arg-type]
        return await original(**kwargs)  # type: ignore[arg-type]

    channel._handle_message = _spy  # type: ignore[method-assign]

    await channel._on_message(_group("group-a", "user-a", "hello", "m-group"), is_group=True)
    await channel._on_message(_group("group-b", "user-b", "nope", "m-other"), is_group=True)
    allowed = await channel.bus.consume_inbound()

    assert allowed.chat_id == "group-a"
    assert allowed.sender_id == "user-a"
    assert allowed.metadata["chat_type"] == "group"
    assert authorizations == ["group-a"]
    assert channel.bus.inbound_size == 0

    listed_sender = _channel(tmp_path / "sender", allow_from=["user-a"])
    await listed_sender._on_message(_group("group-b", "user-a", "hello", "m-sender"), is_group=True)
    sender_msg = await listed_sender.bus.consume_inbound()
    assert sender_msg.content == "hello"
    assert sender_msg.metadata["raw_content"] == "hello"


@pytest.mark.asyncio
async def test_group_wildcard_does_not_authorize_direct_messages(
    isolated_config: None,
    tmp_path: Path,
) -> None:
    channel = _channel(tmp_path, group_allow_from=["*"])
    channel._client = _client(_Api())

    await channel._on_message(_group("group-a", "user-a", "group line", "m-wild"), is_group=True)
    await channel._on_message(_direct("user-b", "direct line", "m-dm"), is_group=False)

    group_msg = await channel.bus.consume_inbound()
    assert group_msg.chat_id == "group-a"
    assert channel.bus.inbound_size == 0
    assert channel._client.api.c2c_calls


@pytest.mark.asyncio
async def test_empty_mention_uses_placeholder_and_plain_group_text_does_not(
    isolated_config: None,
    tmp_path: Path,
) -> None:
    channel = _channel(tmp_path, allow_from=["user-a"])
    await channel._on_message(_group("group-a", "user-a", "   ", "m-empty"), is_group=True)
    mentioned = await channel.bus.consume_inbound()
    assert mentioned.content == "[有人只 @ 了你，没有写其他文字]"
    assert mentioned.metadata["raw_content"] == mentioned.content

    mark_plain_group_message("m-plain")
    await channel._on_message(_group("group-a", "user-a", "", "m-plain"), is_group=True)
    assert channel.bus.inbound_size == 0
    assert consume_plain_group_message("m-plain") is False


@pytest.mark.asyncio
async def test_group_context_stays_in_one_group_and_commands_stay_raw(
    isolated_config: None,
    tmp_path: Path,
) -> None:
    channel = _channel(tmp_path, group_allow_from=["group-a", "group-b"])
    await channel._on_message(_group("group-a", "user-a", "earlier", "m-earlier"), is_group=True)
    await channel.bus.consume_inbound()
    await channel._on_message(_group("group-a", "user-b", "what was that", "m-follow"), is_group=True)
    follow = await channel.bus.consume_inbound()
    assert "earlier" in follow.content
    assert follow.content.startswith("[最近群消息")
    assert follow.metadata["raw_content"] == "what was that"

    await channel._on_message(_group("group-b", "user-a", "other room", "m-other-room"), is_group=True)
    other = await channel.bus.consume_inbound()
    assert "earlier" not in other.content

    await channel._on_message(_group("group-a", "user-a", "/new", "m-new"), is_group=True)
    command = await channel.bus.consume_inbound()
    assert command.content == "/new"
    assert command_text(command) == "/new"


@pytest.mark.asyncio
async def test_outbound_group_route_strips_prefix_and_retries_expired_msgid(
    isolated_config: None,
    tmp_path: Path,
) -> None:
    channel = _channel(tmp_path, group_allow_from=["group-a"])
    api = _Api()
    channel._client = _client(api)
    await channel.send(OutboundMessage(channel="qq", chat_id="qq:group-a", content="later"))
    assert api.group_calls[0]["group_openid"] == "group-a"
    assert not api.c2c_calls

    expired = _Api(fail_with_id=True)
    channel._client = _client(expired)
    channel._chat_type_cache["group-a"] = "group"
    await channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="group-a",
            content="again",
            metadata={"message_id": "m-old"},
        )
    )
    assert expired.group_calls[0]["msg_id"] == "m-old"
    assert expired.group_calls[1]["msg_id"] is None


@pytest.mark.asyncio
async def test_send_failure_log_has_route_and_not_the_chat_id(
    isolated_config: None,
    tmp_path: Path,
) -> None:
    from loguru import logger

    channel = _channel(tmp_path)
    channel._client = _client(_Api())

    async def _boom(**_kwargs: object) -> None:
        raise RuntimeError("transport failed")

    channel._client.api.post_c2c_message = _boom
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        with pytest.raises(RuntimeError):
            await channel.send(OutboundMessage(channel="qq", chat_id="user-a", content="hello"))
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    assert "user-a" not in rendered
    assert "route=c2c" in rendered
    assert "RuntimeError" in rendered


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_group_context_uses_private_permissions(tmp_path: Path) -> None:
    target = tmp_path / "private"
    log = GroupContextLog(target)
    assert (target.stat().st_mode & 0o777) == 0o700
    assert (log.path.stat().st_mode & 0o777) == 0o600


def test_group_context_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "private"
    target.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        pytest.skip("Windows runner cannot create symlinks")
    with pytest.raises(OSError, match="QQ group context directory must not be a symlink"):
        GroupContextLog(link)


def test_mention_gate_strips_the_bot_and_drops_other_group_text() -> None:
    mentioned = decide_group_event(
        {
            "id": "event-at",
            "d": {
                "id": "msg-at",
                "content": "<@!bot-a> hello",
                "group_openid": "group-a",
                "author": {"member_openid": "user-a"},
                "mentions": [{"id": "bot-a", "is_you": True}],
            },
        },
        state=MentionState(),
        robot_id="bot-a",
        require_mention=True,
    )
    plain = decide_group_event(
        {
            "id": "event-plain",
            "d": {
                "id": "msg-plain",
                "content": "hello",
                "group_openid": "group-a",
                "author": {"member_openid": "user-b"},
            },
        },
        state=MentionState(),
        robot_id="bot-a",
        require_mention=True,
    )

    assert mentioned.dispatch is True
    assert mentioned.content == "hello"
    assert mentioned.message_id == "msg-at"
    assert plain.dispatch is False
    assert plain.message_id == "msg-plain"


def test_unmentioned_message_uses_the_body_id_not_the_event_id() -> None:
    decision = decide_group_event(
        {
            "id": "event-1",
            "d": {
                "id": "msg-1",
                "content": "",
                "group_openid": "group-a",
                "author": {"member_openid": "user-a"},
            },
        },
        state=MentionState(),
        robot_id="bot-a",
        require_mention=False,
    )

    assert decision.dispatch is True
    assert decision.at_bot is False
    assert decision.message_id == "msg-1"
    mark_plain_group_message(decision.message_id)
    assert consume_plain_group_message("msg-1") is True
    assert consume_plain_group_message("event-1") is False


def test_group_parser_is_registered_in_process_without_editing_qq_botpy() -> None:
    pytest.importorskip("botpy")
    from botpy.connection import ConnectionState

    library_source = inspect.getsource(ConnectionState)
    assert "def parse_group_message_create" not in library_source
    installed = install_group_message_parser(lambda _decision: None, require_mention=lambda: True)
    assert installed is True
    assert parser_installed() is True
    discovered = {
        name
        for name, _member in inspect.getmembers(ConnectionState)
        if name.startswith("parse_")
    }
    assert "parse_group_message_create" in discovered
