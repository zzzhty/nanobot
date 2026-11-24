# pyright: reportMissingTypeStubs=false

"""QQ group events that qq-botpy does not register.

qq-botpy discovers ``parse_*`` methods when a connection starts. This module
adds ``parse_group_message_create`` on the connection class in memory. It does
not rewrite installed package files. Messages that do not mention the bot stay
out of the agent; mention tokens are removed before dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from loguru import logger

_MENTION_TOKENS = ("<@!", "<!", "<@")


@dataclass
class MentionState:
    """Remember this bot's mention id after the first event that names it."""

    self_openid: str = ""


@dataclass(frozen=True, slots=True)
class GroupEventDecision:
    """One group payload after the mention gate."""

    dispatch: bool
    content: str
    raw_content: str
    chat_id: str
    sender_id: str
    message_id: str
    at_bot: bool
    attachment_count: int


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    raw = cast(dict[object, object], value)
    return {str(key): item for key, item in raw.items() if isinstance(key, str)}


def _objects(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in cast(list[object], value):
        parsed = _object(item)
        if parsed:
            rows.append(parsed)
    return rows


def decide_group_event(
    payload: Mapping[str, Any],
    *,
    state: MentionState,
    robot_id: str,
    require_mention: bool,
) -> GroupEventDecision:
    """Return whether a group payload should enter the normal mention handler."""
    body = _object(payload.get("d"))
    raw = str(body.get("content") or "")
    rows = _objects(body.get("mentions"))
    attachments = body.get("attachments")
    attachment_count = len(cast(list[object], attachments)) if isinstance(attachments, list) else 0
    author_body = _object(body.get("author"))
    chat_id = str(body.get("group_openid") or "")
    sender_id = str(author_body.get("member_openid") or "")
    message_id = str(payload.get("id") or "")

    own = [str(item.get("id") or "") for item in rows if item.get("is_you")]
    if not state.self_openid:
        for mention_id in own:
            if mention_id:
                state.self_openid = mention_id
                break
    candidates = [item for item in (state.self_openid, robot_id) if item]
    id_hit = any(
        str(item.get("id") or "") == candidate
        for item in rows
        for candidate in candidates
    )
    token_hit = any(
        token + candidate + ">" in raw
        for candidate in candidates
        for token in _MENTION_TOKENS
    )
    at_bot = bool(own or id_hit or token_hit)
    if require_mention and not at_bot:
        return GroupEventDecision(
            False, raw.strip(), raw, chat_id, sender_id, message_id, False, attachment_count,
        )

    strip_ids = own or [
        candidate
        for candidate in candidates
        if any(token + candidate + ">" in raw for token in _MENTION_TOKENS)
    ]
    content = raw
    for mention_id in strip_ids:
        if not mention_id:
            continue
        for token in _MENTION_TOKENS:
            content = content.replace(token + mention_id + ">", "")
    return GroupEventDecision(
        True, content.strip(), raw, chat_id, sender_id, message_id, at_bot, attachment_count,
    )


_handler: Callable[[GroupEventDecision], None] | None = None
_mention_state = MentionState()
_installed = False
_plain_order: list[str] = []
_plain_ids: set[str] = set()


def mark_plain_group_message(message_id: str) -> None:
    """Remember a dispatched group message that did not mention the bot."""
    if not message_id or message_id in _plain_ids:
        return
    _plain_ids.add(message_id)
    _plain_order.append(message_id)
    while len(_plain_order) > 1000:
        _plain_ids.discard(_plain_order.pop(0))


def consume_plain_group_message(message_id: str) -> bool:
    """Return True once for a group message that was not a mention."""
    if message_id not in _plain_ids:
        return False
    _plain_ids.discard(message_id)
    return True


def _require_mention() -> bool:
    return True


def install_group_message_parser(
    handler: Callable[[GroupEventDecision], None],
    *,
    require_mention: Callable[[], bool],
    state: MentionState | None = None,
) -> bool:
    """Register the group parser if this qq-botpy build does not provide one.

    Returns False when qq-botpy is missing or already defines the parser. An
    existing library method is left unchanged.
    """
    global _handler, _require_mention, _mention_state, _installed
    _handler = handler
    _require_mention = require_mention
    if state is not None:
        _mention_state = state
    try:
        from botpy.connection import ConnectionState
    except ImportError:
        return False
    existing = ConnectionState.__dict__.get("parse_group_message_create")
    if existing is not None and existing is not _parse_group_message_create:
        return False
    if existing is None:
        ConnectionState.parse_group_message_create = _parse_group_message_create  # type: ignore[attr-defined]
    _installed = True
    return True


def parser_installed() -> bool:
    """Whether this process added the group parser to the connection class."""
    return _installed


def _parse_group_message_create(self: Any, payload: Mapping[str, Any]) -> None:
    robot = getattr(self, "robot", None)
    robot_id = str(getattr(robot, "id", "") or "")
    decision = decide_group_event(
        payload,
        state=_mention_state,
        robot_id=robot_id,
        require_mention=_require_mention(),
    )
    if not decision.dispatch:
        if _handler is not None:
            _handler(decision)
        logger.debug("QQ group message was not a mention")
        return
    if not decision.at_bot:
        mark_plain_group_message(decision.message_id)
    logger.info("QQ group message routed to the mention handler")
    body = _object(payload.get("d"))
    body["content"] = decision.content
    from botpy.message import GroupMessage

    message = GroupMessage(self.api, payload.get("id"), cast(Any, body))
    self._dispatch("group_at_message_create", message)


_parse_group_message_create.__name__ = "parse_group_message_create"
