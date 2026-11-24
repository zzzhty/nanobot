"""Record a channel delivery, then queue it for the channel adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager


async def deliver_to_channel(
    msg: OutboundMessage,
    *,
    bus: MessageBus,
    session_manager: SessionManager,
    record: bool = False,
    session_key: str | None = None,
    session_generation: str | None = None,
) -> None:
    """Persist an explicit delivery and hand the same message to the channel."""
    metadata = dict(msg.metadata or {})
    record = record or bool(metadata.pop("_record_channel_delivery", False))
    raw_key = metadata.pop("_session_key", None)
    raw_generation = metadata.pop("_session_generation", None)
    expected_generation = (
        session_generation
        if session_generation is not None
        else raw_generation if isinstance(raw_generation, str) else None
    )
    key = session_key or (raw_key if isinstance(raw_key, str) else None)
    if key is None:
        key = f"{msg.channel}:{msg.chat_id}"
    if msg.channel == "websocket":
        from nanobot.agent.turn_delivery import TurnRoute
        from nanobot.session.display_history import HISTORY_GENERATION_METADATA_KEY
        from nanobot.session.webui_turns import WebuiTurnRoutePolicy
        from nanobot.webui.session_identity import webui_session_key

        route = await WebuiTurnRoutePolicy(session_manager).admit(
            TurnRoute(msg.channel, msg.chat_id, metadata), key, expected_generation,
        )
        metadata = route.metadata
        if key == webui_session_key(msg.chat_id):
            expected_generation = cast(str, metadata[HISTORY_GENERATION_METADATA_KEY])
    if metadata != (msg.metadata or {}):
        msg = replace(msg, metadata=metadata)
    if record and msg.channel != "cli" and msg.content.strip():
        if expected_generation is None:
            target = await session_manager.state.get_header(key)
            expected_generation = target.generation
        extra: dict[str, Any] = {"_channel_delivery": True}
        if msg.media:
            extra["media"] = list(msg.media)
        await session_manager.state.record_delivery(
            key, msg.content, extra, expected_generation=expected_generation,
        )
    await bus.publish_outbound(msg)
