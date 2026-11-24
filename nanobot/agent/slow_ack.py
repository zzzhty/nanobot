"""Optional delayed QQ group receipt while a turn is still running."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import cast

from nanobot.bus.events import InboundMessage, OutboundMessage


def plan_slow_ack(
    msg: InboundMessage,
    *,
    previous: str = "",
    choose: Callable[[Sequence[str]], str] = random.choice,
) -> tuple[float, str] | None:
    """Return a delay and text when this group turn should send a late receipt."""
    if msg.channel != "qq":
        return None
    meta = msg.metadata or {}
    if str(meta.get("chat_type") or "") != "group":
        return None
    try:
        delay = float(meta.get("slow_ack_seconds") or 0)
    except (TypeError, ValueError):
        return None
    if delay <= 0:
        return None
    single = str(meta.get("slow_ack_text") or "").strip()
    if single:
        return delay, single
    raw_texts = meta.get("slow_ack_texts")
    if not isinstance(raw_texts, list):
        return None
    pool: list[str] = []
    for item in cast(list[object], raw_texts):
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            pool.append(text)
    if not pool:
        return None
    candidates = [text for text in pool if text != previous] or pool
    return delay, choose(candidates)


class QQSlowAck:
    """Schedule one receipt and cancel it when the turn finishes first."""

    def __init__(self) -> None:
        self._last = ""

    def start(
        self,
        msg: InboundMessage,
        publish: Callable[[OutboundMessage], Awaitable[None]],
    ) -> asyncio.Task[None] | None:
        plan = plan_slow_ack(msg, previous=self._last)
        if plan is None:
            return None
        delay, text = plan
        self._last = text

        async def _send() -> None:
            try:
                await asyncio.sleep(delay)
                await publish(OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=text))
            except asyncio.CancelledError:
                raise
            except Exception:
                return

        task = asyncio.create_task(_send())
        task.add_done_callback(_consume_task_error)
        return task


def _consume_task_error(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    with suppress(Exception):
        task.exception()
