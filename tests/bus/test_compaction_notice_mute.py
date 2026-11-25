"""NANOBOT_COMPACTION_NOTICE_MUTE defaults to qq and can be cleared."""

from __future__ import annotations

import pytest

from nanobot.bus.notification_delivery import notification_is_deliverable
from nanobot.events import ContextCompactionEvent


def test_compaction_mute_defaults_to_qq_and_empty_clears_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NANOBOT_COMPACTION_NOTICE_MUTE", raising=False)
    assert notification_is_deliverable(
        ContextCompactionEvent, channel="qq", publish_lifecycle=True,
    ) is False
    assert notification_is_deliverable(
        ContextCompactionEvent, channel="telegram", publish_lifecycle=True,
    ) is True

    monkeypatch.setenv("NANOBOT_COMPACTION_NOTICE_MUTE", "")
    assert notification_is_deliverable(
        ContextCompactionEvent, channel="qq", publish_lifecycle=True,
    ) is True

    monkeypatch.setenv("NANOBOT_COMPACTION_NOTICE_MUTE", "telegram, qq")
    assert notification_is_deliverable(
        ContextCompactionEvent, channel="qq", publish_lifecycle=True,
    ) is False
    assert notification_is_deliverable(
        ContextCompactionEvent, channel="telegram", publish_lifecycle=True,
    ) is False
