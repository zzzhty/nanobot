"""Ambient archive: permissions, context identity, and attachment downloads."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.channels.qq import ambient
from nanobot.channels.qq.group_events import (
    MentionState,
    archive_group_event,
    decide_group_event,
)


def _use_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db = tmp_path / "ambient" / "qq_messages.db"
    monkeypatch.setenv("QQ_AMBIENT_DB", str(db))
    monkeypatch.setenv("QQ_AMBIENT_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("QQ_AMBIENT_SENDER_NAMES", "")
    return db


def _log(chat: str, msg: str, body: str, **kwargs: object) -> None:
    ambient.log_message(
        chat_id=chat,
        chat_type=str(kwargs.pop("chat_type", "group")),
        msg_id=msg,
        sender_id=str(kwargs.pop("sender_id", "SYNTHETIC_USER")),
        content=body,
        at_bot=bool(kwargs.pop("at_bot", False)),
        routed=bool(kwargs.pop("routed", False)),
        attachments=kwargs.pop("attachments", None),
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_new_database_and_sidecars_are_private(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = _use_db(monkeypatch, tmp_path)
    previous = os.umask(0o000)
    try:
        with ambient.connect() as held:
            _log("GROUP_A", "a1", "hello")
            assert held.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
            assert _mode(db.parent) == 0o700
            for path in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
                assert path.is_file(), path
                assert _mode(path) == 0o600
    finally:
        os.umask(previous)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_existing_loose_files_are_tightened_without_dropping_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    db = _use_db(monkeypatch, tmp_path)
    db.parent.mkdir()
    raw = sqlite3.connect(db)
    try:
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute("CREATE TABLE legacy (value TEXT)")
        raw.execute("INSERT INTO legacy VALUES ('preserved')")
        raw.commit()
        for path in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
            os.chmod(path, 0o666)
        os.chmod(db.parent, 0o777)
    finally:
        raw.close()
    with ambient.connect() as conn:
        assert _mode(db.parent) == 0o700
        for path in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
            assert _mode(path) == 0o600
        assert conn.execute("SELECT value FROM legacy").fetchone()[0] == "preserved"


@pytest.mark.skipif(os.name != "posix", reason="symlink checks")
def test_symlink_database_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "unrelated.db"
    target.write_bytes(b"not an ambient database")
    original = _mode(target)
    db = tmp_path / "ambient" / "messages.db"
    db.parent.mkdir()
    db.symlink_to(target)
    monkeypatch.setenv("QQ_AMBIENT_DB", str(db))
    monkeypatch.setenv("QQ_AMBIENT_SENDER_NAMES", "")
    with pytest.raises(OSError):
        with ambient.connect():
            pass
    assert ambient.log_message(
        chat_id="GROUP_A", chat_type="group", msg_id="m", sender_id="u", content="nope",
    ) is False
    assert target.read_bytes() == b"not an ambient database"
    assert _mode(target) == original

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "link-dir"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("QQ_AMBIENT_DB", str(linked / "messages.db"))
    with pytest.raises(OSError):
        with ambient.connect():
            pass
    assert not (real / "messages.db").exists()


def test_missing_group_id_is_not_archived(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = _use_db(monkeypatch, tmp_path)
    assert ambient.log_message(
        chat_id="", chat_type="group", msg_id="unknown", sender_id="u", content="unsafe",
    ) is False
    assert ambient.context_block(chat_id="") == ""
    assert not db.exists()


def test_context_stays_in_one_group_and_excludes_the_stored_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _use_db(monkeypatch, tmp_path)
    _log("GROUP_A", "shared-id", "only in A")
    _log("GROUP_B", "shared-id", "SECRET_FROM_B")
    _log("GROUP_B", "b2", "second B message")
    _log("GROUP_A", "body-id", "exclude current")
    _log("GROUP_A", "dm", "PRIVATE_DM", chat_type="c2c")
    _log("GROUP_A", "shared-id", "duplicate A")

    block = ambient.context_block(chat_id="GROUP_A", exclude_msg_id="body-id")
    other = ambient.context_block(chat_id="GROUP_B")
    assert "only in A" in block
    assert "exclude current" not in block
    for forbidden in ("SECRET_FROM_B", "second B message", "PRIVATE_DM", "duplicate A"):
        assert forbidden not in block
    assert "SECRET_FROM_B" in other
    assert "only in A" not in other
    wrong = ambient.context_block(chat_id="GROUP_A", exclude_msg_id="envelope-id")
    assert "exclude current" in wrong


def test_attachment_only_line_is_rendered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_db(monkeypatch, tmp_path)
    _log(
        "GROUP_A",
        "pic",
        "",
        attachments=[{
            "url": "https://multimedia.nt.qq.com.cn/a.png",
            "filename": "a.png",
            "content_type": "image/png",
        }],
    )
    _log(
        "GROUP_A",
        "doc",
        "",
        attachments=[{
            "url": "https://multimedia.nt.qq.com.cn/notes.pdf",
            "filename": "课件 1.pdf",
            "content_type": "application/pdf",
        }],
    )
    block = ambient.context_block(chat_id="GROUP_A")
    assert "图片 1 张" in block
    assert "文件 课件 1.pdf" in block


def test_old_database_gains_attachments_without_a_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    db = _use_db(monkeypatch, tmp_path)
    db.parent.mkdir(parents=True)
    raw = sqlite3.connect(db)
    raw.execute(
        """CREATE TABLE messages (
            id INTEGER PRIMARY KEY, channel TEXT, chat_type TEXT, chat_id TEXT,
            msg_id TEXT, ts TEXT, sender_id TEXT, content TEXT,
            at_bot INTEGER, routed INTEGER, mentions TEXT
        )"""
    )
    recent = datetime.now().isoformat(timespec="seconds")
    raw.execute(
        "INSERT INTO messages (channel, chat_type, chat_id, msg_id, ts, sender_id, content, at_bot, routed) "
        "VALUES ('qq', 'group', 'GROUP_A', 'old', ?, 'u', 'kept', 0, 0)",
        (recent,),
    )
    raw.commit()
    raw.close()
    _log("GROUP_A", "new", "fresh")
    with ambient.connect() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        assert "attachments" in cols
        kept = conn.execute("SELECT content FROM messages WHERE msg_id = 'old'").fetchone()
        assert kept["content"] == "kept"


def test_long_url_is_not_truncated_into_another_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _use_db(monkeypatch, tmp_path)
    url = "https://multimedia.nt.qq.com.cn/" + ("a" * 4000)
    _log("GROUP_A", "long", "", attachments=[{"url": url, "filename": "a.png"}])
    with ambient.connect() as conn:
        raw = conn.execute("SELECT attachments FROM messages").fetchone()[0]
    assert raw in (None, "")


def test_download_writes_allowlisted_https_and_records_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _use_db(monkeypatch, tmp_path)
    payload = b"\x89PNG\r\n\x1a\n" + b"x" * 32
    saved: dict[str, object] = {}

    class _Resp:
        def __init__(self) -> None:
            self._offset = 0

        def read(self, size: int) -> bytes:
            chunk = payload[self._offset:self._offset + size]
            self._offset += size
            return chunk

        def geturl(self) -> str:
            return "https://multimedia.nt.qq.com.cn/math.png"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    opener = MagicMock()
    opener.open.side_effect = lambda *args, **kwargs: _Resp()
    monkeypatch.setattr("urllib.request.build_opener", lambda *args, **kwargs: opener)
    meta = [{
        "url": "https://multimedia.nt.qq.com.cn/math.png",
        "filename": "数学题.png",
        "content_type": "image/png",
    }]
    _log("GROUP_A", "pic", "", attachments=meta)
    ambient._download_all("GROUP_A", "pic", ambient._normalize_attachments(meta))
    with ambient.connect() as conn:
        stored = json.loads(conn.execute("SELECT attachments FROM messages").fetchone()[0])
    saved_path = Path(str(stored[0]["saved_path"]))
    assert saved_path.read_bytes() == payload
    assert saved_path.parent == tmp_path / "media"
    if os.name == "posix":
        assert _mode(saved_path) == 0o600
    saved["path"] = saved_path

    ambient._download_all("GROUP_A", "bad", [{
        "url": "http://127.0.0.1/a.png",
        "filename": "a.png",
    }])
    # The row may not exist; the helper still must not raise. Record via a row.
    _log("GROUP_A", "http", "", attachments=[{"url": "http://127.0.0.1/a.png", "filename": "a.png"}])
    http_meta = ambient._normalize_attachments([{"url": "http://127.0.0.1/a.png", "filename": "a.png"}])
    ambient._download_all("GROUP_A", "http", http_meta)
    assert http_meta[0]["error"]
    assert "https required" in str(http_meta[0]["error"])

    evil = [{"url": "https://evil.example/a.png", "filename": "a.png"}]
    ambient._download_all("GROUP_A", "evil", evil)
    assert "host not allowed" in str(evil[0]["error"])

    monkeypatch.setenv("QQ_AMBIENT_MAX_ATTACH_BYTES", "4")
    huge = [{"url": "https://multimedia.nt.qq.com.cn/big.bin", "filename": "big.bin"}]
    ambient._download_all("GROUP_A", "huge", huge)
    assert "exceeds" in str(huge[0]["error"])
    assert list((tmp_path / "media").glob("*.part")) == []


def test_redirect_and_oversize_do_not_keep_partial_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    monkeypatch.setenv("QQ_AMBIENT_MEDIA_DIR", str(media))
    monkeypatch.setenv("QQ_AMBIENT_MAX_ATTACH_BYTES", "4")

    class _Resp:
        def read(self, size: int) -> bytes:
            return b"abcdef"

        def geturl(self) -> str:
            return "https://multimedia.nt.qq.com.cn/test"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    opener = MagicMock()
    opener.open.return_value = _Resp()
    monkeypatch.setattr("urllib.request.build_opener", lambda *args, **kwargs: opener)
    with pytest.raises(ValueError, match="exceeds"):
        ambient._download_one("https://multimedia.nt.qq.com.cn/test", "test.txt", "synthetic")
    assert not media.exists() or list(media.iterdir()) == []


def test_prune_removes_expired_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_db(monkeypatch, tmp_path)
    media = tmp_path / "media"
    media.mkdir()
    target = media / "old.png"
    target.write_bytes(b"old")
    os.chmod(target, 0o600)
    _log("GROUP_A", "old", "stale")
    old = (datetime.now() - timedelta(days=40)).isoformat(timespec="seconds")
    with ambient.connect() as conn:
        conn.execute(
            "UPDATE messages SET ts = ?, attachments = ? WHERE msg_id = 'old'",
            (old, json.dumps([{"saved_path": str(target)}], ensure_ascii=False)),
        )
        conn.commit()
        removed = ambient._prune(conn, 30)
    assert removed == 1
    assert not target.exists()


def test_sender_names_from_config_and_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _use_db(monkeypatch, tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "agents": {"defaults": {"botName": "小助手"}},
        "channels": {"qq": {
            "senderNames": {"user-a": "甲"},
            "botOpenid": "bot-self",
        }},
    }), encoding="utf-8")
    monkeypatch.delenv("QQ_AMBIENT_SENDER_NAMES", raising=False)
    from nanobot.config import loader as config_loader

    previous = config_loader._current_config_path
    config_loader.set_config_path(config)
    try:
        _log("GROUP_A", "m1", "hi", sender_id="user-a")
        _log("GROUP_A", "m2", "bot", sender_id="bot-self")
        block = ambient.context_block(chat_id="GROUP_A")
        assert "甲" in block
        assert "小助手" in block
    finally:
        config_loader._current_config_path = previous
    monkeypatch.setenv("QQ_AMBIENT_SENDER_NAMES", "user-a=乙")
    named = ambient._sender_names()
    assert named == {"user-a": "乙"}
    assert "bot-self" not in named


def test_cli_lists_files_and_prunes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _use_db(monkeypatch, tmp_path)
    _log("GROUP_A", "m1", "hello from search")
    assert ambient.main(["ambient", "recent", "--chat", "GROUP_A"]) == 0
    assert "hello from search" in capsys.readouterr().out
    assert ambient.main(["ambient", "search", "search"]) == 0
    assert "hello from search" in capsys.readouterr().out
    assert ambient.main(["ambient", "stats"]) == 0
    assert "total=1" in capsys.readouterr().out
    assert ambient.main(["ambient", "prune", "--days", "0"]) == 0
    assert "removed=" in capsys.readouterr().out


def test_parser_archives_plain_and_mentioned_without_mixing_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _use_db(monkeypatch, tmp_path)
    queued: list[str] = []
    monkeypatch.setattr(
        ambient,
        "queue_download",
        lambda **kwargs: queued.append(str(kwargs["msg_id"])),
    )
    plain = {
        "id": "envelope-plain",
        "d": {
            "id": "body-plain",
            "content": "旁白",
            "group_openid": "unauthorized",
            "author": {"member_openid": "user-b"},
            "attachments": [{
                "url": "https://multimedia.nt.qq.com.cn/a.png",
                "filename": "a.png",
                "content_type": "image/png",
            }],
        },
    }
    mentioned = {
        "id": "envelope-at",
        "d": {
            "id": "body-at",
            "content": "<@!bot-a> 看图",
            "group_openid": "group-a",
            "author": {"member_openid": "user-a"},
            "mentions": [{"id": "bot-a", "is_you": True}],
            "attachments": [{
                "url": "https://multimedia.nt.qq.com.cn/b.png",
                "filename": "b.png",
                "content_type": "image/png",
            }],
        },
    }
    plain_decision = decide_group_event(
        plain, state=MentionState(), robot_id="bot-a", require_mention=True,
    )
    at_decision = decide_group_event(
        mentioned, state=MentionState(), robot_id="bot-a", require_mention=True,
    )
    assert plain_decision.dispatch is False
    assert at_decision.dispatch is True
    archive_group_event(plain, plain_decision)
    archive_group_event(mentioned, at_decision)
    with ambient.connect() as conn:
        rows = {
            row["msg_id"]: row
            for row in conn.execute("SELECT chat_id, msg_id, routed, at_bot FROM messages")
        }
    assert set(rows) == {"body-plain", "body-at"}
    assert "envelope-plain" not in rows
    assert rows["body-plain"]["chat_id"] == "unauthorized"
    assert rows["body-plain"]["routed"] == 0
    assert rows["body-at"]["routed"] == 1
    assert queued == ["body-plain"]


def test_plain_events_are_archived_and_not_dispatched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from nanobot.channels.qq import group_events

    _use_db(monkeypatch, tmp_path)

    class _Conn:
        def __init__(self) -> None:
            self.robot = SimpleNamespace(id="bot-a")
            self.api = object()
            self.dispatched: list[str] = []

        def _dispatch(self, name: str, message: object) -> None:
            self.dispatched.append(name)

    previous_handler = group_events._handler
    previous_require = group_events._require_mention
    seen: list[str] = []
    group_events._handler = lambda decision: seen.append(decision.chat_id)
    group_events._require_mention = lambda: True
    conn = _Conn()
    try:
        for chat_id in ("allowed-group", "other-group"):
            group_events._parse_group_message_create(conn, {
                "id": f"envelope-{chat_id}",
                "d": {
                    "id": f"body-{chat_id}",
                    "content": "未艾特",
                    "group_openid": chat_id,
                    "author": {"member_openid": "user-b"},
                },
            })
    finally:
        group_events._handler = previous_handler
        group_events._require_mention = previous_require
    assert conn.dispatched == []
    assert seen == ["allowed-group", "other-group"]
    with ambient.connect() as db:
        chats = {row["chat_id"] for row in db.execute("SELECT chat_id FROM messages")}
    assert chats == {"allowed-group", "other-group"}


def test_archive_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ambient, "log_message", MagicMock(side_effect=RuntimeError("disk")))
    decision = decide_group_event(
        {"id": "env", "d": {"id": "body", "content": "hi", "group_openid": "g", "author": {}}},
        state=MentionState(),
        robot_id="bot",
        require_mention=True,
    )
    archive_group_event(
        {"id": "env", "d": {"id": "body", "content": "hi", "group_openid": "g"}},
        decision,
    )


def test_local_http_server_is_not_fetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hits = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hits["n"] += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"nope")

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _use_db(monkeypatch, tmp_path)
        monkeypatch.setenv("QQ_AMBIENT_ALLOWED_HOSTS", "127.0.0.1")
        meta = [{"url": f"http://127.0.0.1:{port}/a.png", "filename": "a.png"}]
        ambient._download_all("GROUP_A", "http", meta)
        assert "https required" in str(meta[0]["error"])
        assert hits["n"] == 0
    finally:
        server.shutdown()
        server.server_close()
