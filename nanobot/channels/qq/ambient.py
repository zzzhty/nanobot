"""Side-channel archive of inbound QQ group messages.

Records mentioned and unmentioned group messages, including ones that never
reach the agent. ``log_message``, ``queue_download``, and ``context_block``
never raise. Callers store ``d.id`` — the same id as ``GroupMessage.id`` —
not the gateway envelope id. Old rows are left as they were written.

The database defaults to the data directory's ``ambient/qq_messages.db``
(``QQ_AMBIENT_DB`` overrides it).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

from loguru import logger

_BUSY_TIMEOUT_MS = 3000
_ATTACHMENT_JSON_LIMIT = 8000
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    "application/zip": ".zip",
    "text/plain": ".txt",
}
_TOKEN_RE = re.compile(r"<@!?[^>]{1,64}>")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    channel     TEXT NOT NULL DEFAULT 'qq',
    chat_type   TEXT NOT NULL,
    chat_id     TEXT NOT NULL,
    msg_id      TEXT,
    ts          TEXT NOT NULL,
    sender_id   TEXT,
    content     TEXT,
    at_bot      INTEGER NOT NULL DEFAULT 0,
    routed      INTEGER NOT NULL DEFAULT 0,
    mentions    TEXT,
    attachments TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_msg
    ON messages (chat_id, msg_id) WHERE msg_id IS NOT NULL AND msg_id <> '';
CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages (chat_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages (ts);
CREATE TABLE IF NOT EXISTS senders (
    openid     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_state_lock = threading.Lock()
_ready_for: str | None = None
_prune_day_for: tuple[str, str] | None = None
_executor: Any = None
_executor_lock = threading.Lock()


def database_path() -> Path:
    """Return the archive path. ``QQ_AMBIENT_DB`` overrides the data directory."""
    raw = os.environ.get("QQ_AMBIENT_DB")
    if raw:
        return Path(raw).expanduser()
    from nanobot.config.paths import get_data_dir

    return get_data_dir() / "ambient" / "qq_messages.db"


def media_dir() -> Path:
    """Return the attachment directory. ``QQ_AMBIENT_MEDIA_DIR`` overrides it."""
    raw = os.environ.get("QQ_AMBIENT_MEDIA_DIR")
    if raw:
        return Path(raw).expanduser()
    from nanobot.config.paths import get_media_dir

    return get_media_dir("qq") / "ambient"


def retention_days() -> int:
    return _int_env("QQ_AMBIENT_RETENTION_DAYS", 30)


def max_attach_bytes() -> int:
    return _int_env("QQ_AMBIENT_MAX_ATTACH_BYTES", 20 * 1024 * 1024)


def download_timeout() -> int:
    return _int_env("QQ_AMBIENT_DOWNLOAD_TIMEOUT", 30)


def allowed_hosts() -> tuple[str, ...]:
    raw = os.environ.get("QQ_AMBIENT_ALLOWED_HOSTS") or "multimedia.nt.qq.com.cn"
    return tuple(host.strip().lower().rstrip(".") for host in raw.split(",") if host.strip())


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    result: dict[str, Any] = {}
    for key, item in raw.items():
        if isinstance(key, str):
            result[key] = item
    return result


def _sender_names() -> dict[str, str]:
    """Display names from the environment, else the config file. Never raises."""
    raw = os.environ.get("QQ_AMBIENT_SENDER_NAMES")
    if raw is not None:
        return _parse_name_list(raw)
    try:
        from nanobot.config.paths import get_config_path

        path = get_config_path()
        if not path.is_file() or path.is_symlink():
            return {}
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    data = _as_object(loaded)
    if data is None:
        return {}
    channels = _as_object(data.get("channels")) or {}
    qq = _as_object(channels.get("qq")) or {}
    names: dict[str, str] = {}
    configured = _as_object(qq.get("senderNames")) or _as_object(qq.get("sender_names")) or {}
    for openid, name in configured.items():
        key = str(openid).strip()
        label = str(name).strip()
        if key and label:
            names[key] = label
    agents = _as_object(data.get("agents")) or {}
    defaults = _as_object(agents.get("defaults")) or {}
    bot_name = str(defaults.get("botName") or defaults.get("bot_name") or "").strip()
    bot_openid = str(qq.get("botOpenid") or qq.get("bot_openid") or "").strip()
    if bot_name and bot_openid and bot_openid not in names:
        names[bot_openid] = bot_name
    return names


def _parse_name_list(raw: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in raw.split(","):
        openid, sep, name = item.partition("=")
        if sep and openid.strip() and name.strip():
            names[openid.strip()] = name.strip()
    return names


def _restrict_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise OSError(f"unsafe ambient database file: {path.name}")
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _open_connection() -> sqlite3.Connection:
    path = database_path()
    parent = path.parent
    if parent.is_symlink() or path.is_symlink():
        raise OSError("ambient database path is a symlink")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.is_symlink() or path.is_symlink():
        raise OSError("ambient database path is a symlink")
    if os.name == "posix":
        dir_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        dir_fd = os.open(parent, dir_flags)
        try:
            info = os.fstat(dir_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                raise OSError("unsafe ambient database directory")
            os.fchmod(dir_fd, 0o700)
        finally:
            os.close(dir_fd)
        try:
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(fd)
        for suffix in ("", "-wal", "-shm"):
            _restrict_file(path.with_name(path.name + suffix))

    conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_MS / 1000)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        if os.name == "posix":
            for suffix in ("-wal", "-shm"):
                _restrict_file(path.with_name(path.name + suffix))
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def connect() -> Generator[sqlite3.Connection, None, None]:
    """Open the archive. Raises on an unsafe path; callers that must not fail catch it."""
    conn = _open_connection()
    try:
        yield conn
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        if cols and "attachments" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")
            conn.commit()
    except Exception:
        logger.debug("QQ ambient schema migration failed")


def _sync_senders(conn: sqlite3.Connection) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    for openid, name in _sender_names().items():
        conn.execute(
            "INSERT INTO senders (openid, name, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(openid) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
            (openid, name, now),
        )
    conn.commit()


def _ensure(conn: sqlite3.Connection) -> None:
    global _ready_for, _prune_day_for
    key = str(database_path())
    with _state_lock:
        if _ready_for != key:
            conn.executescript(_SCHEMA)
            _migrate(conn)
            _ready_for = key
            _prune_day_for = None
        _sync_senders(conn)
        today = datetime.now().strftime("%Y-%m-%d")
        if _prune_day_for != (key, today):
            _prune_day_for = (key, today)
            _prune(conn, retention_days())


def _prune(conn: sqlite3.Connection, days: int) -> int:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    _prune_media(conn, cutoff)
    cur = conn.execute("DELETE FROM messages WHERE ts < ?", (cutoff,))
    removed = cur.rowcount or 0
    conn.commit()
    if removed:
        _vacuum_monthly(conn)
    return removed


def _inside_media(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _prune_media(conn: sqlite3.Connection, cutoff: str) -> None:
    try:
        rows = conn.execute(
            "SELECT attachments FROM messages WHERE ts < ? "
            "AND attachments IS NOT NULL AND attachments <> ''",
            (cutoff,),
        ).fetchall()
    except Exception:
        return
    root = media_dir()
    for row in rows:
        for item in _loads(row["attachments"]):
            saved = str(item.get("saved_path") or "")
            if not saved:
                continue
            candidate = Path(saved)
            if not _inside_media(candidate, root):
                continue
            try:
                candidate.unlink()
            except OSError:
                pass


def _vacuum_monthly(conn: sqlite3.Connection) -> None:
    try:
        today = datetime.now()
        if today.day != 1:
            return
        stamp = today.strftime("%Y-%m-%d")
        row = conn.execute("SELECT value FROM meta WHERE key = 'last_vacuum'").fetchone()
        if row is not None and row["value"] == stamp:
            return
        conn.execute("VACUUM")
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('last_vacuum', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (stamp,),
        )
        conn.commit()
    except Exception:
        logger.debug("QQ ambient vacuum failed")


def log_message(
    *,
    chat_id: str,
    chat_type: str,
    msg_id: str | None,
    sender_id: str | None,
    content: str | None,
    at_bot: bool = False,
    routed: bool = True,
    mentions: object = None,
    attachments: object = None,
) -> bool:
    """Record one inbound message. Never raises. Returns whether a row was stored."""
    if chat_type == "group" and not str(chat_id or "").strip():
        return False
    try:
        att_meta = _normalize_attachments(attachments)
        payload = _attachment_payload(att_meta) if att_meta else None
        mention_payload = None
        if mentions is not None:
            try:
                mention_payload = json.dumps(mentions, ensure_ascii=False)[:2000]
            except TypeError:
                mention_payload = None
        with connect() as conn:
            _ensure(conn)
            conn.execute(
                "INSERT OR IGNORE INTO messages "
                "(channel, chat_type, chat_id, msg_id, ts, sender_id, content, "
                "at_bot, routed, mentions, attachments) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "qq",
                    chat_type,
                    str(chat_id or ""),
                    str(msg_id or ""),
                    datetime.now().isoformat(timespec="seconds"),
                    str(sender_id or ""),
                    (content or "")[:8000],
                    1 if at_bot else 0,
                    1 if routed else 0,
                    mention_payload,
                    payload,
                ),
            )
            if payload:
                conn.execute(
                    "UPDATE messages SET attachments = ? WHERE chat_id = ? AND msg_id = ? "
                    "AND (attachments IS NULL OR attachments = '')",
                    (payload, str(chat_id or ""), str(msg_id or "")),
                )
            conn.commit()
        return True
    except Exception:
        logger.debug("QQ ambient log failed")
        return False


def _loads(raw: object) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        loaded: object = json.loads(str(raw))
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in cast(list[object], loaded):
        parsed = _as_object(item)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _att_field(att: object, name: str) -> object:
    if isinstance(att, dict):
        return cast(dict[object, object], att).get(name)
    return getattr(att, name, None)


def _attachment_items(attachments: object) -> list[object]:
    if isinstance(attachments, list):
        return list(cast(list[object], attachments))
    if isinstance(attachments, tuple):
        return list(cast(tuple[object, ...], attachments))
    return [attachments]


def _attachment_payload(meta: list[dict[str, Any]]) -> str:
    payload = json.dumps(meta, ensure_ascii=False)
    if len(payload) > _ATTACHMENT_JSON_LIMIT:
        raise ValueError("attachment metadata exceeds storage budget")
    return payload


def _bounded_text(value: object, limit: int = 300) -> str:
    text = str(value)[:limit]
    while len(json.dumps(text, ensure_ascii=False)) > limit + 2 and text:
        text = text[:-1]
    return text


def _normalize_attachments(attachments: object) -> list[dict[str, Any]]:
    if not attachments:
        return []
    out: list[dict[str, Any]] = []
    budget = 2
    reserve = len(json.dumps({"saved_path": "x" * 512, "error": "x" * 300}, ensure_ascii=False))
    for att in _attachment_items(attachments)[:1000]:
        url = str(_att_field(att, "url") or "").strip()
        if not url:
            continue
        if url.startswith("//"):
            url = "https:" + url
        if len(url) > 4000 or any(char.isspace() for char in url):
            continue
        entry: dict[str, Any] = {"url": url}
        for key in ("filename", "content_type", "size", "width", "height", "id"):
            value = _att_field(att, key)
            if value in (None, ""):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                entry[key] = _bounded_text(value)
            elif math.isfinite(value) and abs(value) < 1e20:
                entry[key] = value
            else:
                entry[key] = _bounded_text(value)
        cost = len(json.dumps(entry, ensure_ascii=False)) + reserve + 2
        if budget + cost > _ATTACHMENT_JSON_LIMIT:
            continue
        budget += cost
        out.append(entry)
    return out


def _safe_name(name: str, fallback: str = "file") -> str:
    cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", (name or "").strip())
    cleaned = cleaned.strip("._")[:80]
    return cleaned or fallback


def _name_from_url(url: str) -> str:
    from urllib.parse import unquote, urlparse

    try:
        base = os.path.basename(unquote(urlparse(url).path))
    except Exception:
        base = ""
    return _safe_name(base, "")


def _url_allowed(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    return any(host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts())


def _prepare_media_dir() -> Path:
    root = media_dir()
    if root.is_symlink():
        raise OSError("ambient media directory is a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise OSError("ambient media directory is a symlink")
    if os.name == "posix":
        os.chmod(root, 0o700)
        if root.stat().st_uid != os.geteuid():
            raise OSError("ambient media directory is owned by another user")
    return root


def _download_one(url: str, name: str, stem: str) -> str:
    import urllib.request

    if not _url_allowed(url):
        if not url.lower().startswith("https://"):
            raise ValueError("https required")
        raise ValueError(f"host not allowed: {url[:120]}")

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            raise ValueError(f"redirect blocked: {str(newurl)[:120]}")

    root = _prepare_media_dir()
    target = root / f"{stem}_{name}"
    tmp = target.with_name(target.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "nanobot-ambient/1.0"})
    opener = urllib.request.build_opener(_NoRedirect)
    total = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        with opener.open(req, timeout=download_timeout()) as resp:
            if not _url_allowed(str(resp.geturl())):
                raise ValueError("final url host not allowed")
            fd = os.open(tmp, flags, 0o600)
            with os.fdopen(fd, "wb") as fh:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_attach_bytes():
                        raise ValueError(f"attachment exceeds {max_attach_bytes()} bytes")
                    fh.write(chunk)
        tmp.replace(target)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return str(target)


def _download_all(chat_id: str, msg_id: str, meta: list[dict[str, Any]]) -> None:
    stem = hashlib.sha1(f"{chat_id}:{msg_id}".encode()).hexdigest()[:8]
    changed = False
    for index, item in enumerate(meta):
        if item.get("saved_path"):
            continue
        url = str(item.get("url") or "")
        name = _safe_name(str(item.get("filename") or "") or _name_from_url(url), "file")
        if "." not in name:
            ext = _MIME_EXT.get(str(item.get("content_type") or "").split(";")[0].strip(), "")
            if ext:
                name += ext
        try:
            item["saved_path"] = _download_one(url, name, f"{stem}_{index}")
            item.pop("error", None)
        except Exception as exc:
            item["error"] = _bounded_text(exc)
        changed = True
    if not changed:
        return
    try:
        with connect() as conn:
            _ensure(conn)
            conn.execute(
                "UPDATE messages SET attachments = ? WHERE chat_id = ? AND msg_id = ?",
                (_attachment_payload(meta), str(chat_id), str(msg_id)),
            )
            conn.commit()
    except Exception:
        logger.debug("QQ ambient attachment update failed")


def queue_download(*, chat_id: str, msg_id: str, attachments: object) -> None:
    """Fetch undispatched attachments in the background. Never raises."""
    try:
        meta = _normalize_attachments(attachments)
        if not meta or not str(msg_id or "").strip() or not str(chat_id or "").strip():
            return
        _download_executor().submit(_download_all, str(chat_id), str(msg_id), meta)
    except Exception:
        logger.debug("QQ ambient download queue failed")


def _download_executor() -> Any:
    global _executor
    with _executor_lock:
        if _executor is None:
            from concurrent.futures import ThreadPoolExecutor

            _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qq-ambient-dl")
        return _executor


def _attachment_summary(raw: object) -> str:
    items = _loads(raw)
    if not items:
        return ""
    labels: list[str] = []
    images = 0
    for item in items:
        name = str(item.get("filename") or "") or _name_from_url(str(item.get("url") or ""))
        ctype = str(item.get("content_type") or "")
        ext = os.path.splitext(name)[1].lower()
        if ctype.startswith("image/") or ext in _IMAGE_EXTS:
            images += 1
        else:
            labels.append(f"文件 {name[:60]}" if name else "文件")
    parts: list[str] = []
    if images:
        parts.append(f"图片 {images} 张")
    parts.extend(labels)
    return "[" + "、".join(parts) + "]"


def _lookup_name(openid: str | None) -> str:
    if not openid:
        return "?"
    named = _sender_names().get(openid)
    if named:
        return named
    try:
        with connect() as conn:
            found = conn.execute("SELECT name FROM senders WHERE openid = ?", (openid,)).fetchone()
            if found:
                return str(found["name"])
    except Exception:
        logger.debug("QQ ambient sender lookup failed")
    return f"…{openid[-6:]}"


def _clean(text: str | None) -> str:
    return _TOKEN_RE.sub("", (text or "").replace("\n", " ")).strip()


def context_block(
    *,
    chat_id: str,
    minutes: int | None = None,
    limit: int | None = None,
    chars: int = 160,
    exclude_msg_id: str | None = None,
) -> str:
    """Render recent group lines. Never raises. Excludes ``exclude_msg_id`` exactly."""
    if not str(chat_id or "").strip():
        return ""
    try:
        if minutes is None:
            minutes = _int_env("QQ_AMBIENT_CONTEXT_MINUTES", 120)
        if limit is None:
            limit = _int_env("QQ_AMBIENT_CONTEXT_LIMIT", 40)
        cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds")
        sql = (
            "SELECT * FROM (SELECT * FROM messages "
            "WHERE ts >= ? AND chat_type = 'group' AND chat_id = ? "
            "AND (msg_id IS NULL OR msg_id != ?) "
            "ORDER BY id DESC LIMIT ?) ORDER BY id"
        )
        with connect() as conn:
            _ensure(conn)
            rows = conn.execute(
                sql, (cutoff, str(chat_id), str(exclude_msg_id or ""), limit)
            ).fetchall()
        if not rows:
            return ""
        lines: list[str] = []
        for row in rows:
            body = _clean(row["content"])[:chars]
            atts = _attachment_summary(row["attachments"])
            if atts:
                body = f"{body} {atts}".strip() if body else atts
            if not body:
                continue
            stamp = row["ts"][11:16] if len(row["ts"]) >= 16 else row["ts"]
            mark = "*" if row["at_bot"] else " "
            lines.append(f"{stamp}{mark}{_lookup_name(row['sender_id'])}: {body}")
        if not lines:
            return ""
        header = f"[群里最近 {minutes} 分钟的对话（* = 当时 @ 了我）]"
        return header + "\n" + "\n".join(lines)
    except Exception:
        logger.debug("QQ ambient context failed")
        return ""


def _print_rows(rows: list[sqlite3.Row], *, chars: int, show_date: bool = False) -> None:
    for row in rows:
        ts = row["ts"]
        stamp = ts[5:16].replace("T", " ") if show_date and len(ts) >= 16 else (ts[11:16] if len(ts) >= 16 else ts)
        mark = "*" if row["at_bot"] else " "
        body = _clean(row["content"])[:chars]
        atts = _attachment_summary(row["attachments"])
        if atts:
            body = f"{body} {atts}".strip() if body else atts
        print(f"{stamp}{mark}{_lookup_name(row['sender_id'])}: {body}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry for ``python -m nanobot.channels.qq.ambient``."""
    args = list(sys.argv if argv is None else argv)
    args = args[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: python -m nanobot.channels.qq.ambient "
            "recent|since|search|around|files|stats|prune"
        )
        return 0
    cmd, rest = args[0], args[1:]
    opts: dict[str, str] = {}
    positional: list[str] = []
    index = 0
    while index < len(rest):
        item = rest[index]
        if item.startswith("--"):
            key = item[2:]
            if "=" in key:
                key, value = key.split("=", 1)
            elif index + 1 < len(rest):
                index += 1
                value = rest[index]
            else:
                value = ""
            opts[key] = value
        else:
            positional.append(item)
        index += 1
    try:
        limit = int(opts.get("limit", "30"))
        chars = int(opts.get("chars", "200"))
    except ValueError:
        print("invalid number", file=sys.stderr)
        return 2
    chat = opts.get("chat") or None
    try:
        conn_cm = connect()
        conn = conn_cm.__enter__()
    except OSError as exc:
        print(f"ambient database unavailable: {type(exc).__name__}", file=sys.stderr)
        return 1
    try:
        _ensure(conn)
        if cmd == "recent":
            sql = "SELECT * FROM (SELECT * FROM messages {} ORDER BY id DESC LIMIT ?) ORDER BY id"
            where, params = ("WHERE chat_id = ?", [chat]) if chat else ("", [])
            rows = conn.execute(sql.format(where), (*params, limit)).fetchall()
            _print_rows(rows, chars=chars)
            return 0
        if cmd == "since":
            minutes = int(opts.get("minutes", "60"))
            cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds")
            sql = (
                "SELECT * FROM (SELECT * FROM messages WHERE ts >= ? {} "
                "ORDER BY id DESC LIMIT ?) ORDER BY id"
            )
            where, params = ("AND chat_id = ?", [chat]) if chat else ("", [])
            rows = conn.execute(sql.format(where), (cutoff, *params, limit)).fetchall()
            _print_rows(rows, chars=chars)
            return 0
        if cmd == "search":
            if not positional:
                print("search needs a pattern", file=sys.stderr)
                return 2
            pattern = f"%{positional[0]}%"
            sql = (
                "SELECT * FROM (SELECT * FROM messages WHERE content LIKE ? {} "
                "ORDER BY id DESC LIMIT ?) ORDER BY id"
            )
            where, params = ("AND chat_id = ?", [chat]) if chat else ("", [])
            rows = conn.execute(sql.format(where), (pattern, *params, limit)).fetchall()
            _print_rows(rows, chars=chars, show_date=True)
            return 0
        if cmd == "around":
            if not positional:
                print("around needs a message id", file=sys.stderr)
                return 2
            before, after = int(opts.get("before", "10")), int(opts.get("after", "10"))
            anchor = conn.execute(
                "SELECT * FROM messages WHERE msg_id = ? OR id = ?",
                (positional[0], int(positional[0]) if positional[0].isdigit() else -1),
            ).fetchone()
            if anchor is None:
                print("message not found", file=sys.stderr)
                return 1
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE id <= ? ORDER BY id DESC LIMIT ?) "
                "UNION ALL SELECT * FROM (SELECT * FROM messages WHERE id > ? ORDER BY id LIMIT ?) "
                "ORDER BY id",
                (anchor["id"], before + 1, anchor["id"], after),
            ).fetchall()
            _print_rows(rows, chars=chars, show_date=True)
            return 0
        if cmd == "files":
            rows = conn.execute(
                "SELECT * FROM messages WHERE attachments IS NOT NULL AND attachments <> '' "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            for row in reversed(rows):
                stamp = row["ts"][5:16].replace("T", " ") if len(row["ts"]) >= 16 else row["ts"]
                print(f"{stamp} {_lookup_name(row['sender_id'])}: {_attachment_summary(row['attachments']) or '-'}")
                for item in _loads(row["attachments"]):
                    label = str(item.get("filename") or "file")
                    state = item.get("saved_path") or item.get("error") or "-"
                    print(f"    {label}")
                    print(f"    -> {state}")
            return 0
        if cmd == "stats":
            total = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
            chats = conn.execute(
                "SELECT chat_id, chat_type, COUNT(*) AS n, MAX(ts) AS last "
                "FROM messages GROUP BY chat_id, chat_type ORDER BY n DESC"
            ).fetchall()
            path = database_path()
            size = path.stat().st_size if path.exists() else 0
            print(f"total={total} db={size / 1024:.0f}KB path={path}")
            for row in chats:
                print(f"  {row['chat_type']} {row['chat_id']} n={row['n']} last={row['last']}")
            return 0
        if cmd == "prune":
            days = int(opts.get("days", str(retention_days())))
            removed = _prune(conn, days)
            print(f"removed={removed} older_than={days}d")
            return 0
    except ValueError:
        print("invalid number", file=sys.stderr)
        return 2
    finally:
        conn_cm.__exit__(None, None, None)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
