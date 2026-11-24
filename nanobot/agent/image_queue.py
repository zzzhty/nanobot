"""Persistent image queue for interactive QQ and WebUI sessions.

Interactive turns admit work in order. A draft is frozen in one SQLite
transaction before the provider is called. The first 180 seconds are a soft wait: the same provider call keeps running and
the user is asked whether to observe it for five more minutes or start a new
attempt. A provider that has already answered with a failure is not described
as a timeout. Restart cannot recover a call, because providers have no task
query.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from filelock import FileLock, Timeout
from loguru import logger

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.image_generation import (
    ImageGenerationTool,
    ImageRequestRejectedError,
    public_image_reason,
)
from nanobot.bus.channel_delivery import deliver_to_channel
from nanobot.bus.delivery import DeliveryRequest
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    INBOUND_META_USER_SHELL,
    OutboundMessage,
)
from nanobot.command.router import command_text
from nanobot.config.paths import get_data_dir
from nanobot.providers.image_generation import image_http_timeout
from nanobot.runtime_context import RuntimeContextBlock, wrap_runtime_context_lines
from nanobot.session.automation_turns import automation_history_overrides
from nanobot.session.display_history import HISTORY_GENERATION_METADATA_KEY
from nanobot.session.recovery import RECOVERY_INBOUND_METADATA_KEY
from nanobot.session.turn_continuation import internal_continuation_inbound

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage

TOKEN_KEY = "_local_image_queue_input"
DEBOUNCE = 8.0
PROVIDER_TIMEOUT = 180.0
DELIVERY_TIMEOUT = 30.0
WATCH_EXTEND = 300.0
_BACKGROUND_GRACE = 1.0
_IMAGE_CHOICES = {"/image-wait", "/image-retry"}


def _as_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    result: dict[str, Any] = {}
    for key, item in raw.items():
        if isinstance(key, str):
            result[key] = item
    return result


def _json_object(value: str) -> dict[str, Any]:
    parsed: object = json.loads(value)
    return _as_object(parsed) or {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    raw = cast(list[object], value)
    return [item for item in raw if isinstance(item, str) and item]


def _json_strings(value: str) -> list[str]:
    parsed: object = json.loads(value)
    return _string_list(parsed)
MAX_DRAFTS = 12
MAX_INPUTS = 64
_MAX_REFERENCE_BYTES = 20 * 1024 * 1024
_TURN: ContextVar[tuple[ImageQueue, Input | None] | None] = ContextVar(
    "image_queue_turn",
    default=None,
)


@dataclass(frozen=True, slots=True)
class Input:
    """One interactive message admitted ahead of the inbound queue."""

    id: str
    key: str
    origin: str
    channel: str
    chat: str
    sender: str
    message_id: str | None
    generation: str | None
    text: str
    media: tuple[str, ...]


def _span(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    return f"{minutes} 分钟"


def _soft_prompt(waited: float) -> str:
    waited_text = _span(waited)
    return (
        f"已经等了 {waited_text}，这条请求可能还在生成。我没有把它当成失败，也没有重新发送。"
        "直接说再等，我会继续看同一条请求，不会重新生图。"
        "重新生成是另一次请求，原来的可能已经计费。"
        "如果接下来不再说，我会停止观察，之后不能再等回结果。"
    )


def _terminal_text(state: str, detail: str) -> str:
    if state == "done":
        return "图画好了。"
    if state == "failed":
        reason = detail or "生图设置或参考图已变更"
        return (
            f"生图失败：{reason}。这是已经返回的结果，不是超时。"
            "没有自动重试。若要再试一次，直接说要重试。"
        )
    if detail == "extended":
        return (
            "又等了一段时间，仍然没有结果。已停止观察，没有重新生图。"
            "继续等不能再取回结果。若要重新生成，直接说要重试；原来的请求可能已经计费。"
        )
    if detail == "silence":
        return (
            "这段时间内没有选择，已停止观察这条请求，没有重新生图。"
            "继续等不能再取回结果。若要重新生成，直接说要重试；原来的请求可能已经计费。"
        )
    if detail == "retried":
        return (
            "已开始一次新的生图。上一条请求如果仍在服务端处理，可能已经计费，它的结果不会当作这次的图。"
        )
    return (
        "没能确认服务端是否已经处理这条请求，所以不能把它说成失败，也没有自动重试。"
        "原来的连接已经结束，继续等不能取回结果。服务端没有查询接口，重启后也不能补回这次结果。"
        "若要重新生成，直接说要重试。这会新开一次请求，原来的请求可能已经计费。"
    )


def _artifact_paths(result: object) -> list[str]:
    if not isinstance(result, str):
        return []
    try:
        parsed = _json_object(result)
    except (json.JSONDecodeError, TypeError):
        return []
    paths: list[str] = []
    artifacts = parsed.get("artifacts")
    rows = cast(list[object], artifacts) if isinstance(artifacts, list) else []
    for item in rows:
        artifact = _as_object(item)
        if artifact is None:
            continue
        artifact_path = artifact.get("path")
        if isinstance(artifact_path, str) and artifact_path:
            paths.append(artifact_path)
    return paths


def _safe_detail(text: str) -> str:
    return public_image_reason(RuntimeError(text.removeprefix("Error:").strip()))


class ImageQueue:
    """FIFO image drafts owned by the gateway process."""

    def __init__(
        self,
        loop: AgentLoop,
        root: Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        debounce: float = DEBOUNCE,
        provider_timeout: float = PROVIDER_TIMEOUT,
        delivery_timeout: float = DELIVERY_TIMEOUT,
        watch_extend: float = WATCH_EXTEND,
    ) -> None:
        self.loop = loop
        self.clock = clock
        self.debounce = debounce
        self.provider_timeout = provider_timeout
        self.delivery_timeout = delivery_timeout
        self.watch_extend = watch_extend
        self.root = Path(root) if root is not None else get_data_dir() / "image-queue"
        self._prepare_dir(self.root)
        self.refs = self.root / "refs"
        self._prepare_dir(self.refs)
        self.wake = asyncio.Event()
        self.workers: dict[str, asyncio.Task[None]] = {}
        self.inputs: dict[str, Input] = {}
        self.by_session: dict[str, dict[str, None]] = {}
        self.deferred: set[str] = set()
        self.pump: asyncio.Task[None] | None = None
        self.decision = asyncio.Event()
        self._background: dict[str, set[asyncio.Task[Any]]] = {}
        self._owner = FileLock(str(self.root / "owner.lock"))
        try:
            self._owner.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("image queue is already running") from exc
        try:
            self._prepare_file(self.root / "jobs.sqlite3")
            with self._tx() as db:
                db.execute(
                    """CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY, session TEXT NOT NULL, generation TEXT NOT NULL,
                        channel TEXT NOT NULL, chat TEXT NOT NULL, message_id TEXT,
                        provider TEXT NOT NULL, model TEXT NOT NULL, params TEXT,
                        version INTEGER NOT NULL, mergeable INTEGER NOT NULL,
                        state TEXT NOT NULL CHECK (state IN (
                            'draft','running','done','unknown','cancelled','failed','acknowledged'
                        )),
                        due REAL NOT NULL, changed REAL NOT NULL, cancelled INTEGER NOT NULL DEFAULT 0,
                        paths TEXT NOT NULL DEFAULT '[]', notice INTEGER NOT NULL DEFAULT 0,
                        delivery_error INTEGER NOT NULL DEFAULT 0,
                        watch_until REAL NOT NULL DEFAULT 0,
                        prompted INTEGER NOT NULL DEFAULT 0,
                        extensions INTEGER NOT NULL DEFAULT 0,
                        superseded INTEGER NOT NULL DEFAULT 0,
                        prompt_error INTEGER NOT NULL DEFAULT 0,
                        detail TEXT NOT NULL DEFAULT '',
                        retry_of TEXT)"""
                )
                columns = {str(row[1]) for row in db.execute("PRAGMA table_info(jobs)")}
                for name, ddl in (
                    ("delivery_error", "delivery_error INTEGER NOT NULL DEFAULT 0"),
                    ("watch_until", "watch_until REAL NOT NULL DEFAULT 0"),
                    ("prompted", "prompted INTEGER NOT NULL DEFAULT 0"),
                    ("extensions", "extensions INTEGER NOT NULL DEFAULT 0"),
                    ("superseded", "superseded INTEGER NOT NULL DEFAULT 0"),
                    ("prompt_error", "prompt_error INTEGER NOT NULL DEFAULT 0"),
                    ("detail", "detail TEXT NOT NULL DEFAULT ''"),
                    ("retry_of", "retry_of TEXT"),
                ):
                    if name not in columns:
                        db.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")
                db.execute("DROP INDEX IF EXISTS one_paid_call_per_session")
                db.execute(
                    """CREATE UNIQUE INDEX one_paid_call_per_session
                       ON jobs(session) WHERE state='running'"""
                )
                db.execute("CREATE INDEX IF NOT EXISTS image_queue_fifo ON jobs(session, state, due)")
                db.execute(
                    """CREATE TABLE IF NOT EXISTS inbox (
                        id TEXT PRIMARY KEY, session TEXT NOT NULL, generation TEXT NOT NULL,
                        payload TEXT NOT NULL, created REAL NOT NULL,
                        reason TEXT NOT NULL DEFAULT 'hold')"""
                )
                db.execute(
                    "CREATE INDEX IF NOT EXISTS image_inbox_session ON inbox(session, generation)"
                )
                db.execute(
                    """CREATE TABLE IF NOT EXISTS submissions (
                        session TEXT NOT NULL, generation TEXT NOT NULL, origin TEXT NOT NULL,
                        job TEXT NOT NULL, created REAL NOT NULL,
                        PRIMARY KEY(session, generation, origin))"""
                )
                db.execute(
                    "UPDATE jobs SET state='unknown', changed=?, notice=cancelled WHERE state='running'",
                    (self._now(),),
                )
        except BaseException:
            self._owner.release()
            raise

    def start(self) -> None:
        """Start the worker and admit new inbound messages before they queue."""
        if self.pump is not None:
            return
        self.pump = asyncio.create_task(self.run(), name="image-queue")
        self.loop.bus.image_queue_admit = self.admit
        logger.info("Persistent image queue enabled for QQ and WebUI")

    async def close(self) -> None:
        """Stop workers. Drafts stay; an in-flight provider call becomes unknown."""
        if not self._owner.is_locked:
            return
        self.loop.bus.image_queue_admit = None
        pump = self.pump
        self.pump = None
        if pump is not None:
            pump.cancel()
        tasks = tuple(self.workers.values())
        for task in tasks:
            task.cancel()
        background = tuple(task for group in self._background.values() for task in group)
        for task in background:
            task.cancel()
        try:
            pending = tuple(task for task in (pump, *tasks) if task is not None)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if background:
                await asyncio.wait(background, timeout=_BACKGROUND_GRACE)
            with self._tx() as db:
                db.execute(
                    """UPDATE jobs SET state='unknown', changed=?, notice=cancelled
                       WHERE state='running'""",
                    (self._now(),),
                )
        finally:
            self._owner.release()

    def backlog(self) -> bool:
        if self.workers:
            return True
        try:
            with self._tx() as db:
                return db.execute(
                    """SELECT 1 FROM jobs WHERE state IN ('draft','running')
                       OR (notice=0 AND state IN ('done','failed','unknown')) LIMIT 1"""
                ).fetchone() is not None
        except sqlite3.Error:
            return True

    async def planning_context(self, ctx: Any) -> RuntimeContextBlock | None:
        bound = _TURN.get()
        if bound is None or bound[0] is not self or not ctx.session_generation:
            return None
        record = bound[1]
        if (
            record is None
            or record.id not in self.inputs
            or not ctx.persist_session
            or not ctx.log_content
        ):
            return None
        if (ctx.session_key, ctx.channel, ctx.chat_id, ctx.sender_id) != (
            record.key, record.channel, record.chat, record.sender,
        ):
            return None
        try:
            with self._tx() as db:
                rows = db.execute(
                    """SELECT DISTINCT state FROM jobs
                       WHERE session=? AND generation=?
                         AND state IN ('draft','running','unknown','failed')""",
                    (record.key, ctx.session_generation),
                ).fetchall()
                undelivered = db.execute(
                    """SELECT 1 FROM jobs
                       WHERE session=? AND generation=? AND delivery_error=1 LIMIT 1""",
                    (record.key, ctx.session_generation),
                ).fetchone() is not None
        except sqlite3.Error:
            return None
        states = sorted({str(row[0]) for row in rows})
        if not states and not undelivered:
            return None
        lines: list[str] = []
        if states:
            lines.extend((
                f"Image queue in this conversation: {', '.join(states)}.",
                "If the current input adds or changes image requirements, use generate_image status and update the full draft. If clarification is needed, hold the draft with defer before asking.",
                "Do not queue a new image for unrelated chat. A running image cannot be changed. Do not call the provider again for an unconfirmed call. If the user asks to keep waiting, call generate_image with queue_mode=wait. That keeps the same provider call and does not generate again. If the user asks to try again, call queue_mode=retry. That starts one new attempt; say that an unknown or still-running original may already have been billed. Do not send /image-wait or /image-retry yourself.",
            ))
        if undelivered:
            lines.append(
                "An earlier image result was not delivered. Read status for its paths and do not generate that image again."
            )
        return RuntimeContextBlock("local-image-queue", wrap_runtime_context_lines(lines))

    def admit(self, msg: InboundMessage) -> None:
        """Register an interactive input before the bus publishes it."""
        msg.metadata = {key: value for key, value in msg.metadata.items() if key != TOKEN_KEY}
        meta = msg.metadata
        raw_text = meta.get("raw_content") if msg.channel == "qq" else None
        text = raw_text if isinstance(raw_text, str) else command_text(msg)
        stripped = text.strip()
        if (
            msg.channel not in {"qq", "websocket"}
            or not msg.is_user_input
            or msg.sender_id == "subagent"
            or meta.get(INBOUND_META_RUNTIME_CONTROL)
            or meta.get(INBOUND_META_USER_SHELL)
            or meta.get(RECOVERY_INBOUND_METADATA_KEY)
            or meta.get("original_command")
            or internal_continuation_inbound(meta)
            or automation_history_overrides(meta)[1]
            or self.loop.commands.is_priority(stripped)
            or self.loop.commands.is_dispatchable_command(stripped)
            or stripped.lower() in _IMAGE_CHOICES
        ):
            return
        key = msg.session_key
        if not key.startswith(msg.channel + ":"):
            return
        header = self.loop.sessions.state.peek_header(key)
        if header is not None and not (header.policy.persist and header.policy.log_content):
            return
        if len(self.by_session.get(key, ())) >= MAX_INPUTS:
            raise RuntimeError("Too many pending inputs in this session; try again later")
        mid = meta.get("message_id") if msg.channel == "qq" else meta.get("webui_turn_id")
        token = uuid4().hex
        record = Input(
            token,
            key,
            str(mid or token),
            msg.channel,
            msg.chat_id,
            msg.sender_id,
            str(mid) if mid and msg.channel == "qq" else None,
            msg.session_generation,
            str(text or msg.content)[:24000],
            tuple(msg.media[:16]),
        )
        with self._tx() as db:
            open_job = db.execute(
                """SELECT generation FROM jobs
                   WHERE session=? AND state IN ('draft','running','unknown')
                   ORDER BY rowid DESC LIMIT 1""",
                (key,),
            ).fetchone()
            if (
                open_job is not None
                and (header is None or str(open_job["generation"]) == header.generation)
                and record.generation in (None, str(open_job["generation"]))
            ):
                self._hold(db, record, str(open_job["generation"]))
        self.inputs[token] = record
        self.by_session.setdefault(key, {})[token] = None
        msg.metadata[TOKEN_KEY] = token

    def tracked(self, msg: InboundMessage) -> bool:
        token = msg.metadata.get(TOKEN_KEY)
        return isinstance(token, str) and token in self.inputs

    def bind(self, msg: InboundMessage) -> Token[tuple[ImageQueue, Input | None] | None]:
        """Expose this input to image-queue tools for the duration of its turn."""
        token = msg.metadata.get(TOKEN_KEY)
        record = self.inputs.get(token) if isinstance(token, str) else None
        return _TURN.set((self, record))

    def unbind(
        self,
        reset: Token[tuple[ImageQueue, Input | None] | None],
        msg: InboundMessage,
    ) -> None:
        """Restore the previous turn and drop the in-memory input record."""
        _TURN.reset(reset)
        token = msg.metadata.get(TOKEN_KEY)
        record = self.inputs.get(token) if isinstance(token, str) else None
        if record is not None:
            self._drop_input(record)

    @contextmanager
    def turn(self, msg: InboundMessage) -> Generator[None, None, None]:
        reset = self.bind(msg)
        try:
            yield
        finally:
            self.unbind(reset, msg)

    def finish(self, msg: InboundMessage, completed: bool) -> None:
        """Release a normal turn's held input. Errors stay for explicit recovery."""
        token = msg.metadata.get(TOKEN_KEY)
        record = self.inputs.get(token) if isinstance(token, str) else None
        if record is not None and completed and record.id not in self.deferred:
            with self._tx() as db:
                db.execute("DELETE FROM inbox WHERE id=?", (record.id,))
            self.wake.set()

    def cancel(self, key: str) -> int:
        """Drop drafts for a stopped session and keep an in-flight call from delivering."""
        with self._tx() as db:
            queued = db.execute(
                """UPDATE jobs SET state='cancelled', params=NULL, notice=1, changed=?
                   WHERE session=? AND state='draft'""",
                (self._now(), key),
            ).rowcount
            running = db.execute(
                """UPDATE jobs SET cancelled=1
                   WHERE session=? AND state='running' AND cancelled=0""",
                (key,),
            ).rowcount
            db.execute("DELETE FROM inbox WHERE session=?", (key,))
            db.execute(
                """UPDATE jobs SET state='cancelled', params=NULL, cancelled=1, notice=1, changed=?
                   WHERE session=? AND state='unknown'""",
                (self._now(), key),
            )
        worker = self.workers.get(key)
        if worker is not None:
            worker.cancel()
        for task in tuple(self._background.get(key, ())):
            task.cancel()
        self.decision.set()
        for token in self.by_session.pop(key, {}):
            self.inputs.pop(token, None)
            self.deferred.discard(token)
        self.wake.set()
        return int(queued) + int(running)

    async def intercept(self, msg: InboundMessage, key: str) -> bool:
        """Handle /image-resume, and keep raw /stop and /new visible as commands."""
        if not msg.is_user_input or msg.channel not in {"qq", "websocket"} or not key.startswith(msg.channel + ":"):
            return False
        raw = msg.metadata.get("raw_content") if msg.channel == "qq" else None
        command = (raw if isinstance(raw, str) else command_text(msg)).strip().lower()
        if command in {"/stop", "/new"} and isinstance(raw, str):
            msg.content = command
            return False
        if command in _IMAGE_CHOICES:
            await self._answer_choice(msg, key, command)
            return True
        token = msg.metadata.get(TOKEN_KEY)
        record = self.inputs.get(token) if isinstance(token, str) else None
        if command != "/image-resume" or record is None:
            return False
        try:
            header = await self.loop.sessions.state.read_header(key)
            if (
                header is not None
                and header.policy.persist
                and header.policy.log_content
                and record.generation in (None, header.generation)
                and record.id in self.inputs
            ):
                self.recovery(record, header.generation, "resume")
                self.finish(msg, True)
                text = "已确认未完成的生图记录。旧请求不会重试，未处理的追加已清除。新的请求可以继续。"
            else:
                text = "这个会话没有可恢复的生图队列。"
            delivery = self.loop.turn_delivery_factory.create(msg, key)
            if header is not None:
                await delivery.bind_session_generation(header.generation)
            await delivery.complete(
                OutboundMessage(msg.channel, msg.chat_id, text),
                publish_completion=True,
            )
        finally:
            with self.turn(msg):
                pass
        return True

    def status(self, key: str, generation: str) -> dict[str, Any]:
        with self._tx() as db:
            jobs = db.execute(
                """SELECT id, version, state, mergeable, cancelled, delivery_error, notice,
                          prompt_error, extensions, detail, params, paths FROM jobs
                   WHERE session=? AND generation=?
                   ORDER BY CASE WHEN state IN ('draft','running','unknown') THEN 0 ELSE 1 END,
                   rowid DESC LIMIT 18""",
                (key, generation),
            ).fetchall()
            held = db.execute(
                "SELECT id FROM inbox WHERE session=? AND generation=? ORDER BY rowid LIMIT 65",
                (key, generation),
            ).fetchall()
            paused = db.execute(
                "SELECT 1 FROM jobs WHERE session=? AND state='unknown'",
                (key,),
            ).fetchone() is not None
        visible: list[dict[str, Any]] = []
        for job in jobs:
            params = _json_object(str(job["params"] or "{}"))
            row: dict[str, Any] = {
                "id": str(job["id"]),
                "version": int(job["version"]),
                "state": str(job["state"]),
                "mergeable": int(job["mergeable"]),
                "cancelled": int(job["cancelled"]),
                "delivery_error": int(job["delivery_error"]),
                "notice": int(job["notice"]),
                "prompt_error": int(job["prompt_error"]),
                "extensions": int(job["extensions"]),
                "detail": str(job["detail"] or ""),
                "paths": _json_strings(str(job["paths"] or "[]")),
            }
            row.update({name: value for name, value in params.items() if name != "reference_images"})
            visible.append(row)
        unresolved = sum(str(item["id"]) not in self.inputs for item in held)
        can_wait = key in self.workers and any(
            item["state"] == "running" and item["cancelled"] == 0 for item in visible
        )
        return {
            "jobs": visible,
            "unconfirmed_call": paused,
            "unresolved_inputs": unresolved,
            "can_wait": can_wait,
        }

    def recovery(self, record: Input, generation: str, mode: str) -> str:
        with self._tx() as db:
            held = db.execute(
                "SELECT id, payload FROM inbox WHERE session=? AND generation=? ORDER BY rowid",
                (record.key, generation),
            ).fetchall()
            orphaned = [row for row in held if str(row["id"]) not in self.inputs]
            if mode == "recover":
                payloads: list[Any] = []
                for row in orphaned[:15]:
                    payloads.append(_json_object(str(row["payload"])))
                return json.dumps({"unresolved_inputs": payloads}, ensure_ascii=False)
            db.executemany("DELETE FROM inbox WHERE id=?", [(str(row["id"]),) for row in orphaned])
            db.execute(
                """UPDATE jobs SET state='acknowledged', notice=1, changed=?
                   WHERE session=? AND state='unknown'""",
                (self._now(), record.key),
            )
        self.wake.set()
        return "Unconfirmed image call acknowledged. It was not retried."

    def defer(self, record: Input, generation: str) -> str:
        with self._tx() as db:
            exists = db.execute(
                "SELECT 1 FROM jobs WHERE session=? AND generation=? AND state='draft'",
                (record.key, generation),
            ).fetchone()
            if exists is None:
                return ToolResult.error("No draft to hold for clarification.")
            self._hold(db, record, generation, reason="defer")
            self.deferred.add(record.id)
        return "Draft held. Read unresolved inputs with recover and revise before resuming."

    def submit(
        self,
        record: Input,
        generation: str,
        tool: ImageGenerationTool,
        params: dict[str, Any],
        mode: str,
        draft_id: str | None,
    ) -> str:
        now = self._now()
        snapshot = self.refs / uuid4().hex
        originals = params.get("reference_images")
        copied: list[str] | None = None
        try:
            if originals:
                snapshot.mkdir(mode=0o700)
                copied = []
                for index, source in enumerate(originals):
                    target = snapshot / f"{index}{Path(str(source)).suffix[:12]}"
                    with open(source, "rb") as src, target.open("xb") as dst:
                        data = src.read(_MAX_REFERENCE_BYTES + 1)
                        if len(data) > _MAX_REFERENCE_BYTES:
                            raise ValueError("Reference image exceeds 20 MiB")
                        dst.write(data)
                        dst.flush()
                        os.fsync(dst.fileno())
                    copied.append(str(target))
                params = {**params, "reference_images": copied}
            with self._tx() as db:
                prior = db.execute(
                    """SELECT job FROM submissions
                       WHERE session=? AND generation=? AND origin=?""",
                    (record.key, generation, record.origin),
                ).fetchone()
                if prior is not None:
                    if mode != "new":
                        self._release_deferred(db, record.key, generation)
                    return json.dumps({
                        "job": str(prior["job"]),
                        "status": "already accepted; do not submit again",
                    })
                row = None
                base = None
                if mode != "new":
                    current = db.execute(
                        """SELECT id, version, params FROM jobs
                           WHERE session=? AND generation=? AND state='draft' AND mergeable=1
                           ORDER BY rowid DESC LIMIT 1""",
                        (record.key, generation),
                    ).fetchone()
                    if draft_id:
                        target = db.execute(
                            """SELECT id, version, params, state FROM jobs
                               WHERE session=? AND generation=? AND id=?""",
                            (record.key, generation, draft_id),
                        ).fetchone()
                        if target is not None and str(target["state"]) == "draft":
                            row = target
                        elif (
                            target is not None
                            and str(target["state"]) in {"running", "done", "unknown", "acknowledged"}
                            and target["params"]
                            and current is None
                        ):
                            base = target
                        else:
                            return ToolResult.error(
                                "That draft is unavailable. Inspect status and choose the current draft explicitly."
                            )
                    else:
                        row = current
                selected = row if row is not None else base
                if selected is not None:
                    old = _json_object(str(selected["params"] or "{}"))
                    if originals is None:
                        params["reference_images"] = _string_list(old.get("reference_images"))
                    for name in ("aspect_ratio", "image_size", "count"):
                        if params.get(name) is None:
                            params[name] = old.get(name)
                params["reference_images"] = params.get("reference_images") or []
                job = str(row["id"]) if row is not None else uuid4().hex
                version = int(row["version"]) + 1 if row is not None else 1
                if row is not None:
                    db.execute(
                        """UPDATE jobs SET params=?, version=?, due=?, changed=?, message_id=?,
                           provider=?, model=? WHERE id=? AND state='draft'""",
                        (
                            json.dumps(params, ensure_ascii=False),
                            version,
                            now + self.debounce,
                            now,
                            record.message_id,
                            tool.config.provider,
                            tool.config.model,
                            job,
                        ),
                    )
                else:
                    count_row = db.execute(
                        "SELECT count(*) FROM jobs WHERE session=? AND state='draft'",
                        (record.key,),
                    ).fetchone()
                    count = int(count_row[0]) if count_row is not None else 0
                    if count >= MAX_DRAFTS:
                        return ToolResult.error(
                            "Too many queued image drafts in this conversation; cancel or wait."
                        )
                    db.execute(
                        """INSERT INTO jobs (
                            id, session, generation, channel, chat, message_id, provider, model,
                            params, version, mergeable, state, due, changed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)""",
                        (
                            job,
                            record.key,
                            generation,
                            record.channel,
                            record.chat,
                            record.message_id,
                            tool.config.provider,
                            tool.config.model,
                            json.dumps(params, ensure_ascii=False),
                            version,
                            int(mode != "new"),
                            now + self.debounce,
                            now,
                        ),
                    )
                if mode != "new":
                    self._release_deferred(db, record.key, generation)
                for pending_id in self.by_session.get(record.key, {}):
                    pending = self.inputs[pending_id]
                    if pending.generation in (None, generation):
                        self._hold(db, pending, generation)
                db.execute(
                    "INSERT INTO submissions VALUES (?, ?, ?, ?, ?)",
                    (record.key, generation, record.origin, job, now),
                )
            snapshot_done = snapshot
            snapshot = None
            del snapshot_done
        finally:
            if snapshot is not None and snapshot.exists():
                shutil.rmtree(snapshot)
        self.wake.set()
        return json.dumps({
            "job": job,
            "version": version,
            "status": "draft queued; worker sends result automatically. Do not send another image request for this input.",
        })

    async def execute(self, job: sqlite3.Row) -> None:
        if not await self._header_ok(job):
            self._settle(job, "cancelled")
            return
        tool = self.loop.tools.get("generate_image")
        params = _json_object(str(job["params"] or "{}"))
        if not params:
            self._settle(job, "failed")
            return
        refs = _string_list(params.get("reference_images"))
        raw_count = params.get("count")
        count = raw_count if isinstance(raw_count, int) and raw_count > 0 else 1
        if (
            not isinstance(tool, ImageGenerationTool)
            or (tool.config.provider, tool.config.model) != (str(job["provider"]), str(job["model"]))
            or count > tool.config.max_images_per_turn
            or any(not Path(ref).is_file() for ref in refs)
        ):
            self._settle(job, "failed")
            return
        selected = self._freeze(job)
        if selected is None:
            return
        prompt = params.get("prompt")
        aspect_ratio = params.get("aspect_ratio")
        image_size = params.get("image_size")
        self._arm_watch(selected)
        call = asyncio.create_task(
            self._invoke_provider(
                tool,
                prompt if isinstance(prompt, str) else "",
                refs,
                aspect_ratio if isinstance(aspect_ratio, str) else None,
                image_size if isinstance(image_size, str) else None,
                count,
            )
        )
        self._track(str(selected["session"]), call, str(selected["id"]))
        try:
            await self._watch(call, str(selected["id"]))
        except asyncio.CancelledError:
            call.cancel()
            self._settle(selected, "unknown")
            raise

    async def notice(self, job: sqlite3.Row) -> None:
        if not await self._header_ok(job):
            self._mark_notice(job, error=False)
            return
        paths = _json_strings(str(job["paths"] or "[]"))
        text = _terminal_text(str(job["state"]), str(job["detail"] or ""))
        metadata: dict[str, Any] = {
            "_record_channel_delivery": True,
            "_session_key": str(job["session"]),
            "_session_generation": str(job["generation"]),
        }
        message_id = job["message_id"]
        if isinstance(message_id, str) and message_id:
            metadata["message_id"] = message_id
        if str(job["channel"]) == "websocket":
            metadata[HISTORY_GENERATION_METADATA_KEY] = str(job["generation"])
        call = asyncio.create_task(
            deliver_to_channel(
                OutboundMessage(
                    str(job["channel"]),
                    str(job["chat"]),
                    text,
                    media=paths,
                    metadata=metadata,
                    delivery=DeliveryRequest(
                        "image-" + str(job["id"]),
                        str(job["session"]),
                        str(job["generation"]),
                    ),
                ),
                bus=self.loop.bus,
                session_manager=self.loop.sessions,
            )
        )
        self._track(str(job["session"]), call)
        try:
            done, _pending = await asyncio.wait({call}, timeout=self.delivery_timeout)
        except asyncio.CancelledError:
            call.cancel()
            raise
        if call not in done:
            call.cancel()
            logger.warning("Image result delivery timed out")
            self._mark_notice(job, error=True)
            return
        try:
            call.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Image result was not delivered: {}", type(exc).__name__)
            self._mark_notice(job, error=True)
            return
        self._mark_notice(job, error=False)

    async def run(self) -> None:
        while True:
            self.wake.clear()
            try:
                self._release_abandoned_running()
                for job in self._candidates():
                    key = str(job["session"])
                    if key in self.workers:
                        continue
                    worker = self.execute if str(job["state"]) == "draft" else self.notice
                    task = asyncio.create_task(worker(job))
                    self.workers[key] = task
                    task.add_done_callback(lambda done, session_key=key: self._finished(session_key, done))
            except Exception as exc:
                logger.warning("Image queue scan failed: {}", type(exc).__name__)
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1)
            except TimeoutError:
                pass

    def _finished(self, key: str, task: asyncio.Task[None]) -> None:
        if self.workers.get(key) is task:
            self.workers.pop(key, None)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.warning("Image queue worker failed: {}", type(exc).__name__)
        self.wake.set()

    def _candidates(self) -> list[sqlite3.Row]:
        with self._tx() as db:
            rows = db.execute(
                """SELECT j.* FROM jobs j WHERE
                   (j.state='draft' AND j.due<=? AND NOT EXISTS (
                       SELECT 1 FROM jobs b WHERE b.session=j.session AND
                       (b.state='running' OR (b.state='draft' AND b.rowid<j.rowid))
                   ) AND NOT EXISTS (
                       SELECT 1 FROM inbox i WHERE i.session=j.session AND i.generation=j.generation
                   ))
                   OR (j.notice=0 AND j.state IN ('done','unknown','failed'))
                   ORDER BY j.rowid""",
                (self._now(),),
            ).fetchall()
        return list(rows)

    def _freeze(self, job: sqlite3.Row) -> sqlite3.Row | None:
        now = self._now()
        with self._tx() as db:
            return db.execute(
                """UPDATE jobs AS j SET state='running', changed=?
                   WHERE j.id=? AND j.version=? AND j.state='draft' AND j.due<=?
                   AND NOT EXISTS (
                       SELECT 1 FROM inbox i WHERE i.session=j.session AND i.generation=j.generation
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM jobs other WHERE other.session=j.session AND
                       (other.state='running' OR (other.state='draft' AND other.rowid<j.rowid))
                   )
                   RETURNING *""",
                (now, str(job["id"]), int(job["version"]), now),
            ).fetchone()

    async def _header_ok(self, job: sqlite3.Row) -> bool:
        session = await self.loop.sessions.state.read_header(str(job["session"]))
        return (
            session is not None
            and session.generation == str(job["generation"])
            and session.policy.persist
            and session.policy.log_content
        )

    def _settle(
        self,
        job: sqlite3.Row,
        state: str,
        paths: list[str] | None = None,
        detail: str | None = None,
    ) -> None:
        saved_paths = paths or []
        assignments = (
            "state=CASE WHEN cancelled=1 THEN 'cancelled' ELSE ? END, "
            "paths=CASE WHEN cancelled=1 THEN '[]' ELSE ? END, "
            "changed=?, notice=cancelled"
        )
        values: list[Any] = [state, json.dumps(saved_paths), self._now()]
        if detail is not None:
            assignments += ", detail=?"
            values.append(detail)
        values.extend((str(job["id"]), str(job["state"]), int(job["version"])))
        with self._tx() as db:
            db.execute(
                f"UPDATE jobs SET {assignments} WHERE id=? AND state=? AND version=?",
                tuple(values),
            )

    def _hold(
        self,
        db: sqlite3.Connection,
        record: Input,
        generation: str,
        *,
        reason: str = "hold",
    ) -> None:
        db.execute(
            """INSERT INTO inbox (id, session, generation, payload, created, reason)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 payload=excluded.payload,
                 generation=excluded.generation,
                 reason=CASE WHEN excluded.reason='defer' THEN 'defer' ELSE inbox.reason END""",
            (
                record.id,
                record.key,
                generation,
                json.dumps({"text": record.text, "media": record.media}, ensure_ascii=False),
                self._now(),
                reason,
            ),
        )

    def _release_deferred(self, db: sqlite3.Connection, key: str, generation: str) -> None:
        """Drop clarification holds once a revised draft is accepted."""
        db.execute(
            "DELETE FROM inbox WHERE session=? AND generation=? AND reason='defer'",
            (key, generation),
        )

    async def _invoke_provider(
        self,
        tool: ImageGenerationTool,
        prompt: str,
        refs: list[str],
        aspect_ratio: str | None,
        image_size: str | None,
        count: int,
    ) -> str:
        token = image_http_timeout.set(self.provider_timeout + (2 * self.watch_extend))
        try:
            return await tool.generate_now(
                prompt,
                refs,
                aspect_ratio,
                image_size,
                count,
                references_are_resolved=True,
                propagate_provider_errors=True,
            )
        finally:
            image_http_timeout.reset(token)

    async def _watch(self, call: asyncio.Task[str], job_id: str) -> None:
        while True:
            row = self._job(job_id)
            if (
                row is None
                or str(row["state"]) != "running"
                or int(row["cancelled"])
                or int(row["superseded"])
            ):
                call.cancel()
                return
            if call.done():
                self._record_call_outcome(call, job_id)
                return
            remaining = float(row["watch_until"]) - self._now()
            if remaining <= 0:
                action = self._advance_watch(job_id)
                if action == "prompt":
                    await self._deliver_prompt(job_id)
                    continue
                if action == "wait":
                    continue
                if action != "close":
                    call.cancel()
                    return
                fresh = self._job(job_id)
                if fresh is None or str(fresh["state"]) != "running":
                    call.cancel()
                    return
                kind = "extended" if int(fresh["extensions"]) else "silence"
                call.cancel()
                if call.done() and not call.cancelled():
                    self._record_call_outcome(call, job_id)
                    return
                self._settle(fresh, "unknown", detail=kind)
                return
            self.decision.clear()
            waiter = asyncio.create_task(self.decision.wait())
            try:
                await asyncio.wait({call, waiter}, timeout=min(max(remaining, 0.0), 1.0))
            finally:
                waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await waiter

    def _arm_watch(self, job: sqlite3.Row) -> None:
        with self._tx() as db:
            db.execute(
                """UPDATE jobs SET watch_until=?
                   WHERE id=? AND state='running' AND watch_until=0""",
                (self._now() + self.provider_timeout, str(job["id"])),
            )

    def _advance_watch(self, job_id: str) -> str:
        now = self._now()
        with self._tx() as db:
            row = db.execute(
                """SELECT prompted, extensions, state, superseded, cancelled, watch_until
                   FROM jobs WHERE id=?""",
                (job_id,),
            ).fetchone()
            if (
                row is None
                or str(row["state"]) != "running"
                or int(row["superseded"])
                or int(row["cancelled"])
            ):
                return "stop"
            if float(row["watch_until"]) > now:
                return "wait"
            if int(row["prompted"]) == 0:
                db.execute(
                    """UPDATE jobs SET prompted=1, watch_until=?
                       WHERE id=? AND state='running' AND prompted=0""",
                    (now + self.watch_extend, job_id),
                )
                return "prompt"
            return "close"

    def _job(self, job_id: str) -> sqlite3.Row | None:
        with self._tx() as db:
            return db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def _record_call_outcome(self, call: asyncio.Task[str], job_id: str) -> None:
        if not call.done() or call.cancelled():
            return
        try:
            exc = call.exception()
        except asyncio.CancelledError:
            return
        if isinstance(exc, ImageRequestRejectedError):
            self._claim_terminal(job_id, "failed", detail=exc.reason)
            return
        if exc is not None:
            logger.warning("Image provider call has no confirmed result: {}", type(exc).__name__)
            self._claim_terminal(job_id, "unknown")
            return
        result = call.result()
        paths = _artifact_paths(result)
        if paths:
            self._claim_terminal(job_id, "done", paths=paths)
            return
        if isinstance(result, ToolResult) and result.is_error:
            self._claim_terminal(job_id, "failed", detail=_safe_detail(str(result)))
            return
        self._claim_terminal(job_id, "unknown")

    def _claim_terminal(
        self,
        job_id: str,
        state: str,
        paths: list[str] | None = None,
        detail: str = "",
    ) -> None:
        with self._tx() as db:
            db.execute(
                """UPDATE jobs SET state=?, paths=?, detail=?, changed=?, notice=0
                   WHERE id=? AND state='running' AND cancelled=0 AND superseded=0""",
                (state, json.dumps(paths or []), detail, self._now(), job_id),
            )

    async def _deliver_prompt(self, job_id: str) -> None:
        row = self._job(job_id)
        if row is None or str(row["state"]) != "running":
            return
        try:
            await self._send(row, _soft_prompt(self.provider_timeout), [])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Image wait prompt was not delivered: {}", type(exc).__name__)
            with self._tx() as db:
                db.execute("UPDATE jobs SET prompt_error=1 WHERE id=?", (job_id,))

    async def _send(self, job: sqlite3.Row, text: str, media: list[str]) -> None:
        metadata: dict[str, Any] = {
            "_record_channel_delivery": True,
            "_session_key": str(job["session"]),
            "_session_generation": str(job["generation"]),
        }
        message_id = job["message_id"]
        if isinstance(message_id, str) and message_id:
            metadata["message_id"] = message_id
        if str(job["channel"]) == "websocket":
            metadata[HISTORY_GENERATION_METADATA_KEY] = str(job["generation"])
        call = asyncio.create_task(
            deliver_to_channel(
                OutboundMessage(
                    str(job["channel"]),
                    str(job["chat"]),
                    text,
                    media=media,
                    metadata=metadata,
                    delivery=DeliveryRequest(
                        "image-" + str(job["id"]),
                        str(job["session"]),
                        str(job["generation"]),
                    ),
                ),
                bus=self.loop.bus,
                session_manager=self.loop.sessions,
            )
        )
        try:
            done, _pending = await asyncio.wait({call}, timeout=self.delivery_timeout)
        except asyncio.CancelledError:
            call.cancel()
            raise
        if call not in done:
            call.cancel()
            raise TimeoutError("image delivery timed out")
        call.result()

    async def _answer_choice(self, msg: InboundMessage, key: str, command: str) -> None:
        header = await self.loop.sessions.state.read_header(key)
        generation = header.generation if header is not None else msg.session_generation
        if not isinstance(generation, str) or not generation:
            text = "没有可操作的生图请求。"
            generation = ""
        elif command == "/image-wait":
            text = self.choose_wait(key, generation)
        else:
            text = self.choose_retry(key, generation)
        try:
            await self._send_chat(msg, key, generation, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Image choice was not delivered: {}", type(exc).__name__)
            self._note_prompt_error(key, generation)

    async def _send_chat(self, msg: InboundMessage, key: str, generation: str, text: str) -> None:
        metadata: dict[str, Any] = {"_record_channel_delivery": True, "_session_key": key}
        if generation:
            metadata["_session_generation"] = generation
        if msg.channel == "websocket" and generation:
            metadata[HISTORY_GENERATION_METADATA_KEY] = generation
        message_id = msg.metadata.get("message_id")
        if isinstance(message_id, str) and message_id:
            metadata["message_id"] = message_id
        call = asyncio.create_task(
            deliver_to_channel(
                OutboundMessage(
                    msg.channel,
                    msg.chat_id,
                    text,
                    metadata=metadata,
                    delivery=DeliveryRequest("image-choice-" + (message_id or key), key, generation or "none"),
                ),
                bus=self.loop.bus,
                session_manager=self.loop.sessions,
            )
        )
        try:
            done, _pending = await asyncio.wait({call}, timeout=self.delivery_timeout)
        except asyncio.CancelledError:
            call.cancel()
            raise
        if call not in done:
            call.cancel()
            raise TimeoutError("image choice delivery timed out")
        call.result()

    def choose_wait(self, key: str, generation: str) -> str:
        closed = (
            "这条请求已经不在观察中。继续等不能取回结果，也没有重新生图。"
            "服务端没有查询接口，重启或停止观察后不能补回原来的结果。"
            "若要重新生成，直接说要重试。这可能重复计费。"
        )
        if key not in self.workers:
            return closed
        now = self._now()
        with self._tx() as db:
            row = db.execute(
                """SELECT id, prompted, extensions FROM jobs
                   WHERE session=? AND generation=? AND state='running'
                     AND superseded=0 AND cancelled=0
                   ORDER BY rowid DESC LIMIT 1""",
                (key, generation),
            ).fetchone()
            if row is None:
                return closed
            if int(row["prompted"]) == 0:
                return "这条请求还在第一次等待中，没有重新生图。"
            if int(row["extensions"]) >= 1:
                return "已经在多等这条请求，不会重新生图。"
            db.execute(
                """UPDATE jobs SET extensions=1, watch_until=?
                   WHERE id=? AND state='running' AND extensions=0""",
                (now + self.watch_extend, str(row["id"])),
            )
        self.decision.set()
        return "继续观察同一条请求，不会重新生图。"

    def choose_retry(self, key: str, generation: str) -> str:
        accepted = "已经接受重新生成，不用重复发送。"
        started = (
            "已开始一次新的生图。上一条请求如果仍在服务端处理，可能已经计费，它的结果不会当作这次的图。"
        )
        finished = "上一条生图已经完成，没有重新生成。"
        nothing = "没有可重试的生图请求。"
        now = self._now()
        launch = False
        with self._tx() as db:
            newest = db.execute(
                """SELECT * FROM jobs WHERE session=? AND generation=?
                   ORDER BY rowid DESC LIMIT 1""",
                (key, generation),
            ).fetchone()
            if newest is None or not newest["params"]:
                return nothing
            state = str(newest["state"])
            if state in {"draft", "running"} and newest["retry_of"]:
                return accepted
            if state == "done":
                return finished
            if state == "cancelled" or int(newest["cancelled"]):
                return nothing
            if state == "running":
                replaced = db.execute(
                    """UPDATE jobs SET state='unknown', superseded=1, notice=1, detail='retried', changed=?
                       WHERE id=? AND state='running' AND superseded=0 AND cancelled=0""",
                    (now, str(newest["id"])),
                ).rowcount
                if replaced:
                    self._insert_retry(db, newest, now)
                    launch = True
                else:
                    current = db.execute(
                        "SELECT * FROM jobs WHERE id=?",
                        (str(newest["id"]),),
                    ).fetchone()
                    if current is None or not current["params"]:
                        return nothing
                    if str(current["state"]) == "done":
                        return finished
                    if str(current["state"]) == "running" or int(current["cancelled"]):
                        return accepted if int(current["superseded"]) or current["retry_of"] else nothing
                    newest = current
                    state = str(current["state"])
            if not launch and state in {"failed", "unknown"} and not int(newest["superseded"]):
                origin = f"retry:{newest['id']}"
                existing = db.execute(
                    """SELECT job FROM submissions
                       WHERE session=? AND generation=? AND origin=?""",
                    (key, generation, origin),
                ).fetchone()
                if existing is not None:
                    return accepted
                self._insert_retry(db, newest, now)
                launch = True
            elif not launch:
                return nothing
        for task in tuple(self._background.get(key, ())):
            task.cancel()
        self.decision.set()
        self.wake.set()
        return started

    def _insert_retry(self, db: sqlite3.Connection, source: sqlite3.Row, now: float) -> None:
        job = uuid4().hex
        origin = f"retry:{source['id']}"
        db.execute(
            "INSERT INTO submissions VALUES (?, ?, ?, ?, ?)",
            (str(source["session"]), str(source["generation"]), origin, job, now),
        )
        db.execute(
            """INSERT INTO jobs (
                id, session, generation, channel, chat, message_id, provider, model,
                params, version, mergeable, state, due, changed, retry_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 'draft', ?, ?, ?)""",
            (
                job,
                str(source["session"]),
                str(source["generation"]),
                str(source["channel"]),
                str(source["chat"]),
                source["message_id"],
                str(source["provider"]),
                str(source["model"]),
                str(source["params"]),
                now,
                now,
                str(source["id"]),
            ),
        )

    def _note_prompt_error(self, key: str, generation: str) -> None:
        if not generation:
            return
        with self._tx() as db:
            db.execute(
                """UPDATE jobs SET prompt_error=1
                   WHERE id=(
                       SELECT id FROM jobs WHERE session=? AND generation=?
                       ORDER BY rowid DESC LIMIT 1
                   )""",
                (key, generation),
            )

    def _mark_notice(self, job: sqlite3.Row, *, error: bool) -> None:
        with self._tx() as db:
            db.execute(
                """UPDATE jobs SET notice=1, delivery_error=?
                   WHERE id=? AND state=? AND notice=0""",
                (int(error), str(job["id"]), str(job["state"])),
            )

    def _release_abandoned_running(self) -> None:
        """A running row with no live worker is unconfirmed. Do not send it again."""
        live = set(self.workers)
        now = self._now()
        with self._tx() as db:
            rows = db.execute(
                "SELECT id, session, cancelled FROM jobs WHERE state='running'"
            ).fetchall()
            for row in rows:
                if str(row["session"]) in live:
                    continue
                if int(row["cancelled"]):
                    db.execute(
                        """UPDATE jobs SET state='cancelled', notice=1, changed=?
                           WHERE id=? AND state='running'""",
                        (now, str(row["id"])),
                    )
                else:
                    db.execute(
                        """UPDATE jobs SET state='unknown', notice=0, changed=?
                           WHERE id=? AND state='running'""",
                        (now, str(row["id"])),
                    )

    def _track(self, key: str, task: asyncio.Task[Any], job_id: str | None = None) -> None:
        bucket = self._background.setdefault(key, set())
        bucket.add(task)

        def _drop(done: asyncio.Task[Any], session_key: str = key, tracked_job: str | None = job_id) -> None:
            if tracked_job is not None:
                self._record_call_outcome(done, tracked_job)
            elif not done.cancelled():
                done.exception()
            tasks = self._background.get(session_key)
            if tasks is None:
                return
            tasks.discard(done)
            if not tasks:
                self._background.pop(session_key, None)

        task.add_done_callback(_drop)

    def _drop_input(self, record: Input) -> None:
        self.inputs.pop(record.id, None)
        self.deferred.discard(record.id)
        members = self.by_session.get(record.key)
        if members is not None:
            members.pop(record.id, None)
            if not members:
                self.by_session.pop(record.key, None)

    def _now(self) -> float:
        return float(self.clock())

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=0.15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _prepare_dir(path: Path) -> None:
        if path.is_symlink():
            raise OSError("image queue path must not be a symlink")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(path, 0o700)

    @staticmethod
    def _prepare_file(path: Path) -> None:
        if path.is_symlink():
            raise OSError("image queue database must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags | nofollow, 0o600)
        os.close(fd)
        if os.name == "posix":
            os.chmod(path, 0o600)


def interactive_queue_turn() -> bool:
    """Whether the current turn was admitted to the QQ or WebUI image queue."""
    return _active_record() is not None


def _active_record() -> tuple[ImageQueue, Input] | None:
    bound = _TURN.get()
    if bound is None or bound[1] is None:
        return None
    return bound[0], bound[1]


async def handle_image_request(
    tool: ImageGenerationTool,
    prompt: str | None,
    reference_images: list[str] | None,
    aspect_ratio: str | None,
    image_size: str | None,
    count: int | None,
    mode: str,
    draft_id: str | None,
) -> str:
    """Queue one interactive image request, or explain why this turn cannot."""
    active = _active_record()
    ctx = current_request_context()
    if active is None or ctx is None:
        return ToolResult.error(
            "Queued image generation requires an interactive QQ or WebUI user turn (not automation or replay)."
        )
    queue, record = active
    if (
        not ctx.persist_session
        or not ctx.log_content
        or not ctx.session_generation
        or (ctx.session_key, ctx.channel, ctx.chat_id, ctx.sender_id)
        != (record.key, record.channel, record.chat, record.sender)
        or record.id not in queue.inputs
        or record.generation not in (None, ctx.session_generation)
    ):
        return ToolResult.error("This turn cannot use the persistent image queue.")
    generation = ctx.session_generation
    try:
        if mode == "status":
            return json.dumps(queue.status(record.key, generation), ensure_ascii=False)
        if mode == "wait":
            return queue.choose_wait(record.key, generation)
        if mode == "retry":
            return queue.choose_retry(record.key, generation)
        if mode == "defer":
            return queue.defer(record, generation)
        if mode == "recover":
            return queue.recovery(record, generation, mode)
        if mode == "resume":
            return ToolResult.error("Only the user can resume the queue, using /image-resume.")
        if mode not in {"merge", "new"} or not prompt or not prompt.strip():
            return ToolResult.error("A full image prompt is required for merge/new.")
        if len(prompt) > 24000 or len(reference_images or []) > 8 or (count or 1) > tool.config.max_images_per_turn:
            return ToolResult.error("Image request exceeds queue size or count limits.")
        from nanobot.providers.image_generation import ImageGenerationError

        try:
            refs = tool.resolve_reference_images(reference_images) if reference_images is not None else None
        except ImageGenerationError as exc:
            return ToolResult.error(f"Image references: {exc}")
        return queue.submit(
            record,
            generation,
            tool,
            {
                "prompt": prompt,
                "reference_images": refs,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "count": count,
            },
            mode,
            draft_id,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        logger.warning("Image queue could not accept input: {}", type(exc).__name__)
        return ToolResult.error("Image queue could not apply this request. Do not start a separate provider call.")

