"""Persistent image queue for interactive QQ and WebUI sessions.

Interactive turns admit work in order. A draft is frozen in one SQLite
transaction before the provider is called. Closing the gateway keeps unsent
drafts and marks an in-flight call unknown instead of retrying it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from filelock import FileLock, Timeout
from loguru import logger

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.image_generation import ImageGenerationTool
from nanobot.bus.channel_delivery import deliver_to_channel
from nanobot.bus.delivery import DeliveryRequest
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    INBOUND_META_USER_SHELL,
    OutboundMessage,
)
from nanobot.command.router import command_text
from nanobot.config.paths import get_data_dir
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


class ImageQueue:
    """FIFO image drafts owned by the gateway process."""

    def __init__(
        self,
        loop: AgentLoop,
        root: Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        debounce: float = DEBOUNCE,
    ) -> None:
        self.loop = loop
        self.clock = clock
        self.debounce = debounce
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
                        paths TEXT NOT NULL DEFAULT '[]', notice INTEGER NOT NULL DEFAULT 0)"""
                )
                db.execute(
                    """CREATE UNIQUE INDEX IF NOT EXISTS one_paid_call_per_session
                       ON jobs(session) WHERE state IN ('running','unknown')"""
                )
                db.execute("CREATE INDEX IF NOT EXISTS image_queue_fifo ON jobs(session, state, due)")
                db.execute(
                    """CREATE TABLE IF NOT EXISTS inbox (
                        id TEXT PRIMARY KEY, session TEXT NOT NULL, generation TEXT NOT NULL,
                        payload TEXT NOT NULL, created REAL NOT NULL)"""
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
            await asyncio.gather(pump, return_exceptions=True)
        tasks = tuple(self.workers.values())
        for task in tasks:
            task.cancel()
        try:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
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
                       WHERE session=? AND generation=? AND state IN ('draft','running','unknown')""",
                    (record.key, ctx.session_generation),
                ).fetchall()
        except sqlite3.Error:
            return None
        states = sorted({str(row[0]) for row in rows})
        if not states:
            return None
        lines = [
            f"Image queue in this conversation: {', '.join(states)}.",
            "If the current input adds or changes image requirements, use generate_image status and update the full draft. If clarification is needed, hold the draft with defer before asking.",
            "Do not queue a new image for unrelated chat. A running image cannot be changed; an unknown provider call must not be retried.",
        ]
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
                "UPDATE jobs SET cancelled=1, notice=1 WHERE session=? AND state='unknown'",
                (key,),
            )
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
                text = "已解除生图暂停。未确认的旧请求不会重试，未处理的追加已清除。"
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
                """SELECT id, version, state, mergeable, cancelled, params, paths FROM jobs
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
                "paths": _json_strings(str(job["paths"] or "[]")),
            }
            row.update({name: value for name, value in params.items() if name != "reference_images"})
            visible.append(row)
        unresolved = sum(str(item["id"]) not in self.inputs for item in held)
        return {
            "jobs": visible,
            "paused_after_unknown_call": paused,
            "unresolved_inputs": unresolved,
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
        return "Queue unpaused. No interrupted provider call has been retried."

    def defer(self, record: Input, generation: str) -> str:
        with self._tx() as db:
            exists = db.execute(
                "SELECT 1 FROM jobs WHERE session=? AND generation=? AND state='draft'",
                (record.key, generation),
            ).fetchone()
            if exists is None:
                return ToolResult.error("No draft to hold for clarification.")
            self._hold(db, record, generation)
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
        try:
            result = await tool.generate_now(
                prompt if isinstance(prompt, str) else "",
                refs,
                aspect_ratio if isinstance(aspect_ratio, str) else None,
                image_size if isinstance(image_size, str) else None,
                count,
                references_are_resolved=True,
            )
            parsed = _json_object(result)
            paths: list[str] = []
            artifacts = parsed.get("artifacts")
            artifact_rows = cast(list[object], artifacts) if isinstance(artifacts, list) else []
            for item in artifact_rows:
                artifact = _as_object(item)
                if artifact is None:
                    continue
                artifact_path = artifact.get("path")
                if isinstance(artifact_path, str) and artifact_path:
                    paths.append(artifact_path)
            if not paths:
                raise ValueError("no artifact paths")
        except asyncio.CancelledError:
            self._settle(selected, "unknown")
            raise
        except Exception as exc:
            logger.warning("Image provider call has no confirmed result: {}", type(exc).__name__)
            self._settle(selected, "unknown")
        else:
            self._settle(selected, "done", paths)

    async def notice(self, job: sqlite3.Row) -> None:
        allowed = await self._header_ok(job)
        with self._tx() as db:
            accepted = db.execute(
                "UPDATE jobs SET notice=1 WHERE id=? AND state=? AND notice=0",
                (str(job["id"]), str(job["state"])),
            ).rowcount
        if not (accepted and allowed):
            return
        paths = _json_strings(str(job["paths"] or "[]"))
        text = {
            "done": "图画好了。",
            "failed": "草稿没发出：生图设置或参考图已变更。",
            "unknown": "上一份生图未能确认结果，没有重试。队列已暂停；可以先让我查看未处理的追加，再发 /image-resume 解除暂停。",
        }[str(job["state"])]
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
        media = paths
        await deliver_to_channel(
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

    async def run(self) -> None:
        while True:
            self.wake.clear()
            try:
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
        self.workers.pop(key, None)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("Image queue worker failed: {}", type(task.exception()).__name__)
            return
        self.wake.set()

    def _candidates(self) -> list[sqlite3.Row]:
        with self._tx() as db:
            rows = db.execute(
                """SELECT j.* FROM jobs j WHERE
                   (j.state='draft' AND j.due<=? AND NOT EXISTS (
                       SELECT 1 FROM jobs b WHERE b.session=j.session AND
                       (b.state IN ('running','unknown') OR (b.state='draft' AND b.rowid<j.rowid))
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
                       (other.state IN ('running','unknown')
                        OR (other.state='draft' AND other.rowid<j.rowid))
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

    def _settle(self, job: sqlite3.Row, state: str, paths: list[str] | None = None) -> None:
        saved_paths = paths or []
        with self._tx() as db:
            db.execute(
                """UPDATE jobs SET
                   state=CASE WHEN cancelled=1 AND ?!='unknown' THEN 'cancelled' ELSE ? END,
                   paths=CASE WHEN cancelled=1 THEN '[]' ELSE ? END,
                   changed=?, notice=cancelled
                   WHERE id=? AND state=? AND version=?""",
                (
                    state,
                    state,
                    json.dumps(saved_paths),
                    self._now(),
                    str(job["id"]),
                    str(job["state"]),
                    int(job["version"]),
                ),
            )

    def _hold(self, db: sqlite3.Connection, record: Input, generation: str) -> None:
        db.execute(
            "INSERT OR IGNORE INTO inbox VALUES (?, ?, ?, ?, ?)",
            (
                record.id,
                record.key,
                generation,
                json.dumps({"text": record.text, "media": record.media}, ensure_ascii=False),
                self._now(),
            ),
        )

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

