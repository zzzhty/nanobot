#!/bin/sh
# Confirm the in-process archive and group parser.
# Success does not depend on site-packages patch markers.
# Set QQ_AMBIENT_DB to the archive to open. When unset, this check uses a
# temporary database so a dry run does not create the production file.
set -eu

python="${NANOBOT_PYTHON:-python3}"
"$python" - <<'PY'
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QQ_AMBIENT_SENDER_NAMES", "")
if not os.environ.get("QQ_AMBIENT_DB"):
    os.environ["QQ_AMBIENT_DB"] = str(Path(tempfile.mkdtemp()) / "qq_messages.db")

from nanobot.channels.qq import ambient
from nanobot.channels.qq.group_events import (
    install_group_message_parser,
    install_unknown_event_logger,
)

if ambient.main(["ambient", "stats"]) != 0:
    raise SystemExit("ambient database could not be opened")
try:
    from botpy.connection import ConnectionState
except ImportError as exc:
    raise SystemExit("qq-botpy is not installed") from exc
installed = install_group_message_parser(
    lambda _decision: None,
    require_mention=lambda: True,
)
has_parser = installed or callable(getattr(ConnectionState, "parse_group_message_create", None))
if not has_parser:
    raise SystemExit("in-process group parser is not available")
logger_installed = install_unknown_event_logger()
print(f"ambient ok parser_callable={has_parser} unknown_event_logger={logger_installed}")
PY
