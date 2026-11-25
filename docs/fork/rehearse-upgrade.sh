#!/usr/bin/env bash
# Install this fork at a fixed commit into a temporary tool directory and
# confirm the archive module and in-process parser. Does not install the
# official wheel, does not run qq_group_patch.py, and does not change the
# production uv tool install or gateway.
set -eu

if [ "${1:-}" = "--source" ]; then
  source_path=$(cd -- "$2" && pwd -P)
  shift 2
else
  source_path=""
fi

commit="${NANOBOT_FORK_COMMIT:-}"
if [ -z "$source_path" ] && [ -z "$commit" ]; then
  echo "Set NANOBOT_FORK_COMMIT or pass --source /path/to/this/fork" >&2
  exit 2
fi

workdir=$(mktemp -d)
trap 'rm -rf -- "$workdir"' EXIT
export HOME="$workdir/home"
export UV_TOOL_DIR="$workdir/tools"
export UV_TOOL_BIN_DIR="$workdir/bin"
export UV_CACHE_DIR="$workdir/uv-cache"
export UV_LINK_MODE=copy
mkdir -p -- "$HOME"

if [ -n "$source_path" ]; then
  uv tool install --no-config --editable "$source_path"
else
  uv tool install --no-config \
    "nanobot-ai @ git+https://github.com/zzzhty/nanobot.git@${commit}"
fi

python="$UV_TOOL_DIR/nanobot-ai/bin/python"
export QQ_AMBIENT_DB="$workdir/qq_messages.db"
export QQ_AMBIENT_SENDER_NAMES=""
"$python" - <<'PY'
from nanobot.channels.qq import ambient
from nanobot.channels.qq.group_events import (
    install_group_message_parser,
    install_unknown_event_logger,
)

assert ambient.log_message(
    chat_id="rehearsal",
    chat_type="group",
    msg_id="rehearsal",
    sender_id="rehearsal",
    content="ok",
    at_bot=False,
    routed=False,
)
with ambient.connect() as conn:
    assert conn.execute("select count(*) from messages").fetchone()[0] == 1
install_group_message_parser(lambda _decision: None, require_mention=lambda: True)
install_unknown_event_logger()
print("fork rehearsal ok")
PY
