# 当前部署

网关使用本仓库 `main` 的最新提交，安装在 uv 工具目录。单元没有 `ExecStartPre`，也不把补丁仓库放进 `PYTHONPATH`。配置、旁路库、媒体和生图队列仍在数据目录，不随安装目录搬家。

单元模板是 [`nanobot-gateway.service`](./nanobot-gateway.service)。保留 `QQ_GROUP_REQUIRE_MENTION`、`QQ_SLOW_ACK_SECONDS` 和 `NANOBOT_IMAGE_QUEUE`。uv 工具环境没有 pip，`PATH` 里要有 uv。不要写 `EnvironmentFile`。

旁路库是数据目录下的 `ambient/qq_messages.db`，可用 `QQ_AMBIENT_DB` 覆盖。上下文窗口由 `QQ_AMBIENT_CONTEXT_MINUTES` 和 `QQ_AMBIENT_CONTEXT_LIMIT` 控制，默认 120 分钟、40 条。新写入的消息 id 是事件体 `d.id`。

检查旁路库和进程内解析器：

```bash
QQ_AMBIENT_DB=/path/to/data/ambient/qq_messages.db \
QQ_AMBIENT_SENDER_NAMES= \
NANOBOT_PYTHON=/path/to/venv/bin/python \
  sh docs/fork/qq-ambient-check.sh
```

升级时安装本仓库 `main`，不要安装官方 wheel，也不要再跑补丁脚本。隔离演练用 [`rehearse-upgrade.sh`](./rehearse-upgrade.sh)，并设置 `NANOBOT_FORK_COMMIT` 为当时的 `main` 提交。
