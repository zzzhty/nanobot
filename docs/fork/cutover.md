# 从补丁部署换到本 fork

这份说明只描述维护窗口里的替换顺序。写它的时候没有改正在跑的网关、生产 uv 安装、systemd 单元或 `config.json`。窗口开始之前不要执行下面的停机和重装命令。

固定提交是 `8a8abc408a2234516481bf9faf306dab212cfc77`（`feat: archive QQ group traffic inside the fork`）。不要再安装官方 `nanobot-ai` wheel，也不要再跑 `qq_group_patch.py`。

## 现在不要做的事

不要把补丁仓库放进 `PYTHONPATH`，不要保留 `ExecStartPre`。不要把 `groupContextDir` 指到旁路库。不要改写 `~/.nanobot/ambient/qq_messages.db` 里的旧行，也不要把 `config.json`、旁路库、附件或队列库提交进 git。

## 窗口之前

用隔离目录演练安装，确认解释器来自临时工具目录，而不是正在使用的 uv 工具目录。

```bash
NANOBOT_FORK_COMMIT=8a8abc408a2234516481bf9faf306dab212cfc77 \
  bash docs/fork/rehearse-upgrade.sh
```

演练通过后再记录回退材料：当前 unit 全文、`ExecStart` 的解释器、补丁仓库提交、`uv-receipt.toml`，以及用该解释器导出的依赖清单。配置、会话、`ambient/qq_messages.db`、`media/qq/ambient` 和 `image-queue/` 按现有备份流程各留一份。这些文件留在本机，不进 git。

## 维护窗口

先停止网关，再停掉补丁巡检，避免巡检在替换过程中重打 site-packages。

```bash
systemctl --user stop nanobot-gateway.service
systemctl --user stop nanobot-qqpatch-check.timer
systemctl --user stop nanobot-qqpatch-check.service
```

把现有 uv 工具目录改名留作回退，不要直接删除。然后在原来的工具目录安装本 fork 的固定提交，并带上 QQ 依赖。安装命令使用维护窗口里确认过的 `UV_TOOL_DIR` 和 `UV_TOOL_BIN_DIR`，不要沿用演练的临时目录。

```bash
UV_LINK_MODE=copy uv tool install --force \
  "nanobot-ai @ git+https://github.com/zzzhty/nanobot.git@8a8abc408a2234516481bf9faf306dab212cfc77" \
  --with 'qq-botpy==1.2.1' --with 'aiohttp<4.0' --python 3.13
```

新 unit 以 [`nanobot-gateway.service`](./nanobot-gateway.service) 为模板。去掉 `ExecStartPre`，去掉指向补丁仓库的 `PYTHONPATH`。保留原来的端口、`WorkingDirectory`、`QQ_GROUP_REQUIRE_MENTION=1`、`QQ_SLOW_ACK_SECONDS=0` 和 `NANOBOT_IMAGE_QUEUE=1`。uv 工具环境没有 pip，把 uv 所在目录放进 `PATH`。不要写 `EnvironmentFile`，密钥继续只放在 `config.json`。配置路径不变时，旁路库默认就是数据目录下的 `ambient/qq_messages.db`，媒体目录默认是 `media/qq/ambient`，不需要为了换进程而搬家。

`daemon-reload` 之后再启动。启动后看 journal，确认没有执行 `qq_group_patch.py`，也没有把补丁仓库放进 `PYTHONPATH`。然后用新解释器打开旁路库并安装进程内解析器：

```bash
QQ_AMBIENT_DB=/path/to/data/ambient/qq_messages.db \
QQ_AMBIENT_SENDER_NAMES= \
NANOBOT_PYTHON=/path/to/venv/bin/python \
  sh docs/fork/qq-ambient-check.sh
```

把路径换成现场确认的数据目录和解释器。这条命令只读统计并检查模块，不往库里插探测行。

再在一个测试群里发一条未 @ 的文本和一张未 @ 的图片，确认它们进了旁路库、图片落到 `media/qq/ambient`，而且没有进入 agent。随后 @ 机器人，确认回复能看到刚才的旁路文本和 `[图片 N 张]`，且当前这句话没有重复出现。私聊和已有定时任务各发一次，确认群目标没有打到私聊接口。

## 数据兼容

旁路库沿用 `messages`、`senders`、`meta` 和 `attachments` 列，打开旧库不需要删库。新写入的 `msg_id` 是事件体 `d.id`，与 `GroupMessage.id` 相同；旧行保持原来的信封 id，不要批量改写。上下文排除只用当前消息的 `d.id`，所以旧行不会被当成“当前这条”删掉。

生图队列仍在数据目录的 `image-queue/jobs.sqlite3`。先停网关再让新进程打开它，避免锁文件冲突。新代码会补齐后来增加的列。如果打开失败，把整个 `image-queue/` 改名留存，让新进程重建空库；未完成的旧草稿不会自动继续，也不要手改库内行。

`channels.qq.botOpenid` 现在会被配置模型保留，只用于旁路显示名，不是授权名单。`groupContextDir` 即使还在配置里也不再写入，不要把它指到旁路库。

## 切换之后

停用并禁用 `nanobot-qqpatch-check.timer`。那个巡检仍以 site-packages 里的补丁标记为成功标准，留着会在新安装上误报。补丁仓库先留着，只作为回退材料，不要再挂到启动命令上。

稳定之前保留改名后的旧工具目录和旧 unit 副本。确认新网关能收发、旁路库继续增长、生图队列能打开之后，再考虑删除旧目录。

## 回退

保持网关停止。恢复改名留下的旧工具目录和切换前保存的 unit。旧 unit 里的 `ExecStartPre` 和 `PYTHONPATH` 只在回退时恢复。用切换前的补丁仓库提交在旧解释器的 site-packages 上重新应用并 `--check`，成功后再启动，并重新启用补丁巡检。只切 Git 提交不会恢复已经换掉的 uv 工具目录。旁路库可以留在原地，旧进程不读新写入的 `d.id` 也不会破坏旧行。
