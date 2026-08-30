# Spire Agent 安装指南

[English](install.md) | [简体中文](install_CN.md)

本文介绍如何从全新克隆开始安装 Spire Agent，并启动一个可见的本地
《杀戮尖塔》run。内容涵盖 macOS、Linux 和 Windows，以及可选的 replay 和
故障排查说明。

本仓库不包含游戏本体和 Steam Workshop JAR 文件。

一套可工作的安装由四部分组成：

1. 本仓库中的 Python Agent；
2. 负责启动游戏并与之通信的 `gym-sts`；
3. 提供战斗 MCTS 和卡牌评估的 `sts_lightspeed`；
4. 游戏 JAR、Workshop Mod，以及 Spire Agent 的 Java 支持 Mod。

## 1. 必备软件

所有平台都需要：

- Steam 版《杀戮尖塔》；
- Git；
- Python 3.11 或更高版本，通常由 `uv` 管理；
- `uv`；
- CMake 3.19 或更高版本；
- 支持 C++17 的编译器。

## 2. 克隆仓库

```bash
git clone --recurse-submodules \
  https://github.com/AttemorySystem/spire-agent.git
cd spire-agent
```

如果 clone 时没有使用 `--recurse-submodules`，可以随后初始化依赖：

```bash
git submodule update --init --recursive --depth 1
```

## 3. 安装 Python 环境

在 Agent 仓库根目录执行：

```bash
uv sync
```

## 4. 准备游戏和 Workshop JAR

```bash
mkdir -p runtime/lib runtime/mods
```

最终目录结构必须是：

```text
runtime/lib/
  desktop-1.0.jar
  ModTheSpire.jar

runtime/mods/
  BaseMod.jar
  CommunicationMod.jar
```

对应的 Steam Workshop 项目目录为：

| JAR | Workshop 项目 |
|---|---:|
| `ModTheSpire.jar` | `1605060445` |
| `BaseMod.jar` | `1605833019` |
| `CommunicationMod.jar` | `2131373661` |

在 Steam 中订阅这些项目，等待 Steam 完成下载后再复制文件。

### macOS

Steam 默认根目录是：

```text
~/Library/Application Support/Steam
```

在 Agent 仓库根目录执行：

```bash
STEAM_ROOT="$HOME/Library/Application Support/Steam"
STS_DIR="$STEAM_ROOT/steamapps/common/SlayTheSpire"
WORKSHOP_DIR="$STEAM_ROOT/steamapps/workshop/content/646570"

cp "$STS_DIR/desktop-1.0.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605060445/ModTheSpire.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605833019/BaseMod.jar" runtime/mods/
cp "$WORKSHOP_DIR/2131373661/CommunicationMod.jar" runtime/mods/
```

如果游戏存放在其他 Steam 游戏库中，请修改 `STEAM_ROOT` 等环境变量。

### Linux

常见的 Steam 根目录包括：

```text
~/.local/share/Steam
~/.steam/steam
~/.var/app/com.valvesoftware.Steam/.local/share/Steam
```

对于普通的原生 Steam 安装：

```bash
STEAM_ROOT="$HOME/.local/share/Steam"
STS_DIR="$STEAM_ROOT/steamapps/common/SlayTheSpire"
WORKSHOP_DIR="$STEAM_ROOT/steamapps/workshop/content/646570"

cp "$STS_DIR/desktop-1.0.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605060445/ModTheSpire.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605833019/BaseMod.jar" runtime/mods/
cp "$WORKSHOP_DIR/2131373661/CommunicationMod.jar" runtime/mods/
```

如果使用 Flatpak Steam 或自定义 Steam 游戏库，请修改 `STEAM_ROOT`。

### Windows

在仓库根目录通过 PowerShell 执行以下命令。示例假设 Steam 安装在默认位置：

```powershell
$SteamRoot = "${env:ProgramFiles(x86)}\Steam"
$StsDir = Join-Path $SteamRoot "steamapps\common\SlayTheSpire"
$WorkshopDir = Join-Path $SteamRoot "steamapps\workshop\content\646570"

New-Item -ItemType Directory -Force runtime\lib, runtime\mods | Out-Null
Copy-Item (Join-Path $StsDir "desktop-1.0.jar") runtime\lib\
Copy-Item (Join-Path $WorkshopDir "1605060445\ModTheSpire.jar") runtime\lib\
Copy-Item (Join-Path $WorkshopDir "1605833019\BaseMod.jar") runtime\mods\
Copy-Item (Join-Path $WorkshopDir "2131373661\CommunicationMod.jar") runtime\mods\
```

如果游戏存放在其他 Steam 游戏库中，请修改 `$SteamRoot`。

## 5. 安装 Java 支持 Mod

所有实时 run 和 replay 都必须加载 `AgentStateFixes`。它提供稳定决策边界、
战斗转换和 RNG 恢复所需的状态。

只有在设置 `run.hud: true` 时才需要 `AgentVisualizer`。

[GitHub Releases](https://github.com/AttemorySystem/spire-agent/releases)
包含这两个 JAR 文件，下载后放入 `runtime/mods/` 即可。

### 从源码构建（可选）

安装包含 `javac` 和 `jar` 的完整 JDK。JDK 8 是最稳妥的构建环境；因为脚本以
Java 8 字节码为目标，更新版本的 JDK 也可以使用。

#### macOS 和 Linux

```bash
./game_mods/agent_state_fixes/build.sh
./game_mods/agent_visualizer/build.sh
```

#### Windows

确认 JDK 在终端中可用后，通过 Git Bash 或 MSYS2 UCRT64 终端运行相同脚本：

```bash
javac -version
./game_mods/agent_state_fixes/build.sh
./game_mods/agent_visualizer/build.sh
```

此时所有平台都应具有以下文件：

```text
runtime/mods/
  AgentStateFixes.jar
  AgentVisualizer.jar
  BaseMod.jar
  CommunicationMod.jar
```

## 6. 编译 `sts_lightspeed`

实时运行需要两个 Release 可执行文件：

- 用于战斗 MCTS 的 `battle-sim`；
- 用于卡牌和遭遇评估的 `card-reward-eval`。

### macOS 和 Linux

```bash
cmake -S 3rd/sts_lightspeed -B 3rd/sts_lightspeed/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 3rd/sts_lightspeed/build \
  --target battle-sim card-reward-eval -j
```

### Windows

在 MSYS2 UCRT64 终端中执行：

```bash
cmake -S 3rd/sts_lightspeed -B 3rd/sts_lightspeed/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build 3rd/sts_lightspeed/build \
  --target battle-sim card-reward-eval -j
3rd/sts_lightspeed/build/card-reward-eval.exe --self-test
```

Windows 可执行文件带有 `.exe` 后缀。如果 Python 不能自动解析该后缀，请在启动
Agent 时显式传入两个可执行文件的路径：

```powershell
uv run spire-agent `
  --no-tui `
  --mcts-binary 3rd\sts_lightspeed\build\battle-sim.exe `
  --card-eval-binary 3rd\sts_lightspeed\build\card-reward-eval.exe
```

## 7. 配置 Agent 和 LLM

`config.yaml` 是运行时默认配置的唯一来源。CLI 参数可以在单次调用中覆盖它。
常规配置如下：

```yaml
prompt_language: en
agents:
  map: llm
  build: winning_path
  combat: mcts

run:
  character: IRONCLAD
  ascension: 20
  seed: random
  window_size: 1600x900
  hud: false

paths:
  runtime_dir: runtime
  log_dir: .
  mcts_binary: 3rd/sts_lightspeed/build/battle-sim
  card_eval_binary: 3rd/sts_lightspeed/build/card-reward-eval

llm:
  base_url: https://provider.example/v1
  model: provider/model-name

communication:
  timeout: 5.0

mcts:
  simulations: 100000
  threads: 12
  max_time_ms: 10000
  adaptive_time_ms: 30000
  adaptive_simulations: 500000
```

支持的实现名称如下：

| 配置项 | 可选值 |
|---|---|
| `agents.map` | `llm` |
| `agents.build` | `winning_path`、`llm` |
| `agents.combat` | `mcts`、`llm` |
| `prompt_language` | `en`、`zh` |
| `run.window_size` | `WIDTHxHEIGHT`、`fullscreen` |
| `run.hud` | `true`、`false` |

相对路径以 YAML 文件所在目录为基准解析。

`--config` 只读取 YAML，不会加载 shell 环境文件。`llm.base_url` 和
`llm.model` 可以留空，此时分别使用 `MODEL_URL` 和 `MODEL`。API key 始终来自
启动进程的 shell，或操作系统正常的密钥管理机制。

### macOS 和 Linux

```bash
export MODEL_URL="https://provider.example/v1"
export MODEL="provider/model-name"
export API_KEY="replace-with-your-key"
```

### Windows PowerShell

```powershell
$env:MODEL_URL = "https://provider.example/v1"
$env:MODEL = "provider/model-name"
$env:API_KEY = "replace-with-your-key"
```

## 8. 启动第一个 run

铁血战士：

```bash
uv run spire-agent --no-tui --character IRONCLAD --ascension 0 --seed 0
```

故障机器人：

```bash
uv run spire-agent --no-tui --character DEFECT --ascension 0 --seed 0
```

十进制种子会初始化 gym-sts 的确定性种子转换。游戏返回规范的字母数字
《杀戮尖塔》种子，并用它命名 run 目录。字母数字种子会原样传给游戏：

```bash
uv run spire-agent --no-tui --character IRONCLAD --ascension 20 --seed ABC123
```

普通 run 会加载以下 Mod ID：

```text
basemod, CommunicationMod, agentstatefixes
```

启动时，游戏文件会被复制到 `runtime/tmp/`；Steam 安装不会被修改。Bridge 文件和
游戏 stderr 会写入 `runtime/out/`；两者都可以在 run 之间安全清理。

使用默认的终端界面：

```bash
uv run spire-agent
```

控制台中的常用命令：

```text
run
run ironclad a17 ABC123
replay ABC123
agent off
agent on
window
fullscreen
hud=on
hud=off
```

使用上/下方向键浏览和编辑命令历史；`PageUp`/`PageDown` 滚动输出区域。HUD 和显示
命令会应用到下一次启动。

## 9. 显示模式和 HUD

将 `run.window_size` 设置为 `fullscreen`，即可使用当前显示器分辨率。也可以通过
单次启动参数启用相同模式：

```bash
uv run spire-agent --no-tui --character IRONCLAD --ascension 20 --seed 0 --fullscreen
```

每个 run 都会记录 HUD 历史。在 `config.yaml` 中设置 `run.hud: true`，即可加载
`AgentVisualizer.jar` 并在游戏内显示这些数据。窗口和全屏模式都支持 HUD。HUD
模式会加载：

```text
basemod, CommunicationMod, agentstatefixes, agentvisualizer
```

HUD 模式只会额外加入 `agentvisualizer`；无论是否启用 HUD，游戏逻辑使用的 Mod
集合都相同。在 macOS 上，全屏 run 还会在游戏运行期间阻止机器休眠。

## 10. 日志和 replay

每个 run 都会以规范种子为目录名写入持久化文件：

```text
runs/<seed>/
  run_config.json
  replay.jsonl
  run_history.jsonl
  hud_history.jsonl
  card_choices.jsonl
  mcts.log
  llm/
  mcts/
```

游戏进程输出写入：

```text
runtime/out/stderr.log
```

显式恢复一个崩溃的新格式 run：

```bash
uv run spire-agent --no-tui --replay runs/ABC123
```

Replay 执行下一条历史动作前，会将当前已确认状态保持
`replay.action_delay_seconds` 秒（默认 `0.5`）。将其设为 `0`，或使用
`--replay-action-delay 0`，即可最快回放。这个延迟只影响显示，不参与稳定检查和
replay 校验。

Replay 会校验每个已记录的决策边界和地下城 RNG 状态。不要编辑
`replay.jsonl`、弱化 mismatch 检查，或复用 run 目录。Replay mismatch 表示安装
或确定性发生故障，必须进行调查。

普通启动会拒绝覆盖已经存在的 `runs/<seed>` 目录。

## 11. 故障排查

### 缺少依赖 JAR

Java 构建脚本会输出缺失文件的准确路径。请确认文件名和大小写完全一致：

```bash
ls -l runtime/lib runtime/mods
```

不要只把 `ModTheSpire.jar` 放在 `runtime/mods/`；构建和启动程序要求它位于
`runtime/lib/`。

### 缺少 `transition_pending` 或 replay RNG 状态

加载的 `AgentStateFixes.jar` 不存在或已经过期。请从同一个 checkout 重新构建，
并确认 `runtime/mods/` 中生成的文件会在下次启动时被复制到
`runtime/tmp/mods/`。

### 游戏在窗口出现前退出

检查：

```bash
tail -200 runtime/out/stderr.log
```

常见原因包括缺少 JAR、Mod 版本错误，或者找不到 Java。启动程序会自动尝试定位
Steam 游戏自带的 JRE。对于自定义安装，请将环境变量指向 JRE 目录：

```bash
export STS_JRE_DIR="/path/to/SlayTheSpire/jre"
```

PowerShell：

```powershell
$env:STS_JRE_DIR = "D:\SteamLibrary\steamapps\common\SlayTheSpire\jre"
```

### 找不到 `battle-sim` 或 `card-reward-eval`

重新构建 `sts_lightspeed` 并确认文件位于预期路径：

```bash
ls -l 3rd/sts_lightspeed/build/battle-sim \
  3rd/sts_lightspeed/build/card-reward-eval
```

在 Windows 上，请使用带 `.exe` 后缀的路径，并传入前文展示的两个 CLI 覆盖参数。

### run 目录已经存在

这是有意设置的保护机制。请使用其他种子、把旧目录移动到归档位置，或显式 replay。
不要仅仅为了绕过检查而删除或覆盖一个 run。

### Steam 安装在自定义游戏库

找到三个 Workshop JAR 和 `desktop-1.0.jar`，然后将它们复制到本仓库要求的目录。
在 macOS 或 Linux 上：

```bash
find /path/to/SteamLibrary -type f \( \
  -name 'desktop-1.0.jar' -o \
  -name 'ModTheSpire.jar' -o \
  -name 'BaseMod.jar' -o \
  -name 'CommunicationMod.jar' \
\)
```
