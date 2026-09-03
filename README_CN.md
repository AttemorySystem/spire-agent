# Spire Agent

[English](README.md) | [简体中文](README_CN.md)

**Spire Agent: An Autonomous Agent for Slay the Spire**

## 核心特点

- **极简框架** 极简的 Agent 系统框架，可以灵活定制逻辑和动作；复杂业务逻辑
  由独立、可插拔的 Tool 实现。
- **确定性优先** 尽量消除 LLM 推理的不确定性：harness 保障交互确定性，独立 Tool
  实现具有稳定证据的领域决策，只有仍需开放式权衡的判断才交给 LLM。
- **持续改进** 每个 run 都会被完整记录并可回放，成为后续策略
  优化的固定测试证据。

## 当前状态

- 角色：铁甲战士、故障机器人
- 平台：已在 Linux 和 macOS 上测试；Windows 尚未测试
- 当前成绩：铁甲战士和故障机器人均已多次完成进阶 20 心脏通关
  （[铁甲战士](https://www.bilibili.com/video/BV1ewuo66EP7/)、
  [故障机器人](https://www.bilibili.com/video/BV14Sto65EQz/)）。
    - 铁甲战士：在最近的 12 局样本中，平均终局层数为 38.6；其中 6 局
      进入第三幕，包括 2 局进入第四幕。
    - 故障机器人：评测仍在进行；初步结果与上述铁甲战士成绩大致相当。

![Spire Agent 正在游玩杀戮尖塔](docs/assets/demo.gif)

## 为什么这很难

即使对有经验的玩家而言，通关《杀戮尖塔》的进阶 20 心脏（A20H）也很困难。一局游戏
跨越 50 多层，包含数百次相互耦合的决策：卡组决定战斗选择；战斗决定生命值；生命
值约束路线；路线又塑造卡组。早期的一次选择可能到最终 Boss 才显现代价。结果还会随
seed 改变而不同，而稀疏的终局奖励使失败难以归因。

LLM 能提供有价值的上下文判断，它难以胜任一个要求精确计算、协议
正确性和跨多个时间尺度保持决策一致性的过程(很多用LLM来控制一切的尝试，甚至无法通关A0)。一次看似轻微的幻觉、计算错误或资源
误用，最终都可能断送一局胜利。更多 Prompt 上下文并不能让这些性质变得确定，或保证得到提升。

| 问题 | 仅 LLM | Spire Agent |
|---|---|---|
| 终局奖励稀疏与整局随机性 | LLM的推理很难找到成功或失败的根因，因此无法可靠地比较策略变更。 | Replay 在相同运行配置下记录语义边界与 RNG；冻结的轨迹和配对战斗回归测试在相同 RNG 世界中比较改动。 |
| 随机战斗中的精确数值约束 | 长动作序列、精确计算和极端分支都容易被误判。 | Combat Tool 将完整状态交给高速模拟器和 MCTS，而不是要求 LLM 计算整场战斗。 |
| 卡牌价值依赖上下文 | 选择容易退化为静态强度排名或表面协同。 | Winning Path 结合构筑结构、即时生存需求和上下文专家证据，返回确定决策或受限 shortlist。 |
| 耦合的规划时间尺度 | LLM推理很难同时保持回合级精度和整局规划的合理性。 | Map、Build 和 Combat Agent 分别拥有决策权，只在边界间交换结构化证据。 |
| 有状态的游戏协议 | 模型可能在过期页面操作、选择错误索引，或在状态转换后丢失嵌套任务。 | 稳定状态适配、命令验证、确认后提交和带作用域的 continuation 将交互正确性移出模型推理。 |

Spire Agent中，harness 保证交互正确性；独立 Tool 负责有证据支撑的领域决策；LLM 处理尚未解决的权衡。
EvolveAgent 将 run 日志转化为冻结的评测样本，只有在 benchmark 提升且没有回归后才推广改动。

## 架构

```text
          +--------------------------+
          | Slay the Spire           |
          +-------------+------------+
                        ^
                        | gym-sts
                        v
          +-------------+------------+
          | GameAgent                |<-----> replay
          | loop + router            |------> logs-----+
          +-------------+------------+                 |
                        ^                              |
                        | agent templates              |
                        v                              |
             +----------+----------+                   |
             |          |          |                   |
             v          v          v                   v
          MapAgent   BuildAgent  CombatAgent      EvolveAgent
             |          |          |                   |
             v          v          v                   |
          MapTool   CardPicker   CombatTool            |
           (LLM)  (Winning Path)   (MCTS)              |
                        ^                              |
                        +------------------------------+
```

`GameAgent` 是一个很小的状态—动作循环和路由器。`MapAgent`、`BuildAgent`
和 `CombatAgent` 通过相同的请求/决策接口连接游戏，并分别把领域逻辑委托给
外部 Tool。所有状态、决策、LLM 调用和 MCTS 搜索都会写入
`runs/<seed>/`，同一个 run 可以通过 replay journal 精确复现。

## 安装

Spire Agent 依赖《杀戮尖塔》游戏本体。安装游戏并在 Steam 上订阅
ModTheSpire、BaseMod 和 CommunicationMod，然后克隆仓库：

```bash
git clone --recurse-submodules https://github.com/AttemorySystem/spire-agent.git
cd spire-agent
uv sync
mkdir -p runtime/lib runtime/mods
```

将 `desktop-1.0.jar` 和 `ModTheSpire.jar` 复制到 `runtime/lib/`，将
`BaseMod.jar` 和 `CommunicationMod.jar` 复制到 `runtime/mods/`。

然后构建 `sts_lightspeed`：

```bash
cmake -S 3rd/sts_lightspeed -B 3rd/sts_lightspeed/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 3rd/sts_lightspeed/build \
  --target battle-sim card-reward-eval -j
```

从 [GitHub Releases](https://github.com/AttemorySystem/spire-agent/releases)
下载 `AgentStateFixes.jar` 和 `AgentVisualizer.jar`，并将它们放入
`runtime/mods/`。

在 [config.yaml](config.yaml) 中设置模型，并设置 `API_KEY` 环境变量。

[详细安装指南](docs/install_CN.md) 包含各平台依赖配置、Steam 路径设置、安装验证和
故障排查说明。

## 运行

配置 [config.yaml](config.yaml) 后，启动终端界面：

```bash
uv run spire-agent
```

每个 run 的日志、LLM 调用、MCTS 记录、选牌记录和 replay journal 都会写入
`runs/<canonical-seed>/`。`run_config.json` 保存解析后的非敏感配置。即使不在
游戏内显示 HUD，HUD 历史也会记录。已有的 run 目录不会被覆盖。

常用控制台命令：

```text
run                         # 使用 config.yaml 默认配置启动
run ironclad a17 ABC123     # 覆盖角色、进阶等级和种子
replay ABC123               # 回放并继续一个已有 run
agent off                   # 暂停 Agent，切换为人工操作
agent on                    # 游戏稳定后恢复 Agent 操作
window                      # 下一个 run 使用配置的窗口大小
fullscreen                  # 下一个 run 使用全屏模式
hud=on                      # 启用游戏内 Agent HUD
hud=off                     # 禁用游戏内 Agent HUD
```

按 `Ctrl+D` 退出。使用上/下方向键浏览和编辑命令历史，使用
`PageUp`/`PageDown` 滚动输出区域。

不使用 TUI 界面运行：

```bash
uv run spire-agent --no-tui
```

## Agents

我们将游戏分为地图、构筑和战斗三个决策单元，每个单元由一个 Agent 负责。在代码
层面，三个 Agent 都接收稳定的 `DecisionRequest`，并返回一个合法的 `Decision`。
具体 Tool 通过这一边界注入，因此替换实现时不需要修改 `GameAgent` 或其他模块。

- **Map Agent** 使用 LLM，因为路线选择包含难以归纳为固定分数的上下文权衡。
  它会评估到本幕 Boss 为止的完整路线。确定性门控负责保留钥匙，并根据当前卡组的
  模拟结果排除无法支持的连续战斗；LLM 根据成长机会、恢复、商店和本幕特有风险对
  剩余路线进行排序。

- **Build Agent** 负责卡组构筑、遗物选择，以及商店、事件和休息处决策。默认的
  Winning Path CardPicker 不使用神经网络预测单次选牌，而是在专家通关卡组形成的
  稀疏模板图上搜索。模板距离、当前生存需求和同期专家选择共同决定拿牌或跳过；
  当前仍有一部分无法确定的候选会交给 LLM 进行最终判断。

- **Combat Agent** 接收完整战斗状态。默认 Tool 使用 `sts_lightspeed` 中的 MCTS
  搜索，返回一个根动作以及必要的后续选牌操作。搜索最初不会开放药水；
  只有无药水搜索预计损失过多 HP 时，才会有选择地向 MCTS 开放药水。当前实现中
  仍然包含一些需要清理的 dirty hacks。

在 [config.yaml](config.yaml) 的 `agents` 配置中选择具体实现。

## 说明

### Bugs

- MCTS 在完全没有胜率时的选择当前还有很多优化空间
- MCTS 并不喜欢随机

### LLM 实现与 Prompt

如果希望由配置的 LLM 接管所有 Build 和 Combat 决策，可以设置：

```yaml
agents:
  map: llm
  build: llm
  combat: llm
```

Build 和 Combat 可以独立切换。

Build 和 Combat 共用 [config.yaml](config.yaml) 中 `llm` 下的 endpoint 和模型。
每次请求都会包含完整、规范化的游戏状态。

纯 `build: llm` 和 `combat: llm` 实现都定义在
[`tools/llm_agents.py`](src/spire_agent/tools/llm_agents.py) 中。该文件包含发给模型的
指令，并要求两个 Agent 返回相同的 JSON 对象：
`{"command":"...","reason":"..."}`。

自定义这些指令时，不要改变命令契约：模型必须从 CommunicationMod 当前
提供的命令中只选择一个，并遵守文档中的索引规则。Tool 会自动组装动态
游戏状态，prompt 只需要包含稳定的决策规则。

使用默认的 `agents.build: winning_path` 时，Winning Path 会在证据充分时直接处理
选牌。无法确定的选牌和其他不属于 fast path 的 Build 场景，会使用以下分场景
prompt：
[`en.toml`](src/spire_agent/subagents/prompts/build/en.toml) 和
[`zh.toml`](src/spire_agent/subagents/prompts/build/zh.toml)；纯
`build: llm` 实现不会使用这两个文件。

### MCTS 搜索时间和质量

当前最大的时间开销是 MCTS 搜索。[config.yaml](config.yaml) 中的 `mcts` 配置控制
Combat 搜索预算。

提高每个 worker 的 `simulations` 和 `max_time_ms` 可以为普通搜索提供更多空间；
`adaptive_simulations` 和 `adaptive_time_ms` 对困难状态起相同作用。
`threads` 控制并行 worker 数和 RNG world 覆盖。搜索可能先达到模拟次数或时间限制，
因此当其中一个已经成为瓶颈时，需要同时提高两项限制。更大的配置可能改善困难战斗
中的决策，但也会占用更多 CPU，并延长每个动作的等待时间；请根据可用 CPU 核心数
设置 `threads`。

### 地图决策延迟

MapTool 在调用 Map LLM 对路线排序前，会使用当前卡组评估未来具有代表性的普通战斗
和精英战斗，并在需要时评估篝火后的状态。这些确定性的 readiness 模拟会使地图决策
明显变慢。它们使用的较小搜索预算目前固定在
[`tools/map/readiness.py`](src/spire_agent/tools/map/readiness.py) 中，会在同一个 run
内缓存，但尚未暴露到 `config.yaml`。Combat 的 `mcts` 配置不会改变这部分地图评估
预算。

## 致谢

- [sts_lightspeed](https://github.com/Attemory/sts_lightspeed)：高速战斗模拟器和
  树搜索引擎。
- [gym-sts](https://github.com/Attemory/gym-sts)：连接游戏和
  CommunicationMod 的交互桥梁。

特别感谢 [Baalorlord](https://baalorlord.tv/) 发布游戏记录归档，其中的专家选牌
历史为 Winning Path 提供了重要证据和评测数据。
