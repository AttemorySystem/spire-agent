# Spire Agent

[English](README.md) | [简体中文](README_CN.md)

**Spire Agent——一个可以自主游玩《杀戮尖塔》的智能体。**

## 核心特点

- **极简框架** 极简的 Agent 系统框架，可以灵活定制逻辑和动作；复杂业务逻辑
  由独立、可插拔的 Tool 实现。
- **确定性优先** 尽量消除 LLM 推理的不确定性：harness 保障交互确定性，独立 Tool
  实现具有稳定证据的领域决策，只有仍需开放式权衡的判断才交给 LLM。
- **从每局游戏中持续改进** 每个 run 都会被完整记录并可回放，成为后续策略
  优化的固定测试证据。

## 当前状态

- 角色：铁甲战士、故障机器人
- 平台：已在 Linux 和 macOS 上测试；Windows 尚未测试
- 当前成绩：
    - 铁甲战士：多次完成进阶 20 心脏通关（[视频](https://www.bilibili.com/video/BV1ewuo66EP7/)）
    - 故障机器人：多次进入第四幕

![Spire Agent 正在游玩杀戮尖塔](docs/assets/demo.gif)

## 为什么这很难

《杀戮尖塔》不是一个只要“理解规则”就能玩好的游戏。一局进阶 20 心脏（A20H）通常
跨越 50 多层，包含数百次相互依赖的决策：当前回合的出牌顺序会改变剩余生命，剩余
生命会改变路线，路线会改变卡牌和遗物的获得机会，而早期的一次选牌可能到最终 Boss
才显示出真正代价。同一策略面对不同 seed 时，结果可能完全相反。

LLM 擅长根据语义和上下文进行开放式判断，但 A20H 要求大量精确、连续而且跨时间
尺度一致的决策。一个看似很小的推理幻觉、计算错误或资源误用，都可能导致最终的失败；
更困难的是，仅凭终局结果也很难判断前几十层的哪一步需要修改。把完整
历史不断追加到 Prompt，可以增加信息，却不能保证计算精度、策略一致性或改进后的
行为不会破坏以前已经解决的问题。

| 困难 | LLM Only | Spire Agent |
|---|---|---|
| 最终奖励稀疏且环境随机，失败难以归因，策略也难以公平比较 | 复盘容易变成事后叙事；一次成功或失败不足以证明策略优劣 | Replay 在相同运行配置下保存语义边界和 RNG；历史轨迹评测生成固定决策样本与战斗检查点，并在相同 RNG worlds 上进行配对回归。 |
| 战斗具有巨大的组合空间，并要求精确处理出牌顺序、目标、能量和随机抽牌 | 模型可以提出合理战术，但容易在伤害计算、长动作序列和极端分支上出错 | Combat Tool 将完整战斗状态交给高速模拟器和 MCTS；LLM 不负责逐步心算整场战斗。 |
| 单张卡牌的价值取决于已有卡组、未来遭遇和尚未完成的构筑结构 | 决策容易退化为静态卡牌强度、表面协同或前后不一致的 LLM 推理幻觉 | Winning Path 算法显式维护构筑模块、近期生存需求和上下文专家证据，仅将确定结果或受限 shortlist 交给 Build Agent。 |
| 战斗、构筑、路线和药水资源处在不同时间尺度，但会相互影响 | 一个通用上下文很难同时保持回合级精度和整局规划，局部合理动作可能破坏长期目标 | Harness 按决策所有权组合 Combat、Build 和 Map Agent，并把战斗模拟结果、路线事实和构筑状态作为结构化证据跨层传递。 |
| 一局包含数百次命令以及大量嵌套选择页面，单次协议错误就可能中断运行 | 模型容易选择过期动作、错误索引，或在页面切换后丢失原任务 | harness 通过稳定状态适配、命令验证、确认后提交和带作用域的 continuation，将交互正确性从模型推理中移出。 |

Spire Agent harness 保证交互正确性；独立 Tool 利用模拟、专家经验和结构化证据扩大确定性决策范围。
EvolveAgent 负责把积累的 run 日志转换为固定数据集，在受约束的参数空间内反复评测策略；只有
通过 benchmark 且没有能力回退才会固化为可执行 Tool。LLM 保留那些证据仍然不足、需要开放式权衡的判断。

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

`EvolveAgent` 是离线优化循环。它把积累的 run 日志转换为固定评测数据集，
并在固定参数空间内优化 Deck Building 算法。每次优化都有固定的验收标准：新策略和
当前策略会在相同的历史选牌与 MCTS 战斗检查点上进行比较，包括新策略重建的卡组
能否战胜原 run 已经通过的普通敌人、各幕 Boss，以及最终导致失败的敌人。只有取得
可衡量提升且 benchmark 没有回归的修改，才会合入正式 CardPicker。

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
