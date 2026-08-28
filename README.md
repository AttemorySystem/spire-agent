# Spire Agent

**Spire Agent — an autonomous Slay the Spire agent.**

Spire Agent autonomously plays complete runs of Slay the Spire.

- **Minimal harness.** Small agents route stable game states; complexity lives
  in external tools.
- **Deterministic first.** Policies and MCTS handle repeatable decisions; LLMs
  are reserved for choices that need contextual judgment.
- **Improves over runs.** Every run is recorded and replayable, becoming fixed
  regression evidence for the next policy improvement.

## Current status

- Characters: Ironclad and Defect
- Platforms: Linux and macOS tested; Windows untested
- Performance: under active improvement
    - Ironclad: several Ascension 20 Heart wins with GPT-5.6 Luna ([video](https://www.bilibili.com/video/BV1ewuo66EP7/))
    - Defect: reached the Act 4 stage multiple times

![Spire Agent playing Slay the Spire](docs/assets/demo.gif)

## Architecture

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

`GameAgent` is a small state-action loop and router. `MapAgent`, `BuildAgent`,
and `CombatAgent` use the same request/decision interface to connect to the
game, and each delegates domain work through an external tool interface. Every
state, decision, LLM call, and MCTS search is written under `runs/<seed>/`; the
same run can be reproduced from its replay journal.

`EvolveAgent` is the offline improvement loop. It turns accumulated run logs
into fixed evaluation datasets and proposes bounded Winning Path changes. This
works because improvement has a fixed acceptance test: the new policy is
compared with the current policy on the same historical choices and MCTS combat
checkpoints, including whether its rebuilt deck can defeat the enemies and Act
bosses reached by the original run—and the encounter that ended it. Only
measurable gains without benchmark regressions are promoted to the card picker.

## Install

Spire Agent requires a licensed Steam copy of Slay the Spire. Linux and macOS
are tested; Windows has not yet been tested. Install the game and subscribe to
ModTheSpire, BaseMod, and CommunicationMod, then clone the agent:

```bash
git clone --recurse-submodules https://github.com/AttemorySystem/spire-agent.git
cd spire-agent
uv sync
mkdir -p runtime/lib runtime/mods
```

Copy `desktop-1.0.jar` and `ModTheSpire.jar` to `runtime/lib/`, and copy
`BaseMod.jar` and `CommunicationMod.jar` to `runtime/mods/`. Then build both C++
tools in one invocation:

```bash
cmake -S 3rd/sts_lightspeed -B 3rd/sts_lightspeed/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 3rd/sts_lightspeed/build \
  --target battle-sim card-reward-eval -j
```

GitHub release bundles include the platform-independent `AgentStateFixes.jar`
and `AgentVisualizer.jar`, so release users do not need a JDK. From a source
checkout, copy the matching release JARs or build them locally with a JDK:

```bash
./game_mods/agent_state_fixes/build.sh
./game_mods/agent_visualizer/build.sh
```

Set the model in [config.yaml](config.yaml), export `API_KEY`, and follow the
[detailed installation guide](docs/install.md) for platform prerequisites,
exact Steam paths, verification, and troubleshooting.

## How to run

Review [config.yaml](config.yaml), then start the terminal UI:

```bash
uv run spire-agent
```

Each run writes logs, LLM calls, MCTS records, card choices, and a replay
journal under `runs/<canonical-seed>/`. `run_config.json` records the resolved
non-secret settings. An existing run directory is never overwritten.

Common console commands are:

```text
run                         # Start with config.yaml defaults
run ironclad a17 ABC123     # Override character, ascension, and seed
replay ABC123               # Replay and continue a recorded run
agent off                   # Pause the agent for manual control
agent on                    # Resume agent control after the game settles
window                      # Use the configured window size
fullscreen                  # Use fullscreen for the next run
hud=on                      # Enable the in-game agent HUD
hud=off                     # Disable the in-game agent HUD
```

Press `Ctrl+D` to exit. Use Up/Down to edit command history and
PageUp/PageDown to scroll the output.

To run directly without the terminal UI:

```bash
uv run spire-agent --no-tui
```

## Agents

All three agents receive a stable `DecisionRequest` and return one legal
`Decision`. Their tools are injected behind that boundary, so an implementation
can be replaced without changing `GameAgent` or replay.

- **Map Agent** evaluates complete routes to the Act boss. Deterministic gates
  preserve keys and reject unsupported consecutive fights using current-deck
  simulations; the LLM ranks the remaining routes by growth, recovery, Shops,
  and Act-specific risk.
- **Build Agent** owns card rewards, shops, events, and rest sites. Its default
  Winning Path picker searches toward a small set of expert winning-deck
  templates. Template distance, immediate survival needs, and contextual expert
  choices drive deterministic picks and skips; the LLM handles only unresolved
  frontiers.
- **Combat Agent** receives the complete combat state. Its default tool searches
  `sts_lightspeed` with MCTS and returns one root action plus any required card
  selections. Potions are first withheld, then selectively exposed to MCTS when
  the no-potion search predicts excessive HP loss.

Select implementations under `agents` in [config.yaml](config.yaml).

## Thanks

- [sts_lightspeed](https://github.com/Attemory/sts_lightspeed), the fast combat
  simulator and tree-search engine.
- [gym-sts](https://github.com/Attemory/gym-sts), the Slay the Spire game and
  CommunicationMod bridge;

Special thanks to [Baalorlord](https://baalorlord.tv/) for publishing his run
archive. Its expert card-choice history provides important evidence and
evaluation data for Winning Path.
