# Spire Agent

[English](README.md) | [简体中文](README_CN.md)

**Spire Agent — an autonomous Slay the Spire agent.**

## Highlights

- **Minimal framework.** A minimal agent framework makes logic and actions easy
  to customize, while independent, pluggable tools contain complex domain
  logic.
- **Deterministic first.** The harness handles everything that can be implemented
  deterministically, minimizing uncertainty from LLM inference. Only complex
  judgments are delegated to an LLM.
- **Improves with every run.** Every run is fully recorded and replayable,
  becoming fixed test evidence for later policy improvements.

## Current status

- Characters: Ironclad and Defect
- Platforms: Linux and macOS tested; Windows untested
- Current results:
    - Ironclad: several Ascension 20 Heart wins ([video](https://www.bilibili.com/video/BV1ewuo66EP7/))
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
into fixed evaluation datasets and optimizes the deck-building algorithm within
a fixed parameter space. Every improvement has a fixed acceptance test: the new
policy is compared with the current policy on the same historical choices and
MCTS combat checkpoints, including whether its rebuilt deck can defeat the
enemies and Act bosses reached by the original run—and the encounter that ended
it. Only measurable gains without benchmark regressions are merged into the
production card picker.

## Install

Spire Agent requires Slay the Spire. Install the game and subscribe to
ModTheSpire, BaseMod, and CommunicationMod on Steam, then clone the repository:

```bash
git clone --recurse-submodules https://github.com/AttemorySystem/spire-agent.git
cd spire-agent
uv sync
mkdir -p runtime/lib runtime/mods
```

Copy `desktop-1.0.jar` and `ModTheSpire.jar` to `runtime/lib/`, and copy
`BaseMod.jar` and `CommunicationMod.jar` to `runtime/mods/`.

Then build `sts_lightspeed`:

```bash
cmake -S 3rd/sts_lightspeed -B 3rd/sts_lightspeed/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 3rd/sts_lightspeed/build \
  --target battle-sim card-reward-eval -j
```

Download `AgentStateFixes.jar` and `AgentVisualizer.jar` from the
[GitHub Releases](https://github.com/AttemorySystem/spire-agent/releases) page
and place both files in `runtime/mods/`.

Set the model in [config.yaml](config.yaml) and set the `API_KEY` environment
variable.

The [detailed installation guide](docs/install.md) covers platform
prerequisites, Steam paths, installation verification, and troubleshooting.

## How to run

Configure [config.yaml](config.yaml), then start the terminal UI:

```bash
uv run spire-agent
```

Each run writes logs, LLM calls, MCTS records, card choices, and a replay
journal under `runs/<canonical-seed>/`. `run_config.json` records the resolved
non-secret settings. HUD history is recorded even when its in-game display is
off. An existing run directory is never overwritten.

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

Press `Ctrl+D` to exit. Use Up/Down to browse and edit command history, and use
`PageUp`/`PageDown` to scroll the output.

To run without the TUI:

```bash
uv run spire-agent --no-tui
```

## Agents

We divide the game into three decision units: map, build, and combat. One agent
owns each unit. At the code level, all three agents receive a stable
`DecisionRequest` and return one legal `Decision`. Tools are injected behind
this boundary, so an implementation can be replaced without changing
`GameAgent` or any other module.

- **Map Agent** uses an LLM because route selection depends on contextual
  tradeoffs that are difficult to reduce to one fixed score. It evaluates
  complete routes to the Act boss. Deterministic gates
  preserve keys and reject unsupported consecutive fights using current-deck
  simulations; the LLM ranks the remaining routes by growth, recovery, shops,
  and Act-specific risk.

- **Build Agent** owns deck construction, relic selection, shops, events, and
  rest sites. Instead of using a neural network to predict individual card
  picks, its default Winning Path picker searches a sparse graph of expert
  winning-deck templates. Template distance, immediate survival needs, and
  contextual expert choices determine whether to pick or skip; some unresolved
  candidates still require a final judgment from the LLM.

- **Combat Agent** receives the complete combat state. Its default tool searches
  `sts_lightspeed` with MCTS and returns one root action plus any required card
  selections. Potions are first withheld, then selectively exposed to MCTS when
  the no-potion search predicts excessive HP loss. The current implementation
  still contains some dirty hacks that need cleanup.

Select implementations under `agents` in [config.yaml](config.yaml).

## Notes

### LLM implementations and prompts

To let the configured LLM control every Build and Combat decision, set:

```yaml
agents:
  map: llm
  build: llm
  combat: llm
```

Build and Combat can be switched independently.

Build and Combat share the endpoint and model under `llm` in
[config.yaml](config.yaml). The full normalized game state is included in every
request.

The pure `build: llm` and `combat: llm` implementations are both defined in
[`tools/llm_agents.py`](src/spire_agent/tools/llm_agents.py). This file contains
the instructions sent to the model and requires both agents to return the same
JSON object: `{"command":"...","reason":"..."}`.

When customizing these instructions, keep the command contract unchanged: the
model must choose exactly one command currently exposed by CommunicationMod and
follow the documented index rules. The Tool assembles the changing game-state
payload automatically, so the prompt only needs stable decision rules.

With the default `agents.build: winning_path`, Winning Path resolves card
rewards when it has sufficient evidence. Unresolved card rewards and other
non-fast-path Build scenes use the scene-specific prompts in
[`en.toml`](src/spire_agent/subagents/prompts/build/en.toml) and
[`zh.toml`](src/spire_agent/subagents/prompts/build/zh.toml); these files are not
used by the pure `build: llm` implementation.

### MCTS search quality

The `mcts` section in [config.yaml](config.yaml) controls Combat search budget.
Increasing per-worker `simulations` and `max_time_ms` gives normal searches more
room; `adaptive_simulations` and `adaptive_time_ms` do the same for difficult
states.
`threads` controls parallel workers and RNG-world coverage. A search can reach
either its simulation or time limit, so increase both limits when one is already
binding. Larger values can improve difficult combat decisions, but consume more
CPU and make each action take longer; set `threads` according to the available
CPU cores.

### Map decision latency

Before asking the Map LLM to rank routes, MapTool evaluates the current deck
against representative future hallway and elite encounters, including relevant
post-Rest projections. These deterministic readiness simulations can make Map
decisions noticeably slower. Their small search budget is currently fixed in
[`tools/map/readiness.py`](src/spire_agent/tools/map/readiness.py), cached within
each run, and not exposed in `config.yaml`. The Combat `mcts` settings do not
change this Map evaluation budget.

## Thanks

- [sts_lightspeed](https://github.com/Attemory/sts_lightspeed), the fast combat
  simulator and tree-search engine.
- [gym-sts](https://github.com/Attemory/gym-sts), the Slay the Spire game and
  CommunicationMod bridge.

Special thanks to [Baalorlord](https://baalorlord.tv/) for publishing his run
archive. Its expert card-choice history provides important evidence and
evaluation data for Winning Path.
