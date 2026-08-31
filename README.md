# Spire Agent

[English](README.md) | [简体中文](README_CN.md)

**Spire Agent: An Autonomous Agent for Slay the Spire**

## Highlights

- **Minimal framework.** A minimal agent framework makes logic and actions easy
  to customize, while independent, pluggable tools contain complex domain
  logic.
- **Deterministic first.** The harness guarantees deterministic interaction,
  while independent tools implement decisions backed by stable evidence. Only
  judgments that still require open-ended tradeoffs are delegated to an LLM.
- **Continuous improvement.** Every run is fully recorded and replayable,
  becoming fixed test evidence for later policy improvements.

## Current status

- Characters: Ironclad and Defect
- Platforms: Linux and macOS tested; Windows untested
- Current results:
    - Ironclad: several Ascension 20 Heart wins ([video](https://www.bilibili.com/video/BV1ewuo66EP7/))
    - Defect: reached the Act 4 stage multiple times

![Spire Agent playing Slay the Spire](docs/assets/demo.gif)

## Why this is hard

Slay the Spire is not a game that can be mastered simply by understanding its
rules. An Ascension 20 Heart (A20H) run usually spans more than 50 floors and
contains hundreds of interdependent decisions: the order of cards played this
turn changes the HP that remains, remaining HP changes the route, the route
changes opportunities to acquire cards and relics, and the true cost of an
early card choice may not appear until the final boss. The same strategy can
produce completely opposite results on different seeds.

LLMs excel at open-ended judgment based on semantics and context, but A20H
demands a long sequence of precise, continuous decisions that remain consistent
across time scales. A seemingly minor hallucination, calculation error, or
misuse of resources can ultimately end the run. More importantly, the final
outcome alone rarely reveals which decision dozens of floors earlier needs to
change. Appending the complete history to the prompt adds information, but
cannot guarantee numerical precision, policy consistency, or that an
improvement will preserve behavior that already works.

| Difficulty | LLM Only | Spire Agent |
|---|---|---|
| Terminal rewards are sparse and the environment is stochastic, making failures hard to attribute and policies hard to compare fairly | Reviews easily become post-hoc narratives; one success or failure does not establish which policy is better | Under the same runtime configuration, Replay preserves semantic boundaries and RNG; historical trajectory evaluation produces fixed decision samples and combat checkpoints, then runs paired regressions over the same RNG worlds. |
| Combat has an enormous combinatorial space and requires exact handling of card order, targets, energy, and random draws | A model can propose plausible tactics but easily makes mistakes in damage arithmetic, long action sequences, and extreme branches | Combat Tool sends the complete combat state to a fast simulator and MCTS; the LLM does not mentally simulate an entire battle step by step. |
| A card's value depends on the existing deck, future encounters, and unfinished build structures | Decisions can collapse into static card rankings, superficial synergies, or inconsistent hallucinated reasoning | The Winning Path algorithm explicitly maintains build modules, immediate survival needs, and contextual expert evidence, passing only deterministic results or a constrained shortlist to Build Agent. |
| Combat, deck building, routing, and potion resources operate on different time scales but affect one another | One generic context struggles to preserve both turn-level precision and run-level planning; locally reasonable actions can undermine long-term goals | The Harness composes Combat, Build, and Map Agents by decision ownership and passes combat simulations, route facts, and build state across layers as structured evidence. |
| A run contains hundreds of commands and many nested selection screens; one protocol error can stop it | A model can select stale actions, use the wrong index, or lose the original task after a screen transition | The harness removes interaction correctness from model reasoning through stable-state adaptation, command validation, commit-after-confirmation, and scoped continuations. |

The Spire Agent harness guarantees interaction correctness, while independent
Tools use simulation, expert experience, and structured evidence to expand the
range of deterministic decisions. EvolveAgent turns accumulated run logs into
fixed datasets and repeatedly evaluates policies within a constrained parameter
space; only changes that pass the benchmark without regressing existing
capabilities are encoded as executable Tools. The LLM retains judgments where
evidence remains insufficient and open-ended tradeoffs are still required.

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

### MCTS search time and quality

MCTS search is currently the largest source of latency. The `mcts` section in
[config.yaml](config.yaml) controls the Combat search budget.
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
