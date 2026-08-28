# Agent Visualizer

`AgentVisualizer` is a display-only BaseMod extension. It polls the atomic JSON
snapshot named by `AGENT_OVERLAY_STATE` and renders five read-only views:

- the Map Agent route through the current Act Boss;
- the permanent deck, grouped by card and upgrade count;
- the current Winning Path modules when that evidence exists;
- the latest completed MCTS root actions, win rates, and expected winning HP;
- confirmed actions for the current combat, or the latest non-combat decision.

There is no TODO or Boss-readiness panel. The mod never calls an LLM, runs a
search, sends a game command, or writes data back to the agent.
HUD labels and entity names follow the Slay the Spire language: simplified or
traditional Chinese uses the bundled Chinese names; other languages use English.

Python records each confirmed frame in `runs/<seed>/hud_history.jsonl`. With
the HUD enabled, provider-exposed LLM reasoning—or response text when reasoning
is unavailable—is streamed to the live action panel. Final reasoning is retained
in the confirmed frame. Replay reads that frame without calling the LLM or
reproducing token timing. A missing or malformed display artifact disables only
the overlay; gameplay and strict replay remain authoritative.

Build from the repository root:

```bash
game_mods/agent_visualizer/build.sh
```

Enable `run.hud: true` in `config.yaml` to load `agentvisualizer` in either a
windowed or fullscreen run.
