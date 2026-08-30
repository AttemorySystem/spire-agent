# Spire Agent installation

[English](install.md) | [简体中文](install_CN.md)

This guide installs Spire Agent from a fresh clone and starts a visible local
Slay the Spire run. It covers macOS, Linux, and Windows, followed by optional
replay and troubleshooting instructions.

The repository does not contain the game or Steam Workshop JAR files.

A working installation has four parts:

1. the Python agent in this repository;
2. `gym-sts`, which launches and communicates with the game;
3. `sts_lightspeed`, which provides combat MCTS and card evaluation;
4. the game JARs, Workshop mods, and Spire Agent's Java support mods.

## 1. Required software

All platforms need:

- a Steam installation of Slay the Spire;
- Git;
- Python 3.11 or newer, normally managed by `uv`;
- `uv`;
- CMake 3.19 or newer;
- a C++17 compiler.

## 2. Clone the repository

```bash
git clone --recurse-submodules \
  https://github.com/AttemorySystem/spire-agent.git
cd spire-agent
```

If the repository was cloned without `--recurse-submodules`, initialize the
dependencies afterward:

```bash
git submodule update --init --recursive --depth 1
```

## 3. Install the Python environment

From the agent repository root:

```bash
uv sync
```

## 4. Prepare the game and Workshop JARs

```bash
mkdir -p runtime/lib runtime/mods
```

The final input layout must be:

```text
runtime/lib/
  desktop-1.0.jar
  ModTheSpire.jar

runtime/mods/
  BaseMod.jar
  CommunicationMod.jar
```

The relevant Steam Workshop item directories are:

| JAR | Workshop item |
|---|---:|
| `ModTheSpire.jar` | `1605060445` |
| `BaseMod.jar` | `1605833019` |
| `CommunicationMod.jar` | `2131373661` |

Subscribe to those items in Steam and let Steam finish downloading them before
copying the files.

### macOS

The default Steam root is:

```text
~/Library/Application Support/Steam
```

From the agent repository root:

```bash
STEAM_ROOT="$HOME/Library/Application Support/Steam"
STS_DIR="$STEAM_ROOT/steamapps/common/SlayTheSpire"
WORKSHOP_DIR="$STEAM_ROOT/steamapps/workshop/content/646570"

cp "$STS_DIR/desktop-1.0.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605060445/ModTheSpire.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605833019/BaseMod.jar" runtime/mods/
cp "$WORKSHOP_DIR/2131373661/CommunicationMod.jar" runtime/mods/
```

If the game is stored in another Steam library, change `STEAM_ROOT` and the
other environment variables as needed.

### Linux

Common Steam roots are:

```text
~/.local/share/Steam
~/.steam/steam
~/.var/app/com.valvesoftware.Steam/.local/share/Steam
```

For the normal native Steam installation:

```bash
STEAM_ROOT="$HOME/.local/share/Steam"
STS_DIR="$STEAM_ROOT/steamapps/common/SlayTheSpire"
WORKSHOP_DIR="$STEAM_ROOT/steamapps/workshop/content/646570"

cp "$STS_DIR/desktop-1.0.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605060445/ModTheSpire.jar" runtime/lib/
cp "$WORKSHOP_DIR/1605833019/BaseMod.jar" runtime/mods/
cp "$WORKSHOP_DIR/2131373661/CommunicationMod.jar" runtime/mods/
```

Change `STEAM_ROOT` for Flatpak Steam or a custom Steam library.

### Windows

Run these commands in PowerShell from the repository root. The example assumes
Steam is installed in its default location:

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

Change `$SteamRoot` when the game is stored in another Steam library.

## 5. Install the Java support mods

`AgentStateFixes` is required for every live run and replay. It exposes state
needed for stable decision boundaries, combat conversion, and RNG restoration.

`AgentVisualizer` is required only when `run.hud: true`.

[GitHub Releases](https://github.com/AttemorySystem/spire-agent/releases)
contain both JAR files. Download them and place them in `runtime/mods/`.

### Build from source (optional)

Install a full JDK containing `javac` and `jar`. JDK 8 is the safest build
environment; newer JDKs also work because the scripts target Java 8 bytecode.

#### macOS and Linux

```bash
./game_mods/agent_state_fixes/build.sh
./game_mods/agent_visualizer/build.sh
```

#### Windows

After ensuring the JDK is available, run the same scripts from Git Bash or the
MSYS2 UCRT64 terminal:

```bash
javac -version
./game_mods/agent_state_fixes/build.sh
./game_mods/agent_visualizer/build.sh
```

All platforms should now have:

```text
runtime/mods/
  AgentStateFixes.jar
  AgentVisualizer.jar
  BaseMod.jar
  CommunicationMod.jar
```

## 6. Compile `sts_lightspeed`

The live runtime expects two Release executables:

- `battle-sim` for combat MCTS;
- `card-reward-eval` for card and encounter evaluation.

### macOS and Linux

```bash
cmake -S 3rd/sts_lightspeed -B 3rd/sts_lightspeed/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 3rd/sts_lightspeed/build \
  --target battle-sim card-reward-eval -j
```

### Windows

Run from the MSYS2 UCRT64 terminal:

```bash
cmake -S 3rd/sts_lightspeed -B 3rd/sts_lightspeed/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build 3rd/sts_lightspeed/build \
  --target battle-sim card-reward-eval -j
3rd/sts_lightspeed/build/card-reward-eval.exe --self-test
```

Windows executables have an `.exe` suffix. If Python does not resolve that
suffix automatically, pass both executable paths explicitly when starting the
agent:

```powershell
uv run spire-agent `
  --no-tui `
  --mcts-binary 3rd\sts_lightspeed\build\battle-sim.exe `
  --card-eval-binary 3rd\sts_lightspeed\build\card-reward-eval.exe
```

## 7. Configure the agent and LLM

`config.yaml` is the single source of runtime defaults. CLI options override it
for one invocation. The normal configuration is:

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

Supported implementation names are:

| Setting | Values |
|---|---|
| `agents.map` | `llm` |
| `agents.build` | `winning_path`, `llm` |
| `agents.combat` | `mcts`, `llm` |
| `prompt_language` | `en`, `zh` |
| `run.window_size` | `WIDTHxHEIGHT`, `fullscreen` |
| `run.hud` | `true`, `false` |

Relative paths are resolved from the YAML file's directory.

`--config` reads YAML only; it does not load shell environment files.
`llm.base_url` and `llm.model` may be left empty to use `MODEL_URL` and
`MODEL`. The API key always comes from the launching shell or the operating
system's normal secret-management mechanism.

### macOS and Linux

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

## 8. Start the first run

```bash
uv run spire-agent --no-tui --character IRONCLAD --ascension 0 --seed 0
```

For Defect:

```bash
uv run spire-agent --no-tui --character DEFECT --ascension 0 --seed 0
```

A decimal seed initializes gym-sts's deterministic seed conversion. The game
returns its canonical alphanumeric Slay the Spire seed, which becomes the name
of the run directory. An alphanumeric seed is passed to the game exactly:

```bash
uv run spire-agent --no-tui --character IRONCLAD --ascension 20 --seed ABC123
```

The normal run loads these mod IDs:

```text
basemod, CommunicationMod, agentstatefixes
```

The game files are copied into `runtime/tmp/` for launch; the Steam installation
is not modified. Bridge files and game stderr go to `runtime/out/`; both
directories are disposable between runs.

Use the default terminal UI:

```bash
uv run spire-agent
```

Inside the console, common commands are:

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

Use Up/Down to browse and edit command history, and use `PageUp`/`PageDown` to
scroll the output area. HUD and display commands apply to the next launch.

## 9. Display mode and HUD

Set `run.window_size: fullscreen` to use the active display resolution. The
same mode can also be enabled for one invocation with:

```bash
uv run spire-agent --no-tui --character IRONCLAD --ascension 20 --seed 0 --fullscreen
```

HUD history is recorded for every run. Set `run.hud: true` in `config.yaml` to
load `AgentVisualizer.jar` and display that data in game. It works in either
windowed or fullscreen mode. HUD mode loads:

```text
basemod, CommunicationMod, agentstatefixes, agentvisualizer
```

HUD mode only adds `agentvisualizer`; gameplay uses the same mod set with or
without the HUD. On macOS a fullscreen run also keeps the machine awake while
the game is active.

## 10. Logs and replay

Each run writes durable artifacts under its canonical seed:

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

The game's process output is written to:

```text
runtime/out/stderr.log
```

Resume a crashed new-format run explicitly:

```bash
uv run spire-agent --no-tui --replay runs/ABC123
```

Replay holds each confirmed historical state for the configured
`replay.action_delay_seconds` (default `0.5`) before executing the next action.
Set it to `0`, or use `--replay-action-delay 0`, for the fastest replay. This is
a display-only delay and does not change settling or replay validation.

Replay validates every recorded decision boundary and dungeon RNG state. Do
not edit `replay.jsonl`, weaken a mismatch, or reuse a run directory. A replay
mismatch is an installation or determinism failure that must be investigated.

The normal startup rejects an existing `runs/<seed>` directory instead of
overwriting it.

## 11. Troubleshooting

### A dependency JAR is missing

The Java build scripts print the exact missing path. Confirm file names and
case exactly:

```bash
ls -l runtime/lib runtime/mods
```

Do not put `ModTheSpire.jar` only in `runtime/mods/`; the build and launcher
expect it in `runtime/lib/`.

### `transition_pending` or replay RNG state is missing

The loaded `AgentStateFixes.jar` is absent or stale. Rebuild it from the same
checkout and verify that the generated file in `runtime/mods/` is the one
copied into `runtime/tmp/mods/` on the next launch.

### The game closes before opening

Inspect:

```bash
tail -200 runtime/out/stderr.log
```

Typical causes are a missing JAR, a wrong mod version, or Java not being found.
The launcher tries to locate Steam's bundled game JRE automatically. For a
custom installation, point it at the JRE directory:

```bash
export STS_JRE_DIR="/path/to/SlayTheSpire/jre"
```

In PowerShell:

```powershell
$env:STS_JRE_DIR = "D:\SteamLibrary\steamapps\common\SlayTheSpire\jre"
```

### `battle-sim` or `card-reward-eval` is not found

Rebuild `sts_lightspeed` and confirm the expected path:

```bash
ls -l 3rd/sts_lightspeed/build/battle-sim \
  3rd/sts_lightspeed/build/card-reward-eval
```

On Windows, use the `.exe` paths and pass the two CLI overrides shown above.

### A run directory already exists

This is intentional protection. Use a different seed, move the old directory
to an archive, or explicitly replay it. Never delete or overwrite a run merely
to bypass the check.

### Steam is installed in a custom library

Find the three Workshop JARs and `desktop-1.0.jar`, then copy them into the
repository layout. On macOS or Linux:

```bash
find /path/to/SteamLibrary -type f \( \
  -name 'desktop-1.0.jar' -o \
  -name 'ModTheSpire.jar' -o \
  -name 'BaseMod.jar' -o \
  -name 'CommunicationMod.jar' \
\)
```
