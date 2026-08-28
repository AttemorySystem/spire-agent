"""Load the concrete runtime settings from config.yaml."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from spire_agent.subagents.llm import PromptLanguage


class AgentConfigError(ValueError):
    """The YAML runtime configuration is invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    map: str
    build: str
    combat: str
    prompt_language: PromptLanguage
    character: str
    ascension: int
    seed: str
    fullscreen: bool
    window_size: tuple[int, int]
    hud: bool
    runtime_dir: Path
    log_dir: Path
    mcts_binary: Path
    card_eval_binary: Path
    llm_base_url: str
    llm_model: str
    communication_timeout: float
    mcts_simulations: int
    mcts_threads: int
    mcts_max_time_ms: int
    mcts_adaptive_time_ms: int
    mcts_adaptive_simulations: int


_IMPLEMENTATIONS = {
    "map": {"llm"},
    "build": {"winning_path", "llm"},
    "combat": {"mcts", "llm"},
}
_KEYS = {
    "agents": set(_IMPLEMENTATIONS),
    "run": {"character", "ascension", "seed", "window_size", "hud"},
    "paths": {"runtime_dir", "log_dir", "mcts_binary", "card_eval_binary"},
    "llm": {"base_url", "model"},
    "communication": {"timeout"},
    "mcts": {
        "simulations",
        "threads",
        "max_time_ms",
        "adaptive_time_ms",
        "adaptive_simulations",
    },
}


def load_runtime_config(path: Path) -> RuntimeConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AgentConfigError(f"cannot read agent config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise AgentConfigError(f"invalid YAML in agent config {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise AgentConfigError("config must be a YAML mapping")
    unknown = set(raw) - ({"prompt_language"} | set(_KEYS))
    if unknown:
        raise AgentConfigError(f"unknown config keys: {sorted(unknown)!r}")
    groups = {name: _group(raw, name) for name in _KEYS}
    agents = groups["agents"]
    if not agents:
        raise AgentConfigError("agent config requires an 'agents' mapping")
    implementations = {
        name: str(agents.get(name) or "").strip() for name in _IMPLEMENTATIONS
    }
    for name, value in implementations.items():
        if value not in _IMPLEMENTATIONS[name]:
            raise AgentConfigError(
                f"unsupported {name} agent {value!r}; expected one of "
                f"{sorted(_IMPLEMENTATIONS[name])!r}"
            )
    try:
        language = PromptLanguage.parse(str(raw.get("prompt_language", "en")))
    except ValueError as error:
        raise AgentConfigError("prompt_language must be 'en' or 'zh'") from error

    run, paths = groups["run"], groups["paths"]
    character = str(run.get("character", "IRONCLAD")).strip().upper()
    if character not in {"IRONCLAD", "DEFECT"}:
        raise AgentConfigError("run.character must be IRONCLAD or DEFECT")
    seed = str(run.get("seed", "random")).strip()
    if not seed:
        raise AgentConfigError("run.seed must not be empty")
    window_size, fullscreen = _display(run.get("window_size", "1600x900"))
    hud = run.get("hud", False)
    if not isinstance(hud, bool):
        raise AgentConfigError("run.hud must be true or false")
    timeout = groups["communication"].get("timeout", 5.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise AgentConfigError("communication.timeout must be positive")
    mcts = {
        name: _integer(groups["mcts"], name, default)
        for name, default in {
            "simulations": 100_000,
            "threads": 12,
            "max_time_ms": 10_000,
            "adaptive_time_ms": 30_000,
            "adaptive_simulations": 500_000,
        }.items()
    }
    if not all(mcts.values()):
        raise AgentConfigError("mcts values must be positive")

    root = path.resolve().parent
    llm = groups["llm"]
    return RuntimeConfig(
        **implementations,
        prompt_language=language,
        character=character,
        ascension=_integer(run, "ascension", 20, maximum=20),
        seed=seed,
        fullscreen=fullscreen,
        window_size=window_size,
        hud=hud,
        runtime_dir=_path(root, paths.get("runtime_dir"), "runtime"),
        log_dir=_path(root, paths.get("log_dir"), "."),
        mcts_binary=_path(
            root, paths.get("mcts_binary"), "3rd/sts_lightspeed/build/battle-sim"
        ),
        card_eval_binary=_path(
            root,
            paths.get("card_eval_binary"),
            "3rd/sts_lightspeed/build/card-reward-eval",
        ),
        llm_base_url=str(llm.get("base_url") or "").strip(),
        llm_model=str(llm.get("model") or "").strip(),
        communication_timeout=float(timeout),
        **{f"mcts_{name}": value for name, value in mcts.items()},
    )


def _group(raw: Mapping, name: str) -> Mapping:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise AgentConfigError(f"config.{name} must be a mapping")
    unknown = set(value) - _KEYS[name]
    if unknown:
        raise AgentConfigError(
            f"unknown config keys: {sorted(f'{name}.{key}' for key in unknown)!r}"
        )
    return value


def _integer(
    group: Mapping, name: str, default: int, *, maximum: int | None = None
) -> int:
    value = group.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentConfigError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise AgentConfigError(f"{name} must be at most {maximum}")
    return value


def _path(root: Path, value: object, default: str) -> Path:
    path = Path(str(default if value is None else value)).expanduser()
    return path if path.is_absolute() else root / path


def _display(value: object) -> tuple[tuple[int, int], bool]:
    if not isinstance(value, str):
        raise AgentConfigError("run.window_size must be WIDTHxHEIGHT or fullscreen")
    normalized = value.strip().casefold()
    if normalized == "fullscreen":
        return (1600, 900), True
    parts = normalized.split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise AgentConfigError("run.window_size must be WIDTHxHEIGHT or fullscreen")
    width, height = map(int, parts)
    if width < 800 or height < 450:
        raise AgentConfigError("run.window_size must be at least 800x450")
    return (width, height), False


__all__ = ["AgentConfigError", "RuntimeConfig", "load_runtime_config"]
