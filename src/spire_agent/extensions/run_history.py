"""Public durable run facts for observers and offline consumers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from spire_agent.contracts import ContextEntry, Decision, GameState
from .log_io import append_jsonl, jsonable, write_jsonl
from .run_directory import RunDirectory


HISTORY_FILENAME = "run_history.jsonl"
HISTORY_VERSION = 1
_FACT_KEYS = (
    "act",
    "floor",
    "room_type",
    "class",
    "ascension_level",
    "act_boss",
    "current_hp",
    "max_hp",
    "gold",
    "has_ruby_key",
    "has_sapphire_key",
    "has_emerald_key",
    "victory",
    "heart_kill",
    "killed_by",
)
_COMBAT_KEYS = (
    "turn",
    "player",
    "monsters",
    "enemies",
    "hand",
    "draw_pile",
    "discard_pile",
    "exhaust_pile",
)


class RunHistoryError(ValueError):
    """A run history file is malformed or unsupported."""


class RunHistoryRecorder:
    """Record confirmed facts for replay-independent offline consumers."""

    def __init__(self, directory: RunDirectory, replay: object) -> None:
        self._directory = directory
        self._replay = replay
        self._previous: GameState | None = None
        self._started: bool | None = None

    def on_entry(self, entry: ContextEntry) -> None:
        previous, self._previous = self._previous, entry.state
        replayed = bool(getattr(self._replay, "resume", False)) and (
            entry.index == 0
            or bool(getattr(self._replay, "last_execution_replayed", False))
        )
        if replayed:
            return
        self._start(previous or entry.state, partial=entry.index != 0)
        if previous is None:
            return
        append_jsonl(
            self.path,
            {
                "schema_version": HISTORY_VERSION,
                "type": "resync" if entry.command is None else "action",
                "entry_index": entry.index,
                "before": state_snapshot(previous),
                "action": _action(entry, previous),
                "after": state_snapshot(entry.state),
            },
        )

    @property
    def path(self) -> Path:
        return self._directory.path / HISTORY_FILENAME

    def _start(self, state: GameState, *, partial: bool) -> None:
        if self._started is None:
            _repair_partial(self.path)
            self._started = self.path.is_file() and self.path.stat().st_size > 0
        if self._started:
            return
        append_jsonl(
            self.path,
            {
                "schema_version": HISTORY_VERSION,
                "type": "run_start",
                "partial": partial,
                "state": state_snapshot(state),
            },
        )
        self._started = True


def state_snapshot(state: GameState) -> dict[str, Any]:
    """Return compact run facts without RNG or full map duplication."""

    facts = state.facts
    run = {key: jsonable(facts[key]) for key in _FACT_KEYS if key in facts}
    run["deck"] = _deck(facts.get("deck"))
    run["relics"] = _items(facts.get("relics"))
    run["potions"] = _items(facts.get("potions"), empty_as_none=True)
    combat = state.combat or {}
    return {
        "owner": state.owner_hint.value,
        "scope_id": state.scope_id,
        "terminal": state.terminal,
        "run": run,
        "screen": {
            "type": state.screen.type,
            "commands": list(state.screen.commands),
            "choices": jsonable(state.screen.choices),
            "current_action": state.screen.current_action,
            "details": jsonable(state.screen.details),
        },
        "combat": {
            key: jsonable(combat[key]) for key in _COMBAT_KEYS if key in combat
        }
        if combat
        else None,
    }


def _events(path: Path) -> list[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RunHistoryError(f"cannot read run history {path}: {error}") from error
    result = []
    lines = text.splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) and not text.endswith("\n"):
                break
            raise RunHistoryError(f"invalid JSON on history line {index}: {error}") from error
        if not isinstance(value, Mapping):
            raise RunHistoryError(f"history line {index} is not an object")
        if value.get("schema_version") != HISTORY_VERSION:
            raise RunHistoryError(f"unsupported history version on line {index}")
        result.append(value)
    return result


def _repair_partial(path: Path) -> None:
    if not path.is_file():
        return
    try:
        complete = path.read_text(encoding="utf-8").endswith("\n")
    except OSError as error:
        raise RunHistoryError(f"cannot read run history {path}: {error}") from error
    if not complete:
        write_jsonl(path, list(_events(path)))


def _action(entry: ContextEntry, before: GameState) -> dict[str, Any]:
    decision = entry.decision
    return {
        "command": entry.command,
        "label": _action_label(before, entry.command),
        "confirmed": entry.confirmed,
        "error": entry.error,
        "scope": None
        if entry.scope is None
        else {"owner": entry.scope.owner.value, "id": entry.scope.id},
        "decision": _decision(decision),
    }


def _decision(decision: Decision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    payload = {
        key: jsonable(value)
        for key, value in decision.payload.items()
        if key != "build_exchange"
    }
    return {
        "source": decision.source,
        "reason": decision.reason,
        "payload": payload,
        "metrics": jsonable(decision.metrics),
    }


def _action_label(state: GameState, command: str | None) -> str | None:
    if not command:
        return "external state synchronization"
    parts = command.split()
    if len(parts) >= 2 and parts[1].isdigit():
        index = int(parts[1])
        if parts[0] == "choose" and index < len(state.screen.choices):
            return f"choose {_name(state.screen.choices[index])}"
        hand = state.combat.get("hand", ()) if state.combat else ()
        hand_index = index - 1
        if (
            parts[0] == "play"
            and _is_sequence(hand)
            and 0 <= hand_index < len(hand)
        ):
            label = f"play {_name(hand[hand_index])}"
            monsters = _monsters(state.combat or {})
            if len(parts) >= 3 and parts[2].isdigit() and int(parts[2]) < len(monsters):
                label += f" on {_name(monsters[int(parts[2])])}"
            return label
    return command


def _deck(value: object) -> list[dict[str, Any]]:
    counts: dict[str, list[int]] = {}
    for card in value if _is_sequence(value) else ():
        name = _name(card).removesuffix("+")
        count, upgraded = counts.setdefault(name, [0, 0])
        counts[name] = [count + 1, upgraded + int(_upgraded(card))]
    return [
        {"name": name, "count": count, "upgrades": upgrades}
        for name, (count, upgrades) in sorted(counts.items())
    ]


def _items(value: object, *, empty_as_none: bool = False) -> list[str | None]:
    result: list[str | None] = []
    for item in value if _is_sequence(value) else ():
        name = _name(item)
        if empty_as_none and name in {"", "Potion Slot"}:
            result.append(None)
        elif name:
            result.append(name)
    return result


def _monsters(combat: Mapping[str, Any]) -> Sequence[Any]:
    value = combat.get("monsters", combat.get("enemies", ()))
    return value if _is_sequence(value) else ()


def _name(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "label", "id"):
            if value.get(key):
                return str(value[key])
    return str(value)


def _upgraded(value: object) -> bool:
    if not isinstance(value, Mapping):
        return str(value).endswith("+")
    raw = value.get("upgrades", value.get("upgrade_count", 0))
    return bool(raw) or str(value.get("name") or "").endswith("+")


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


__all__ = [
    "HISTORY_FILENAME",
    "HISTORY_VERSION",
    "RunHistoryError",
    "RunHistoryRecorder",
    "state_snapshot",
]
