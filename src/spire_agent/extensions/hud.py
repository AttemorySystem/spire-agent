"""Record and replay display-only frames for the in-game HUD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, MutableMapping

from spire_agent.contracts import ContextEntry, GameState
from spire_agent.extensions.log_io import append_jsonl, write_json
from spire_agent.extensions.run_directory import RunDirectory
from spire_agent.extensions.run_history import state_snapshot


HISTORY_FILENAME = "hud_history.jsonl"
HISTORY_VERSION = 1
HUD_SCHEMA_VERSION = 4
_ROOM_SYMBOLS = {
    "Monster": "M",
    "Elite": "E",
    "Burning Elite": "E+",
    "Event": "?",
    "Rest": "R",
    "Shop": "$",
    "Chest": "T",
}
_STARTER_ORDER = ("strike", "defend")
_MAX_LLM_TEXT = 8_000
_STREAM_REFRESH_SECONDS = 0.1


def prepare_display(
    *,
    environ: MutableMapping[str, str] = os.environ,
    socket_dir: Path = Path("/tmp/.X11-unix"),
    platform: str = sys.platform,
) -> str | None:
    """Recover the sole local Linux display, or return a useful error."""

    if not platform.startswith("linux") or environ.get("DISPLAY"):
        return None
    displays = sorted(
        path.name[1:]
        for path in socket_dir.glob("X[0-9]*")
        if path.name[1:].isdigit()
    )
    if len(displays) == 1:
        environ["DISPLAY"] = f":{displays[0]}"
        return None
    if not displays:
        return "game display requires DISPLAY or one local X11/Xwayland socket"
    return "game display cannot infer DISPLAY from multiple local sockets"


class HudObserver:
    """Record live frames and replay those exact frames at confirmed entries."""

    def __init__(
        self,
        directory: RunDirectory,
        replay: object,
        overlay_path: str | Path,
    ) -> None:
        self.directory = directory
        self.replay = replay
        self.overlay_path = Path(overlay_path)
        self.projector = HudProjector()
        self.records: dict[int, Mapping[str, Any]] = {}
        self.sequence = 0
        self.loaded = False
        self.recording = True
        self.displaying = True
        self._frame: dict[str, Any] | None = None
        self._llm_activity: dict[str, str] | None = None
        self._stream_updated = 0.0
        try:
            self.overlay_path.unlink(missing_ok=True)
        except OSError:
            self.displaying = False

    def on_entry(self, entry: ContextEntry) -> None:
        if not entry.confirmed:
            return
        self._load()
        replayed = bool(getattr(self.replay, "resume", False)) and (
            entry.index == 0
            or bool(getattr(self.replay, "last_execution_replayed", False))
        )
        if replayed:
            record = self.records.get(entry.index)
            if record is not None:
                frame = record.get("frame")
                if isinstance(frame, Mapping):
                    self.projector.restore(frame, entry.state)
                    self._frame = deepcopy(dict(frame))
                    self._llm_activity = None
                    self._show(self._frame)
                    return
            self._frame = None
            self._llm_activity = None
            self._hide()
            return

        self.sequence += 1
        frame = self.projector.project(
            entry,
            self.directory.path / "mcts",
            sequence=self.sequence,
        )
        if self._llm_activity:
            reasoning = self._llm_activity["reasoning"].strip()
            if reasoning:
                frame["action_panel"]["history"] = reasoning[-_MAX_LLM_TEXT:]
        self._frame = frame
        self._llm_activity = None
        record = {
            "version": HISTORY_VERSION,
            "entry_index": entry.index,
            "frame": frame,
        }
        if not self.recording:
            return
        try:
            append_jsonl(self.directory.path / HISTORY_FILENAME, record)
            self.records[entry.index] = record
        except OSError:
            self.recording = False
            self.displaying = False
            return
        self._show(frame)

    def on_llm_event(self, event: str, value: str) -> None:
        """Display one provider stream without affecting game decisions."""

        if event == "start":
            self._llm_activity = {
                "context": _llm_context(value),
                "reasoning": "",
                "content": "",
                "error": "",
            }
            self._show_llm_activity(force=True)
            return
        if self._llm_activity is None:
            return
        if event in {"reasoning", "content", "error"} and value:
            current = self._llm_activity[event]
            self._llm_activity[event] = (current + value)[-_MAX_LLM_TEXT:]
        self._show_llm_activity(force=event in {"done", "error"})

    def _show_llm_activity(self, *, force: bool) -> None:
        if self._frame is None or self._llm_activity is None:
            return
        now = time.monotonic()
        if not force and now - self._stream_updated < _STREAM_REFRESH_SECONDS:
            return
        activity = self._llm_activity
        frame = deepcopy(self._frame)
        frame["action_panel"] = {
            "context": activity["context"],
            "action": "LLM failed" if activity["error"] else "Thinking ...",
            "history": (
                activity["error"]
                or activity["reasoning"]
                or activity["content"]
            ),
        }
        self._show(frame)
        self._stream_updated = now

    def _load(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        path = self.directory.path / HISTORY_FILENAME
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                record = json.loads(line)
                index = record.get("entry_index")
                frame = record.get("frame")
                if (
                    record.get("version") == HISTORY_VERSION
                    and type(index) is int
                    and isinstance(frame, Mapping)
                    and frame.get("schema_version") == HUD_SCHEMA_VERSION
                ):
                    self.records[index] = record
                    sequence = frame.get("sequence")
                    if type(sequence) is int:
                        self.sequence = max(self.sequence, sequence)
            except (AttributeError, json.JSONDecodeError):
                continue

    def _show(self, frame: Mapping[str, Any]) -> None:
        if not self.displaying:
            return
        try:
            self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.overlay_path, dict(frame))
        except OSError:
            self.displaying = False

    def _hide(self) -> None:
        try:
            self.overlay_path.unlink(missing_ok=True)
        except OSError:
            self.displaying = False


class HudProjector:
    """Deterministically project immutable run facts into one small UI frame."""

    def __init__(self) -> None:
        self.previous: GameState | None = None
        self.route: dict[str, Any] = _empty_route()
        self.strategy = {"status": "", "name": "Waiting for a build"}
        self.mcts = {"actions": []}
        self.card_order: list[str] = []
        self.combat_actions: list[str] = []

    def restore(self, frame: Mapping[str, Any], state: GameState) -> None:
        self.route = _mapping_copy(frame.get("map_panel"), _empty_route())
        self.strategy = _mapping_copy(
            frame.get("strategy_panel"), self.strategy
        )
        self.mcts = _mapping_copy(frame.get("mcts_panel"), self.mcts)
        build = frame.get("build_panel")
        cards = build.get("cards") if isinstance(build, Mapping) else ()
        self.card_order = [
            str(card.get("name") or "").casefold()
            for card in _sequence(cards)
            if isinstance(card, Mapping) and card.get("name")
        ]
        action = frame.get("action_panel")
        history = action.get("history") if isinstance(action, Mapping) else ""
        self.combat_actions = str(history or "").splitlines() if state.combat else []
        self.previous = state

    def project(
        self,
        entry: ContextEntry,
        mcts_directory: Path,
        *,
        sequence: int,
    ) -> dict[str, Any]:
        before, after = self.previous, entry.state
        run = state_snapshot(after)["run"]
        self._update_route(entry, run)
        self._update_strategy(entry)
        self._update_mcts(entry, before, mcts_directory)
        action = self._action_panel(entry, before, after)
        self.previous = after
        return {
            "schema_version": HUD_SCHEMA_VERSION,
            "sequence": sequence,
            "run": {"floor": run.get("floor")},
            "map_panel": deepcopy(self.route),
            "strategy_panel": deepcopy(self.strategy),
            "mcts_panel": deepcopy(self.mcts),
            "build_panel": self._build_panel(run.get("deck")),
            "action_panel": action,
        }

    def _update_route(self, entry: ContextEntry, run: Mapping[str, Any]) -> None:
        payload = entry.decision.payload if entry.decision is not None else {}
        raw_route = payload.get("run_route")
        rooms = raw_route.get("planned_rooms") if isinstance(raw_route, Mapping) else None
        floor = _integer(run.get("floor"))
        act = _integer(run.get("act"))
        if rooms is not None:
            planned = [_room_name(room) for room in _sequence(rooms)]
            visible = [room for room in planned if room.casefold() != "boss"]
            self.route = {
                "act": act,
                "rooms": [
                    {
                        "floor": None if floor is None else floor + offset,
                        "symbol": _ROOM_SYMBOLS.get(name, "?"),
                        "name": name,
                        "current": offset == 0,
                    }
                    for offset, name in enumerate(visible)
                ],
                "boss": str(run.get("act_boss") or ""),
                "boss_floor": None if floor is None else floor + len(visible),
                "boss_current": False,
            }
        elif self.route.get("act") not in (None, act):
            self.route = {
                **_empty_route(),
                "act": act,
                "boss": str(run.get("act_boss") or ""),
            }
        else:
            self.route["rooms"] = [
                {**room, "current": room.get("floor") == floor}
                for room in _sequence(self.route.get("rooms"))
                if isinstance(room, Mapping)
                and (floor is None
                or _integer(room.get("floor")) is None
                or _integer(room.get("floor")) >= floor)
            ]
            self.route["boss_current"] = "boss" in str(
                run.get("room_type") or ""
            ).casefold()

    def _update_strategy(self, entry: ContextEntry) -> None:
        payload = entry.decision.payload if entry.decision is not None else {}
        modules = _active_modules(payload)
        if modules is None:
            return
        self.strategy = {
            "status": "ONLINE" if modules else "",
            "name": (
                " + ".join(_display_module(module) for module in modules[:3])
                if modules
                else "No active Winning Path module"
            ),
        }

    def _update_mcts(
        self,
        entry: ContextEntry,
        before: GameState | None,
        directory: Path,
    ) -> None:
        decision = entry.decision
        search_id = decision.metrics.get("search_id") if decision is not None else None
        if not isinstance(search_id, str) or not search_id:
            return
        try:
            audit = json.loads((directory / f"{search_id}.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        raw = audit.get("raw_result") if isinstance(audit, Mapping) else None
        roots = raw.get("rootActions") if isinstance(raw, Mapping) else None
        if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)):
            return
        selected = decision.command
        rows = [row for row in roots if isinstance(row, Mapping)]
        rows.sort(
            key=lambda row: (
                str(row.get("action") or "") == selected,
                _number(row.get("selectionValue")),
                _number(row.get("visits")),
            ),
            reverse=True,
        )
        self.mcts = {
            "actions": [
                _root_row(row, selected, before)
                for row in rows[:20]
            ]
        }

    def _build_panel(self, cards: object) -> dict[str, Any]:
        rows = [dict(card) for card in _sequence(cards) if isinstance(card, Mapping)]
        active = {str(card.get("name") or "").casefold() for card in rows}
        self.card_order = [name for name in self.card_order if name in active]
        if not self.card_order:
            self.card_order.extend(name for name in _STARTER_ORDER if name in active)
        for card in rows:
            name = str(card.get("name") or "").casefold()
            if name and name not in self.card_order:
                self.card_order.append(name)
        order = {name: index for index, name in enumerate(self.card_order)}
        rows.sort(key=lambda card: order.get(str(card.get("name") or "").casefold(), 9999))
        return {
            "card_count": sum(_integer(card.get("count")) or 0 for card in rows),
            "cards": rows,
        }

    def _action_panel(
        self,
        entry: ContextEntry,
        before: GameState | None,
        after: GameState,
    ) -> dict[str, Any]:
        label = _action_label(before, entry.command)
        if after.combat:
            if before is None or not before.combat:
                self.combat_actions = []
            if before is not None and before.combat and label:
                self.combat_actions.append(label)
            return {
                "context": "Combat",
                "action": "",
                "history": "\n".join(self.combat_actions),
            }
        self.combat_actions = []
        decision = entry.decision
        owner = entry.scope.owner.value.title() if entry.scope is not None else "Run"
        return {
            "context": owner,
            "action": label,
            "history": "" if decision is None else decision.reason[:8_000],
        }


def _active_modules(payload: Mapping[str, Any]) -> list[str] | None:
    for key in (
        "winning_path_review",
        "card_choice_review",
    ):
        review = payload.get(key)
        if not isinstance(review, Mapping):
            continue
        winning_path = review.get("winning_path", review)
        if not isinstance(winning_path, Mapping):
            continue
        state = winning_path.get("state")
        if isinstance(state, Mapping) and "active_modules" in state:
            return [str(value) for value in _sequence(state.get("active_modules"))]
    return None


def _llm_context(purpose: str) -> str:
    owner = str(purpose).partition(".")[0].casefold()
    return {
        "map": "Map Agent",
        "build": "Build Agent",
        "combat": "Combat Agent",
    }.get(owner, "Agent")


def _root_row(
    row: Mapping[str, Any],
    selected: str,
    state: GameState | None,
) -> dict[str, Any]:
    action = str(row.get("action") or "")
    rate = row.get("winSampleRate")
    win_rate = float(rate) if isinstance(rate, (int, float)) else None
    return {
        "label": _combat_label(action, state),
        "target": _combat_target(action, state),
        "selected": action == selected,
        "win_rate": win_rate,
        "end_hp": (
            row.get("expectedEndHpOnWin")
            if win_rate is not None and win_rate > 0
            else None
        ),
    }


def _combat_label(command: str, state: GameState | None) -> str:
    parts = command.split()
    if parts[:1] == ["play"] and len(parts) >= 2:
        hand = state.combat.get("hand", ()) if state is not None and state.combat else ()
        card = _indexed(hand, parts[1], one_based=True)
        return _name(card) or f"Card #{parts[1]}"
    if parts[:2] == ["potion", "use"] and len(parts) >= 3:
        potions = state.facts.get("potions", ()) if state is not None else ()
        return _name(_indexed(potions, parts[2])) or f"Potion #{parts[2]}"
    return {"end": "End turn"}.get(command, command)


def _combat_target(command: str, state: GameState | None) -> str:
    parts = command.split()
    raw_index = (
        parts[2]
        if parts[:1] == ["play"] and len(parts) >= 3
        else parts[3]
        if parts[:2] == ["potion", "use"] and len(parts) >= 4
        else None
    )
    if raw_index is None or state is None or not state.combat:
        return "-"
    monsters = state.combat.get("monsters", state.combat.get("enemies", ()))
    return _name(_indexed(monsters, raw_index)) or f"Monster #{raw_index}"


def _action_label(state: GameState | None, command: str | None) -> str:
    if state is None or not command:
        return ""
    parts = command.split()
    if parts[:1] == ["play"]:
        label = _combat_label(command, state)
        target = _combat_target(command, state)
        return f"Play {label}" + (f" -> {target}" if target != "-" else "")
    if parts[:1] == ["choose"] and len(parts) >= 2:
        choice = _indexed(state.screen.choices, parts[1])
        return f"Choose {_name(choice) or '#' + parts[1]}"
    return {
        "end": "End turn",
        "skip": "Skip",
        "proceed": "Proceed",
        "leave": "Leave",
        "confirm": "Confirm",
    }.get(command, command)


def _empty_route() -> dict[str, Any]:
    return {
        "act": None,
        "rooms": [],
        "boss": "",
        "boss_floor": None,
        "boss_current": False,
    }


def _mapping_copy(value: object, fallback: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else deepcopy(dict(fallback))


def _sequence(value: object) -> tuple[Any, ...]:
    return (
        tuple(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _indexed(value: object, raw_index: object, *, one_based: bool = False) -> Any:
    items = _sequence(value)
    try:
        index = int(str(raw_index)) - int(one_based)
    except (TypeError, ValueError):
        return None
    return items[index] if 0 <= index < len(items) else None


def _name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("name", "label", "id"):
            if value.get(key):
                return str(value[key])
    return ""


def _integer(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _display_module(value: str) -> str:
    return value.replace("_", " ").title()


def _room_name(value: object) -> str:
    name = str(value)
    return "Event" if name.casefold() in {"unknown", "?"} else name


__all__ = [
    "HISTORY_FILENAME",
    "HISTORY_VERSION",
    "HUD_SCHEMA_VERSION",
    "HudObserver",
    "prepare_display",
]
