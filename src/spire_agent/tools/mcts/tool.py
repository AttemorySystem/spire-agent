"""Self-contained sts_lightspeed combat search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from spire_agent.contracts import (
    AgentKind,
    Continuation,
    ContinuationChange,
    Decision,
    DecisionRequest,
    GameState,
    frozen_mapping,
)
from spire_agent.subagents.combat import CombatTool
from spire_agent.tools.mcts.human_log import append_search_log


class MCTSError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MCTSResult:
    command: str
    follow_up: Mapping[str, Any] | None
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "follow_up", None if self.follow_up is None else frozen_mapping(self.follow_up))
        object.__setattr__(self, "metrics", frozen_mapping(self.metrics))


_SELECTION_KIND = "mcts_card_selection"
_PROCESS_GRACE_SECONDS = 30
_SELECTION_SCREENS = ("HAND_SELECT", "GRID", "CARD_REWARD")
_MAX_NATIVE_OUTPUT_BYTES = 16 * 1024 * 1024
_MAX_DIAGNOSTIC_BYTES = 128 * 1024


class DefaultCombatTool(CombatTool):
    """Default combat flow around an injected battle search."""

    def __init__(self, search: object, potion_gate: object | None = None) -> None:
        self._search = search
        self._potion_gate = potion_gate

    def try_decide(self, request: DecisionRequest) -> Decision | None:
        continuation = request.continuation
        if (
            continuation is not None
            and continuation.kind == _SELECTION_KIND
            and request.state.screen.type in continuation.expected_screens
        ):
            try:
                command, remaining = resolve_selection(
                    request.state, continuation.data
                )
            except MCTSError:
                if continuation.data.get("task") != generated_task(request.state):
                    raise
            else:
                change = (
                    ContinuationChange.clear()
                    if remaining is None
                    else ContinuationChange.set(
                        _selection_continuation(request, remaining)
                    )
                )
                return Decision(
                    command,
                    "combat.mcts_selection",
                    f"execute MCTS {continuation.data.get('task', '')} selection",
                    continuation=change,
                )

        state = request.state
        if not is_mcts_state(state):
            return None
        if (
            state.screen.type == "CARD_REWARD"
            and generated_task(state) is not None
            and len(state.screen.choices) == 1
            and "choose" in state.screen.commands
        ):
            return Decision(
                "choose 0",
                "combat.single_choice",
                "only generated-card candidate",
                continuation=_clear_stale(request),
            )
        choose = getattr(self._search, "choose", None)
        if not callable(choose):
            raise TypeError("CombatAgent search has no choose() method")
        active_slots = ()
        active = getattr(self._potion_gate, "active_slots", None)
        if callable(active):
            active_slots = tuple(active(state))
        result = (
            choose(
                state,
                potion_slots=active_slots,
                search_role="authorized_potion",
            )
            if active_slots
            else choose(state)
        )
        select = getattr(self._potion_gate, "select", None)
        if callable(select):
            result = select(state, result, choose)
        follow_up = getattr(result, "follow_up", None)
        if follow_up is not None and not isinstance(follow_up, Mapping):
            raise TypeError("MCTS follow_up must be a mapping or None")
        change = (
            ContinuationChange.set(_selection_continuation(request, follow_up))
            if follow_up is not None
            else _clear_stale(request)
        )
        return Decision(
            result.command,
            "combat.mcts",
            "battle search completed",
            continuation=change,
            metrics=result.metrics,
        )


class CombatMCTS:
    """Turn one stable GameState into one certified root command."""

    def __init__(
        self,
        binary: str | Path,
        run_directory: object,
        *,
        simulations: int = 100_000,
        threads: int = 12,
        max_time_ms: int = 10_000,
        adaptive_time_ms: int = 30_000,
        adaptive_simulations: int = 500_000,
    ) -> None:
        self.binary = Path(binary).resolve()
        self.runs = run_directory
        self.simulations = simulations
        self.threads = threads
        self.max_time_ms = max_time_ms
        self.adaptive_time_ms = adaptive_time_ms
        self.adaptive_simulations = adaptive_simulations
        if min(simulations, threads, max_time_ms, adaptive_time_ms, adaptive_simulations) <= 0:
            raise ValueError("MCTS limits must be positive")

    def choose(
        self,
        state: GameState,
        *,
        potion_slots: Sequence[int] = (),
        probe: bool = False,
        search_role: str = "baseline",
    ) -> MCTSResult:
        payload = encode_state(state)
        simulations, max_time = self._limits(state)
        adaptive_time = self.adaptive_time_ms
        adaptive_simulations = self.adaptive_simulations
        if probe:
            simulations = min(simulations, 20_000)
            max_time = min(max_time, 2_000)
            adaptive_time = min(adaptive_time, max_time)
            adaptive_simulations = min(adaptive_simulations, simulations)
        potion_slots = tuple(sorted({int(slot) for slot in potion_slots}))
        if potion_slots and not probe:
            simulations = min(simulations, 50_000)
            max_time = min(max_time, 5_000)
            adaptive_time = min(adaptive_time, max_time)
            adaptive_simulations = min(adaptive_simulations, simulations)
        process_timeout = (
            adaptive_time + max_time
        ) / 1000 + _PROCESS_GRACE_SECONDS
        settings = {
            "simulations_per_thread": simulations,
            "threads": self.threads,
            "max_time_ms": max_time,
            "adaptive_max_time_ms": adaptive_time,
            "adaptive_max_simulations": adaptive_simulations,
            "process_timeout_seconds": process_timeout,
            "allowed_potion_slots": list(potion_slots),
            "search_role": search_role,
        }
        binary = self._binary_info()
        started = time.perf_counter()
        stdout = stderr = ""
        stdout_overflow = False
        raw: object = None
        result: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            if not binary["available"]:
                raise MCTSError(f"battle-sim is unavailable: {self.binary}")
            with tempfile.TemporaryDirectory(prefix="sts-mcts-") as directory:
                input_file = Path(directory) / "input.json"
                input_file.write_text(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                stdout_file = Path(directory) / "stdout"
                stderr_file = Path(directory) / "stderr"
                command = [
                        str(self.binary),
                        str(input_file),
                        str(simulations),
                        str(self.threads),
                        str(max_time),
                        "0",
                    ]
                if potion_slots:
                    command.append(
                        "potion_slots="
                        + ",".join(str(slot) for slot in potion_slots)
                    )
                command.extend(
                    [
                        f"adaptive_max_time_ms={adaptive_time}",
                        f"adaptive_max_simulations={adaptive_simulations}",
                    ]
                )
                with stdout_file.open("w", encoding="utf-8") as stdout_stream, \
                        stderr_file.open("w", encoding="utf-8") as stderr_stream:
                    process = subprocess.run(
                        command,
                        cwd=directory,
                        text=True,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        timeout=process_timeout,
                        check=False,
                    )
                if isinstance(process.stdout, str):
                    stdout = process.stdout
                else:
                    stdout, stdout_overflow = _read_native_stdout(stdout_file)
                stderr = (
                    process.stderr
                    if isinstance(process.stderr, str)
                    else _read_bounded(stderr_file, _MAX_DIAGNOSTIC_BYTES)
                )
            if process.returncode:
                raise MCTSError(
                    f"battle-sim exited with status {process.returncode}: {stderr.strip()}"
                )
            if stdout_overflow:
                raise MCTSError(
                    "battle-sim output exceeded 16 MiB; native diagnostics are flooding"
                )
            try:
                raw = json.loads(stdout)
            except json.JSONDecodeError as cause:
                raise MCTSError(f"battle-sim returned invalid JSON: {cause}") from cause
            result = _parse_result(raw, state, potion_slots=potion_slots)
        except Exception as cause:
            error = cause if isinstance(cause, MCTSError) else MCTSError(
                f"battle-sim exceeded its {process_timeout:g}s process deadline"
                if isinstance(cause, subprocess.TimeoutExpired)
                else str(cause)
            )

        elapsed = round((time.perf_counter() - started) * 1000, 3)
        search_id = self._record(
            payload, settings, binary, raw, result, elapsed, error, stdout, stderr
        )
        if error is not None:
            raise error
        assert result is not None
        metrics = dict(result["metrics"])
        metrics.update(
            {
                "search_id": search_id,
                "latency_ms": elapsed,
                "allowed_potion_slots": list(potion_slots),
                "search_role": search_role,
            }
        )
        return MCTSResult(result["command"], result["follow_up"], metrics)

    def _limits(self, state: GameState) -> tuple[int, int]:
        monsters = state.combat.get("monsters", ()) if state.combat else ()
        ids = {
            str(monster.get("id") or "")
            for monster in _sequence(monsters)
            if isinstance(monster, Mapping)
        }
        if ids & {"SpireShield", "SpireSpear", "CorruptHeart"}:
            return self.adaptive_simulations, self.adaptive_time_ms
        return self.simulations, self.max_time_ms

    def _binary_info(self) -> dict[str, object]:
        try:
            stat = self.binary.stat()
        except OSError:
            return {"path": str(self.binary), "available": False}
        return {
            "path": str(self.binary),
            "available": self.binary.is_file() and bool(stat.st_mode & 0o111),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _record(
        self,
        request: Mapping[str, Any],
        settings: Mapping[str, Any],
        binary: Mapping[str, Any],
        raw: object,
        result: Mapping[str, Any] | None,
        elapsed: float,
        error: Exception | None,
        stdout: str,
        stderr: str,
    ) -> str:
        run_path = getattr(self.runs, "path", None)
        if not isinstance(run_path, Path):
            raise MCTSError("MCTS run directory is not bound")
        directory = run_path / "mcts"
        directory.mkdir(exist_ok=True)
        existing = [
            int(path.stem)
            for path in directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")
            if path.stem.isdigit()
        ]
        search_id = f"{max(existing, default=0) + 1:06d}"
        record = {
            "schema_version": 1,
            "search_id": search_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "status": "error" if error else "success",
            "elapsed_ms": elapsed,
            "request": request,
            "settings": settings,
            "binary": binary,
            "raw_result": raw,
            "result": _jsonable(result),
            "stdout": stdout if error else "",
            "stderr": stderr,
            "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
        }
        path = directory / f"{search_id}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        append_search_log(
            run_path, search_id, request, settings, raw, elapsed, error
        )
        return search_id


def _read_native_stdout(path: Path) -> tuple[str, bool]:
    overflow = path.stat().st_size > _MAX_NATIVE_OUTPUT_BYTES
    limit = _MAX_DIAGNOSTIC_BYTES if overflow else _MAX_NATIVE_OUTPUT_BYTES
    return _read_bounded(path, limit), overflow


def _read_bounded(path: Path, limit: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        text = stream.read(limit + 1)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[native output truncated]"


def is_mcts_state(state: GameState) -> bool:
    return bool(
        state.combat is not None
        and (
            state.screen.type == "NONE"
            and bool({"play", "end"} & set(state.screen.commands))
            or generated_task(state) is not None
        )
    )


def generated_task(state: GameState) -> str | None:
    if state.combat is None:
        return None
    task = {
        ("GRID", "BetterDiscardPileToHandAction"): "HOLOGRAM",
        ("HAND_SELECT", "RecycleAction"): "RECYCLE",
        ("GRID", "SeekAction"): "SEEK",
    }.get((state.screen.type, state.screen.current_action))
    if task is not None:
        return task
    if (
        state.screen.type == "HAND_SELECT"
        and state.screen.current_action == "GamblingChipAction"
        and state.screen.details.get("can_pick_zero") is True
        and "confirm" in state.screen.commands
    ):
        return "GAMBLE"
    if state.screen.type != "CARD_REWARD":
        return None
    return {
        "DiscoveryAction": "DISCOVERY",
        "CodexAction": "CODEX",
        "ChooseOneColorless": "TOOLBOX",
    }.get(state.screen.current_action)


def encode_state(state: GameState) -> dict[str, Any]:
    if state.combat is None:
        raise MCTSError("combat search requires an active combat state")
    game = {
        str(key): _jsonable(value)
        for key, value in state.facts.items()
        if key
        not in {
            "bridge",
            "replay_boundary_key",
            "replay_rng_state",
            "sts_seed",
        }
    }
    game.update(
        {
            "screen_type": state.screen.type,
            "screen_state": _jsonable(state.screen.details),
            "choice_list": _jsonable(state.screen.choices),
            "current_action": state.screen.current_action or None,
            "combat_state": _jsonable(state.combat),
        }
    )
    bridge = state.facts.get("bridge")
    bridge = bridge if isinstance(bridge, Mapping) else {}
    payload = {
        "available_commands": list(state.screen.commands),
        "ready_for_command": bridge.get("ready_for_command", True),
        "in_game": bridge.get("in_game", True),
        "game_state": game,
    }
    if task := generated_task(state):
        payload["mcts_card_select"] = {"task": task, "copy_count": 1}
    return payload


def resolve_selection(
    state: GameState,
    plan: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if state.screen.type not in {"HAND_SELECT", "GRID", "CARD_REWARD"}:
        raise MCTSError(f"MCTS selection reached unexpected screen {state.screen.type}")
    cards = list(_sequence(plan.get("cards")))
    completion = plan.get("completionCommand")
    if (
        cards
        and "choose" not in state.screen.commands
        and "confirm" in state.screen.commands
    ):
        return "confirm", dict(plan)
    if not cards:
        if completion is None and "confirm" in state.screen.commands:
            completion = "confirm"
        if completion not in {"confirm", "skip"}:
            raise MCTSError("completed MCTS selection has no final command")
        return _validate_command(state, str(completion)), None

    target = cards[0]
    if not isinstance(target, Mapping):
        raise MCTSError("MCTS selection card is malformed")
    choices = _visible_choices(state)
    selected = _selected_choice_indexes(state, choices)
    matches = [
        i
        for i, choice in enumerate(choices)
        if i not in selected and _card_matches(target, choice)
    ]
    source = target.get("sourceIndex")
    if isinstance(source, int) and source in matches:
        index = source
    elif matches:
        index = matches[0]
    else:
        raise MCTSError(f"MCTS selection target is not visible: {dict(target)!r}")
    remaining = dict(plan)
    remaining["cards"] = cards[1:]
    command = _validate_command(state, f"choose {index}")
    return command, remaining


def _selection_continuation(
    request: DecisionRequest,
    data: Mapping[str, object],
) -> Continuation:
    return Continuation(
        AgentKind.COMBAT,
        _SELECTION_KIND,
        request.scope.id,
        expected_screens=_SELECTION_SCREENS,
        data=data,
    )


def _clear_stale(request: DecisionRequest) -> ContinuationChange:
    continuation = request.continuation
    return (
        ContinuationChange.clear()
        if continuation is not None and continuation.kind == _SELECTION_KIND
        else ContinuationChange.keep()
    )


def _parse_result(
    raw: object,
    state: GameState,
    *,
    potion_slots: Sequence[int] = (),
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("protocolVersion") != 1:
        version = raw.get("protocolVersion") if isinstance(raw, Mapping) else None
        raise MCTSError(f"unsupported battle-sim protocolVersion: {version!r}")
    root_command = str(raw.get("rootCommand") or "")
    if generated_task(state) == "GAMBLE":
        command, follow_up = _gamble_selection(state, root_command)
    else:
        command = _validate_command(
            state, root_command, potion_slots=potion_slots
        )
        follow_up = raw.get("followUp")
    if follow_up is not None:
        if not isinstance(follow_up, Mapping) or follow_up.get("kind") != "card_selection":
            raise MCTSError("battle-sim followUp is malformed")
        cards = follow_up.get("cards")
        completion = follow_up.get("completionCommand")
        if not isinstance(cards, list) or completion not in {None, "confirm", "skip"}:
            raise MCTSError("battle-sim card selection is malformed")
        if any(
            not isinstance(card, Mapping)
            or not (card.get("id") or card.get("name"))
            for card in cards
        ):
            raise MCTSError("battle-sim selection card is malformed")
        follow_up = dict(follow_up)
    score = raw.get("score")
    score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None
    roots = raw.get("rootActions")
    roots = roots if isinstance(roots, Sequence) else ()
    selected = next(
        (
            row
            for row in roots
            if isinstance(row, Mapping) and row.get("action") == root_command
        ),
        {},
    )
    risk_keys = (
        "winSamples",
        "lossSamples",
        "cutoffSamples",
        "winSampleRate",
        "expectedEndHpOnWin",
        "expectedEffectiveEndHpOnWin",
        "expectedHpLossOnWin",
        "meanBestWinEndHp",
        "bestValue",
        "value",
        "visits",
    )
    return {
        "command": command,
        "follow_up": follow_up,
        "score": score,
        "metrics": {
            "score": score,
            "selection_policy": raw.get("rootSelectionPolicy"),
            "stop_reason": raw.get("searchStopReason"),
            "credible_win_evidence": raw.get("credibleWinEvidence"),
            "recovery_search": raw.get("recoverySearch"),
            "risk": {
                key: selected.get(key)
                for key in risk_keys
                if key in selected
            },
        },
    }


def _gamble_selection(
    state: GameState, command: str
) -> tuple[str, dict[str, Any] | None]:
    command = command.strip()
    if command == "choose none":
        return _validate_command(state, "confirm"), None
    names = re.findall(r"\((.*?)\)", command)
    if not names or not command.startswith("choose "):
        raise MCTSError(f"battle-sim returned malformed GAMBLE action {command!r}")
    choices = _visible_choices(state)
    cards, unused = [], set(range(len(choices)))
    for name in names:
        matches = [
            index
            for index in unused
            if str(choices[index].get("name") or "").casefold()
            == name.casefold()
        ]
        if not matches:
            raise MCTSError(f"GAMBLE target is not visible: {name!r}")
        index = min(matches)
        unused.remove(index)
        cards.append({**choices[index], "sourceIndex": index})
    first, remaining = cards[0], cards[1:]
    root = _validate_command(state, f"choose {first['sourceIndex']}")
    return root, {
        "kind": "card_selection",
        "task": "GAMBLE",
        "cards": remaining,
        "completionCommand": "confirm",
    }


def _validate_command(
    state: GameState,
    command: str,
    *,
    potion_slots: Sequence[int] = (),
) -> str:
    command = command.strip()
    family = command.split(" ", 1)[0] if command else ""
    if family not in state.screen.commands:
        raise MCTSError(f"battle-sim returned unavailable command {command!r}")
    if command == "end" or command in {"confirm", "skip"}:
        return command
    if match := re.fullmatch(r"choose (\d+)", command):
        if int(match.group(1)) >= len(_visible_choices(state)):
            raise MCTSError(f"battle-sim choose index is out of range: {command}")
        return command
    if match := re.fullmatch(r"play (\d+)(?: (\d+))?", command):
        hand = state.combat.get("hand", ()) if state.combat else ()
        if not 1 <= int(match.group(1)) <= len(_sequence(hand)):
            raise MCTSError(f"battle-sim hand index is out of range: {command}")
        if match.group(2) is not None:
            monsters = state.combat.get("monsters", ()) if state.combat else ()
            if int(match.group(2)) >= len(_sequence(monsters)):
                raise MCTSError(f"battle-sim target is out of range: {command}")
        return command
    if match := re.fullmatch(r"potion use (\d+)(?: (\d+))?", command):
        slot = int(match.group(1))
        if slot not in potion_slots:
            raise MCTSError(
                f"battle-sim returned unauthorized potion slot: {command}"
            )
        potions = _sequence(state.facts.get("potions"))
        if not 0 <= slot < len(potions):
            raise MCTSError(f"battle-sim potion slot is out of range: {command}")
        potion = potions[slot]
        if not isinstance(potion, Mapping) or potion.get("can_use") is False:
            raise MCTSError(f"battle-sim potion slot is not usable: {command}")
        target = match.group(2)
        requires_target = bool(potion.get("requires_target")) and str(
            potion.get("id") or potion.get("name") or ""
        ) != "Explosive Potion"
        if requires_target != (target is not None):
            raise MCTSError(f"battle-sim potion target is malformed: {command}")
        if target is not None:
            monsters = state.combat.get("monsters", ()) if state.combat else ()
            if int(target) >= len(_sequence(monsters)):
                raise MCTSError(
                    f"battle-sim potion target is out of range: {command}"
                )
        return command
    if command.startswith("potion "):
        raise MCTSError(f"battle-sim returned malformed potion command: {command}")
    raise MCTSError(f"battle-sim returned malformed command {command!r}")


def _visible_choices(state: GameState) -> list[dict[str, Any]]:
    choices = list(state.screen.choices)
    details = state.screen.details.get("cards", state.screen.details.get("hand", ()))
    details = list(_sequence(details))
    if not choices:
        choices = details
    result = []
    for index, value in enumerate(choices):
        choice = dict(value) if isinstance(value, Mapping) else {"name": str(value)}
        if len(details) == len(choices) and isinstance(details[index], Mapping):
            choice = {**details[index], **choice}
        result.append(choice)
    return result


def _card_matches(target: Mapping[str, Any], choice: Mapping[str, Any]) -> bool:
    for key in ("id", "name"):
        expected, actual = str(target.get(key) or ""), str(choice.get(key) or "")
        if expected and actual and expected.casefold() != actual.casefold():
            return False
    if not choice.get("id") and not choice.get("name"):
        return False
    for key in ("upgrades", "cost"):
        if target.get(key) is not None and choice.get(key) is not None:
            if int(target[key]) != int(choice[key]):
                return False
    return True


def _selected_choice_indexes(
    state: GameState, choices: Sequence[Mapping[str, Any]]
) -> set[int]:
    selected = _sequence(
        state.screen.details.get(
            "selected_cards", state.screen.details.get("selected", ())
        )
    )
    result: set[int] = set()
    for raw in selected:
        if not isinstance(raw, Mapping):
            continue
        uuid = str(raw.get("uuid") or "")
        matches = [
            index
            for index, choice in enumerate(choices)
            if index not in result
            and (
                uuid
                and uuid == str(choice.get("uuid") or "")
                or not uuid
                and _card_matches(raw, choice)
            )
        ]
        if matches:
            result.add(matches[0])
    return result


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


__all__ = [
    "CombatMCTS",
    "DefaultCombatTool",
    "MCTSError",
    "MCTSResult",
    "encode_state",
    "generated_task",
    "is_mcts_state",
    "resolve_selection",
]
