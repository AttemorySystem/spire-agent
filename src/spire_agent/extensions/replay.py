"""Write-ahead deterministic replay runtime extension for new Spire Agent runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import time
from typing import Any

from spire_agent.contracts import (
    AgentKind,
    ContextView,
    Continuation,
    ContinuationChange,
    ContinuationOperation,
    Decision,
    DecisionScope,
    ExecutionResult,
    GameState,
    RoutedDecision,
    SessionRefresh,
)
from spire_agent.extensions.log_io import append_jsonl, jsonable, write_json, write_jsonl
from spire_agent.extensions.run_directory import RunDirectory
from spire_agent.ports import DecisionProvider, GameSession, RunObserver
from spire_agent.tools.game_stability import stable_boundary_key


REPLAY_FILENAME = "replay.jsonl"
REPLAY_UNAVAILABLE_FILENAME = "replay_unavailable.json"
REPLAY_VERSION = 5


class ReplayError(RuntimeError):
    """A replay artifact is missing, invalid, or diverged from the game."""


def _boundary_key(state: GameState) -> str:
    value = state.facts.get("replay_boundary_key")
    if not isinstance(value, str) or not value:
        raise ReplayError("game state has no replay boundary key")
    return value


def _rng_state(value: object) -> dict[str, tuple[int, int, int]]:
    if not isinstance(value, Mapping):
        raise ReplayError(
            "AgentStateFixes did not expose replay_rng_state; rebuild the mod"
        )
    result = {}
    for name, raw in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(raw, (list, tuple))
            or len(raw) != 3
            or not all(type(item) is int for item in raw)
        ):
            raise ReplayError(f"invalid replay RNG entry {name!r}")
        result[name] = (raw[0], raw[1], raw[2])
    if not result:
        raise ReplayError("replay RNG state is empty")
    return result


def rng_restore_command(state: Mapping[str, tuple[int, int, int]]) -> str:
    entries = (
        f"{name},{values[0]},{values[1]},{values[2]}"
        for name, values in sorted(state.items())
    )
    return "rng_restore " + ";".join(entries)


def restore_game_rng(
    send: Callable[[str], object],
    expected: Mapping[str, tuple[int, int, int]],
    boundary_key: str,
) -> None:
    """Restore RNG without creating an agent-visible gameplay transition."""

    try:
        observation = send(rng_restore_command(expected))
    except Exception as error:
        raise ReplayError(f"rng_restore command failed: {error}") from error
    raw = getattr(observation, "state", observation)
    if not isinstance(raw, Mapping):
        raise ReplayError("rng_restore returned an invalid observation")
    error = raw.get("error")
    if error not in (None, ""):
        raise ReplayError(f"rng_restore failed: {error}")
    game = raw.get("game_state")
    actual = _rng_state(
        game.get("replay_rng_state") if isinstance(game, Mapping) else None
    )
    if actual != dict(expected):
        raise ReplayError("game did not restore the recorded RNG state")
    if stable_boundary_key(observation) != boundary_key:
        raise ReplayError("rng_restore changed the logical game boundary")


def _boundary(state: GameState) -> dict[str, Any]:
    combat = state.combat or {}
    return {
        "act": state.facts.get("act"),
        "floor": state.facts.get("floor"),
        "screen": state.screen.type,
        "turn": combat.get("turn"),
        "choices": jsonable(state.screen.choices),
    }


def _continuation(change: ContinuationChange) -> dict[str, Any]:
    value = change.value
    return {
        "operation": change.operation.value,
        "value": None
        if value is None
        else {
            "owner": value.owner.value,
            "kind": value.kind,
            "scope_id": value.scope_id,
            "expected_screens": list(value.expected_screens),
            "data": jsonable(value.data),
        },
    }


def _routed_value(routed: RoutedDecision) -> dict[str, Any]:
    decision = routed.decision
    return {
        "scope": {"owner": routed.scope.owner.value, "id": routed.scope.id},
        "decision": {
            "command": decision.command,
            "source": decision.source,
            "reason": decision.reason,
            "continuation": _continuation(decision.continuation),
            "payload": jsonable(decision.payload),
        },
    }


def _routed(record: Mapping[str, Any]) -> RoutedDecision:
    raw_scope = record.get("scope")
    raw_decision = record.get("decision")
    if not isinstance(raw_scope, Mapping) or not isinstance(raw_decision, Mapping):
        raise ReplayError("replay action has no routed decision")
    raw_change = raw_decision.get("continuation")
    if not isinstance(raw_change, Mapping):
        raise ReplayError("replay decision has no continuation change")
    operation = ContinuationOperation(str(raw_change.get("operation")))
    raw_value = raw_change.get("value")
    value = None
    if isinstance(raw_value, Mapping):
        value = Continuation(
            owner=AgentKind(str(raw_value.get("owner"))),
            kind=str(raw_value.get("kind") or ""),
            scope_id=str(raw_value.get("scope_id") or ""),
            expected_screens=tuple(raw_value.get("expected_screens") or ()),
            data=raw_value.get("data") or {},
        )
    return RoutedDecision(
        DecisionScope(
            AgentKind(str(raw_scope.get("owner"))),
            str(raw_scope.get("id") or ""),
        ),
        Decision(
            command=str(raw_decision.get("command") or ""),
            source=str(raw_decision.get("source") or "replay"),
            reason=str(raw_decision.get("reason") or ""),
            continuation=ContinuationChange(operation, value),
            payload=raw_decision.get("payload") or {},
        ),
    )


class ReplayJournal:
    """Append live inputs before execution and consume them during resume."""

    def __init__(self, directory: RunDirectory, *, resume: bool = False) -> None:
        self.directory = directory
        self.resume = bool(resume)
        self.header: Mapping[str, Any] | None = None
        self.actions: list[Mapping[str, Any]] = []
        self.results: dict[int, Mapping[str, Any]] = {}
        self.cursor = 0
        self.active: Mapping[str, Any] | None = None
        self.active_record: Mapping[str, Any] | None = None
        self.active_replay = False
        self.last_execution_replayed = False
        self.disabled = False
        self.next_index = 0
        if self.resume:
            unavailable = self.directory.path / REPLAY_UNAVAILABLE_FILENAME
            if unavailable.is_file():
                raise ReplayError(
                    f"run is not replayable after manual takeover: {unavailable}"
                )
            self._load()

    @property
    def seed(self) -> str:
        if self.header is None:
            raise ReplayError("replay has no run header")
        return str(self.header.get("seed") or "")

    @property
    def character(self) -> str:
        if self.header is None:
            raise ReplayError("replay has no run header")
        return str(self.header.get("character") or "IRONCLAD")

    @property
    def ascension(self) -> int:
        if self.header is None:
            raise ReplayError("replay has no run header")
        value = self.header.get("ascension")
        if type(value) is not int or not 0 <= value <= 20:
            raise ReplayError(f"replay has invalid ascension: {value!r}")
        return value

    @property
    def path(self) -> Path:
        return self.directory.path / REPLAY_FILENAME

    def begin(self, state: GameState) -> None:
        key = _boundary_key(state)
        seed = str(state.facts.get("sts_seed") or "")
        if self.resume:
            if seed != self.seed:
                self.fail("seed_mismatch", expected=self.seed, actual=seed)
            expected = str(self.header.get("boundary_key") or "")
            if key != expected:
                self.fail("initial_boundary_mismatch", expected=expected, actual=key)
            return
        if self.header is not None:
            raise ReplayError("replay journal was already started")
        self.header = {
            "version": REPLAY_VERSION,
            "kind": "run",
            "seed": seed,
            "character": state.facts.get("class"),
            "ascension": state.facts.get("ascension_level"),
            "boundary_key": key,
            "boundary": _boundary(state),
        }
        append_jsonl(self.path, self.header)

    def disable(self, reason: str, **details: Any) -> None:
        """Permanently stop recording after unjournaled external input."""

        if self.disabled:
            return
        if self.active is not None:
            raise ReplayError("cannot disable replay while an action is active")
        write_json(
            self.directory.path / REPLAY_UNAVAILABLE_FILENAME,
            {
                "reason": str(reason),
                **jsonable(details),
            },
            sort_keys=True,
        )
        self.disabled = True
        self.last_execution_replayed = False

    def stage_live(self, context: ContextView, routed: RoutedDecision) -> None:
        if self.active is not None:
            raise ReplayError("previous replay action has not completed")
        state = context.state
        raw = {
            "version": REPLAY_VERSION,
            "kind": "action",
            "index": self.next_index,
            "boundary_key": _boundary_key(state),
            "boundary": _boundary(state),
            "rng": {
                name: list(values)
                for name, values in _rng_state(
                    state.facts.get("replay_rng_state")
                ).items()
            },
            **_routed_value(routed),
        }
        self.active = raw
        self.active_record = raw
        self.active_replay = False

    def stage_replay(self, context: ContextView) -> RoutedDecision | None:
        if self.cursor >= len(self.actions):
            return None
        if self.active is not None:
            raise ReplayError("previous replay action has not completed")
        action = self.actions[self.cursor]
        actual = _boundary_key(context.state)
        expected = str(action.get("boundary_key") or "")
        if actual != expected:
            self.fail(
                "boundary_mismatch",
                action_index=action["index"],
                expected=expected,
                actual=actual,
                expected_boundary=action.get("boundary") or {},
                actual_boundary=_boundary(context.state),
            )
        self.active = action
        self.active_record = None
        self.active_replay = True
        return _routed(action)

    def prepare_execute(self, command: str) -> None:
        """Flush a validated live input immediately before game execution."""

        action = self.active
        if action is None or _routed(action).decision.command != command:
            raise ReplayError("game command does not match the staged replay action")
        if self.active_replay or self.active_record is None:
            return
        append_jsonl(self.path, self.active_record)
        self.active_record = None
        self.next_index += 1

    def complete(self, result: ExecutionResult) -> None:
        action = self.active
        if action is None or result.command != _routed(action).decision.command:
            raise ReplayError("execution does not match the staged replay action")
        index = int(action["index"])
        value = {
            "version": REPLAY_VERSION,
            "kind": "result",
            "index": index,
            "confirmed": result.confirmed,
            "error": result.error,
            "boundary_key": _boundary_key(result.state),
            "boundary": _boundary(result.state),
            "terminal": result.state.terminal,
        }
        expected = self.results.get(index)
        if self.active_replay and expected is not None:
            comparable = ("confirmed", "boundary_key", "terminal")
            if any(expected.get(key) != value.get(key) for key in comparable):
                next_rng = (
                    self.actions[self.cursor + 1].get("rng")
                    if self.cursor + 1 < len(self.actions)
                    else None
                )
                self.fail(
                    "result_mismatch",
                    action_index=index,
                    expected=dict(expected),
                    actual=value,
                    expected_next_rng=next_rng,
                    actual_rng=result.state.facts.get("replay_rng_state"),
                )
        else:
            append_jsonl(self.path, value)
            self.results[index] = value
        replayed = self.active_replay
        if replayed:
            self.cursor += 1
        if result.state.terminal and self.cursor >= len(self.actions):
            (self.directory.path / "replay_failure.json").unlink(missing_ok=True)
        self.active = None
        self.active_record = None
        self.active_replay = False
        self.last_execution_replayed = replayed

    def active_rng(self) -> tuple[Mapping[str, tuple[int, int, int]], str] | None:
        if self.active is None or not self.active_replay:
            return None
        return (
            _rng_state(self.active.get("rng")),
            str(self.active.get("boundary_key") or ""),
        )

    def _load(self) -> None:
        path = self.path
        if not path.is_file():
            raise ReplayError(f"new-format replay is missing: {path}")
        raw_actions: dict[int, Mapping[str, Any]] = {}
        rejected: set[int] = set()
        lines = path.read_bytes().splitlines()
        records: list[Mapping[str, Any]] = []
        for line_number, raw_line in enumerate(lines, 1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if line_number == len(lines):
                    write_jsonl(path, records)
                    break
                raise ReplayError(
                    f"invalid replay line {line_number}: {error}"
                ) from error
            if not isinstance(value, Mapping) or value.get("version") != REPLAY_VERSION:
                raise ReplayError(f"invalid replay record at line {line_number}")
            records.append(value)
            kind = value.get("kind")
            if kind == "run":
                if self.header is not None:
                    raise ReplayError("replay contains multiple run headers")
                self.header = value
            elif kind in {"action", "result"} and type(value.get("index")) is int:
                index = int(value["index"])
                if kind == "action":
                    raw_actions[index] = value
                else:
                    self.results[index] = value
                    if not value.get("confirmed"):
                        rejected.add(index)
            else:
                raise ReplayError(f"unknown replay record at line {line_number}")
        if self.header is None:
            raise ReplayError("replay has no run header")
        self.actions = [
            raw_actions[index]
            for index in sorted(raw_actions)
            if index not in rejected
        ]
        for action in self.actions:
            _rng_state(action.get("rng"))
            _routed(action)
            if not action.get("boundary_key"):
                raise ReplayError("replay action has no boundary key")
        self.next_index = max(raw_actions, default=-1) + 1

    def fail(self, reason: str, **details: Any) -> None:
        path = self.directory.path / "replay_failure.json"
        write_json(path, {"reason": reason, **jsonable(details)}, sort_keys=True)
        raise ReplayError(f"{reason}; details written to {path}")


class ReplayRuntime:
    """Replay-aware implementation of both application-level runtime ports."""

    def __init__(
        self,
        session: GameSession,
        decisions: DecisionProvider,
        journal: ReplayJournal,
        restore_rng: Callable[[Mapping[str, tuple[int, int, int]], str], None],
        *,
        action_delay_seconds: float = 0.0,
    ) -> None:
        if action_delay_seconds < 0:
            raise ValueError("replay action delay must be non-negative")
        self.session = session
        self.decisions = decisions
        self.journal = journal
        self.restore_rng = restore_rng
        self.action_delay_seconds = action_delay_seconds
        self.current_state: GameState | None = None

    def decide(self, context: ContextView) -> RoutedDecision:
        if self.journal.disabled:
            return self.decisions.decide(context)
        recorded = self.journal.stage_replay(context)
        if recorded is not None:
            return recorded
        routed = self.decisions.decide(context)
        self.journal.stage_live(context, routed)
        return routed

    def reset(self) -> GameState:
        state = self.session.reset()
        self.journal.begin(state)
        self.current_state = state
        return state

    def execute(self, command: str) -> ExecutionResult:
        if self.journal.disabled:
            result = self.session.execute(command)
            self.current_state = result.state
            self.journal.last_execution_replayed = False
            return result
        self.journal.prepare_execute(command)
        if self.journal.active_replay and self.action_delay_seconds:
            time.sleep(self.action_delay_seconds)
        restore = self.journal.active_rng()
        try:
            current_rng = (
                None
                if self.current_state is None
                else self.current_state.facts.get("replay_rng_state")
            )
            if restore is not None and _rng_state(current_rng) != dict(restore[0]):
                self.restore_rng(*restore)
            result = self.session.execute(command)
        except Exception as error:
            if restore is not None:
                action = self.journal.active
                self.journal.fail(
                    "replay_execution_failed",
                    action_index=None if action is None else action.get("index"),
                    command=command,
                    error=str(error),
                )
            raise
        self.journal.complete(result)
        self.current_state = result.state
        return result

    def refresh(self) -> SessionRefresh:
        refreshed = self.session.refresh()
        before = self.current_state
        self.current_state = refreshed.state
        if refreshed.changed:
            self.journal.disable(
                "manual game-window takeover changed the settled game state",
                before_boundary=None if before is None else _boundary(before),
                after_boundary=_boundary(refreshed.state),
            )
        return refreshed

    def close(self) -> None:
        self.session.close()


class LiveOnlyObserver:
    """Keep replay reconstruction out of append-only business logs."""

    def __init__(self, observer: RunObserver, journal: ReplayJournal) -> None:
        self.observer = observer
        self.journal = journal

    def on_entry(self, entry) -> None:
        if not self.journal.last_execution_replayed:
            self.observer.on_entry(entry)


__all__ = [
    "LiveOnlyObserver",
    "REPLAY_FILENAME",
    "REPLAY_UNAVAILABLE_FILENAME",
    "ReplayError",
    "ReplayJournal",
    "ReplayRuntime",
    "restore_game_rng",
    "rng_restore_command",
]
