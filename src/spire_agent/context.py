"""Authoritative command/state context for one Spire Agent game run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import (
    ContextEntry,
    ContextView,
    Continuation,
    ContinuationOperation,
    DecisionScope,
    ExecutionResult,
    GameState,
    RoutedDecision,
    frozen_mapping,
)
from .errors import ContextError
from .ports import ContextReducer


class EmptyContextReducer:
    """Default reducer for a framework-only run with no shared artifacts."""

    def initialize(self, state: GameState) -> Mapping[str, object]:
        return {}

    def reduce(
        self,
        shared: Mapping[str, object],
        entry: ContextEntry,
    ) -> Mapping[str, object]:
        return shared


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    base_entry_index: int
    routed: RoutedDecision


class GameContext:
    """The sole mutable framework state.

    SubAgents only receive ``ContextView``/``DecisionRequest`` projections.
    A continuation and application shared state change only after the matching
    command has been confirmed by GameSession.
    """

    def __init__(self, reducer: ContextReducer | None = None) -> None:
        self._reducer = reducer or EmptyContextReducer()
        self._entries: list[ContextEntry] = []
        self._shared: Mapping[str, object] = frozen_mapping()
        self._continuation: Continuation | None = None
        self._active_scope: DecisionScope | None = None
        self._pending: _PendingDecision | None = None

    @property
    def started(self) -> bool:
        return bool(self._entries)

    @property
    def current_state(self) -> GameState:
        if not self._entries:
            raise ContextError("GameContext has not been started")
        return self._entries[-1].state

    @property
    def entries(self) -> tuple[ContextEntry, ...]:
        """Detached append-only history for recorders and offline review."""

        return tuple(self._entries)

    def start(self, state: GameState) -> ContextEntry:
        if self._entries:
            raise ContextError("GameContext.start may only be called once")
        entry = ContextEntry(
            index=0,
            command=None,
            state=state,
            confirmed=True,
        )
        self._entries.append(entry)
        self._shared = frozen_mapping(self._reducer.initialize(state))
        return entry

    def view(self) -> ContextView:
        if not self._entries:
            raise ContextError("GameContext has not been started")
        return ContextView(
            state=self.current_state,
            active_scope=self._active_scope,
            continuation=self._continuation,
            shared=self._shared,
            last_entry=self._entries[-1],
            entry_count=len(self._entries),
        )

    def stage(self, routed: RoutedDecision) -> None:
        if not self._entries:
            raise ContextError("cannot stage before GameContext.start")
        if self._pending is not None:
            raise ContextError("previous decision has not been confirmed")
        if self.current_state.terminal:
            raise ContextError("cannot stage a decision for terminal state")
        self._active_scope = routed.scope
        self._pending = _PendingDecision(
            base_entry_index=self._entries[-1].index,
            routed=routed,
        )

    def resync(self, state: GameState) -> ContextEntry:
        """Replace stale state after an explicit external-control boundary."""

        if not self._entries:
            raise ContextError("cannot resync before GameContext.start")
        if self._pending is not None:
            raise ContextError("cannot resync while a decision is pending")
        entry = ContextEntry(
            index=len(self._entries),
            command=None,
            state=state,
            confirmed=True,
        )
        self._entries.append(entry)
        self._continuation = None
        self._active_scope = None
        self._shared = frozen_mapping(self._reducer.initialize(state))
        return entry

    def confirm(self, result: ExecutionResult) -> ContextEntry:
        pending = self._pending
        if pending is None:
            raise ContextError("no staged decision to confirm")
        if pending.base_entry_index != self._entries[-1].index:
            raise ContextError("context advanced after the decision was staged")
        decision = pending.routed.decision
        if result.command != decision.command:
            raise ContextError(
                "execution result does not match staged command: "
                f"{result.command!r} != {decision.command!r}"
            )

        entry = ContextEntry(
            index=len(self._entries),
            command=result.command,
            state=result.state,
            confirmed=result.confirmed,
            error=result.error,
            scope=pending.routed.scope,
            decision=decision,
        )
        self._entries.append(entry)
        self._pending = None

        if not result.confirmed:
            return entry

        change = decision.continuation
        if change.operation is ContinuationOperation.CLEAR:
            self._continuation = None
        elif change.operation is ContinuationOperation.SET:
            self._continuation = change.value

        self._shared = frozen_mapping(
            self._reducer.reduce(self._shared, entry)
        )
        return entry


__all__ = ["EmptyContextReducer", "GameContext"]
