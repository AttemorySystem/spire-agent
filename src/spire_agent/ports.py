"""Narrow behavioral ports used by the frozen Spire Agent framework."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .contracts import (
    AgentKind,
    ContextEntry,
    ContextView,
    Decision,
    DecisionRequest,
    ExecutionResult,
    GameState,
    RoutedDecision,
    SessionRefresh,
)


class SubAgent(Protocol):
    """One BUILD, MAP, or COMBAT decision owner.

    Implementations may use deterministic fast paths, domain tools, an LLM,
    or any ordered combination of them.  The framework treats them equally.
    """

    kind: AgentKind

    def decide(self, request: DecisionRequest) -> Decision:
        ...


class DecisionProvider(Protocol):
    """Select a decision source for the current bounded context view.

    Normal play uses room routing plus the SubAgent registry.  Replay can
    provide recorded decisions through the same port without adding replay
    branches to GameAgent.
    """

    def decide(self, context: ContextView) -> RoutedDecision:
        ...


class GameSession(Protocol):
    """The sole port allowed to communicate with the live game."""

    def reset(self) -> GameState:
        ...

    def execute(self, command: str) -> ExecutionResult:
        ...

    def refresh(self) -> SessionRefresh:
        """Read and settle state after possible external game-window input."""

        ...

    def close(self) -> None:
        ...


class ActionValidator(Protocol):
    """Validate a concrete SubAgent command against the current state."""

    def validate(self, state: GameState, decision: Decision) -> None:
        ...


class DecisionControl(Protocol):
    """Pause automatic decisions and request synchronization when resumed."""

    def before_decision(self, context: ContextView) -> bool:
        """Return true when the session must refresh before deciding."""

        ...


class ContextReducer(Protocol):
    """Own application-specific shared state outside the framework."""

    def initialize(self, state: GameState) -> Mapping[str, object]:
        ...

    def reduce(
        self,
        shared: Mapping[str, object],
        entry: ContextEntry,
    ) -> Mapping[str, object]:
        ...


class RunObserver(Protocol):
    """Read confirmed or rejected command/state entries without deciding."""

    def on_entry(self, entry: ContextEntry) -> None:
        ...


class ObserverErrorSink(Protocol):
    def __call__(self, observer: RunObserver, error: Exception) -> None:
        ...


ObserverCollection = Sequence[RunObserver]


__all__ = [
    "ActionValidator",
    "ContextReducer",
    "DecisionProvider",
    "DecisionControl",
    "GameSession",
    "ObserverCollection",
    "ObserverErrorSink",
    "RunObserver",
    "SubAgent",
]
