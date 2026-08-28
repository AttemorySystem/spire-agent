"""Frozen data exchanged across the Spire Agent framework boundaries.

The framework deliberately knows nothing about individual cards, rooms,
screens, prompts, or simulator formats.  An adapter normalizes those details
into ``GameState``; a selected SubAgent receives one detached request and
returns one concrete command proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


def freeze(value: Any) -> Any:
    """Recursively detach JSON-like data and make it read-only."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(
        "framework payloads must contain only mappings, sequences, sets, "
        f"and scalar values; got {type(value).__name__}"
    )


def frozen_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    frozen = freeze(value or {})
    if not isinstance(frozen, Mapping):  # Defensive; ``freeze`` guarantees it.
        raise TypeError("expected a mapping")
    return frozen


class AgentKind(str, Enum):
    """The only three decision ownership scopes in spire_agent."""

    BUILD = "build"
    MAP = "map"
    COMBAT = "combat"


@dataclass(frozen=True, slots=True)
class ScreenState:
    """Normalized facts about the current interactive screen."""

    type: str
    commands: tuple[str, ...] = ()
    choices: tuple[Any, ...] = ()
    interaction_id: str = ""
    current_action: str = ""
    details: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", str(self.type).upper())
        object.__setattr__(
            self,
            "commands",
            tuple(
                str(command).strip()
                for command in self.commands
                if str(command).strip()
            ),
        )
        object.__setattr__(self, "choices", tuple(freeze(self.choices)))
        object.__setattr__(self, "interaction_id", str(self.interaction_id))
        object.__setattr__(self, "current_action", str(self.current_action))
        object.__setattr__(self, "details", frozen_mapping(self.details))


@dataclass(frozen=True, slots=True)
class GameState:
    """One immutable state returned by the game adapter.

    ``owner_hint`` is the adapter's coarse room/scope classification.  It is
    intentionally not a page handler.  A live continuation can override it so
    a GRID or CARD_REWARD opened by combat remains owned by CombatAgent.
    """

    owner_hint: AgentKind
    scope_id: str
    screen: ScreenState
    terminal: bool = False
    facts: Mapping[str, Any] = field(default_factory=frozen_mapping)
    combat: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", str(self.scope_id))
        object.__setattr__(self, "terminal", bool(self.terminal))
        object.__setattr__(self, "facts", frozen_mapping(self.facts))
        if self.combat is not None:
            object.__setattr__(self, "combat", frozen_mapping(self.combat))


@dataclass(frozen=True, slots=True)
class DecisionScope:
    owner: AgentKind
    id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))


@dataclass(frozen=True, slots=True)
class Continuation:
    """Opaque, owner-bound progress across nested game screens."""

    owner: AgentKind
    kind: str
    scope_id: str
    expected_screens: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "scope_id", str(self.scope_id))
        object.__setattr__(
            self,
            "expected_screens",
            tuple(str(item).upper() for item in self.expected_screens),
        )
        object.__setattr__(self, "data", frozen_mapping(self.data))


class ContinuationOperation(str, Enum):
    KEEP = "keep"
    CLEAR = "clear"
    SET = "set"


@dataclass(frozen=True, slots=True)
class ContinuationChange:
    """Transactional change applied only after a command is confirmed."""

    operation: ContinuationOperation = ContinuationOperation.KEEP
    value: Continuation | None = None

    def __post_init__(self) -> None:
        if self.operation is ContinuationOperation.SET and self.value is None:
            raise ValueError("SET continuation change requires a value")
        if (
            self.operation is not ContinuationOperation.SET
            and self.value is not None
        ):
            raise ValueError("only SET continuation change accepts a value")

    @classmethod
    def keep(cls) -> "ContinuationChange":
        return cls(ContinuationOperation.KEEP)

    @classmethod
    def clear(cls) -> "ContinuationChange":
        return cls(ContinuationOperation.CLEAR)

    @classmethod
    def set(cls, value: Continuation) -> "ContinuationChange":
        return cls(ContinuationOperation.SET, value)


@dataclass(frozen=True, slots=True)
class Decision:
    """One concrete command proposed by exactly one SubAgent."""

    command: str
    source: str
    reason: str = ""
    continuation: ContinuationChange = field(
        default_factory=ContinuationChange.keep
    )
    payload: Mapping[str, Any] = field(default_factory=frozen_mapping)
    metrics: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        command = str(self.command).strip()
        if not command:
            raise ValueError("Decision.command must not be empty")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "payload", frozen_mapping(self.payload))
        object.__setattr__(self, "metrics", frozen_mapping(self.metrics))

    @property
    def command_family(self) -> str:
        return self.command.split(" ", 1)[0]


@dataclass(frozen=True, slots=True)
class ContextEntry:
    """One command and the state observed after attempting it."""

    index: int
    command: str | None
    state: GameState
    confirmed: bool
    error: str | None = None
    scope: DecisionScope | None = None
    decision: Decision | None = None


@dataclass(frozen=True, slots=True)
class ContextView:
    """Bounded read-only projection; it never exposes full run history."""

    state: GameState
    active_scope: DecisionScope | None
    continuation: Continuation | None
    shared: Mapping[str, Any]
    last_entry: ContextEntry
    entry_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared", frozen_mapping(self.shared))


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """The complete framework input visible to one SubAgent call."""

    state: GameState
    scope: DecisionScope
    continuation: Continuation | None
    shared: Mapping[str, Any]
    previous: ContextEntry

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared", frozen_mapping(self.shared))


@dataclass(frozen=True, slots=True)
class RoutedDecision:
    scope: DecisionScope
    decision: Decision


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """The only result accepted at the command/state confirmation boundary."""

    command: str
    state: GameState
    confirmed: bool
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", str(self.command).strip())
        object.__setattr__(self, "confirmed", bool(self.confirmed))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))


@dataclass(frozen=True, slots=True)
class SessionRefresh:
    """A settled state read after external control may have changed the game."""

    state: GameState
    changed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed", bool(self.changed))


__all__ = [
    "AgentKind",
    "ContextEntry",
    "ContextView",
    "Continuation",
    "ContinuationChange",
    "ContinuationOperation",
    "Decision",
    "DecisionRequest",
    "DecisionScope",
    "ExecutionResult",
    "GameState",
    "RoutedDecision",
    "SessionRefresh",
    "ScreenState",
    "freeze",
    "frozen_mapping",
]
