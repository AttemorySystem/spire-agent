"""Stable framework contracts for Spire Agent."""

from .context import GameContext
from .contracts import (
    AgentKind,
    ContextEntry,
    ContextView,
    Continuation,
    ContinuationChange,
    Decision,
    DecisionRequest,
    DecisionScope,
    ExecutionResult,
    GameState,
    RoutedDecision,
    ScreenState,
    SessionRefresh,
)
from .game_agent import GameAgent
from .registry import SubAgentRegistry
from .router import RoomScopeRouter, RoutedDecisionProvider

__all__ = [
    "AgentKind",
    "ContextEntry",
    "ContextView",
    "Continuation",
    "ContinuationChange",
    "Decision",
    "DecisionRequest",
    "DecisionScope",
    "ExecutionResult",
    "GameAgent",
    "GameContext",
    "GameState",
    "RoomScopeRouter",
    "RoutedDecision",
    "RoutedDecisionProvider",
    "ScreenState",
    "SessionRefresh",
    "SubAgentRegistry",
]
