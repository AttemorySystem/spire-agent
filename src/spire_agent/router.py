"""Coarse room/scope routing and normal routed decision provision."""

from __future__ import annotations

from dataclasses import replace

from .contracts import (
    Continuation,
    ContinuationChange,
    ContinuationOperation,
    ContextView,
    DecisionRequest,
    DecisionScope,
    RoutedDecision,
)
from .registry import SubAgentRegistry


class RoomScopeRouter:
    """Route by active continuation, otherwise by adapter owner hint.

    Page-specific interpretation intentionally belongs to the selected
    SubAgent.  The observation adapter is responsible only for the coarse
    BUILD/MAP/COMBAT room hint.
    """

    def route(self, context: ContextView) -> DecisionScope:
        continuation = _active_continuation(context)
        if continuation is not None:
            return DecisionScope(
                owner=continuation.owner,
                id=continuation.scope_id,
            )
        state = context.state
        return DecisionScope(owner=state.owner_hint, id=state.scope_id)


class RoutedDecisionProvider:
    """Build one bounded request and dispatch it through the registry."""

    def __init__(
        self,
        router: RoomScopeRouter,
        registry: SubAgentRegistry,
    ) -> None:
        self._router = router
        self._registry = registry

    def decide(self, context: ContextView) -> RoutedDecision:
        scope = self._router.route(context)
        continuation = _active_continuation(context)
        request = DecisionRequest(
            state=context.state,
            scope=scope,
            continuation=continuation,
            shared=context.shared,
            previous=context.last_entry,
        )
        decision = self._registry.get(scope.owner).decide(request)
        if (
            context.continuation is not None
            and continuation is None
            and decision.continuation.operation is ContinuationOperation.KEEP
        ):
            decision = replace(
                decision,
                continuation=ContinuationChange.clear(),
            )
        return RoutedDecision(scope=scope, decision=decision)


def _active_continuation(context: ContextView) -> Continuation | None:
    continuation = context.continuation
    if continuation is None:
        return None
    expected = continuation.expected_screens
    if expected and context.state.screen.type not in expected:
        return None
    return continuation


__all__ = ["RoomScopeRouter", "RoutedDecisionProvider"]
