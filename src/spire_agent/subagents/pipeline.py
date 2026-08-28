"""Shared decision pipeline for every small Spire Agent SubAgent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from spire_agent.contracts import Decision, DecisionRequest


class SubAgentDecisionError(RuntimeError):
    """A SubAgent cannot produce a valid decision for its request."""


class DecisionStage(Protocol):
    """One optional deterministic decision stage.

    A stage returns ``None`` when it does not own the current situation.  It
    must not execute game commands or mutate framework context.
    """

    def try_decide(self, request: DecisionRequest) -> Decision | None:
        ...


class DecisionFallback(Protocol):
    """The optional final policy, normally an LLM-backed implementation."""

    def decide(self, request: DecisionRequest) -> Decision:
        ...


@dataclass(frozen=True, slots=True)
class DecisionPipeline:
    """Run the four fixed SubAgent phases in order and fail closed."""

    continuation_stages: tuple[DecisionStage, ...] = ()
    fast_paths: tuple[DecisionStage, ...] = ()
    tool_stages: tuple[DecisionStage, ...] = ()
    fallback: DecisionFallback | None = None

    def decide(self, request: DecisionRequest) -> Decision:
        groups = (
            (
                "continuation",
                self.continuation_stages if request.continuation is not None else (),
            ),
            ("fast_path", self.fast_paths),
            ("tool", self.tool_stages),
        )
        for phase, stages in groups:
            for stage in stages:
                decision = stage.try_decide(request)
                if decision is not None:
                    return self._require_decision(decision, phase, stage)

        if self.fallback is None:
            raise SubAgentDecisionError(
                "no decision stage handled "
                f"{request.scope.owner.value}:{request.state.screen.type}"
            )
        decision = self.fallback.decide(request)
        return self._require_decision(decision, "fallback", self.fallback)

    @staticmethod
    def _require_decision(
        value: object,
        phase: str,
        source: object,
    ) -> Decision:
        if isinstance(value, Decision):
            return value
        name = type(source).__name__
        raise SubAgentDecisionError(
            f"{phase} stage {name} returned {type(value).__name__}, "
            "expected Decision or None"
        )


__all__ = [
    "DecisionFallback",
    "DecisionPipeline",
    "DecisionStage",
    "SubAgentDecisionError",
]
