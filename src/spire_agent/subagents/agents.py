"""The three small SubAgent owners supported by spire_agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from spire_agent.contracts import AgentKind, Decision, DecisionRequest

from .pipeline import DecisionFallback, DecisionPipeline, DecisionStage


class PipelineSubAgent:
    """Thin owner wrapper around the shared decision pipeline."""

    __slots__ = ("_pipeline",)
    kind: ClassVar[AgentKind]

    def __init__(
        self,
        *,
        continuation_stages: Iterable[DecisionStage] = (),
        fast_paths: Iterable[DecisionStage] = (),
        tool_stages: Iterable[DecisionStage] = (),
        fallback: DecisionFallback | None = None,
    ) -> None:
        if not isinstance(getattr(type(self), "kind", None), AgentKind):
            raise TypeError("PipelineSubAgent subclasses must declare an AgentKind")
        self._pipeline = DecisionPipeline(
            continuation_stages=tuple(continuation_stages),
            fast_paths=tuple(fast_paths),
            tool_stages=tuple(tool_stages),
            fallback=fallback,
        )

    def decide(self, request: DecisionRequest) -> Decision:
        if request.scope.owner is not self.kind:
            raise ValueError(
                f"{self.kind.value} SubAgent received "
                f"{request.scope.owner.value} scope"
            )
        return self._pipeline.decide(request)


class BuildAgent(PipelineSubAgent):
    kind = AgentKind.BUILD


class MapAgent(PipelineSubAgent):
    kind = AgentKind.MAP


class CombatAgent(PipelineSubAgent):
    kind = AgentKind.COMBAT


__all__ = ["BuildAgent", "CombatAgent", "MapAgent", "PipelineSubAgent"]
