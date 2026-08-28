"""MapAgent contract and composition."""

from __future__ import annotations

from abc import ABC, abstractmethod

from spire_agent.contracts import Decision, DecisionRequest

from .agents import MapAgent


class MapDecisionError(RuntimeError):
    pass


class MapTool(ABC):
    """Replaceable implementation used by MapAgent."""

    @abstractmethod
    def try_decide(self, request: DecisionRequest) -> Decision | None:
        """Return a map decision, or None when the request is not a map screen."""


def create_map_agent(tool: MapTool) -> MapAgent:
    return MapAgent(tool_stages=(tool,))


__all__ = ["MapDecisionError", "MapTool", "create_map_agent"]
