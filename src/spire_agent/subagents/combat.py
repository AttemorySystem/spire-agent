"""CombatAgent contract and composition."""

from __future__ import annotations

from abc import ABC, abstractmethod

from spire_agent.contracts import Decision, DecisionRequest

from .agents import CombatAgent


class CombatTool(ABC):
    """Replaceable implementation used by CombatAgent."""

    @abstractmethod
    def try_decide(self, request: DecisionRequest) -> Decision | None:
        """Return a combat decision, including any continuation work."""


def create_combat_agent(tool: CombatTool) -> CombatAgent:
    return CombatAgent(tool_stages=(tool,))


__all__ = ["CombatTool", "create_combat_agent"]
