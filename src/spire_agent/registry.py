"""Immutable mapping from decision scope to one SubAgent."""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping

from .contracts import AgentKind
from .errors import RegistryError
from .ports import SubAgent


class SubAgentRegistry:
    """Register exactly one BUILD, MAP, and COMBAT owner at construction."""

    def __init__(self, agents: Iterable[SubAgent]) -> None:
        registered: dict[AgentKind, SubAgent] = {}
        for agent in agents:
            kind = agent.kind
            if kind in registered:
                raise RegistryError(f"duplicate SubAgent for {kind.value}")
            registered[kind] = agent
        missing = set(AgentKind) - set(registered)
        if missing:
            names = ", ".join(sorted(kind.value for kind in missing))
            raise RegistryError(f"missing SubAgent owners: {names}")
        self._agents: Mapping[AgentKind, SubAgent] = MappingProxyType(registered)

    def get(self, kind: AgentKind) -> SubAgent:
        try:
            return self._agents[kind]
        except KeyError as exc:
            raise RegistryError(f"no SubAgent for {kind.value}") from exc


__all__ = ["SubAgentRegistry"]
