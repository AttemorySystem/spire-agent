"""Immutable payload contracts for Winning Path analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from spire_agent.contracts import freeze, frozen_mapping

from .protocol import PROTOCOL_VERSION, canonical_sha256


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """The three positive evidence sources for one offered card."""

    choice_id: int
    name: str
    card: Mapping[str, Any]
    hard_constraints: tuple[Mapping[str, Any], ...] = ()
    limitations: tuple[Mapping[str, Any], ...] = ()
    template: Mapping[str, Any] = field(default_factory=frozen_mapping)
    transition: Mapping[str, Any] = field(default_factory=frozen_mapping)
    expert: Mapping[str, Any] = field(default_factory=frozen_mapping)
    schema_version: int = 1
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("CandidateEvidence.schema_version must equal 1")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("CandidateEvidence protocol version mismatch")
        if isinstance(self.choice_id, bool) or int(self.choice_id) < 0:
            raise ValueError("CandidateEvidence.choice_id must be non-negative")
        if not str(self.name).strip():
            raise ValueError("CandidateEvidence.name must be non-empty")
        object.__setattr__(self, "choice_id", int(self.choice_id))
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "card", frozen_mapping(self.card))
        for name in ("hard_constraints", "limitations"):
            object.__setattr__(self, name, tuple(freeze(getattr(self, name))))
        for name in ("template", "transition", "expert"):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))

    @property
    def rejected(self) -> bool:
        return bool(self.hard_constraints)

    def as_dict(self) -> dict[str, Any]:
        return _plain(
            {
                "schema_version": self.schema_version,
                "protocol_version": self.protocol_version,
                "choice_id": self.choice_id,
                "name": self.name,
                "card": self.card,
                "rejected": self.rejected,
                "hard_constraints": self.hard_constraints,
                "limitations": self.limitations,
                "template": self.template,
                "transition": self.transition,
                "expert": self.expert,
            }
        )


@dataclass(frozen=True, slots=True)
class DecisionState:
    """Canonical, replayable input to all evidence providers."""

    run: Mapping[str, Any]
    deck: Mapping[str, Any]
    assets: Mapping[str, Any]
    route: Mapping[str, Any]
    reward: Mapping[str, Any]
    missing_facts: tuple[str, ...] = ()
    schema_version: int = 1
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("DecisionState.schema_version must equal 1")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("DecisionState protocol version mismatch")
        for name in ("run", "deck", "assets", "route", "reward"):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))
        object.__setattr__(
            self,
            "missing_facts",
            tuple(sorted({str(item) for item in self.missing_facts if str(item)})),
        )

    def as_dict(self) -> dict[str, Any]:
        return _plain(
            {
                "schema_version": self.schema_version,
                "protocol_version": self.protocol_version,
                "run": self.run,
                "deck": self.deck,
                "assets": self.assets,
                "route": self.route,
                "reward": self.reward,
                "missing_facts": self.missing_facts,
            }
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class DeckPlan:
    """Reconstructed structural plan; it never stores an LLM assertion."""

    active_modules: tuple[str, ...] = ()
    committed_modules: tuple[str, ...] = ()
    blocked_modules: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    hard_resource_constraints: tuple[Mapping[str, Any], ...] = ()
    resource_pressures: tuple[Mapping[str, Any], ...] = ()
    goals: tuple[Mapping[str, Any], ...] = ()
    exit_conditions: tuple[Mapping[str, Any], ...] = ()
    dynamic_verification_required: tuple[str, ...] = ()
    schema_version: int = 1
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("DeckPlan.schema_version must equal 1")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("DeckPlan protocol version mismatch")
        for name in (
            "active_modules",
            "committed_modules",
            "blocked_modules",
            "capabilities",
            "dynamic_verification_required",
        ):
            object.__setattr__(
                self,
                name,
                tuple(sorted({str(item) for item in getattr(self, name)})),
            )
        for name in (
            "hard_resource_constraints",
            "resource_pressures",
            "goals",
            "exit_conditions",
        ):
            object.__setattr__(self, name, tuple(freeze(getattr(self, name))))

    def as_dict(self) -> dict[str, Any]:
        return _plain(
            {
                "schema_version": self.schema_version,
                "protocol_version": self.protocol_version,
                "active_modules": self.active_modules,
                "committed_modules": self.committed_modules,
                "blocked_modules": self.blocked_modules,
                "capabilities": self.capabilities,
                "hard_resource_constraints": self.hard_resource_constraints,
                "resource_pressures": self.resource_pressures,
                "goals": self.goals,
                "exit_conditions": self.exit_conditions,
                "dynamic_verification_required": self.dynamic_verification_required,
            }
        )


@dataclass(frozen=True, slots=True)
class TargetPlan:
    """Candidate-independent prospective encounter selection."""

    groups: tuple[Mapping[str, Any], ...]
    targets: tuple[str, ...]
    missing_facts: tuple[str, ...] = ()
    limitations: tuple[Mapping[str, Any], ...] = ()
    schema_version: int = 1
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("TargetPlan.schema_version must equal 1")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("TargetPlan protocol version mismatch")
        object.__setattr__(self, "groups", tuple(freeze(self.groups)))
        object.__setattr__(self, "targets", tuple(dict.fromkeys(map(str, self.targets))))
        object.__setattr__(
            self, "missing_facts", tuple(sorted(set(map(str, self.missing_facts))))
        )
        object.__setattr__(self, "limitations", tuple(freeze(self.limitations)))

    def as_dict(self) -> dict[str, Any]:
        return _plain(
            {
                "schema_version": self.schema_version,
                "protocol_version": self.protocol_version,
                "candidate_independent": True,
                "groups": self.groups,
                "targets": self.targets,
                "missing_facts": self.missing_facts,
                "limitations": self.limitations,
            }
        )


@dataclass(frozen=True, slots=True)
class NeedProfile:
    """Discrete current-deck readiness against one fixed target plan."""

    target_plan_sha256: str
    current_capabilities: tuple[str, ...]
    needs: tuple[Mapping[str, Any], ...]
    blocking_deficits: tuple[str, ...] = ()
    limitations: tuple[Mapping[str, Any], ...] = ()
    schema_version: int = 1
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("NeedProfile.schema_version must equal 1")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("NeedProfile protocol version mismatch")
        if not str(self.target_plan_sha256):
            raise ValueError("NeedProfile.target_plan_sha256 must be non-empty")
        object.__setattr__(
            self,
            "current_capabilities",
            tuple(sorted(set(map(str, self.current_capabilities)))),
        )
        object.__setattr__(self, "needs", tuple(freeze(self.needs)))
        object.__setattr__(
            self,
            "blocking_deficits",
            tuple(sorted(set(map(str, self.blocking_deficits)))),
        )
        object.__setattr__(self, "limitations", tuple(freeze(self.limitations)))

    def as_dict(self) -> dict[str, Any]:
        return _plain(
            {
                "schema_version": self.schema_version,
                "protocol_version": self.protocol_version,
                "target_plan_sha256": self.target_plan_sha256,
                "current_capabilities": self.current_capabilities,
                "needs": self.needs,
                "blocking_deficits": self.blocking_deficits,
                "limitations": self.limitations,
            }
        )


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    return value


__all__ = [
    "CandidateEvidence",
    "DecisionState",
    "DeckPlan",
    "NeedProfile",
    "TargetPlan",
]
