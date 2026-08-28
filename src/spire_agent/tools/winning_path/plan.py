"""Reconstruct a deterministic DeckPlan from state and reviewed knowledge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import DecisionState, DeckPlan


class DeckPlanError(ValueError):
    """A catalog cannot be used by the frozen DeckPlan analyzer."""


def module_progress(
    state: DecisionState,
    module: Mapping[str, Any],
    capabilities: Sequence[str] = (),
) -> dict[str, Any]:
    """Return discrete within-module progress, never a cross-module score."""

    module_id = _module_id(module)
    counts = {
        str(name): int(count)
        for name, count in _object(
            state.deck.get("counts"), "state.deck.counts"
        ).items()
    }
    max_upgrades_raw = state.deck.get("max_upgrades")
    fallback = _object_optional(state.deck.get("upgrade_counts"))
    max_upgrades = {
        str(name): int(level)
        for name, level in (
            max_upgrades_raw.items()
            if isinstance(max_upgrades_raw, Mapping)
            else fallback.items()
        )
    }
    relics = {
        str(row.get("id") or "")
        for row in _mapping_rows(state.assets.get("relics"))
        if row.get("id")
    }
    activation = _object(
        module.get("activation"), f"module {module_id}.activation"
    )
    slots = _mapping_rows(activation.get("slots"))
    required = tuple(slot for slot in slots if bool(slot.get("required", True)))
    satisfied = tuple(
        sorted(
            str(slot.get("id") or "")
            for slot in required
            if _slot_satisfied(slot, counts, max_upgrades, relics)
        )
    )
    anchors = {
        str(item) for item in _array_optional(activation.get("anchor_slots"))
    }
    required_capabilities = {
        str(item)
        for item in _array_optional(activation.get("requires_capabilities"))
    }
    owned_capabilities = set(map(str, capabilities))
    return {
        "module_id": module_id,
        "required_slots": tuple(
            str(slot.get("id") or "") for slot in required
        ),
        "satisfied_slots": satisfied,
        "satisfied_anchor_slots": tuple(sorted(anchors.intersection(satisfied))),
        "required_capabilities": tuple(sorted(required_capabilities)),
        "satisfied_capabilities": tuple(
            sorted(required_capabilities.intersection(owned_capabilities))
        ),
        "complete": len(satisfied) == len(required)
        and required_capabilities.issubset(owned_capabilities),
    }


def analyze_deck_plan(
    state: DecisionState, catalog: Mapping[str, Any]
) -> DeckPlan:
    knowledge = _object(catalog.get("knowledge"), "catalog.knowledge")
    raw_modules = _array(knowledge.get("modules"), "catalog.knowledge.modules")
    modules = tuple(
        sorted(
            (_object(raw, "module") for raw in raw_modules),
            key=lambda row: str(row.get("module_id") or ""),
        )
    )
    support = _object(knowledge.get("support"), "catalog.knowledge.support")
    counts = {
        str(name): int(count)
        for name, count in _object(state.deck.get("counts"), "state.deck.counts").items()
    }
    upgrade_counts = {
        str(name): int(count)
        for name, count in _object(
            state.deck.get("upgrade_counts"), "state.deck.upgrade_counts"
        ).items()
    }
    max_upgrades_raw = state.deck.get("max_upgrades")
    max_upgrades = {
        str(name): int(level)
        for name, level in (
            _object(max_upgrades_raw, "state.deck.max_upgrades").items()
            if isinstance(max_upgrades_raw, Mapping)
            else upgrade_counts.items()
        )
    }
    relics = {
        str(row.get("id") or "")
        for row in _mapping_rows(state.assets.get("relics"))
        if row.get("id")
    }

    capabilities = _support_capabilities(support, counts, max_upgrades)
    blocked = {
        _module_id(module)
        for module in modules
        if any(
            _fact_satisfied(fact, counts, max_upgrades, relics)
            for fact in _mapping_rows(module.get("blocked_by"))
        )
    }
    active: set[str] = set()
    changed = True
    while changed:
        changed = False
        for module in modules:
            module_id = _module_id(module)
            if module_id in active or module_id in blocked:
                continue
            activation = _object(module.get("activation"), f"module {module_id}.activation")
            required_caps = {
                str(item) for item in _array_optional(activation.get("requires_capabilities"))
            }
            if not required_caps.issubset(capabilities):
                continue
            slots = _mapping_rows(activation.get("slots"))
            required_slots = tuple(
                slot for slot in slots if bool(slot.get("required", True))
            )
            if all(
                _slot_satisfied(slot, counts, max_upgrades, relics)
                for slot in required_slots
            ):
                active.add(module_id)
                capabilities.update(_provided_capabilities(module))
                changed = True

    committed: set[str] = set()
    for module in modules:
        module_id = _module_id(module)
        if module_id in active or module_id in blocked:
            continue
        activation = _object(module.get("activation"), f"module {module_id}.activation")
        anchors = {
            str(item) for item in _array_optional(activation.get("anchor_slots"))
        }
        slots = {
            str(slot.get("id") or ""): slot
            for slot in _mapping_rows(activation.get("slots"))
        }
        if anchors and any(
            anchor in slots
            and _slot_satisfied(slots[anchor], counts, max_upgrades, relics)
            for anchor in anchors
        ):
            committed.add(module_id)

    participating = active | committed
    hard_constraints: list[dict[str, Any]] = []
    pressures: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    dynamic: set[str] = set()
    for module in modules:
        module_id = _module_id(module)
        if module_id not in participating:
            continue
        hard = _object_optional(module.get("hard_resource_constraints"))
        if hard:
            hard_constraints.append(
                {"declaring_module_id": module_id, "constraints": dict(hard)}
            )
        pressures.extend(
            _attributed(module_id, row)
            for row in _mapping_rows(module.get("soft_resource_pressures"))
        )
        goals.extend(
            _attributed(module_id, row)
            for row in _mapping_rows(module.get("goals"))
        )
        exits.extend(
            _attributed(module_id, row)
            for row in _mapping_rows(module.get("exit_conditions"))
        )
        if bool(module.get("dynamic_verification")):
            dynamic.add(module_id)

    return DeckPlan(
        active_modules=tuple(active),
        committed_modules=tuple(committed),
        blocked_modules=tuple(blocked),
        capabilities=tuple(capabilities),
        hard_resource_constraints=tuple(hard_constraints),
        resource_pressures=tuple(pressures),
        goals=tuple(goals),
        exit_conditions=tuple(exits),
        dynamic_verification_required=tuple(dynamic),
    )


def _support_capabilities(
    support: Mapping[str, Any],
    counts: Mapping[str, int],
    upgrades: Mapping[str, int],
) -> set[str]:
    capabilities: set[str] = set()
    owned = {name for name, count in counts.items() if count > 0}
    for row in _mapping_rows(support.get("cards")):
        card = str(row.get("card") or "")
        if not card or counts.get(card, 0) <= 0:
            continue
        requirements = {
            str(item) for item in _array_optional(row.get("requires_any_owned"))
        }
        if requirements and not requirements.intersection(owned):
            continue
        capabilities.update(
            str(item) for item in _array_optional(row.get("provides")) if str(item)
        )
        if upgrades.get(card, 0) > 0:
            capabilities.update(
                str(item)
                for item in _array_optional(row.get("upgraded_provides"))
                if str(item)
            )
    return capabilities


def _slot_satisfied(
    slot: Mapping[str, Any],
    counts: Mapping[str, int],
    upgrades: Mapping[str, int],
    relics: set[str],
) -> bool:
    alternatives = _mapping_rows(slot.get("any"))
    if alternatives:
        return any(
            _clause_satisfied(clause, counts, upgrades, relics)
            for clause in alternatives
        )
    return _clause_satisfied(slot, counts, upgrades, relics)


def _clause_satisfied(
    clause: Mapping[str, Any],
    counts: Mapping[str, int],
    upgrades: Mapping[str, int],
    relics: set[str],
) -> bool:
    facts = _mapping_rows(clause.get("all"))
    if facts and not all(
        _fact_satisfied(fact, counts, upgrades, relics) for fact in facts
    ):
        return False
    group = _object_optional(clause.get("group"))
    if group:
        cards = tuple(str(card) for card in _array_optional(group.get("cards")))
        distinct = sum(counts.get(card, 0) > 0 for card in cards)
        copies = sum(max(0, counts.get(card, 0)) for card in cards)
        if distinct < int(group.get("minimum_distinct") or 0):
            return False
        if copies < int(group.get("minimum_total_copies") or 0):
            return False
    return bool(facts or group)


def _fact_satisfied(
    fact: Mapping[str, Any],
    counts: Mapping[str, int],
    upgrades: Mapping[str, int],
    relics: set[str],
) -> bool:
    kind = str(fact.get("kind") or "").upper()
    name = str(fact.get("name") or "")
    if kind == "RELIC":
        return name in relics
    if kind != "CARD":
        return False
    required_count = max(1, int(fact.get("count") or 1))
    minimum_upgrade = max(0, int(fact.get("minimum_upgrade") or 0))
    if counts.get(name, 0) < required_count:
        return False
    return upgrades.get(name, 0) >= minimum_upgrade


def _provided_capabilities(module: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("capability"))
        for row in _mapping_rows(module.get("provides"))
        if row.get("capability")
    }


def _attributed(module_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {"declaring_module_id": module_id, **dict(row)}


def _module_id(module: Mapping[str, Any]) -> str:
    module_id = str(module.get("module_id") or "")
    if not module_id:
        raise DeckPlanError("module.module_id must be non-empty")
    return module_id


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeckPlanError(f"{path} must be an object")
    return value


def _object_optional(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _array(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DeckPlanError(f"{path} must be an array")
    return value


def _array_optional(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _mapping_rows(value: object) -> tuple[Mapping[str, Any], ...]:
    return tuple(item for item in _array_optional(value) if isinstance(item, Mapping))


__all__ = [
    "DeckPlanError",
    "analyze_deck_plan",
    "module_progress",
]
