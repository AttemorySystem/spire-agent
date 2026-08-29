"""Frozen target selection and discrete encounter-readiness analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import re
from typing import Any

from spire_agent.contracts import frozen_mapping

from .contracts import DecisionState, DeckPlan, NeedProfile, TargetPlan
from .plan import analyze_deck_plan
from .parameters import load_parameters, load_policy
from .protocol import PROTOCOL_VERSION, canonical_sha256, normalize_character


class EncounterModelError(ValueError):
    """The reviewed encounter model is malformed or incompatible."""


@lru_cache(maxsize=None)
def load_encounter_model(character: object = "IRONCLAD") -> Mapping[str, Any]:
    character = normalize_character(character)
    transition = load_parameters(character)["transition"]
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scope": {
            "character": character,
            "interpretation": (
                "Discrete reviewed combat capabilities. Absence is a modeled "
                "deficit, not proof that the deck cannot win."
            ),
        },
        **dict(transition),
    }
    payload.pop("capabilities")
    payload.pop("target_rules")
    validate_encounter_model(payload, character)
    return frozen_mapping(payload)


@lru_cache(maxsize=None)
def encounter_model_sha256(character: object = "IRONCLAD") -> str:
    return canonical_sha256(load_encounter_model(character))


def validate_encounter_model(
    payload: Mapping[str, Any], character: object = "IRONCLAD"
) -> None:
    character = normalize_character(character)
    expected = {
        "schema_version", "protocol_version", "scope", "target_pools",
        "capability_aliases", "card_capabilities", "encounter_requirements",
        "foundation_needs",
    }
    if set(payload) != expected:
        raise EncounterModelError("encounter model has an unexpected top-level schema")
    if payload.get("schema_version") != 1 or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise EncounterModelError("encounter model version mismatch")
    if _object(payload.get("scope"), "scope").get("character") != character:
        raise EncounterModelError("encounter model character mismatch")
    immediate = set(
        map(str, load_policy(character)["parameters"]["transition"]["capabilities"])
    )
    aliases = _object(payload.get("capability_aliases"), "capability_aliases")
    if set(map(str, aliases)) != immediate:
        raise EncounterModelError("capability aliases must cover every immediate evidence type")
    for need, values in aliases.items():
        _strings(values, f"capability_aliases.{need}")
    foundations = _object(payload.get("foundation_needs"), "foundation_needs")
    if not set(map(str, foundations)).issubset(immediate):
        raise EncounterModelError("foundation needs must be declared capabilities")
    for need, raw in foundations.items():
        value = _object(raw, f"foundation_needs.{need}")
        if set(value) != {"density_bands"}:
            raise EncounterModelError("foundation needs contain an unreviewed field")
        bands = _rows(value["density_bands"])
        if not bands:
            raise EncounterModelError("foundation density bands must not be empty")
        previous = previous_required = 0
        for index, band_raw in enumerate(bands):
            band = _object(band_raw, f"foundation_needs.{need}.density_bands[{index}]")
            if set(band) != {"max_deck_size", "required_sources"}:
                raise EncounterModelError("foundation density band has an unexpected schema")
            maximum, required = int(band["max_deck_size"]), int(band["required_sources"])
            if maximum <= previous or required < max(1, previous_required):
                raise EncounterModelError("foundation density bands must be positive and ordered")
            previous, previous_required = maximum, required
    pools = _object(payload.get("target_pools"), "target_pools")
    if set(map(str, pools)) != {"1", "2", "3", "4"}:
        raise EncounterModelError("target pools must cover Acts 1 through 4")
    known_targets: set[str] = set()
    for act, raw in pools.items():
        pool = _object(raw, f"target_pools.{act}")
        if set(pool) != {"elites", "bosses"}:
            raise EncounterModelError(f"target_pools.{act} must contain elites and bosses")
        for family in ("elites", "bosses"):
            known_targets.update(_strings(pool[family], f"target_pools.{act}.{family}"))
    seen_cards: set[str] = set()
    for index, raw in enumerate(_rows(payload.get("card_capabilities"))):
        row = _object(raw, f"card_capabilities[{index}]")
        if set(row) != {"card", "provides"}:
            raise EncounterModelError("card capability rows must contain card and provides")
        card = _text(row["card"], f"card_capabilities[{index}].card")
        if card in seen_cards:
            raise EncounterModelError(f"duplicate card capability {card!r}")
        seen_cards.add(card)
        provided = set(_strings(row["provides"], f"card_capabilities[{index}].provides"))
        if not provided.issubset(immediate):
            raise EncounterModelError(f"card {card!r} provides undeclared immediate evidence")
    seen_targets: set[str] = set()
    for index, raw in enumerate(_rows(payload.get("encounter_requirements"))):
        row = _object(raw, f"encounter_requirements[{index}]")
        if set(row) != {"encounter", "critical"}:
            raise EncounterModelError(
                "encounter requirement rows must contain encounter and critical"
            )
        target = _text(row["encounter"], f"encounter_requirements[{index}].encounter")
        if target not in known_targets or target in seen_targets:
            raise EncounterModelError(f"unknown or duplicate encounter requirement {target!r}")
        seen_targets.add(target)
        needs = set(_strings(row["critical"], f"encounter_requirements[{index}].critical"))
        if not needs.issubset(immediate):
            raise EncounterModelError(f"encounter {target!r} uses undeclared need")
    if seen_targets != known_targets:
        raise EncounterModelError("every target pool encounter must have requirements")


def plan_targets(
    state: DecisionState,
    model: Mapping[str, Any] | None = None,
) -> TargetPlan:
    """Select targets from state only; offered candidates are never inspected."""

    character = _character(state)
    model = model or load_encounter_model(character)
    pools = _object(model.get("target_pools"), "target_pools")
    act = int(state.run.get("act") or 0)
    groups: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []
    missing: list[str] = []
    if str(act) not in pools:
        return TargetPlan(
            groups=(), targets=(), missing_facts=("run.act",),
            limitations=({"type": "TARGET_PLAN_UNAVAILABLE", "reason": "act is outside 1..4"},),
        )

    if _planned_elite_before_rest(state.route) and act <= 3:
        _append_group(groups, "PLANNED_ELITE_BEFORE_REST", pools[str(act)]["elites"], character)
    elif _is_boss_reward(state) and act in (1, 2):
        _append_group(groups, "NEXT_ACT_BOSS_POOL", pools[str(act + 1)]["bosses"], character)
    elif act == 3 and _is_boss_reward(state):
        _append_group(groups, "HEART_OBJECTIVE", ("Shield And Spear", "The Heart"), character)
    elif act == 3:
        current = _current_boss_targets(state, pools, act, limitations)
        if int(state.run.get("ascension") or 0) >= 20:
            current = tuple(pools["3"]["bosses"])
            limitations.append({
                "type": "BOSS_SEQUENCE_POOL",
                "reason": "A20 second Act 3 boss is unknown at card-reward time",
            })
        _append_group(groups, "CURRENT_BOSS_SEQUENCE", current, character)
        _append_group(groups, "HEART_OBJECTIVE", ("Shield And Spear", "The Heart"), character)
    elif act == 4:
        _append_group(groups, "HEART_OBJECTIVE", ("Shield And Spear", "The Heart"), character)
    else:
        current = _current_boss_targets(state, pools, act, limitations)
        _append_group(groups, "CURRENT_ACT_BOSS", current, character)
        if not str(state.run.get("boss") or "").strip():
            missing.append("run.boss")

    targets = tuple(
        dict.fromkeys(
            str(target)
            for group in groups
            for target in _array(group.get("targets"))
        )
    )
    return TargetPlan(
        groups=tuple(groups), targets=targets,
        missing_facts=tuple(missing), limitations=tuple(limitations),
    )


def analyze_need_profile(
    state: DecisionState,
    catalog: Mapping[str, Any],
    target_plan: TargetPlan | None = None,
    model: Mapping[str, Any] | None = None,
) -> NeedProfile:
    model = model or load_encounter_model(_character(state))
    target_plan = target_plan or plan_targets(state, model)
    aliases = {
        str(need): set(map(str, _array(values)))
        for need, values in _object(model.get("capability_aliases"), "capability_aliases").items()
    }
    plan = analyze_deck_plan(state, catalog)
    raw_capabilities = set(plan.capabilities)
    unverified = _dynamic_capability_evidence(state, plan, catalog)
    raw_capabilities.difference_update(capability for _, capability in unverified)
    raw_capabilities.update(
        _owned_card_capabilities(state, model, catalog, unverified)
    )
    current = {
        need for need, accepted in aliases.items()
        if accepted.intersection(raw_capabilities)
    }
    foundations = _foundation_status(state, plan, aliases, model, catalog)
    for need, status in foundations.items():
        if status["satisfied"]:
            current.add(need)
        else:
            current.discard(need)
    requirements = {
        _compact(row["encounter"]): row
        for row in _rows(model.get("encounter_requirements"))
    }
    required_by: dict[str, set[str]] = {}
    unknown_targets: list[str] = []
    for target in target_plan.targets:
        row = requirements.get(_compact(target))
        if row is None:
            unknown_targets.append(target)
            continue
        for need in _array(row.get("critical")):
            required_by.setdefault(str(need), set()).add(str(target))
    for need in foundations:
        required_by.setdefault(need, set()).add("DECK_FOUNDATION")
    needs = []
    blocking_horizon = _blocking_horizon(state, target_plan)
    for need in sorted(required_by):
        satisfied = need in current
        blocking = (
            not satisfied
            and blocking_horizon is not None
            and required_by[need] != {"DECK_FOUNDATION"}
        )
        row = {
            "type": need,
            "status": "SATISFIED" if satisfied else "DEFICIT_WITHIN_MODEL",
            "strength": "HIGH" if satisfied else "NONE",
            "blocking": blocking,
            "blocking_horizon": blocking_horizon if blocking else None,
            "required_by_targets": tuple(sorted(required_by[need])),
            "satisfied_by": tuple(sorted(aliases[need].intersection(raw_capabilities))),
        }
        if need in foundations:
            row.update(foundations[need])
            row.pop("satisfied")
        needs.append(row)
    limitations = list(target_plan.limitations)
    limitations.append({
        "type": "STATIC_CAPABILITY_ABSTRACTION",
        "reason": (
            "a modeled deficit is not proof of combat loss; the combat "
            "benchmark remains separate evidence"
        ),
    })
    if unknown_targets:
        limitations.append({"type": "UNMODELED_TARGETS", "targets": tuple(unknown_targets)})
    target_payload = target_plan.as_dict()
    return NeedProfile(
        target_plan_sha256=canonical_sha256(target_payload),
        current_capabilities=tuple(current),
        needs=tuple(needs),
        blocking_deficits=tuple(
            row["type"] for row in needs if row["blocking"]
        ),
        limitations=tuple(limitations),
    )


def candidate_need_coverage(
    state: DecisionState,
    need_profile: NeedProfile,
    model: Mapping[str, Any] | None = None,
) -> dict[int, tuple[str, ...]]:
    """Return direct candidate coverage of current modeled deficits."""

    model = model or load_encounter_model(_character(state))
    cards = {
        str(row["card"]): set(map(str, _array(row.get("provides"))))
        for row in _rows(model.get("card_capabilities"))
    }
    deficits = {
        str(row.get("type") or "")
        for row in need_profile.needs
        if row.get("status") != "SATISFIED"
    }
    result: dict[int, tuple[str, ...]] = {}
    for choice_id, raw in enumerate(_array(state.reward.get("offered_cards"))):
        card = raw if isinstance(raw, Mapping) else {}
        provided = cards.get(str(card.get("id") or ""), set())
        result[choice_id] = tuple(sorted(deficits.intersection(provided)))
    return result


def _owned_card_capabilities(
    state: DecisionState,
    model: Mapping[str, Any],
    catalog: Mapping[str, Any],
    unverified: set[tuple[str, str]] | None = None,
) -> set[str]:
    owned = {
        str(name) for name, count in _object(state.deck.get("counts"), "deck.counts").items()
        if int(count) > 0
    }
    knowledge = _object(catalog.get("knowledge"), "catalog.knowledge")
    support = _object(knowledge.get("support"), "catalog.knowledge.support")
    requirements = {
        str(row.get("card") or row.get("name") or ""): set(
            map(str, _array(row.get("requires_any_owned")))
        )
        for row in (
            *_rows(knowledge.get("conditional_cards")),
            *_rows(support.get("cards")),
        )
        if row.get("requires_any_owned")
    }
    result: set[str] = set()
    unverified = unverified or set()
    for row in _rows(model.get("card_capabilities")):
        card = str(row.get("card") or "")
        required = requirements.get(card, set())
        if card in owned and (not required or required & owned):
            result.update(
                capability
                for capability in map(str, _array(row.get("provides")))
                if (card, capability) not in unverified
            )
    return result


def _dynamic_capability_evidence(
    state: DecisionState,
    plan: DeckPlan,
    catalog: Mapping[str, Any],
) -> set[tuple[str, str]]:
    dynamic = set(plan.dynamic_verification_required)
    if not dynamic:
        return set()
    modules = _rows(
        _object(catalog.get("knowledge"), "catalog.knowledge").get("modules")
    )
    verified = {
        str(row.get("capability") or "")
        for module in modules
        if str(module.get("module_id") or "") in plan.active_modules
        and str(module.get("module_id") or "") not in dynamic
        for row in _rows(module.get("provides"))
    }
    evidence = set()
    for module in modules:
        if str(module.get("module_id") or "") not in dynamic:
            continue
        capabilities = {
            str(row.get("capability") or "")
            for row in _rows(module.get("provides"))
        } - verified
        activation = _object(module.get("activation"), "module.activation")
        anchors = set(map(str, _array(activation.get("anchor_slots"))))
        for slot in _rows(activation.get("slots")):
            if str(slot.get("id") or "") in anchors:
                evidence.update(
                    (card, capability)
                    for card in _slot_cards(slot)
                    for capability in capabilities
                )
    return evidence


def _slot_cards(value: Mapping[str, Any]) -> set[str]:
    group = value.get("group")
    cards = (
        set(map(str, _array(group.get("cards"))))
        if isinstance(group, Mapping)
        else set()
    )
    cards.update(
        str(fact.get("name") or "")
        for fact in _rows(value.get("all"))
        if str(fact.get("kind") or "").upper() == "CARD"
    )
    for alternative in _rows(value.get("any")):
        cards.update(_slot_cards(alternative))
    return cards


def _foundation_status(
    state: DecisionState,
    plan: DeckPlan,
    aliases: Mapping[str, set[str]],
    model: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    counts = {
        str(name): max(0, int(count))
        for name, count in _object(state.deck.get("counts"), "deck.counts").items()
    }
    size = int(state.deck.get("physical_size") or sum(counts.values()))
    providers = {
        str(row["card"]): set(map(str, _array(row.get("provides"))))
        for row in _rows(model.get("card_capabilities"))
    }
    active = set(plan.active_modules)
    modules = _rows(
        _object(catalog.get("knowledge"), "catalog.knowledge").get("modules")
    )
    engine_capabilities = {
        str(item.get("capability"))
        for module in modules
        if str(module.get("module_id") or "") in active
        and str(module.get("module_id") or "")
        not in plan.dynamic_verification_required
        for item in _rows(module.get("provides"))
    }
    result = {}
    for need, raw in _object(model.get("foundation_needs"), "foundation_needs").items():
        bands = _rows(_object(raw, f"foundation_needs.{need}").get("density_bands"))
        required = next(
            int(row["required_sources"])
            for row in bands
            if size <= int(row["max_deck_size"])
        )
        sources = sum(
            count for card, count in counts.items() if need in providers.get(card, ())
        )
        engine = tuple(sorted(aliases[need].intersection(engine_capabilities)))
        result[str(need)] = {
            "satisfied": bool(engine) or sources >= required,
            "current_sources": sources,
            "required_sources": required,
            "completed_engine_capabilities": engine,
        }
    return result


def _planned_elite_before_rest(route: Mapping[str, Any]) -> bool:
    rooms = tuple(_compact(item) for item in _array(route.get("planned_rooms")))
    first_rest = next(
        (
            index
            for index, room in enumerate(rooms)
            if room in {"r", "rest", "restroom"}
        ),
        len(rooms),
    )
    return any(
        room in {"e", "elite", "burningelite", "monsteroomelite"}
        for room in rooms[:first_rest]
    )


def _blocking_horizon(
    state: DecisionState, target_plan: TargetPlan
) -> str | None:
    rules = {str(group.get("rule") or "") for group in target_plan.groups}
    if "PLANNED_ELITE_BEFORE_REST" in rules:
        return "PLANNED_ELITE_BEFORE_REST"
    if int(state.run.get("act") or 0) == 4:
        return "ACT4_OBJECTIVE"
    rooms = tuple(_compact(item) for item in _array(state.route.get("planned_rooms")))
    first_rest = next(
        (index for index, room in enumerate(rooms) if room in {"r", "rest", "restroom"}),
        len(rooms),
    )
    if any(
        room in {"b", "boss", "monsterroomboss"}
        for room in rooms[:first_rest]
    ):
        return "PLANNED_BOSS_BEFORE_REST"
    return None


def _is_boss_reward(state: DecisionState) -> bool:
    kind = _compact(state.reward.get("kind"))
    room = _compact(state.run.get("room_type"))
    return "boss" in kind or "boss" in room


def _current_boss_targets(
    state: DecisionState,
    pools: Mapping[str, Any],
    act: int,
    limitations: list[dict[str, Any]],
) -> tuple[str, ...]:
    candidates = tuple(map(str, _array(_object(pools[str(act)], "act pool").get("bosses"))))
    raw = str(state.run.get("boss") or "").strip()
    if raw:
        normalized = _compact(raw)
        matches = tuple(
            target for target in candidates
            if _compact(target) == normalized
            or _compact("The " + target) == normalized
            or _compact(target.removeprefix("The ")) == normalized
        )
        if len(matches) == 1:
            return matches
    limitations.append({
        "type": "CURRENT_BOSS_UNKNOWN_USING_POOL",
        "reason": "target planner used the full current-Act boss pool",
    })
    return candidates


def _append_group(
    groups: list[dict[str, Any]],
    rule: str,
    targets: Sequence[object],
    character: str,
) -> None:
    allowed = set(
        map(str, load_policy(character)["parameters"]["transition"]["target_rules"])
    )
    if rule not in allowed:
        raise EncounterModelError(f"target rule {rule!r} is not frozen in protocol")
    groups.append({"rule": rule, "targets": tuple(map(str, targets))})


def _character(state: DecisionState) -> str:
    return normalize_character(state.run.get("character"))


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EncounterModelError(f"{path} must be an object")
    return value


def _rows(value: object) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in _array(value) if isinstance(row, Mapping))


def _array(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _strings(value: object, path: str) -> tuple[str, ...]:
    result = tuple(_text(item, f"{path}[]") for item in _array(value))
    if not result or len(result) != len(set(result)):
        raise EncounterModelError(f"{path} must contain unique non-empty strings")
    return result


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EncounterModelError(f"{path} must be a non-empty string")
    return value.strip()


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


__all__ = [
    "EncounterModelError", "analyze_need_profile",
    "candidate_need_coverage", "encounter_model_sha256",
    "load_encounter_model", "plan_targets", "validate_encounter_model",
]
