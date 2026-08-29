"""Template-distance evidence for Winning Path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .template_search import analyze_routes
from .contracts import DecisionState
from .plan import module_progress
from .parameters import load_policy
from .protocol import normalize_character


def analyze_templates(
    state: DecisionState, catalog: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    character = normalize_character(state.run.get("character"))
    policy = load_policy(character)["parameters"]["templates"]
    algorithm = policy["distance"]["algorithm"]
    if algorithm not in {"CERTIFICATE_LEXICOGRAPHIC", "MODULE_LEXICOGRAPHIC"}:
        raise ValueError("unsupported template distance algorithm")
    knowledge = catalog.get("knowledge") or {}
    offered = list(state.reward.get("offered_cards") or ())
    if algorithm == "MODULE_LEXICOGRAPHIC":
        return {
            choice_id: _module_evidence(state, card, catalog)
            for choice_id, card in enumerate(offered)
            if isinstance(card, Mapping)
        }
    result = analyze_routes(
        {
            **dict(state.run),
            "deck": list(state.deck.get("cards") or ()),
            "relics": list(state.assets.get("relics") or ()),
        },
        offered,
        catalog=_search_catalog(catalog),
    )
    return {
        int(row["choice_id"]): _template_evidence(
            state, str(row.get("name") or ""), row.get("path_evidence"), knowledge
        )
        for row in result.get("candidates") or ()
        if isinstance(row, Mapping)
    }


def _template_evidence(
    state: DecisionState,
    card: str,
    path: object,
    knowledge: Mapping[str, Any],
) -> dict[str, Any]:
    dominant = _dominant(state, card, knowledge)
    if dominant:
        return {"level": "CORE_ACTIVATION", "source": "DOMINANT_TEMPLATE", "reason": dominant}
    if not isinstance(path, Mapping):
        return {"level": "NONE"}
    level = {
        "BOTTLENECK_ANCHOR": "REACHABLE_ENTRY",
        "COMMITTED_SLOT": "COMMITTED_PROGRESS",
        "CORE_ACTIVATION": "CORE_ACTIVATION",
    }[str(path["kind"])]
    return {
        "level": level,
        "source": "TEMPLATE_DISTANCE",
        "route_id": path.get("route_id"),
        "completed_core_gain": int(path.get("completed_core_gain") or 0),
        "anchor_reduction": int(path.get("anchor_reduction") or 0),
        "missing_card_reduction": int(path.get("missing_card_reduction") or 0),
        "completion_probability": float(path.get("completion_probability") or 0.0),
    }


def _dominant(state: DecisionState, card: str, knowledge: Mapping[str, Any]) -> str | None:
    act = int(state.run.get("act") or 0)
    owned = int((state.deck.get("counts") or {}).get(card) or 0)
    for row in knowledge.get("dominant_cards") or ():
        if not isinstance(row, Mapping) or str(row.get("name") or row.get("card") or "") != card:
            continue
        acts = {int(value) for value in row.get("acts") or ()}
        if (not acts or act in acts) and owned <= int(row.get("maximum_owned") or 0):
            return str(row.get("reason") or "reviewed dominant template")
    return None


def _module_evidence(
    state: DecisionState,
    card: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    knowledge = catalog.get("knowledge") or {}
    name = str(card.get("id") or "")
    dominant = _dominant(state, name, knowledge)
    if dominant:
        return {
            "level": "CORE_ACTIVATION",
            "source": "DOMINANT_TEMPLATE",
            "reason": dominant,
        }
    after_state = _with_card(state, card)
    rates = (catalog.get("derived") or {}).get("offer_rates") or {}
    rows = []
    for module in knowledge.get("modules") or ():
        if not isinstance(module, Mapping):
            continue
        before = module_progress(state, module)
        after = module_progress(after_state, module)
        gained = set(after["satisfied_slots"]) - set(before["satisfied_slots"])
        optional_gain = sum(
            int(
                after["optional_slot_progress"][slot]
                > before["optional_slot_progress"][slot]
            )
            for slot in before["optional_slot_progress"]
        )
        anchors_before = set(before["satisfied_anchor_slots"])
        anchors_after = set(after["satisfied_anchor_slots"])
        if after["complete"] and not before["complete"]:
            level = "CORE_ACTIVATION"
        elif anchors_before and (gained or optional_gain):
            level = "COMMITTED_PROGRESS"
        elif anchors_after - anchors_before:
            level = "REACHABLE_ENTRY"
        else:
            continue
        observed_level = level
        if module.get("candidate_policy") == "ADVISORY_ONLY" or (
            module.get("bottleneck_requires_prerequisites")
            and (not anchors_before or not after["complete"])
        ):
            level = "NONE"
        rows.append(
            {
                "level": level,
                "observed_level": observed_level,
                "source": "MODULE_DISTANCE",
                "route_id": after["module_id"],
                "completed_core_gain": int(after["complete"]),
                "anchor_reduction": len(anchors_after - anchors_before),
                "missing_card_reduction": max(len(gained), optional_gain),
                "completion_probability": _completion_probability(
                    after_state, module, rates, catalog
                ),
            }
        )
    if not rows:
        return {"level": "NONE"}
    order = {name: index for index, name in enumerate(
        ("NONE", "REACHABLE_ENTRY", "COMMITTED_PROGRESS", "CORE_ACTIVATION")
    )}
    return max(
        rows,
        key=lambda row: (
            order[row["level"]], row["completed_core_gain"],
            row["anchor_reduction"], row["missing_card_reduction"],
            row["completion_probability"], str(row["route_id"]),
        ),
    )


def _with_card(
    state: DecisionState, card: Mapping[str, Any]
) -> DecisionState:
    name = str(card.get("id") or "")
    deck = dict(state.deck)
    counts = dict(deck.get("counts") or {})
    counts[name] = int(counts.get(name) or 0) + 1
    deck["counts"] = counts
    deck["physical_size"] = int(deck.get("physical_size") or 0) + 1
    return DecisionState(
        run=state.run, deck=deck, assets=state.assets, route=state.route,
        reward=state.reward, missing_facts=state.missing_facts,
    )


def _completion_probability(
    state: DecisionState,
    module: Mapping[str, Any],
    rates: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> float:
    progress = module_progress(state, module)
    missing = set(progress["required_slots"]) - set(progress["satisfied_slots"])
    slots = {
        str(row.get("id") or ""): row
        for row in (module.get("activation") or {}).get("slots") or ()
        if isinstance(row, Mapping)
    }
    horizon = _horizon(state, catalog)
    chance = 1.0
    for slot_id in missing:
        rate = min(1.0, sum(float(rates.get(name) or 0.0) for name in _slot_cards(slots[slot_id])))
        chance *= 1.0 - (1.0 - rate) ** horizon
    return round(chance, 10)


def _slot_cards(slot: Mapping[str, Any]) -> set[str]:
    clauses: Sequence[object] = slot.get("any") or (slot,)
    return {
        str(fact.get("name") or "")
        for clause in clauses
        if isinstance(clause, Mapping)
        for fact in clause.get("all") or ()
        if isinstance(fact, Mapping) and fact.get("kind") == "CARD"
    } | {
        str(name)
        for clause in clauses
        if isinstance(clause, Mapping)
        for name in (clause.get("group") or {}).get("cards") or ()
    }


def _horizon(state: DecisionState, catalog: Mapping[str, Any]) -> int:
    act, floor = int(state.run.get("act") or 0), int(state.run.get("floor") or 0)
    rows = (catalog.get("derived") or {}).get("horizons", {}).get(str(act), ())
    values = [float(value) for known_floor, value in rows if int(known_floor) <= floor]
    return max(0, round(values[-1] if values else 0))


def _search_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt reviewed parameters to the small route-search input."""

    knowledge, derived = catalog.get("knowledge") or {}, catalog.get("derived") or {}
    return {
        "model": catalog.get("model") or {},
        "modules": [
            {
                "id": row["module_id"],
                "activation": row["activation"],
                "aspect": row["aspect"],
                "candidate_policy": row["candidate_policy"],
                "phase": row["phase"],
                "provides": [item["capability"] for item in row["provides"]],
            }
            for row in knowledge.get("modules") or ()
        ],
        "routes": knowledge.get("routes") or (),
        "dominant_cards": knowledge.get("dominant_cards") or (),
        "forbidden_cards": knowledge.get("forbidden_cards") or (),
        "conditional_cards": knowledge.get("conditional_cards") or (),
        "bridges": knowledge.get("bridges") or (),
        "candidate_bridges": knowledge.get("candidate_bridges") or (),
        "support_cards": (knowledge.get("support") or {}).get("cards") or (),
        "offer_rates": derived.get("offer_rates") or {},
        "horizons": derived.get("horizons") or {},
    }

__all__ = ["analyze_templates"]
