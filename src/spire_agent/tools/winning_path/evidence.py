"""Build the three candidate evidence sources used by Winning Path."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .preference import SKIP, PreferenceTable, normalize_card
from .contracts import CandidateEvidence, DecisionState, DeckPlan
from .needs import analyze_need_profile, candidate_need_coverage, plan_targets
from .plan import analyze_deck_plan
from .parameters import load_policy
from .protocol import normalize_character
from .templates import analyze_templates


class CandidateEvidenceError(ValueError):
    pass


def analyze_candidate_evidence(
    state: DecisionState, catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every offered card with Skip using only reviewed parameters."""

    cards = _offered_cards(state)
    if not cards:
        raise CandidateEvidenceError("state.reward.offered_cards must not be empty")
    knowledge = _object(catalog.get("knowledge"), "catalog.knowledge")
    derived = _object(catalog.get("derived"), "catalog.derived")
    before = analyze_deck_plan(state, catalog)
    templates = analyze_templates(state, catalog)
    targets = plan_targets(state)
    needs = analyze_need_profile(state, catalog, targets)
    coverage = candidate_need_coverage(state, needs)
    need_rows = {
        str(row.get("type") or ""): row
        for row in needs.needs
        if isinstance(row, Mapping) and row.get("status") != "SATISFIED"
    }
    expert_config = load_policy(_character(state))["parameters"]["expert"]
    table = PreferenceTable(
        _object(derived.get("expert_preferences"), "derived.expert_preferences"),
        context_order=expert_config["context_order"],
        deck_size_limits=expert_config["deck_size_limits"],
    )
    rows = []
    for choice_id, card in enumerate(cards):
        name = str(card["id"])
        after = analyze_deck_plan(add_card_to_state(state, card), catalog)
        rows.append(
            CandidateEvidence(
                choice_id=choice_id,
                name=name,
                card=card,
                hard_constraints=tuple(
                    _hard_constraints(state, name, before, after, knowledge)
                ),
                template=templates.get(choice_id, {"level": "NONE"}),
                transition=_transition_evidence(coverage.get(choice_id, ()), need_rows),
                expert=_expert_evidence(state, name, table),
            )
        )
    return {
        "schema_version": 1,
        "protocol_version": state.protocol_version,
        "skip": {"kind": "SKIP", "deck_plan": before.as_dict()},
        "target_plan": targets.as_dict(),
        "need_profile": needs.as_dict(),
        "candidates": [row.as_dict() for row in rows],
    }


def add_card_to_state(
    state: DecisionState, card: Mapping[str, Any]
) -> DecisionState:
    name = normalize_card(card)
    if not name:
        raise CandidateEvidenceError("candidate card has no canonical name")
    upgrades = max(0, int(card.get("upgrades") or 0))
    deck = _plain(state.deck)
    deck["cards"] = [*deck.get("cards", ()), _plain(card)]
    for field, increment in (
        ("counts", 1),
        ("upgrade_counts", int(upgrades > 0)),
    ):
        values = dict(deck.get(field) or {})
        if increment:
            values[name] = int(values.get(name) or 0) + increment
        deck[field] = dict(sorted(values.items()))
    maximum = dict(deck.get("max_upgrades") or {})
    maximum[name] = max(int(maximum.get(name) or 0), upgrades)
    deck["max_upgrades"] = dict(sorted(maximum.items()))
    deck["physical_size"] = int(deck.get("physical_size") or 0) + 1
    return DecisionState(
        run=_plain(state.run),
        deck=deck,
        assets=_plain(state.assets),
        route=_plain(state.route),
        reward=_plain(state.reward),
        missing_facts=state.missing_facts,
    )


def _transition_evidence(
    covered: Sequence[str], needs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    matched = tuple(sorted(set(map(str, covered)).intersection(needs)))
    if any(bool(needs[name].get("blocking")) for name in matched):
        level = "BLOCKING_NEED"
    elif matched:
        level = "CRITICAL_NEED"
    else:
        level = "NONE"
    return {"level": level, "needs": matched}


def _expert_evidence(
    state: DecisionState, card: str, table: PreferenceTable
) -> dict[str, Any]:
    config = load_policy(_character(state))["parameters"]["expert"]
    comparison = table.compare(
        state.run.get("act"),
        card,
        SKIP,
        owned=state.deck.get("counts") or {},
        deck_size=int(state.deck.get("physical_size") or 0),
    )
    if comparison["left"] == normalize_card(card):
        wins, losses = comparison["left_wins"], comparison["right_wins"]
    else:
        wins, losses = comparison["right_wins"], comparison["left_wins"]
    observations = wins + losses
    score = (wins - losses) / math.sqrt(observations) if observations else 0.0
    level = (
        "DIRECT"
        if score >= float(config["direct_z"])
        else "POSITIVE"
        if score > float(config["positive_z"])
        else "NEGATIVE"
        if score < -float(config["positive_z"])
        else "NONE"
    )
    return {
        "level": level,
        "score": round(score, 10),
        "wins": wins,
        "losses": losses,
        "observations": observations,
        "comparison": comparison,
    }


def _hard_constraints(
    state: DecisionState,
    card: str,
    before: DeckPlan,
    after: DeckPlan,
    knowledge: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    act = int(state.run.get("act") or 0)
    policies = _rows(knowledge.get("card_policies")) or _rows(
        knowledge.get("forbidden_cards")
    )
    for row in policies:
        acts = {int(value) for value in _array(row.get("acts"))}
        if (
            str(row.get("card") or row.get("name") or "") == card
            and str(row.get("policy") or "FORBID").upper() == "FORBID"
            and (not acts or act in acts)
        ):
            result.append({"type": "FORBIDDEN", "reason": row.get("reason")})
    owned = {
        str(name)
        for name, count in _object(state.deck.get("counts"), "deck.counts").items()
        if int(count) > 0
    }
    support = _object(knowledge.get("support"), "knowledge.support")
    rows = (*_rows(knowledge.get("conditional_cards")), *_rows(support.get("cards")))
    for row in rows:
        required = set(map(str, _array(row.get("requires_any_owned"))))
        if (
            str(row.get("card") or row.get("name") or "") == card
            and required
            and not required & owned
        ):
            result.append(
                {
                    "type": "MISSING_PREREQUISITE",
                    "requires_any_owned": sorted(required),
                }
            )
            break
    previous = _resource_conflicts(before, knowledge)
    for key, detail in _resource_conflicts(after, knowledge).items():
        if key not in previous:
            result.append({"type": "HARD_RESOURCE_CONFLICT", **detail})
    return result


def _resource_conflicts(
    plan: DeckPlan, knowledge: Mapping[str, Any]
) -> dict[tuple[str, tuple[str, ...]], dict[str, Any]]:
    declared: dict[str, dict[str, set[str]]] = {}
    for row in plan.hard_resource_constraints:
        for resource, direction in (row.get("constraints") or {}).items():
            declared.setdefault(str(resource), {}).setdefault(str(direction), set()).add(
                str(row.get("declaring_module_id") or "")
            )
    result = {}
    rules = knowledge.get("resource_rules")
    for resource, rule in (rules.items() if isinstance(rules, Mapping) else ()):
        if not isinstance(rule, Mapping):
            continue
        for raw in _array(rule.get("hard_conflicts")):
            directions = tuple(sorted(map(str, _array(raw))))
            known = declared.get(str(resource), {})
            if len(directions) == 2 and all(value in known for value in directions):
                result[(str(resource), directions)] = {
                    "resource": str(resource), "directions": directions
                }
    return result


def _offered_cards(state: DecisionState) -> tuple[Mapping[str, Any], ...]:
    rows = _rows(state.reward.get("offered_cards"))
    return rows or tuple(
        {"id": str(name), "upgrades": 0}
        for name in _array(state.reward.get("offered"))
        if str(name)
    )


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateEvidenceError(f"{path} must be an object")
    return value


def _array(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _rows(value: object) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in _array(value) if isinstance(row, Mapping))


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _character(state: DecisionState) -> str:
    return normalize_character(state.run.get("character"))


__all__ = ["CandidateEvidenceError", "add_card_to_state", "analyze_candidate_evidence"]
