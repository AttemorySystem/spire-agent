"""Search progress over reviewed construction modules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any


@dataclass
class _Deck:
    cards: Counter[str]
    upgrades: dict[str, int]
    relics: set[str]

    def with_card(self, value: object) -> _Deck:
        result = _Deck(self.cards.copy(), dict(self.upgrades), set(self.relics))
        name = _base_card(_name(value))
        result.cards[name] += 1
        result.upgrades[name] = max(result.upgrades.get(name, 0), _upgrade(value))
        return result


def analyze_routes(
    game_state: Mapping[str, Any], candidates: Sequence[object],
    *, catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Return certificate-route progress for template-distance evidence."""

    data = catalog
    deck = _deck(game_state)
    horizon = _horizon(
        data, _integer(game_state.get("act")), _integer(game_state.get("floor"))
    )
    active_before, capabilities_before = _active(deck, data["modules"])
    routes_before = _route_frontier(
        deck, active_before, capabilities_before, horizon, data
    )
    rows = []
    for choice_id, raw in enumerate(candidates):
        after = deck.with_card(raw)
        active_after, capabilities_after = _active(after, data["modules"])
        routes_after = _route_frontier(
            after, active_after, capabilities_after, horizon, data
        )
        progress = _route_progress(routes_before, routes_after)
        rows.append(
            {
                "choice_id": choice_id,
                "name": _name(raw),
                "route_progress": progress,
                "path_evidence": _path_evidence(
                    _base_card(_name(raw)), progress, deck, data
                ),
            }
        )
    return {
        "model": "expert_certificate_routes",
        "route_count": len(data.get("routes") or ()),
        "template_count": len(routes_before),
        "nearest_before": routes_before[:5],
        "candidates": rows,
    }



def _active(deck: _Deck, modules: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    active: set[str] = set()
    capabilities: set[str] = set()
    changed = True
    while changed:
        changed = False
        for module in modules:
            module_id = str(module["id"])
            activation = module.get("activation") or {}
            if module_id in active or not set(
                activation.get("requires_capabilities") or ()
            ) <= capabilities:
                continue
            if all(
                not slot.get("required", True) or _slot_satisfied(slot, deck)
                for slot in activation.get("slots") or ()
            ):
                active.add(module_id)
                capabilities.update(map(str, module.get("provides") or ()))
                changed = True
    return active, capabilities


def _slot_satisfied(slot: Mapping[str, Any], deck: _Deck) -> bool:
    alternatives = slot.get("any") or ()
    return (
        any(_clause_satisfied(row, deck) for row in alternatives)
        if alternatives
        else _clause_satisfied(slot, deck)
    )


def _clause_satisfied(clause: Mapping[str, Any], deck: _Deck) -> bool:
    for fact in clause.get("all") or ():
        name = str(fact.get("name") or "")
        if fact.get("kind") == "RELIC":
            if name not in deck.relics:
                return False
        elif deck.cards.get(name, 0) < int(fact.get("count") or 1) or deck.upgrades.get(
            name, 0
        ) < int(fact.get("minimum_upgrade") or 0):
            return False
    group = clause.get("group")
    if isinstance(group, Mapping):
        names = list(map(str, group.get("cards") or ()))
        if sum(deck.cards.get(name, 0) > 0 for name in names) < int(
            group.get("minimum_distinct") or 1
        ) or sum(deck.cards.get(name, 0) for name in names) < int(
            group.get("minimum_total_copies") or 1
        ):
            return False
    return True


def _module_plan(
    module: Mapping[str, Any],
    deck: _Deck,
    capabilities: set[str],
    horizon: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    activation = module.get("activation") or {}
    missing_cards: Counter[str] = Counter()
    missing_relics: set[str] = set()
    for slot in activation.get("slots") or ():
        if not slot.get("required", True):
            continue
        plan = _slot_plan(slot, deck, horizon, data)
        for card, count in plan["missing_cards"].items():
            missing_cards[card] = max(missing_cards[card], count)
        missing_relics.update(plan["missing_relics"])
    missing_capabilities = sorted(
        set(activation.get("requires_capabilities") or ()) - capabilities
    )
    probability = _plan_probability(
        missing_cards, missing_relics, missing_capabilities, horizon, data
    )
    return {
        "missing_cards": dict(sorted(missing_cards.items())),
        "missing_relics": sorted(missing_relics),
        "missing_capabilities": missing_capabilities,
        "probability": probability,
    }


def _slot_plan(
    slot: Mapping[str, Any], deck: _Deck, horizon: int, data: Mapping[str, Any]
) -> dict[str, Any]:
    alternatives = slot.get("any") or ()
    plans = [
        _clause_plan(row, deck, horizon, data)
        for row in (alternatives or (slot,))
    ]
    return min(
        plans,
        key=lambda row: (
            bool(row["missing_relics"]),
            -float(row["probability"]),
            sum(row["missing_cards"].values()),
            tuple(row["missing_cards"]),
        ),
    )


def _clause_plan(
    clause: Mapping[str, Any], deck: _Deck, horizon: int, data: Mapping[str, Any]
) -> dict[str, Any]:
    cards = deck.cards.copy()
    missing: Counter[str] = Counter()
    relics: set[str] = set()
    for fact in clause.get("all") or ():
        name = str(fact.get("name") or "")
        if fact.get("kind") == "RELIC":
            if name not in deck.relics:
                relics.add(name)
            continue
        required = int(fact.get("count") or 1)
        count = max(0, required - cards.get(name, 0))
        if not count and deck.upgrades.get(name, 0) < int(
            fact.get("minimum_upgrade") or 0
        ):
            count = 1
        if count:
            missing[name] += count
            cards[name] += count
    group = clause.get("group")
    if isinstance(group, Mapping):
        names = list(map(str, group.get("cards") or ()))
        rates = data.get("offer_rates") or {}
        absent = sorted(
            (name for name in names if not cards.get(name)),
            key=lambda name: (-float(rates.get(name, 0.0)), name),
        )
        distinct = sum(cards.get(name, 0) > 0 for name in names)
        for name in absent[: max(0, int(group.get("minimum_distinct") or 1) - distinct)]:
            missing[name] += 1
            cards[name] += 1
        total = sum(cards.get(name, 0) for name in names)
        ordered = sorted(names, key=lambda name: (-float(rates.get(name, 0.0)), name))
        while total < int(group.get("minimum_total_copies") or 1) and ordered:
            missing[ordered[0]] += 1
            cards[ordered[0]] += 1
            total += 1
    return {
        "missing_cards": dict(sorted(missing.items())),
        "missing_relics": sorted(relics),
        "probability": _plan_probability(missing, relics, (), horizon, data),
    }


def _plan_probability(
    cards: Mapping[str, int],
    relics: Sequence[str] | set[str],
    capabilities: Sequence[str] | set[str],
    horizon: int,
    data: Mapping[str, Any],
) -> float:
    if relics or capabilities or sum(cards.values()) > horizon:
        return 0.0
    probability = 1.0
    rates = data.get("offer_rates") or {}
    for card, count in cards.items():
        probability *= _binomial_tail(horizon, float(rates.get(card, 0.0)), count)
    return probability



def _route_frontier(
    deck: _Deck,
    active: set[str],
    capabilities: set[str],
    horizon: int,
    data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return every certificate signature with a reviewed mainline."""

    modules = {str(row["id"]): row for row in data["modules"]}
    result = []
    for route in data.get("routes") or ():
        route_ids = tuple(
            module_id
            for value in route.get("modules") or ()
            if (module_id := str(value)) in modules
        )
        core = tuple(
            sorted(
                module_id
                for module_id in route_ids
                if str(modules[module_id].get("candidate_policy") or "")
                == "MAINLINE_ONLY"
            )
        )
        if not core:
            continue
        result.append(
            _route_variant(
                core,
                route,
                deck,
                active,
                capabilities,
                horizon,
                modules,
                data,
            )
        )
    return sorted(result, key=_route_key)


def _route_variant(
    core: tuple[str, ...],
    route: Mapping[str, Any],
    deck: _Deck,
    active: set[str],
    capabilities: set[str],
    horizon: int,
    modules: Mapping[str, Mapping[str, Any]],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    route_ids = {
        str(value) for value in route.get("modules") or () if str(value) in modules
    }
    selected = {
        module_id
        for module_id in route_ids
        if str(modules[module_id].get("candidate_policy") or "")
        in {"MAINLINE_ONLY", "COMPATIBLE_ONLY"}
    }
    external: set[str] = set()
    all_route_capabilities = capabilities | {
        str(value)
        for module_id in route_ids
        for value in modules[module_id].get("provides") or ()
    }
    while True:
        provided = capabilities | {
            str(value)
            for module_id in selected
            for value in modules[module_id].get("provides") or ()
        }
        missing = {
            str(value)
            for module_id in selected
            for value in (modules[module_id].get("activation") or {}).get(
                "requires_capabilities"
            )
            or ()
            if str(value) not in provided
        }
        if not missing:
            break
        changed = False
        for capability in sorted(missing):
            providers = [
                module_id
                for module_id in route_ids - selected
                if capability
                in {str(value) for value in modules[module_id].get("provides") or ()}
            ]
            if not providers:
                external.add(capability)
                continue
            selected.add(
                min(
                    providers,
                    key=lambda module_id: _module_route_key(
                        _module_plan(
                            modules[module_id],
                            deck,
                            all_route_capabilities,
                            horizon,
                            data,
                        ),
                        module_id,
                    ),
                )
            )
            changed = True
        if not changed:
            break

    potential = capabilities | {
        str(value)
        for module_id in selected
        for value in modules[module_id].get("provides") or ()
    }
    plans = {
        module_id: _module_plan(
            modules[module_id], deck, potential, horizon, data
        )
        for module_id in selected
    }
    missing_cards: Counter[str] = Counter()
    missing_relics: set[str] = set()
    for plan in plans.values():
        for card, count in plan["missing_cards"].items():
            missing_cards[card] = max(missing_cards[card], int(count))
        missing_relics.update(plan["missing_relics"])
    missing_anchor_cards: Counter[str] = Counter()
    missing_anchor_relics: set[str] = set()
    for module_id in core:
        for slot in _anchor_slots(modules[module_id]):
            plan = _slot_plan(slot, deck, horizon, data)
            for card, count in plan["missing_cards"].items():
                missing_anchor_cards[card] = max(
                    missing_anchor_cards[card], int(count)
                )
            missing_anchor_relics.update(plan["missing_relics"])
    probability = _plan_probability(
        missing_cards, missing_relics, external, horizon, data
    )
    return {
        "id": str(route["id"]),
        "core_modules": list(core),
        "required_modules": sorted(selected),
        "completed_core_modules": sum(module_id in active for module_id in core),
        "active_route_modules": sum(module_id in active for module_id in route_ids),
        "missing_cards": dict(sorted(missing_cards.items())),
        "missing_relics": sorted(missing_relics),
        "missing_capabilities": sorted(external),
        "missing_anchor_cards": dict(sorted(missing_anchor_cards.items())),
        "missing_anchor_relics": sorted(missing_anchor_relics),
        "completion_probability": round(probability, 10),
        "examples": list(route.get("examples") or ()),
    }


def _anchor_slots(module: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    activation = module.get("activation") or {}
    slots = list(activation.get("slots") or ())
    ids = {str(value) for value in activation.get("anchor_slots") or ()}
    if not ids:
        ids = {
            str(slot.get("id"))
            for slot in slots
            if str(slot.get("id")) in {"core", "payoff"}
        }
    return [slot for slot in slots if str(slot.get("id")) in ids]


def _module_route_key(
    plan: Mapping[str, Any], module_id: str
) -> tuple[Any, ...]:
    return (
        len(plan["missing_capabilities"]),
        len(plan["missing_relics"]),
        sum(plan["missing_cards"].values()),
        -float(plan["probability"]),
        module_id,
    )


def _route_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        len(row["missing_capabilities"]),
        len(row["missing_relics"]),
        sum(row["missing_cards"].values()),
        -float(row["completion_probability"]),
        -int(row["completed_core_modules"]),
        -int(row["active_route_modules"]),
        str(row["id"]),
    )


def _route_progress(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before_by_id = {str(row["id"]): row for row in before}
    improvements = []
    for current in after:
        previous = before_by_id[str(current["id"])]
        if _route_key(current) >= _route_key(previous):
            continue
        improvements.append(
            {
                "id": current["id"],
                "missing_card_reduction": (
                    sum(previous["missing_cards"].values())
                    - sum(current["missing_cards"].values())
                ),
                "completed_core_gain": (
                    int(current["completed_core_modules"])
                    - int(previous["completed_core_modules"])
                ),
                "anchor_reduction": (
                    sum(previous["missing_anchor_cards"].values())
                    + len(previous["missing_anchor_relics"])
                    - sum(current["missing_anchor_cards"].values())
                    - len(current["missing_anchor_relics"])
                ),
                "probability_gain": round(
                    float(current["completion_probability"])
                    - float(previous["completion_probability"]),
                    10,
                ),
                "before": dict(previous),
                "after": dict(current),
            }
        )
    improvements.sort(
        key=lambda row: (
            -int(row["completed_core_gain"]),
            -int(row["anchor_reduction"]),
            -int(row["missing_card_reduction"]),
            -float(row["probability_gain"]),
            str(row["id"]),
        )
    )
    return {
        "improved_route_count": len(improvements),
        "best_completed_core_gain": max(
            (int(row["completed_core_gain"]) for row in improvements), default=0
        ),
        "best_anchor_reduction": max(
            (int(row["anchor_reduction"]) for row in improvements), default=0
        ),
        "best_missing_card_reduction": max(
            (int(row["missing_card_reduction"]) for row in improvements), default=0
        ),
        "best_probability_gain": round(
            max((float(row["probability_gain"]) for row in improvements), default=0.0),
            10,
        ),
        "improved_routes": improvements[:5],
        "nearest_after": list(after[:3]),
    }


def _path_evidence(
    card: str,
    progress: Mapping[str, Any],
    deck: _Deck,
    data: Mapping[str, Any],
) -> dict[str, Any] | None:
    rates = data.get("offer_rates") or {}
    card_rate = float(rates.get(card, 0.0))
    modules = data["modules"]
    committed = _supports_owned_anchor(card, deck, modules)
    anchor = _is_anchor_card(card, modules)
    choices = []
    for row in progress.get("improved_routes") or ():
        after = row["after"]
        missing = after["missing_cards"]
        reachable = not (
            after["missing_relics"] or after["missing_capabilities"]
        ) and (not missing or float(after["completion_probability"]) > 0.0)
        if not reachable:
            continue
        bottleneck = bool(row["anchor_reduction"]) and all(
            float(rates.get(name, 0.0)) > card_rate
            for name in missing
        )
        kind = (
            "CORE_ACTIVATION"
            if int(row["completed_core_gain"]) > 0
            else "COMMITTED_SLOT"
            if committed and int(row["missing_card_reduction"]) > 0
            else "BOTTLENECK_ANCHOR"
            if bottleneck and anchor
            else None
        )
        if kind:
            choices.append({"kind": kind, **row})
    if not choices:
        return None
    tiers = {
        "CORE_ACTIVATION": 3,
        "COMMITTED_SLOT": 2,
        "BOTTLENECK_ANCHOR": 1,
    }
    best = max(
        choices,
        key=lambda row: (
            tiers[row["kind"]],
            int(row["completed_core_gain"]),
            int(row["anchor_reduction"]),
            int(row["missing_card_reduction"]),
            float(row["after"]["completion_probability"]),
            str(row["id"]),
        ),
    )
    return {
        "kind": best["kind"],
        "route_id": best["id"],
        "core_modules": best["after"]["core_modules"],
        "missing_before": best["before"]["missing_cards"],
        "missing_after": best["after"]["missing_cards"],
        "completion_probability": best["after"]["completion_probability"],
        "completed_core_gain": best["completed_core_gain"],
        "anchor_reduction": best["anchor_reduction"],
        "missing_card_reduction": best["missing_card_reduction"],
    }


def _supports_owned_anchor(
    card: str, deck: _Deck, modules: Sequence[Mapping[str, Any]]
) -> bool:
    for module in modules:
        anchors = _anchor_slots(module)
        if not anchors or not any(_slot_satisfied(slot, deck) for slot in anchors):
            continue
        for slot in (module.get("activation") or {}).get("slots") or ():
            if slot not in anchors and slot.get("required", True) and card in _slot_cards(slot):
                return True
    return False


def _is_anchor_card(card: str, modules: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        card in _slot_cards(slot)
        for module in modules
        for slot in _anchor_slots(module)
    )


def _slot_cards(slot: Mapping[str, Any]) -> set[str]:
    clauses = slot.get("any") or (slot,)
    return {
        str(fact["name"])
        for clause in clauses
        for fact in clause.get("all") or ()
        if fact.get("kind") == "CARD"
    } | {
        str(name)
        for clause in clauses
        for name in (clause.get("group") or {}).get("cards") or ()
    }




@lru_cache(maxsize=4096)
def _binomial_tail(trials: int, probability: float, minimum: int) -> float:
    if minimum <= 0:
        return 1.0
    if trials < minimum or probability <= 0:
        return 0.0
    return sum(
        math.comb(trials, hits)
        * probability**hits
        * (1.0 - probability) ** (trials - hits)
        for hits in range(minimum, trials + 1)
    )


def _horizon(data: Mapping[str, Any], act: int, floor: int) -> int:
    rows = data.get("horizons", {}).get(str(act)) or ()
    if not rows:
        return 0
    _, value = min(rows, key=lambda row: (abs(int(row[0]) - floor), int(row[0])))
    return max(0, round(float(value)))



def _deck(game_state: Mapping[str, Any]) -> _Deck:
    cards: Counter[str] = Counter()
    upgrades: dict[str, int] = {}
    for item in _sequence(game_state.get("deck")):
        name = _base_card(_name(item))
        if not name:
            continue
        cards[name] += 1
        upgrades[name] = max(upgrades.get(name, 0), _upgrade(item))
    return _Deck(cards, upgrades, _names(game_state.get("relics")))


def _names(value: object) -> set[str]:
    return {_name(item) for item in _sequence(value) if _name(item)}


def _name(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("id") or value.get("value")
    return str(value or "").strip()


def _base_card(value: object) -> str:
    name = re.sub(r"\+\d*$", "", str(value or "").strip())
    return {
        "AscendersBane": "Ascender's Bane", "Defend_R": "Defend", "Strike_R": "Strike",
    }.get(name, name)


def _upgrade(value: object) -> int:
    explicit = 0
    if isinstance(value, Mapping):
        try:
            explicit = int(value.get("upgrade", value.get("upgrades", 0)) or 0)
        except (TypeError, ValueError):
            explicit = 0
    match = re.search(r"\+(\d*)$", _name(value))
    return max(explicit, int(match.group(1) or 1) if match else 0)


def _sequence(value: object) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _integer(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["analyze_routes"]
