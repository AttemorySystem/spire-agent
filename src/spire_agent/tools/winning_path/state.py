"""Project a framework DecisionRequest into canonical Winning Path state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re
from typing import Any

from spire_agent.contracts import DecisionRequest
from spire_agent.tools.run_keys import RUN_ROUTE_KEY, key_view

from .preference import normalize_card
from .contracts import DecisionState


class StateProjectionError(ValueError):
    """A request is outside the frozen card-reward state contract."""


_ROUTE_FIELDS = (
    "node",
    "room",
    "future_rests",
    "burning_elite_reachable",
    "forced_segment",
    "planned_path",
    "planned_rooms",
)


def project_state(request: DecisionRequest) -> DecisionState:
    if request.state.screen.type != "CARD_REWARD":
        raise StateProjectionError("state projection requires CARD_REWARD")

    facts = request.state.facts
    missing: list[str] = []
    run_fields = {
        "seed": _integer(_first(facts, "seed")),
        "character": _first(facts, "class", "character"),
        "act": _integer(_first(facts, "act")),
        "floor": _integer(_first(facts, "floor")),
        "ascension": _integer(_first(facts, "ascension_level", "ascension")),
        "boss": _string(_first(facts, "act_boss", "boss")),
        "hp": _integer(_first(facts, "current_hp", "hp")),
        "max_hp": _integer(_first(facts, "max_hp")),
        "gold": _integer(_first(facts, "gold")),
        "room_type": _string(_first(facts, "room_type")),
        "keys": key_view(request.shared, request.state),
    }
    required_aliases = {
        "run.seed": ("seed",),
        "run.character": ("class", "character"),
        "run.act": ("act",),
        "run.floor": ("floor",),
        "run.hp": ("current_hp", "hp"),
        "run.max_hp": ("max_hp",),
    }
    for label, aliases in required_aliases.items():
        if _first(facts, *aliases) is None:
            missing.append(label)

    cards = tuple(_card_instance(raw) for raw in _array(facts.get("deck")))
    cards = tuple(card for card in cards if card["id"])
    counts = Counter(str(card["id"]) for card in cards)
    upgrade_counts = Counter(
        str(card["id"]) for card in cards if int(card["upgrades"]) > 0
    )
    max_upgrades: dict[str, int] = {}
    for card in cards:
        name = str(card["id"])
        max_upgrades[name] = max(
            max_upgrades.get(name, 0), int(card["upgrades"])
        )
    if "deck" not in facts:
        missing.append("deck.cards")

    relics = tuple(
        relic for relic in (_relic(raw) for raw in _array(facts.get("relics")))
        if relic["id"]
    )
    potion_slots = tuple(
        potion
        for potion in (_potion(raw) for raw in _array(facts.get("potions")))
        if potion["id"]
    )
    potions = tuple(
        potion for potion in potion_slots
        if _compact(potion["id"]) not in {"potionslot", "none"}
    )
    if "potions" not in facts:
        missing.append("assets.potion_slots")

    route_raw = request.shared.get(RUN_ROUTE_KEY)
    route = (
        {key: _plain(route_raw[key]) for key in _ROUTE_FIELDS if key in route_raw}
        if isinstance(route_raw, Mapping)
        else {}
    )
    details = request.state.screen.details
    offered_raw = _array(details.get("cards")) or request.state.screen.choices
    offered_cards = tuple(_card_instance(raw) for raw in offered_raw)
    offered_cards = tuple(card for card in offered_cards if card["id"])
    offered = tuple(str(card["id"]) for card in offered_cards)
    if not offered:
        missing.append("reward.offered")

    return DecisionState(
        run=run_fields,
        deck={
            "cards": cards,
            "counts": dict(sorted(counts.items())),
            "upgrade_counts": dict(sorted(upgrade_counts.items())),
            "max_upgrades": dict(sorted(max_upgrades.items())),
            "physical_size": len(cards),
        },
        assets={
            "relics": relics,
            "potions": potions,
            "potion_slots": potion_slots,
        },
        route=route,
        reward={
            "kind": _reward_kind(request),
            "offered": offered,
            "offered_cards": offered_cards,
            "singing_bowl": _singing_bowl_available(request),
        },
        missing_facts=tuple(missing),
    )


def _singing_bowl_available(request: DecisionRequest) -> bool:
    details = request.state.screen.details
    return bool(
        details.get("singing_bowl")
        or details.get("bowl_available")
        or request.state.facts.get("singing_bowl")
        or any(
            str(row.get("name") or row.get("id") or "") == "Singing Bowl"
            if isinstance(row, Mapping)
            else str(row) == "Singing Bowl"
            for row in request.state.facts.get("relics") or ()
        )
    )


def _card_instance(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    display = normalize_card(value)
    raw_name = str(
        raw.get("name") or raw.get("id") or raw.get("value") or value or ""
    ).strip()
    explicit = raw.get("upgrades", raw.get("upgrade"))
    match = re.search(r"\+(\d*)$", raw_name)
    upgrades = _integer(explicit)
    if explicit is None and match:
        upgrades = int(match.group(1) or 1)
    result: dict[str, Any] = {"id": display, "upgrades": max(0, upgrades)}
    if "misc" in raw and raw["misc"] is not None:
        result["misc"] = _plain(raw["misc"])
    if any(bool(raw.get(field)) for field in ("bottled", "is_bottled", "bottle")):
        result["bottled"] = True
    return result


def _relic(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    name = _string(raw.get("name") or raw.get("id") or value)
    result: dict[str, Any] = {"id": name}
    if "counter" in raw and raw["counter"] is not None:
        result["counter"] = _integer(raw["counter"])
    return result


def _potion(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    name = _string(
        raw.get("name") or raw.get("id") or raw.get("potion") or value
    )
    result: dict[str, Any] = {"id": name}
    for field in ("can_use", "requires_target"):
        if field in raw:
            result[field] = bool(raw[field])
    return result


def _reward_kind(request: DecisionRequest) -> str:
    details = request.state.screen.details
    raw = details.get("reward_type") or details.get("kind")
    if raw:
        return str(raw).strip().casefold().replace(" ", "_")
    return "combat_card_reward"


def _first(value: Mapping[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return None


def _array(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _string(value: object) -> str:
    return "" if value is None else str(value).strip()


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = ["StateProjectionError", "project_state"]
