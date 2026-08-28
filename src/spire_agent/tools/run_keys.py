"""Three-key policy reconstructed from confirmed game commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from spire_agent.contracts import DecisionRequest, GameState


RUN_KEYS_KEY = "run_keys"
RUN_ROUTE_KEY = "run_route"
_COLOURS = ("ruby", "emerald", "sapphire")


def initial_keys(state: GameState) -> dict[str, object]:
    owned = {
        colour: bool(state.facts.get(f"has_{colour}_key"))
        for colour in _COLOURS
    }
    return _key_view(owned)


def key_view(shared: Mapping[str, object], state: GameState) -> dict[str, object]:
    raw = shared.get(RUN_KEYS_KEY)
    owned = {
        colour: bool(raw.get(colour)) if isinstance(raw, Mapping) else False
        for colour in _COLOURS
    }
    for colour in _COLOURS:
        owned[colour] = owned[colour] or bool(
            state.facts.get(f"has_{colour}_key")
        )
    return _key_view(owned)


def acquire(
    shared: Mapping[str, object], state: GameState, colour: object
) -> dict[str, object]:
    name = _normalize(colour)
    if name not in _COLOURS:
        raise ValueError(f"unknown run key colour {colour!r}")
    owned = key_view(shared, state)
    owned[name] = True
    return _key_view(owned)


def rest_policy(request: DecisionRequest) -> dict[str, Any] | None:
    """Apply Ruby Key deadlines and proven survival-critical resting."""

    if request.state.screen.type != "REST":
        return None
    choices = tuple(_choice_label(choice) for choice in request.state.screen.choices)
    recall = next(
        (
            index
            for index, choice in enumerate(choices)
            if _normalize(choice) == "recall"
        ),
        None,
    )
    act = _integer(request.state.facts.get("act"))
    legal = tuple(range(len(choices)))
    needs_ruby = recall is not None and not key_view(request.shared, request.state)["ruby"]
    if needs_ruby and act >= 3:
        route = request.shared.get(RUN_ROUTE_KEY)
        future_rests = _integer(route.get("future_rests")) if isinstance(route, Mapping) else 0
        floor = _integer(request.state.facts.get("floor"))
        if future_rests <= 0 or floor >= 49:
            return {
                "forced_choice_id": recall,
                "legal_choice_ids": (recall,),
                "acquired_key": "ruby",
                "reason": "take the Ruby Key at the final guaranteed Act 3 rest site",
            }

    route = request.shared.get(RUN_ROUTE_KEY)
    threat = route_threat(route)
    rest = next(
        (index for index, choice in enumerate(choices) if _normalize(choice) == "rest"),
        None,
    )
    readiness = route.get("encounter_readiness") if isinstance(route, Mapping) else None
    rested = route.get("rest_readiness") if isinstance(route, Mapping) else None
    fresh = isinstance(readiness, Mapping) and _integer(
        readiness.get("entry_hp")
    ) == _integer(request.state.facts.get("current_hp"))
    relics = {_normalize(_choice_label(item)) for item in request.state.facts.get("relics") or ()}
    cards = {_normalize(_choice_label(item)) for item in request.state.facts.get("deck") or ()}
    rested_threat = route_threat(route, rested)
    apotheosis_rest = (
        "apotheosis" in cards
        and threat["family"] == rested_threat["family"]
        and threat["status"] != "SUPPORTED"
        and _expected_end_hp(rested, threat["family"])
        > _expected_end_hp(readiness, threat["family"])
    )
    if (
        rest is not None
        and fresh
        and (
            (
                threat["status"] in {"AT_RISK", "INCONCLUSIVE"}
                and rested_threat["status"] == "SUPPORTED"
            )
            or apotheosis_rest
        )
        and threat["rests_before"] > 0
        and _integer(request.state.facts.get("current_hp"))
        < _integer(request.state.facts.get("max_hp"))
        and not relics & {"coffeedripper", "markofthebloom"}
    ):
        return {
            "forced_choice_id": rest,
            "legal_choice_ids": (rest,),
            "reason": (
                "rest before the unresolved "
                f"{threat['family'].lower().replace('_', ' ')} encounter"
            ),
        }
    if needs_ruby and act < 3:
        return {
            "legal_choice_ids": tuple(index for index in legal if index != recall),
            "reason": "Ruby Key recall is deferred until Act 3",
        }
    if needs_ruby:
        future_rests = _integer(route.get("future_rests")) if isinstance(route, Mapping) else 0
        return {
            "legal_choice_ids": legal,
            "reason": (
                f"Ruby Key may be deferred across {future_rests} "
                "reachable future rest site(s)"
            ),
        }
    return None


def route_threat(
    route: object, readiness: object | None = None
) -> dict[str, object]:
    """Return the strongest cached threat before the next recovery boundary."""

    if not isinstance(route, Mapping):
        return {"family": "", "rests_before": 0, "status": "UNAVAILABLE"}
    readiness = readiness or route.get("encounter_readiness")
    weak = (
        _integer(readiness.get("weak_hallways_remaining"))
        if isinstance(readiness, Mapping)
        else 0
    )
    groups = readiness.get("groups") if isinstance(readiness, Mapping) else None
    hallway_evidence = isinstance(groups, Mapping) and (
        "WEAK_HALLWAY" in groups or "STRONG_HALLWAY" in groups
    )
    rooms = route.get("planned_rooms")
    if not isinstance(rooms, Sequence) or isinstance(rooms, (str, bytes)):
        segment = route.get("forced_segment")
        rooms = [row.get("room") for row in segment or () if isinstance(row, Mapping)]
    names = [_normalize(room) for room in rooms or ()]
    encounters = {"monster", "m", "elite", "burningelite", "e"}
    first_encounter = next(
        (index for index, name in enumerate(names) if name in encounters), len(names)
    )
    first_rest = next(
        (index for index, name in enumerate(names) if name in {"rest", "r"}),
        len(names),
    )
    start = first_rest + 1 if first_rest < first_encounter else 0
    stop = next(
        (index for index in range(start, len(names)) if names[index] in {"rest", "r"}),
        len(names),
    )
    families = []
    hallways = 0
    for name in names[start:stop]:
        if name in {"elite", "burningelite", "e"}:
            families.append("ELITE")
        elif hallway_evidence and name in {"monster", "m"}:
            families.append("WEAK_HALLWAY" if hallways < weak else "STRONG_HALLWAY")
            hallways += 1
    rests = int(start > 0)
    statuses = {}
    for family in dict.fromkeys(families):
        group = groups.get(family) if isinstance(groups, Mapping) else None
        statuses[family] = (
            str(group.get("status") or "UNAVAILABLE")
            if isinstance(group, Mapping)
            else str(readiness.get("status") or "UNAVAILABLE")
            if isinstance(readiness, Mapping) and family == "ELITE"
            else "UNAVAILABLE"
        )
    order = {"AT_RISK": 0, "INCONCLUSIVE": 1, "UNAVAILABLE": 2, "SUPPORTED": 3}
    family = min(statuses, key=lambda item: order.get(statuses[item], 2), default="")
    status = statuses.get(family, "UNAVAILABLE")
    return {"family": family, "rests_before": rests, "status": status}


def reward_key(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    kind = _normalize(value.get("reward_type") or value.get("type"))
    if kind in {"emeraldkey", "sapphirekey"}:
        return kind.removesuffix("key")
    if kind != "key":
        return ""
    colour = _normalize(value.get("value") or value.get("key"))
    return colour if colour in {"emerald", "sapphire"} else ""


def route_context(option: Mapping[str, object]) -> dict[str, object]:
    return {
        key: option[key]
        for key in (
            "node",
            "room",
            "future_rests",
            "burning_elite_reachable",
            "combats_before_rest",
            "elites_before_rest",
            "encounter_readiness",
            "rest_readiness",
            "forced_segment",
            "planned_path",
            "planned_rooms",
        )
        if key in option
    }


def _key_view(owned: Mapping[str, object]) -> dict[str, object]:
    missing = [colour for colour in _COLOURS if not bool(owned.get(colour))]
    return {
        **{colour: bool(owned.get(colour)) for colour in _COLOURS},
        "complete": not missing,
        "missing": missing,
        "objective": "A20_HEART",
    }


def _choice_label(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("value") or value.get("text") or "")
    return str(value)


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _expected_end_hp(readiness: object, family: object) -> float:
    groups = readiness.get("groups") if isinstance(readiness, Mapping) else None
    row = groups.get(family) if isinstance(groups, Mapping) else None
    try:
        return float(row.get("expected_end_hp_on_win") or 0) if isinstance(row, Mapping) else 0.0
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "RUN_KEYS_KEY",
    "RUN_ROUTE_KEY",
    "acquire",
    "initial_keys",
    "key_view",
    "rest_policy",
    "reward_key",
    "route_context",
    "route_threat",
]
