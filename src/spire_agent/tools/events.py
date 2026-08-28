"""Known event guidance and evidence-based safety choices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re

from spire_agent.contracts import GameState
from spire_agent.tools.run_keys import route_threat


_RULES = {
    "neow": ("neow",),
    "the_nest": ("the nest", "nest"),
    "nloth": ("n loth", "nloth"),
    "woman_in_blue": ("the woman in blue", "woman in blue"),
    "masked_bandits": ("masked bandits",),
    "golden_idol": ("golden idol",),
    "liars_game": ("liars game", "liar s game"),
    "mind_bloom": ("mind bloom", "mindbloom"),
    "dead_adventurer": ("dead adventurer",),
    "knowing_skull": ("knowing skull",),
    "match_and_keep": ("match and keep",),
}


def event_rule(state: GameState) -> str | None:
    """Return the stable rule key for a recognized event."""

    if state.screen.type != "EVENT":
        return None
    name = _event_name(state.screen.details)
    if not name:
        return None
    for key, names in _RULES.items():
        if any(name == _normalize(alias) for alias in names):
            return key
    return None


def forced_event_choice(
    state: GameState, shared: Mapping[str, object] | None = None
) -> int | None:
    """Return choices whose downside is forbidden by available evidence."""

    rule = event_rule(state)
    labels = [_choice_label(choice) for choice in state.screen.choices]
    normalized = [_normalize(label) for label in labels]
    survival = _survival_choice(state, shared or {})
    if survival is not None:
        return survival
    if rule == "match_and_keep":
        # Replaying the leading pair keeps a match, but spends mismatched
        # attempts instead of deliberately adding random cards to the deck.
        return 0 if labels else None
    if rule == "dead_adventurer":
        current = _number(state.facts.get("current_hp"))
        maximum = _number(state.facts.get("max_hp"))
        if maximum > 0 and current / maximum < 0.80:
            return _find(normalized, "leave")
        return None
    if rule == "knowing_skull":
        current = _number(state.facts.get("current_hp"))
        maximum = _number(state.facts.get("max_hp"))
        return _find(normalized, "leave") if maximum > 0 and current / maximum < 0.55 else None
    if rule == "liars_game" and not _shop_before_combat(shared or {}):
        return _find(normalized, "disagree", "leave", "no")
    if rule != "the_nest":
        return None
    for marker in ("smash and grab", "gold"):
        for index, label in enumerate(normalized):
            if marker in label:
                return index
    for index, label in enumerate(normalized):
        if "leave" in label:
            return index
    forbidden = {
        index
        for index, label in enumerate(normalized)
        if "ritual dagger" in label or "stay in line" in label
    }
    safe = [index for index in range(len(labels)) if index not in forbidden]
    return safe[0] if forbidden and safe else None


def _survival_choice(state: GameState, shared: Mapping[str, object]) -> int | None:
    """Reject visible HP loss when it lowers entry HP for an at-risk fight."""

    route = shared.get("run_route")
    if not isinstance(route, Mapping):
        return None
    readiness = route.get("encounter_readiness")
    if (
        not isinstance(readiness, Mapping)
        or int(_number(readiness.get("entry_hp"))) != int(_number(state.facts.get("current_hp")))
    ):
        return None
    threat = route_threat(route)
    if not threat["family"] or threat["status"] != "AT_RISK":
        return None
    losses = _event_costs(state)
    if not losses or not any(loss > 0 for loss, _ in losses.values()):
        return None
    current = int(_number(state.facts.get("current_hp")))
    maximum = int(_number(state.facts.get("max_hp")))
    relics = {_normalize(_choice_label(item)) for item in state.facts.get("relics") or ()}

    def projected(cost: tuple[int, int]) -> int:
        loss, max_gain = cost
        hp, max_hp = max(0, current - loss), maximum + max_gain
        if "mark of the bloom" in relics or "coffee dripper" in relics:
            return hp
        heal = math.floor(max_hp * 0.30) + (15 if "regal pillow" in relics else 0)
        for _ in range(int(threat["rests_before"])):
            hp = min(max_hp, hp + heal)
        return hp

    safe = [(projected(cost), index) for index, cost in losses.items() if cost[0] == 0]
    unsafe = [projected(cost) for cost in losses.values() if cost[0] > 0]
    if safe and unsafe and max(safe)[0] > max(unsafe):
        return max(safe)[1]
    return None


def _event_costs(state: GameState) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    options = state.screen.details.get("options")
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for fallback, option in enumerate(options):
            if not isinstance(option, Mapping) or option.get("disabled"):
                continue
            index = option.get("choice_index", fallback)
            if not isinstance(index, int) or not 0 <= index < len(state.screen.choices):
                continue
            text = str(option.get("text") or option.get("label") or "")
            loss = re.search(r"\blose\s+(\d+)\s+hp\b", text, re.IGNORECASE)
            gain = re.search(r"\bgain\s+(\d+)\s+max\s+hp\b", text, re.IGNORECASE)
            result[index] = (
                int(loss.group(1)) if loss else 0,
                int(gain.group(1)) if gain else 0,
            )
    return result


def _shop_before_combat(shared: Mapping[str, object]) -> bool:
    route = shared.get("run_route")
    segment = route.get("planned_rooms") if isinstance(route, Mapping) else ()
    if not isinstance(segment, Sequence) or isinstance(segment, (str, bytes)):
        segment = route.get("forced_segment") if isinstance(route, Mapping) else ()
    if not isinstance(segment, Sequence) or isinstance(segment, (str, bytes)):
        return False
    for row in segment[1:]:
        room = str(row.get("room") or "") if isinstance(row, Mapping) else str(row)
        name = _normalize(room)
        if room == "$" or name == "shop":
            return True
        if name in {"m", "e", "monster", "elite", "burning elite"}:
            return False
    return False


def _find(labels: list[str], *markers: str) -> int | None:
    return next(
        (index for marker in markers for index, label in enumerate(labels) if marker in label),
        None,
    )


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _event_name(details: Mapping[str, object]) -> str:
    for key in ("event_name", "event", "name", "event_id"):
        value = details.get(key)
        if value:
            return _normalize(value)
    return ""


def _choice_label(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("value") or value.get("text") or "")
    return str(value)


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


__all__ = ["event_rule", "forced_event_choice"]
