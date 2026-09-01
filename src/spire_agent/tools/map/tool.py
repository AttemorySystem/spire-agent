"""Map rendering and compact run context."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import resources
import json
import re
from typing import Any

from spire_agent.contracts import AgentKind, Decision, DecisionRequest, GameState
from spire_agent.subagents.map import MapDecisionError, MapTool
from spire_agent.subagents.llm import LLMMessage, LLMRequest, PromptLanguage
from spire_agent.tools.run_keys import RUN_ROUTE_KEY, key_view, route_context
from spire_agent.tools.sts_db import StsDB


class MapError(ValueError):
    pass


_CHOICE_X = re.compile(r"x\s*=\s*(-?\d+)")
_NODE_ID = re.compile(r"L(?P<layer>\d+)C(?P<column>\d+)")
_ROOM_NAMES = {
    "M": "Monster",
    "?": "Event",
    "E": "Elite",
    "E*": "Burning Elite",
    "R": "Rest",
    "T": "Chest",
    "$": "Shop",
    "BOSS": "Boss",
}
_SCHEMA = {
    "type": "object",
    "properties": {
        "choice_id": {"type": "integer", "minimum": 0},
        "path": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["choice_id", "path", "reason"],
    "additionalProperties": False,
}


class DefaultMapTool(MapTool):
    """Default map rendering and LLM implementation."""

    def __init__(
        self,
        llm: object,
        language: PromptLanguage | str = PromptLanguage.ENGLISH,
        encounter_readiness: object | None = None,
    ) -> None:
        self._llm = llm
        self._language = PromptLanguage.parse(language)
        self._encounter_readiness = encounter_readiness

    def try_decide(self, request: DecisionRequest) -> Decision | None:
        state = request.state
        if request.scope.owner is not AgentKind.MAP or state.screen.type != "MAP":
            return None
        if "choose" not in state.screen.commands:
            raise MapDecisionError("MAP screen does not expose the choose command")

        forced = forced_map_choice(state)
        if forced is not None:
            return _decision(forced, "map.single_choice", "only legal boss entrance")
        graph, options = render_map(state)
        options, gate = _policy_options(request, options, self._encounter_readiness)
        if len(options) == 1 and options[0].get("planned_path"):
            return _decision(
                options[0],
                str(gate.get("source") or "map.single_choice"),
                str(gate.get("reason") or "only legal exit"),
            )

        complete = getattr(self._llm, "complete", None)
        if not callable(complete):
            raise TypeError("MapAgent LLM has no complete() method")
        response = complete(build_prompt(request, graph, options, gate, self._language))
        data = getattr(response, "data", None)
        if not isinstance(data, Mapping):
            raise MapDecisionError("LLM response data must be an object")
        choice_id = data.get("choice_id")
        legal = {int(option["choice_id"]): option for option in options}
        if isinstance(choice_id, bool) or not isinstance(choice_id, int):
            raise MapDecisionError("LLM response choice_id must be an integer")
        if choice_id not in legal:
            raise MapDecisionError(
                f"LLM selected illegal map choice {choice_id}; "
                f"legal ids are {sorted(legal)}"
            )
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise MapDecisionError("LLM response reason must be a non-empty string")
        route_error = ""
        try:
            route = _validated_route(state, legal[choice_id], data.get("path"))
        except MapDecisionError as error:
            route, route_error = {}, str(error)
        option = {**legal[choice_id], **route}
        decision = _decision(
            option,
            str(gate.get("source") or "map.llm"),
            str(gate.get("reason") or reason.strip()),
        )
        model, usage = getattr(response, "model", ""), getattr(response, "usage", {})
        metrics = {
            **({"model": model} if model else {}),
            **({"usage": usage} if usage else {}),
            **({"route_error": route_error} if route_error else {}),
        }
        if not metrics:
            return decision
        return Decision(
            decision.command,
            decision.source,
            decision.reason,
            payload=decision.payload,
            metrics=metrics,
        )


def render_map(state: GameState) -> tuple[str, tuple[dict[str, object], ...]]:
    """Return the proven reachable graph and legal CommunicationMod choices."""

    nodes = _map_nodes(state)

    if not state.screen.choices:
        raise MapError("map choice list is empty")
    columns = tuple(_choice_x(choice) for choice in state.screen.choices)
    raw_current = state.screen.details.get("current_node")
    current = _coord(raw_current) if isinstance(raw_current, Mapping) else None
    current = current or (-1, 0)
    root_y = current[0] + 1

    def roots_exist(layer: int) -> bool:
        return all((layer, column) in nodes for column in columns)

    if not roots_exist(root_y):
        first_y = min(y for y, _ in nodes)
        last_y = max(y for y, _ in nodes)
        if current[0] > last_y and roots_exist(first_y):
            current, root_y = (first_y - 1, 0), first_y

    roots = tuple((root_y, column) for column in columns)
    for root in roots:
        if root not in nodes:
            raise MapError(f"map choice points to missing node {_node_id(root)}")

    missing_edges = [
        (parent, child)
        for parent, (_, children) in nodes.items()
        for child in children
        if child not in nodes and not _is_boss(child, nodes)
    ]
    implicit_boss = set()
    if int(state.facts.get("act") or 0) == 4 and len(missing_edges) == 1:
        parent, child = missing_edges[0]
        if (
            parent[0] == max(y for y, _ in nodes)
            and nodes[parent][0] in {"E", "E*"}
            and child[0] == parent[0] + 1
        ):
            implicit_boss.add(child)

    reachable: set[tuple[int, int]] = set()
    pending = list(roots)
    reaches_boss = False
    while pending:
        coord = pending.pop()
        if coord in reachable:
            continue
        if coord not in nodes:
            raise MapError(f"reachable edge points to missing node {_node_id(coord)}")
        reachable.add(coord)
        for child in nodes[coord][1]:
            if _is_boss(child, nodes) or child in implicit_boss:
                reaches_boss = True
            elif child not in nodes:
                raise MapError(
                    f"reachable edge points to missing node {_node_id(child)}"
                )
            else:
                pending.append(child)

    lines = [
        "MAP_GRAPH v1",
        "direction: layer increases toward BOSS",
        "node_id: L<layer>C<column>",
        "legend: M=Combat ?=Unknown(Event/Combat/Shop) E=Elite "
        "E*=BurningElite R=Rest T=Chest $=Shop",
    ]
    if current[0] < 0:
        lines.append("current: START")
    else:
        symbol = f" {nodes[current][0]}" if current in nodes else ""
        lines.append(f"current: {_node_id(current)}{symbol}")
    lines.append(
        "choices: "
        + " | ".join(
            f"{index}->{_node_id(root)}" for index, root in enumerate(roots)
        )
    )
    for layer in sorted({y for y, _ in reachable}):
        entries = []
        for coord in sorted(item for item in reachable if item[0] == layer):
            symbol, children = nodes[coord]
            targets = ",".join(
                "BOSS"
                if _is_boss(child, nodes) or child in implicit_boss
                else _node_id(child)
                for child in children
            )
            entries.append(f"C{coord[1]} {symbol}->[{targets}]")
        lines.append(f"L{layer:02d}: " + " | ".join(entries))
    boss = str(state.facts.get("act_boss") or state.facts.get("boss") or "UNKNOWN")
    suffix = "" if reaches_boss else " (no explicit reachable edge in input)"
    lines.append(f"BOSS: {boss}{suffix}")

    options = tuple(
        {
            "choice_id": index,
            "node": _node_id(root),
            "room": nodes[root][0],
            **_route_facts(root, nodes),
        }
        for index, root in enumerate(roots)
    )
    return "\n".join(lines) + "\n", options


def forced_map_choice(state: GameState) -> dict[str, object] | None:
    """Return the sole non-coordinate transition exposed by the map screen."""

    choices = state.screen.choices
    if len(choices) == 1 and str(choices[0]).strip().casefold() == "boss":
        return {"choice_id": 0, "node": "BOSS", "room": "BOSS"}
    return None


def _policy_options(
    request: DecisionRequest,
    options: tuple[dict[str, object], ...],
    readiness_tool: object | None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    state = request.state
    act = int(state.facts.get("act") or 0)
    keys = key_view(request.shared, state)
    selected = options
    constraints = []
    if act == 3 and not keys["emerald"]:
        capable = tuple(row for row in selected if row["burning_elite_reachable"])
        if capable:
            selected = capable
            constraints.append("preserve a route to the Burning Elite")
    if act == 3 and not keys["ruby"]:
        capable = tuple(
            row
            for row in selected
            if row["room"] == "R" or row["rest_reachable"]
        )
        if capable:
            selected = capable
            constraints.append("preserve a route to a Ruby Key rest site")

    readiness: Mapping[str, object] = {}
    rested_readiness: Mapping[str, object] = {}
    families = _needed_families(act, selected)
    if families:
        evaluate = getattr(readiness_tool, "evaluate", None)
        if callable(evaluate):
            readiness = evaluate(state, families)
            projected = _rested_state(state)
            rested_families = _needed_families(
                act, tuple(row for row in selected if _has_safe_rest(row))
            )
            if projected is not None and rested_families:
                rested_readiness = evaluate(projected, rested_families)
        selected = tuple(
            {
                **row,
                "encounter_readiness": _compact_readiness(readiness),
                **(
                    {"rest_readiness": _compact_readiness(rested_readiness)}
                    if _has_safe_rest(row) and rested_readiness
                    else {}
                ),
            }
            for row in selected
        )

    rotation = (
        readiness.get("elite_rotation")
        if isinstance(readiness, Mapping)
        else None
    )

    safer = tuple(row for row in selected if int(row["elites_before_rest"]) <= 1)
    if safer and len(safer) != len(selected):
        selected = safer
        constraints.append("avoid committing to multiple elites before the next rest")

    exposed = tuple(
        row for row in selected
        if readiness and _route_at_risk(row, act, readiness)
    )
    safe = tuple(row for row in selected if row not in exposed)
    if exposed and safe:
        selected = safe
        constraints.append("avoid a current-HP encounter that lacks survival support")
    elif exposed:
        selected = (min(exposed, key=lambda row: _danger_key(row, act, readiness)),)
        constraints.append("all routes are dangerous; choose the least committed risk")

    gate: dict[str, object] = {"options": list(selected)}
    if isinstance(rotation, Mapping):
        gate["elite_rotation"] = dict(rotation)
    if constraints:
        gate.update(
            source="map.run_policy",
            reason="; ".join(constraints),
        )
    return selected, gate


def _needed_families(
    act: int, options: Sequence[Mapping[str, object]]
) -> tuple[str, ...]:
    families = []
    for option in options:
        for row in option.get("forced_segment", ()):
            if not isinstance(row, Mapping):
                continue
            room = row.get("room")
            family = (
                "BURNING_ELITE" if room == "E*"
                else "ELITE" if room == "E"
                else None
            )
            if room == "M":
                family = "HALLWAY" if act in {2, 3} else None
            if room in {"M", "E", "E*"}:
                if family:
                    if family not in families:
                        families.append(family)
                    break
    return tuple(families)


def _has_safe_rest(option: Mapping[str, object]) -> bool:
    for row in option.get("forced_segment", ()):
        if not isinstance(row, Mapping):
            continue
        room = row.get("room")
        if room == "R":
            return True
        if room not in {"$", "T"}:
            return False
    return False


def _route_at_risk(
    option: Mapping[str, object], act: int, readiness: Mapping[str, object]
) -> bool:
    weak = int(readiness.get("weak_hallways_remaining") or 0)
    hallway = 0
    evidence = readiness
    combat_evidence_used = False
    for row in option.get("forced_segment", ()):
        if not isinstance(row, Mapping):
            continue
        room = row.get("room")
        if room == "R":
            rested = option.get("rest_readiness")
            if isinstance(rested, Mapping):
                evidence = rested
            elif combat_evidence_used:
                evidence = {}
            combat_evidence_used = False
            continue
        family = (
            "BURNING_ELITE" if room == "E*"
            else "ELITE" if room == "E"
            else None
        )
        if act in {2, 3} and room == "M":
            family = "WEAK_HALLWAY" if hallway < weak else "STRONG_HALLWAY"
            hallway += 1
        if room in {"M", "E", "E*"}:
            if combat_evidence_used:
                return True
            if family and _group_status(evidence, family) != "SUPPORTED":
                return True
            combat_evidence_used = True
    return False


def _group_status(readiness: Mapping[str, object], family: str) -> str:
    groups = readiness.get("groups")
    row = groups.get(family) if isinstance(groups, Mapping) else None
    if isinstance(row, Mapping):
        return str(row.get("status") or "UNAVAILABLE")
    return str(readiness.get("status") or "UNAVAILABLE") if family == "ELITE" else "UNAVAILABLE"


def _danger_key(
    option: Mapping[str, object], act: int, readiness: Mapping[str, object]
) -> tuple[int, float, float, int, int, int]:
    rooms = [
        row.get("room")
        for row in option.get("forced_segment", ())
        if isinstance(row, Mapping)
    ]
    before_rest = rooms[: rooms.index("R")] if "R" in rooms else rooms
    evidence = readiness
    family = None
    for room in rooms:
        if room == "R":
            rested = option.get("rest_readiness")
            evidence = rested if isinstance(rested, Mapping) else {}
        if room in {"E", "E*"}:
            family = "BURNING_ELITE" if room == "E*" else "ELITE"
            break
        if room == "M":
            if act in {2, 3}:
                family = (
                    "WEAK_HALLWAY"
                    if int(readiness.get("weak_hallways_remaining") or 0) > 0
                    else "STRONG_HALLWAY"
                )
            break
    groups = evidence.get("groups") if isinstance(evidence, Mapping) else None
    row = groups.get(family) if isinstance(groups, Mapping) else None
    survival = float(row.get("estimated_survival") or 0) if isinstance(row, Mapping) else 0
    end_hp = float(row.get("expected_end_hp_on_win") or 0) if isinstance(row, Mapping) else 0
    return (
        sum(room in {"M", "E", "E*"} for room in before_rest),
        -survival,
        -end_hp,
        sum(room in {"E", "E*"} for room in before_rest),
        len(before_rest),
        int(option["choice_id"]),
    )


def _compact_readiness(value: Mapping[str, object]) -> dict[str, object]:
    groups = value.get("groups") if isinstance(value, Mapping) else None
    return {
        key: item
        for key, item in {
            "status": value.get("status") if isinstance(value, Mapping) else "UNAVAILABLE",
            "entry_hp": value.get("entry_hp") if isinstance(value, Mapping) else None,
            "weak_hallways_remaining": (
                value.get("weak_hallways_remaining")
                if isinstance(value, Mapping)
                else None
            ),
            "groups": {
                family: {
                    key: row[key]
                    for key in (
                        "status",
                        "estimated_survival",
                        "expected_end_hp_on_win",
                        "worst_target",
                    )
                    if key in row
                }
                for family, row in groups.items()
                if isinstance(row, Mapping)
            } if isinstance(groups, Mapping) else {},
        }.items()
        if item not in (None, {})
    }


def _rested_state(state: GameState) -> GameState | None:
    facts = state.facts
    current, maximum = int(facts.get("current_hp") or 0), int(facts.get("max_hp") or 0)
    if current >= maximum or _relic_blocker(facts, "mark of the bloom"):
        return None
    relics = set(_names(facts.get("relics")))
    gain = (
        3 * (_deck_size(facts.get("deck")) // 5)
        if "Eternal Feather" in relics
        else 0
    )
    if "Coffee Dripper" not in relics:
        gain += int(maximum * 0.30) + (15 if "Regal Pillow" in relics else 0)
    if gain <= 0:
        return None
    healed = min(maximum, current + gain)
    return GameState(
        state.owner_hint,
        f"{state.scope_id}:rest_projection",
        state.screen,
        state.terminal,
        {**facts, "current_hp": healed},
        state.combat,
    )


def _route_facts(
    root: tuple[int, int],
    nodes: Mapping[tuple[int, int], tuple[str, tuple[tuple[int, int], ...]]],
) -> dict[str, object]:
    reachable: set[tuple[int, int]] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in reachable or current not in nodes:
            continue
        reachable.add(current)
        pending.extend(child for child in nodes[current][1] if child in nodes)

    segment = []
    current = root
    reaches_boss = False
    visited: set[tuple[int, int]] = set()
    while current in nodes and current not in visited:
        visited.add(current)
        symbol, children = nodes[current]
        segment.append({"node": _node_id(current), "room": symbol})
        live = [
            child
            for child in children
            if child in nodes and not _is_boss(child, nodes)
        ]
        reaches_boss = bool(children) and not live and all(
            child not in nodes or _is_boss(child, nodes) for child in children
        )
        if len(live) != 1:
            break
        current = live[0]
    forced_rooms = [row["room"] for row in segment]
    result = {
        "rest_reachable": any(
            nodes[node][0] == "R" for node in reachable if node != root
        ),
        "burning_elite_reachable": any(nodes[node][0] == "E*" for node in reachable),
        **_route_counts(forced_rooms),
        "forced_shop_count": forced_rooms.count("$"),
        "forced_segment": segment,
    }
    if reaches_boss:
        path = tuple(row["node"] for row in segment) + ("BOSS",)
        result.update(_route_payload(path, nodes))
    return result


def _validated_route(
    state: GameState,
    option: Mapping[str, object],
    raw_path: object,
) -> dict[str, object]:
    if not _sequence(raw_path) or len(raw_path) < 2:
        raise MapDecisionError("LLM response path must include rooms and BOSS")
    labels = tuple(str(item).strip() for item in raw_path)
    if labels[0] != option["node"] or labels[-1].upper() != "BOSS":
        raise MapDecisionError(
            "LLM response path must start at the selected choice and end at BOSS"
        )
    path = (*labels[:-1], "BOSS")
    nodes = _map_nodes(state)
    coordinates = []
    for label in path[:-1]:
        match = _NODE_ID.fullmatch(label)
        coordinate = (
            (int(match.group("layer")), int(match.group("column")))
            if match else None
        )
        if coordinate not in nodes:
            raise MapDecisionError(
                f"LLM response path references missing node {label!r}"
            )
        coordinates.append(coordinate)
    for parent, child in zip(coordinates, coordinates[1:]):
        if child not in nodes[parent][1]:
            raise MapDecisionError(
                f"LLM response path contains non-edge "
                f"{_node_id(parent)} -> {_node_id(child)}"
            )
    children = nodes[coordinates[-1]][1]
    if not any(
        child not in nodes or _is_boss(child, nodes) for child in children
    ):
        raise MapDecisionError(
            "LLM response path final room does not lead to BOSS"
        )
    return _route_payload(path, nodes)


def _route_payload(
    path: Sequence[str],
    nodes: Mapping[tuple[int, int], tuple[str, object]],
) -> dict[str, object]:
    rooms = []
    for label in path:
        match = _NODE_ID.fullmatch(label)
        symbol = (
            nodes[(int(match.group("layer")), int(match.group("column")))][0]
            if match else "BOSS"
        )
        rooms.append(_ROOM_NAMES.get(symbol, symbol))
    return {
        "planned_path": list(path),
        "planned_rooms": rooms,
        "future_rests": sum(room == "Rest" for room in rooms[1:]),
        **_route_counts(
            next((symbol for symbol, name in _ROOM_NAMES.items() if name == room), room)
            for room in rooms
        ),
    }


def _route_counts(rooms: Sequence[str] | object) -> dict[str, int]:
    values = list(rooms)
    before_rest = values[: values.index("R")] if "R" in values else values
    return {
        "combats_before_rest": sum(room in {"M", "E", "E*"} for room in before_rest),
        "elites_before_rest": sum(room in {"E", "E*"} for room in before_rest),
    }


def run_summary(state: GameState) -> dict[str, Any]:
    facts = state.facts
    result = {
        key: _jsonable(facts[key])
        for key in (
            "class",
            "ascension_level",
            "act",
            "floor",
            "current_hp",
            "max_hp",
            "gold",
            "act_boss",
            "has_ruby_key",
            "has_sapphire_key",
            "has_emerald_key",
        )
        if key in facts
    }
    deck = Counter(_names(facts.get("deck")))
    if deck:
        result["deck"] = [
            f"{count}x {name}" if count > 1 else name
            for name, count in sorted(deck.items())
        ]
    for key in ("relics", "potions"):
        names = [name for name in _names(facts.get(key)) if name != "Potion Slot"]
        if names:
            result[key] = names
    rest_blocker = _relic_blocker(
        facts, "coffee dripper", "mark of the bloom"
    )
    smithing_blocker = _relic_blocker(facts, "fusion hammer")
    healing_sources = []
    if not rest_blocker:
        healing_sources.append("Rest")
    if (
        not _relic_blocker(facts, "mark of the bloom")
        and _relic_blocker(facts, "eternal feather")
    ):
        healing_sources.append("Eternal Feather")
    result["campfire_rest_healing_available"] = not rest_blocker
    result["campfire_smithing_available"] = not smithing_blocker
    result["rest_site_healing_sources"] = healing_sources
    if rest_blocker:
        result["campfire_rest_healing_blocked_by"] = rest_blocker
    if smithing_blocker:
        result["campfire_smithing_blocked_by"] = smithing_blocker
    entity_facts = _entity_facts(facts)
    if entity_facts:
        result["entity_facts"] = entity_facts
    return result


def build_prompt(
    request: DecisionRequest,
    graph: str,
    options: tuple[dict[str, object], ...],
    gate: Mapping[str, object],
    language: PromptLanguage,
) -> LLMRequest:
    values = {
        "{{MAP_GRAPH}}": graph.rstrip(),
        "{{RUN_JSON}}": json.dumps(run_summary(request.state), ensure_ascii=False, indent=2),
        "{{SHARED_JSON}}": json.dumps(_jsonable(request.shared), ensure_ascii=False, indent=2),
        "{{ROUTE_JSON}}": json.dumps(_jsonable(gate), ensure_ascii=False, indent=2),
        "{{ALLOWED_IDS}}": json.dumps([option["choice_id"] for option in options]),
    }
    user = _prompt_text("user", language)
    for key, value in values.items():
        user = user.replace(key, value)
    return LLMRequest(
        "map.choose_exit",
        (
            LLMMessage("system", _prompt_text("system", language)),
            LLMMessage("user", user),
        ),
        _SCHEMA,
    )


def _entity_facts(facts: Mapping[str, object]) -> dict[str, object]:
    db, result = StsDB(), {}
    for key, query in (("relics", db.relic), ("potions", db.potion)):
        rows = [
            query(name)
            for name in _names(facts.get(key))
            if name != "Potion Slot"
        ]
        if known := [row for row in rows if row is not None]:
            result[key] = known
    return result


def _decision(option: Mapping[str, object], source: str, reason: str) -> Decision:
    return Decision(
        f"choose {option['choice_id']}",
        source,
        reason,
        payload={
            "choice_id": option["choice_id"],
            "next_node": option["node"],
            "room": option["room"],
            RUN_ROUTE_KEY: route_context(option),
        },
    )


@lru_cache(maxsize=4)
def _prompt_text(name: str, language: PromptLanguage) -> str:
    return (
        resources.files("spire_agent.subagents")
        .joinpath("prompts", "map", f"{name}.{language.value}.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def _coord(value: object) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        x, y = int(value.get("x", -1)), int(value.get("y", -1))
    except (TypeError, ValueError):
        return None
    return (y, x) if x >= 0 and y >= 0 else None


def _map_nodes(
    state: GameState,
) -> dict[tuple[int, int], tuple[str, tuple[tuple[int, int], ...]]]:
    raw_map = state.facts.get("map")
    if not _sequence(raw_map):
        raise MapError("game state has no map array")
    nodes = {}
    for raw in raw_map:
        if not isinstance(raw, Mapping) or (coord := _coord(raw)) is None:
            continue
        symbol = str(raw.get("symbol") or "UNKNOWN")
        if symbol == "E" and raw.get("is_burning"):
            symbol = "E*"
        nodes[coord] = (
            symbol,
            tuple(
                sorted(
                    child
                    for value in _sequence(raw.get("children"))
                    if (child := _coord(value)) is not None
                )
            ),
        )
    if not nodes:
        raise MapError("game state map has no valid nodes")
    return nodes


def _choice_x(value: object) -> int:
    if isinstance(value, Mapping):
        raw = value.get("x")
    else:
        match = _CHOICE_X.search(str(value))
        raw = match.group(1) if match else -1
    try:
        column = int(raw)
    except (TypeError, ValueError):
        column = -1
    if column < 0:
        raise MapError(f"could not parse map choice {value!r}")
    return column


def _is_boss(
    coord: tuple[int, int],
    nodes: Mapping[tuple[int, int], tuple[str, object]],
) -> bool:
    return coord[0] >= 15 or (coord in nodes and nodes[coord][0] == "B")


def _node_id(coord: tuple[int, int]) -> str:
    return f"L{coord[0]:02d}C{coord[1]}"


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _names(value: object) -> list[str]:
    result = []
    for item in _sequence(value):
        name = item.get("name") or item.get("id") if isinstance(item, Mapping) else item
        if name not in (None, ""):
            result.append(str(name))
    return result


def _deck_size(value: object) -> int:
    return sum(
        max(1, int(item.get("count") or 1)) if isinstance(item, Mapping) else 1
        for item in _sequence(value)
    )


def _relic_blocker(facts: Mapping[str, object], *blocked: str) -> str:
    return next(
        (name for name in _names(facts.get("relics")) if name.casefold() in blocked),
        "",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if _sequence(value):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


__all__ = [
    "DefaultMapTool",
    "MapError",
    "build_prompt",
    "forced_map_choice",
    "render_map",
    "run_summary",
]
