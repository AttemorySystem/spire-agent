"""Small fixed resolver for Winning Path evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .preference import PreferenceTable, normalize_card
from .contracts import DecisionState
from .parameters import load_policy
from .protocol import PROTOCOL_VERSION, normalize_character


class ResolverError(ValueError):
    pass


def resolve(
    state: DecisionState,
    catalog: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {**_resolve(state, catalog, evidence), "mode": "LIVE_POLICY"}


def _resolve(
    state: DecisionState,
    catalog: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = _rows(evidence.get("candidates"))
    choice_ids = [int(row.get("choice_id", -1)) for row in candidates]
    if not candidates or choice_ids != list(range(len(candidates))):
        raise ResolverError(
            "candidate evidence must be non-empty, ordered, and contiguous"
        )
    eligible = tuple(row for row in candidates if not row.get("rejected"))
    if not eligible:
        if state.reward.get("must_pick"):
            return _advice(candidates, "COMMITTED_REWARD_ALL_REJECTED")
        return _non_card(state, len(candidates), "ALL_HARD_REJECTED", ("ALL_REJECTED",))

    policy = load_policy(_character(state))
    fixed, parameters = policy["fixed"], policy["parameters"]
    template_levels = tuple(fixed["template_levels"])
    transition_levels = tuple(fixed["transition_levels"])
    template_min = template_levels.index(parameters["authority"]["template"])
    transition_min = transition_levels.index(parameters["authority"]["transition"])

    blocking = tuple(
        row
        for row in eligible
        if _level(row, "transition", transition_levels)
        == transition_levels.index("BLOCKING_NEED")
    )
    result = _resolve_frontier(state, catalog, blocking, "BLOCKING_SURVIVAL")
    if result:
        return result

    template = tuple(
        row
        for row in eligible
        if _level(row, "template", template_levels) >= template_min
    )
    if template:
        best = max(_template_key(row, template_levels) for row in template)
        result = _resolve_frontier(
            state,
            catalog,
            tuple(
                row
                for row in template
                if _template_key(row, template_levels) == best
            ),
            "TEMPLATE_PROGRESS",
        )
        if result:
            return result

    direct = tuple(
        row
        for row in eligible
        if _field(row, "expert").get("level") == "DIRECT"
    )
    transition = tuple(
        row
        for row in eligible
        if _level(row, "transition", transition_levels) >= transition_min
    )
    if transition:
        best = max(
            _level(row, "transition", transition_levels) for row in transition
        )
        frontier_ids = {
            int(row["choice_id"])
            for row in transition
            if _level(row, "transition", transition_levels) == best
        }
        result = _resolve_frontier(
            state,
            catalog,
            tuple(
                row
                for row in eligible
                if int(row["choice_id"]) in frontier_ids
            ),
            "TRANSITION_NEED",
        )
        if result:
            return result

    result = _resolve_frontier(state, catalog, direct, "EXPERT_EXPERIENCE")
    if result:
        return result

    positive = tuple(
        row
        for row in eligible
        if _field(row, "template").get("level") != "NONE"
        or _field(row, "transition").get("level") != "NONE"
        or _field(row, "expert").get("level") in {"POSITIVE", "DIRECT"}
    )
    if not positive:
        if state.reward.get("must_pick"):
            return _advice(eligible, "COMMITTED_REWARD")
        return _non_card(
            state,
            len(candidates),
            "NO_POSITIVE_EVIDENCE",
            ("NO_POSITIVE_EVIDENCE",),
        )
    return _advice(positive, "POSITIVE_CONFLICT")


def _resolve_frontier(
    state: DecisionState,
    catalog: Mapping[str, Any],
    frontier: Sequence[Mapping[str, Any]],
    policy: str,
) -> dict[str, Any] | None:
    if not frontier:
        return None
    chosen, ranking = _rank(state, catalog, frontier)
    if chosen is None:
        return _advice(frontier, f"{policy}_CONFLICT", ranking)
    return _pick(chosen, frontier, policy, ranking)


def _rank(
    state: DecisionState,
    catalog: Mapping[str, Any],
    frontier: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if len(frontier) == 1:
        return frontier[0], None
    expert = load_policy(_character(state))["parameters"]["expert"]
    table = PreferenceTable(
        _field(_field(catalog, "derived"), "expert_preferences"),
        context_order=expert["context_order"],
        deck_size_limits=expert["deck_size_limits"],
    )
    scores = {int(row["choice_id"]): 0 for row in frontier}
    defeated = {int(row["choice_id"]): set() for row in frontier}
    comparisons = []
    for index, left_row in enumerate(frontier):
        for right_row in frontier[index + 1 :]:
            comparison = table.compare(
                state.run.get("act"), left_row["name"], right_row["name"],
                owned=state.deck.get("counts") or {},
                deck_size=int(state.deck.get("physical_size") or 0),
            )
            left_name = normalize_card(left_row["name"])
            if comparison["left"] == left_name:
                left_wins, right_wins = (
                    comparison["left_wins"],
                    comparison["right_wins"],
                )
            else:
                left_wins, right_wins = (
                    comparison["right_wins"],
                    comparison["left_wins"],
                )
            total = left_wins + right_wins
            z = abs(left_wins - right_wins) / math.sqrt(total) if total else 0.0
            winner = None
            if total and z >= float(expert["direct_z"]) and left_wins != right_wins:
                winner_row, loser_row = (
                    (left_row, right_row)
                    if left_wins > right_wins
                    else (right_row, left_row)
                )
                winner = int(winner_row["choice_id"])
                loser = int(loser_row["choice_id"])
                scores[winner] += 1
                scores[loser] -= 1
                defeated[winner].add(loser)
            comparisons.append(
                {
                    **comparison,
                    "z": round(z, 10),
                    "winner_choice_id": winner,
                }
            )
    winners = [
        row
        for row in frontier
        if len(defeated[int(row["choice_id"])]) == len(frontier) - 1
    ]
    return (winners[0] if len(winners) == 1 else None), {
        "scores": {str(key): value for key, value in sorted(scores.items())},
        "comparisons": comparisons,
    }


def _template_key(row: Mapping[str, Any], levels: Sequence[str]) -> tuple[Any, ...]:
    value = _field(row, "template")
    return (
        levels.index(str(value.get("level") or "NONE")),
        int(value.get("completed_core_gain") or 0),
        int(value.get("anchor_reduction") or 0),
        int(value.get("missing_card_reduction") or 0),
        float(value.get("completion_probability") or 0.0),
    )


def _level(row: Mapping[str, Any], field: str, levels: Sequence[str]) -> int:
    value = str(_field(row, field).get("level") or "NONE")
    if value not in levels:
        raise ResolverError(f"unknown {field} level {value!r}")
    return levels.index(value)


def _pick(
    chosen: Mapping[str, Any],
    frontier: Sequence[Mapping[str, Any]],
    policy: str,
    ranking: Mapping[str, Any] | None,
) -> dict[str, Any]:
    choice_id = int(chosen["choice_id"])
    return _result(
        outcome="PICK",
        policy=policy,
        proposed={"kind": "PICK", "choice_id": choice_id},
        allowed=(choice_id,),
        frontier=frontier,
        allow_skip=False,
        ranking=ranking,
    )


def _advice(
    frontier: Sequence[Mapping[str, Any]],
    policy: str,
    ranking: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _result(
        outcome="ADVICE_REQUIRED",
        policy=policy,
        proposed=None,
        allowed=[int(row["choice_id"]) for row in frontier],
        frontier=frontier,
        allow_skip=False,
        ranking=ranking,
    )


def _non_card(
    state: DecisionState, count: int, policy: str, reasons: Sequence[str]
) -> dict[str, Any]:
    action = _alternative(state, count)
    return _result(
        outcome=action["kind"],
        policy=policy,
        proposed=action,
        allowed=(
            (int(action["choice_id"]),)
            if action["kind"] == "SINGING_BOWL"
            else ()
        ),
        frontier=(),
        allow_skip=action["kind"] == "SKIP",
        reasons=reasons,
    )


def _result(
    *,
    outcome: str,
    policy: str,
    proposed: Mapping[str, Any] | None,
    allowed: Sequence[int],
    frontier: Sequence[Mapping[str, Any]],
    allow_skip: bool,
    ranking: Mapping[str, Any] | None = None,
    alternative: Mapping[str, Any] | None = None,
    reasons: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "outcome": outcome,
        "policy": policy,
        "proposed_action": proposed,
        "allowed_choice_ids": list(allowed),
        "allow_skip": allow_skip,
        "frontier_choice_ids": [int(row["choice_id"]) for row in frontier],
        "alternative": alternative,
        "card_preference": ranking,
        "reason_codes": list(reasons or (policy,)),
    }


def _alternative(state: DecisionState, count: int) -> dict[str, Any]:
    if state.reward.get("singing_bowl"):
        return {"kind": "SINGING_BOWL", "choice_id": count}
    return {"kind": "SKIP"}


def _field(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise ResolverError(f"{name} must be an object")
    return result


def _rows(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))


def _character(state: DecisionState) -> str:
    return normalize_character(state.run.get("character"))


__all__ = ["ResolverError", "resolve"]
