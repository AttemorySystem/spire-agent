"""Deterministic constraints for Boss relic choices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from spire_agent.contracts import DecisionRequest


_HEALING_CARDS = frozenset({"bandage up", "bite", "reaper", "self repair"})
_HEALING_RELICS = frozenset(
    {
        "bird-faced urn",
        "black blood",
        "blood vial",
        "bloody idol",
        "burning blood",
        "eternal feather",
        "meal ticket",
        "meat on the bone",
        "pantograph",
        "toy ornithopter",
    }
)


def boss_relic_policy(request: DecisionRequest) -> dict[str, object] | None:
    """Forbid Coffee Dripper when the run has no future healing source."""

    state = request.state
    if state.screen.type != "BOSS_REWARD":
        return None
    choices = tuple(_name(choice) for choice in state.screen.choices)
    coffee = next(
        (index for index, name in enumerate(choices) if name == "coffee dripper"),
        None,
    )
    if coffee is None:
        return None
    sources = sorted(
        {
            name
            for name in _names(state.facts.get("deck"))
            if name in _HEALING_CARDS
        }
        | {
            name
            for name in _names(state.facts.get("relics"))
            if name in _HEALING_RELICS
        }
    )
    legal = tuple(range(len(choices)))
    if not sources:
        legal = tuple(index for index in legal if index != coffee)
    return {
        "legal_choice_ids": legal,
        "coffee_dripper_choice_id": coffee,
        "recurring_healing_sources": sources,
        "reason": (
            "Coffee Dripper is legal because the run already has future healing"
            if sources
            else (
                "Coffee Dripper is forbidden without an existing future "
                "healing source"
            )
        ),
    }


def _names(value: object) -> set[str]:
    return {
        name
        for item in _sequence(value)
        if (name := _name(item))
    }


def _name(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("id")
    return " ".join(str(value or "").strip().removesuffix("+").casefold().split())


def _sequence(value: object) -> Sequence[object]:
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


__all__ = ["boss_relic_policy"]
