"""Shared approval and shop helpers for the Winning Path card picker."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from spire_agent.contracts import DecisionRequest, GameState, ScreenState


CARD_REWARD_RESULT_KEY = "card_reward_policy_result"
CARD_CHOICE_REVIEW_KEY = "card_choice_review"
WINNING_PATH_REVIEW_KEY = "winning_path_review"
SHOP_CARD_POLICY_KEY = "shop_card_policy"


class CardRewardError(RuntimeError):
    pass


def review_shop(
    request: DecisionRequest,
    *,
    reviewer: Callable[[DecisionRequest], Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the card-reward policy to shop card choices."""

    offered = _shop_cards(request)
    if not offered:
        return {
            "owner": "CardRewardPolicy",
            "card_choices": [],
            "allowed_card_choice_ids": [],
            "policy_result": None,
        }
    state = request.state
    shop = {
        "gold": _number(state.facts.get("gold")),
        "purge_cost": _number(state.screen.details.get("purge_cost")),
        "purge_choice_id": _choice_id(state.screen.choices, "purge"),
    }
    card_state = GameState(
        state.owner_hint,
        state.scope_id,
        ScreenState(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=tuple(row["name"] for row in offered),
            details={
                "cards": tuple({"name": row["name"]} for row in offered),
            },
        ),
        facts=state.facts,
    )
    result = reviewer(
        DecisionRequest(
            card_state,
            request.scope,
            None,
            request.shared,
            request.previous,
        )
    )
    strong = result.get("mode") == "DIRECT"
    allowed = {int(value) for value in result.get("allowed_choice_ids") or ()}
    command = str(result.get("command") or "")
    if strong and command.startswith("choose "):
        allowed.add(int(command.split()[1]))
    return {
        "owner": "CardRewardPolicy",
        "card_choices": offered,
        "allowed_card_choice_ids": [
            int(row["choice_id"])
            for index, row in enumerate(offered)
            if index in allowed
        ],
        "policy_result": result,
        **shop,
    }


def shop_prompt_context(result: Mapping[str, Any]) -> dict[str, Any]:
    policy = result.get("policy_result")
    policy = policy if isinstance(policy, Mapping) else {}
    allowed = {int(value) for value in result.get("allowed_card_choice_ids") or ()}
    return {
        "owner": "CardRewardPolicy",
        "policy": policy.get("policy"),
        "reason": policy.get("reason"),
        "card_choices": [
            {
                "choice_id": int(row["choice_id"]),
                "name": row["name"],
                "purchase_allowed": int(row["choice_id"]) in allowed,
            }
            for row in result.get("card_choices") or ()
            if isinstance(row, Mapping)
        ],
        "instruction": (
            "Never buy a card with purchase_allowed=false. This restriction "
            "does not apply to removal, relics, potions, or leaving the shop."
        ),
    }


def approve_shop(
    result: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    """Approve a shop action or replace a forbidden card purchase with Leave."""

    action, choice_id = str(proposal.get("action") or ""), proposal.get("choice_id")
    allowed = {int(value) for value in result.get("allowed_card_choice_ids") or ()}
    if (
        action == "choose"
        and choice_id == result.get("purge_choice_id")
        and len(allowed) == 1
        and _basic_purge(proposal)
    ):
        row = next(
            (
                item
                for item in result.get("card_choices") or ()
                if isinstance(item, Mapping)
                and int(item["choice_id"]) in allowed
            ),
            {},
        )
        gold, price = _number(result.get("gold")), _number(row.get("price"))
        if gold >= price > gold - _number(result.get("purge_cost")):
            fallback = next(iter(allowed))
            return {
                "data": {
                    "action": "choose",
                    "choice_id": fallback,
                    "targets": [],
                    "reason": "Buy the unique approved card before it becomes unaffordable.",
                },
                "approved": False,
                "veto_reason": "basic removal would forfeit the unique approved card",
            }
    card_ids = {
        int(row["choice_id"])
        for row in result.get("card_choices") or ()
        if isinstance(row, Mapping)
    }
    if action != "choose" or choice_id not in card_ids or choice_id in allowed:
        return {"data": dict(proposal), "approved": True, "veto_reason": None}
    fallback = next(iter(allowed)) if len(allowed) == 1 else None
    return {
        "data": {
            "action": "choose" if fallback is not None else "leave",
            "choice_id": fallback,
            "targets": [],
            "reason": (
                "CardRewardPolicy selected its unique approved shop card."
                if fallback is not None
                else "CardRewardPolicy vetoed an unapproved shop card purchase."
            ),
        },
        "approved": False,
        "veto_reason": (
            f"shop card choice_id={choice_id!r} is outside allowed card choice "
            f"ids {sorted(allowed)}"
        ),
    }


def _shop_cards(request: DecisionRequest) -> list[dict[str, object]]:
    details = request.state.screen.details.get("cards")
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return []
    cards = {
        _shop_name(row.get("name") or row.get("id")): row
        for row in details
        if isinstance(row, Mapping) and (row.get("name") or row.get("id"))
    }
    result = []
    for choice_id, raw in enumerate(request.state.screen.choices):
        label = (
            raw.get("name") or raw.get("value") or raw.get("text")
            if isinstance(raw, Mapping)
            else raw
        )
        card = cards.get(_shop_name(label))
        if card:
            result.append({
                "choice_id": choice_id,
                "name": str(card.get("name") or card.get("id") or "").strip(),
                "price": _number(card.get("price")),
            })
    return result


def _shop_name(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _choice_id(choices: Sequence[object], marker: str) -> int | None:
    return next(
        (
            index
            for index, choice in enumerate(choices)
            if marker in _shop_name(
                choice.get("name") or choice.get("value") or choice.get("text")
                if isinstance(choice, Mapping)
                else choice
            )
        ),
        None,
    )


def _basic_purge(proposal: Mapping[str, Any]) -> bool:
    return any(
        str(target).strip().removesuffix("+") in {"Strike", "Defend"}
        for target in proposal.get("targets") or ()
    )


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def approve(
    result: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    """Approve a bounded LLM proposal."""

    if result.get("mode") != "ADVICE_REQUIRED":
        raise CardRewardError("LLM approval requested for a direct policy result")
    action, choice_id = str(proposal.get("action") or ""), proposal.get("choice_id")
    allowed = {int(value) for value in result.get("allowed_choice_ids") or ()}
    valid_skip = (
        bool(result.get("allow_skip", True))
        and action == "skip"
        and choice_id is None
    )
    valid_pick = (
        action == "choose"
        and isinstance(choice_id, int)
        and not isinstance(choice_id, bool)
        and choice_id in allowed
    )
    if valid_skip or valid_pick:
        return {
            "data": dict(proposal),
            "approved": True,
            "veto_reason": None,
            "selection_kind": "SKIP" if valid_skip else str(result["policy"]),
        }
    raise CardRewardError(
        f"proposal action={action!r} choice_id={choice_id!r} is outside "
        f"allowed choice ids {sorted(allowed)}"
    )


__all__ = [
    "CARD_CHOICE_REVIEW_KEY",
    "CARD_REWARD_RESULT_KEY",
    "SHOP_CARD_POLICY_KEY",
    "WINNING_PATH_REVIEW_KEY",
    "CardRewardError",
    "approve",
    "approve_shop",
    "review_shop",
    "shop_prompt_context",
]
