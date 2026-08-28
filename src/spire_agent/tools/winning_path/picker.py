"""Live implementation of the CardPicker contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spire_agent.contracts import DecisionRequest
from spire_agent.subagents.build import CardPicker
from spire_agent.subagents.build_context import RUN_CONSTRUCTION_KEY

from . import card_policy
from .catalog import load_default_catalog
from .analysis import analyze
from .protocol import PROTOCOL_VERSION, normalize_character


class WinningPathCardPicker(CardPicker):
    __slots__ = ("character",)

    def __init__(self, character: object = "IRONCLAD") -> None:
        self.character = normalize_character(character)

    def review(self, request: DecisionRequest) -> dict[str, Any]:
        actual = normalize_character(request.state.facts.get("class"))
        if actual != self.character:
            raise ValueError(
                f"{self.character} card picker received character {actual!r}"
            )
        return _result(self.character, analyze(request))

    def approve(
        self, result: Mapping[str, Any], proposal: Mapping[str, Any]
    ) -> dict[str, Any]:
        return card_policy.approve(result, proposal)

    def prompt_context(self, result: Mapping[str, Any]) -> dict[str, Any]:
        analysis = _mapping(result.get("winning_path"))
        candidates = {
            int(row["choice_id"]): row
            for row in result.get("candidates") or ()
            if isinstance(row, Mapping)
        }
        allowed = [int(value) for value in result.get("allowed_choice_ids") or ()]
        return {
            "owner": result["owner"],
            "policy": result["policy"],
            "allowed_actions": [
                *(("skip",) if result.get("allow_skip") else ()),
                *(f"choose {choice_id}" for choice_id in allowed),
            ],
            "state": analysis.get("state"),
            "deck_plan": analysis.get("deck_plan"),
            "target_plan": analysis.get("target_plan"),
            "need_profile": analysis.get("need_profile"),
            "candidates": [
                dict(candidates[choice_id])
                for choice_id in allowed
            ],
        }

    def decision_payload(
        self,
        result: Mapping[str, Any],
        *,
        command: str,
        approval: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        analysis = _mapping(result.get("winning_path"))
        choice_id = _choice_id(command)
        selected = next(
            (
                str(row.get("name") or "")
                for row in result.get("candidates") or ()
                if isinstance(row, Mapping) and row.get("choice_id") == choice_id
            ),
            (
                "Singing Bowl"
                if result.get("bowl_choice_id") is not None
                and choice_id == result.get("bowl_choice_id")
                else None
            ),
        )
        return {
            RUN_CONSTRUCTION_KEY: _construction_update(analysis, command),
            card_policy.CARD_CHOICE_REVIEW_KEY: {
                "character": self.character,
                "picker_id": f"{self.character.casefold()}.winning_path",
                "policy_version": PROTOCOL_VERSION,
                "state": _record_state(analysis),
                "candidates": list(result.get("candidates") or ()),
                "allowed_choice_ids": list(
                    result.get("allowed_choice_ids") or ()
                ),
                "fingerprints": dict(_mapping(analysis.get("fingerprints"))),
            },
            card_policy.CARD_REWARD_RESULT_KEY: {
                "policy": result.get("policy"),
                "selection_kind": (
                    approval.get("selection_kind")
                    if approval is not None
                    else result.get("policy")
                ),
                "act": _mapping(_mapping(analysis.get("state")).get("run")).get(
                    "act"
                ),
                "command": command,
                "choice_id": choice_id,
                "card": selected,
                "llm_approved": approval.get("approved") if approval else None,
                "veto_reason": approval.get("veto_reason") if approval else None,
            },
            card_policy.WINNING_PATH_REVIEW_KEY: dict(result),
        }

    def review_shop(self, request: DecisionRequest) -> dict[str, Any]:
        return card_policy.review_shop(request, reviewer=self.review)

    def approve_shop(
        self, result: Mapping[str, Any], proposal: Mapping[str, Any]
    ) -> dict[str, Any]:
        return card_policy.approve_shop(result, proposal)

    def shop_decision_payload(
        self, result: Mapping[str, Any], approval: Mapping[str, Any]
    ) -> dict[str, Any]:
        policy = _mapping(result.get("policy_result"))
        analysis = _mapping(policy.get("winning_path"))
        if not analysis:
            return {}
        selected = _mapping(approval.get("data")).get("choice_id")
        local = next(
            (
                index
                for index, row in enumerate(result.get("card_choices") or ())
                if isinstance(row, Mapping) and row.get("choice_id") == selected
            ),
            None,
        )
        return {
            RUN_CONSTRUCTION_KEY: _construction_update(
                analysis, f"choose {local}" if local is not None else "skip"
            )
        }

    def shop_prompt_context(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return card_policy.shop_prompt_context(result)


def review_card_reward(
    request: DecisionRequest,
    *,
    preference_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the same card-reward policy used by the live picker."""

    character = normalize_character(request.state.facts.get("class"))
    catalog: Mapping[str, Any] | None = None
    if preference_payload is not None:
        current = load_default_catalog(character)
        catalog = {
            **current,
            "derived": {
                **_mapping(current.get("derived")),
                "expert_preferences": preference_payload,
            },
        }
    return _result(character, analyze(request, catalog))


def _result(character: str, analysis: Mapping[str, Any]) -> dict[str, Any]:
    resolution = _mapping(analysis["resolution"])
    candidates = list(
        _mapping(analysis["candidate_evidence"]).get("candidates") or ()
    )
    command = _command(resolution.get("proposed_action"))
    return {
        "owner": "CardRewardPolicy",
        "mode": "DIRECT" if command is not None else "ADVICE_REQUIRED",
        "policy": resolution["policy"],
        "command": command,
        "reason": ", ".join(map(str, resolution.get("reason_codes") or ())),
        "allowed_choice_ids": list(resolution.get("allowed_choice_ids") or ()),
        "allow_skip": bool(resolution.get("allow_skip")),
        "bowl_choice_id": _bowl_choice_id(resolution),
        "candidates": candidates,
        "winning_path": analysis,
    }


def create_card_picker(character: object) -> CardPicker:
    return WinningPathCardPicker(character)


def _command(action: object) -> str | None:
    action = _mapping(action)
    kind = action.get("kind")
    if kind == "SKIP":
        return "skip"
    choice_id = action.get("choice_id")
    if kind in {"PICK", "SINGING_BOWL"} and type(choice_id) is int:
        return f"choose {choice_id}"
    return None


def _bowl_choice_id(resolution: Mapping[str, Any]) -> int | None:
    for key in ("proposed_action", "alternative"):
        action = _mapping(resolution.get(key))
        if (
            action.get("kind") == "SINGING_BOWL"
            and type(action.get("choice_id")) is int
        ):
            return int(action["choice_id"])
    return None


def _choice_id(command: str) -> int | None:
    parts = command.split()
    return int(parts[1]) if len(parts) == 2 and parts[0] == "choose" else None


def _construction_update(
    analysis: Mapping[str, Any], command: str
) -> dict[str, Any]:
    plan = _mapping(analysis.get("deck_plan"))
    targets = _mapping(analysis.get("target_plan"))
    needs = _mapping(analysis.get("need_profile"))
    candidates = {
        row.get("choice_id"): row
        for row in _mapping(analysis.get("candidate_evidence")).get(
            "candidates", ()
        )
        if isinstance(row, Mapping)
    }
    selected = _mapping(candidates.get(_choice_id(command)))
    return {
        "schema_version": 1,
        "basis": "PRE_ACTION_WITH_CONFIRMED_SELECTION",
        "modules": {
            "active": list(plan.get("active_modules") or ()),
            "committed": list(plan.get("committed_modules") or ()),
        },
        "targets": list(targets.get("targets") or ()),
        "capabilities": list(needs.get("current_capabilities") or ()),
        "deficits": [
            row.get("type")
            for row in needs.get("needs") or ()
            if isinstance(row, Mapping) and row.get("status") != "SATISFIED"
        ],
        "confirmed_selection": {
            "command": command,
            "card": selected.get("name"),
            "template": selected.get("template"),
            "transition": selected.get("transition"),
        },
    }


def _record_state(analysis: Mapping[str, Any]) -> dict[str, Any]:
    state = _mapping(analysis.get("state"))
    run = _mapping(state.get("run"))
    deck = _mapping(state.get("deck"))
    assets = _mapping(state.get("assets"))
    plan = _mapping(analysis.get("deck_plan"))
    return {
        "act": run.get("act"),
        "floor": run.get("floor"),
        "act_boss": run.get("boss"),
        "current_hp": run.get("hp"),
        "max_hp": run.get("max_hp"),
        "gold": run.get("gold"),
        "ascension_level": run.get("ascension"),
        "room_type": run.get("room_type"),
        "deck": dict(_mapping(deck.get("counts"))),
        "upgrades": dict(_mapping(deck.get("upgrade_counts"))),
        "relics": [
            row.get("id")
            for row in assets.get("relics") or ()
            if isinstance(row, Mapping)
        ],
        "active_modules": list(plan.get("active_modules") or ()),
        "committed_modules": list(plan.get("committed_modules") or ()),
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "WinningPathCardPicker",
    "create_card_picker",
    "review_card_reward",
]
