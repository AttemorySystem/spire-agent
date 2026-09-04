"""Deterministic Build command and continuation mechanics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Any

from spire_agent.contracts import (
    AgentKind,
    Continuation,
    ContinuationChange,
    Decision,
    DecisionRequest,
    GameState,
)
from spire_agent.tools.boss_relics import boss_relic_policy
from spire_agent.tools.events import event_choice_policy, event_rule, forced_event_choice
from spire_agent.tools.run_keys import key_view, rest_policy, reward_key
from spire_agent.subagents.build_context import BUILD_EXCHANGE_KEY
from spire_agent.subagents.llm import LLMRequest


class BuildError(RuntimeError):
    pass


_KIND = "build_flow"
_SELECT_SCREENS = ("GRID", "HAND_SELECT")
_REWARD_SCREENS = ("COMBAT_REWARD", "CARD_REWARD")
_ACTIONS = frozenset({"choose", "skip", "proceed", "leave", "confirm"})
BUILD_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "choice_id": {"type": ["integer", "null"], "minimum": 0},
        "targets": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action", "choice_id", "targets", "reason"],
    "additionalProperties": False,
}


def build_choice_policy(
    request: DecisionRequest,
    encounter_evaluator: Callable[
        [GameState, Mapping[str, Mapping[str, float]]], Mapping[str, object]
    ] | None = None,
) -> dict[str, object] | None:
    """Combine independent non-card choice constraints for BuildAgent."""

    return (
        event_choice_policy(request.state, request.shared, encounter_evaluator)
        or boss_relic_policy(request)
        or rest_policy(request)
    )


_SELECT_MARKERS = (
    ("purge", "remove"),
    ("toke", "remove"),
    ("empty cage", "remove"),
    ("smith", "upgrade"),
    ("astrolabe", "transform"),
    ("dolly's mirror", "duplicate"),
)
_MANUAL_CARD_EFFECT = re.compile(
    r"\b(remove|transform|duplicate|upgrade)\b"
    r"(?:(?!\b(?:all|random)\b)[^.;])*?\bcards?\b"
)
_UNREMOVABLE_CARDS = frozenset(
    {"ascender's bane", "curse of the bell", "necronomicurse"}
)


def continue_build(request: DecisionRequest) -> Decision | None:
    """Execute an active Build continuation without another model call."""

    continuation = request.continuation
    if continuation is None or continuation.kind != _KIND:
        return None
    flow = str(continuation.data.get("flow") or "")
    if flow == "selection":
        return _continue_selection(request)
    if flow == "rewards":
        state = request.state
        skipped = _nonnegative_int(continuation.data.get("skipped_card_rows"))
        if state.screen.type == "COMBAT_REWARD":
            return _reward_decision(request, skipped)
        if state.screen.type == "CARD_REWARD":
            if continuation.data.get("retry_skip") and "skip" in state.screen.commands:
                data = _reward_data(skipped, retry_skip=False)
                return Decision(
                    "skip",
                    "build.reward_retry",
                    "retry the confirmed card reward skip",
                    continuation=ContinuationChange.set(_flow(request, data, _REWARD_SCREENS)),
                )
            return None
        raise BuildError(
            f"reward continuation reached unexpected screen {state.screen.type}"
        )
    if flow == "shop_exit":
        return _continue_shop_exit(request)
    raise BuildError(f"unknown Build continuation flow {flow!r}")


def fast_decision(request: DecisionRequest) -> Decision | None:
    """Return only decisions that require no strategic judgment."""

    state = request.state
    if request.scope.owner is not AgentKind.BUILD:
        return None
    fruit_juice = _potion_slot(state, "fruit juice")
    if fruit_juice is not None and "potion" in state.screen.commands:
        return Decision(
            f"potion use {fruit_juice}",
            "build.potion",
            "consume Fruit Juice for its permanent max HP",
        )
    key_rule = rest_policy(request)
    if key_rule is not None and key_rule.get("forced_choice_id") is not None:
        choice = int(key_rule["forced_choice_id"])
        acquired = key_rule.get("acquired_key")
        targets = tuple(
            str(value) for value in key_rule.get("selection_targets") or ()
        )
        return Decision(
            f"choose {choice}",
            (
                "build.key_policy" if acquired else
                "build.curse_policy" if targets else
                "build.survival_policy"
            ),
            str(key_rule["reason"]),
            continuation=_llm_continuation(request, "choose", targets),
            payload={
                "choice_id": choice,
                **({"targets": targets} if targets else {}),
                **({"acquired_key": acquired} if acquired else {}),
            },
        )
    if state.screen.type == "EVENT" and "choose" in state.screen.commands:
        choice = forced_event_choice(state, request.shared)
        if choice is not None:
            rule = event_rule(state) or "event"
            return Decision(
                f"choose {choice}",
                "build.event_rule",
                f"apply the reviewed {rule.replace('_', ' ')} constraint",
                payload={"choice_id": choice, "event_rule": rule},
            )
    if state.screen.type == "COMBAT_REWARD":
        return _reward_decision(request, 0)

    commands = set(state.screen.commands)
    choices = state.screen.choices
    if "confirm" in commands and not (
        state.screen.type == "HAND_SELECT" and "choose" in commands and choices
    ):
        return Decision(
            "confirm",
            "build.confirm",
            "the current selection is ready for confirmation",
            continuation=_clear_build(request),
        )
    if (
        state.screen.type == "SHOP_ROOM"
        and "choose" in commands
        and len(choices) == 1
    ):
        return Decision("choose 0", "build.enter_shop", "enter the merchant screen")
    if (
        "choose" in commands
        and len(choices) == 1
        and state.screen.type != "SHOP_SCREEN"
        and not (state.screen.type == "CARD_REWARD" and "skip" in commands)
    ):
        return Decision(
            "choose 0",
            "build.single_choice",
            "only legal choice",
            continuation=_clear_build(request),
        )
    executable = [action for action in ("proceed", "leave", "skip") if action in commands]
    if not choices and len(executable) == 1:
        action = executable[0]
        return Decision(
            action,
            "build.single_action",
            "only executable screen action",
            continuation=(
                _shop_exit_continuation(request)
                if state.screen.type == "SHOP_SCREEN" and action == "leave"
                else _clear_build(request)
            ),
        )
    return None


def policy_decision(
    request: DecisionRequest,
    command: str,
    source: str,
    reason: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> Decision:
    """Translate one domain-policy action through Build UI continuation rules."""

    action = str(command).split(" ", 1)[0]
    if action not in request.state.screen.commands:
        raise BuildError(
            f"policy selected unavailable action {action!r}; "
            f"available actions are {list(request.state.screen.commands)!r}"
        )
    return Decision(
        command,
        source,
        reason,
        continuation=_llm_continuation(request, action, ()),
        payload=payload or {},
    )


def llm_decision(
    request: DecisionRequest,
    response: object,
    prompt: LLMRequest,
    *,
    snapshot: Mapping[str, Any],
    legal_choice_ids: Sequence[int] | None = None,
) -> Decision:
    """Validate model output and translate it to one concrete command."""

    data = getattr(response, "data", None)
    if not isinstance(data, Mapping):
        raise BuildError("LLM response data must be an object")
    action = data.get("action")
    if not isinstance(action, str) or action not in _ACTIONS:
        raise BuildError(f"LLM response action must be one of {sorted(_ACTIONS)}")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise BuildError("LLM response reason must be a non-empty string")
    choice_id = data.get("choice_id")
    targets = data.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise BuildError("LLM response targets must be an array")
    target_names = tuple(str(target).strip() for target in targets)
    if any(not target for target in target_names):
        raise BuildError("LLM response targets must contain non-empty names")

    state = request.state
    if action == "choose":
        if isinstance(choice_id, bool) or not isinstance(choice_id, int):
            raise BuildError("choose requires an integer choice_id")
        legal = (
            tuple(range(len(state.screen.choices)))
            if legal_choice_ids is None
            else tuple(int(value) for value in legal_choice_ids)
        )
        if choice_id not in legal:
            raise BuildError(
                f"LLM selected illegal choice {choice_id}; "
                f"legal ids are {list(legal)}"
            )
        command = f"choose {choice_id}"
    else:
        if choice_id is not None:
            raise BuildError(f"{action} requires choice_id=null")
        command = action
    if action not in state.screen.commands:
        raise BuildError(
            f"LLM selected unavailable action {action!r}; "
            f"available actions are {list(state.screen.commands)!r}"
        )

    expects_selection = (
        action == "choose"
        and int(choice_id) < len(state.screen.choices)
        and _opens_selector(state, int(choice_id))
    )
    if expects_selection and not target_names:
        raise BuildError("the selected action opens a card selector but targets are empty")
    if target_names and not expects_selection:
        raise BuildError("targets are only valid for an action that opens a card selector")
    if expects_selection:
        target_names = _legalize_removal_targets(
            state, int(choice_id), target_names
        )
        _validate_selection_targets(state, int(choice_id), target_names)

    change = _llm_continuation(request, action, target_names)
    metrics = {}
    model = getattr(response, "model", "")
    usage = getattr(response, "usage", {})
    if model or usage:
        metrics = {"model": model, "usage": usage}
    raw_text = str(getattr(response, "raw_text", "") or "")
    assistant = raw_text or json.dumps(
        _jsonable(data),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    acquired = _selected_key(state, action, choice_id)
    user = next(
        message.content
        for message in reversed(prompt.messages)
        if message.role == "user"
        and (
            "# CURRENT STATE\n" in message.content
            or "# CONFIRMED STATE UPDATE\n" in message.content
        )
    )
    return Decision(
        command,
        "build.llm",
        reason.strip(),
        continuation=change,
        payload={
            "scene": prompt.purpose.removeprefix("build."),
            "choice_id": choice_id,
            "targets": target_names,
            **({"acquired_key": acquired} if acquired else {}),
            BUILD_EXCHANGE_KEY: {
                "scope_id": request.scope.id,
                "system": prompt.messages[0].content,
                "user": user,
                "assistant": assistant,
                "snapshot": _jsonable(snapshot),
            },
        },
        metrics=metrics,
    )


def _continue_selection(request: DecisionRequest) -> Decision | None:
    state = request.state
    if state.screen.type not in _SELECT_SCREENS:
        raise BuildError(
            f"selection continuation reached unexpected screen {state.screen.type}"
        )
    continuation = request.continuation
    if continuation is None:
        raise BuildError("selection continuation is missing")
    targets = tuple(str(value) for value in _sequence(continuation.data.get("targets")))
    used = {
        _nonnegative_int(value)
        for value in _sequence(continuation.data.get("used_choice_ids"))
    }
    if not targets:
        if "confirm" not in state.screen.commands:
            raise BuildError("selection is complete but confirm is unavailable")
        return Decision(
            "confirm",
            "build.selection_confirm",
            "confirm the completed Build selection",
            continuation=ContinuationChange.clear(),
        )
    wanted = _normalize_name(targets[0])
    if (
        str(state.screen.details.get("grid_operation") or "").upper() == "UPGRADE"
        and wanted.endswith("+")
    ):
        wanted = wanted.removesuffix("+")
    choices = tuple(
        _normalize_name(_choice_label(choice)) for choice in state.screen.choices
    )
    choice_id = next(
        (
            index for index, choice in enumerate(choices)
            if index not in used and choice == wanted
        ),
        None,
    )
    if choice_id is None and not wanted.endswith("+"):
        choice_id = next(
            (
                index for index, choice in enumerate(choices)
                if index not in used and choice.removesuffix("+") == wanted
            ),
            None,
        )
    if choice_id is None:
        return None
    remaining = targets[1:]
    change = ContinuationChange.clear()
    if remaining:
        change = ContinuationChange.set(
            _flow(
                request,
                {
                    "flow": "selection",
                    "targets": remaining,
                    "used_choice_ids": tuple(sorted(used | {choice_id})),
                },
                _SELECT_SCREENS,
            )
        )
    return Decision(
        f"choose {choice_id}",
        "build.selection",
        f"select planned target {targets[0]}",
        continuation=change,
    )


def _continue_shop_exit(request: DecisionRequest) -> Decision:
    state = request.state
    if state.screen.type != "SHOP_ROOM":
        raise BuildError(
            f"shop exit continuation reached unexpected screen {state.screen.type}"
        )
    action = next(
        (value for value in ("proceed", "leave") if value in state.screen.commands),
        None,
    )
    if action is None:
        raise BuildError("shop exit continuation found no exit action")
    return Decision(
        action,
        "build.shop_exit",
        "continue past the merchant after leaving the shop screen",
        continuation=ContinuationChange.clear(),
    )


def _reward_decision(request: DecisionRequest, skipped_cards: int) -> Decision | None:
    state = request.state
    rewards = _sequence(state.screen.details.get("rewards"))
    if not rewards:
        if "proceed" in state.screen.commands:
            return Decision(
                "proceed",
                "build.rewards_done",
                "no collectible rewards remain",
                continuation=ContinuationChange.clear(),
            )
        return None
    reward_types = tuple(reward_type(value) for value in rewards)
    if "UNKNOWN" in reward_types:
        return None
    keys = key_view(request.shared, state)
    act = _nonnegative_int(state.facts.get("act"))
    for colour in ("emerald", "sapphire"):
        if keys[colour] or (colour == "sapphire" and act < 3):
            continue
        index = next(
            (i for i, reward in enumerate(rewards) if reward_key(reward) == colour),
            None,
        )
        if index is not None:
            data = _reward_data(skipped_cards)
            return Decision(
                f"choose {index}",
                "build.key_policy",
                f"collect the required {colour.title()} Key",
                continuation=ContinuationChange.set(_flow(request, data, _REWARD_SCREENS)),
                payload={
                    "reward_index": index,
                    "reward_type": f"{colour.upper()}_KEY",
                    "acquired_key": colour,
                },
            )
    skipped_seen = 0
    potion_space = _has_potion_slot(state)
    for index, reward_kind in enumerate(reward_types):
        if reward_kind == "CARD":
            if skipped_seen < skipped_cards:
                skipped_seen += 1
                continue
        elif reward_kind == "POTION" and not potion_space:
            continue
        elif reward_kind in {"EMERALD_KEY", "SAPPHIRE_KEY"}:
            continue
        data = _reward_data(skipped_cards)
        return Decision(
            f"choose {index}",
            "build.collect_reward",
            f"collect {reward_kind.lower()} reward",
            continuation=ContinuationChange.set(_flow(request, data, _REWARD_SCREENS)),
            payload={"reward_index": index, "reward_type": reward_kind},
        )
    if "proceed" not in state.screen.commands:
        return None
    return Decision(
        "proceed",
        "build.rewards_done",
        "all remaining rewards were deliberately skipped",
        continuation=ContinuationChange.clear(),
    )


def _llm_continuation(
    request: DecisionRequest,
    action: str,
    targets: tuple[str, ...],
) -> ContinuationChange:
    state = request.state
    active = request.continuation
    skipped = (
        _nonnegative_int(active.data.get("skipped_card_rows"))
        if active is not None and active.kind == _KIND
        else 0
    )
    if state.screen.type == "CARD_REWARD" and active is not None and active.kind == _KIND:
        if action == "skip":
            skipped += 1
        if action in {"choose", "skip"}:
            return ContinuationChange.set(
                _flow(
                    request,
                    _reward_data(skipped, retry_skip=action == "skip"),
                    _REWARD_SCREENS,
                )
            )
    if state.screen.type == "COMBAT_REWARD" and action == "choose":
        return ContinuationChange.set(
            _flow(request, _reward_data(skipped), _REWARD_SCREENS)
        )
    if state.screen.type == "SHOP_SCREEN" and action == "leave":
        return _shop_exit_continuation(request)
    if targets:
        return ContinuationChange.set(
            _flow(
                request,
                {"flow": "selection", "targets": targets, "used_choice_ids": ()},
                _SELECT_SCREENS,
            )
        )
    return _clear_build(request)


def _shop_exit_continuation(request: DecisionRequest) -> ContinuationChange:
    return ContinuationChange.set(
        _flow(request, {"flow": "shop_exit"}, ("SHOP_ROOM",))
    )


def _opens_selector(state: GameState, choice_id: int) -> bool:
    return bool(_selection_kinds(state, choice_id))


def _choice_text(state: GameState, choice_id: int) -> str:
    text = _choice_label(state.screen.choices[choice_id])
    options = _sequence(state.screen.details.get("options"))
    if choice_id < len(options):
        text += " " + _choice_label(options[choice_id])
    return _normalize_name(text)


def _selection_kinds(state: GameState, choice_id: int) -> frozenset[str]:
    text = _choice_text(state, choice_id)
    kinds = set(_MANUAL_CARD_EFFECT.findall(text))
    kinds.update(kind for marker, kind in _SELECT_MARKERS if marker in text)
    return frozenset(kinds)


def _validate_selection_targets(
    state: GameState,
    choice_id: int,
    targets: tuple[str, ...],
) -> None:
    available: dict[str, list[int]] = {}
    for card in _sequence(state.facts.get("deck")):
        if isinstance(card, Mapping):
            raw_name = card.get("name") or card.get("id")
            count = max(1, _nonnegative_int(card.get("count", 1)))
            upgrades = _nonnegative_int(card.get("upgrades", card.get("upgrade", 0)))
        else:
            raw_name, count, upgrades = card, 1, 0
        name = _normalize_name(raw_name).removesuffix("+")
        if not name:
            continue
        upgraded = min(count, upgrades or int(_normalize_name(raw_name).endswith("+")))
        row = available.setdefault(name, [0, 0])
        row[0] += count
        row[1] += count - upgraded

    kinds = _selection_kinds(state, choice_id)
    if "remove" in kinds:
        for name in _UNREMOVABLE_CARDS:
            available.pop(name, None)
    upgrade_only = "upgrade" in kinds and "remove" not in kinds
    requested: dict[str, int] = {}
    for target in targets:
        name = _normalize_name(target).removesuffix("+")
        requested[name] = requested.get(name, 0) + 1
    illegal = [
        target
        for target, count in requested.items()
        if count > available.get(target, [0, 0])[int(upgrade_only)]
    ]
    if illegal:
        kind = "unupgraded " if upgrade_only else ""
        raise BuildError(
            f"selection target has no matching {kind}deck copy: {illegal[0]!r}"
        )


def _legalize_removal_targets(
    state: GameState, choice_id: int, targets: tuple[str, ...]
) -> tuple[str, ...]:
    if "remove" not in _selection_kinds(state, choice_id) or not any(
        _normalize_name(target).removesuffix("+") in _UNREMOVABLE_CARDS
        for target in targets
    ):
        return targets
    names = [
        str(card.get("name") or card.get("id") or "").strip().removesuffix("+")
        for card in _sequence(state.facts.get("deck"))
        if isinstance(card, Mapping)
    ]
    legal = [name for name in names if _normalize_name(name) not in _UNREMOVABLE_CARDS]
    if not legal:
        return targets
    replacement = next(
        (name for preferred in ("strike", "defend") for name in legal
         if _normalize_name(name) == preferred),
        legal[0],
    )
    return tuple(
        replacement
        if _normalize_name(target).removesuffix("+") in _UNREMOVABLE_CARDS
        else target
        for target in targets
    )


def _flow(
    request: DecisionRequest,
    data: Mapping[str, object],
    screens: tuple[str, ...],
) -> Continuation:
    return Continuation(
        AgentKind.BUILD,
        _KIND,
        request.scope.id,
        expected_screens=screens,
        data=data,
    )


def _clear_build(request: DecisionRequest) -> ContinuationChange:
    continuation = request.continuation
    return (
        ContinuationChange.clear()
        if continuation is not None and continuation.kind == _KIND
        else ContinuationChange.keep()
    )


def _reward_data(skipped: int, *, retry_skip: bool = False) -> dict[str, object]:
    return {
        "flow": "rewards",
        "skipped_card_rows": skipped,
        "retry_skip": retry_skip,
    }


def reward_type(value: object) -> str:
    if isinstance(value, Mapping):
        raw = value.get("reward_type") or value.get("type")
    else:
        raw = value
    text = str(raw or "").strip().upper().replace(" ", "_")
    key = reward_key(value)
    if key:
        return f"{key.upper()}_KEY"
    if text in {"GOLD", "STOLEN_GOLD", "POTION", "RELIC", "CARD", "EMERALD_KEY", "SAPPHIRE_KEY"}:
        return text
    return "UNKNOWN"


def _selected_key(state: GameState, action: str, choice_id: object) -> str:
    if action != "choose" or not isinstance(choice_id, int):
        return ""
    if state.screen.type == "REST" and choice_id < len(state.screen.choices):
        return "ruby" if _normalize_name(_choice_label(state.screen.choices[choice_id])) == "recall" else ""
    if state.screen.type == "COMBAT_REWARD":
        rewards = _sequence(state.screen.details.get("rewards"))
        return reward_key(rewards[choice_id]) if choice_id < len(rewards) else ""
    return ""


def _has_potion_slot(state: GameState) -> bool:
    for value in _sequence(state.facts.get("potions")):
        if isinstance(value, Mapping):
            name = value.get("name") or value.get("potion") or value.get("id")
        else:
            name = value
        if _normalize_name(name) in {"potion slot", "empty", "empty slot"}:
            return True
    return False


def _potion_slot(state: GameState, target: str) -> int | None:
    for index, value in enumerate(_sequence(state.facts.get("potions"))):
        if isinstance(value, Mapping) and value.get("can_use") is False:
            continue
        name = (
            value.get("name") or value.get("potion") or value.get("id")
            if isinstance(value, Mapping)
            else value
        )
        if _normalize_name(name) == target:
            slot = value.get("slot", index) if isinstance(value, Mapping) else index
            return (
                slot
                if isinstance(slot, int)
                and not isinstance(slot, bool)
                and 0 <= slot < 5
                else None
            )
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if str(key) not in {"uuid", "replay_rng_state", "map", "bridge"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _choice_label(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("value") or value.get("text") or "")
    return str(value)


def _normalize_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "BUILD_ACTION_SCHEMA",
    "BuildError",
    "build_choice_policy",
    "continue_build",
    "fast_decision",
    "llm_decision",
    "policy_decision",
    "reward_type",
]
