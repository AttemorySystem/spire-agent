"""BuildAgent prompt and compact state projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import resources
import json
import re
import tomllib
from typing import Any

from spire_agent.contracts import DecisionRequest, GameState
from spire_agent.subagents.build_context import BUILD_CONVERSATION_KEY, room_messages
from spire_agent.tools.build_flow import BUILD_ACTION_SCHEMA, BuildError, reward_type
from spire_agent.tools.events import event_rule
from spire_agent.subagents.llm import LLMMessage, LLMRequest, PromptLanguage
from spire_agent.tools.sts_db import StsDB


_SCENES = {
    "BOSS_REWARD": "boss_reward",
    "CARD_REWARD": "card_reward",
    "CHEST": "chest",
    "COMBAT_REWARD": "rewards",
    "EVENT": "event",
    "GRID": "selector",
    "HAND_SELECT": "selector",
    "REST": "rest",
    "SHOP_SCREEN": "shop",
}
_RUN_KEYS = (
    "class",
    "ascension_level",
    "act",
    "floor",
    "current_hp",
    "max_hp",
    "gold",
    "act_boss",
    "room_type",
    "has_ruby_key",
    "has_sapphire_key",
    "has_emerald_key",
    "keys",
)


def build_prompt(
    request: DecisionRequest,
    language: PromptLanguage | str,
    *,
    policy_context: Mapping[str, Any] | None = None,
    choice_policy: Mapping[str, Any] | None = None,
) -> LLMRequest:
    parsed = PromptLanguage.parse(language)
    scene = _scene(request.state)
    prompts = _prompt_file(parsed)
    context = {
        "scene": scene,
        "run": {
            key: _jsonable(request.state.facts[key])
            for key in _RUN_KEYS
            if key in request.state.facts
        },
        "assets": {
            "deck": _deck_summary(request.state.facts.get("deck")),
            "relics": _entity_names(request.state.facts.get("relics")),
            "potions": _potion_slots(request.state.facts.get("potions")),
        },
        "screen": {
            "type": request.state.screen.type,
            "available_actions": [
                action
                for action in request.state.screen.commands
                if action in BUILD_ACTION_SCHEMA["properties"]["action"]["enum"]
            ],
            "choices": _jsonable(request.state.screen.choices),
            "details": _jsonable(request.state.screen.details),
            "current_action": request.state.screen.current_action,
        },
        "shared": _jsonable(
            {
                key: value
                for key, value in request.shared.items()
                if key != BUILD_CONVERSATION_KEY
            }
        ),
    }
    entity_facts = _entity_facts(request.state, StsDB())
    if entity_facts:
        context["entity_facts"] = entity_facts
    if policy_context is not None:
        context["card_reward_policy"] = _jsonable(policy_context)
    if choice_policy is not None:
        context["choice_policy"] = _jsonable(choice_policy)
    rule_key = event_rule(request.state)
    if rule_key is not None:
        rules = prompts.get("event_rules")
        guidance = rules.get(rule_key) if isinstance(rules, Mapping) else None
        if not isinstance(guidance, str) or not guidance.strip():
            raise BuildError(f"Build event rule {rule_key!r} is missing")
        context["event_rule"] = {"key": rule_key, "prompt": guidance.strip()}

    prior = room_messages(request.shared, request.scope.id)
    messages = tuple(
        LLMMessage(item["role"], item["content"])
        for item in prior
    ) or (LLMMessage("system", _prompt_value(prompts, "common")),)
    user = (
        f"# CURRENT TASK\n{_prompt_value(prompts, scene)}\n\n"
        f"# CURRENT STATE\n{json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True)}"
    )
    return LLMRequest(
        f"build.{scene}",
        messages + (LLMMessage("user", user),),
        BUILD_ACTION_SCHEMA,
    )


def _scene(state: GameState) -> str:
    if state.screen.type == "COMBAT_REWARD" and any(
        reward_type(value) == "SAPPHIRE_KEY"
        for value in _sequence(state.screen.details.get("rewards"))
    ):
        return "chest"
    try:
        return _SCENES[state.screen.type]
    except KeyError as error:
        raise BuildError(f"unsupported Build screen {state.screen.type}") from error


@lru_cache(maxsize=2)
def _prompt_file(language: PromptLanguage) -> Mapping[str, object]:
    text = (
        resources.files("spire_agent.subagents")
        .joinpath("prompts", "build", f"{language.value}.toml")
        .read_text(encoding="utf-8")
    )
    return tomllib.loads(text)


def _prompt_value(prompts: Mapping[str, object], key: str) -> str:
    section = prompts.get(key)
    value = section.get("prompt") if isinstance(section, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise BuildError(f"Build prompt section {key!r} is missing")
    return value.strip()


def _deck_summary(value: object) -> list[dict[str, object]]:
    cards: dict[str, dict[str, object]] = {}
    for item in _sequence(value):
        if isinstance(item, Mapping):
            raw_name = item.get("name") or item.get("id")
            explicit_upgrade = item.get("upgrades", item.get("upgrade", 0))
        else:
            raw_name, explicit_upgrade = item, 0
        name, suffix_upgrade = _card_name(raw_name)
        if not name:
            continue
        try:
            upgrade = max(0, int(explicit_upgrade or 0), suffix_upgrade)
        except (TypeError, ValueError):
            upgrade = suffix_upgrade
        row = cards.setdefault(name, {"name": name, "count": 0, "upgrades": 0})
        row["count"] = int(row["count"]) + 1
        row["upgrades"] = int(row["upgrades"]) + int(upgrade > 0)
    return list(cards.values())


def _entity_facts(state: GameState, db: StsDB) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    explicit = {
        "cards": [row["name"] for row in _deck_summary(state.facts.get("deck"))],
        "relics": _entity_names(state.facts.get("relics")),
        "potions": [name for name in _potion_slots(state.facts.get("potions")) if name],
    }
    queries = {"cards": db.card, "relics": db.relic, "potions": db.potion}
    for kind, names in explicit.items():
        facts = [fact for name in names if (fact := queries[kind](name)) is not None]
        if facts:
            result[kind] = facts

    details = json.dumps(_jsonable(state.screen.details), ensure_ascii=False)
    choices = [_choice_label(choice) for choice in state.screen.choices]
    for kind, facts in db.mentions(details, *choices).items():
        existing = {
            str(row.get("name") or "").casefold() for row in result.get(kind, ())
        }
        result.setdefault(kind, []).extend(
            row
            for row in facts
            if str(row.get("name") or "").casefold() not in existing
        )
    return result


def _entity_names(value: object) -> list[str]:
    names = []
    for item in _sequence(value):
        name = item.get("name") or item.get("id") if isinstance(item, Mapping) else item
        if name not in (None, ""):
            names.append(str(name))
    return names


def _potion_slots(value: object) -> list[str | None]:
    slots = []
    for item in _sequence(value):
        if isinstance(item, Mapping):
            name = item.get("name") or item.get("potion") or item.get("id")
        else:
            name = item
        normalized = _normalize_name(name)
        slots.append(
            None
            if normalized in {"", "potion slot", "empty", "empty slot"}
            else str(name)
        )
    return slots


def _card_name(value: object) -> tuple[str, int]:
    rendered = str(value or "").strip()
    match = re.search(r"\+(\d*)$", rendered)
    if match is None:
        return rendered, 0
    return rendered[: match.start()].strip(), int(match.group(1) or 1)


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


__all__ = ["build_prompt"]
