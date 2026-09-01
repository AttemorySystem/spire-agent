"""Drop-in BUILD and COMBAT agents whose every command comes from an LLM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from spire_agent.contracts import (
    AgentKind,
    ContinuationChange,
    Decision,
    DecisionRequest,
    GameState,
)
from spire_agent.subagents import BuildAgent, CombatAgent
from spire_agent.subagents.build_context import (
    BUILD_EXCHANGE_KEY,
    context_delta,
    room_messages,
    room_snapshot,
)
from spire_agent.subagents.combat import CombatTool, create_combat_agent
from spire_agent.subagents.llm import LLMMessage, LLMRequest, PromptLanguage
from spire_agent.tools.sts_db import StsDB


class LLMAgentError(RuntimeError):
    """An LLM agent cannot produce one executable command."""


_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["command", "reason"],
    "additionalProperties": False,
}
_HIDDEN_KEYS = frozenset(
    {
        "bridge",
        "build_conversation",
        "replay_boundary_key",
        "replay_rng_state",
        "uuid",
    }
)
_CONTROL_COMMANDS = frozenset({"click", "key", "state", "wait"})


class LLMAgentTool(CombatTool):
    """Owner-bound BUILD or COMBAT implementation driven only by an LLM."""

    def __init__(
        self,
        owner: AgentKind,
        llm: object,
        language: PromptLanguage | str = PromptLanguage.ENGLISH,
    ) -> None:
        if owner not in {AgentKind.BUILD, AgentKind.COMBAT}:
            raise ValueError("LLM agent tool supports only BUILD or COMBAT")
        self._owner = owner
        self._llm = llm
        self._language = PromptLanguage.parse(language)

    def try_decide(self, request: DecisionRequest) -> Decision | None:
        if request.scope.owner is not self._owner:
            return None
        return _decide(self._llm, request, self._owner, self._language)


def create_llm_build_agent(
    llm: object,
    language: PromptLanguage | str = PromptLanguage.ENGLISH,
) -> BuildAgent:
    """Compose the existing BuildAgent with only the LLM tool stage."""

    return BuildAgent(tool_stages=(LLMAgentTool(AgentKind.BUILD, llm, language),))


def create_llm_combat_agent(
    llm: object,
    language: PromptLanguage | str = PromptLanguage.ENGLISH,
) -> CombatAgent:
    """Compose the existing CombatAgent with the alternate CombatTool."""

    return create_combat_agent(LLMAgentTool(AgentKind.COMBAT, llm, language))


def _decide(
    llm: object,
    request: DecisionRequest,
    owner: AgentKind,
    language: PromptLanguage,
) -> Decision:
    complete = getattr(llm, "complete", None)
    if not callable(complete):
        raise TypeError("LLM agent has no complete() method")
    prompt, snapshot = _prompt(request, owner, language)
    response = complete(prompt)
    data = getattr(response, "data", None)
    if not isinstance(data, Mapping):
        raise LLMAgentError("LLM response data must be an object")
    command = str(data.get("command") or "").strip()
    reason = str(data.get("reason") or "").strip()
    if not reason:
        raise LLMAgentError("LLM response reason must be non-empty")
    return Decision(
        command,
        f"{owner.value}.llm",
        reason,
        continuation=ContinuationChange.clear(),
        payload=(
            {
                BUILD_EXCHANGE_KEY: {
                    "scope_id": request.scope.id,
                    "system": prompt.messages[0].content,
                    "user": prompt.messages[-1].content,
                    "assistant": str(getattr(response, "raw_text", ""))
                    or json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                    "snapshot": snapshot,
                }
            }
            if owner is AgentKind.BUILD
            else {}
        ),
        metrics={
            "model": str(getattr(response, "model", "")),
            "usage": _jsonable(getattr(response, "usage", {})),
        },
    )


def _prompt(
    request: DecisionRequest,
    owner: AgentKind,
    language: PromptLanguage,
) -> tuple[LLMRequest, Mapping[str, Any]]:
    state = request.state
    system = (
        f"You are the LLM {owner.value.upper()} agent for "
        "Slay the Spire. Choose exactly one executable CommunicationMod "
        "command from the current state. Every decision, including an obvious "
        "or forced one, is yours. Choice, monster, and potion indexes are "
        "zero-based; PLAY hand positions are one-based. Arguments must be "
        "numeric indexes, never choice labels: select screen.choices[N] with "
        "the exact command choose N. A play N command may reference only "
        "combat.hand[N-1], never deck, draw_pile, discard_pile, or exhaust_pile; "
        "verify 1 <= N <= len(combat.hand) and is_playable=true. Common forms "
        "are: choose <index>, "
        "play <hand-position> [monster], "
        "potion use <slot> [monster], potion discard <slot>, and exact action "
        "names such as end, skip, proceed, confirm, or leave. Never invent an "
        "unavailable command family. Return only the requested JSON object."
    )
    system += (
        " Write the reason in Chinese."
        if language is PromptLanguage.CHINESE
        else " Write the reason in English."
    )
    payload = {
        "owner": owner.value,
        "scope_id": request.scope.id,
        "state_owner_hint": state.owner_hint.value,
        "state_scope_id": state.scope_id,
        "terminal": state.terminal,
        "continuation": _continuation(request),
        "screen": {
            "type": state.screen.type,
            "interaction_id": state.screen.interaction_id,
            "available_command_families": list(_available_commands(state)),
            "choices": _jsonable(state.screen.choices),
            "details": _jsonable(state.screen.details),
            "current_action": state.screen.current_action,
        },
        "facts": _jsonable(state.facts),
        "combat": _jsonable(state.combat),
        "shared": _jsonable(request.shared),
        "previous": {
            "command": request.previous.command,
            "confirmed": request.previous.confirmed,
            "error": request.previous.error,
        },
    }
    entity_facts = _entity_facts(state)
    if entity_facts:
        payload["entity_facts"] = entity_facts
    messages = tuple(
        LLMMessage(item["role"], item["content"])
        for item in room_messages(request.shared, request.scope.id)
    )
    previous = room_snapshot(request.shared, request.scope.id) if messages else None
    current = (
        payload
        if previous is None
        else {
            "instruction": (
                "Apply this update to the preceding confirmed state; omitted "
                "fields are unchanged and numeric path segments are list indexes."
            ),
            "state_update": context_delta(previous, payload),
        }
    )
    return (
        LLMRequest(
            f"{owner.value}.llm",
            (*messages, LLMMessage("user", json.dumps(current, ensure_ascii=False)))
            if messages
            else (
                LLMMessage("system", system),
                LLMMessage("user", json.dumps(current, ensure_ascii=False)),
            ),
            _SCHEMA,
        ),
        payload,
    )


def _continuation(request: DecisionRequest) -> Mapping[str, Any] | None:
    value = request.continuation
    if value is None:
        return None
    return {
        "owner": value.owner.value,
        "kind": value.kind,
        "scope_id": value.scope_id,
        "expected_screens": list(value.expected_screens),
        "data": _jsonable(value.data),
    }


def _entity_facts(state: GameState) -> Mapping[str, Any]:
    db = StsDB()
    values = _named_entities((state.facts, state.combat, state.screen.details))
    result = {}
    for kind, query in (
        ("cards", db.card),
        ("relics", db.relic),
        ("potions", db.potion),
    ):
        facts = {
            str(fact["name"]): fact
            for value in sorted(values, key=str.casefold)
            if (fact := query(value)) is not None
        }
        if facts:
            result[kind] = list(facts.values())
    mentioned = db.mentions(
        json.dumps(_jsonable(state.screen.choices), ensure_ascii=False),
        json.dumps(_jsonable(state.screen.details), ensure_ascii=False),
    )
    for kind, facts in mentioned.items():
        current = {str(row["name"]): row for row in result.get(kind, ())}
        current.update({str(row["name"]): row for row in facts})
        result[kind] = list(current.values())
    return result


def _named_entities(value: object) -> set[str]:
    names = set()
    if isinstance(value, Mapping):
        name = value.get("name") or value.get("id")
        if name:
            names.add(str(name))
        for item in value.values():
            names.update(_named_entities(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            names.update(_named_entities(item))
    return names


def _available_commands(state: GameState) -> tuple[str, ...]:
    return tuple(
        command
        for command in state.screen.commands
        if command not in _CONTROL_COMMANDS
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if str(key) not in _HIDDEN_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


__all__ = [
    "LLMAgentError",
    "LLMAgentTool",
    "create_llm_build_agent",
    "create_llm_combat_agent",
]
