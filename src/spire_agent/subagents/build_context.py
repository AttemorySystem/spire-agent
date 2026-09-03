"""Transactional room-scoped context owned by BuildAgent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spire_agent.contracts import AgentKind, ContextEntry, GameState
from spire_agent.tools.run_keys import (
    RUN_KEYS_KEY,
    RUN_ROUTE_KEY,
    acquire,
    initial_keys,
    readiness_fingerprint,
)
BUILD_CONVERSATION_KEY = "build_conversation"
BUILD_EXCHANGE_KEY = "build_exchange"
RUN_CONSTRUCTION_KEY = "run_construction"


class BuildConversationReducer:
    """Commit confirmed LLM exchanges and discard them outside their room."""

    def initialize(self, state: GameState) -> Mapping[str, object]:
        return {RUN_KEYS_KEY: initial_keys(state)}

    def reduce(
        self,
        shared: Mapping[str, object],
        entry: ContextEntry,
    ) -> Mapping[str, object]:
        result = dict(shared)
        if not entry.confirmed:
            return result
        decision = entry.decision
        route = decision.payload.get(RUN_ROUTE_KEY) if decision is not None else None
        if decision is not None:
            acquired = decision.payload.get("acquired_key")
            if acquired:
                result[RUN_KEYS_KEY] = acquire(result, entry.state, acquired)
            if isinstance(route, Mapping):
                result[RUN_ROUTE_KEY] = _invalidate_readiness(route, entry.state)
            construction = decision.payload.get(RUN_CONSTRUCTION_KEY)
            if isinstance(construction, Mapping):
                result[RUN_CONSTRUCTION_KEY] = dict(construction)
        if RUN_ROUTE_KEY in result and not isinstance(route, Mapping):
            result[RUN_ROUTE_KEY] = _invalidate_readiness(
                result[RUN_ROUTE_KEY], entry.state
            )
        if entry.state.terminal or entry.state.owner_hint is not AgentKind.BUILD:
            result.pop(BUILD_CONVERSATION_KEY, None)
            return result

        scope_id = entry.state.scope_id
        current = result.get(BUILD_CONVERSATION_KEY)
        if not isinstance(current, Mapping) or current.get("scope_id") != scope_id:
            current = None
            result.pop(BUILD_CONVERSATION_KEY, None)

        exchange = (
            decision.payload.get(BUILD_EXCHANGE_KEY)
            if decision is not None
            else None
        )
        if not isinstance(exchange, Mapping) or exchange.get("scope_id") != scope_id:
            return result

        system = exchange.get("system")
        user = exchange.get("user")
        assistant = exchange.get("assistant")
        if not all(isinstance(value, str) and value for value in (system, user, assistant)):
            raise ValueError("Build conversation exchange is malformed")

        messages = []
        if current is not None:
            raw_messages = current.get("messages")
            if isinstance(raw_messages, (list, tuple)):
                messages.extend(dict(item) for item in raw_messages if isinstance(item, Mapping))
        if not messages:
            messages.append({"role": "system", "content": system})
        elif messages[0] != {"role": "system", "content": system}:
            raise ValueError("Build conversation system prompt changed within a room")
        messages.extend(
            (
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            )
        )
        result[BUILD_CONVERSATION_KEY] = {
            "scope_id": scope_id,
            "messages": messages,
            **(
                {"snapshot": _plain(exchange["snapshot"])}
                if isinstance(exchange.get("snapshot"), Mapping)
                else {}
            ),
        }
        return result


def room_messages(
    shared: Mapping[str, object],
    scope_id: str,
) -> tuple[dict[str, str], ...]:
    """Return validated messages for the current room only."""

    conversation = shared.get(BUILD_CONVERSATION_KEY)
    if not isinstance(conversation, Mapping) or conversation.get("scope_id") != scope_id:
        return ()
    raw_messages = conversation.get("messages")
    if not isinstance(raw_messages, (list, tuple)):
        return ()
    messages = []
    for item in raw_messages:
        if not isinstance(item, Mapping):
            return ()
        role, content = item.get("role"), item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return ()
        messages.append({"role": role, "content": content})
    if len(messages) < 3 or len(messages) % 2 == 0:
        return ()
    expected = ("system",) + ("user", "assistant") * ((len(messages) - 1) // 2)
    if tuple(message["role"] for message in messages) != expected:
        return ()
    return tuple(messages)


def room_snapshot(
    shared: Mapping[str, object], scope_id: str
) -> Mapping[str, Any] | None:
    conversation = shared.get(BUILD_CONVERSATION_KEY)
    if (
        not isinstance(conversation, Mapping)
        or conversation.get("scope_id") != scope_id
    ):
        return None
    snapshot = conversation.get("snapshot")
    return _plain(snapshot) if isinstance(snapshot, Mapping) else None


def context_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, object]:
    """Return changed values and removals using dotted object/list paths."""

    changed: dict[str, object] = {}
    removed: list[str] = []

    def visit(path: tuple[str, ...], before: object, after: object) -> None:
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            for key in sorted(set(before) | set(after)):
                child = (*path, str(key))
                if key not in after:
                    removed.append(".".join(child))
                elif key not in before:
                    changed[".".join(child)] = _plain(after[key])
                else:
                    visit(child, before[key], after[key])
        elif isinstance(before, list) and isinstance(after, list):
            for index in range(max(len(before), len(after))):
                child = (*path, str(index))
                if index >= len(after):
                    removed.append(".".join(child))
                elif index >= len(before):
                    changed[".".join(child)] = _plain(after[index])
                else:
                    visit(child, before[index], after[index])
        elif _plain(before) != _plain(after):
            changed[".".join(path)] = _plain(after)

    visit((), _plain(previous), _plain(current))
    return {"set": changed, "remove": removed}


def _invalidate_readiness(route: object, state: GameState) -> object:
    if not isinstance(route, Mapping):
        return route
    expected = route.get("readiness_fingerprint")
    has_readiness = "encounter_readiness" in route or "rest_readiness" in route
    if not has_readiness or (
        expected is not None and expected == readiness_fingerprint(state)
    ):
        return dict(route)
    result = dict(route)
    result.pop("encounter_readiness", None)
    result.pop("rest_readiness", None)
    result.pop("readiness_fingerprint", None)
    result["readiness_stale"] = True
    return result


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "BUILD_CONVERSATION_KEY",
    "BUILD_EXCHANGE_KEY",
    "RUN_CONSTRUCTION_KEY",
    "BuildConversationReducer",
    "context_delta",
    "room_messages",
    "room_snapshot",
]
