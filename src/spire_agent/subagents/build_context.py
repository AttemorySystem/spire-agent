"""Transactional room-scoped context owned by BuildAgent."""

from __future__ import annotations

from collections.abc import Mapping

from spire_agent.contracts import AgentKind, ContextEntry, GameState
from spire_agent.tools.run_keys import (
    RUN_KEYS_KEY,
    RUN_ROUTE_KEY,
    acquire,
    initial_keys,
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
        if decision is not None:
            acquired = decision.payload.get("acquired_key")
            if acquired:
                result[RUN_KEYS_KEY] = acquire(result, entry.state, acquired)
            route = decision.payload.get(RUN_ROUTE_KEY)
            if isinstance(route, Mapping):
                result[RUN_ROUTE_KEY] = dict(route)
            construction = decision.payload.get(RUN_CONSTRUCTION_KEY)
            if isinstance(construction, Mapping):
                result[RUN_CONSTRUCTION_KEY] = dict(construction)
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
    return tuple(messages)


__all__ = [
    "BUILD_CONVERSATION_KEY",
    "BUILD_EXCHANGE_KEY",
    "RUN_CONSTRUCTION_KEY",
    "BuildConversationReducer",
    "room_messages",
]
