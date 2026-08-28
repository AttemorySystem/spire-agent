"""SubAgent-owned, provider-neutral boundary for structured LLM calls.

SubAgents own prompts and validate domain output.  Provider adapters only turn
this request into an API call and return the decoded JSON object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


def _mapping_copy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


class PromptLanguage(str, Enum):
    ENGLISH = "en"
    CHINESE = "zh"

    @classmethod
    def parse(cls, value: "PromptLanguage | str") -> "PromptLanguage":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().casefold())
        except ValueError as error:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unsupported prompt language {value!r}; expected one of: {supported}"
            ) from error


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        content = str(self.content).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported LLM message role: {role!r}")
        if not content:
            raise ValueError("LLM message content must not be empty")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)


@dataclass(frozen=True, slots=True)
class LLMRequest:
    purpose: str
    messages: tuple[LLMMessage, ...]
    response_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        purpose = str(self.purpose).strip()
        if not purpose:
            raise ValueError("LLM request purpose must not be empty")
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("LLM request requires at least one message")
        if not all(isinstance(message, LLMMessage) for message in messages):
            raise TypeError("LLM request messages must be LLMMessage values")
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "response_schema", _mapping_copy(self.response_schema))


@dataclass(frozen=True, slots=True)
class LLMResponse:
    data: Mapping[str, Any]
    raw_text: str = ""
    model: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _mapping_copy(self.data))
        object.__setattr__(self, "raw_text", str(self.raw_text))
        object.__setattr__(self, "model", str(self.model))
        object.__setattr__(self, "usage", _mapping_copy(self.usage))
        object.__setattr__(self, "reasoning", str(self.reasoning))


class LLMOutputError(ValueError):
    """A provider returned output that cannot satisfy the structured boundary."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        reasoning: str = "",
    ) -> None:
        self.raw_text = str(raw_text)
        self.reasoning = str(reasoning)
        super().__init__(message)


__all__ = [
    "LLMMessage",
    "LLMOutputError",
    "LLMRequest",
    "LLMResponse",
    "PromptLanguage",
]
