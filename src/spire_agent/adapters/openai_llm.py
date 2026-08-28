"""OpenAI-compatible structured LLM adapter configured by three variables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable

from openai import OpenAI

from spire_agent.subagents.llm import LLMOutputError, LLMRequest, LLMResponse


@dataclass(frozen=True, slots=True)
class LLMSettings:
    base_url: str
    model: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, *, base_url: str = "", model: str = "") -> "LLMSettings":
        def setting(name: str) -> str:
            return os.environ.get(name, "").strip()

        values = {
            "base_url": str(base_url).strip() or setting("MODEL_URL"),
            "model": str(model).strip() or setting("MODEL"),
            "api_key": setting("API_KEY"),
        }
        missing = [
            environment_name
            for field_name, environment_name in (
                ("base_url", "MODEL_URL"),
                ("model", "MODEL"),
                ("api_key", "API_KEY"),
            )
            if not values[field_name]
        ]
        if missing:
            raise ValueError(
                "missing LLM configuration: " + ", ".join(missing)
            )
        return cls(**values)


class OpenAICompatibleLLMClient:
    """Call an OpenAI-compatible JSON-mode chat completion endpoint."""

    __slots__ = ("_client", "_settings", "_stream_event")

    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: object | None = None,
        stream_event: Callable[[str, str], object] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=120.0,
            max_retries=0,
        )
        self._stream_event = stream_event

    def complete(self, request: LLMRequest) -> LLMResponse:
        chat = getattr(self._client, "chat", None)
        completions = getattr(chat, "completions", None)
        create = getattr(completions, "create", None)
        if not callable(create):
            raise TypeError("OpenAI-compatible client has no chat.completions.create")

        arguments = {
            "model": self._settings.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "response_format": {"type": "json_object"},
        }
        if self._stream_event is not None:
            return self._complete_stream(create, arguments, request.purpose)

        response = create(
            **arguments,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMOutputError("LLM response has no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        return _response(
            str(content or ""),
            model=str(getattr(response, "model", None) or self._settings.model),
            usage=_usage_mapping(getattr(response, "usage", None)),
        )

    def _complete_stream(
        self,
        create: Callable[..., object],
        arguments: dict[str, Any],
        purpose: str,
    ) -> LLMResponse:
        self._emit("start", purpose)
        content, reasoning, model, usage = [], [], self._settings.model, {}
        try:
            stream = create(
                **arguments,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                model = str(getattr(chunk, "model", None) or model)
                current_usage = _usage_mapping(getattr(chunk, "usage", None))
                if current_usage:
                    usage = current_usage
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                reasoning_delta = _reasoning_text(delta)
                content_delta = str(getattr(delta, "content", None) or "")
                if reasoning_delta:
                    reasoning.append(reasoning_delta)
                    self._emit("reasoning", reasoning_delta)
                if content_delta:
                    content.append(content_delta)
                    self._emit("content", content_delta)
            result = _response(
                "".join(content),
                model=model,
                usage=usage,
                reasoning="".join(reasoning),
            )
        except BaseException as error:
            self._emit("error", str(error))
            raise
        self._emit("done", "")
        return result

    def _emit(self, event: str, value: str) -> None:
        if self._stream_event is None:
            return
        try:
            self._stream_event(event, value)
        except Exception:
            pass


def _response(
    raw_text: str,
    *,
    model: str,
    usage: Mapping[str, Any],
    reasoning: str = "",
) -> LLMResponse:
    if not raw_text:
        raise LLMOutputError(
            "LLM response content is empty",
            raw_text=raw_text,
            reasoning=reasoning,
        )
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise LLMOutputError(
            f"LLM response is not valid JSON: {error.msg}",
            raw_text=raw_text,
            reasoning=reasoning,
        ) from error
    if not isinstance(data, Mapping):
        raise LLMOutputError(
            "LLM response must decode to a JSON object",
            raw_text=raw_text,
            reasoning=reasoning,
        )

    return LLMResponse(data, raw_text, model, usage, reasoning)


def _reasoning_text(delta: object) -> str:
    for name in ("reasoning", "reasoning_content"):
        value = getattr(delta, name, None)
        if isinstance(value, str):
            return value
    details = getattr(delta, "reasoning_details", None)
    if isinstance(details, list):
        return "".join(
            str(
                item.get("text") or item.get("content") or ""
                if isinstance(item, Mapping)
                else getattr(item, "text", None)
                or getattr(item, "content", None)
                or ""
            )
            for item in details
        )
    return ""


def _usage_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        result = model_dump()
        return dict(result) if isinstance(result, Mapping) else {}
    return {}


__all__ = [
    "LLMSettings",
    "OpenAICompatibleLLMClient",
]
