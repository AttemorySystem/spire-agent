"""Mandatory one-file audit trail for every structured LLM call."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any

from spire_agent.adapters.openai_llm import LLMSettings, OpenAICompatibleLLMClient
from spire_agent.subagents.llm import LLMRequest, LLMResponse

from .log_io import jsonable, write_json
from .run_directory import RunDirectory


class LLMAuditError(RuntimeError):
    """An LLM call cannot satisfy the mandatory audit contract."""


@dataclass(slots=True)
class _ActiveCall:
    path: Path
    record: dict[str, Any]
    started_at: float


class LLMCallRecorder:
    """Reserve and atomically finalize one JSON file per logical call."""

    __slots__ = ("_directory", "_lock", "_next_sequence", "_secrets")

    def __init__(
        self,
        directory: RunDirectory,
        *,
        secrets: Sequence[str] = (),
    ) -> None:
        self._directory = directory
        self._lock = Lock()
        self._next_sequence: int | None = None
        self._secrets = tuple(str(item) for item in secrets if str(item))

    def begin(self, request: LLMRequest) -> _ActiveCall:
        try:
            llm_dir = self._directory.path / "llm"
            llm_dir.mkdir(exist_ok=True)
            with self._lock:
                sequence = self._reserve_sequence(llm_dir)
                label = _filename_label(request.purpose)
                path = llm_dir / f"{sequence:06d}-{label}.json"
                record = {
                    "schema_version": 1,
                    "call_id": f"{sequence:06d}",
                    "purpose": request.purpose,
                    "status": "in_progress",
                    "started_at": _utc_now(),
                    "completed_at": None,
                    "elapsed_ms": None,
                    "request": {
                        "messages": [
                            {"role": item.role, "content": item.content}
                            for item in request.messages
                        ],
                        "response_schema": jsonable(request.response_schema),
                    },
                    "response": None,
                    "error": None,
                }
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(self._redact(record), stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                self._next_sequence = sequence + 1
            return _ActiveCall(path, record, time.perf_counter())
        except Exception as error:
            if isinstance(error, LLMAuditError):
                raise
            raise LLMAuditError(f"cannot create LLM audit log: {error}") from error

    def success(self, call: _ActiveCall, response: LLMResponse) -> None:
        record = dict(call.record)
        record.update(
            {
                "status": "success",
                "completed_at": _utc_now(),
                "elapsed_ms": round(
                    (time.perf_counter() - call.started_at) * 1000,
                    3,
                ),
                "response": {
                    "raw_text": response.raw_text,
                    "data": jsonable(response.data),
                    "model": response.model,
                    "usage": jsonable(response.usage),
                    **({"reasoning": response.reasoning} if response.reasoning else {}),
                },
            }
        )
        self._finalize(call.path, record)

    def failure(self, call: _ActiveCall, error: BaseException) -> None:
        raw_text = str(getattr(error, "raw_text", ""))
        reasoning = str(getattr(error, "reasoning", ""))
        record = dict(call.record)
        record.update(
            {
                "status": "error",
                "completed_at": _utc_now(),
                "elapsed_ms": round(
                    (time.perf_counter() - call.started_at) * 1000,
                    3,
                ),
                "response": (
                    {
                        "raw_text": raw_text,
                        "data": None,
                        "model": "",
                        "usage": {},
                        **({"reasoning": reasoning} if reasoning else {}),
                    }
                    if raw_text or reasoning
                    else None
                ),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        self._finalize(call.path, record)

    def _reserve_sequence(self, directory: Path) -> int:
        if self._next_sequence is not None:
            return self._next_sequence
        existing = []
        pattern = re.compile(r"^(\d{6})-[A-Za-z0-9._-]+\.json$")
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match is not None:
                existing.append(int(match.group(1)))
        return max(existing, default=0) + 1

    def _finalize(self, path: Path, record: Mapping[str, Any]) -> None:
        try:
            write_json(path, self._redact(record))
        except Exception as error:
            raise LLMAuditError(f"cannot finalize LLM audit log {path}: {error}") from error

    def _redact(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): self._redact(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "***REDACTED***")
        return value


class RecordingLLMClient:
    """Fail-closed LLM decorator backed by LLMCallRecorder."""

    __slots__ = ("_client", "_recorder")

    def __init__(self, client: object, recorder: LLMCallRecorder) -> None:
        self._client = client
        self._recorder = recorder

    def complete(self, request: LLMRequest) -> LLMResponse:
        complete = getattr(self._client, "complete", None)
        if not callable(complete):
            raise TypeError("recorded LLM has no complete() method")
        call = self._recorder.begin(request)
        try:
            response = complete(request)
        except BaseException as error:
            try:
                self._recorder.failure(call, error)
            except LLMAuditError as audit_error:
                raise audit_error from error
            raise
        self._recorder.success(call, response)
        return response


def create_run_llm_client(
    run_directory: RunDirectory,
    *,
    base_url: str = "",
    model: str = "",
    provider_client: object | None = None,
    stream_event: Callable[[str, str], object] | None = None,
) -> RecordingLLMClient:
    settings = LLMSettings.from_env(base_url=base_url, model=model)
    provider = OpenAICompatibleLLMClient(
        settings,
        client=provider_client,
        stream_event=stream_event,
    )
    return RecordingLLMClient(
        provider,
        LLMCallRecorder(run_directory, secrets=(settings.api_key,)),
    )


def _filename_label(value: object) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return label or "llm-call"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

__all__ = [
    "LLMAuditError",
    "LLMCallRecorder",
    "RecordingLLMClient",
    "create_run_llm_client",
]
