"""Render recorded LLM JSON calls as a plain-text conversation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, TextIO


class LLMLogViewError(ValueError):
    """A file is not a supported recorded LLM call."""


def render_llm_log(value: Mapping[str, Any]) -> str:
    request = value.get("request")
    if not isinstance(request, Mapping):
        raise LLMLogViewError("LLM log has no request object")
    messages = request.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise LLMLogViewError("LLM log request has no messages array")

    sections: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise LLMLogViewError("LLM log contains an invalid request message")
        role = str(message.get("role") or "unknown").upper()
        content = message.get("content")
        sections.append(_section(role, _text(content)))

    response = value.get("response")
    assistant = ""
    if isinstance(response, Mapping):
        raw_text = response.get("raw_text")
        if raw_text not in (None, ""):
            assistant = _text(raw_text)
        elif response.get("data") is not None:
            assistant = json.dumps(
                response["data"],
                ensure_ascii=False,
                indent=2,
            )
        else:
            assistant = ""
    sections.append(_section("ASSISTANT", assistant))

    return "\n\n".join(sections) + "\n"


def load_llm_log(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise LLMLogViewError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise LLMLogViewError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise LLMLogViewError(f"LLM log root must be an object: {path}")
    return value


def view_path(path: Path, *, output: TextIO = sys.stdout) -> int:
    files = _log_files(path)
    for index, file_path in enumerate(files):
        if len(files) > 1:
            if index:
                output.write("\n")
            output.write(f"========== {file_path.name} ==========\n\n")
        output.write(render_llm_log(load_llm_log(file_path)))
    return len(files)


def _log_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix == ".json")
        if files:
            return files
        raise LLMLogViewError(f"directory contains no JSON logs: {path}")
    raise LLMLogViewError(f"LLM log path does not exist: {path}")


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2)


def _section(role: str, content: str) -> str:
    return f"========== {role} ==========\n{content}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="one LLM audit JSON file or a directory containing JSON files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        view_path(args.path)
    except LLMLogViewError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LLMLogViewError",
    "load_llm_log",
    "main",
    "render_llm_log",
    "view_path",
]
