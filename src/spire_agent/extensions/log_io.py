"""Small durable JSON primitives shared by run recorders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    def dump(stream: TextIO) -> None:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        stream.write("\n")

    _replace(path, dump)


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    def dump(stream: TextIO) -> None:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

    _replace(path, dump)


def _replace(path: Path, dump: Callable[[TextIO], None]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            dump(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["append_jsonl", "jsonable", "write_json", "write_jsonl"]
