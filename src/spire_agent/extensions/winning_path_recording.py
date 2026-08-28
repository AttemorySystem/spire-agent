"""One durable audit record for every CardRewardPolicy decision."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Lock

from spire_agent.contracts import ContextEntry
from spire_agent.tools.winning_path import WINNING_PATH_REVIEW_KEY

from .log_io import jsonable, write_json
from .run_directory import RunDirectory


class WinningPathRecorder:
    __slots__ = ("_directory", "_lock", "_next_sequence")

    def __init__(self, directory: RunDirectory) -> None:
        self._directory = directory
        self._lock = Lock()
        self._next_sequence: int | None = None

    def on_entry(self, entry: ContextEntry) -> None:
        decision = entry.decision
        review = (
            decision.payload.get(WINNING_PATH_REVIEW_KEY)
            if decision is not None
            else None
        )
        if not isinstance(review, Mapping):
            return
        directory = self._directory.path / "winning_path"
        directory.mkdir(exist_ok=True)
        with self._lock:
            sequence = self._reserve(directory)
            path = directory / f"{sequence:06d}.json"
            payload = {
                "schema_version": 1,
                "entry_index": entry.index,
                "confirmed": entry.confirmed,
                "command": entry.command,
                "source": decision.source,
                "reason": decision.reason,
                "policy_result": jsonable(
                    decision.payload.get("card_reward_policy_result")
                ),
                "llm_proposal": jsonable(decision.payload.get("llm_proposal")),
                "review": jsonable(review),
            }
            write_json(path, payload, sort_keys=True)
            self._next_sequence = sequence + 1

    def _reserve(self, directory: Path) -> int:
        if self._next_sequence is not None:
            return self._next_sequence
        existing = [
            int(path.stem)
            for path in directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")
            if path.stem.isdigit()
        ]
        return max(existing, default=0) + 1

__all__ = ["WinningPathRecorder"]
