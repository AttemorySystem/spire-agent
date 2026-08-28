"""Append confirmed permanent card rewards to one compatible JSONL journal."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from threading import Lock
from typing import Any

from spire_agent.contracts import ContextEntry
from spire_agent.tools.winning_path import (
    CARD_CHOICE_REVIEW_KEY,
)

from .log_io import append_jsonl, jsonable, write_jsonl
from .run_directory import RunDirectory


class CardChoiceRecorder:
    """Observe decisions without participating in routing or execution."""

    __slots__ = (
        "_directory", "_lock", "_records", "_floor_counts", "_last_enemies",
    )

    def __init__(self, directory: RunDirectory) -> None:
        self._directory = directory
        self._lock = Lock()
        self._records: list[dict[str, Any]] = []
        self._floor_counts: Counter[int] = Counter()
        self._last_enemies: tuple[str, ...] = ()

    def on_entry(self, entry: ContextEntry) -> None:
        self._observe_combat(entry)
        if entry.confirmed:
            self._record_choice(entry)
        if entry.state.terminal:
            self._finish(entry)

    def _record_choice(self, entry: ContextEntry) -> None:
        decision = entry.decision
        review = (
            decision.payload.get(CARD_CHOICE_REVIEW_KEY)
            if decision is not None
            else None
        )
        if not isinstance(review, Mapping):
            return
        path_state = _mapping(review.get("state"))
        policy = _mapping(decision.payload.get("card_reward_policy_result"))
        candidates = [
            row for row in review.get("candidates") or () if isinstance(row, Mapping)
        ]
        offered = [str(row.get("name") or "") for row in candidates]
        if not offered:
            return
        floor = int(path_state.get("floor") or 0)
        choice_index = self._floor_counts[floor]
        self._floor_counts[floor] += 1
        selected = str(policy.get("card") or "")
        bowl = selected == "Singing Bowl"
        skipped = entry.command == "skip"
        picked = None if skipped or bowl else selected or None
        run_id = self._directory.seed
        record = {
            "schema_version": 1,
            "choice_id": f"{run_id}:f{floor}:c{choice_index}",
            "entry_index": entry.index,
            "context": {
                "act_boss": path_state.get("act_boss"),
                "deck_before_counts": jsonable(path_state.get("deck") or {}),
                "deck_before_upgrades": jsonable(
                    path_state.get("upgrades") or {}
                ),
                "hp_after_floor": path_state.get("current_hp"),
                "max_hp_after_floor": path_state.get("max_hp"),
                "gold_after_floor": path_state.get("gold"),
                "relics_before_floor_rewards": list(
                    path_state.get("relics") or ()
                ),
                "active_modules": list(path_state.get("active_modules") or ()),
            },
            "decision": {
                "act": int(path_state.get("act") or 0),
                "floor": floor,
                "kind": _choice_kind(path_state.get("room_type")),
                "offered": offered,
                "picked": picked,
                "skipped": skipped,
                "used_singing_bowl": bowl,
                "action": entry.command,
                "source": decision.source,
                "reason": decision.reason,
                "policy": policy.get("policy"),
                "selection_kind": policy.get("selection_kind"),
                "allowed_choice_ids": list(review.get("allowed_choice_ids") or ()),
                "picker_id": review.get("picker_id"),
                "policy_version": review.get("policy_version"),
            },
            "run": {
                "run_id": run_id,
                "character": review.get("character"),
                "ascension": int(path_state.get("ascension_level") or 0),
                "floor_reached": None,
                "victory": None,
                "heart_kill": None,
                "killed_by": None,
            },
        }
        with self._lock:
            self._records.append(record)
            append_jsonl(self._directory.path / "card_choices.jsonl", record)

    def _finish(self, entry: ContextEntry) -> None:
        if not self._records:
            return
        facts = entry.state.facts
        floor = int(facts.get("floor") or 0)
        room = str(facts.get("room_type") or "")
        explicit = facts.get("victory")
        victory = bool(explicit) if isinstance(explicit, bool) else "victory" in room.casefold()
        heart = facts.get("heart_kill")
        heart_kill = bool(heart) if isinstance(heart, bool) else victory and int(
            facts.get("act") or 0
        ) >= 4
        outcome = {
            "floor_reached": floor,
            "victory": victory,
            "heart_kill": heart_kill,
            "killed_by": (
                None
                if victory
                else facts.get("killed_by") or ", ".join(self._last_enemies) or None
            ),
        }
        with self._lock:
            for record in self._records:
                record["run"].update(outcome)
            write_jsonl(self._directory.path / "card_choices.jsonl", self._records)

    def _observe_combat(self, entry: ContextEntry) -> None:
        combat = entry.state.combat
        monsters = combat.get("monsters") if isinstance(combat, Mapping) else ()
        names = tuple(
            str(row.get("name") or row.get("id") or "").strip()
            for row in monsters or ()
            if isinstance(row, Mapping) and not row.get("is_gone")
        )
        if any(names):
            self._last_enemies = tuple(name for name in names if name)


def _choice_kind(value: object) -> str:
    room = str(value or "").casefold()
    if "boss" in room:
        return "boss_card_reward"
    if "elite" in room:
        return "elite_card_reward"
    if "monster" in room:
        return "combat_card_reward"
    return "event_card_reward"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["CardChoiceRecorder"]
