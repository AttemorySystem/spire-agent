"""Deterministic cross-combat gate for MCTS potion access."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

from spire_agent.contracts import GameState
from spire_agent.extensions.log_io import append_jsonl, jsonable

from .tool import MCTSResult


SAFE, DANGER, EMERGENCY, UNKNOWN = "SAFE", "DANGER", "EMERGENCY", "UNKNOWN"
MAX_POTION_SLOTS = 5
_SEVERITY = {SAFE: 0, DANGER: 1, EMERGENCY: 2, UNKNOWN: 3}


class PotionGate:
    """Open zero, one, or two potion slots after a no-potion search."""

    def __init__(self, run_directory: object) -> None:
        self._runs = run_directory
        self._scope = self._decided_level = ""
        self._decided_inventory: tuple[int, ...] = ()
        self._authorized: set[int] = set()
        self._released: set[int] = set()

    def active_slots(self, state: GameState) -> tuple[int, ...]:
        self._enter(state)
        if state.screen.type != "NONE" or "potion" not in state.screen.commands:
            return ()
        return tuple(sorted(self._authorized & set(potion_slots(state))))

    def select(
        self,
        state: GameState,
        baseline: MCTSResult,
        search: Callable[..., MCTSResult],
    ) -> MCTSResult:
        self._enter(state)
        active = self.active_slots(state)
        available = tuple(
            slot for slot in potion_slots(state) if slot not in self._released
        )
        if "potion" not in state.screen.commands:
            return baseline

        before = assess_risk(state, baseline)
        level = str(before["level"])
        smoke = _smoke_slots(state)
        if smoke and before.get("win_samples") == 0 and not _boss_combat(state):
            slot = smoke[0]
            self._released.add(slot)
            self._record(state, before, (), (slot,), "SMOKE_BOMB_ESCAPE")
            return MCTSResult(
                f"potion use {slot}",
                None,
                {**dict(baseline.metrics), "potion_gate": "SMOKE_BOMB_ESCAPE"},
            )
        live_inventory = tuple(potion_slots(state))
        if not available or level in {SAFE, UNKNOWN} or not self._may_decide(level, live_inventory):
            self._record(state, before, (), (), "NO_RELEASE")
            return baseline

        probes: list[dict[str, Any]] = []
        if is_heart(state):
            selected, reason = available, "HEART_RELEASE_ALL"
        elif 2 - len(self._released) <= 0:
            selected, reason = (), "COMBAT_BUDGET_EXHAUSTED"
        else:
            probes = self._probe(
                state,
                search,
                (tuple(sorted((*active, slot))) for slot in available),
            )
            selected = _best(before, probes, size=len(active) + 1)
            reason = "SINGLE_MATERIAL_GAIN" if selected else "NO_MATERIAL_SINGLE_GAIN"
            if selected and _risk_of(probes, selected)["level"] != level:
                reason = "SINGLE_DEESCALATES_RISK"
            selected_risk = _risk_of(probes, selected) if selected else before
            if (
                not active
                and 2 - len(self._released) >= 2
                and _severity(selected_risk) >= _SEVERITY[DANGER]
                and len(available) > 1
            ):
                probes += self._probe(state, search, combinations(available, 2))
                pair = _best(before, probes, size=2)
                if pair and (
                    not selected
                    or _meaningfully_better(
                        _risk_of(probes, pair), _risk_of(probes, selected)
                    )
                ):
                    selected, reason = pair, "PAIR_REQUIRED_FOR_EMERGENCY"

        self._decided_level = level
        self._decided_inventory = live_inventory
        if not selected:
            self._record(state, before, probes, (), reason)
            return baseline

        self._authorized.update(selected)
        self._released.update(selected)
        final = search(state, potion_slots=selected, search_role="potion_final")
        self._record(state, before, probes, selected, reason, final)
        return final

    def _probe(
        self,
        state: GameState,
        search: Callable[..., MCTSResult],
        candidates: Iterable[tuple[int, ...]],
    ) -> list[dict[str, Any]]:
        rows = []
        for slots in candidates:
            slots = tuple(slots)
            try:
                result = search(
                    state,
                    potion_slots=slots,
                    probe=True,
                    search_role="potion_probe",
                )
                rows.append(
                    {"slots": slots, "result": result, "risk": assess_risk(state, result)}
                )
            except Exception as error:
                rows.append({"slots": slots, "error": str(error)})
        return rows

    def _may_decide(self, level: str, inventory: tuple[int, ...]) -> bool:
        return not self._decided_level or (
            self._decided_level == DANGER and level == EMERGENCY
        ) or inventory != self._decided_inventory

    def _enter(self, state: GameState) -> None:
        if state.scope_id == self._scope:
            return
        self._scope, self._decided_level = state.scope_id, ""
        self._decided_inventory = ()
        self._authorized.clear()
        self._released.clear()

    def _record(
        self,
        state: GameState,
        baseline: Mapping[str, Any],
        probes: Sequence[Mapping[str, Any]],
        selected: Sequence[int],
        reason: str,
        final: MCTSResult | None = None,
    ) -> None:
        path = getattr(self._runs, "path", None)
        if path is None:
            return
        rows = []
        for row in probes:
            result = row.get("result")
            rows.append(
                {
                    "slots": row.get("slots"),
                    "search_id": result.metrics.get("search_id")
                    if isinstance(result, MCTSResult)
                    else None,
                    "risk": row.get("risk"),
                    "material": _material(baseline, row.get("risk") or {}),
                    "error": row.get("error"),
                }
            )
        append_jsonl(
            path / "potion_decisions.jsonl",
            jsonable(
                {
                    "schema_version": 1,
                    "scope_id": state.scope_id,
                    "interaction_id": state.screen.interaction_id,
                    "baseline": baseline,
                    "probes": rows,
                    "selected_slots": selected,
                    "reason": reason,
                    "final_search_id": final.metrics.get("search_id") if final else None,
                }
            ),
        )


def potion_slots(state: GameState) -> tuple[int, ...]:
    values = state.facts.get("potions")
    values = values if isinstance(values, Sequence) else ()
    result = []
    for index, raw in enumerate(values[:MAX_POTION_SLOTS]):
        if not isinstance(raw, Mapping) or raw.get("can_use") is False:
            continue
        name = str(raw.get("id") or raw.get("name") or "")
        name = "".join(char for char in name.casefold() if char.isalnum())
        if not name or name in {"potionslot", "emptypotionslot"}:
            continue
        if name.startswith("smokebomb"):
            continue
        slot = raw.get("slot", index)
        if isinstance(slot, int) and not isinstance(slot, bool) and 0 <= slot < 5:
            result.append(slot)
    return tuple(dict.fromkeys(result))


def _smoke_slots(state: GameState) -> tuple[int, ...]:
    values = state.facts.get("potions")
    values = values if isinstance(values, Sequence) else ()
    result = []
    for index, raw in enumerate(values[:MAX_POTION_SLOTS]):
        if not isinstance(raw, Mapping) or raw.get("can_use") is False:
            continue
        name = "".join(
            char
            for char in str(raw.get("id") or raw.get("name") or "").casefold()
            if char.isalnum()
        )
        slot = raw.get("slot", index)
        if name.startswith("smokebomb") and isinstance(slot, int) and 0 <= slot < 5:
            result.append(slot)
    return tuple(dict.fromkeys(result))


def _boss_combat(state: GameState) -> bool:
    return "boss" in str(state.facts.get("room_type") or "").casefold()


def assess_risk(state: GameState, result: MCTSResult) -> dict[str, Any]:
    risk = result.metrics.get("risk")
    risk = risk if isinstance(risk, Mapping) else {}
    current_hp, max_hp = _health(state)
    end_hp = _first_number(
        risk,
        "expectedEndHpOnWin",
        "expectedEffectiveEndHpOnWin",
        "meanBestWinEndHp",
    )
    if current_hp is None or max_hp is None or end_hp is None:
        return {"level": UNKNOWN, "reason": "missing HP projection"}
    loss = max(0.0, current_hp - end_hp)
    loss_current, loss_max, end_max = loss / current_hp, loss / max_hp, end_hp / max_hp
    no_win = result.metrics.get("credible_win_evidence") is False or (
        isinstance(risk.get("winSamples"), (int, float)) and risk["winSamples"] <= 0
    )
    emergency = no_win or loss_current >= 0.50 or loss_max >= 0.35 or (
        end_max <= 0.15 and loss_max >= 0.10
    )
    danger = loss_max >= 0.20 or (end_max <= 0.25 and loss_max >= 0.10)
    return {
        "level": EMERGENCY if emergency else DANGER if danger else SAFE,
        "current_hp": current_hp,
        "max_hp": max_hp,
        "expected_end_hp": end_hp,
        "expected_hp_loss": loss,
        "loss_of_current_hp": round(loss_current, 4),
        "loss_of_max_hp": round(loss_max, 4),
        "expected_end_hp_of_max": round(end_max, 4),
        "no_credible_win": no_win,
        "win_samples": risk.get("winSamples"),
        "win_sample_rate": risk.get("winSampleRate"),
    }


def is_heart(state: GameState) -> bool:
    boss = "".join(
        char for char in str(state.facts.get("act_boss") or "").casefold() if char.isalnum()
    )
    return (
        state.facts.get("act") == 4
        and "boss" in str(state.facts.get("room_type") or "").casefold()
        and boss in {"theheart", "corruptheart"}
    )


def _best(
    before: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], size: int = 1
) -> tuple[int, ...]:
    candidates = [
        row
        for row in rows
        if len(row.get("slots") or ()) == size
        and _material(before, row.get("risk") or {})
    ]
    return tuple(min(candidates, key=_candidate_key)["slots"]) if candidates else ()


def _material(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    if _severity(after) < _severity(before):
        return True
    hp_gain = _number(after.get("expected_end_hp")) - _number(
        before.get("expected_end_hp")
    )
    threshold = max(3.0, 0.05 * _number(before.get("max_hp")))
    rate_gain = _number(after.get("win_sample_rate")) - _number(
        before.get("win_sample_rate")
    )
    return hp_gain >= threshold or rate_gain >= 0.03


def _candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    risk = row.get("risk") or {}
    return (
        _severity(risk),
        -_number(risk.get("win_sample_rate"), -1.0),
        -_number(risk.get("expected_end_hp"), -1.0),
        tuple(row.get("slots") or ()),
    )


def _meaningfully_better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _severity(left) < _severity(right) or (
        _severity(left) == _severity(right)
        and (
            _number(left.get("win_sample_rate"))
            >= _number(right.get("win_sample_rate")) + 0.02
            or _number(left.get("expected_end_hp"))
            >= _number(right.get("expected_end_hp")) + 3.0
        )
    )


def _risk_of(
    rows: Sequence[Mapping[str, Any]], slots: Sequence[int]
) -> Mapping[str, Any]:
    return next(
        (row.get("risk") or {} for row in rows if tuple(row.get("slots") or ()) == tuple(slots)),
        {},
    )


def _health(state: GameState) -> tuple[float | None, float | None]:
    player = state.combat.get("player", {}) if state.combat else {}
    player = player if isinstance(player, Mapping) else {}
    current = state.facts.get("current_hp", player.get("current_hp"))
    maximum = state.facts.get("max_hp", player.get("max_hp"))
    if not isinstance(current, (int, float)) or not isinstance(maximum, (int, float)):
        return None, None
    return float(current), float(maximum)


def _first_number(value: Mapping[str, Any], *keys: str) -> float | None:
    return next(
        (float(value[key]) for key in keys if isinstance(value.get(key), (int, float))),
        None,
    )


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _severity(risk: Mapping[str, Any]) -> int:
    return _SEVERITY.get(str(risk.get("level")), 3)


__all__ = [
    "DANGER",
    "EMERGENCY",
    "MAX_POTION_SLOTS",
    "PotionGate",
    "SAFE",
    "UNKNOWN",
    "assess_risk",
    "is_heart",
    "potion_slots",
]
