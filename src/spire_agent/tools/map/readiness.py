"""Fight-now survival evidence for route encounters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from spire_agent.contracts import GameState
from spire_agent.extensions.log_io import append_jsonl, jsonable


_ELITES = {
    1: ("Gremlin Nob", "Lagavulin", "Three Sentries"),
    2: ("Gremlin Leader", "Slavers", "Book Of Stabbing"),
    3: ("Giant Head", "Nemesis", "Reptomancer"),
}
_ACT2_WEAK = {
    name: 1 / 5
    for name in ("Spheric Guardian", "Chosen", "Shell Parasite", "Three Byrds", "Two Thieves")
}
_ACT2_STRONG = {
    "Chosen And Byrds": 2 / 29,
    "Sentry And Sphere": 2 / 29,
    "Cultist And Chosen": 3 / 29,
    "Three Cultist": 3 / 29,
    "Shelled Parasite And Fungi": 3 / 29,
    "Snecko": 4 / 29,
    "Snake Plant": 6 / 29,
    "Centurion And Healer": 6 / 29,
}
_ACT3_WEAK = {
    name: 1 / 3 for name in ("Three Darklings", "Orb Walker", "Three Shapes")
}
_ACT3_STRONG = {
    name: 1 / 8
    for name in (
        "Spire Growth", "Transient", "Four Shapes", "Maw",
        "Sphere And Two Shapes", "Jaw Worm Horde", "Three Darklings",
        "Writhing Mass",
    )
}
_BOTTLES = {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"}
_ELITE_ALIASES = {
    "sentry": "Three Sentries",
    "blueslaver": "Slavers",
    "redslaver": "Slavers",
    "slaverblue": "Slavers",
    "slaverred": "Slavers",
    "slaverboss": "Slavers",
}


class EncounterReadiness:
    """Evaluate current HP/deck against independently simulated encounters."""

    def __init__(
        self,
        binary: str | Path,
        run_directory: object,
        *,
        worlds: int = 8,
        simulations: int = 100,
        max_time_ms: int = 25,
        max_decisions: int = 160,
        minimum_survival: float = 0.60,
    ) -> None:
        self.binary = Path(binary).resolve()
        self.runs = run_directory
        self.worlds = worlds
        self.simulations = simulations
        self.max_time_ms = max_time_ms
        self.max_decisions = max_decisions
        self.minimum_survival = minimum_survival
        self._cache: dict[str, dict[str, Any]] = {}

    def evaluate(self, state: GameState) -> dict[str, Any]:
        act = _integer(state.facts.get("act"))
        if act not in _ELITES:
            return {"status": "NOT_APPLICABLE", "act": act}
        history = _history(self.runs)
        rotation = _elite_rotation(act, history[2].get(act))
        groups = _groups(act, rotation["last_elite_if_known"])
        try:
            spec = _spec(
                state,
                tuple(name for group in groups.values() for name in group),
                history[1],
            )
            fingerprint = sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if fingerprint in self._cache:
                return {**self._cache[fingerprint], "cache_hit": True}
            if not self.binary.is_file():
                raise FileNotFoundError(self.binary)
            with tempfile.TemporaryDirectory(prefix="sts-encounter-") as directory:
                path = Path(directory) / "input.json"
                path.write_text(json.dumps(spec, separators=(",", ":")), "utf-8")
                completed = subprocess.run(
                    (
                        str(self.binary), "--battle-eval", str(path),
                        str(self.simulations), "0", str(self.worlds),
                        str(self.max_time_ms), str(self.max_decisions),
                    ),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip())
            result = {
                "status": "AVAILABLE",
                "act": act,
                "entry_hp": _integer(state.facts.get("current_hp")),
                "potions_modeled": False,
                "cache_hit": False,
                "fingerprint": fingerprint,
                "elite_rotation": rotation,
                **_summarize(json.loads(completed.stdout), groups, self.minimum_survival),
            }
            if act in {2, 3}:
                completed_hallways = sum(
                    room_act == act for room_act, _ in history[0]
                ) + int(
                    str(state.facts.get("room_type") or "").casefold() == "monsterroom"
                    and (act, _integer(state.facts.get("floor"))) not in history[0]
                )
                result["weak_hallways_remaining"] = max(0, 2 - completed_hallways)
        except Exception as error:
            result = {
                "status": "UNAVAILABLE",
                "act": act,
                "elite_rotation": rotation,
                "reason": f"{type(error).__name__}: {error}",
            }
        if "fingerprint" in result:
            self._cache[result["fingerprint"]] = dict(result)
        self._record(state, result)
        return result

    def _record(self, state: GameState, result: Mapping[str, Any]) -> None:
        try:
            path = self.runs.path / "encounter_readiness.jsonl"
        except Exception:
            return
        append_jsonl(
            path,
            jsonable(
                {
                    "schema_version": 1,
                    "scope_id": state.scope_id,
                    "floor": state.facts.get("floor"),
                    **dict(result),
                }
            ),
        )


def _groups(act: int, last_elite: object = None) -> dict[str, Mapping[str, float]]:
    possible = (
        tuple(name for name in _ELITES[act] if name != last_elite)
        or _ELITES[act]
    )
    elite = {name: 1 / len(possible) for name in possible}
    if act == 1:
        return {"ELITE": elite}
    weak, strong = (
        (_ACT2_WEAK, _ACT2_STRONG)
        if act == 2 else (_ACT3_WEAK, _ACT3_STRONG)
    )
    return {"WEAK_HALLWAY": weak, "STRONG_HALLWAY": strong, "ELITE": elite}


def _spec(
    state: GameState,
    targets: Sequence[str],
    bottles: Mapping[str, tuple[str, int]],
) -> dict[str, Any]:
    facts = state.facts
    required = (
        "seed", "ascension_level", "act", "floor", "current_hp", "max_hp",
        "gold", "class", "deck", "relics", "potions",
    )
    missing = [key for key in required if facts.get(key) is None]
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    bottled = set(bottles.values())
    deck = []
    for raw in _sequence(facts.get("deck")):
        if not isinstance(raw, Mapping) or not (card_id := raw.get("id") or raw.get("name")):
            continue
        upgrades = min(1, max(0, _integer(raw.get("upgrades", raw.get("upgrade")))))
        card = {"id": str(card_id), "upgrades": upgrades}
        if raw.get("misc") is not None:
            card["misc"] = raw["misc"]
        if raw.get("bottled") or (_card_key(raw.get("name") or card_id), upgrades) in bottled:
            card["bottled"] = True
        deck.append(card)
    if not deck:
        raise ValueError("permanent deck is empty")
    relics = []
    for raw in _sequence(facts.get("relics")):
        if not isinstance(raw, Mapping) or not (relic_id := raw.get("id") or raw.get("name")):
            continue
        relics.append({"id": str(relic_id), "counter": _integer(raw.get("counter", -1))})
    potions = []
    for _ in _sequence(facts.get("potions"))[:5]:
        potions.append({"id": "Potion Slot"})
    return {
        "game_state": {
            **{key: facts[key] for key in required[:8]},
            "deck": deck,
            "relics": relics,
            "potions": potions,
        },
        "candidates": [],
        "targets": list(dict.fromkeys(targets)),
    }


def _summarize(
    raw: Mapping[str, Any],
    groups: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> dict[str, Any]:
    targets = {}
    for row in raw.get("targets") or ():
        aggregate = row.get("aggregate") if isinstance(row, Mapping) else None
        if not isinstance(aggregate, Mapping):
            continue
        wins, attempts = _integer(aggregate.get("wins")), _integer(aggregate.get("attempts"))
        lower, upper = _wilson(wins, attempts)
        name = str(row.get("target") or "")
        targets[name] = {
            "target": name,
            "status": _status(lower, upper, threshold),
            "wins": wins,
            "attempts": attempts,
            "survival": round(wins / attempts, 4) if attempts else 0.0,
            "confidence_interval": {"lower": round(lower, 4), "upper": round(upper, 4)},
            "end_hp_on_win": aggregate.get("expected_end_hp_on_win"),
        }
    return {
        "minimum_survival": threshold,
        "groups": {
            family: _group_summary(weights, targets)
            for family, weights in groups.items()
        },
        "targets": list(targets.values()),
    }


def _group_summary(
    weights: Mapping[str, float],
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [(weight, targets[name]) for name, weight in weights.items() if name in targets]
    if not rows:
        return {"status": "UNAVAILABLE"}
    total = sum(weight for weight, _ in rows)
    rows = [(weight / total, row) for weight, row in rows]
    survival = sum(weight * float(row["survival"]) for weight, row in rows)
    trials = 1 / sum(weight * weight / max(1, int(row["attempts"])) for weight, row in rows)
    lower, upper = _wilson(survival * trials, trials)
    winning_mass = sum(weight * float(row["survival"]) for weight, row in rows)
    end_hp = (
        sum(
            weight * float(row["survival"]) * float(row["end_hp_on_win"] or 0)
            for weight, row in rows
        )
        / winning_mass
        if winning_mass else 0.0
    )
    worst = min(rows, key=lambda item: float(item[1]["survival"]))[1]
    statuses = {str(row["status"]) for _, row in rows}
    status = (
        "AT_RISK" if "AT_RISK" in statuses
        else "SUPPORTED" if statuses == {"SUPPORTED"}
        else "INCONCLUSIVE"
    )
    return {
        "status": status,
        "estimated_survival": round(survival, 4),
        "confidence_interval": {"lower": round(lower, 4), "upper": round(upper, 4)},
        "expected_end_hp_on_win": round(end_hp, 2),
        "worst_target": worst["target"],
    }


def _history(
    run_directory: object,
) -> tuple[set[tuple[int, int]], dict[str, tuple[str, int]], dict[int, str]]:
    try:
        path = run_directory.path / "run_history.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return set(), {}, {}
    hallways: set[tuple[int, int]] = set()
    bottles: dict[str, tuple[str, int]] = {}
    elites: dict[int, str] = {}
    pending = ""
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        before, after = event.get("before"), event.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            continue
        for snapshot in (before, after):
            run = snapshot.get("run")
            if (
                isinstance(run, Mapping)
                and str(run.get("room_type") or "").casefold() == "monsterroom"
            ):
                hallways.add((_integer(run.get("act")), _integer(run.get("floor"))))
            if elite := _elite(snapshot):
                elites[elite[0]] = elite[1]
        before_run, after_run = before.get("run"), after.get("run")
        before_relics = (
            set(before_run.get("relics") or ())
            if isinstance(before_run, Mapping)
            else set()
        )
        after_relics = (
            set(after_run.get("relics") or ())
            if isinstance(after_run, Mapping)
            else set()
        )
        added = (after_relics - before_relics) & _BOTTLES
        if added and str(after.get("screen", {}).get("type") or "") == "GRID":
            pending = next(iter(added))
            continue
        action = event.get("action")
        command = action.get("command") if isinstance(action, Mapping) else ""
        if (
            pending
            and str(before.get("screen", {}).get("type") or "") == "GRID"
            and str(command).startswith("choose ")
        ):
            try:
                index = int(str(command).split()[1])
                cards = before["screen"]["details"]["cards"]
                card = cards[index]
                bottles[pending] = (
                    _card_key(card.get("name") or card.get("id")),
                    min(1, max(0, _integer(card.get("upgrades")))),
                )
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            pending = ""
    return hallways, bottles, elites


def _elite(snapshot: Mapping[str, Any]) -> tuple[int, str] | None:
    run, combat = snapshot.get("run"), snapshot.get("combat")
    if not isinstance(run, Mapping) or not isinstance(combat, Mapping):
        return None
    act = _integer(run.get("act"))
    if (
        act not in _ELITES
        or "elite" not in str(run.get("room_type") or "").casefold()
    ):
        return None
    for row in _sequence(combat.get("monsters") or combat.get("enemies")):
        if not isinstance(row, Mapping):
            continue
        raw = row.get("name") or row.get("id")
        candidate = _ELITE_ALIASES.get(_entity_key(raw), str(raw or ""))
        match = next(
            (
                name
                for name in _ELITES[act]
                if _entity_key(name) == _entity_key(candidate)
            ),
            None,
        )
        if match:
            return act, match
    return None


def _elite_rotation(act: int, last: object) -> dict[str, Any]:
    latest = str(last) if last in _ELITES[act] else None
    return {
        "last_elite_if_known": latest,
        "next_candidates_if_known": [name for name in _ELITES[act] if name != latest],
    }


def _card_key(value: object) -> str:
    return str(value or "").casefold().removesuffix("+")


def _entity_key(value: object) -> str:
    return "".join(
        character
        for character in str(value or "").casefold()
        if character.isalnum()
    )


def _status(lower: float, upper: float, threshold: float) -> str:
    return "SUPPORTED" if lower >= threshold else "AT_RISK" if upper < threshold else "INCONCLUSIVE"


def _wilson(wins: float, trials: float) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = wins / trials
    denominator = 1 + z * z / trials
    center = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, (center - margin) / denominator), min(1.0, (center + margin) / denominator)


def _sequence(value: object) -> tuple[Any, ...]:
    return (
        tuple(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["EncounterReadiness"]
