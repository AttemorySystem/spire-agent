"""Evaluate historical and counterfactual decks at recorded combat milestones."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .evaluation import (
    _apply_fixed_delta,
    _mapping,
    _picked_card,
    _request,
    _review,
)


SCHEMA_VERSION = 1
BOSS_ENCOUNTERS = {
    "slimeboss": "Slime Boss",
    "theguardian": "The Guardian",
    "hexaghost": "Hexaghost",
    "bronzeautomaton": "Automaton",
    "thecollector": "Collector",
    "champ": "Champ",
    "awakenedone": "Awakened One",
    "timeeater": "Time Eater",
    "decadonu": "Donu And Deca",
    "corruptheart": "The Heart",
}
ENCOUNTERS_BY_SIGNATURE = {
    "acidslimelacidslimemacidslimem": "Lots Of Slimes",
    "awakenedonecultistcultist": "Awakened One",
    "byrdchosen": "Chosen And Byrds",
    "centurionhealer": "Centurion And Healer",
    "chosencultist": "Cultist And Chosen",
    "corruptheart": "The Heart",
    "decadonu": "Donu And Deca",
    "gremlinfatgremlinfatgremlinleadergremlinthiefgremlintsundere": "Gremlin Leader",
    "gremlinfatgremlinfatgremlinleadergremlinthiefgremlintsunderegremlinwarrior": "Gremlin Leader",
    "gremlinfatgremlinleadergremlinwarriorgremlinwarriorgremlinwizard": "Gremlin Leader",
    "jawwormspikeslimem": "Exordium Wildlife",
    "sentrysentrysentry": "Three Sentries",
    "fungibeastshelledparasite": "Shelled Parasite And Fungi",
    "slaverblueslaverbossslaverred": "Slavers",
    "slimeboss": "Slime Boss",
    "acidslimelacidslimemacidslimemslimebossspikeslimel": "Slime Boss",
    "spireshieldspirespear": "Shield And Spear",
    "thecollectortorchheadtorchhead": "Collector",
}
SINGLE_ENCOUNTER_ALIASES = {
    "acidslimel": "Large Slime",
    "bookofstabbing": "Book Of Stabbing",
    "bronzeautomaton": "Automaton",
    "champ": "Champ",
    "gianthead": "Giant Head",
    "gremlinnob": "Gremlin Nob",
    "hexaghost": "Hexaghost",
    "maw": "Maw",
    "serpent": "Spire Growth",
    "shelledparasite": "Shell Parasite",
    "snakeplant": "Snake Plant",
    "spikeslimel": "Large Slime",
    "thecollector": "Collector",
    "theguardian": "The Guardian",
    "timeeater": "Time Eater",
}
CARD_ID_ALIASES = {
    "Ascender's Bane": "AscendersBane",
    "Curse of the Bell": "CurseOfTheBell",
    "Defend": "Defend_R",
    "Strike": "Strike_R",
}
BOTTLES = {
    "Bottled Flame": "ATTACK",
    "Bottled Lightning": "SKILL",
    "Bottled Tornado": "POWER",
}


class CombatEvaluationError(ValueError):
    pass


def prepare(
    dataset_dir: Path,
    source_root: Path,
    output_dir: Path,
    *,
    fallback: str = "history",
    run_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic deck timelines and recorded combat checkpoints."""

    if fallback not in {"history", "skip"}:
        raise CombatEvaluationError(f"unknown fallback {fallback!r}")
    cases = _read_jsonl(dataset_dir / "cases.jsonl")
    runs = _read_jsonl(dataset_dir / "runs.jsonl")
    by_id = {str(case["case_id"]): case for case in cases}
    selected = [run for run in runs if not run_ids or str(run["run_id"]) in run_ids]
    missing = (run_ids or set()) - {str(run["run_id"]) for run in selected}
    if missing:
        raise CombatEvaluationError(f"unknown run ids: {', '.join(sorted(missing))}")

    choice_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run in selected:
        run_id = str(run["run_id"])
        run_cases = [by_id[str(case_id)] for case_id in run.get("case_ids") or ()]
        if not run_cases:
            quarantined.append({"run_id": run_id, "kind": "EMPTY_RUN"})
            continue
        steps, trajectory_warnings = _counterfactual_steps(
            run, run_cases, fallback=fallback
        )
        choice_rows.extend(steps)
        found, issues = discover_checkpoints(source_root / run_id, run)
        quarantined.extend({"run_id": run_id, **issue} for issue in issues)
        if _mapping(run.get("outcome")).get("status") == "INCOMPLETE":
            quarantined.append({"run_id": run_id, "kind": "INCOMPLETE_RUN"})
        run_checkpoints = []
        for checkpoint in found:
            prepared = _prepare_checkpoint(checkpoint, steps)
            checkpoints.append(prepared)
            run_checkpoints.append(prepared["checkpoint_id"])
            if not prepared.get("encounter"):
                quarantined.append(
                    {
                        "run_id": run_id,
                        "kind": "UNKNOWN_ENCOUNTER",
                        "checkpoint_id": prepared["checkpoint_id"],
                        "enemies": prepared["enemies"],
                    }
                )
            if prepared.get("unsupported"):
                quarantined.append(
                    {
                        "run_id": run_id,
                        "kind": "UNSUPPORTED_CHECKPOINT",
                        "checkpoint_id": prepared["checkpoint_id"],
                        "reason": prepared["unsupported"],
                    }
                )
        if not found:
            quarantined.append({"run_id": run_id, "kind": "NO_CHECKPOINTS"})
        run_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "outcome": run.get("outcome"),
                "choice_count": len(steps),
                "changed_choices": sum(row["changed"] for row in steps),
                "direct_choices": sum(row["direct"] for row in steps),
                "fallback_choices": sum(not row["direct"] for row in steps),
                "checkpoint_ids": run_checkpoints,
                "trajectory_warnings": trajectory_warnings,
            }
        )

    summary = _inventory(run_rows, checkpoints, quarantined, fallback)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "choices.jsonl", choice_rows)
    _write_jsonl(output_dir / "runs.jsonl", run_rows)
    _write_jsonl(output_dir / "checkpoints.jsonl", checkpoints)
    _write_jsonl(output_dir / "quarantined.jsonl", quarantined)
    _write_json(output_dir / "inventory.json", summary)
    (output_dir / "report.md").write_text(
        _render_report(summary, checkpoints, []), encoding="utf-8"
    )
    return summary


def evaluate_battles(
    output_dir: Path,
    binary: Path,
    *,
    simulations: int,
    worlds: int,
    world_start: int = 0,
    max_time_ms: int = 100,
    max_decisions: int = 300,
    timeout: int = 600,
    clean_only: bool = True,
    jobs: int = 1,
) -> dict[str, Any]:
    """Run paired independent battles for every prepared checkpoint."""

    if min(simulations, worlds, max_time_ms, max_decisions, timeout, jobs) <= 0:
        raise CombatEvaluationError("battle budgets must be positive")
    if not binary.is_file():
        raise CombatEvaluationError(f"combat evaluator not found: {binary}")
    checkpoints = _read_jsonl(output_dir / "checkpoints.jsonl")
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    binary_hash = _sha256(binary)
    config = {
        "simulations_per_decision": simulations,
        "rng_worlds": worlds,
        "world_start": world_start,
        "max_time_ms_per_decision": max_time_ms,
        "max_decisions_per_battle": max_decisions,
        "potions_allowed": False,
        "binary_sha256": binary_hash,
    }

    def evaluate(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        if clean_only and not checkpoint["clean"]:
            return _skipped_result(checkpoint, "COUNTERFACTUAL_WARNINGS")
        if not checkpoint.get("encounter"):
            return _skipped_result(checkpoint, "UNKNOWN_ENCOUNTER")
        if checkpoint.get("unsupported"):
            return _skipped_result(checkpoint, str(checkpoint["unsupported"]))
        sides = {}
        for side in ("historical", "candidate"):
            spec = _battle_spec(checkpoint, side)
            sides[side] = _run_cached(
                binary,
                spec,
                config,
                cache,
                timeout=timeout,
            )
        return _paired_result(checkpoint, sides, config)

    if jobs == 1:
        results = list(map(evaluate, checkpoints))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(evaluate, checkpoints))

    _write_jsonl(output_dir / "combat_results.jsonl", results)
    _write_jsonl(
        output_dir / "regressions.jsonl",
        [row for row in results if row.get("classification") == "REGRESSED"],
    )
    _write_jsonl(
        output_dir / "improvements.jsonl",
        [row for row in results if row.get("classification") == "IMPROVED"],
    )
    inventory = json.loads((output_dir / "inventory.json").read_text("utf-8"))
    summary = _combat_summary(inventory, results, config)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        _render_report(summary, checkpoints, results), encoding="utf-8"
    )
    return summary


def _counterfactual_steps(
    run: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    fallback: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deck = Counter(_mapping(cases[0]["state"]).get("deck_counts") or {})
    transitions = {
        str(row.get("from_case_id")): row
        for row in run.get("transitions") or ()
        if isinstance(row, Mapping)
    }
    shared: dict[str, Any] = {}
    steps = []
    all_warnings = []
    for case in cases:
        outcome = _review(_request(case, deck, shared))
        command = outcome.get("command")
        direct = isinstance(command, str) and bool(command)
        historical_action = str(_mapping(case.get("observed_action")).get("action"))
        action = str(command) if direct else historical_action if fallback == "history" else "skip"
        offered = list(_mapping(case.get("reward")).get("offered") or ())
        historical_card = _mapping(case.get("observed_action")).get("picked")
        candidate_card = _picked_card(action, offered)
        before = dict(sorted(deck.items()))
        if candidate_card is not None:
            deck[candidate_card] += 1
        warnings = _apply_fixed_delta(deck, transitions.get(str(case["case_id"])))
        all_warnings.extend({"case_id": case["case_id"], **row} for row in warnings)
        steps.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run["run_id"],
                "case_id": case["case_id"],
                "sequence": case["sequence"],
                "act": case["state"]["act"],
                "floor": case["state"]["floor"],
                "offered": offered,
                "historical_action": historical_action,
                "historical_card": historical_card,
                "candidate_action": action,
                "candidate_card": candidate_card,
                "direct": direct,
                "fallback": None if direct else fallback,
                "changed": historical_card != candidate_card,
                "action_changed": historical_action != action,
                "policy": {
                    "mode": outcome.get("mode"),
                    "policy": outcome.get("policy"),
                    "command": outcome.get("command"),
                    "reason": outcome.get("reason"),
                },
                "deck_before": before,
                "deck_after": dict(sorted(deck.items())),
                "warnings": warnings,
            }
        )
    return steps, all_warnings


def discover_checkpoints(
    run_dir: Path, run: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract first-turn boss and fatal encounter states from one run."""

    if not run_dir.is_dir():
        return [], [{"kind": "MISSING_RUN_DIRECTORY", "path": str(run_dir)}]
    starts = []
    issues = []
    seen = set()
    outcome = _mapping(run.get("outcome"))
    full_scan = (
        outcome.get("status") == "DEFEAT"
        and int(outcome.get("floor_reached") or 0) == 51
    )
    for path in _turn_one_paths(run_dir, full=full_scan):
        try:
            raw = json.loads(path.read_text("utf-8"))
            state = _mapping(raw.get("game_state"))
            combat = _mapping(state.get("combat_state"))
            monsters = [row for row in combat.get("monsters") or () if isinstance(row, Mapping)]
            if combat.get("turn") != 1 or not monsters or int(state.get("current_hp") or 0) <= 0:
                continue
            signature = _monster_signature(monsters)
            key = (
                path.name.split("-", 1)[0],
                int(state.get("act") or 0),
                int(state.get("floor") or 0),
            )
            if key in seen:
                continue
            seen.add(key)
            starts.append(
                {
                    "path": path,
                    "order": _natural_key(path.name),
                    "state": state,
                    "signature": signature,
                    "enemies": [str(row.get("name") or row.get("id") or "") for row in monsters],
                }
            )
        except (OSError, ValueError, TypeError) as error:
            issues.append({"kind": "INVALID_COMBAT_SNAPSHOT", "path": str(path), "error": str(error)})

    fatal = None
    if outcome.get("status") == "DEFEAT":
        floor = outcome.get("floor_reached")
        candidates = [row for row in starts if row["state"].get("floor") == floor]
        killed_signature = _text_signature(str(outcome.get("killed_by") or ""))
        matching = [row for row in candidates if _same_encounter(row["signature"], killed_signature)]
        fatal = (matching or candidates)[-1] if (matching or candidates) else None
        if fatal is None:
            issues.append(
                {
                    "kind": "MISSING_FATAL_START",
                    "floor": floor,
                    "killed_by": outcome.get("killed_by"),
                }
            )

    result = []
    for row in sorted(starts, key=lambda value: value["order"]):
        state = row["state"]
        act = int(state.get("act") or 0)
        floor = int(state.get("floor") or 0)
        marker = run_dir / f"post-combat-act{act}-floor{floor}-combat-agent"
        room_type = str(state.get("room_type") or "")
        kind = "FATAL_ENCOUNTER" if row is fatal else "PASSED_BOSS" if "Boss" in room_type and marker.exists() else None
        if kind is None:
            continue
        encounter = encounter_name(
            row["enemies"], state.get("act_boss"), kind, signature=row["signature"]
        )
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_id": f"{run['run_id']}:a{act}:f{floor}:{kind.lower()}",
                "run_id": str(run["run_id"]),
                "kind": kind,
                "act": act,
                "floor": floor,
                "encounter": encounter,
                "enemies": row["enemies"],
                "source": path_relative(row["path"], run_dir.parent),
                "game_state": _combat_game_state(state),
            }
        )
    return result, issues


def encounter_name(
    enemies: Sequence[str],
    act_boss: object,
    kind: str,
    *,
    signature: str | None = None,
) -> str | None:
    signature = signature or _text_signature(",".join(enemies))
    mapped = _encounter_from_signature(signature)
    if mapped:
        return mapped
    if len(enemies) == 1:
        if signature in SINGLE_ENCOUNTER_ALIASES:
            return SINGLE_ENCOUNTER_ALIASES[signature]
        display = str(enemies[0]).strip()
        return display or None
    if kind == "PASSED_BOSS":
        return BOSS_ENCOUNTERS.get(_normalized(str(act_boss or "")))
    return None


def _encounter_from_signature(signature: str) -> str | None:
    mapped = ENCOUNTERS_BY_SIGNATURE.get(signature)
    if mapped:
        return mapped
    if "gremlinleader" in signature:
        return "Gremlin Leader"
    if "acidslimel" in signature and signature.count("acidslimem") >= 2:
        return "Lots Of Slimes"
    return None


def _prepare_checkpoint(
    checkpoint: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    state = deepcopy(dict(_mapping(checkpoint.get("game_state"))))
    historical = _recover_bottled_cards(state)
    candidate, warnings, changed = apply_reward_changes(
        historical,
        [
            row
            for row in steps
            if (int(row["act"]), int(row["floor"]))
            < (int(checkpoint["act"]), int(checkpoint["floor"]))
        ],
    )
    same_floor = [
        row["case_id"]
        for row in steps
        if (int(row["act"]), int(row["floor"]))
        == (int(checkpoint["act"]), int(checkpoint["floor"]))
    ]
    if same_floor:
        warnings.append({"kind": "SAME_FLOOR_CHOICES_EXCLUDED", "case_ids": same_floor})
    unsupported = (
        _unsupported_deck(historical)
        or _unsupported_deck(candidate)
        or _bottle_problem(state, historical)
        or _bottle_problem(state, candidate)
    )
    blocking = {"BOTTLED_CARD_REMOVED", "HISTORICAL_PICK_NOT_PRESENT"}
    result = deepcopy(dict(checkpoint))
    result.update(
        {
            "historical_deck": historical,
            "candidate_deck": candidate,
            "historical_deck_size": len(historical),
            "candidate_deck_size": len(candidate),
            "changed_choice_ids": changed,
            "warnings": warnings,
            "clean": not any(row.get("kind") in blocking for row in warnings),
            "unsupported": unsupported,
        }
    )
    return result


def apply_reward_changes(
    historical_deck: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Apply card-reward differences to an exact historical checkpoint deck."""

    deck = deepcopy([dict(card) for card in historical_deck])
    warnings = []
    changed = []
    for step in steps:
        old = step.get("historical_card")
        new = step.get("candidate_card")
        if old == new:
            continue
        changed.append(str(step["case_id"]))
        if old:
            index, exact = _find_card(deck, str(old))
            if index is None:
                warnings.append(
                    {
                        "kind": "HISTORICAL_PICK_NOT_PRESENT",
                        "case_id": step["case_id"],
                        "card": old,
                    }
                )
            else:
                removed = deck.pop(index)
                if removed.get("bottled"):
                    warnings.append(
                        {
                            "kind": "BOTTLED_CARD_REMOVED",
                            "case_id": step["case_id"],
                            "card": old,
                        }
                    )
                if not exact:
                    warnings.append(
                        {
                            "kind": "REMOVED_UPGRADED_HISTORICAL_PICK",
                            "case_id": step["case_id"],
                            "card": old,
                            "actual": _card_name(removed),
                        }
                    )
        if new:
            deck.append(_new_card(str(new)))
    return deck, warnings, changed


def _find_card(deck: Sequence[Mapping[str, Any]], name: str) -> tuple[int | None, bool]:
    base, upgrades = _split_card(name)
    same_base = []
    for index, card in enumerate(deck):
        card_base, card_upgrades = _split_card(_card_name(card))
        if card_base != base:
            continue
        if card_upgrades == upgrades:
            return index, True
        same_base.append((card_upgrades, index))
    return (max(same_base)[1], False) if same_base else (None, False)


def _combat_game_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": state.get("seed"),
        "ascension_level": state.get("ascension_level"),
        "act": state.get("act"),
        "floor": state.get("floor"),
        "current_hp": state.get("current_hp"),
        "max_hp": state.get("max_hp"),
        "gold": state.get("gold"),
        "class": state.get("class"),
        "deck": deepcopy(list(state.get("deck") or ())),
        "relics": deepcopy(list(state.get("relics") or ())),
        "potions": deepcopy(list(state.get("potions") or ())),
        "combat_state": deepcopy(dict(_mapping(state.get("combat_state")))),
    }


def _recover_bottled_cards(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_deck = [dict(card) for card in state.get("deck") or () if isinstance(card, Mapping)]
    deck = [_simple_card(card) for card in raw_deck]
    hand = [card for card in _mapping(state.get("combat_state")).get("hand") or () if isinstance(card, Mapping)]
    relics = {str(row.get("name") or row.get("id") or "") for row in state.get("relics") or () if isinstance(row, Mapping)}
    for relic, card_type in BOTTLES.items():
        if relic not in relics:
            continue
        for hand_card in hand:
            if str(hand_card.get("type") or "") != card_type:
                continue
            uuid = hand_card.get("uuid")
            match = next(
                (
                    index
                    for index, raw in enumerate(raw_deck)
                    if uuid and raw.get("uuid") == uuid
                ),
                None,
            )
            if match is not None:
                deck[match]["bottled"] = True
                break
    return deck


def _simple_card(card: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "id": str(card.get("id") or card.get("name") or ""),
        "upgrades": int(card.get("upgrades") or 0),
    }
    if "misc" in card:
        result["misc"] = card["misc"]
    if card.get("bottled"):
        result["bottled"] = True
    return result


def _new_card(name: str) -> dict[str, Any]:
    base, upgrades = _split_card(name)
    return {"id": CARD_ID_ALIASES.get(base, base), "upgrades": upgrades}


def _card_name(card: Mapping[str, Any]) -> str:
    base = str(card.get("name") or card.get("id") or "")
    base = {
        "AscendersBane": "Ascender's Bane",
        "CurseOfTheBell": "Curse of the Bell",
        "Defend_R": "Defend",
        "Strike_R": "Strike",
    }.get(base, base)
    return base + "+" * int(card.get("upgrades") or 0)


def _split_card(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.*?)(?:\+(\d+)|([+]+))?", name.strip())
    if not match:
        return name.strip(), 0
    return match.group(1), int(match.group(2)) if match.group(2) else len(match.group(3) or "")


def _unsupported_deck(deck: Sequence[Mapping[str, Any]]) -> str | None:
    for card in deck:
        if int(card.get("upgrades") or 0) > 1:
            return f"MULTI_UPGRADE_CARD:{_card_name(card)}"
        if (
            _normalized(str(card.get("id") or ""))
            in {"geneticalgorithm", "ritualdagger"}
            and "misc" not in card
        ):
            return f"MISSING_CARD_MISC:{_card_name(card)}"
    bottled_types = 0
    for card in deck:
        bottled_types += bool(card.get("bottled"))
    if bottled_types > 3:
        return "INVALID_BOTTLED_CARDS"
    return None


def _bottle_problem(
    state: Mapping[str, Any], deck: Sequence[Mapping[str, Any]]
) -> str | None:
    relics = {
        str(row.get("name") or row.get("id") or "")
        for row in state.get("relics") or ()
        if isinstance(row, Mapping)
    }
    expected = sum(name in relics for name in BOTTLES)
    actual = sum(bool(card.get("bottled")) for card in deck)
    return "UNIDENTIFIED_BOTTLED_CARD" if expected != actual else None


def _battle_spec(checkpoint: Mapping[str, Any], side: str) -> dict[str, Any]:
    state = deepcopy(dict(_mapping(checkpoint.get("game_state"))))
    state.pop("combat_state", None)
    state["deck"] = deepcopy(list(checkpoint[f"{side}_deck"]))
    return {
        "game_state": state,
        "candidates": [],
        "targets": [checkpoint["encounter"]],
    }


def _run_cached(
    binary: Path,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    cache_dir: Path,
    *,
    timeout: int,
) -> dict[str, Any]:
    key = hashlib.sha256(
        json.dumps({"spec": spec, "config": config}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        json.dump(spec, handle, separators=(",", ":"))
        handle.flush()
        command = [
            str(binary),
            "--battle-eval",
            handle.name,
            str(config["simulations_per_decision"]),
            str(config["world_start"]),
            str(config["rng_worlds"]),
            str(config["max_time_ms_per_decision"]),
            str(config["max_decisions_per_battle"]),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result = {"error": "COMBAT_EVALUATOR_TIMEOUT", "timeout": timeout}
        else:
            if completed.returncode != 0:
                result = {
                    "error": "COMBAT_EVALUATOR_FAILED",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                }
            else:
                try:
                    result = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    result = {
                        "error": "INVALID_COMBAT_EVALUATOR_OUTPUT",
                        "detail": str(error),
                        "stdout": completed.stdout[-4000:],
                    }
    _write_json(path, result)
    return result


def _paired_result(
    checkpoint: Mapping[str, Any],
    sides: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if any(side.get("error") for side in sides.values()):
        return {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "run_id": checkpoint["run_id"],
            "kind": checkpoint["kind"],
            "encounter": checkpoint["encounter"],
            "status": "ERROR",
            "sides": sides,
        }
    historical = sides["historical"]["targets"][0]
    candidate = sides["candidate"]["targets"][0]
    historical_trials = {int(row["world"]): row for row in historical["trials"]}
    candidate_trials = {int(row["world"]): row for row in candidate["trials"]}
    paired = Counter()
    trials = []
    differences = []
    for world in sorted(historical_trials.keys() & candidate_trials.keys()):
        old_trial = historical_trials[world]
        new_trial = candidate_trials[world]
        left = _trial_code(old_trial)
        right = _trial_code(new_trial)
        paired[left + right] += 1
        differences.append(int(bool(new_trial["won"])) - int(bool(old_trial["won"])))
        trials.append(
            {
                "world": world,
                "historical": _compact_trial(old_trial),
                "candidate": _compact_trial(new_trial),
            }
        )
    old_rate = float(historical["aggregate"]["win_rate"])
    new_rate = float(candidate["aggregate"]["win_rate"])
    interval = _paired_interval(differences)
    incomplete = int(historical["aggregate"].get("incomplete") or 0) + int(
        candidate["aggregate"].get("incomplete") or 0
    )
    classification = (
        "INCOMPLETE_TRIALS"
        if incomplete
        else "SCREEN_ONLY"
        if len(differences) < 16
        else "IMPROVED"
        if interval[0] > 0
        else "REGRESSED"
        if interval[1] < 0
        else "INCONCLUSIVE"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "run_id": checkpoint["run_id"],
        "kind": checkpoint["kind"],
        "act": checkpoint["act"],
        "floor": checkpoint["floor"],
        "encounter": checkpoint["encounter"],
        "status": "OK",
        "historical": historical["aggregate"],
        "candidate": candidate["aggregate"],
        "win_rate_delta": round(new_rate - old_rate, 10),
        "paired_delta_95_interval": [round(value, 10) for value in interval],
        "classification": classification,
        "baseline_calibration": (
            "BASELINE_SIGNAL"
            if checkpoint["kind"] != "PASSED_BOSS" or historical["aggregate"]["wins"]
            else "MODEL_MISMATCH_OR_SATURATION"
        ),
        "paired_outcomes": dict(sorted(paired.items())),
        "trials": trials,
        "changed_choice_ids": checkpoint.get("changed_choice_ids") or [],
        "config": dict(config),
    }


def _trial_code(trial: Mapping[str, Any]) -> str:
    return "W" if trial.get("won") else "L" if trial.get("completed") else "I"


def _compact_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "won": bool(trial.get("won")),
        "completed": bool(trial.get("completed")),
        "ending_hp": trial.get("ending_hp"),
        "turns": trial.get("turns"),
        "stop_reason": trial.get("stop_reason"),
    }


def _paired_interval(differences: Sequence[int]) -> tuple[float, float]:
    if not differences:
        return 0.0, 0.0
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        return -1.0, 1.0
    variance = sum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    margin = 1.96 * math.sqrt(variance / len(differences))
    return max(-1.0, mean - margin), min(1.0, mean + margin)


def _skipped_result(checkpoint: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "run_id": checkpoint["run_id"],
        "kind": checkpoint["kind"],
        "encounter": checkpoint.get("encounter"),
        "status": "SKIPPED",
        "reason": reason,
    }


def _inventory(
    runs: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    quarantined: Sequence[Mapping[str, Any]],
    fallback: str,
) -> dict[str, Any]:
    kinds = Counter(str(row["kind"]) for row in checkpoints)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "RECORDED_STATE_DECK_SWAP",
        "policy": "winning_path",
        "advice_fallback": fallback,
        "runs": len(runs),
        "complete_runs": sum(_mapping(row.get("outcome")).get("status") != "INCOMPLETE" for row in runs),
        "choices": sum(int(row["choice_count"]) for row in runs),
        "changed_choices": sum(int(row["changed_choices"]) for row in runs),
        "direct_choices": sum(int(row["direct_choices"]) for row in runs),
        "fallback_choices": sum(int(row["fallback_choices"]) for row in runs),
        "checkpoints": len(checkpoints),
        "checkpoint_kinds": dict(sorted(kinds.items())),
        "clean_checkpoints": sum(
            bool(row["clean"])
            and not row.get("unsupported")
            and bool(row.get("encounter"))
            for row in checkpoints
        ),
        "quarantined": len(quarantined),
        "assumptions": [
            "Only card-reward decisions are changed.",
            "Recorded HP, max HP, relics, potions, route, and encounter are fixed.",
            "Same-floor reward choices are excluded from a combat checkpoint.",
            "Combat starts from a fresh shuffle; recorded hand and draw pile are not reused.",
            "Policy evidence may include evaluated runs; this is a regression benchmark, not a holdout claim.",
        ],
    }


def _combat_summary(
    inventory: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "OK"]
    by_kind = {}
    for kind in ("PASSED_BOSS", "FATAL_ENCOUNTER"):
        rows = [row for row in ok if row.get("kind") == kind]
        deltas = [float(row["win_rate_delta"]) for row in rows]
        by_kind[kind] = {
            "cases": len(rows),
            "improved": sum(value > 0 for value in deltas),
            "regressed": sum(value < 0 for value in deltas),
            "unchanged": sum(value == 0 for value in deltas),
            "mean_win_rate_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
            "classifications": dict(
                sorted(Counter(str(row["classification"]) for row in rows).items())
            ),
        }
    return {
        **dict(inventory),
        "combat_config": dict(config),
        "combat_results": len(results),
        "evaluated": len(ok),
        "skipped": sum(row.get("status") == "SKIPPED" for row in results),
        "errors": sum(row.get("status") == "ERROR" for row in results),
        "by_kind": by_kind,
    }


def _render_report(
    summary: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Winning Path combat evaluation",
        "",
        f"- Runs: {summary.get('runs', 0)}",
        f"- Choices: {summary.get('choices', 0)}",
        f"- Changed choices: {summary.get('changed_choices', 0)}",
        f"- Direct choices: {summary.get('direct_choices', 0)}",
        f"- Historical/skip fallbacks: {summary.get('fallback_choices', 0)}",
        f"- Checkpoints: {summary.get('checkpoints', 0)}",
        f"- Clean checkpoints: {summary.get('clean_checkpoints', 0)}",
        f"- Quarantined issues: {summary.get('quarantined', 0)}",
        "",
    ]
    if not results:
        lines.extend(("Combat simulation has not been run.", ""))
        return "\n".join(lines)
    lines.extend(
        (
            "| Run | Kind | Act/Floor | Encounter | Historical | Candidate | Delta | Result |",
            "|---|---|---:|---|---:|---:|---:|---|",
        )
    )
    by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    for row in results:
        checkpoint = by_id[str(row["checkpoint_id"])]
        if row.get("status") == "OK":
            old = f"{float(row['historical']['win_rate']):.3f}"
            new = f"{float(row['candidate']['win_rate']):.3f}"
            delta = f"{float(row['win_rate_delta']):+.3f}"
        else:
            old = new = delta = "-"
        lines.append(
            f"| {row['run_id']} | {row['kind']} | {checkpoint['act']}/{checkpoint['floor']} "
            f"| {row.get('encounter') or 'UNKNOWN'} | {old} | {new} | {delta} "
            f"| {row.get('classification') or row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _turn_one_paths(run_dir: Path, *, full: bool = False) -> list[Path]:
    try:
        command = [
            "rg",
            "-l",
            r'"turn"\s*:\s*1(?:\D|$)',
        ]
        if not full:
            command.extend(
                [
                    "-g",
                    "*-0.json",
                    "-g",
                    "*-1.json",
                    "-g",
                    "*-2.json",
                ]
            )
        command.append(str(run_dir))
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        paths = [Path(line) for line in completed.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        paths = list(run_dir.glob("*.json"))
    return sorted(
        [path for path in paths if re.match(r"^\d+-\d+\.json$", path.name)],
        key=lambda path: _natural_key(path.name),
    )


def _monster_signature(monsters: Sequence[Mapping[str, Any]]) -> str:
    return _text_signature(",".join(str(row.get("id") or row.get("name") or "") for row in monsters))


def _text_signature(value: str) -> str:
    return "".join(sorted(_normalized(part) for part in value.split(",") if _normalized(part)))


def _same_encounter(left: str, right: str) -> bool:
    return left == right or (left and right and (left in right or right in left))


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def path_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise CombatEvaluationError(f"missing input: {path}")
    result = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise CombatEvaluationError(f"{path}:{line_number}: expected object")
        result.append(value)
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("src/spire_agent/tools/winning_path/data/evaluation")
    parser.add_argument("--dataset", type=Path, default=root)
    parser.add_argument("--source", type=Path, default=Path("../remote"))
    parser.add_argument("--output", type=Path, default=root / "current_policy/combat")
    parser.add_argument("--fallback", choices=("history", "skip"), default="history")
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="reuse checkpoints already present in --output",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("3rd/sts_lightspeed/build/card-reward-eval"),
    )
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--world-start", type=int, default=0)
    parser.add_argument("--max-time-ms", type=int, default=100)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--include-warnings", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_only and args.simulate_only:
        raise CombatEvaluationError(
            "--prepare-only and --simulate-only are mutually exclusive"
        )
    if args.simulate_only:
        summary = json.loads((args.output / "inventory.json").read_text("utf-8"))
    else:
        summary = prepare(
            args.dataset,
            args.source,
            args.output,
            fallback=args.fallback,
            run_ids=set(args.run_ids) if args.run_ids else None,
        )
        print(
            f"Prepared {summary['checkpoints']} checkpoints from {summary['runs']} runs "
            f"({summary['clean_checkpoints']} clean)."
        )
    if args.prepare_only:
        return
    result = evaluate_battles(
        args.output,
        args.binary,
        simulations=args.simulations,
        worlds=args.worlds,
        world_start=args.world_start,
        max_time_ms=args.max_time_ms,
        max_decisions=args.max_decisions,
        timeout=args.timeout,
        clean_only=not args.include_warnings,
        jobs=args.jobs,
    )
    print(
        f"Evaluated {result['evaluated']} checkpoints; "
        f"report: {args.output / 'report.md'}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CombatEvaluationError",
    "apply_reward_changes",
    "discover_checkpoints",
    "encounter_name",
    "evaluate_battles",
    "main",
    "prepare",
]
