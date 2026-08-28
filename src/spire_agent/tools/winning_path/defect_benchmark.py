"""Measure Defect determinism against the public expert archive."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any
import zipfile

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)

from .defect_data import _card_names, _reward_rows
from .picker import WinningPathCardPicker


def benchmark_defect_archive(
    archive_path: Path,
    cards_path: Path,
    case_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    names, picker = _card_names(cards_path), WinningPathCardPicker("DEFECT")
    total = deterministic = matched = advice_covered = 0
    policies: Counter[str] = Counter()
    acts: dict[int, list[int]] = {act: [0, 0, 0] for act in range(1, 5)}
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".run"):
                continue
            run = json.loads(archive.read(member))
            if run.get("character_chosen") != "DEFECT" or int(
                run.get("ascension_level") or 0
            ) != 20:
                continue
            deck = ["Strike_B"] * 4 + ["Defend_B"] * 4 + [
                "Zap", "Dualcast", "AscendersBane"
            ]
            bosses = _bosses(run)
            for act, floor, offered, picked in _reward_rows(run, names):
                result = picker.review(
                    _request(
                        total, act, floor, offered, picked, deck, bosses.get(act, "")
                    )
                )
                expected = _historical_command(offered, picked)
                total += 1
                acts[act][0] += 1
                policies[str(result["policy"])] += 1
                if result["mode"] == "DIRECT":
                    deterministic += 1
                    acts[act][1] += 1
                    if result["command"] == expected:
                        matched += 1
                        acts[act][2] += 1
                elif _advice_covers(result, offered, picked):
                    advice_covered += 1
                if case_rows is not None:
                    case_rows.append(
                        {
                            "index": total - 1,
                            "act": act,
                            "floor": floor,
                            "offered": offered,
                            "historical": expected,
                            "mode": result["mode"],
                            "command": result.get("command"),
                            "policy": result["policy"],
                            "templates": [
                                row.get("template") for row in result["candidates"]
                            ],
                        }
                    )
                if picked not in {"SKIP", "Singing Bowl"}:
                    deck.append(picked)
    return {
        "schema_version": 1,
        "scope": "DEFECT A20 expert standard rewards",
        "limitations": [
            "The archive has no exact deck-before snapshot.",
            "The benchmark replays reward picks but not event, shop, removal, "
            "or transform mutations.",
            "Agreement measures historical alignment, not optimality.",
        ],
        "choices": total,
        "deterministic": deterministic,
        "deterministic_coverage": _rate(deterministic, total),
        "deterministic_matches": matched,
        "deterministic_agreement": _rate(matched, deterministic),
        "overall_direct_match": _rate(matched, total),
        "advice_covering_history": advice_covered,
        "by_act": {
            str(act): {
                "choices": row[0],
                "deterministic": row[1],
                "matches": row[2],
            }
            for act, row in acts.items()
        },
        "policies": dict(sorted(policies.items())),
    }


def _request(
    index: int,
    act: int,
    floor: int,
    offered: list[str],
    picked: str,
    deck: list[str],
    boss: str,
) -> DecisionRequest:
    screen = ScreenState(
        "CARD_REWARD",
        commands=("choose", "skip"),
        choices=tuple(offered),
        details={
            "cards": tuple({"name": card} for card in offered),
            "singing_bowl": picked == "Singing Bowl",
        },
    )
    state = GameState(
        AgentKind.BUILD,
        f"defect-expert:{index}",
        screen,
        facts={
            "seed": 0,
            "class": "DEFECT",
            "act": act,
            "floor": floor,
            "ascension_level": 20,
            "act_boss": boss,
            "current_hp": 50,
            "max_hp": 75,
            "gold": 0,
            "room_type": "MonsterRoomBoss" if floor in (16, 33) else "MonsterRoom",
            "deck": tuple({"name": card} for card in deck),
            "relics": ({"name": "Cracked Core"},),
            "potions": ({"name": "Potion Slot"},),
        },
    )
    scope = DecisionScope(AgentKind.BUILD, state.scope_id)
    return DecisionRequest(
        state, scope, None, {}, ContextEntry(0, None, state, True, scope=scope)
    )


def _bosses(run: Mapping[str, Any]) -> dict[int, str]:
    floors = {16: 1, 33: 2, 50: 3}
    return {
        floors[floor]: str(row.get("enemies") or "")
        for row in run.get("damage_taken") or ()
        if isinstance(row, Mapping)
        and (floor := int(row.get("floor") or 0)) in floors
    }


def _historical_command(offered: list[str], picked: str) -> str:
    if picked == "SKIP":
        return "skip"
    if picked == "Singing Bowl":
        return f"choose {len(offered)}"
    return f"choose {offered.index(picked)}"


def _advice_covers(result: Mapping[str, Any], offered: list[str], picked: str) -> bool:
    if picked == "SKIP":
        return bool(result.get("allow_skip"))
    choice_id = len(offered) if picked == "Singing Bowl" else offered.index(picked)
    return choice_id in result.get("allowed_choice_ids", ())


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases-output", type=Path)
    args = parser.parse_args(argv)
    cases: list[dict[str, Any]] | None = [] if args.cases_output else None
    report = benchmark_defect_archive(args.archive, args.cards, cases)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if args.cases_output:
        args.cases_output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases or ()),
            encoding="utf-8",
        )
    print(text, end="")


if __name__ == "__main__":
    main()


__all__ = ["benchmark_defect_archive", "main"]
