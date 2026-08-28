"""Compare one historical card-choice journal with the current Winning Path."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.tools.winning_path import review


class ComparisonError(ValueError):
    pass


def compare(path: Path) -> str:
    rows = _read(path)
    lines: list[str] = []
    outcomes: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        request, history = _request(row, path, index)
        result = review(request)
        actions = _policy_actions(result)
        command = result.get("command")
        outcome = (
            "DIRECT_MATCH"
            if command == history["action"]
            else "DIRECT_DIFFERENT"
            if command
            else "ADVICE_COVERS_HISTORY"
            if history["action"] in actions
            else "ADVICE_EXCLUDES_HISTORY"
        )
        outcomes[outcome] += 1
        lines.extend(_render_choice(index, len(rows), row, history, result, actions, outcome))

    lines.extend(("", "Summary", "-------", f"choices: {len(rows)}"))
    lines.extend(f"{key}: {value}" for key, value in sorted(outcomes.items()))
    return "\n".join(lines) + "\n"


def _render_choice(
    index: int,
    total: int,
    raw: Mapping[str, Any],
    history: Mapping[str, Any],
    result: Mapping[str, Any],
    actions: list[str],
    outcome: str,
) -> list[str]:
    context = _mapping(raw.get("context"))
    decision = _mapping(raw.get("decision"))
    path = _mapping(result.get("winning_path"))
    state = _mapping(path.get("state"))
    run = _mapping(state.get("run"))
    resolution = _mapping(path.get("resolution"))
    targets = _mapping(path.get("target_plan"))
    offered = [str(value) for value in decision.get("offered") or ()]
    direct = result.get("command")
    current = (
        _action_text(str(direct), offered)
        if isinstance(direct, str) and direct
        else "LLM advice: " + ", ".join(_action_text(action, offered) for action in actions)
    )
    lines = [
        "",
        "=" * 78,
        f"[{index}/{total}] {raw.get('choice_id')}  Act {decision.get('act')} "
        f"Floor {decision.get('floor')}  {decision.get('kind')}",
        "Deck: " + _counts(_mapping(context.get("deck_before_counts"))),
        "Relics: " + ", ".join(map(str, context.get("relics_before_floor_rewards") or ())),
        "Offered: " + " | ".join(f"[{i}] {name}" for i, name in enumerate(offered)),
        "Historical: " + _action_text(str(history["action"]), offered),
        f"Winning Path: {current}  [{result.get('policy')}]",
        f"Comparison: {outcome}",
        f"Reason: {result.get('reason')}",
        f"State: Act {run.get('act')} Floor {run.get('floor')} | Targets: "
        + ", ".join(map(str, targets.get("targets") or ())),
        "Candidates:",
    ]
    for candidate in result.get("candidates") or ():
        row = _mapping(candidate)
        template = _mapping(row.get("template"))
        transition = _mapping(row.get("transition"))
        expert = _mapping(row.get("expert"))
        route = (
            f" route={template['route_id']}" if template.get("route_id") else ""
        )
        lines.append(
            f"  [{row.get('choice_id')}] {row.get('name')}: "
            f"{'REJECTED ' if row.get('rejected') else ''}"
            f"template={template.get('level')} "
            f"transition={transition.get('level')} "
            f"expert={expert.get('level')}{route}"
        )
    preference = _mapping(resolution.get("card_preference"))
    if preference:
        lines.append(
            "Preference scores: "
            + json.dumps(preference.get("scores"), ensure_ascii=False)
        )
        for raw_pair in preference.get("comparisons") or ():
            pair = _mapping(raw_pair)
            lines.append(
                f"  {pair.get('left')} vs {pair.get('right')}: "
                f"{pair.get('left_wins')}:{pair.get('right_wins')} "
                f"bucket={pair.get('bucket')} z={pair.get('z')}"
            )
    return lines


def _request(
    raw: Mapping[str, Any], path: Path, line: int
) -> tuple[DecisionRequest, dict[str, Any]]:
    context, decision = _mapping(raw.get("context")), _mapping(raw.get("decision"))
    run = _mapping(raw.get("run"))
    offered = tuple(map(str, decision.get("offered") or ()))
    deck_counts = _mapping(context.get("deck_before_counts"))
    if not offered or not deck_counts:
        raise ComparisonError(f"{path}:{line}: missing offered cards or deck")
    picked = decision.get("picked")
    if picked is not None:
        if str(picked) not in offered:
            raise ComparisonError(f"{path}:{line}: historical pick is not offered")
        historical = f"choose {offered.index(str(picked))}"
    elif decision.get("used_singing_bowl"):
        historical = f"choose {len(offered)}"
    elif decision.get("skipped"):
        historical = "skip"
    else:
        raise ComparisonError(f"{path}:{line}: historical action is missing")

    deck = tuple(
        {"name": str(name)}
        for name, count in deck_counts.items()
        for _ in range(int(count))
    )
    facts = {
        "class": run.get("character") or "IRONCLAD",
        "act": decision.get("act"),
        "floor": decision.get("floor"),
        "act_boss": context.get("act_boss"),
        "current_hp": context.get("hp_after_floor"),
        "max_hp": context.get("max_hp_after_floor"),
        "gold": context.get("gold_after_floor"),
        "deck": deck,
        "relics": tuple(
            {"name": str(name)}
            for name in context.get("relics_before_floor_rewards") or ()
        ),
    }
    scope_id = f"comparison:{raw.get('choice_id') or line}"
    state = GameState(
        owner_hint=AgentKind.BUILD,
        scope_id=scope_id,
        screen=ScreenState(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=offered,
            details={
                "reward_type": decision.get("kind"),
                "singing_bowl": bool(
                    decision.get("used_singing_bowl")
                    or "Singing Bowl"
                    in (context.get("relics_before_floor_rewards") or ())
                )
            },
        ),
        facts=facts,
    )
    scope = DecisionScope(AgentKind.BUILD, scope_id)
    request = DecisionRequest(
        state, scope, None, {}, ContextEntry(0, None, state, True, scope=scope)
    )
    return request, {"action": historical, "picked": picked}


def _policy_actions(result: Mapping[str, Any]) -> list[str]:
    command = result.get("command")
    if isinstance(command, str) and command:
        return [command]
    return [
        *(("skip",) if result.get("allow_skip", True) else ()),
        *(f"choose {int(value)}" for value in result.get("allowed_choice_ids") or ()),
    ]


def _action_text(action: str, offered: Sequence[str]) -> str:
    if action == "skip":
        return "skip"
    parts = action.split()
    if len(parts) == 2 and parts[0] == "choose":
        index = int(parts[1])
        if index == len(offered):
            return "singing bowl"
        if 0 <= index < len(offered):
            return f"choose {index} ({offered[index]})"
    return action


def _counts(value: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{name} x{count}" for name, count in sorted(value.items())
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    for line, text in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ComparisonError(f"{path}:{line}: expected a JSON object")
        rows.append(value)
    if not rows:
        raise ComparisonError(f"{path}: journal is empty")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    args = parser.parse_args(argv)
    print(compare(args.journal), end="")


if __name__ == "__main__":
    main()


__all__ = ["ComparisonError", "compare", "main"]
