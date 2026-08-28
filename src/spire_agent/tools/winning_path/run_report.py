"""Generate one detailed Winning Path review from a historical run directory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .combat_eval import evaluate_battles, prepare


class RunReportError(ValueError):
    pass


def review_run(
    run_dir: Path,
    dataset: Path,
    output: Path,
    binary: Path,
    *,
    fallback: str = "history",
    simulations: int = 500,
    worlds: int = 16,
    max_time_ms: int = 100,
    max_decisions: int = 300,
    timeout: int = 600,
    include_warnings: bool = False,
) -> Path:
    run_dir = run_dir.resolve()
    if not (run_dir / "card_choices.jsonl").is_file():
        raise RunReportError(f"not a historical run directory: {run_dir}")
    run_id = run_dir.name
    prepare(
        dataset,
        run_dir.parent,
        output,
        fallback=fallback,
        run_ids={run_id},
    )
    evaluate_battles(
        output,
        binary,
        simulations=simulations,
        worlds=worlds,
        max_time_ms=max_time_ms,
        max_decisions=max_decisions,
        timeout=timeout,
        clean_only=not include_warnings,
    )
    runs = _rows(output / "runs.jsonl")
    report = render_report(
        runs[0],
        _rows(output / "choices.jsonl"),
        _rows(output / "checkpoints.jsonl"),
        _rows(output / "combat_results.jsonl"),
    )
    path = output / "run_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def render_report(
    run: Mapping[str, Any],
    choices: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> str:
    outcome = _mapping(run.get("outcome"))
    lines = [
        f"# Winning Path review: run {run.get('run_id')}",
        "",
        f"- Historical outcome: {outcome.get('status')}",
        f"- Floor reached: {outcome.get('floor_reached')}",
        f"- Killed by: {outcome.get('killed_by') or '-'}",
        f"- Card rewards: {len(choices)}",
        f"- Deck-changing decisions: {sum(bool(row.get('changed')) for row in choices)}",
        f"- Advice fallbacks: {sum(not bool(row.get('direct')) for row in choices)}",
        "",
        "Advice-required rows preserve the historical action; no LLM is called.",
        "",
        "## Card choices",
        "",
        "| Act/Floor | Offered | Historical | Winning Path | Difference | Policy / reason |",
        "|---:|---|---|---|---|---|",
    ]
    for row in choices:
        offered = [str(value) for value in row.get("offered") or ()]
        source = "direct" if row.get("direct") else f"{row.get('fallback')} fallback"
        difference = (
            "SAME"
            if row.get("historical_action") == row.get("candidate_action")
            else "CHANGED"
        )
        policy = _mapping(row.get("policy"))
        policy_text = (
            f"{source}; {policy.get('policy')}: {policy.get('reason')}"
        )
        lines.append(
            f"| {row.get('act')}/{row.get('floor')} "
            f"| {_escape('<br>'.join(f'[{i}] {name}' for i, name in enumerate(offered)))} "
            f"| {_escape(_action(row.get('historical_action'), offered))} "
            f"| {_escape(_action(row.get('candidate_action'), offered))} "
            f"| {difference} | {_escape(policy_text)} |"
        )

    by_result = {str(row["checkpoint_id"]): row for row in results}
    lines.extend(("", "## Combat summary", ""))
    lines.extend(
        (
            "| Checkpoint | Historical | Winning Path | Delta | Paired worlds | Result |",
            "|---|---:|---:|---:|---|---|",
        )
    )
    for checkpoint in checkpoints:
        result = by_result.get(str(checkpoint["checkpoint_id"]), {})
        if result.get("status") == "OK":
            old = _aggregate(_mapping(result.get("historical")))
            new = _aggregate(_mapping(result.get("candidate")))
            delta = f"{float(result.get('win_rate_delta') or 0):+.1%}"
            paired = ", ".join(
                f"{key}={value}"
                for key, value in _mapping(result.get("paired_outcomes")).items()
            )
            verdict = str(result.get("classification"))
        else:
            old = new = delta = paired = "-"
            verdict = f"{result.get('status')}: {result.get('reason') or 'error'}"
        lines.append(
            f"| {_checkpoint_name(checkpoint)} | {old} | {new} | {delta} "
            f"| {paired or '-'} | {verdict} |"
        )

    for checkpoint in checkpoints:
        result = by_result.get(str(checkpoint["checkpoint_id"]), {})
        lines.extend(_combat_detail(checkpoint, result))
    lines.extend(
        (
            "",
            "## Paired outcome codes",
            "",
            "- `WW`: both decks won.",
            "- `LW`: historical deck lost and Winning Path won.",
            "- `WL`: historical deck won and Winning Path lost.",
            "- `LL`: both decks lost.",
            "- `I`: the battle hit a decision or execution limit.",
            "",
        )
    )
    return "\n".join(lines)


def _combat_detail(
    checkpoint: Mapping[str, Any], result: Mapping[str, Any]
) -> list[str]:
    historical = Counter(_card_name(row) for row in checkpoint.get("historical_deck") or ())
    candidate = Counter(_card_name(row) for row in checkpoint.get("candidate_deck") or ())
    state = _mapping(checkpoint.get("game_state"))
    lines = [
        "",
        f"## {_checkpoint_name(checkpoint)}",
        "",
        f"- Recorded state: HP {state.get('current_hp')}/{state.get('max_hp')}",
        "- Relics: "
        + ", ".join(
            str(_mapping(row).get("name") or _mapping(row).get("id"))
            for row in state.get("relics") or ()
        ),
        f"- Historical deck ({sum(historical.values())}): {_counts(historical)}",
        f"- Winning Path deck ({sum(candidate.values())}): {_counts(candidate)}",
        f"- Removed by Winning Path: {_counts(historical - candidate)}",
        f"- Added by Winning Path: {_counts(candidate - historical)}",
        "- Changed choices before this fight: "
        + (", ".join(map(str, checkpoint.get("changed_choice_ids") or ())) or "none"),
    ]
    warnings = checkpoint.get("warnings") or ()
    if warnings:
        lines.append(
            "- Counterfactual notes: "
            + "; ".join(
                f"{_mapping(row).get('kind')}({_mapping(row).get('case_id') or '-'})"
                for row in warnings
            )
        )
    if result.get("status") != "OK":
        lines.extend(("", f"Simulation: {result.get('status')} — {result.get('reason') or 'error'}"))
        return lines

    lines.extend(
        (
            "",
            f"Historical: {_aggregate(_mapping(result.get('historical')))}; "
            f"Winning Path: {_aggregate(_mapping(result.get('candidate')))}; "
            f"delta {float(result.get('win_rate_delta') or 0):+.1%}; "
            f"95% interval {_interval(result.get('paired_delta_95_interval'))}; "
            f"{result.get('classification')}.",
            "",
            "| World | Historical | Winning Path |",
            "|---:|---|---|",
        )
    )
    for trial in result.get("trials") or ():
        row = _mapping(trial)
        lines.append(
            f"| {row.get('world')} | {_trial(_mapping(row.get('historical')))} "
            f"| {_trial(_mapping(row.get('candidate')))} |"
        )
    return lines


def _checkpoint_name(row: Mapping[str, Any]) -> str:
    kind = "passed boss" if row.get("kind") == "PASSED_BOSS" else "fatal encounter"
    return f"Act {row.get('act')} Floor {row.get('floor')} {row.get('encounter')} ({kind})"


def _action(value: object, offered: Sequence[str]) -> str:
    action = str(value or "")
    if action == "skip":
        return "Skip"
    parts = action.split()
    if len(parts) == 2 and parts[0] == "choose" and parts[1].isdigit():
        index = int(parts[1])
        if index == len(offered):
            return "Singing Bowl"
        if index < len(offered):
            return f"choose {index}: {offered[index]}"
    return action


def _aggregate(row: Mapping[str, Any]) -> str:
    attempts = int(row.get("attempts") or 0)
    wins = int(row.get("wins") or 0)
    hp = float(row.get("expected_end_hp_on_win") or 0)
    return f"{wins}/{attempts} ({float(row.get('win_rate') or 0):.1%}), win HP {hp:.1f}"


def _trial(row: Mapping[str, Any]) -> str:
    if not row.get("completed"):
        return f"incomplete ({row.get('stop_reason')})"
    result = "win" if row.get("won") else "loss"
    return f"{result}, HP {row.get('ending_hp')}, turn {row.get('turns')}"


def _interval(value: object) -> str:
    values = list(value) if isinstance(value, Sequence) else []
    return (
        f"[{float(values[0]):+.1%}, {float(values[1]):+.1%}]"
        if len(values) == 2
        else "-"
    )


def _card_name(value: object) -> str:
    row = _mapping(value)
    base = {
        "AscendersBane": "Ascender's Bane",
        "CurseOfTheBell": "Curse of the Bell",
        "Defend_R": "Defend",
        "Strike_R": "Strike",
    }.get(str(row.get("id") or row.get("name") or ""), str(row.get("id") or ""))
    return base + "+" * int(row.get("upgrades") or 0)


def _counts(values: Mapping[str, int]) -> str:
    return ", ".join(
        f"{name} x{count}" for name, count in sorted(values.items()) if count
    ) or "none"


def _escape(value: object) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("src/spire_agent/tools/winning_path/data/evaluation"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("3rd/sts_lightspeed/build/card-reward-eval"),
    )
    parser.add_argument("--fallback", choices=("history", "skip"), default="history")
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--max-time-ms", type=int, default=100)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--include-warnings", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_id = args.run_dir.resolve().name
    output = args.output or Path(tempfile.gettempdir()) / "winning-path-run-review" / run_id
    print(f"Evaluating run {run_id}...", file=sys.stderr)
    path = review_run(
        args.run_dir,
        args.dataset,
        output,
        args.binary,
        fallback=args.fallback,
        simulations=args.simulations,
        worlds=args.worlds,
        max_time_ms=args.max_time_ms,
        max_decisions=args.max_decisions,
        timeout=args.timeout,
        include_warnings=args.include_warnings,
    )
    print(path.read_text("utf-8"), end="")
    print(f"Report: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()


__all__ = ["RunReportError", "main", "render_report", "review_run"]
