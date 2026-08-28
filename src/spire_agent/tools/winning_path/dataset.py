"""Build a card-reward evaluation dataset from run journals."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1


class DatasetError(ValueError):
    """Raised when a source journal is internally inconsistent."""


def build_dataset(source_root: Path, output_dir: Path) -> dict[str, Any]:
    """Normalize every run journal and write deterministic dataset files."""

    journals = sorted(
        source_root.glob("*/card_choices.jsonl"), key=lambda path: _natural(path.parent.name)
    )
    if not journals:
        raise DatasetError(f"no card_choices.jsonl files under {source_root}")

    cases: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    seen_case_ids: set[str] = set()

    for journal in journals:
        run_dir = journal.parent
        raw_choices = _read_jsonl(journal)
        templates_path = run_dir / "template_choices.jsonl"
        templates = _read_jsonl(templates_path) if templates_path.exists() else []
        if templates and len(templates) != len(raw_choices):
            raise DatasetError(
                f"{run_dir.name}: {len(raw_choices)} card choices but "
                f"{len(templates)} template choices"
            )

        run_cases: list[dict[str, Any]] = []
        expected_run: dict[str, Any] | None = None
        previous_position = (0, 0)
        for sequence, raw in enumerate(raw_choices):
            source = f"{journal.as_posix()}:{sequence + 1}"
            template = templates[sequence] if templates else None
            case = _normalize_case(
                raw,
                template,
                sequence,
                run_dir.name,
                source,
                journal.relative_to(source_root).as_posix(),
            )
            case_id = case["case_id"]
            if case_id in seen_case_ids:
                raise DatasetError(f"{source}: duplicate case_id {case_id!r}")
            seen_case_ids.add(case_id)

            position = (case["state"]["act"], case["state"]["floor"])
            if position < previous_position:
                raise DatasetError(f"{source}: choices are not in game order")
            previous_position = position

            current_run = case.pop("_run")
            if expected_run is None:
                expected_run = current_run
            elif current_run != expected_run:
                raise DatasetError(f"{source}: run outcome changed within journal")
            run_cases.append(case)
            cases.append(case)

        if not run_cases or expected_run is None:
            issues.append({"run_id": run_dir.name, "kind": "EMPTY_JOURNAL"})
            continue
        if expected_run["outcome"]["status"] == "INCOMPLETE":
            issues.append({"run_id": expected_run["run_id"], "kind": "INCOMPLETE_RUN"})

        runs.append(
            {
                "schema_version": SCHEMA_VERSION,
                **expected_run,
                "case_ids": [case["case_id"] for case in run_cases],
                "transitions": _transitions(run_cases),
            }
        )
        source_entry = {
            "run_id": expected_run["run_id"],
            "card_choices": _source_digest(journal, source_root),
        }
        if templates_path.exists():
            source_entry["template_choices"] = _source_digest(
                templates_path, source_root
            )
        source_files.append(source_entry)

    stats = _statistics(cases, runs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "ironclad_card_reward_history",
        "semantics": {
            "observed_action": "historical behavior, not a correctness label",
            "comparison": "future policies are compared with historical runs",
        },
        "statistics": stats,
        "issues": issues,
        "sources": source_files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cases.jsonl", cases)
    _write_jsonl(output_dir / "runs.jsonl", runs)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "report.md").write_text(
        _render_report(stats, issues), encoding="utf-8"
    )
    return manifest


def build_expert_dataset(source_file: Path, output_dir: Path) -> dict[str, Any]:
    """Build an initial labeled dataset from one aggregated expert journal."""

    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for line, raw in enumerate(_read_jsonl(source_file), 1):
        run = _object(raw.get("run"), f"{source_file}:{line}", "run")
        run_id = _text(run.get("run_id"), f"{source_file}:{line}", "run.run_id")
        grouped.setdefault(run_id, []).append((line, raw))

    cases: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    quality = Counter()
    for run_id, raw_rows in grouped.items():
        run_cases = []
        expected_run = None
        for sequence, (line, original) in enumerate(raw_rows):
            raw = dict(original)
            context = dict(_object(raw.get("context"), str(source_file), "context"))
            context["act_boss"] = str(context.get("act_boss") or "UNKNOWN")
            raw["context"] = context
            decision = dict(_object(raw.get("decision"), str(source_file), "decision"))
            if decision.get("skipped") or decision.get("used_singing_bowl"):
                decision["picked"] = None
            else:
                decision["picked"] = decision.get("selected_card") or decision.get("picked")
            raw["decision"] = decision
            source = f"{source_file}:{line}"
            case = _normalize_case(
                raw,
                None,
                sequence,
                run_id,
                source,
                source_file.name,
            )
            case["provenance"]["line"] = line
            curation = _object(raw.get("curation"), source, "curation")
            evidence_class = str(curation.get("evidence_class") or "unknown")
            verified = bool(curation.get("eligible_modern_verified_seed"))
            case["quality"] = {
                "evidence_class": evidence_class,
                "verified": verified,
                "exclusion_reasons": list(curation.get("exclusion_reasons") or ()),
            }
            quality[evidence_class] += 1
            current_run = case.pop("_run")
            if expected_run is None:
                expected_run = current_run
            elif current_run != expected_run:
                raise DatasetError(f"{source}: run outcome changed within journal")
            run_cases.append(case)
            cases.append(case)
            action = case["observed_action"]["action"]
            labels.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "case_id": case["case_id"],
                    "acceptable_actions": [action],
                    "preferred_action": action,
                    "confidence": "HIGH" if verified else "MEDIUM",
                    "reason": f"Baalorlord expert history ({evidence_class}).",
                    "reviewer": "baalorlord_history",
                    "blind": False,
                }
            )
        if expected_run is None:
            continue
        runs.append(
            {
                "schema_version": SCHEMA_VERSION,
                **expected_run,
                "case_ids": [case["case_id"] for case in run_cases],
                "transitions": _transitions(run_cases),
            }
        )

    stats = _statistics(cases, runs)
    stats["quality"] = dict(sorted(quality.items()))
    source_bytes = source_file.read_bytes()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "baalorlord_ironclad_card_reward_history",
        "semantics": {
            "expert_actions": "immutable historical expert actions",
            "quality": "modern_verified is the highest-confidence subset",
        },
        "statistics": stats,
        "source": {
            "path": source_file.as_posix(),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "bytes": len(source_bytes),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cases.jsonl", cases)
    _write_jsonl(output_dir / "runs.jsonl", runs)
    _write_jsonl(output_dir / "expert_actions.jsonl", labels)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _normalize_case(
    raw: Mapping[str, Any],
    template: Mapping[str, Any] | None,
    sequence: int,
    directory_name: str,
    source: str,
    provenance_file: str,
) -> dict[str, Any]:
    if raw.get("schema_version") != 1:
        raise DatasetError(f"{source}: unsupported source schema")
    context = _object(raw.get("context"), source, "context")
    decision = _object(raw.get("decision"), source, "decision")
    run = _object(raw.get("run"), source, "run")

    run_id = _text(run.get("run_id"), source, "run.run_id")
    if run_id != directory_name:
        raise DatasetError(
            f"{source}: run_id {run_id!r} does not match directory {directory_name!r}"
        )
    source_case_id = _text(raw.get("choice_id"), source, "choice_id")
    offered = _string_list(decision.get("offered"), source, "decision.offered")
    if not offered or len(set(offered)) != len(offered):
        raise DatasetError(f"{source}: offered cards must be non-empty and unique")

    skipped = _boolean(decision.get("skipped"), source, "decision.skipped")
    used_bowl = _boolean(
        decision.get("used_singing_bowl"), source, "decision.used_singing_bowl"
    )
    picked_value = decision.get("picked")
    picked = None if picked_value is None else _text(picked_value, source, "picked")
    modes = int(skipped) + int(used_bowl) + int(picked is not None)
    if modes != 1:
        raise DatasetError(f"{source}: decision must be pick, skip, or Singing Bowl")
    if picked is not None and picked not in offered:
        raise DatasetError(f"{source}: picked card {picked!r} is not offered")
    action = (
        f"choose {offered.index(picked)}"
        if picked is not None
        else f"choose {len(offered)}"
        if used_bowl
        else "skip"
    )

    act = _integer(decision.get("act"), source, "decision.act", minimum=1)
    floor = _integer(decision.get("floor"), source, "decision.floor", minimum=0)
    boss = _text(context.get("act_boss"), source, "context.act_boss")
    observed = {
        "action": action,
        "picked": picked,
        "skipped": skipped,
        "used_singing_bowl": used_bowl,
        "source": "unknown",
        "rationale": None,
    }
    if template is not None:
        _merge_template(observed, template, act, floor, boss, action, source)

    deck_raw = _object(
        context.get("deck_before_counts"), source, "context.deck_before_counts"
    )
    deck: dict[str, int] = {}
    for name, count in sorted(deck_raw.items(), key=lambda item: str(item[0]).casefold()):
        card_name = _text(name, source, "deck card name")
        deck[card_name] = _integer(count, source, f"deck count for {card_name}", minimum=1)
    if not deck:
        raise DatasetError(f"{source}: deck is empty")

    floor_reached = run.get("floor_reached")
    if floor_reached is not None:
        floor_reached = _integer(
            floor_reached, source, "run.floor_reached", minimum=0
        )
    victory = _optional_boolean(run.get("victory"), source, "run.victory")
    heart_kill = _optional_boolean(
        run.get("heart_kill"), source, "run.heart_kill"
    )
    if heart_kill is True and victory is not True:
        raise DatasetError(f"{source}: heart_kill requires victory")
    status = (
        "HEART_KILL"
        if heart_kill is True
        else "VICTORY"
        if victory is True
        else "DEFEAT"
        if victory is False and floor_reached is not None
        else "INCOMPLETE"
    )
    killed_by = run.get("killed_by")
    if killed_by is not None:
        killed_by = str(killed_by).strip() or None

    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": source_case_id,
        "run_id": run_id,
        "sequence": sequence,
        "state": {
            "act": act,
            "floor": floor,
            "boss": boss,
            "hp": _integer(context.get("hp_after_floor"), source, "hp", minimum=0),
            "max_hp": _integer(
                context.get("max_hp_after_floor"), source, "max_hp", minimum=1
            ),
            "gold": _integer(
                context.get("gold_after_floor"), source, "gold", minimum=0
            ),
            "deck_counts": deck,
            "relics": _string_list(
                context.get("relics_before_floor_rewards"), source, "relics"
            ),
        },
        "reward": {
            "kind": _text(decision.get("kind"), source, "decision.kind"),
            "offered": offered,
        },
        "observed_action": observed,
        "provenance": {"file": provenance_file, "line": sequence + 1},
        "_run": {
            "run_id": run_id,
            "ascension": _integer(
                run.get("ascension"), source, "run.ascension", minimum=0
            ),
            "outcome": {
                "status": status,
                "floor_reached": floor_reached,
                "killed_by": killed_by,
            },
        },
    }


def _transitions(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record non-reward deck mutations between consecutive choices."""

    result: list[dict[str, Any]] = []
    for current, following in zip(cases, cases[1:]):
        after_choice = Counter(current["state"]["deck_counts"])
        picked = current["observed_action"]["picked"]
        if picked is not None:
            after_choice[picked] += 1
        following_deck = Counter(following["state"]["deck_counts"])
        result.append(
            {
                "from_case_id": current["case_id"],
                "to_case_id": following["case_id"],
                "fixed_deck_delta": {
                    "removed": _positive_counts(after_choice - following_deck),
                    "added": _positive_counts(following_deck - after_choice),
                },
            }
        )
    return result


def _positive_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {
        name: int(count)
        for name, count in sorted(counts.items(), key=lambda item: item[0].casefold())
        if count > 0
    }


def _merge_template(
    observed: dict[str, Any],
    template: Mapping[str, Any],
    act: int,
    floor: int,
    boss: str,
    action: str,
    source: str,
) -> None:
    if template.get("act") != act or template.get("floor") != floor:
        raise DatasetError(f"{source}: template choice position does not match")
    template_boss = str(template.get("boss") or "")
    if template_boss and template_boss != boss:
        raise DatasetError(f"{source}: template boss does not match")
    final = _object(template.get("final_decision"), source, "template.final_decision")
    if final.get("action") != action:
        raise DatasetError(
            f"{source}: template action {final.get('action')!r} does not match {action!r}"
        )
    observed["source"] = _text(
        final.get("source"), source, "template.final_decision.source"
    )
    rationale = final.get("rationale")
    observed["rationale"] = str(rationale).strip() if rationale else None


def _statistics(
    cases: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    actions = Counter(
        "singing_bowl"
        if case["observed_action"]["used_singing_bowl"]
        else "skip"
        if case["observed_action"]["skipped"]
        else "pick"
        for case in cases
    )
    sources = Counter(case["observed_action"]["source"] for case in cases)
    outcomes = Counter(run["outcome"]["status"] for run in runs)
    acts = Counter(str(case["state"]["act"]) for case in cases)
    kinds = Counter(case["reward"]["kind"] for case in cases)
    return {
        "runs": len(runs),
        "cases": len(cases),
        "actions": dict(sorted(actions.items())),
        "decision_sources": dict(sorted(sources.items())),
        "run_outcomes": dict(sorted(outcomes.items())),
        "acts": dict(sorted(acts.items())),
        "reward_kinds": dict(sorted(kinds.items())),
    }


def _render_report(stats: Mapping[str, Any], issues: list[dict[str, str]]) -> str:
    def lines(title: str, values: Mapping[str, Any]) -> list[str]:
        return [f"### {title}", "", *[f"- {key}: {value}" for key, value in values.items()], ""]

    result = [
        "# Card reward dataset report",
        "",
        f"Runs: {stats['runs']}",
        f"Cases: {stats['cases']}",
        "",
        "Historical actions are observations, not correctness labels.",
        "",
    ]
    result += lines("Actions", stats["actions"])
    result += lines("Decision sources", stats["decision_sources"])
    result += lines("Run outcomes", stats["run_outcomes"])
    result += lines("Acts", stats["acts"])
    result += lines("Reward kinds", stats["reward_kinds"])
    result += ["### Data quality", ""]
    if issues:
        result += [f"- {issue['run_id']}: {issue['kind']}" for issue in issues]
    else:
        result.append("- No issues found.")
    return "\n".join(result) + "\n"


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, Mapping):
            raise DatasetError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _object(value: Any, source: str, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetError(f"{source}: {field} must be an object")
    return value


def _string_list(value: Any, source: str, field: str) -> list[str]:
    if not isinstance(value, list):
        raise DatasetError(f"{source}: {field} must be a list")
    return [_text(item, source, field) for item in value]


def _text(value: Any, source: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{source}: {field} must be a non-empty string")
    return value.strip()


def _integer(value: Any, source: str, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DatasetError(f"{source}: {field} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, source: str, field: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetError(f"{source}: {field} must be boolean")
    return value


def _optional_boolean(value: Any, source: str, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, source, field)


def _source_digest(path: Path, source_root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(source_root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _natural(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../remote"))
    parser.add_argument("--expert-source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/spire_agent/tools/winning_path/data/evaluation"),
    )
    args = parser.parse_args(argv)
    manifest = (
        build_expert_dataset(args.expert_source, args.output)
        if args.expert_source
        else build_dataset(args.source, args.output)
    )
    stats = manifest["statistics"]
    print(f"Wrote {stats['cases']} cases from {stats['runs']} runs to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["DatasetError", "build_dataset", "build_expert_dataset", "main"]
