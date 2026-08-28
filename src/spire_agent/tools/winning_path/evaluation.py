"""Evaluate the card-reward policy against frozen cases."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from .preference import PreferenceBuilder
from .picker import review_card_reward
from .protocol import PROTOCOL_VERSION


class EvaluationError(ValueError):
    pass


def evaluate(
    dataset_dir: Path,
    output_dir: Path,
    *,
    review_quality: str | None = None,
    compact: bool = False,
    preference_folds: int = 0,
) -> dict[str, Any]:
    """Run snapshot and conservative sequential evaluation without an LLM."""

    cases = _read_jsonl(dataset_dir / "cases.jsonl")
    runs = _read_jsonl(dataset_dir / "runs.jsonl")
    reviews = _load_reviews(dataset_dir / "expert_actions.jsonl", cases)
    by_id = {str(case["case_id"]): case for case in cases}
    if len(by_id) != len(cases):
        raise EvaluationError("dataset contains duplicate case_id values")

    results: list[dict[str, Any]] = []
    for run in runs:
        shared: dict[str, Any] = {}
        for case_id in run.get("case_ids") or ():
            case = by_id.get(str(case_id))
            if case is None:
                raise EvaluationError(f"run references unknown case {case_id!r}")
            outcome = _review(
                _request(case, case["state"]["deck_counts"], shared)
            )
            result = _snapshot_result(case, outcome, reviews.get(str(case_id)))
            results.append(result)

    if len(results) != len(cases):
        raise EvaluationError("some cases are missing from run trajectories")
    trajectories = [_evaluate_run(run, by_id) for run in runs]
    differences = [
        row
        for row in results
        if row["comparison"] in {"DIRECT_DIFFERENT", "ADVICE_OBSERVED_OUTSIDE"}
    ]
    review_differences = [
        row
        for row in differences
        if review_quality is None
        or _mapping(row.get("quality")).get("evidence_class") == review_quality
    ]
    summary = _summary(results, trajectories)
    summary["policy_version"] = PROTOCOL_VERSION

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cases.jsonl", review_differences if compact else results)
    _write_jsonl(output_dir / "runs.jsonl", [] if compact else trajectories)
    _write_jsonl(output_dir / "differences.jsonl", review_differences)
    _write_jsonl(
        output_dir / "verified_differences.jsonl",
        [
            row
            for row in review_differences
            if _mapping(row.get("quality")).get("verified")
        ],
    )
    _write_json(output_dir / "summary.json", summary)
    if preference_folds:
        cross_validation = cross_validate_preferences(cases, preference_folds)
        _write_json(
            output_dir / "preference_cross_validation.json", cross_validation
        )
        summary["preference_cross_validation"] = cross_validation
        _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        _render_report(summary, results), encoding="utf-8"
    )
    return summary


def cross_validate_preferences(
    cases: list[Mapping[str, Any]],
    folds: int = 5,
    *,
    quality: str = "modern_verified",
) -> dict[str, Any]:
    """Evaluate preferences with every test run excluded from its training fold."""

    if folds < 2:
        raise EvaluationError("preference cross-validation needs at least two folds")
    fold_by_run = {
        str(case["run_id"]): _fold(str(case["run_id"]), folds) for case in cases
    }
    selected = [
        case
        for case in cases
        if str(_mapping(case.get("quality")).get("evidence_class")) == quality
    ]
    if not selected:
        selected = list(cases)
        quality = "all"
    comparisons: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    directions_by_deck_size: dict[str, Counter[str]] = {}
    directions_by_policy: dict[str, Counter[str]] = {}
    training_observations = []
    for fold in range(folds):
        builder = PreferenceBuilder()
        for case in cases:
            if fold_by_run[str(case["run_id"])] != fold:
                _observe_case(builder, case)
        payload = builder.payload()
        training_observations.append(payload["observations"])
        for case in selected:
            if fold_by_run[str(case["run_id"])] != fold:
                continue
            state = _mapping(case.get("state"))
            outcome = _review(
                _request(case, _mapping(state.get("deck_counts")), {}),
                preference_payload=payload,
            )
            action = _mapping(case.get("observed_action")).get("action")
            command = outcome.get("command")
            if command:
                comparison = "DIRECT_MATCH" if command == action else "DIRECT_DIFFERENT"
            else:
                comparison = "ADVICE"
            comparisons[comparison] += 1
            policy_name = str(outcome.get("policy"))
            policies[policy_name] += 1
            direction = _decision_direction(case, outcome)
            directions[direction] += 1
            directions_by_policy.setdefault(policy_name, Counter())[direction] += 1
            size = sum(
                int(value)
                for value in _mapping(state.get("deck_counts")).values()
            )
            directions_by_deck_size.setdefault(
                _deck_size_band(size), Counter()
            )[direction] += 1
    direct = comparisons["DIRECT_MATCH"] + comparisons["DIRECT_DIFFERENT"]
    return {
        "schema_version": 1,
        "method": "run_grouped_k_fold",
        "folds": folds,
        "quality": quality,
        "cases": len(selected),
        "runs": len({str(case["run_id"]) for case in selected}),
        "training_observations_by_fold": training_observations,
        "comparisons": dict(sorted(comparisons.items())),
        "decision_directions": dict(sorted(directions.items())),
        "decision_directions_by_deck_size": {
            band: dict(sorted(values.items()))
            for band, values in directions_by_deck_size.items()
        },
        "decision_directions_by_policy": {
            name: dict(sorted(values.items()))
            for name, values in sorted(directions_by_policy.items())
        },
        "policies": dict(sorted(policies.items())),
        "deterministic_coverage": round(direct / len(selected), 6) if selected else 0,
        "direct_agreement": (
            round(comparisons["DIRECT_MATCH"] / direct, 6) if direct else 0
        ),
    }


def _decision_direction(
    case: Mapping[str, Any], policy: Mapping[str, Any]
) -> str:
    observed = _mapping(case.get("observed_action"))
    command = policy.get("command")
    if not isinstance(command, str) or not command:
        policy_kind = "ADVICE"
    elif command == "skip" or policy.get("policy") == "SINGING_BOWL":
        policy_kind = "NO_CARD"
    else:
        policy_kind = "PICK"
    observed_kind = "PICK" if observed.get("picked") is not None else "NO_CARD"
    if command == observed.get("action"):
        return "EXACT_MATCH"
    return f"{policy_kind}_VS_{observed_kind}"


def _deck_size_band(size: int) -> str:
    for limit, label in (
        (15, "00-14"),
        (20, "15-19"),
        (25, "20-24"),
        (30, "25-29"),
    ):
        if size < limit:
            return label
    return "30+"


def _observe_case(builder: PreferenceBuilder, case: Mapping[str, Any]) -> None:
    state = _mapping(case.get("state"))
    reward = _mapping(case.get("reward"))
    observed = _mapping(case.get("observed_action"))
    builder.observe(
        state.get("act"),
        reward.get("offered") or (),
        picked=observed.get("picked"),
        skipped=bool(observed.get("skipped")),
        used_singing_bowl=bool(observed.get("used_singing_bowl")),
        owned=state.get("deck_counts") or {},
    )


def _fold(run_id: str, folds: int) -> int:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _request(
    case: Mapping[str, Any],
    deck_counts: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> DecisionRequest:
    state_data = _mapping(case.get("state"))
    reward = _mapping(case.get("reward"))
    deck = tuple(
        {"name": name}
        for name, count in deck_counts.items()
        for _ in range(int(count))
    )
    relics = tuple({"name": name} for name in state_data.get("relics") or ())
    offered = tuple(str(name) for name in reward.get("offered") or ())
    bowl = bool(
        _mapping(case.get("observed_action")).get("used_singing_bowl")
        or "Singing Bowl" in (state_data.get("relics") or ())
    )
    state = GameState(
        AgentKind.BUILD,
        f"evaluation:{case['case_id']}",
        ScreenState(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=offered,
            details={
                "cards": tuple({"name": name} for name in offered),
                "reward_type": reward.get("kind"),
                "singing_bowl": bowl,
            },
        ),
        facts={
            "class": "IRONCLAD",
            "act": state_data.get("act"),
            "floor": state_data.get("floor"),
            "act_boss": state_data.get("boss"),
            "current_hp": state_data.get("hp"),
            "max_hp": state_data.get("max_hp"),
            "gold": state_data.get("gold"),
            "deck": deck,
            "relics": relics,
        },
    )
    scope = DecisionScope(AgentKind.BUILD, state.scope_id)
    return DecisionRequest(
        state,
        scope,
        None,
        shared,
        ContextEntry(0, None, state, True, scope=scope),
    )


def _snapshot_result(
    case: Mapping[str, Any],
    policy: Mapping[str, Any],
    gold: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observed = _mapping(case.get("observed_action"))
    observed_action = str(observed.get("action") or "")
    allowed = _allowed_actions(policy)
    direct = policy.get("command")
    if isinstance(direct, str) and direct:
        comparison = "DIRECT_MATCH" if direct == observed_action else "DIRECT_DIFFERENT"
    else:
        comparison = (
            "ADVICE_OBSERVED_ALLOWED"
            if observed_action in allowed
            else "ADVICE_OBSERVED_OUTSIDE"
        )

    review_result: dict[str, Any] = {"status": "UNREVIEWED"}
    if gold is not None:
        acceptable = set(gold["acceptable_actions"])
        overlap = sorted(acceptable & set(allowed))
        expected = _mapping(gold.get("expected_policy"))
        exact_contract = (
            policy.get("mode") == expected.get("mode")
            and set(allowed) == set(expected.get("actions") or ())
        ) if expected else None
        review_result = {
            "status": (
                "PASS"
                if exact_contract is True
                else "FAIL"
                if exact_contract is False
                else
                "PASS"
                if isinstance(direct, str) and direct in acceptable
                else "COVERED"
                if overlap
                else "FAIL"
            ),
            "acceptable_actions": sorted(acceptable),
            "preferred_action": gold.get("preferred_action"),
            "covered_actions": overlap,
            "confidence": gold.get("confidence"),
            "reason": gold.get("reason"),
            "expected_policy": dict(expected) if expected else None,
            "exact_contract": exact_contract,
        }

    return {
        "schema_version": 1,
        "case_id": case["case_id"],
        "run_id": case["run_id"],
        "sequence": case["sequence"],
        "quality": case.get("quality"),
        "state": case["state"],
        "reward": case["reward"],
        "observed_action": observed,
        "policy_result": _compact_policy(policy),
        "comparison": comparison,
        "review": review_result,
        "review_priority": _review_reasons(case, comparison, policy),
    }


def _compact_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    winning_path = _mapping(policy.get("winning_path"))
    return {
        "mode": policy.get("mode"),
        "policy": policy.get("policy"),
        "structural_policy": policy.get("structural_policy"),
        "command": policy.get("command"),
        "allowed_actions": _allowed_actions(policy),
        "reason": policy.get("reason"),
        "preference": policy.get("preference"),
        "skip": winning_path.get("skip"),
        "nearest_paths": winning_path.get("nearest_paths") or [],
        "candidates": [
            {
                "choice_id": row.get("choice_id"),
                "name": row.get("name"),
                "impact": row.get("impact"),
                "reachability": row.get("reachability"),
                "utility": row.get("utility"),
                "delta_vs_skip": row.get("delta_vs_skip"),
                "activates_modules": row.get("activates_modules") or [],
                "strong_modules": row.get("strong_modules") or [],
                "improves_paths": row.get("improves_paths") or [],
                "reviewed_support": row.get("reviewed_support"),
                "capability_gain": row.get("capability_gain") or [],
                "bridge_needs": row.get("bridge_needs") or [],
            }
            for row in policy.get("candidates") or ()
            if isinstance(row, Mapping)
        ],
    }


def _allowed_actions(policy: Mapping[str, Any]) -> list[str]:
    direct = policy.get("command")
    if isinstance(direct, str) and direct:
        return [direct]
    return [
        *(("skip",) if policy.get("allow_skip", True) else ()),
        *(f"choose {int(value)}" for value in policy.get("allowed_choice_ids") or ()),
    ]


def _evaluate_run(
    run: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case_ids = [str(value) for value in run.get("case_ids") or ()]
    if not case_ids:
        raise EvaluationError(f"run {run.get('run_id')} has no cases")
    deck = Counter(_mapping(by_id[case_ids[0]]["state"]).get("deck_counts") or {})
    transitions = {
        str(row.get("from_case_id")): row
        for row in run.get("transitions") or ()
        if isinstance(row, Mapping)
    }
    shared: dict[str, Any] = {}
    steps = []
    warnings = []
    pick_count = 0
    peak_size = sum(deck.values())
    for case_id in case_ids:
        case = by_id[case_id]
        outcome = _review(_request(case, deck, shared))
        command = outcome.get("command")
        action = command if isinstance(command, str) and command else "skip"
        before_size = sum(deck.values())
        picked = _picked_card(action, _mapping(case.get("reward")).get("offered") or ())
        if picked is not None:
            deck[picked] += 1
            pick_count += 1
        transition = transitions.get(case_id)
        step_warnings = _apply_fixed_delta(deck, transition)
        warnings.extend({"case_id": case_id, **item} for item in step_warnings)
        after_size = sum(deck.values())
        peak_size = max(peak_size, after_size)
        steps.append(
            {
                "case_id": case_id,
                "mode": outcome.get("mode"),
                "policy": outcome.get("policy"),
                "action": action,
                "advice_defaulted_to_skip": not bool(command),
                "deck_size_before": before_size,
                "deck_size_after": after_size,
                "warnings": step_warnings,
            }
        )
    return {
        "schema_version": 1,
        "run_id": run["run_id"],
        "source_outcome": run.get("outcome"),
        "mode": "DIRECT_POLICY_WITH_ADVICE_DEFAULT_SKIP",
        "initial_deck_size": sum(
            _mapping(by_id[case_ids[0]]["state"]).get("deck_counts", {}).values()
        ),
        "final_deck_size": sum(deck.values()),
        "peak_deck_size": peak_size,
        "policy_picks": pick_count,
        "policy_skips": len(case_ids) - pick_count,
        "warnings": warnings,
        "steps": steps,
    }


def _apply_fixed_delta(
    deck: Counter[str], transition: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    if transition is None:
        return []
    delta = _mapping(transition.get("fixed_deck_delta"))
    removed = Counter(_mapping(delta.get("removed")))
    added = Counter(_mapping(delta.get("added")))
    warnings: list[dict[str, Any]] = []
    for name, count in removed.items():
        applied = min(deck.get(name, 0), count)
        if applied:
            deck[name] -= applied
            if deck[name] <= 0:
                del deck[name]
        if applied != count:
            warnings.append(
                {
                    "kind": "FIXED_REMOVAL_UNAVAILABLE",
                    "card": name,
                    "requested": count,
                    "applied": applied,
                }
            )
    for name, count in added.items():
        deck[name] += count
    return warnings


def _picked_card(action: str, offered: Any) -> str | None:
    parts = action.split()
    if len(parts) != 2 or parts[0] != "choose":
        return None
    try:
        index = int(parts[1])
    except ValueError:
        return None
    cards = list(offered)
    return str(cards[index]) if 0 <= index < len(cards) else None


def _review_reasons(
    case: Mapping[str, Any], comparison: str, policy: Mapping[str, Any]
) -> list[str]:
    reasons = []
    if comparison in {"DIRECT_DIFFERENT", "ADVICE_OBSERVED_OUTSIDE"}:
        reasons.append("POLICY_CONFLICT")
    if policy.get("mode") == "DIRECT":
        reasons.append("DIRECT_DECISION")
    state, reward = _mapping(case.get("state")), _mapping(case.get("reward"))
    if reward.get("kind") in {"boss_card_reward", "event_card_reward"} or int(
        state.get("act") or 0
    ) >= 3:
        reasons.append("RARE_CONTEXT")
    if policy.get("mode") == "ADVICE_REQUIRED":
        reasons.append("ADVICE_REQUIRED")
    return reasons


def _review_order(result: Mapping[str, Any]) -> tuple[Any, ...]:
    reasons = set(result.get("review_priority") or ())
    rank = next(
        (
            index
            for index, reason in enumerate(
                ("POLICY_CONFLICT", "DIRECT_DECISION", "RARE_CONTEXT", "ADVICE_REQUIRED")
            )
            if reason in reasons
        ),
        4,
    )
    state = _mapping(result.get("state"))
    quality_rank = 0 if _mapping(result.get("quality")).get("verified") else 1
    return (
        quality_rank,
        rank,
        str(result.get("run_id")),
        int(result.get("sequence") or 0),
        state.get("floor"),
    )


def _summary(
    results: list[dict[str, Any]], trajectories: list[dict[str, Any]]
) -> dict[str, Any]:
    comparisons = Counter(row["comparison"] for row in results)
    modes = Counter(row["policy_result"]["mode"] for row in results)
    policies = Counter(row["policy_result"]["policy"] for row in results)
    references = Counter(row["review"]["status"] for row in results)
    quality: dict[str, Counter[str]] = {}
    for row in results:
        evidence = str(_mapping(row.get("quality")).get("evidence_class") or "unknown")
        quality.setdefault(evidence, Counter())[row["comparison"]] += 1
    return {
        "schema_version": 1,
        "cases": len(results),
        "runs": len(trajectories),
        "comparisons": dict(sorted(comparisons.items())),
        "modes": dict(sorted(modes.items())),
        "policies": dict(sorted(policies.items())),
        "expert_references": dict(sorted(references.items())),
        "differences": sum(
            comparisons[key]
            for key in ("DIRECT_DIFFERENT", "ADVICE_OBSERVED_OUTSIDE")
        ),
        "quality": {
            key: {
                "cases": sum(values.values()),
                "differences": values["DIRECT_DIFFERENT"]
                + values["ADVICE_OBSERVED_OUTSIDE"],
                "comparisons": dict(sorted(values.items())),
            }
            for key, values in sorted(quality.items())
        },
        "sequential": {
            "mode": "DIRECT_POLICY_WITH_ADVICE_DEFAULT_SKIP",
            "policy_picks": sum(row["policy_picks"] for row in trajectories),
            "policy_skips": sum(row["policy_skips"] for row in trajectories),
            "runs_with_warnings": sum(bool(row["warnings"]) for row in trajectories),
        },
    }


def _render_report(summary: Mapping[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Current card reward policy evaluation",
        "",
        f"Cases: {summary['cases']}",
        f"Runs: {summary['runs']}",
        f"Differences requiring confirmation: {summary['differences']}",
        "",
        "Historical agreement is a behavior comparison, not a correctness score.",
        "Advice-required decisions are sets of allowed actions, not predictions.",
        "",
    ]
    for title, key in (
        ("Historical comparison", "comparisons"),
        ("Decision ownership", "modes"),
        ("Policy paths", "policies"),
    ):
        lines += [f"## {title}", ""]
        lines += [f"- {name}: {count}" for name, count in summary[key].items()]
        lines.append("")
    if any(name != "unknown" for name in summary.get("quality", {})):
        lines += ["## Data quality", ""]
        lines += [
            f"- {name}: {row['cases']} cases, {row['differences']} differences"
            for name, row in summary["quality"].items()
        ]
        lines.append("")
    cross_validation = _mapping(summary.get("preference_cross_validation"))
    if cross_validation:
        lines += [
            "## Leakage-safe preference cross-validation",
            "",
            f"- deterministic coverage: {cross_validation['deterministic_coverage']}",
            f"- direct agreement: {cross_validation['direct_agreement']}",
            "",
            "Decision directions:",
            "",
        ]
        lines += [
            f"- {name}: {count}"
            for name, count in _mapping(
                cross_validation.get("decision_directions")
            ).items()
        ]
        lines += ["", "Decision directions by deck size:", ""]
        for band, values in _mapping(
            cross_validation.get("decision_directions_by_deck_size")
        ).items():
            counts = ", ".join(
                f"{name}={count}"
                for name, count in _mapping(values).items()
            )
            lines.append(f"- {band}: {counts}")
        lines.append("")
    sequential = summary["sequential"]
    lines += [
        "## Conservative sequential pass",
        "",
        f"- mode: {sequential['mode']}",
        f"- policy picks: {sequential['policy_picks']}",
        f"- policy skips: {sequential['policy_skips']}",
        f"- runs with fixed-delta warnings: {sequential['runs_with_warnings']}",
        "",
        "This pass executes direct policy decisions and resolves every advice-required "
        "case to Skip. It measures deterministic deck growth; it does not predict wins.",
        "",
        "## First review cases",
        "",
    ]
    for row in sorted(results, key=_review_order)[:30]:
        state, reward = row["state"], row["reward"]
        lines.append(
            f"- {row['case_id']} | A{state['act']} F{state['floor']} | "
            f"{row['comparison']} | observed={row['observed_action']['action']} | "
            f"allowed={','.join(row['policy_result']['allowed_actions'])} | "
            f"offered={', '.join(reward['offered'])}"
        )
    return "\n".join(lines) + "\n"


def _load_reviews(
    path: Path, cases: list[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    legal = {
        str(case["case_id"]): {
            "skip",
            *(f"choose {index}" for index, _ in enumerate(case["reward"]["offered"])),
            *(
                (f"choose {len(case['reward']['offered'])}",)
                if _mapping(case.get("observed_action")).get("used_singing_bowl")
                else ()
            ),
        }
        for case in cases
    }
    result = {}
    for row in _read_jsonl(path):
        case_id = str(row.get("case_id") or "")
        if case_id not in legal:
            raise EvaluationError(f"review references unknown case {case_id!r}")
        if case_id in result:
            raise EvaluationError(f"duplicate review for {case_id}")
        acceptable = row.get("acceptable_actions")
        if not isinstance(acceptable, list) or not acceptable:
            raise EvaluationError(f"review {case_id} has no acceptable_actions")
        if not set(acceptable) <= legal[case_id]:
            raise EvaluationError(f"review {case_id} contains an illegal action")
        preferred = row.get("preferred_action")
        if preferred is not None and preferred not in acceptable:
            raise EvaluationError(f"review {case_id} preferred action is not acceptable")
        expected = row.get("expected_policy")
        if expected is not None:
            if not isinstance(expected, Mapping):
                raise EvaluationError(f"review {case_id} expected_policy is invalid")
            if expected.get("mode") not in {"DIRECT", "ADVICE_REQUIRED"}:
                raise EvaluationError(f"review {case_id} expected_policy mode is invalid")
            actions = expected.get("actions")
            if not isinstance(actions, list) or not actions or not set(actions) <= legal[case_id]:
                raise EvaluationError(f"review {case_id} expected_policy actions are invalid")
            if expected.get("mode") == "DIRECT" and len(actions) != 1:
                raise EvaluationError(f"review {case_id} DIRECT policy needs one action")
        result[case_id] = row
    return result


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise EvaluationError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _review(
    request: DecisionRequest,
    *,
    preference_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return review_card_reward(request, preference_payload=preference_payload)


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
    root = Path("src/spire_agent/tools/winning_path/data/evaluation")
    parser.add_argument("--dataset", type=Path, default=root)
    parser.add_argument(
        "--output", type=Path, default=root / "current_policy"
    )
    parser.add_argument(
        "--review-quality",
        help="write only differences from this evidence class to the review queue",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="store only queued differences instead of every evaluated case",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero when any reviewed policy contract fails",
    )
    parser.add_argument(
        "--preference-folds",
        type=int,
        default=0,
        help="also run leakage-safe expert preference cross-validation",
    )
    parser.add_argument(
        "--preference-only",
        action="store_true",
        help="run only the leakage-safe preference evaluation",
    )
    args = parser.parse_args(argv)
    if args.preference_only:
        if args.preference_folds < 2:
            parser.error("--preference-only requires --preference-folds >= 2")
        cases = _read_jsonl(args.dataset / "cases.jsonl")
        result = cross_validate_preferences(cases, args.preference_folds)
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "preference_cross_validation.json", result)
        (args.output / "report.md").write_text(
            "\n".join(
                (
                    "# Winning Path card reward cross-validation",
                    "",
                    f"- cases: {result['cases']}",
                    f"- deterministic coverage: {result['deterministic_coverage']}",
                    f"- direct agreement: {result['direct_agreement']}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        print(f"Wrote preference evaluation to {args.output}")
        return
    summary = evaluate(
        args.dataset,
        args.output,
        review_quality=args.review_quality,
        compact=args.compact,
        preference_folds=args.preference_folds,
    )
    print(
        f"Evaluated {summary['cases']} cases across {summary['runs']} runs; "
        f"report: {args.output / 'report.md'}"
    )
    failures = int(summary.get("expert_references", {}).get("FAIL", 0))
    if args.check and failures:
        raise SystemExit(f"card reward regression check failed: {failures} review(s)")


if __name__ == "__main__":
    main()


__all__ = ["EvaluationError", "cross_validate_preferences", "evaluate", "main"]
