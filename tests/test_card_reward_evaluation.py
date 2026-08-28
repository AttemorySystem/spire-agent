from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.tools.winning_path.evaluation import (
    EvaluationError,
    _request,
    cross_validate_preferences,
    evaluate,
)
from spire_agent.tools.winning_path.needs import plan_targets
from spire_agent.tools.winning_path.state import project_state


def case(
    case_id: str,
    run_id: str,
    *,
    deck: dict[str, int],
    offered: list[str],
    action: str,
) -> dict:
    picked = None
    if action.startswith("choose "):
        picked = offered[int(action.split()[1])]
    return {
        "schema_version": 1,
        "case_id": case_id,
        "run_id": run_id,
        "sequence": 0,
        "state": {
            "act": 1,
            "floor": 1,
            "boss": "Hexaghost",
            "hp": 70,
            "max_hp": 75,
            "gold": 100,
            "deck_counts": deck,
            "relics": ["Burning Blood"],
        },
        "reward": {"kind": "combat_card_reward", "offered": offered},
        "observed_action": {
            "action": action,
            "picked": picked,
            "skipped": action == "skip",
            "used_singing_bowl": False,
            "source": "llm",
            "rationale": None,
        },
        "provenance": {"file": f"{run_id}/card_choices.jsonl", "line": 1},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class CardRewardEvaluationTests(unittest.TestCase):
    def test_preference_cross_validation_excludes_the_test_run(self):
        first = case(
            "1:f1:c0",
            "1",
            deck={"Strike": 5},
            offered=["Test Card A", "Test Card B"],
            action="choose 0",
        )
        second = case(
            "2:f1:c0",
            "2",
            deck={"Strike": 5},
            offered=["Test Card A", "Test Card B"],
            action="choose 1",
        )

        cases = [
            first,
            case(
                "1:f2:c0",
                "1",
                deck={"Strike": 5},
                offered=["Test Card A", "Test Card B"],
                action="choose 0",
            ),
            second,
            case(
                "2:f2:c0",
                "2",
                deck={"Strike": 5},
                offered=["Test Card A", "Test Card B"],
                action="choose 1",
            ),
        ]
        result = cross_validate_preferences(cases, folds=2)

        self.assertEqual(result["deterministic_coverage"], 1)
        self.assertEqual(result["direct_agreement"], 0)
        self.assertEqual(result["decision_directions"], {"PICK_VS_PICK": 4})
        self.assertEqual(
            result["decision_directions_by_deck_size"],
            {"00-14": {"PICK_VS_PICK": 4}},
        )

    def test_expert_labels_create_a_quality_preserving_difference_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = case(
                "1:f1:c0",
                "1",
                deck={"Strike": 5, "Defend": 4, "Bash": 1},
                offered=["Inflame", "Corruption", "Clash"],
                action="choose 2",
            )
            current["quality"] = {
                "evidence_class": "modern_verified",
                "verified": True,
            }
            write_jsonl(root / "dataset" / "cases.jsonl", [current])
            write_jsonl(
                root / "dataset" / "runs.jsonl",
                [self._run("1", current["case_id"])],
            )
            write_jsonl(
                root / "dataset" / "expert_actions.jsonl",
                [self._review(current["case_id"], ["choose 2"])],
            )
            summary = evaluate(root / "dataset", root / "output")
            differences = self._rows(root / "output" / "verified_differences.jsonl")

            self.assertEqual(summary["differences"], 1)
            self.assertEqual(summary["expert_references"], {"FAIL": 1})
            self.assertEqual(summary["quality"]["modern_verified"]["differences"], 1)
            self.assertEqual(differences[0]["quality"]["verified"], True)

    def test_separates_direct_predictions_from_advice_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = case(
                "1:f1:c0",
                "1",
                deck={
                    "Demon Form": 1,
                    "Heavy Blade": 1,
                    "Strike": 5,
                    "Defend": 4,
                    "Bash": 1,
                    "Ascender's Bane": 1,
                },
                offered=["Reaper", "Clash"],
                action="choose 1",
            )
            advice = case(
                "2:f1:c0",
                "2",
                deck={"Strike": 5, "Defend": 4, "Bash": 1},
                offered=["Corruption", "Corruption", "Clash"],
                action="choose 0",
            )
            write_jsonl(root / "dataset" / "cases.jsonl", [direct, advice])
            write_jsonl(
                root / "dataset" / "runs.jsonl",
                [
                    self._run("1", direct["case_id"]),
                    self._run("2", advice["case_id"]),
                ],
            )
            write_jsonl(
                root / "dataset" / "expert_actions.jsonl",
                [
                    self._review(direct["case_id"], ["choose 0"]),
                    self._review(advice["case_id"], ["choose 0"]),
                ],
            )

            summary = evaluate(root / "dataset", root / "output")
            results = self._rows(root / "output" / "cases.jsonl")

            self.assertEqual(summary["comparisons"]["DIRECT_DIFFERENT"], 1)
            self.assertEqual(summary["comparisons"]["ADVICE_OBSERVED_ALLOWED"], 1)
            self.assertEqual(results[0]["policy_result"]["allowed_actions"], ["choose 0"])
            self.assertEqual(results[0]["review"]["status"], "PASS")
            self.assertEqual(
                results[1]["policy_result"]["allowed_actions"],
                ["choose 0", "choose 1"],
            )
            self.assertEqual(results[1]["review"]["status"], "COVERED")

    def test_boss_reward_targets_the_next_act_in_evaluation(self):
        current = case(
            "1:f16:c0",
            "1",
            deck={"Strike": 5, "Defend": 4, "Bash": 1},
            offered=["Reaper", "Impervious"],
            action="choose 0",
        )
        current["reward"]["kind"] = "boss_card_reward"

        state = project_state(_request(current, current["state"]["deck_counts"], {}))
        targets = plan_targets(state)

        self.assertEqual(targets.groups[0]["rule"], "NEXT_ACT_BOSS_POOL")
        self.assertNotIn("Hexaghost", targets.targets)

    def test_conservative_run_skips_advice_and_executes_direct_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = case(
                "1:f1:c0",
                "1",
                deck={
                    "Demon Form": 1,
                    "Heavy Blade": 1,
                    "Strike": 5,
                    "Defend": 4,
                    "Bash": 1,
                    "Ascender's Bane": 1,
                },
                offered=["Reaper"],
                action="skip",
            )
            advice = case(
                "2:f1:c0",
                "2",
                deck={"Strike": 5},
                offered=["Test Card A"],
                action="choose 0",
            )
            write_jsonl(root / "dataset" / "cases.jsonl", [direct, advice])
            write_jsonl(
                root / "dataset" / "runs.jsonl",
                [self._run("1", direct["case_id"]), self._run("2", advice["case_id"])],
            )
            summary = evaluate(root / "dataset", root / "output")
            runs = self._rows(root / "output" / "runs.jsonl")

            self.assertEqual(summary["sequential"]["policy_picks"], 1)
            self.assertEqual(summary["sequential"]["policy_skips"], 1)
            self.assertEqual(runs[0]["steps"][0]["action"], "choose 0")
            self.assertEqual(runs[1]["steps"][0]["action"], "skip")

    def test_rejects_illegal_review_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = case(
                "1:f1:c0",
                "1",
                deck={"Strike": 5},
                offered=["Anger"],
                action="skip",
            )
            write_jsonl(root / "dataset" / "cases.jsonl", [current])
            write_jsonl(
                root / "dataset" / "runs.jsonl",
                [self._run("1", current["case_id"])],
            )
            write_jsonl(
                root / "dataset" / "expert_actions.jsonl",
                [self._review(current["case_id"], ["choose 9"])],
            )

            with self.assertRaisesRegex(EvaluationError, "illegal action"):
                evaluate(root / "dataset", root / "output")

    @staticmethod
    def _run(run_id: str, case_id: str) -> dict:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "ascension": 20,
            "outcome": {"status": "DEFEAT", "floor_reached": 6},
            "case_ids": [case_id],
            "transitions": [],
        }

    @staticmethod
    def _review(case_id: str, actions: list[str]) -> dict:
        return {
            "schema_version": 1,
            "case_id": case_id,
            "acceptable_actions": actions,
            "preferred_action": actions[0],
            "confidence": "HIGH",
            "reason": "test label",
            "reviewer": "test",
        }

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
