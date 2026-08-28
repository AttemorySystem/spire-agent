from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.tools.winning_path.dataset import (
    DatasetError,
    build_dataset,
    build_expert_dataset,
)


def choice(
    run_id: str,
    sequence: int,
    *,
    deck: dict[str, int],
    offered: list[str],
    picked: str | None,
    floor: int,
    complete: bool = True,
) -> dict:
    return {
        "schema_version": 1,
        "choice_id": f"{run_id}:f{floor}:c{sequence}",
        "context": {
            "act_boss": "Hexaghost",
            "deck_before_counts": deck,
            "hp_after_floor": 60,
            "max_hp_after_floor": 75,
            "gold_after_floor": 100,
            "relics_before_floor_rewards": ["Burning Blood"],
        },
        "decision": {
            "act": 1,
            "floor": floor,
            "kind": "combat_card_reward",
            "offered": offered,
            "picked": picked,
            "skipped": picked is None,
            "used_singing_bowl": False,
        },
        "run": {
            "run_id": run_id,
            "ascension": 20,
            "floor_reached": 6 if complete else None,
            "victory": False if complete else None,
            "heart_kill": False if complete else None,
            "killed_by": "GremlinNob" if complete else None,
        },
    }


def template(action: str, floor: int, source: str = "llm") -> dict:
    return {
        "act": 1,
        "floor": floor,
        "boss": "Hexaghost",
        "final_decision": {
            "action": action,
            "source": source,
            "rationale": "recorded reason",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class CardRewardDatasetTests(unittest.TestCase):
    def test_expert_actions_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = choice(
                "expert:10",
                0,
                deck={"Strike_R": 5, "Defend_R": 4, "Bash": 1},
                offered=["Anger", "Clash"],
                picked="Anger",
                floor=1,
            )
            current["decision"]["selected_card"] = "Anger"
            current["curation"] = {
                "eligible_modern_verified_seed": True,
                "evidence_class": "modern_verified",
                "exclusion_reasons": [],
            }
            source = root / "expert.jsonl"
            write_jsonl(source, [current])

            manifest = build_expert_dataset(source, root / "output")
            build_expert_dataset(source, root / "output")

            labels = self._rows(root / "output" / "expert_actions.jsonl")
            self.assertEqual(manifest["statistics"]["quality"], {"modern_verified": 1})
            self.assertEqual(labels[0]["acceptable_actions"], ["choose 0"])
            self.assertEqual(labels[0]["confidence"], "HIGH")

    def test_builds_snapshots_and_sequential_fixed_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "source" / "10"
            first = choice(
                "10",
                0,
                deck={"Strike": 5, "Defend": 4, "Bash": 1},
                offered=["Anger", "Clash", "Cleave"],
                picked="Anger",
                floor=1,
            )
            second = choice(
                "10",
                1,
                deck={"Strike": 4, "Defend": 4, "Bash+": 1, "Anger": 1},
                offered=["Shrug It Off", "Havoc", "Flex"],
                picked=None,
                floor=2,
            )
            write_jsonl(run_dir / "card_choices.jsonl", [first, second])
            write_jsonl(
                run_dir / "template_choices.jsonl",
                [template("choose 0", 1), template("skip", 2, "winning_path")],
            )

            manifest = build_dataset(root / "source", root / "output")

            self.assertEqual(manifest["statistics"]["cases"], 2)
            cases = self._rows(root / "output" / "cases.jsonl")
            self.assertEqual(cases[0]["observed_action"]["source"], "llm")
            self.assertEqual(cases[0]["provenance"]["file"], "10/card_choices.jsonl")
            runs = self._rows(root / "output" / "runs.jsonl")
            self.assertEqual(
                runs[0]["transitions"][0]["fixed_deck_delta"],
                {
                    "removed": {"Bash": 1, "Strike": 1},
                    "added": {"Bash+": 1},
                },
            )

    def test_output_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = choice(
                "10",
                0,
                deck={"Strike": 5},
                offered=["Anger"],
                picked="Anger",
                floor=1,
                complete=False,
            )
            write_jsonl(root / "source" / "10" / "card_choices.jsonl", [row])

            build_dataset(root / "source", root / "first")
            build_dataset(root / "source", root / "second")

            for name in ("cases.jsonl", "runs.jsonl", "manifest.json", "report.md"):
                self.assertEqual(
                    (root / "first" / name).read_bytes(),
                    (root / "second" / name).read_bytes(),
                )

    def test_rejects_pick_outside_offer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = choice(
                "10",
                0,
                deck={"Strike": 5},
                offered=["Anger"],
                picked="Clash",
                floor=1,
            )
            write_jsonl(root / "source" / "10" / "card_choices.jsonl", [row])

            with self.assertRaisesRegex(DatasetError, "is not offered"):
                build_dataset(root / "source", root / "output")

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
