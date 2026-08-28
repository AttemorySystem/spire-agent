from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.tools.winning_path.combat_eval import (
    _paired_interval,
    apply_reward_changes,
    discover_checkpoints,
    encounter_name,
    evaluate_battles,
)


class WinningPathCombatEvaluationTests(unittest.TestCase):
    def test_parallel_evaluation_preserves_checkpoint_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "evaluator"
            binary.write_text("fixture", encoding="utf-8")
            checkpoints = [
                {
                    "checkpoint_id": name,
                    "run_id": name,
                    "kind": "FATAL_ENCOUNTER",
                    "act": 1,
                    "floor": index,
                    "encounter": "Gremlin Nob",
                    "clean": False,
                }
                for index, name in enumerate(("first", "second"), 1)
            ]
            (root / "checkpoints.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in checkpoints),
                encoding="utf-8",
            )
            (root / "inventory.json").write_text("{}", encoding="utf-8")

            evaluate_battles(
                root,
                binary,
                simulations=1,
                worlds=1,
                max_time_ms=1,
                max_decisions=1,
                timeout=1,
                jobs=2,
            )

            results = [
                json.loads(line)
                for line in (root / "combat_results.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["checkpoint_id"] for row in results], ["first", "second"]
            )

    def test_paired_interval_tracks_direction_and_keeps_small_samples_unstable(self):
        self.assertEqual(_paired_interval([1]), (-1.0, 1.0))
        lower, upper = _paired_interval([1] * 16)
        self.assertGreater(lower, 0)
        self.assertGreater(upper, 0)

    def test_applies_changed_reward_to_exact_checkpoint_deck(self):
        historical = [
            {"id": "Strike_R", "upgrades": 0},
            {"id": "Carnage", "upgrades": 1},
        ]
        candidate, warnings, changed = apply_reward_changes(
            historical,
            [
                {
                    "case_id": "run:f7:c0",
                    "historical_card": "Carnage",
                    "candidate_card": "Immolate",
                }
            ],
        )

        self.assertEqual(
            candidate,
            [
                {"id": "Strike_R", "upgrades": 0},
                {"id": "Immolate", "upgrades": 0},
            ],
        )
        self.assertEqual(changed, ["run:f7:c0"])
        self.assertEqual(warnings[0]["kind"], "REMOVED_UPGRADED_HISTORICAL_PICK")

    def test_extracts_passed_boss_and_fatal_encounter(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "7"
            run_dir.mkdir()
            self._state(run_dir / "15-0.json", 1, 16, "MonsterRoomBoss", ["Hexaghost"])
            (run_dir / "post-combat-act1-floor16-combat-agent").write_text("ok")
            self._state(
                run_dir / "28-0.json",
                2,
                29,
                "MonsterRoomElite",
                ["SlaverBlue", "SlaverBoss", "SlaverRed"],
            )
            checkpoints, issues = discover_checkpoints(
                run_dir,
                {
                    "run_id": "7",
                    "outcome": {
                        "status": "DEFEAT",
                        "floor_reached": 29,
                        "killed_by": "SlaverBlue, SlaverBoss, SlaverRed",
                    },
                },
            )

        self.assertEqual(issues, [])
        self.assertEqual(
            [(row["kind"], row["encounter"]) for row in checkpoints],
            [("PASSED_BOSS", "Hexaghost"), ("FATAL_ENCOUNTER", "Slavers")],
        )

    def test_maps_recorded_multi_monster_encounters(self):
        self.assertEqual(
            encounter_name(
                ["Cultist", "Cultist", "AwakenedOne"], "Awakened One", "FATAL_ENCOUNTER"
            ),
            "Awakened One",
        )
        self.assertEqual(
            encounter_name(["The Collector"], "Collector", "PASSED_BOSS"),
            "Collector",
        )

    @staticmethod
    def _state(path: Path, act: int, floor: int, room: str, enemies: list[str]) -> None:
        payload = {
            "game_state": {
                "seed": 1,
                "ascension_level": 20,
                "act": act,
                "floor": floor,
                "current_hp": 50,
                "max_hp": 75,
                "gold": 100,
                "class": "IRONCLAD",
                "room_type": room,
                "deck": [{"id": "Strike_R", "upgrades": 0}],
                "relics": [{"id": "Burning Blood", "counter": -1}],
                "potions": [{"id": "Potion Slot"}],
                "combat_state": {
                    "turn": 1,
                    "hand": [],
                    "monsters": [
                        {"id": name, "name": name, "current_hp": 10} for name in enemies
                    ],
                },
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
