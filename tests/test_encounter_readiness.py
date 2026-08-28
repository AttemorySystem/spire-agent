from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.contracts import AgentKind, GameState, ScreenState
from spire_agent.tools.map.readiness import _group_summary, _groups, _history, _spec


class EncounterReadinessTests(unittest.TestCase):
    def test_group_cannot_average_away_an_inconclusive_target(self):
        targets = {
            "Gremlin Nob": {
                "target": "Gremlin Nob", "status": "INCONCLUSIVE",
                "survival": 0.875, "attempts": 8, "end_hp_on_win": 7,
            },
            "Lagavulin": {
                "target": "Lagavulin", "status": "SUPPORTED",
                "survival": 1.0, "attempts": 8, "end_hp_on_win": 16,
            },
            "Three Sentries": {
                "target": "Three Sentries", "status": "INCONCLUSIVE",
                "survival": 0.875, "attempts": 8, "end_hp_on_win": 17,
            },
        }

        group = _group_summary({name: 1 / 3 for name in targets}, targets)

        self.assertEqual(group["status"], "INCONCLUSIVE")
        self.assertEqual(group["worst_target"], "Gremlin Nob")

    def test_history_recovers_hallways_and_the_confirmed_bottled_card(self):
        bottle = {
            "schema_version": 1,
            "type": "action",
            "before": {
                "run": {"act": 1, "floor": 9, "room_type": "TreasureRoom", "relics": []},
                "screen": {"type": "COMBAT_REWARD"},
            },
            "after": {
                "run": {"act": 1, "floor": 9, "room_type": "TreasureRoom", "relics": ["Bottled Flame"]},
                "screen": {"type": "GRID"},
            },
            "action": {"command": "choose 0"},
        }
        select = {
            "schema_version": 1,
            "type": "action",
            "before": {
                "run": {"act": 1, "floor": 9, "room_type": "TreasureRoom", "relics": ["Bottled Flame"]},
                "screen": {
                    "type": "GRID",
                    "details": {"cards": [{"name": "Carnage+", "upgrades": 1}]},
                },
            },
            "after": {
                "run": {"act": 1, "floor": 9, "room_type": "TreasureRoom", "relics": ["Bottled Flame"]},
                "screen": {"type": "COMBAT_REWARD"},
            },
            "action": {"command": "choose 0"},
        }
        hallway = {
            "schema_version": 1,
            "type": "action",
            "before": {
                "run": {"act": 2, "floor": 18, "room_type": "MonsterRoom", "relics": ["Bottled Flame"]},
                "screen": {"type": "NONE"},
            },
            "after": {
                "run": {"act": 2, "floor": 18, "room_type": "MonsterRoom", "relics": ["Bottled Flame"]},
                "screen": {"type": "NONE"},
            },
            "action": {"command": "end"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_history.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (bottle, select, hallway)) + "\n")
            rooms, bottles, elites = _history(
                type("Run", (), {"path": Path(directory)})()
            )

        self.assertEqual(rooms, {(2, 18)})
        self.assertEqual(bottles, {"Bottled Flame": ("carnage", 1)})
        self.assertEqual(elites, {})

    def test_history_recovers_the_latest_same_act_elite(self):
        event = {
            "schema_version": 1,
            "type": "action",
            "before": {
                "run": {
                    "act": 1,
                    "floor": 7,
                    "room_type": "MonsterRoomElite",
                    "relics": [],
                },
                "combat": {"monsters": [{"id": "Sentry"}] * 3},
                "screen": {"type": "NONE"},
            },
            "after": {
                "run": {
                    "act": 1,
                    "floor": 7,
                    "room_type": "MonsterRoomElite",
                    "relics": [],
                },
                "combat": {"monsters": [{"id": "Sentry"}] * 3},
                "screen": {"type": "NONE"},
            },
            "action": {"command": "end"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_history.jsonl"
            path.write_text(json.dumps(event) + "\n")
            _, _, elites = _history(
                type("Run", (), {"path": Path(directory)})()
            )

        self.assertEqual(elites, {1: "Three Sentries"})
        self.assertNotIn("Three Sentries", _groups(1, elites[1])["ELITE"])

    def test_act_three_readiness_includes_hallway_encounters(self):
        groups = _groups(3)

        self.assertEqual(
            set(groups), {"WEAK_HALLWAY", "STRONG_HALLWAY", "ELITE"}
        )
        self.assertIn("Spire Growth", groups["STRONG_HALLWAY"])

    def test_requested_hallway_resolves_to_only_the_current_pool(self):
        self.assertEqual(
            set(_groups(2, families=("HALLWAY",), weak_hallways_remaining=1)),
            {"WEAK_HALLWAY"},
        )
        self.assertEqual(
            set(_groups(2, families=("HALLWAY",), weak_hallways_remaining=0)),
            {"STRONG_HALLWAY"},
        )

    def test_spec_marks_the_bottle_and_removes_potions_from_survival_trials(self):
        state = GameState(
            AgentKind.MAP,
            "test",
            ScreenState("MAP"),
            facts={
                "seed": 0,
                "ascension_level": 20,
                "act": 2,
                "floor": 24,
                "current_hp": 40,
                "max_hp": 80,
                "gold": 10,
                "class": "IRONCLAD",
                "deck": ({"name": "Carnage", "upgrades": 1}, {"name": "Strike"}),
                "relics": ({"name": "Bottled Flame"},),
                "potions": ({"name": "Elixir"}, {"name": "Potion Slot"}),
            },
        )

        spec = _spec(state, ("Slavers",), {"Bottled Flame": ("carnage", 1)})

        self.assertTrue(spec["game_state"]["deck"][0]["bottled"])
        self.assertNotIn("bottled", spec["game_state"]["deck"][1])
        self.assertEqual(
            spec["game_state"]["potions"],
            [{"id": "Potion Slot"}, {"id": "Potion Slot"}],
        )


if __name__ == "__main__":
    unittest.main()
