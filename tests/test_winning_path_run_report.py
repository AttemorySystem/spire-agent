from __future__ import annotations

import unittest

from spire_agent.tools.winning_path.run_report import render_report


class WinningPathRunReportTests(unittest.TestCase):
    def test_renders_choice_deck_difference_and_paired_trials(self):
        run = {
            "run_id": "7",
            "outcome": {
                "status": "DEFEAT",
                "floor_reached": 16,
                "killed_by": "Hexaghost",
            },
        }
        choices = [
            {
                "act": 1,
                "floor": 1,
                "offered": ["Body Slam", "Shrug It Off"],
                "historical_action": "choose 0",
                "candidate_action": "choose 1",
                "changed": True,
                "direct": True,
                "policy": {
                    "policy": "TRANSITION_TAKE",
                    "reason": "Shrug It Off wins ranking",
                },
            }
        ]
        checkpoint = {
            "checkpoint_id": "7:a1:f16:fatal_encounter",
            "kind": "FATAL_ENCOUNTER",
            "act": 1,
            "floor": 16,
            "encounter": "Hexaghost",
            "game_state": {
                "current_hp": 30,
                "max_hp": 75,
                "relics": [{"name": "Burning Blood"}],
            },
            "historical_deck": [{"id": "Body Slam", "upgrades": 1}],
            "candidate_deck": [{"id": "Shrug It Off", "upgrades": 0}],
            "changed_choice_ids": ["7:f1:c0"],
            "warnings": [],
        }
        result = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "status": "OK",
            "historical": self._aggregate(1, 2, 15),
            "candidate": self._aggregate(2, 2, 20),
            "win_rate_delta": 0.5,
            "paired_delta_95_interval": [-0.1, 1.0],
            "classification": "SCREEN_ONLY",
            "paired_outcomes": {"LW": 1, "WW": 1},
            "trials": [
                {
                    "world": 0,
                    "historical": self._trial(False, 0),
                    "candidate": self._trial(True, 20),
                }
            ],
        }

        report = render_report(run, choices, [checkpoint], [result])

        self.assertIn("choose 0: Body Slam", report)
        self.assertIn("choose 1: Shrug It Off", report)
        self.assertIn("Removed by Winning Path: Body Slam+ x1", report)
        self.assertIn("Added by Winning Path: Shrug It Off x1", report)
        self.assertIn("1/2 (50.0%)", report)
        self.assertIn("LW=1, WW=1", report)
        self.assertIn("| 0 | loss, HP 0, turn 10 | win, HP 20, turn 10 |", report)

    @staticmethod
    def _aggregate(wins: int, attempts: int, hp: int) -> dict:
        return {
            "wins": wins,
            "attempts": attempts,
            "win_rate": wins / attempts,
            "expected_end_hp_on_win": hp,
        }

    @staticmethod
    def _trial(won: bool, hp: int) -> dict:
        return {
            "won": won,
            "completed": True,
            "ending_hp": hp,
            "turns": 10,
            "stop_reason": "terminal",
        }


if __name__ == "__main__":
    unittest.main()
