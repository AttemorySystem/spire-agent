from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from spire_agent.adapters.gym_sts import GymStsObservationAdapter
from spire_agent.extensions.run_directory import RunDirectory
from spire_agent.tools.mcts import (
    CombatMCTS,
    MCTSError,
    encode_state,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture_state():
    raw = json.loads((ROOT / "tests" / "fixtures" / "combat.json").read_text())
    state = GymStsObservationAdapter().adapt(raw, sts_seed="FIXTURESEED")
    return raw, state


def gambling_state():
    raw, _ = fixture_state()
    hand = raw["game_state"]["combat_state"]["hand"]
    raw["available_commands"] = ["choose", "confirm", "state"]
    raw["game_state"].update(
        {
            "screen_type": "HAND_SELECT",
            "current_action": "GamblingChipAction",
            "choice_list": [card["name"] for card in hand],
            "screen_state": {
                "can_pick_zero": True,
                "max_cards": 10,
                "selected": [],
                "hand": hand,
            },
        }
    )
    return GymStsObservationAdapter().adapt(raw, sts_seed="GAMBLESEED")


def hologram_state():
    raw, _ = fixture_state()
    raw["available_commands"] = ["choose", "state"]
    raw["game_state"].update(
        {
            "screen_type": "GRID",
            "current_action": "BetterDiscardPileToHandAction",
            "choice_list": ["Strike"],
            "screen_state": {
                "num_cards": 1,
                "cards": [{"id": "Strike_B", "name": "Strike"}],
            },
        }
    )
    return GymStsObservationAdapter().adapt(raw, sts_seed="HOLOGRAMSEED")


class CombatMCTSToolTests(unittest.TestCase):
    def test_encoder_marks_a_live_hologram_selector_for_search(self):
        self.assertEqual(
            encode_state(hologram_state())["mcts_card_select"],
            {"task": "HOLOGRAM", "copy_count": 1},
        )

    def test_gambling_chip_subset_becomes_choose_then_confirm_continuation(self):
        state = gambling_state()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "battle-sim"
            binary.touch(mode=0o755)
            run_directory = RunDirectory(root / "runs")
            run_directory.bind("ABC123")
            search = CombatMCTS(binary, run_directory)
            output = {
                "protocolVersion": 1,
                "rootCommand": "choose (Strike), (Bash)",
                "followUp": None,
                "score": 1.0,
                "rootActions": [
                    {"action": "choose (Strike), (Bash)", "winSamples": 1}
                ],
            }
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(output), stderr=""
            )

            with patch(
                "spire_agent.tools.mcts.tool.subprocess.run", return_value=completed
            ):
                result = search.choose(state)

        self.assertEqual(
            encode_state(state)["mcts_card_select"],
            {"task": "GAMBLE", "copy_count": 1},
        )
        self.assertEqual(result.command, "choose 0")
        self.assertEqual(result.follow_up["completionCommand"], "confirm")
        self.assertEqual([card["name"] for card in result.follow_up["cards"]], ["Bash"])

    def test_encoder_reconstructs_battle_sim_input_without_v1_metadata(self):
        raw, state = fixture_state()

        encoded = encode_state(state)

        self.assertEqual(encoded["available_commands"], raw["available_commands"])
        self.assertEqual(encoded["ready_for_command"], raw["ready_for_command"])
        self.assertEqual(encoded["in_game"], raw["in_game"])
        for key, value in raw["game_state"].items():
            self.assertEqual(encoded["game_state"][key], value)
        self.assertNotIn("sts_seed", encoded["game_state"])
        self.assertNotIn("bridge", encoded["game_state"])
        self.assertNotIn("replay_boundary_key", encoded["game_state"])
        self.assertNotIn("replay_rng_state", encoded["game_state"])

    def test_adapter_accepts_all_loss_search_and_records_one_file(self):
        raw, _ = fixture_state()
        raw["game_state"]["class"] = "DEFECT"
        raw["game_state"]["combat_state"]["player"]["orbs"] = [
            {"name": "Lightning", "passive_amount": 3, "evoke_amount": 8},
            {"name": "Frost", "passive_amount": 2, "evoke_amount": 5},
            {"name": "Orb Slot", "passive_amount": 0, "evoke_amount": 0},
        ]
        state = GymStsObservationAdapter().adapt(raw, sts_seed="FIXTURESEED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "battle-sim"
            binary.touch(mode=0o755)
            run_directory = RunDirectory(root / "runs")
            run_directory.bind("ABC123")
            search = CombatMCTS(binary, run_directory)
            output = {
                "protocolVersion": 1,
                "rootCommand": "end",
                "followUp": None,
                "score": -100.0,
                "rootSelectionPolicy": "safe_replan_no_win_fallback",
                "searchStopReason": "simulation_limit",
                "credibleWinEvidence": False,
                "rootActions": [
                    {"action": "end", "winSampleRate": 0.0},
                    {
                        "action": "play 2 1",
                        "winSampleRate": 0.0,
                        "lowerQuartileWinSampleRate": 0.0,
                        "expectedEndHpOnWin": 0.0,
                        "visits": 120,
                    },
                ],
            }
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(output),
                stderr="",
            )

            with patch(
                "spire_agent.tools.mcts.tool.subprocess.run",
                return_value=completed,
            ):
                result = search.choose(state)

            self.assertEqual(result.command, "end")
            self.assertFalse(result.metrics["credible_win_evidence"])
            logs = list((run_directory.path / "mcts").glob("*.json"))
            self.assertEqual(len(logs), 1)
            record = json.loads(logs[0].read_text())
            self.assertEqual(record["status"], "success")
            self.assertEqual(record["result"]["command"], "end")
            human = (run_directory.path / "mcts.log").read_text()
            self.assertIn("Monsters:", human)
            self.assertIn("Player:", human)
            self.assertIn("Orbs: [Lightning, Frost, Orb Slot]", human)
            self.assertIn("Relics:", human)
            self.assertIn("Counters: cards_played=", human)
            self.assertIn("CardManager:", human)
            self.assertIn("play 2 1 Impervious -> [1] Sentry", human)
            self.assertIn("win=0.00%", human)
            self.assertIn("Chosen: end turn", human)

    def test_protocol_failure_is_recorded_and_fails_closed(self):
        _, state = fixture_state()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "battle-sim"
            binary.touch(mode=0o755)
            run_directory = RunDirectory(root / "runs")
            run_directory.bind("ABC123")
            search = CombatMCTS(binary, run_directory)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({"actions": ["end"]}),
                stderr="",
            )

            with patch(
                "spire_agent.tools.mcts.tool.subprocess.run",
                return_value=completed,
            ):
                with self.assertRaisesRegex(
                    MCTSError, "protocolVersion"
                ):
                    search.choose(state)

            log = next((run_directory.path / "mcts").glob("*.json"))
            record = json.loads(log.read_text())
            self.assertEqual(record["status"], "error")
            self.assertIn(
                "Search: ERROR",
                (run_directory.path / "mcts.log").read_text(),
            )

    def test_native_output_flood_is_bounded_and_fails_closed(self):
        _, state = fixture_state()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "battle-sim"
            binary.touch(mode=0o755)
            run_directory = RunDirectory(root / "runs")
            run_directory.bind("ABC123")
            search = CombatMCTS(binary, run_directory)

            def flood(command, **kwargs):
                kwargs["stdout"].write("x" * 11)
                return subprocess.CompletedProcess(command, 0)

            with (
                patch("spire_agent.tools.mcts.tool._MAX_NATIVE_OUTPUT_BYTES", 10),
                patch("spire_agent.tools.mcts.tool._MAX_DIAGNOSTIC_BYTES", 4),
                patch("spire_agent.tools.mcts.tool.subprocess.run", side_effect=flood),
            ):
                with self.assertRaisesRegex(MCTSError, "output exceeded"):
                    search.choose(state)

            record = json.loads(next(
                (run_directory.path / "mcts").glob("*.json")
            ).read_text())
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["stdout"], "xxxx\n...[native output truncated]")

    def test_authorized_potion_command_and_slot_are_forwarded(self):
        raw, _ = fixture_state()
        raw["game_state"]["potions"][0] = {
            "requires_target": True,
            "can_use": True,
            "can_discard": True,
            "name": "Explosive Potion",
            "id": "Explosive Potion",
        }
        state = GymStsObservationAdapter().adapt(raw, sts_seed="FIXTURESEED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "battle-sim"
            binary.touch(mode=0o755)
            run_directory = RunDirectory(root / "runs")
            run_directory.bind("ABC123")
            search = CombatMCTS(binary, run_directory)
            output = {
                "protocolVersion": 1,
                "rootCommand": "potion use 0",
                "followUp": None,
                "score": 1.0,
                "credibleWinEvidence": True,
                "rootActions": [
                    {
                        "action": "potion use 0",
                        "winSamples": 10,
                        "winSampleRate": 1.0,
                        "meanBestWinEndHp": 68,
                    }
                ],
            }
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(output), stderr=""
            )

            with patch(
                "spire_agent.tools.mcts.tool.subprocess.run", return_value=completed
            ) as invoked:
                result = search.choose(
                    state,
                    potion_slots=(0,),
                    probe=True,
                    search_role="potion_probe",
                )

            command = invoked.call_args.args[0]
            self.assertIn("potion_slots=0", command)
            self.assertIn("adaptive_max_time_ms=2000", command)
            self.assertIn("adaptive_max_simulations=20000", command)
            self.assertEqual(invoked.call_args.kwargs["timeout"], 34)
            self.assertEqual(result.command, "potion use 0")
            self.assertEqual(result.metrics["allowed_potion_slots"], (0,))
            self.assertEqual(result.metrics["risk"]["meanBestWinEndHp"], 68)

    def test_authorized_potion_search_has_a_hard_memory_budget(self):
        _, state = fixture_state()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "battle-sim"
            binary.touch(mode=0o755)
            run_directory = RunDirectory(root / "runs")
            run_directory.bind("ABC123")
            search = CombatMCTS(binary, run_directory)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "protocolVersion": 1,
                        "rootCommand": "end",
                        "followUp": None,
                        "score": 0.0,
                        "credibleWinEvidence": True,
                        "rootActions": [],
                    }
                ),
                stderr="",
            )

            with patch(
                "spire_agent.tools.mcts.tool.subprocess.run", return_value=completed
            ) as invoked:
                search.choose(state, potion_slots=(0, 1))

            command = invoked.call_args.args[0]
            self.assertEqual(command[2:5], ["50000", "12", "5000"])
            self.assertIn("adaptive_max_time_ms=5000", command)
            self.assertIn("adaptive_max_simulations=50000", command)


if __name__ == "__main__":
    unittest.main()
