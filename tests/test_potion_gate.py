from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.contracts import AgentKind, GameState, ScreenState
from spire_agent.extensions import RunDirectory
from spire_agent.tools.mcts import MCTSResult, PotionGate
from spire_agent.tools.mcts.potion_gate import assess_risk, potion_slots


def combat_state(*, heart=False, potion_count=5):
    potions = [
        {
            "id": f"TestPotion{index}",
            "name": f"Test Potion {index}",
            "can_use": True,
            "requires_target": False,
        }
        for index in range(potion_count)
    ]
    return GameState(
        AgentKind.COMBAT,
        "seed:a4:f56:boss:combat" if heart else "seed:a1:f7:elite:combat",
        ScreenState("NONE", commands=("play", "end", "potion")),
        facts={
            "act": 4 if heart else 1,
            "act_boss": "Corrupt Heart" if heart else "Slime Boss",
            "room_type": "MonsterRoomBoss" if heart else "MonsterRoomElite",
            "current_hp": 50,
            "max_hp": 100,
            "potions": potions,
        },
        combat={
            "player": {"current_hp": 50, "max_hp": 100},
            "hand": ({"id": "Strike_R", "name": "Strike"},),
            "monsters": ({"id": "TestMonster", "current_hp": 100},),
        },
    )


def result(end_hp, *, credible=True, search_id="test"):
    return MCTSResult(
        "end",
        None,
        {
            "search_id": search_id,
            "credible_win_evidence": credible,
            "risk": {
                "winSamples": 100 if credible else 0,
                "lossSamples": 20,
                "winSampleRate": 0.8 if credible else 0.0,
                "meanBestWinEndHp": end_hp,
                "expectedEndHpOnWin": end_hp,
                "visits": 120,
            },
        },
    )


class FakeSearch:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def __call__(self, state, **kwargs):
        slots = tuple(kwargs.get("potion_slots") or ())
        self.calls.append((slots, kwargs.get("probe"), kwargs.get("search_role")))
        value = self.outcomes.get(slots, self.outcomes.get("default", 5))
        return result(value, search_id=f"search-{len(self.calls)}")


class PotionGateTests(unittest.TestCase):
    def test_smoke_bomb_escapes_a_non_boss_without_a_credible_win(self):
        state = combat_state(potion_count=1)
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={
                **state.facts,
                "potions": ({"id": "SmokeBomb", "name": "Smoke Bomb", "can_use": True},),
            },
            combat=state.combat,
        )
        with tempfile.TemporaryDirectory() as directory:
            runs = RunDirectory(Path(directory) / "runs")
            runs.bind("ABC123")
            selected = PotionGate(runs).select(
                state, result(1, credible=False), FakeSearch({})
            )

        self.assertEqual(selected.command, "potion use 0")
        self.assertEqual(selected.metrics["potion_gate"], "SMOKE_BOMB_ESCAPE")

    def test_pair_is_probed_when_one_potion_only_reduces_emergency_to_danger(self):
        state = combat_state(potion_count=2)
        search = FakeSearch({(0,): 30, (1,): 25, (0, 1): 45, "default": 10})
        with tempfile.TemporaryDirectory() as directory:
            runs = RunDirectory(Path(directory) / "runs")
            runs.bind("ABC123")
            selected = PotionGate(runs).select(
                state, result(5, credible=False), search
            )

        self.assertEqual(search.calls[-1][0], (0, 1))
        self.assertEqual(selected.metrics["search_id"], "search-4")

    def test_inventory_supports_all_five_slots(self):
        self.assertEqual(potion_slots(combat_state()), (0, 1, 2, 3, 4))

    def test_risk_uses_expected_hp_instead_of_optimistic_best_hp(self):
        state = combat_state(potion_count=1)
        result = MCTSResult(
            "end",
            None,
            {
                "credible_win_evidence": True,
                "risk": {
                    "winSamples": 100,
                    "meanBestWinEndHp": 45,
                    "expectedEndHpOnWin": 5,
                },
            },
        )

        risk = assess_risk(state, result)

        self.assertEqual(risk["level"], "EMERGENCY")
        self.assertEqual(risk["expected_end_hp"], 5)

    def test_danger_checks_five_singles_and_releases_only_one(self):
        state = combat_state()
        baseline = result(30, search_id="baseline")
        search = FakeSearch(
            {
                (0,): 40,
                (1,): 32,
                (2,): 31,
                (3,): 30,
                (4,): 29,
                "default": 40,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            runs = RunDirectory(Path(directory) / "runs")
            runs.bind("ABC123")
            selected = PotionGate(runs).select(state, baseline, search)
            trace = json.loads(
                (runs.path / "potion_decisions.jsonl").read_text().splitlines()[0]
            )

        self.assertEqual(selected.metrics["search_id"], "search-6")
        self.assertEqual(search.calls[-1], ((0,), None, "potion_final"))
        self.assertEqual(
            [call[0] for call in search.calls[:-1]],
            [(0,), (1,), (2,), (3,), (4,)],
        )
        self.assertEqual(trace["selected_slots"], [0])

    def test_emergency_uses_pair_only_when_no_single_is_enough(self):
        state = combat_state()
        baseline = result(5, credible=False, search_id="baseline")
        outcomes = {
            (0,): 12,
            (1,): 14,
            (2,): 6,
            (3,): 6,
            (4,): 6,
            (0, 1): 35,
            "default": 10,
        }
        search = FakeSearch(outcomes)
        with tempfile.TemporaryDirectory() as directory:
            runs = RunDirectory(Path(directory) / "runs")
            runs.bind("ABC123")
            selected = PotionGate(runs).select(state, baseline, search)
            trace = json.loads(
                (runs.path / "potion_decisions.jsonl").read_text().splitlines()[0]
            )

        pair_probes = [call for call in search.calls if call[1] is True and len(call[0]) == 2]
        self.assertEqual(len(pair_probes), 10)
        self.assertEqual(search.calls[-1], ((0, 1), None, "potion_final"))
        self.assertEqual(selected.metrics["search_id"], "search-16")
        self.assertEqual(trace["selected_slots"], [0, 1])
        self.assertEqual(trace["reason"], "PAIR_REQUIRED_FOR_EMERGENCY")

    def test_heart_releases_all_five_without_counterfactuals(self):
        state = combat_state(heart=True)
        baseline = result(5, credible=False)
        search = FakeSearch({"default": 40})
        with tempfile.TemporaryDirectory() as directory:
            runs = RunDirectory(Path(directory) / "runs")
            runs.bind("ABC123")
            PotionGate(runs).select(state, baseline, search)

        self.assertEqual(
            search.calls,
            [((0, 1, 2, 3, 4), None, "potion_final")],
        )


if __name__ == "__main__":
    unittest.main()
