from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.context import GameContext
from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    ExecutionResult,
    GameState,
    RoutedDecision,
)
from spire_agent.extensions import CardChoiceRecorder, RunDirectory
from spire_agent.extensions.replay import ReplayJournal
from spire_agent.subagents.build import create_build_agent
from spire_agent.subagents.build_context import RUN_CONSTRUCTION_KEY
from spire_agent.tools.winning_path import (
    WINNING_PATH_REVIEW_KEY,
    create_card_picker,
)
from spire_agent.tools.winning_path.card_policy import CardRewardError
from spire_agent.tools.winning_path.catalog import load_default_catalog
from spire_agent.tools.winning_path.contracts import DecisionState
from spire_agent.tools.winning_path.templates import analyze_templates

from tests.test_winning_path import _request
from tests.test_card_reward_policy import FakeLLM, card_state, request
from tests.test_build_agent import build_state, request as build_request


def _defect_reward(
    deck,
    offered,
    *,
    act=2,
    floor=18,
    boss="Collector",
    relics=("Cracked Core",),
):
    state = build_state(
        "CARD_REWARD",
        commands=("choose", "skip"),
        choices=tuple(name.casefold() for name in offered),
        details={"cards": tuple({"name": name} for name in offered)},
        facts={
            "class": "DEFECT",
            "act": act,
            "floor": floor,
            "act_boss": boss,
            "deck": tuple({"name": name} for name in deck),
            "relics": tuple({"name": name} for name in relics),
        },
    )
    return create_card_picker("DEFECT").review(build_request(state))


class WinningPathLiveTests(unittest.TestCase):
    def test_defect_genetic_algorithm_is_limited_to_act_one(self):
        late = _defect_reward(
            ("Zap", "Dualcast", "Coolheaded", "Ball Lightning"),
            ("Genetic Algorithm", "Leap", "Cold Snap"),
            act=2,
            floor=18,
        )
        early = _defect_reward(
            ("Zap", "Dualcast", "Coolheaded", "Ball Lightning"),
            ("Genetic Algorithm", "Leap", "Cold Snap"),
            act=1,
            floor=8,
            boss="Hexaghost",
        )

        self.assertEqual(
            late["candidates"][0]["hard_constraints"][0]["type"], "FORBIDDEN"
        )
        self.assertNotIn(0, late["allowed_choice_ids"])
        self.assertFalse(early["candidates"][0]["rejected"])

    def test_defect_existing_auto_shields_does_not_promote_another_block_card(self):
        result = _defect_reward(
            (
                "Zap", "Dualcast", "Strike", "Strike", "Strike", "Strike",
                "Defend", "Defend", "Defend", "Defend", "Auto-Shields",
            ),
            ("Beam Cell", "Leap", "Ball Lightning"),
            act=1,
            floor=2,
            boss="The Guardian",
        )

        self.assertEqual(result["command"], "choose 2")
        self.assertEqual(result["policy"], "EXPERT_EXPERIENCE")
        self.assertEqual(result["candidates"][1]["transition"]["level"], "NONE")

    def test_defect_consume_requires_nonstarter_orb_supply(self):
        result = _defect_reward(
            (
                "Zap", "Dualcast", "Strike", "Strike", "Strike", "Strike",
                "Defend", "Defend", "Defend", "Defend", "Capacitor",
            ),
            ("Tempest", "Ball Lightning", "Consume"),
            act=1,
            floor=11,
            boss="Hexaghost",
        )

        self.assertEqual(result["command"], "choose 1")
        self.assertEqual(
            result["candidates"][2]["hard_constraints"][0]["type"],
            "MISSING_PREREQUISITE",
        )

    def test_defectxx08_does_not_treat_consume_as_slot_free_focus(self):
        deck = (
            "Aggregate", "Ascender's Bane", "Ball Lightning", "Beam Cell",
            "Blizzard", "Chaos", "Coolheaded", "Core Surge",
            "Doom and Gloom", "Dualcast", "Genetic Algorithm", "Hologram",
            "Hologram", "Overclock", "Rainbow", "Reinforced Body",
            "Reinforced Body", "Static Discharge", "Streamline", "Zap",
        )
        result = _defect_reward(
            deck,
            ("Claw", "Stack", "Consume"),
            act=2,
            floor=28,
            boss="Automaton",
        )

        self.assertEqual(result["command"], "skip")
        self.assertEqual(result["candidates"][2]["template"]["level"], "NONE")
        self.assertEqual(
            result["candidates"][2]["hard_constraints"][0]["type"],
            "MISSING_PREREQUISITE",
        )

    def test_defect_consume_still_advances_orb_capacity(self):
        result = _defect_reward(
            ("Zap", "Dualcast", "Coolheaded", "Cold Snap", "Capacitor"),
            ("Claw", "Stack", "Consume"),
            act=2,
            floor=20,
            boss="Automaton",
        )

        self.assertEqual(result["command"], "choose 2")
        self.assertEqual(
            result["candidates"][2]["template"]["observed_level"],
            "CORE_ACTIVATION",
        )
        self.assertEqual(
            result["candidates"][2]["template"]["route_id"],
            "orb_capacity_scaling",
        )

    def test_defect_data_disk_and_two_frost_sources_complete_focus_frost(self):
        result = _defect_reward(
            (
                "Zap", "Dualcast", "Strike", "Strike", "Strike", "Strike",
                "Defend", "Defend", "Defend", "Defend", "Coolheaded",
            ),
            ("Cold Snap", "Go for the Eyes", "Stack"),
            act=2,
            floor=24,
            boss="Automaton",
            relics=("Cracked Core", "Data Disk"),
        )

        self.assertEqual(result["command"], "choose 0")
        self.assertEqual(
            result["candidates"][0]["template"]["level"], "CORE_ACTIVATION"
        )

    def test_defect_fourth_frost_source_advances_density_target(self):
        deck = (
            "Zap", "Dualcast", "Coolheaded", "Cold Snap", "Glacier",
            "Consume",
        )
        result = _defect_reward(
            deck,
            ("Beam Cell", "Coolheaded", "Recursion"),
            act=3,
            floor=48,
            boss="Awakened One",
        )

        self.assertEqual(result["command"], "choose 1")
        self.assertEqual(
            result["candidates"][1]["template"]["level"], "COMMITTED_PROGRESS"
        )

        full = _defect_reward(
            (*deck, "Coolheaded"),
            ("Beam Cell", "Coolheaded", "Recursion"),
            act=3,
            floor=48,
            boss="Awakened One",
        )
        self.assertEqual(full["command"], "skip")
        self.assertEqual(full["candidates"][1]["template"]["level"], "NONE")

    def test_defect_does_not_count_partial_required_slot_progress(self):
        catalog = load_default_catalog("DEFECT")
        blizzard = next(
            row
            for row in catalog["knowledge"]["modules"]
            if row["module_id"] == "blizzard_frost_cycle"
        )
        state = DecisionState(
            run={"character": "DEFECT", "act": 2, "floor": 18},
            deck={
                "counts": {"Blizzard": 1, "Cold Snap": 1, "Coolheaded": 1},
                "physical_size": 3,
            },
            assets={"relics": []},
            route={},
            reward={"offered_cards": [{"id": "Chill", "upgrades": 0}]},
        )
        result = analyze_templates(
            state,
            {
                "knowledge": {"modules": [blizzard], "dominant_cards": []},
                "derived": catalog["derived"],
            },
        )

        self.assertEqual(result[0]["level"], "NONE")

    def test_defectxx03_prefers_frost_density_over_turbo(self):
        result = _defect_reward(
            (
                "Ascender's Bane", "Strike", "Strike", "Strike", "Defend",
                "Defend", "Defend", "Defend", "Zap", "Dualcast",
                "Coolheaded", "Ball Lightning", "Seek", "Sweeping Beam",
                "Consume", "Cold Snap", "Machine Learning", "Rebound",
            ),
            ("TURBO", "Coolheaded", "Scrape"),
            act=1,
            floor=14,
            boss="Hexaghost",
            relics=(
                "Cracked Core", "Strike Dummy", "Matryoshka",
                "Fossilized Helix",
            ),
        )

        self.assertEqual(result["command"], "choose 1")
        self.assertEqual(
            result["candidates"][1]["template"]["level"], "COMMITTED_PROGRESS"
        )

    def test_defect_reprogram_requires_a_supported_physical_plan(self):
        result = _defect_reward(
            ("Zap", "Dualcast", "Glacier", "Chill"),
            ("Reprogram", "Stack", "Melter"),
        )
        reprogram = result["candidates"][0]
        self.assertTrue(reprogram["rejected"])
        self.assertEqual(
            reprogram["hard_constraints"][0]["type"], "MISSING_PREREQUISITE"
        )
        self.assertNotEqual(result.get("command"), "choose 0")
        self.assertNotIn(0, result["allowed_choice_ids"])

    def test_defect_reprogram_completes_a_supported_physical_plan(self):
        result = _defect_reward(
            ("Zap", "Dualcast", "Beam Cell", "Claw"),
            ("Reprogram", "Stack", "Leap"),
        )
        self.assertEqual(result["command"], "choose 0")
        self.assertEqual(
            result["candidates"][0]["template"]["level"], "CORE_ACTIVATION"
        )

    def test_defect_reprogram_conflicts_with_committed_focus(self):
        result = _defect_reward(
            ("Zap", "Dualcast", "Glacier", "Chill", "Beam Cell"),
            ("Reprogram", "Stack", "Leap"),
        )
        reprogram = result["candidates"][0]
        self.assertTrue(reprogram["rejected"])
        self.assertEqual(
            reprogram["hard_constraints"][0]["type"], "HARD_RESOURCE_CONFLICT"
        )

    def test_uses_the_existing_card_picker_contract_for_a_direct_pick(self):
        picker = create_card_picker("IRONCLAD")
        decision = create_build_agent(FakeLLM({}), picker).decide(_request())

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(
            decision.payload["card_choice_review"]["picker_id"],
            "ironclad.winning_path",
        )
        self.assertEqual(
            decision.payload[WINNING_PATH_REVIEW_KEY]["winning_path"]["mode"],
            "LIVE_POLICY",
        )
        self.assertEqual(
            decision.payload[RUN_CONSTRUCTION_KEY]["confirmed_selection"]["card"],
            "Shrug It Off",
        )

    def test_llm_cannot_escape_its_shortlist(self):
        llm = FakeLLM(
            {
                "action": "choose",
                "choice_id": 2,
                "targets": [],
                "reason": "outside the shortlist",
            }
        )
        picker = create_card_picker("IRONCLAD")
        with self.assertRaisesRegex(CardRewardError, "outside allowed choice ids"):
            create_build_agent(llm, picker).decide(
                request(card_state(("Corruption", "Dark Embrace", "Clash")))
            )
        self.assertEqual(len(llm.requests), 1)

    def test_choice_log_is_compatible_with_the_existing_dataset(self):
        current = _request()
        picker = create_card_picker("IRONCLAD")
        decision = create_build_agent(FakeLLM({}), picker).decide(current)

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            run_path = directory.bind("ABC123")
            CardChoiceRecorder(directory).on_entry(
                ContextEntry(
                    7,
                    decision.command,
                    current.state,
                    True,
                    scope=current.scope,
                    decision=decision,
                )
            )
            row = json.loads(
                (run_path / "card_choices.jsonl").read_text("utf-8")
            )

        self.assertEqual(row["decision"]["picker_id"], "ironclad.winning_path")
        self.assertEqual(row["decision"]["offered"], ["Entrench", "Shrug It Off"])
        self.assertEqual(row["context"]["deck_before_counts"]["Barricade"], 1)

    def test_shop_filter_does_not_expose_picker_details_to_build_agent(self):
        state = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("purge", "wild strike", "true grit"),
            details={
                "cards": (
                    {"name": "Wild Strike", "price": 28},
                    {"name": "True Grit", "price": 53},
                )
            },
            facts={
                "class": "IRONCLAD",
                "act": 1,
                "floor": 3,
                "deck": (
                    {"name": "Strike"},
                    {"name": "Defend"},
                    {"name": "Barricade"},
                ),
            },
        )
        picker = create_card_picker("IRONCLAD")

        result = picker.review_shop(build_request(state))

        self.assertTrue(result["policy_result"]["policy"].startswith(""))

    def test_decision_payload_round_trips_through_replay(self):
        current = _request()
        facts = {
            **dict(current.state.facts),
            "sts_seed": "ABC123",
            "replay_boundary_key": "before",
            "replay_rng_state": {"card": (1, 2, 3)},
        }
        initial = GameState(
            current.state.owner_hint,
            current.state.scope_id,
            current.state.screen,
            facts=facts,
        )
        scope = DecisionScope(AgentKind.BUILD, initial.scope_id)
        live_request = DecisionRequest(
            initial,
            scope,
            None,
            current.shared,
            ContextEntry(0, None, initial, True, scope=scope),
        )
        picker = create_card_picker("IRONCLAD")
        decision = create_build_agent(FakeLLM({}), picker).decide(live_request)

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("ABC123")
            journal = ReplayJournal(directory)
            journal.begin(initial)
            context = GameContext()
            context.start(initial)
            journal.stage_live(context.view(), RoutedDecision(scope, decision))
            journal.prepare_execute(decision.command)
            final = GameState(
                initial.owner_hint,
                initial.scope_id,
                initial.screen,
                terminal=True,
                facts={**facts, "replay_boundary_key": "after"},
            )
            journal.complete(ExecutionResult(decision.command, final, True))

            resumed = ReplayJournal(RunDirectory.open(directory.path), resume=True)
            replay_context = GameContext()
            replay_context.start(initial)
            replayed = resumed.stage_replay(replay_context.view())

        self.assertIsNotNone(replayed)
        self.assertEqual(replayed.decision.command, decision.command)
        self.assertEqual(
            replayed.decision.payload["card_choice_review"]["picker_id"],
            "ironclad.winning_path",
        )
        self.assertEqual(
            replayed.decision.payload[RUN_CONSTRUCTION_KEY],
            decision.payload[RUN_CONSTRUCTION_KEY],
        )


if __name__ == "__main__":
    unittest.main()
