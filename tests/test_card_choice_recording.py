from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.contracts import ContextEntry, GameState
from spire_agent.extensions import CardChoiceRecorder, RunDirectory
from spire_agent.subagents.build import create_build_agent as compose_build_agent
from spire_agent.tools.winning_path import WinningPathCardPicker

from tests.test_card_reward_policy import FakeLLM, card_state, request


def create_build_agent(llm):
    return compose_build_agent(llm, WinningPathCardPicker())


class CardChoiceRecorderTests(unittest.TestCase):
    def test_records_confirmed_choice_and_rewrites_terminal_outcome(self):
        before = card_state(("Armaments", "Shrug It Off", "Anger"), floor=2)
        current_request = request(before)
        decision = create_build_agent(FakeLLM({})).decide(current_request)
        after = GameState(
            before.owner_hint,
            before.scope_id,
            before.screen,
            facts=before.facts,
        )

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            run_path = directory.bind("ABC123")
            recorder = CardChoiceRecorder(directory)
            recorder.on_entry(
                ContextEntry(
                    3,
                    decision.command,
                    after,
                    True,
                    scope=current_request.scope,
                    decision=decision,
                )
            )
            path = run_path / "card_choices.jsonl"
            row = json.loads(path.read_text("utf-8"))

            self.assertEqual(row["choice_id"], "ABC123:f2:c0")
            self.assertEqual(row["decision"]["offered"], [
                "Armaments", "Shrug It Off", "Anger"
            ])
            self.assertEqual(row["decision"]["action"], decision.command)
            self.assertEqual(row["run"]["floor_reached"], None)

            combat = GameState(
                before.owner_hint,
                before.scope_id,
                before.screen,
                facts=before.facts,
                combat={
                    "monsters": (
                        {"name": "Gremlin Nob", "is_gone": False},
                    )
                },
            )
            recorder.on_entry(ContextEntry(4, "end", combat, True))

            terminal = GameState(
                before.owner_hint,
                before.scope_id,
                before.screen,
                terminal=True,
                facts={**dict(before.facts), "floor": 39, "room_type": "MonsterRoom"},
            )
            recorder.on_entry(ContextEntry(5, "end", terminal, True))
            finished = json.loads(path.read_text("utf-8"))

            self.assertEqual(finished["run"]["floor_reached"], 39)
            self.assertFalse(finished["run"]["victory"])
            self.assertEqual(finished["run"]["killed_by"], "Gremlin Nob")

    def test_rejected_choice_is_not_recorded(self):
        state = card_state(("Armaments", "Shrug It Off", "Anger"), floor=2)
        current_request = request(state)
        decision = create_build_agent(FakeLLM({})).decide(current_request)

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            run_path = directory.bind("ABC123")
            CardChoiceRecorder(directory).on_entry(
                ContextEntry(
                    3,
                    decision.command,
                    state,
                    False,
                    error="rejected",
                    decision=decision,
                )
            )

            self.assertFalse((run_path / "card_choices.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
