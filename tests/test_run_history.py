from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    Decision,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.extensions.run_directory import RunDirectory
from spire_agent.extensions.run_history import RunHistoryRecorder


class FakeReplay:
    resume = False
    last_execution_replayed = False


def state(
    floor: int,
    *,
    screen: str = "NONE",
    owner: AgentKind = AgentKind.BUILD,
    hp: int = 80,
    terminal: bool = False,
    choices=(),
    combat=None,
) -> GameState:
    return GameState(
        owner,
        f"seed:a1:f{floor}:{owner.value}",
        ScreenState(screen, commands=("choose", "play", "end"), choices=choices),
        terminal=terminal,
        facts={
            "sts_seed": "HISTORY1",
            "act": 1,
            "floor": floor,
            "room_type": "MonsterRoom",
            "class": "IRONCLAD",
            "ascension_level": 20,
            "current_hp": hp,
            "max_hp": 80,
            "gold": 99,
            "deck": [
                {"name": "Strike", "upgrades": 0},
                {"name": "Strike", "upgrades": 1},
                {"name": "Defend", "upgrades": 0},
            ],
            "relics": [{"name": "Burning Blood"}],
            "potions": [{"name": "Fire Potion"}, {"name": "Potion Slot"}],
        },
        combat=combat,
    )


def entry(index, before, after, command, *, source="test") -> ContextEntry:
    decision = Decision(command, source, "recorded test action")
    return ContextEntry(
        index,
        command,
        after,
        True,
        scope=DecisionScope(before.owner_hint, before.scope_id),
        decision=decision,
    )


class RunHistoryTests(unittest.TestCase):
    def test_records_facts_and_attributes_transition_to_previous_room(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("HISTORY1")
            recorder = RunHistoryRecorder(directory, FakeReplay())
            combat = {
                "turn": 1,
                "hand": [{"name": "Strike", "cost": 1}],
                "monsters": [{"name": "Jaw Worm", "current_hp": 40}],
                "player": {"current_hp": 80, "max_hp": 80},
            }
            first = state(1, owner=AgentKind.COMBAT, combat=combat)
            reward = state(
                1,
                screen="CARD_REWARD",
                choices=("Cleave", "Anger", "Warcry"),
                hp=74,
            )
            next_room = state(2, owner=AgentKind.COMBAT, hp=74, combat=combat)
            terminal = state(2, hp=0, terminal=True)

            recorder.on_entry(ContextEntry(0, None, first, True))
            recorder.on_entry(entry(1, first, reward, "play 1 0"))
            recorder.on_entry(entry(2, reward, next_room, "choose 0"))

            events = [
                json.loads(line)
                for line in (directory.path / "run_history.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(events), 3)
            self.assertEqual(
                events[1]["action"]["label"],
                "play Strike on Jaw Worm",
            )
            self.assertEqual(
                events[2]["action"]["label"], "choose Cleave"
            )
            deck = events[0]["state"]["run"]["deck"]
            self.assertEqual(deck[1], {"name": "Strike", "count": 2, "upgrades": 1})
            self.assertEqual(
                events[0]["state"]["run"]["potions"], ["Fire Potion", None]
            )

            recorder.on_entry(entry(3, next_room, terminal, "end"))
            last = json.loads(
                (directory.path / "run_history.jsonl").read_text().splitlines()[-1]
            )
            self.assertTrue(last["after"]["terminal"])

    def test_replay_reconstruction_is_not_duplicated_before_live_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("HISTORY1")
            original = FakeReplay()
            recorder = RunHistoryRecorder(directory, original)
            first, second, third = state(1), state(2), state(3)
            recorder.on_entry(ContextEntry(0, None, first, True))
            recorder.on_entry(entry(1, first, second, "choose 0"))

            replay = FakeReplay()
            replay.resume = True
            resumed = RunHistoryRecorder(directory, replay)
            resumed.on_entry(ContextEntry(0, None, first, True))
            replay.last_execution_replayed = True
            resumed.on_entry(entry(1, first, second, "choose 0"))
            replay.last_execution_replayed = False
            resumed.on_entry(entry(2, second, third, "choose 0"))

            lines = (directory.path / "run_history.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[-1])["entry_index"], 2)

    def test_recorder_repairs_an_incomplete_last_append_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("HISTORY1")
            recorder = RunHistoryRecorder(directory, FakeReplay())
            first, second = state(1), state(2)
            recorder.on_entry(ContextEntry(0, None, first, True))
            recorder.on_entry(entry(1, first, second, "choose 0"))
            with (directory.path / "run_history.jsonl").open("a") as stream:
                stream.write('{"schema_version":1,"type":"action"')

            resumed = RunHistoryRecorder(directory, FakeReplay())
            resumed.on_entry(ContextEntry(2, None, second, True))
            lines = (directory.path / "run_history.jsonl").read_text().splitlines()

            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[-1])["entry_index"], 1)


if __name__ == "__main__":
    unittest.main()
