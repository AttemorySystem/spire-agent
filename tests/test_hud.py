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
from spire_agent.extensions.hud import (
    HISTORY_FILENAME,
    HudObserver,
    prepare_display,
)
from spire_agent.extensions.run_directory import RunDirectory


class FakeReplay:
    def __init__(self, *, resume: bool = False) -> None:
        self.resume = resume
        self.last_execution_replayed = False


def state(
    floor: int,
    *,
    owner: AgentKind,
    screen: str,
    combat=None,
    deck=None,
) -> GameState:
    return GameState(
        owner,
        f"DISPLAY:a1:f{floor}:{owner.value}",
        ScreenState(
            screen,
            commands=("choose", "play", "end"),
            choices=("First", "Second"),
        ),
        facts={
            "sts_seed": "DISPLAY",
            "act": 1,
            "floor": floor,
            "room_type": "MonsterRoom",
            "act_boss": "Hexaghost",
            "current_hp": 70,
            "max_hp": 80,
            "gold": 99,
            "deck": deck
            or [
                {"name": "Ascender's Bane"},
                {"name": "Bash"},
                {"name": "Defend"},
                {"name": "Defend"},
                {"name": "Strike"},
                {"name": "Strike", "upgrades": 1},
            ],
            "relics": [{"name": "Burning Blood"}],
        },
        combat=combat,
    )


def entry(
    index: int,
    after: GameState,
    command: str | None,
    decision: Decision | None = None,
) -> ContextEntry:
    return ContextEntry(
        index,
        command,
        after,
        True,
        scope=None
        if decision is None
        else DecisionScope(after.owner_hint, after.scope_id),
        decision=decision,
    )


class HudTests(unittest.TestCase):
    def test_live_frames_project_route_deck_strategy_and_mcts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            overlay = root / "runtime" / "agent_overlay.json"
            observer = HudObserver(directory, FakeReplay(), overlay)
            initial = state(0, owner=AgentKind.MAP, screen="MAP")
            combat = {
                "turn": 1,
                "hand": (
                    {"name": "Strike"},
                    {"name": "Bash"},
                ),
                "monsters": ({"name": "Jaw Worm"},),
            }
            fight = state(1, owner=AgentKind.COMBAT, screen="NONE", combat=combat)
            observer.on_entry(entry(0, initial, None))
            observer.on_entry(
                entry(
                    1,
                    fight,
                    "choose 0",
                    Decision(
                        "choose 0",
                        "map.llm",
                        payload={
                            "run_route": {
                                "planned_rooms": (
                                    "Monster", "Unknown", "Elite", "Boss"
                                )
                            },
                            "winning_path_review": {
                                "winning_path": {
                                    "state": {
                                        "active_modules": ("strength_damage",)
                                    }
                                }
                            },
                        },
                    ),
                )
            )
            mcts = directory.path / "mcts"
            mcts.mkdir()
            (mcts / "000001.json").write_text(
                json.dumps(
                    {
                        "raw_result": {
                            "rootActions": [
                                {
                                    "action": "end",
                                    "selectionValue": 0.1,
                                    "visits": 20,
                                    "winSampleRate": 0.4,
                                    "expectedEndHpOnWin": 20,
                                },
                                {
                                    "action": "play 2 0",
                                    "selectionValue": 0.8,
                                    "visits": 80,
                                    "winSampleRate": 0.75,
                                    "expectedEndHpOnWin": 55,
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            observer.on_entry(
                entry(
                    2,
                    fight,
                    "play 2 0",
                    Decision(
                        "play 2 0",
                        "combat.mcts",
                        metrics={"search_id": "000001"},
                    ),
                )
            )

            frame = json.loads(overlay.read_text(encoding="utf-8"))
            self.assertLess(overlay.stat().st_size, 128 * 1024)
            self.assertNotIn("todo_panel", frame)
            self.assertNotIn("boss_panel", frame)
            self.assertEqual(
                [room["name"] for room in frame["map_panel"]["rooms"]],
                ["Monster", "Event", "Elite"],
            )
            self.assertEqual(
                [card["name"] for card in frame["build_panel"]["cards"][:2]],
                ["Strike", "Defend"],
            )
            self.assertEqual(frame["build_panel"]["card_count"], 6)
            self.assertEqual(
                frame["strategy_panel"]["name"], "Strength Damage"
            )
            selected = frame["mcts_panel"]["actions"][0]
            self.assertEqual(
                (selected["label"], selected["target"], selected["win_rate"]),
                ("Bash", "Jaw Worm", 0.75),
            )
            self.assertIn(
                "Play Bash -> Jaw Worm", frame["action_panel"]["history"]
            )

    def test_mcts_panel_is_retained_without_a_new_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            overlay = root / "overlay.json"
            observer = HudObserver(directory, FakeReplay(), overlay)
            current = state(1, owner=AgentKind.COMBAT, screen="NONE", combat={"hand": ()})
            observer.on_entry(entry(0, current, None))
            observer.projector.mcts = {"actions": [{"label": "Previous"}]}
            observer.on_entry(
                entry(1, current, "end", Decision("end", "combat.mcts_selection"))
            )

            frame = json.loads(overlay.read_text(encoding="utf-8"))
            self.assertEqual(frame["mcts_panel"]["actions"], [{"label": "Previous"}])

    def test_llm_stream_is_live_but_only_final_reasoning_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            overlay = root / "overlay.json"
            observer = HudObserver(directory, FakeReplay(), overlay)
            current = state(0, owner=AgentKind.MAP, screen="MAP")
            observer.on_entry(entry(0, current, None))

            observer.on_llm_event("start", "map.choose_exit")
            observer.on_llm_event("reasoning", "Compare elite paths. ")
            observer.on_llm_event("reasoning", "Prefer the safer route.")
            observer.on_llm_event("done", "")

            live = json.loads(overlay.read_text(encoding="utf-8"))
            self.assertEqual(live["action_panel"]["context"], "Map Agent")
            self.assertEqual(live["action_panel"]["action"], "Thinking ...")
            self.assertEqual(
                live["action_panel"]["history"],
                "Compare elite paths. Prefer the safer route.",
            )

            observer.on_entry(
                entry(
                    1,
                    state(1, owner=AgentKind.COMBAT, screen="NONE", combat={}),
                    "choose 0",
                    Decision("choose 0", "map.llm", "short result"),
                )
            )
            final = json.loads(overlay.read_text(encoding="utf-8"))
            self.assertEqual(
                final["action_panel"]["history"],
                "Compare elite paths. Prefer the safer route.",
            )
            records = [
                json.loads(line)
                for line in (directory.path / HISTORY_FILENAME).read_text().splitlines()
            ]
            self.assertEqual(records[-1]["frame"], final)

    def test_replay_reads_the_recorded_frame_without_appending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            original_overlay = root / "live.json"
            live = HudObserver(directory, FakeReplay(), original_overlay)
            initial = state(0, owner=AgentKind.MAP, screen="MAP")
            second = state(1, owner=AgentKind.BUILD, screen="EVENT")
            first_entry = entry(0, initial, None)
            second_entry = entry(
                1, second, "choose 0", Decision("choose 0", "map.llm")
            )
            live.on_entry(first_entry)
            live.on_entry(second_entry)
            path = directory.path / HISTORY_FILENAME
            lines_before = path.read_text(encoding="utf-8").splitlines()

            replay = FakeReplay(resume=True)
            replay_overlay = root / "replay.json"
            player = HudObserver(directory, replay, replay_overlay)
            player.on_entry(first_entry)
            replay.last_execution_replayed = True
            player.on_entry(second_entry)

            expected = json.loads(lines_before[-1])["frame"]
            self.assertEqual(json.loads(replay_overlay.read_text()), expected)
            self.assertEqual(path.read_text().splitlines(), lines_before)

    def test_hidden_live_hud_still_records_exact_replay_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            hidden_overlay = root / "hidden.json"
            initial = entry(
                0, state(0, owner=AgentKind.MAP, screen="MAP"), None
            )
            second = entry(
                1,
                state(1, owner=AgentKind.BUILD, screen="EVENT"),
                "choose 0",
                Decision("choose 0", "map.llm"),
            )
            recorder = HudObserver(
                directory, FakeReplay(), hidden_overlay, display=False
            )

            recorder.on_entry(initial)
            recorder.on_entry(second)

            self.assertFalse(hidden_overlay.exists())
            history = (directory.path / HISTORY_FILENAME).read_text().splitlines()
            self.assertEqual(len(history), 2)

            replay = FakeReplay(resume=True)
            replay_overlay = root / "replay.json"
            player = HudObserver(directory, replay, replay_overlay)
            player.on_entry(initial)
            replay.last_execution_replayed = True
            player.on_entry(second)

            self.assertEqual(
                json.loads(replay_overlay.read_text()),
                json.loads(history[-1])["frame"],
            )
            self.assertEqual(
                (directory.path / HISTORY_FILENAME).read_text().splitlines(),
                history,
            )

    def test_overlay_failure_does_not_stop_history_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            overlay_directory = root / "overlay.json"
            overlay_directory.mkdir()
            observer = HudObserver(
                directory, FakeReplay(), overlay_directory
            )

            observer.on_entry(
                entry(0, state(0, owner=AgentKind.MAP, screen="MAP"), None)
            )

            self.assertFalse(observer.displaying)
            self.assertTrue((directory.path / HISTORY_FILENAME).is_file())

    def test_missing_replay_frames_never_show_a_stale_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            overlay = root / "overlay.json"
            overlay.write_text('{"schema_version":4,"sequence":99}')
            observer = HudObserver(
                directory, FakeReplay(resume=True), overlay
            )

            observer.on_entry(
                entry(0, state(0, owner=AgentKind.MAP, screen="MAP"), None)
            )

            self.assertFalse(overlay.exists())

    def test_missing_later_replay_frame_removes_the_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = RunDirectory(root / "runs")
            directory.bind("DISPLAY")
            frame = {
                "schema_version": 4,
                "sequence": 1,
                "entry_index": 0,
            }
            (directory.path / HISTORY_FILENAME).write_text(
                json.dumps({"version": 1, "entry_index": 0, "frame": frame}) + "\n"
            )
            replay = FakeReplay(resume=True)
            overlay = root / "overlay.json"
            observer = HudObserver(directory, replay, overlay)
            observer.on_entry(
                entry(0, state(0, owner=AgentKind.MAP, screen="MAP"), None)
            )
            self.assertTrue(overlay.exists())

            replay.last_execution_replayed = True
            observer.on_entry(
                entry(1, state(1, owner=AgentKind.BUILD, screen="EVENT"), "choose 0")
            )

            self.assertFalse(overlay.exists())

    def test_prepare_display_is_platform_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            sockets = Path(temporary)
            environ = {}
            self.assertIsNotNone(
                prepare_display(environ=environ, socket_dir=sockets, platform="linux")
            )
            (sockets / "X2").touch()
            self.assertIsNone(
                prepare_display(environ=environ, socket_dir=sockets, platform="linux")
            )
            self.assertEqual(environ["DISPLAY"], ":2")


if __name__ == "__main__":
    unittest.main()
