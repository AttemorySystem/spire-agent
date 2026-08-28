from __future__ import annotations

import curses
from pathlib import Path
from threading import Thread
import unittest
from unittest.mock import patch

from spire_agent.console import (
    AgentControl,
    ConsoleBuffer,
    ConsoleCommandError,
    ConsoleController,
    ConsoleSettings,
    DecisionActivity,
    DecisionDisplayObserver,
    _draw,
    _tui,
    parse_command,
)
from spire_agent.contracts import (
    AgentKind,
    Continuation,
    ContextEntry,
    ContextView,
    Decision,
    DecisionScope,
    GameState,
    RoutedDecision,
    ScreenState,
)


def state(
    screen: str = "CARD_REWARD",
    *,
    owner: AgentKind = AgentKind.BUILD,
    commands=("choose", "skip"),
    choices=("Anger", "Carnage", "Immolate"),
    facts=None,
) -> GameState:
    return GameState(
        owner,
        "scope-1",
        ScreenState(screen, commands=commands, choices=choices),
        facts=facts or {"act": 1, "floor": 7, "room_type": "MonsterRoom"},
    )


def context(
    current: GameState,
    continuation: Continuation | None = None,
) -> ContextView:
    entry = ContextEntry(0, None, current, True)
    return ContextView(current, None, continuation, {}, entry, 1)


class ConsoleCommandTests(unittest.TestCase):
    def test_parses_documented_commands(self):
        self.assertEqual(parse_command("run"), ("run",))
        self.assertEqual(
            parse_command("run ironclad a17 ABC123"),
            ("run", "IRONCLAD", "17", "ABC123"),
        )
        self.assertEqual(parse_command("replay abc123"), ("replay", "ABC123"))
        self.assertEqual(parse_command("agent off"), ("agent", "off"))
        self.assertEqual(parse_command("agent on"), ("agent", "on"))
        self.assertEqual(parse_command("window"), ("view", "window"))
        self.assertEqual(parse_command("full screen"), ("view", "fullscreen"))
        self.assertEqual(parse_command("hud=on"), ("hud", "on"))
        self.assertEqual(parse_command("HUD=OFF"), ("hud", "off"))
        with self.assertRaises(ConsoleCommandError):
            parse_command("choose 2")

    def test_rejects_invalid_run_shape(self):
        for command in ("run ironclad", "run ironclad 17 seed", "run ironclad a21 seed"):
            with self.subTest(command=command), self.assertRaises(ConsoleCommandError):
                parse_command(command)

    def test_control_d_exits_tui(self):
        class Screen:
            def keypad(self, enabled):
                pass

            def timeout(self, milliseconds):
                pass

            def get_wch(self):
                return "\x04"

        class Controller:
            output = ConsoleBuffer()

        with (
            patch("spire_agent.console.curses.curs_set"),
            patch("spire_agent.console._draw", return_value=0),
        ):
            _tui(Screen(), Controller())

    def test_up_and_down_browse_and_edit_command_history(self):
        class Screen:
            keys = iter(
                (
                    "r", "u", "n", "\n",
                    "h", "e", "l", curses.KEY_UP, curses.KEY_DOWN, "p", "\n",
                    curses.KEY_UP, "\b", "r", "\n",
                    "\x04",
                )
            )

            def keypad(self, enabled):
                pass

            def timeout(self, milliseconds):
                pass

            def get_wch(self):
                return next(self.keys)

        class Controller:
            output = ConsoleBuffer()
            commands = []

            def execute(self, command):
                self.commands.append(command)
                return True

        controller = Controller()
        with (
            patch("spire_agent.console.curses.curs_set"),
            patch("spire_agent.console._draw", return_value=0),
        ):
            _tui(Screen(), controller)

        self.assertEqual(controller.commands, ["run", "help", "helr"])

    def test_background_output_is_routed_to_the_display(self):
        output = ConsoleBuffer()
        controller = ConsoleController(
            ConsoleSettings(Path("."), Path("config.yaml")), output
        )

        def fake_run(*args, **kwargs):
            print("W: game bridge is still starting")
            return 0

        with patch("spire_agent.console.run", side_effect=fake_run):
            controller._launch(object())
            controller._thread.join(timeout=1)

        self.assertFalse(controller.active)
        self.assertIn(
            "W: game bridge is still starting",
            output.wrapped(200),
        )

    def test_log_dir_has_the_same_base_directory_semantics_as_sts_agent(self):
        controller = ConsoleController(
            ConsoleSettings(Path("logs"), Path("config.yaml")), ConsoleBuffer()
        )

        args = controller._agent_args()

        self.assertEqual(args.log_dir, Path("logs"))
        self.assertEqual(controller.settings.run_root, Path("logs/runs"))

    def test_settings_display_the_current_model(self):
        class Screen:
            rows = []

            def getmaxyx(self):
                return 10, 200

            def erase(self):
                pass

            def addnstr(self, row, column, value, *args):
                self.rows.append(value)

            def hline(self, *args):
                pass

            def move(self, *args):
                pass

            def refresh(self):
                pass

        screen = Screen()
        controller = ConsoleController(
            ConsoleSettings(
                Path("."), Path("config.yaml"), model="openai/gpt-5.6-luna"
            ),
            ConsoleBuffer(),
        )

        with (
            patch("spire_agent.console.curses.ACS_HLINE", 0, create=True),
            patch("spire_agent.console.curses.A_BOLD", 0, create=True),
        ):
            _draw(screen, controller, "", 0)

        self.assertTrue(
            any("model=openai/gpt-5.6-luna" in row for row in screen.rows)
        )
        setting = next(row for row in screen.rows if row.startswith("setting:"))
        self.assertIn("[1600x900] fullscreen", setting)
        self.assertIn("hud=off", setting)
        self.assertNotIn("session=", setting)
        self.assertNotIn("window", setting)

        controller.settings.fullscreen = True
        controller.settings.hud = True
        screen.rows.clear()
        with (
            patch("spire_agent.console.curses.ACS_HLINE", 0, create=True),
            patch("spire_agent.console.curses.A_BOLD", 0, create=True),
        ):
            _draw(screen, controller, "", 0)
        setting = next(row for row in screen.rows if row.startswith("setting:"))
        self.assertIn("[fullscreen]", setting)
        self.assertIn("hud=on", setting)
        self.assertNotIn("1600x900", setting)

    def test_hud_command_changes_the_next_run_setting(self):
        output = ConsoleBuffer()
        controller = ConsoleController(
            ConsoleSettings(Path("."), Path("config.yaml")), output
        )

        self.assertTrue(controller.execute("hud=on"))
        self.assertTrue(controller.settings.hud)
        self.assertTrue(controller.execute("hud=off"))
        self.assertFalse(controller.settings.hud)


class AgentControlTests(unittest.TestCase):
    def test_enabled_control_does_not_request_refresh(self):
        output = []
        control = AgentControl(output.append)

        result = control.before_decision(context(state()))

        self.assertFalse(result)

    def test_disabled_control_waits_then_requests_refresh(self):
        output = []
        control = AgentControl(output.append)
        control.disable()
        result = []
        worker = Thread(
            target=lambda: result.append(control.before_decision(context(state())))
        )
        worker.start()
        self.assertTrue(worker.is_alive())
        control.enable()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(result[0])
        self.assertTrue(any("game window is under human control" in line for line in output))

    def test_off_is_deferred_until_continuation_completes(self):
        output = []
        control = AgentControl(output.append)
        control.disable()
        active = Continuation(
            AgentKind.BUILD,
            "selection",
            "scope-1",
            ("CARD_REWARD",),
        )

        result = control.before_decision(context(state(), active))

        self.assertFalse(result)
        self.assertFalse(control.paused)
        self.assertTrue(any("continuation completes" in line for line in output))
        control.enable()


class DecisionDisplayTests(unittest.TestCase):
    def test_card_reward_is_one_result_only_line(self):
        output = ConsoleBuffer()
        observer = DecisionDisplayObserver(output)
        before = state(
            choices=("anger", "carnage", "immolate")
        )
        observer.on_entry(ContextEntry(0, None, before, True))
        observer.on_entry(
            ContextEntry(
                1,
                "choose 1",
                state("MAP", commands=("choose",), choices=("x=1",)),
                True,
                scope=DecisionScope(AgentKind.BUILD, "scope-1"),
                decision=Decision("choose 1", "card_reward.policy", "best damage"),
            )
        )

        lines = output.wrapped(200)
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            lines[0],
            "Room  7 | Build Agent | Card reward: "
            "[Anger, Carnage, Immolate] -> Carnage",
        )
        self.assertNotIn("best damage", lines[0])

    def test_confirmed_skip_is_not_displayed_as_singing_bowl(self):
        output = ConsoleBuffer()
        observer = DecisionDisplayObserver(output)
        before = state()
        observer.on_entry(ContextEntry(0, None, before, True))
        observer.on_entry(
            ContextEntry(
                1,
                "skip",
                state("MAP"),
                True,
                decision=Decision(
                    "skip",
                    "card_reward.policy",
                    payload={
                        "card_reward_policy_result": {"card": "Singing Bowl"}
                    },
                ),
            )
        )

        self.assertEqual(
            output.wrapped(200)[0],
            "Room  7 | Build Agent | Card reward: "
            "[Anger, Carnage, Immolate] -> Skip",
        )

    def test_boss_reward_lists_candidates(self):
        output = ConsoleBuffer()
        observer = DecisionDisplayObserver(output)
        before = state(
            "BOSS_REWARD",
            choices=("Black Blood", "Empty Cage", "Runic Cube"),
        )
        observer.on_entry(ContextEntry(0, None, before, True))
        observer.on_entry(
            ContextEntry(
                1,
                "choose 1",
                state("MAP"),
                True,
                decision=Decision("choose 1", "build.llm"),
            )
        )

        self.assertEqual(
            output.wrapped(200)[0],
            "Room  7 | Build Agent | Boss reward: "
            "[Black Blood, Empty Cage, Runic Cube] -> Empty Cage",
        )

    def test_shop_and_campfire_are_displayed(self):
        for screen, choices, command, expected in (
            ("SHOP_SCREEN", ("purge", "Shrug It Off"), "choose 0", "Shop"),
            ("REST", ("rest", "smith"), "choose 1", "Campfire"),
        ):
            with self.subTest(screen=screen):
                output = ConsoleBuffer()
                observer = DecisionDisplayObserver(output)
                before = state(screen, choices=choices)
                observer.on_entry(ContextEntry(0, None, before, True))
                observer.on_entry(
                    ContextEntry(
                        1,
                        command,
                        state("MAP", commands=("choose",), choices=("x=1",)),
                        True,
                        decision=Decision(command, "build.llm", "test reason"),
                    )
                )
                lines = output.wrapped(200)
                self.assertEqual(len(lines), 1)
                self.assertIn(expected, lines[0])
                self.assertNotIn("test reason", lines[0])

    def test_shop_actions_share_one_row_with_remaining_gold(self):
        output = ConsoleBuffer()
        observer = DecisionDisplayObserver(output)
        scope = DecisionScope(AgentKind.BUILD, "shop-1")
        observer.on_entry(
            ContextEntry(
                0,
                None,
                state(
                    "SHOP_SCREEN",
                    choices=("true grit", "cleave", "Leave"),
                    facts={"floor": 2, "gold": 100},
                ),
                True,
            )
        )
        for index, choices, gold, next_screen in (
            (1, ("cleave", "Leave"), 60, "SHOP_SCREEN"),
            (2, ("Leave",), 25, "SHOP_SCREEN"),
            (3, ("x=0",), 25, "MAP"),
        ):
            observer.on_entry(
                ContextEntry(
                    index,
                    "choose 0",
                    state(
                        next_screen,
                        choices=choices,
                        facts={"floor": 2, "gold": gold},
                    ),
                    True,
                    scope=scope,
                    decision=Decision("choose 0", "build.llm"),
                )
            )

        self.assertEqual(
            output.wrapped(200),
            [
                "Room  2 | Build Agent | Shop: "
                "True Grit -> Cleave -> Leave | Gold: 25"
            ],
        )

    def test_thinking_row_is_replaced_by_selected_map_path(self):
        output = ConsoleBuffer()
        before = state(
            "MAP",
            owner=AgentKind.MAP,
            commands=("choose",),
            choices=("x=1", "x=2"),
        )
        decision = Decision(
            "choose 1",
            "map.llm",
            payload={
                "run_route": {
                    "planned_rooms": ("Monster", "Rest", "Elite", "Boss")
                },
            },
        )

        class Provider:
            def decide(self, current):
                return RoutedDecision(
                    DecisionScope(AgentKind.MAP, "scope-1"), decision
                )

        observer = DecisionDisplayObserver(output)
        observer.on_entry(ContextEntry(0, None, before, True))
        activity = DecisionActivity(Provider(), output)
        routed = activity.decide(context(before))
        self.assertEqual(
            output.wrapped(200),
            ["Room  7 | Map Agent is thinking ..."],
        )

        observer.on_entry(
            ContextEntry(
                1,
                routed.decision.command,
                state("EVENT"),
                True,
                scope=routed.scope,
                decision=routed.decision,
            )
        )

        self.assertEqual(
            output.wrapped(200),
            [
                "Room  7 | Map Agent | Path: "
                "Monster -> Rest -> Elite -> Boss"
            ],
        )

    def test_combat_search_uses_one_room_row(self):
        output = ConsoleBuffer()
        current = state(
            "NONE",
            owner=AgentKind.COMBAT,
            commands=("play", "end"),
            choices=(),
        )

        class Provider:
            def decide(self, value):
                return RoutedDecision(
                    DecisionScope(AgentKind.COMBAT, "scope-1"),
                    Decision("end", "combat.mcts"),
                )

        DecisionActivity(Provider(), output).decide(context(current))

        self.assertEqual(
            output.wrapped(200),
            ["Room  7 | Combat Agent is searching ..."],
        )

    def test_combat_search_shows_latest_completed_win_rate(self):
        output = ConsoleBuffer()
        current = state(
            "NONE",
            owner=AgentKind.COMBAT,
            commands=("play", "end"),
            choices=(),
        )
        scope = DecisionScope(AgentKind.COMBAT, "scope-1")
        previous = ContextEntry(
            1,
            "play 1 0",
            current,
            True,
            scope=scope,
            decision=Decision(
                "play 1 0",
                "combat.mcts",
                metrics={
                    "risk": {
                        "winSampleRate": 0.75,
                        "meanBestWinEndHp": 57.5,
                    }
                },
            ),
        )

        class Provider:
            def decide(self, value):
                self.during_search = output.wrapped(200)
                return RoutedDecision(
                    scope,
                    Decision(
                        "end",
                        "combat.mcts",
                        metrics={
                            "risk": {
                                "winSampleRate": 0.80,
                                "meanBestWinEndHp": 60,
                            }
                        },
                    ),
                )

        view = ContextView(current, scope, None, {}, previous, 2)
        provider = Provider()
        DecisionActivity(provider, output).decide(view)

        self.assertEqual(
            provider.during_search,
            [
                "Room  7 | Combat Agent is searching ... | "
                "win-rate 75% | best HP 57.5"
            ],
        )
        self.assertEqual(
            output.wrapped(200),
            [
                "Room  7 | Combat Agent is searching ... | "
                "win-rate 80% | best HP 60"
            ],
        )

    def test_completed_combat_row_shows_health_not_command(self):
        output = ConsoleBuffer()
        observer = DecisionDisplayObserver(output)
        before = state(
            "NONE",
            owner=AgentKind.COMBAT,
            commands=("play", "end"),
            choices=(),
            facts={"floor": 7, "current_hp": 61, "max_hp": 80},
        )
        observer.on_entry(ContextEntry(0, None, before, True))
        observer.on_entry(
            ContextEntry(
                1,
                "play 1 0",
                state(
                    "NONE",
                    owner=AgentKind.COMBAT,
                    commands=("play", "end"),
                    choices=(),
                    facts={"floor": 7, "current_hp": 57, "max_hp": 80},
                ),
                True,
                scope=DecisionScope(AgentKind.COMBAT, "scope-1"),
                decision=Decision("play 1 0", "combat.mcts"),
            )
        )

        self.assertEqual(
            output.wrapped(200),
            ["Room  7 | Combat Agent | HP: 57 / 80"],
        )

    def test_event_actions_share_one_room_row(self):
        output = ConsoleBuffer()
        observer = DecisionDisplayObserver(output)
        scope = DecisionScope(AgentKind.BUILD, "event-11")
        observer.on_entry(
            ContextEntry(
                0,
                None,
                state(
                    "EVENT",
                    choices=("eat", "leave"),
                    facts={"floor": 11},
                ),
                True,
            )
        )
        observer.on_entry(
            ContextEntry(
                1,
                "choose 0",
                state("EVENT", choices=("leave",), facts={"floor": 11}),
                True,
                scope=scope,
                decision=Decision("choose 0", "build.llm"),
            )
        )
        observer.on_entry(
            ContextEntry(
                2,
                "choose 0",
                state("MAP", choices=("x=0",), facts={"floor": 11}),
                True,
                scope=scope,
                decision=Decision("choose 0", "build.llm"),
            )
        )

        self.assertEqual(
            output.wrapped(200),
            ["Room 11 | Build Agent | EVENT: Eat -> Leave"],
        )


if __name__ == "__main__":
    unittest.main()
