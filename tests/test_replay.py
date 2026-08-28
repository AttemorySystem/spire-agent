from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spire_agent.context import GameContext
from spire_agent.contracts import (
    AgentKind,
    Continuation,
    ContinuationChange,
    ContextView,
    Decision,
    DecisionScope,
    ExecutionResult,
    GameState,
    RoutedDecision,
    ScreenState,
    SessionRefresh,
)
from spire_agent.extensions.run_directory import RunDirectory
from spire_agent.errors import ActionValidationError
from spire_agent.game_agent import GameAgent
from spire_agent.extensions.replay import (
    REPLAY_UNAVAILABLE_FILENAME,
    ReplayError,
    ReplayJournal,
    ReplayRuntime,
    restore_game_rng,
)
from spire_agent.tools.game_stability import stable_boundary_key
from spire_agent.validation import AvailableCommandValidator


RNG = {"card": [11, 12, 3], "shuffle": [21, 22, 4]}


def state(key: str, *, terminal: bool = False) -> GameState:
    return GameState(
        owner_hint=AgentKind.BUILD,
        scope_id="TEST:a1:f1:event:build",
        screen=ScreenState(
            type="GAME_OVER" if terminal else "EVENT",
            commands=() if terminal else ("choose",),
            choices=() if terminal else ("accept",),
        ),
        terminal=terminal,
        facts={
            "sts_seed": "TEST",
            "class": "IRONCLAD",
            "ascension_level": 20,
            "act": 1,
            "floor": 1,
            "replay_boundary_key": key,
            "replay_rng_state": RNG,
        },
    )


class OneDecision:
    def __init__(self, *, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def decide(self, context: ContextView) -> RoutedDecision:
        self.calls += 1
        if self.fail:
            raise AssertionError("live provider must not run during replay")
        return RoutedDecision(
            DecisionScope(AgentKind.BUILD, context.state.scope_id),
            Decision(
                "choose 0",
                "build.rule",
                continuation=ContinuationChange.set(
                    Continuation(
                        AgentKind.BUILD,
                        "multi_select",
                        context.state.scope_id,
                        ("GRID",),
                        {"remaining": ["Bash"]},
                    )
                ),
                payload={"acquired_key": "ruby"},
            ),
        )


class OneStepSession:
    def __init__(self, initial: GameState, final: GameState):
        self.initial = initial
        self.final = final
        self.commands = []

    def reset(self):
        return self.initial

    def execute(self, command):
        self.commands.append(command)
        return ExecutionResult(command, self.final, True)

    def close(self):
        pass


class ReplayTests(unittest.TestCase):
    def run_once(self, runtime):
        return GameAgent(
            session=runtime,
            context=GameContext(),
            decisions=runtime,
            validator=AvailableCommandValidator(),
        ).run()

    def test_new_run_records_and_resume_replays_without_live_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            live = OneDecision()
            runtime = ReplayRuntime(
                OneStepSession(state("before"), state("after", terminal=True)),
                live,
                journal,
                lambda rng, key: self.fail("live run must not restore RNG"),
            )
            self.run_once(runtime)
            failure = directory.path / "replay_failure.json"
            failure.write_text("{}", encoding="utf-8")

            replay_directory = RunDirectory.open(directory.path)
            replay = ReplayJournal(replay_directory, resume=True)
            restored = []
            fallback = OneDecision(fail=True)
            replay_runtime = ReplayRuntime(
                OneStepSession(state("before"), state("after", terminal=True)),
                fallback,
                replay,
                lambda rng, key: restored.append((rng, key)),
            )
            result = self.run_once(replay_runtime)

            self.assertTrue(result.state.terminal)
            self.assertFalse(failure.exists())
            self.assertEqual(fallback.calls, 0)
            self.assertEqual(restored, [])
            self.assertEqual(
                result.continuation.data["remaining"],
                ("Bash",),
            )

    def test_replay_stops_and_writes_failure_on_post_state_divergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            self.run_once(
                ReplayRuntime(
                    OneStepSession(state("before"), state("after", terminal=True)),
                    OneDecision(),
                    journal,
                    lambda rng, key: None,
                )
            )

            replay = ReplayJournal(RunDirectory.open(directory.path), resume=True)
            with self.assertRaisesRegex(ReplayError, "result_mismatch"):
                self.run_once(
                    ReplayRuntime(
                        OneStepSession(
                            state("before"), state("different", terminal=True)
                        ),
                        OneDecision(fail=True),
                        replay,
                        lambda rng, key: None,
                    )
                )
            self.assertTrue((directory.path / "replay_failure.json").is_file())

    def test_invalid_proposal_is_not_written_as_an_executable_input(self):
        class InvalidDecision:
            def decide(self, context):
                return RoutedDecision(
                    DecisionScope(AgentKind.BUILD, context.state.scope_id),
                    Decision("skip", "bad-test-decision"),
                )

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            with self.assertRaises(ActionValidationError):
                self.run_once(
                    ReplayRuntime(
                        OneStepSession(
                            state("before"), state("after", terminal=True)
                        ),
                        InvalidDecision(),
                        journal,
                        lambda rng, key: None,
                    )
                )

            self.assertEqual(
                len(journal.path.read_text(encoding="utf-8").splitlines()),
                1,
            )

    def test_validated_input_is_fsynced_before_the_game_receives_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)

            class WriteAheadSession(OneStepSession):
                def execute(self, command):
                    records = journal.path.read_text(encoding="utf-8").splitlines()
                    self.assert_recorded = any('"kind":"action"' in row for row in records)
                    return super().execute(command)

            live = WriteAheadSession(state("before"), state("after", terminal=True))
            self.run_once(
                ReplayRuntime(
                    live, OneDecision(), journal, lambda rng, key: None
                )
            )

            self.assertTrue(live.assert_recorded)

    def test_action_without_result_is_replayed_after_a_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            journal.begin(state("before"))
            context = GameContext()
            context.start(state("before"))
            routed = OneDecision().decide(context.view())
            journal.stage_live(context.view(), routed)
            journal.prepare_execute("choose 0")

            resumed = ReplayJournal(RunDirectory.open(directory.path), resume=True)

            self.assertEqual(len(resumed.actions), 1)
            self.assertEqual(resumed.results, {})

    def test_partial_last_record_is_removed_after_a_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            journal.begin(state("before"))
            context = GameContext()
            context.start(state("before"))
            journal.stage_live(
                context.view(), OneDecision().decide(context.view())
            )
            journal.prepare_execute("choose 0")
            with journal.path.open("ab") as stream:
                stream.write(b'{"version":4,"reason":"\xe4')

            resumed = ReplayJournal(RunDirectory.open(directory.path), resume=True)

            self.assertEqual(len(resumed.actions), 1)
            self.assertEqual(resumed.results, {})
            self.assertTrue(journal.path.read_text(encoding="utf-8").endswith("\n"))

    def test_rejected_input_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            journal.begin(state("before"))
            context = GameContext()
            context.start(state("before"))
            journal.stage_live(
                context.view(), OneDecision().decide(context.view())
            )
            journal.prepare_execute("choose 0")
            journal.complete(
                ExecutionResult("choose 0", state("before"), False, "rejected")
            )

            resumed = ReplayJournal(RunDirectory.open(directory.path), resume=True)

            self.assertEqual(resumed.actions, [])

    def test_pre_action_boundary_mismatch_stops_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            self.run_once(
                ReplayRuntime(
                    OneStepSession(state("before"), state("after", terminal=True)),
                    OneDecision(),
                    journal,
                    lambda rng, key: None,
                )
            )
            resumed = ReplayJournal(RunDirectory.open(directory.path), resume=True)
            context = GameContext()
            context.start(state("different-before"))

            with self.assertRaisesRegex(ReplayError, "boundary_mismatch"):
                resumed.stage_replay(context.view())

    def test_changed_external_refresh_permanently_disables_replay(self):
        class RefreshSession(OneStepSession):
            def refresh(self):
                return SessionRefresh(state("manual-change"), True)

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            runtime = ReplayRuntime(
                RefreshSession(state("before"), state("after", terminal=True)),
                OneDecision(),
                journal,
                lambda rng, key: None,
            )
            runtime.reset()

            refreshed = runtime.refresh()

            self.assertTrue(refreshed.changed)
            self.assertTrue(journal.disabled)
            marker = directory.path / REPLAY_UNAVAILABLE_FILENAME
            self.assertTrue(marker.is_file())
            with self.assertRaisesRegex(ReplayError, "manual takeover"):
                ReplayJournal(RunDirectory.open(directory.path), resume=True)

    def test_agent_resumes_from_refreshed_state_after_manual_takeover(self):
        refreshed_state = state("manual-change")

        class TakeoverSession(OneStepSession):
            def refresh(self):
                return SessionRefresh(refreshed_state, True)

        class ResumeOnce:
            def __init__(self):
                self.first = True

            def before_decision(self, current):
                refresh, self.first = self.first, False
                return refresh

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            runtime = ReplayRuntime(
                TakeoverSession(
                    state("before"),
                    state("after", terminal=True),
                ),
                OneDecision(),
                journal,
                lambda rng, key: None,
            )

            final = GameAgent(
                session=runtime,
                context=GameContext(),
                decisions=runtime,
                validator=AvailableCommandValidator(),
                control=ResumeOnce(),
            ).run()

            self.assertTrue(final.state.terminal)
            self.assertEqual(final.entry_count, 3)
            self.assertEqual(final.last_entry.command, "choose 0")
            self.assertTrue(journal.disabled)
            records = journal.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 1)

    def test_unchanged_external_refresh_keeps_replay_available(self):
        class RefreshSession(OneStepSession):
            def refresh(self):
                return SessionRefresh(state("before"), False)

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            directory.bind("TEST")
            journal = ReplayJournal(directory)
            runtime = ReplayRuntime(
                RefreshSession(state("before"), state("after", terminal=True)),
                OneDecision(),
                journal,
                lambda rng, key: None,
            )
            runtime.reset()

            runtime.refresh()

            self.assertFalse(journal.disabled)
            self.assertFalse(
                (directory.path / REPLAY_UNAVAILABLE_FILENAME).exists()
            )

    def test_restore_rng_validates_state_and_logical_boundary(self):
        raw = {
            "ready_for_command": True,
            "available_commands": ["choose", "state"],
            "game_state": {
                "screen_type": "EVENT",
                "choice_list": ["accept"],
                "room_phase": "COMPLETE",
                "transition_pending": False,
                "replay_rng_state": RNG,
            },
        }
        calls = []

        def send(command):
            calls.append(command)
            return raw

        restore_game_rng(
            send,
            {key: tuple(value) for key, value in RNG.items()},
            stable_boundary_key(raw),
        )

        self.assertTrue(calls[0].startswith("rng_restore "))


if __name__ == "__main__":
    unittest.main()
