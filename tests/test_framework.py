from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from spire_agent.context import GameContext
from spire_agent.contracts import (
    AgentKind,
    Continuation,
    ContinuationChange,
    Decision,
    DecisionRequest,
    DecisionScope,
    ExecutionResult,
    GameState,
    RoutedDecision,
    ScreenState,
    SessionRefresh,
)
from spire_agent.errors import ActionValidationError, ContextError, RegistryError
from spire_agent.game_agent import GameAgent
from spire_agent.observers import ObserverHub
from spire_agent.registry import SubAgentRegistry
from spire_agent.router import RoomScopeRouter, RoutedDecisionProvider
from spire_agent.validation import AvailableCommandValidator


def state(
    owner: AgentKind,
    *,
    scope_id: str = "room-1",
    screen_type: str = "EVENT",
    commands: tuple[str, ...] = ("choose",),
    terminal: bool = False,
    facts: dict | None = None,
) -> GameState:
    return GameState(
        owner_hint=owner,
        scope_id=scope_id,
        screen=ScreenState(
            type=screen_type,
            commands=commands,
            choices=({"id": 0, "name": "first"},),
            interaction_id=f"{scope_id}:{screen_type}",
        ),
        terminal=terminal,
        facts=facts or {},
    )


class StubAgent:
    def __init__(self, kind: AgentKind, decision: Decision):
        self.kind = kind
        self.decision = decision
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> Decision:
        self.requests.append(request)
        return self.decision


def registry_with(
    *,
    build: StubAgent | None = None,
    map_agent: StubAgent | None = None,
    combat: StubAgent | None = None,
) -> SubAgentRegistry:
    return SubAgentRegistry(
        (
            build or StubAgent(AgentKind.BUILD, Decision("choose 0", "build")),
            map_agent or StubAgent(AgentKind.MAP, Decision("choose 0", "map")),
            combat
            or StubAgent(AgentKind.COMBAT, Decision("end", "combat")),
        )
    )


class FrameworkContractTests(unittest.TestCase):
    def test_state_payloads_are_deeply_immutable_and_detached(self):
        source = {"deck": [{"name": "Strike"}]}
        current = state(AgentKind.BUILD, facts=source)
        source["deck"][0]["name"] = "Corruption"

        self.assertEqual(current.facts["deck"][0]["name"], "Strike")
        with self.assertRaises(TypeError):
            current.facts["deck"][0]["name"] = "Bash"
        with self.assertRaises(FrozenInstanceError):
            current.scope_id = "different"

    def test_registry_requires_exactly_one_agent_per_owner(self):
        with self.assertRaisesRegex(RegistryError, "missing SubAgent"):
            SubAgentRegistry(
                (StubAgent(AgentKind.BUILD, Decision("choose 0", "build")),)
            )
        with self.assertRaisesRegex(RegistryError, "duplicate SubAgent"):
            SubAgentRegistry(
                (
                    StubAgent(AgentKind.BUILD, Decision("choose 0", "one")),
                    StubAgent(AgentKind.BUILD, Decision("choose 0", "two")),
                )
            )

    def test_continuation_owner_overrides_page_owner_hint(self):
        context = GameContext()
        context.start(state(AgentKind.BUILD))
        continuation = Continuation(
            owner=AgentKind.COMBAT,
            kind="multi_card_select",
            scope_id="combat-7",
            expected_screens=("HAND_SELECT",),
        )
        context.stage(
            RoutedDecision(
                DecisionScope(AgentKind.BUILD, "room-1"),
                Decision(
                    "choose 0",
                    "build",
                    continuation=ContinuationChange.set(continuation),
                ),
            )
        )
        context.confirm(
            ExecutionResult(
                command="choose 0",
                state=state(
                    AgentKind.BUILD,
                    screen_type="HAND_SELECT",
                    scope_id="room-1",
                ),
                confirmed=True,
            )
        )

        scope = RoomScopeRouter().route(context.view())

        self.assertEqual(scope.owner, AgentKind.COMBAT)
        self.assertEqual(scope.id, "combat-7")

    def test_continuation_expires_after_leaving_expected_screen(self):
        context = GameContext()
        context.start(state(AgentKind.COMBAT, screen_type="NONE"))
        continuation = Continuation(
            owner=AgentKind.COMBAT,
            kind="mcts_selection",
            scope_id="combat-7",
            expected_screens=("GRID",),
        )
        context.stage(
            RoutedDecision(
                DecisionScope(AgentKind.COMBAT, "combat-7"),
                Decision(
                    "play 4",
                    "mcts",
                    continuation=ContinuationChange.set(continuation),
                ),
            )
        )
        context.confirm(
            ExecutionResult(
                command="play 4",
                state=state(
                    AgentKind.COMBAT,
                    screen_type="GRID",
                    commands=("choose",),
                    scope_id="combat-7",
                ),
                confirmed=True,
            )
        )
        context.stage(
            RoutedDecision(
                DecisionScope(AgentKind.COMBAT, "combat-7"),
                Decision("choose 0", "mcts selection"),
            )
        )
        context.confirm(
            ExecutionResult(
                command="choose 0",
                state=state(
                    AgentKind.BUILD,
                    screen_type="COMBAT_REWARD",
                    commands=("choose",),
                    scope_id="act2-floor24",
                ),
                confirmed=True,
            )
        )
        build = StubAgent(
            AgentKind.BUILD,
            Decision("choose 0", "build reward"),
        )
        combat = StubAgent(
            AgentKind.COMBAT,
            Decision("end", "combat"),
        )
        provider = RoutedDecisionProvider(
            RoomScopeRouter(),
            registry_with(build=build, combat=combat),
        )

        routed = provider.decide(context.view())

        self.assertEqual(routed.scope.owner, AgentKind.BUILD)
        self.assertIsNone(build.requests[0].continuation)
        self.assertFalse(combat.requests)
        self.assertEqual(
            routed.decision.continuation.operation.value,
            "clear",
        )
        context.stage(routed)
        context.confirm(
            ExecutionResult(
                command="choose 0",
                state=state(
                    AgentKind.BUILD,
                    screen_type="CARD_REWARD",
                    commands=("choose",),
                    scope_id="act2-floor24",
                ),
                confirmed=True,
            )
        )
        self.assertIsNone(context.view().continuation)

    def test_rejected_command_does_not_commit_continuation(self):
        context = GameContext()
        initial = state(AgentKind.BUILD)
        context.start(initial)
        continuation = Continuation(
            owner=AgentKind.BUILD,
            kind="shop_plan",
            scope_id="shop-1",
        )
        context.stage(
            RoutedDecision(
                DecisionScope(AgentKind.BUILD, "room-1"),
                Decision(
                    "choose 0",
                    "build",
                    continuation=ContinuationChange.set(continuation),
                ),
            )
        )

        entry = context.confirm(
            ExecutionResult(
                command="choose 0",
                state=initial,
                confirmed=False,
                error="game rejected command",
            )
        )

        self.assertFalse(entry.confirmed)
        self.assertIsNone(context.view().continuation)

    def test_external_resync_clears_continuation_and_reinitializes_shared_state(self):
        class Reducer:
            def initialize(self, current):
                return {"floor": current.facts["floor"]}

            def reduce(self, shared, entry):
                return shared

        context = GameContext(Reducer())
        initial = state(AgentKind.BUILD, facts={"floor": 1})
        context.start(initial)
        continuation = Continuation(
            AgentKind.BUILD,
            "selection",
            "room-1",
            ("EVENT",),
        )
        context.stage(
            RoutedDecision(
                DecisionScope(AgentKind.BUILD, "room-1"),
                Decision(
                    "choose 0",
                    "build",
                    continuation=ContinuationChange.set(continuation),
                ),
            )
        )
        context.confirm(ExecutionResult("choose 0", initial, True))

        entry = context.resync(
            state(AgentKind.MAP, facts={"floor": 2}, screen_type="MAP")
        )

        self.assertIsNone(entry.command)
        self.assertIsNone(context.view().continuation)
        self.assertIsNone(context.view().active_scope)
        self.assertEqual(context.view().shared["floor"], 2)

    def test_shared_state_is_reduced_only_after_confirmation(self):
        class Reducer:
            def initialize(self, initial):
                return {"confirmed_commands": 0}

            def reduce(self, shared, entry):
                return {
                    "confirmed_commands": shared["confirmed_commands"] + 1
                }

        context = GameContext(Reducer())
        current = state(AgentKind.BUILD)
        context.start(current)
        routed = RoutedDecision(
            DecisionScope(AgentKind.BUILD, "room-1"),
            Decision("choose 0", "build"),
        )
        context.stage(routed)
        context.confirm(
            ExecutionResult(
                "choose 0",
                current,
                confirmed=False,
                error="rejected",
            )
        )
        self.assertEqual(context.view().shared["confirmed_commands"], 0)

        context.stage(routed)
        context.confirm(
            ExecutionResult("choose 0", current, confirmed=True)
        )
        self.assertEqual(context.view().shared["confirmed_commands"], 1)
        with self.assertRaises(TypeError):
            context.view().shared["confirmed_commands"] = 99

    def test_context_rejects_mismatched_execution_result(self):
        context = GameContext()
        current = state(AgentKind.BUILD)
        context.start(current)
        context.stage(
            RoutedDecision(
                DecisionScope(AgentKind.BUILD, "room-1"),
                Decision("choose 0", "build"),
            )
        )

        with self.assertRaisesRegex(ContextError, "does not match"):
            context.confirm(
                ExecutionResult("choose 1", current, confirmed=True)
            )

    def test_validator_only_checks_framework_command_boundary(self):
        validator = AvailableCommandValidator()
        current = state(AgentKind.COMBAT, commands=("play", "end"))
        validator.validate(current, Decision("play 2 0", "mcts"))

        with self.assertRaises(ActionValidationError):
            validator.validate(current, Decision("choose 2", "llm"))


class FakeSession:
    def __init__(self, initial: GameState, results: list[ExecutionResult]):
        self.initial = initial
        self.results = list(results)
        self.commands: list[str] = []
        self.closed = False

    def reset(self) -> GameState:
        return self.initial

    def execute(self, command: str) -> ExecutionResult:
        self.commands.append(command)
        result = self.results.pop(0)
        if result.command != command:
            raise AssertionError("fake session command mismatch")
        return result

    def close(self) -> None:
        self.closed = True


class RecordingObserver:
    def __init__(self):
        self.entries = []

    def on_entry(self, entry):
        self.entries.append(entry)


class FailingObserver:
    def on_entry(self, entry):
        raise RuntimeError("display unavailable")


class GameAgentTests(unittest.TestCase):
    def test_game_agent_routes_without_domain_fast_path_logic(self):
        build = StubAgent(
            AgentKind.BUILD,
            Decision("choose 0", "build_fast_path", "only event option"),
        )
        map_agent = StubAgent(
            AgentKind.MAP,
            Decision("choose 1", "map"),
        )
        combat = StubAgent(
            AgentKind.COMBAT,
            Decision("end", "mcts"),
        )
        provider = RoutedDecisionProvider(
            RoomScopeRouter(),
            registry_with(build=build, map_agent=map_agent, combat=combat),
        )
        terminal = state(
            AgentKind.BUILD,
            screen_type="GAME_OVER",
            commands=(),
            terminal=True,
        )
        session = FakeSession(
            state(AgentKind.BUILD),
            [ExecutionResult("choose 0", terminal, confirmed=True)],
        )
        recorder = RecordingObserver()
        hub = ObserverHub((recorder, FailingObserver()))
        context = GameContext()
        game = GameAgent(
            session=session,
            context=context,
            decisions=provider,
            validator=AvailableCommandValidator(),
            observers=hub,
        )

        final = game.run()

        self.assertTrue(final.state.terminal)
        self.assertEqual(session.commands, ["choose 0"])
        self.assertTrue(session.closed)
        self.assertEqual(len(build.requests), 1)
        self.assertEqual(len(map_agent.requests), 0)
        self.assertEqual(len(combat.requests), 0)
        self.assertEqual(len(recorder.entries), 2)
        self.assertEqual(len(hub.failures), 2)
        request = build.requests[0]
        self.assertFalse(hasattr(request, "entries"))
        self.assertEqual(request.previous.command, None)

    def test_rejected_result_is_recorded_and_routed_again(self):
        class RetryBuildAgent:
            kind = AgentKind.BUILD

            def __init__(self):
                self.calls = 0

            def decide(self, request):
                self.calls += 1
                return Decision(
                    "choose 0" if self.calls == 1 else "proceed",
                    "build",
                )

        build = RetryBuildAgent()
        provider = RoutedDecisionProvider(
            RoomScopeRouter(), registry_with(build=build)
        )
        refreshed = state(AgentKind.BUILD, commands=("proceed",))
        terminal = state(
            AgentKind.BUILD,
            screen_type="GAME_OVER",
            commands=(),
            terminal=True,
        )
        session = FakeSession(
            state(AgentKind.BUILD),
            [
                ExecutionResult(
                    "choose 0",
                    refreshed,
                    confirmed=False,
                    error="stale choice",
                ),
                ExecutionResult("proceed", terminal, confirmed=True),
            ],
        )
        context = GameContext()

        GameAgent(
            session=session,
            context=context,
            decisions=provider,
            validator=AvailableCommandValidator(),
        ).run()

        self.assertEqual(build.calls, 2)
        self.assertEqual(len(context.entries), 3)
        self.assertFalse(context.entries[1].confirmed)
        self.assertTrue(context.entries[2].confirmed)

    def test_control_refreshes_and_resyncs_before_agent_decides(self):
        refreshed = state(
            AgentKind.MAP,
            scope_id="map-2",
            screen_type="MAP",
            commands=("choose",),
        )
        terminal = state(
            AgentKind.BUILD,
            screen_type="GAME_OVER",
            commands=(),
            terminal=True,
        )

        class RefreshingSession(FakeSession):
            def refresh(self):
                return SessionRefresh(refreshed, True)

        class RefreshOnce:
            def __init__(self):
                self.calls = 0

            def before_decision(self, current):
                self.calls += 1
                return self.calls == 1

        build = StubAgent(AgentKind.BUILD, Decision("choose 0", "build"))
        map_agent = StubAgent(AgentKind.MAP, Decision("choose 0", "map"))
        provider = RoutedDecisionProvider(
            RoomScopeRouter(), registry_with(build=build, map_agent=map_agent)
        )
        session = RefreshingSession(
            state(AgentKind.BUILD),
            [ExecutionResult("choose 0", terminal, True)],
        )
        context = GameContext()

        GameAgent(
            session=session,
            context=context,
            decisions=provider,
            validator=AvailableCommandValidator(),
            control=RefreshOnce(),
        ).run()

        self.assertFalse(build.requests)
        self.assertEqual(len(map_agent.requests), 1)
        self.assertEqual(context.entries[1].command, None)
        self.assertEqual(context.entries[1].state.screen.type, "MAP")


if __name__ == "__main__":
    unittest.main()
