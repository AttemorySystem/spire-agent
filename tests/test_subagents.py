from __future__ import annotations

import unittest

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    Continuation,
    Decision,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.registry import SubAgentRegistry
from spire_agent.subagents import (
    BuildAgent,
    CombatAgent,
    MapAgent,
    SubAgentDecisionError,
)


def request(
    kind: AgentKind,
    *,
    continuation: Continuation | None = None,
) -> DecisionRequest:
    state = GameState(
        owner_hint=kind,
        scope_id=f"{kind.value}-room",
        screen=ScreenState(type="GRID", commands=("choose",)),
    )
    return DecisionRequest(
        state=state,
        scope=DecisionScope(kind, state.scope_id),
        continuation=continuation,
        shared={},
        previous=ContextEntry(
            index=0,
            command=None,
            state=state,
            confirmed=True,
        ),
    )


class Stage:
    def __init__(self, name: str, calls: list[str], result=None):
        self.name = name
        self.calls = calls
        self.result = result

    def try_decide(self, current):
        self.calls.append(self.name)
        return self.result


class Fallback:
    def __init__(self, calls: list[str], result):
        self.calls = calls
        self.result = result

    def decide(self, current):
        self.calls.append("fallback")
        return self.result


class SmallSubAgentTests(unittest.TestCase):
    def test_all_three_agents_share_one_pipeline_shape(self):
        decision = Decision("choose 0", "test")
        registry = SubAgentRegistry(
            (
                BuildAgent(fast_paths=(Stage("build", [], decision),)),
                MapAgent(fast_paths=(Stage("map", [], decision),)),
                CombatAgent(fast_paths=(Stage("combat", [], decision),)),
            )
        )

        for kind in AgentKind:
            self.assertEqual(registry.get(kind).decide(request(kind)), decision)

    def test_continuation_runs_first_and_short_circuits(self):
        calls: list[str] = []
        continuation = Continuation(
            owner=AgentKind.COMBAT,
            kind="card_selection",
            scope_id="combat-room",
        )
        expected = Decision("choose 1", "continuation")
        agent = CombatAgent(
            continuation_stages=(Stage("continuation", calls, expected),),
            fast_paths=(Stage("fast", calls, Decision("end", "fast")),),
            tool_stages=(Stage("tool", calls, Decision("play 1", "tool")),),
            fallback=Fallback(calls, Decision("end", "fallback")),
        )

        result = agent.decide(request(AgentKind.COMBAT, continuation=continuation))

        self.assertEqual(result, expected)
        self.assertEqual(calls, ["continuation"])

    def test_continuation_stages_are_skipped_without_continuation(self):
        calls: list[str] = []
        expected = Decision("choose 0", "fast")
        agent = BuildAgent(
            continuation_stages=(Stage("continuation", calls, None),),
            fast_paths=(Stage("fast", calls, expected),),
        )

        self.assertEqual(agent.decide(request(AgentKind.BUILD)), expected)
        self.assertEqual(calls, ["fast"])

    def test_pipeline_falls_through_in_fixed_order(self):
        calls: list[str] = []
        expected = Decision("choose 2", "fallback")
        agent = MapAgent(
            fast_paths=(Stage("fast-1", calls), Stage("fast-2", calls)),
            tool_stages=(Stage("tool", calls),),
            fallback=Fallback(calls, expected),
        )

        result = agent.decide(request(AgentKind.MAP))

        self.assertEqual(result, expected)
        self.assertEqual(calls, ["fast-1", "fast-2", "tool", "fallback"])

    def test_unhandled_request_fails_closed(self):
        agent = BuildAgent()
        with self.assertRaisesRegex(SubAgentDecisionError, "build:GRID"):
            agent.decide(request(AgentKind.BUILD))

    def test_wrong_owner_and_invalid_stage_result_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "build SubAgent received map"):
            BuildAgent().decide(request(AgentKind.MAP))

        invalid = BuildAgent(fast_paths=(Stage("bad", [], "choose 0"),))
        with self.assertRaisesRegex(SubAgentDecisionError, "expected Decision"):
            invalid.decide(request(AgentKind.BUILD))


if __name__ == "__main__":
    unittest.main()
