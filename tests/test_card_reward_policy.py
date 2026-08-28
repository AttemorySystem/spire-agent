from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.extensions import RunDirectory, WinningPathRecorder
from spire_agent.subagents.build import create_build_agent as compose_build_agent
from spire_agent.subagents.llm import LLMRequest, LLMResponse
from spire_agent.tools.winning_path import WinningPathCardPicker, review
from spire_agent.utils.winning_path_log import render


STARTER = (
    "Ascender's Bane",
    "Bash",
    *("Defend",) * 4,
    *("Strike",) * 5,
)


def create_build_agent(llm):
    return compose_build_agent(llm, WinningPathCardPicker())


def card_state(
    choices,
    *,
    deck=STARTER,
    act=1,
    floor=1,
    singing_bowl=False,
    detail_names=None,
):
    return GameState(
        AgentKind.BUILD,
        f"seed:a{act}:f{floor}:reward:build",
        ScreenState(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=tuple(choices),
            details={
                "cards": tuple(
                    {"name": name} for name in (detail_names or choices)
                ),
                "singing_bowl": singing_bowl,
            },
        ),
        facts={
            "class": "IRONCLAD",
            "act": act,
            "floor": floor,
            "deck": tuple({"name": name} for name in deck),
            "relics": ({"name": "Burning Blood"},),
        },
    )


def request(state):
    scope = DecisionScope(AgentKind.BUILD, state.scope_id)
    return DecisionRequest(
        state,
        scope,
        None,
        {},
        ContextEntry(0, None, state, True, scope=scope),
    )


class FakeLLM:
    def __init__(self, data):
        self.data = data
        self.requests: list[LLMRequest] = []

    def complete(self, prompt):
        self.requests.append(prompt)
        return LLMResponse(
            self.data,
            raw_text=json.dumps(self.data, separators=(",", ":")),
            model="fake",
        )


class CardRewardPolicyTests(unittest.TestCase):
    def test_act_one_apotheosis_is_deterministic(self):
        llm = FakeLLM({})
        decision = create_build_agent(llm).decide(
            request(
                card_state(
                    ("Hand of Greed", "Apotheosis", "Secret Technique"),
                    floor=0,
                )
            )
        )

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(llm.requests, [])
        self.assertEqual(
            decision.payload["winning_path_review"]["policy"],
            "TEMPLATE_PROGRESS",
        )

    def test_live_names_come_from_card_details(self):
        result = review(
            request(
                card_state(
                    ("body slam", "clothesline", "havoc"),
                    deck=(*STARTER, "Barricade"),
                    detail_names=("Body Slam", "Clothesline", "Havoc"),
                )
            )
        )

        self.assertEqual(result["candidates"][0]["name"], "Body Slam")

    def test_worthy_card_precedes_singing_bowl(self):
        decision = create_build_agent(FakeLLM({})).decide(
            request(card_state(("Apotheosis", "Clash"), singing_bowl=True))
        )

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(
            decision.payload["card_reward_policy_result"]["card"],
            "Apotheosis",
        )

    def test_singing_bowl_replaces_skip(self):
        decision = create_build_agent(FakeLLM({})).decide(
            request(card_state(("Clash",), singing_bowl=True))
        )

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(
            decision.payload["card_reward_policy_result"]["card"],
            "Singing Bowl",
        )

    def test_plain_skip_is_not_recorded_as_singing_bowl(self):
        decision = create_build_agent(FakeLLM({})).decide(
            request(card_state(("Clash",)))
        )

        self.assertEqual(decision.command, "skip")
        self.assertIsNone(
            decision.payload["card_reward_policy_result"]["card"]
        )

    def test_audit_log_renders_evidence(self):
        current = request(card_state(("Apotheosis", "Clash")))
        decision = create_build_agent(FakeLLM({})).decide(current)
        entry = ContextEntry(
            1,
            decision.command,
            current.state,
            True,
            scope=current.scope,
            decision=decision,
        )

        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary))
            path = directory.bind("ABC123") / "winning_path" / "000001.json"
            WinningPathRecorder(directory).on_entry(entry)
            text = render(json.loads(path.read_text("utf-8")))

        self.assertIn("policy: TEMPLATE_PROGRESS", text)
        self.assertIn("[0] Apotheosis", text)
        self.assertIn("template=CORE_ACTIVATION", text)


if __name__ == "__main__":
    unittest.main()
