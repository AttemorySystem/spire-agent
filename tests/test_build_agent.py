from __future__ import annotations

import json
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
from spire_agent.subagents.build import create_build_agent as compose_build_agent
from spire_agent.subagents.build_prompt import build_prompt
from spire_agent.tools.boss_relics import build_choice_policy
from spire_agent.tools.build_flow import BuildError
from spire_agent.tools.winning_path.card_policy import review_shop
from spire_agent.subagents.build_context import (
    RUN_CONSTRUCTION_KEY,
    BuildConversationReducer,
)
from spire_agent.tools.winning_path import WinningPathCardPicker
from spire_agent.tools.events import event_rule
from spire_agent.tools.run_keys import RUN_KEYS_KEY
from spire_agent.tools.sts_db import StsDB
from spire_agent.subagents.llm import LLMRequest, LLMResponse, PromptLanguage


def create_build_agent(llm, *, prompt_language=PromptLanguage.ENGLISH):
    return compose_build_agent(
        llm,
        WinningPathCardPicker(),
        prompt_language=prompt_language,
        choice_policy=build_choice_policy,
    )


def build_state(
    screen: str,
    *,
    commands=("choose",),
    choices=(),
    details=None,
    facts=None,
) -> GameState:
    return GameState(
        owner_hint=AgentKind.BUILD,
        scope_id="seed:a1:f2:event:build",
        screen=ScreenState(
            type=screen,
            commands=commands,
            choices=choices,
            details=details or {},
        ),
        facts={
            "class": "IRONCLAD",
            "act": 1,
            "floor": 2,
            "current_hp": 60,
            "max_hp": 80,
            "gold": 99,
            "deck": [
                {"name": "Strike", "uuid": "secret-one"},
                {"name": "Defend", "uuid": "secret-two"},
            ],
            "relics": [{"name": "Burning Blood"}],
            "potions": [{"name": "Potion Slot"}],
            **(facts or {}),
        },
    )


def request(
    state: GameState,
    continuation: Continuation | None = None,
    *,
    shared=None,
) -> DecisionRequest:
    return DecisionRequest(
        state=state,
        scope=DecisionScope(AgentKind.BUILD, state.scope_id),
        continuation=continuation,
        shared=shared or {},
        previous=ContextEntry(0, None, state, True),
    )


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, current: LLMRequest) -> LLMResponse:
        self.requests.append(current)
        data = self.responses.pop(0)
        return LLMResponse(
            data,
            raw_text=json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            model="fake-build-model",
            usage={"input_tokens": 12},
        )


def response(
    action="choose",
    choice_id=0,
    targets=(),
    reason="test decision",
):
    return {
        "action": action,
        "choice_id": choice_id,
        "targets": list(targets),
        "reason": reason,
    }


def prompt_payload(current: LLMRequest) -> dict:
    return json.loads(current.messages[-1].content.split("# CURRENT STATE\n", 1)[1])


class BuildAgentTests(unittest.TestCase):
    def test_card_database_has_complete_cost_and_energy_facts(self):
        db = StsDB()
        zap = db.card("Zap")
        self.assertEqual((zap["cost"], zap["upgraded_cost"]), (1, 0))
        self.assertEqual(db.card("Adrenaline")["cost"], 0)
        self.assertEqual(db.card("Reinforced Body")["cost"], "X")
        self.assertEqual(db.card("Reflex")["cost"], "Unplayable")
        self.assertEqual(db.card("Become Almighty")["cost"], "Unplayable")
        self.assertEqual(db.card("Pride")["cost"], 1)
        self.assertIn("Gain 3 Energy", db.card("TURBO")["upgraded_effect"])
        self.assertIn("Gain 1 Energy", db.card("Fission")["effect"])

    def test_confirmed_key_choice_updates_run_key_state(self):
        state = build_state("COMBAT_REWARD")
        reducer = BuildConversationReducer()
        shared = reducer.initialize(state)
        entry = ContextEntry(
            1,
            "choose 0",
            state,
            True,
            decision=Decision(
                "choose 0", "build.key_policy", payload={"acquired_key": "emerald"}
            ),
        )

        updated = reducer.reduce(shared, entry)

        self.assertTrue(updated[RUN_KEYS_KEY]["emerald"])
        self.assertEqual(updated[RUN_KEYS_KEY]["missing"], ["ruby", "sapphire"])

    def test_confirmed_card_picker_snapshot_updates_run_state(self):
        state = build_state("COMBAT_REWARD")
        reducer = BuildConversationReducer()
        shared = reducer.initialize(state)
        construction = {"schema_version": 1, "targets": ["Hexaghost"]}
        entry = ContextEntry(
            1,
            "choose 0",
            state,
            True,
            decision=Decision(
                "choose 0",
                "card_reward.policy",
                payload={RUN_CONSTRUCTION_KEY: construction},
            ),
        )

        updated = reducer.reduce(shared, entry)

        self.assertEqual(
            updated[RUN_CONSTRUCTION_KEY],
            {"schema_version": 1, "targets": ("Hexaghost",)},
        )

    def test_coffee_dripper_requires_an_existing_future_healing_source(self):
        llm = FakeLLM(
            [
                response(choice_id=0, reason="take Coffee Dripper"),
                response(choice_id=1, reason="take Nuclear Battery"),
            ]
        )
        state = build_state(
            "BOSS_REWARD",
            choices=("coffee dripper", "nuclear battery", "frozen core"),
            facts={
                "class": "DEFECT",
                "deck": ({"name": "Zap"}, {"name": "Dualcast"}),
                "relics": ({"name": "Lee's Waffle"},),
            },
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(len(llm.requests), 2)
        policy = prompt_payload(llm.requests[0])["choice_policy"]
        self.assertEqual(policy["legal_choice_ids"], [1, 2])
        self.assertEqual(policy["recurring_healing_sources"], [])
        self.assertIn(
            "Coffee Dripper removes Rest healing",
            llm.requests[0].messages[-1].content,
        )
        self.assertIn(
            "neither card removal nor\nenergy is universally best",
            llm.requests[0].messages[-1].content,
        )

    def test_existing_healing_source_makes_coffee_dripper_eligible(self):
        for deck, relics, source in (
            (({"name": "Self Repair"},), ({"name": "Cracked Core"},), "self repair"),
            ((), ({"name": "Meat on the Bone"},), "meat on the bone"),
            ((), ({"name": "Eternal Feather"},), "eternal feather"),
            ((), ({"name": "Meal Ticket"},), "meal ticket"),
        ):
            with self.subTest(source=source):
                llm = FakeLLM([response(choice_id=0, reason="healing exists")])
                state = build_state(
                    "BOSS_REWARD",
                    choices=("coffee dripper", "nuclear battery", "frozen core"),
                    facts={"class": "DEFECT", "deck": deck, "relics": relics},
                )

                decision = create_build_agent(llm).decide(request(state))

                self.assertEqual(decision.command, "choose 0")
                policy = prompt_payload(llm.requests[0])["choice_policy"]
                self.assertEqual(policy["legal_choice_ids"], [0, 1, 2])
                self.assertEqual(policy["recurring_healing_sources"], [source])

    def test_shop_keeps_advice_required_shortlist_purchasable(self):
        state = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("purge", "sunder", "charge battery", "self repair"),
            details={
                "cards": (
                    {"name": "Sunder", "price": 86},
                    {"name": "Charge Battery", "price": 51},
                    {"name": "Self Repair", "price": 89},
                )
            },
        )

        result = review_shop(
            request(state),
            reviewer=lambda _: {
                "mode": "ADVICE_REQUIRED",
                "allowed_choice_ids": [0, 2],
                "policy": "EXPERT_EXPERIENCE_CONFLICT",
                "reason": "EXPERT_EXPERIENCE_CONFLICT",
            },
        )

        self.assertEqual(result["allowed_card_choice_ids"], [1, 3])

    def test_recall_is_blocked_before_act_three(self):
        llm = FakeLLM(
            [response(choice_id=1), response(choice_id=0, reason="rest now")]
        )
        state = build_state(
            "REST",
            choices=("rest", "recall"),
            facts={"act": 2, "floor": 24},
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(len(llm.requests), 2)
        self.assertIn("legal ids are [0]", llm.requests[1].messages[-1].content)

    def test_final_act_three_recall_is_forced(self):
        state = build_state(
            "REST",
            choices=("rest", "recall", "smith"),
            facts={"act": 3, "floor": 48},
        )
        shared = {
            RUN_KEYS_KEY: {"ruby": False, "emerald": True, "sapphire": True},
            "run_route": {"future_rests": 0},
        }

        decision = create_build_agent(FakeLLM([])).decide(request(state, shared=shared))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "build.key_policy")
        self.assertEqual(decision.payload["acquired_key"], "ruby")

    def test_dead_adventurer_is_left_at_unready_health(self):
        state = build_state(
            "EVENT",
            choices=("search", "leave"),
            details={"event_name": "Dead Adventurer"},
            facts={"current_hp": 52, "max_hp": 74},
        )

        decision = create_build_agent(FakeLLM([])).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "build.event_rule")

    def test_knowing_skull_is_left_below_the_health_budget(self):
        state = build_state(
            "EVENT",
            choices=("a pick me up?", "riches?", "success?", "how do i leave?"),
            details={"event_name": "Knowing Skull"},
            facts={"current_hp": 35, "max_hp": 71},
        )

        decision = create_build_agent(FakeLLM([])).decide(request(state))

        self.assertEqual(decision.command, "choose 3")
        self.assertEqual(decision.source, "build.event_rule")

    def test_event_hp_loss_is_blocked_before_an_at_risk_elite(self):
        state = build_state(
            "EVENT",
            choices=("sacrifice", "desecrate"),
            details={
                "event_name": "Forgotten Altar",
                "options": (
                    {"choice_index": 0, "text": "[Sacrifice] Gain 5 Max HP. Lose 26 HP."},
                    {"choice_index": 1, "text": "[Desecrate] Become Cursed - Decay."},
                ),
            },
            facts={"current_hp": 53, "max_hp": 75},
        )
        shared = {
            "run_route": {
                "planned_rooms": ("Event", "Rest", "Elite", "Rest", "Boss"),
                "encounter_readiness": {
                    "status": "AVAILABLE",
                    "entry_hp": 53,
                    "groups": {"ELITE": {"status": "AT_RISK"}},
                },
            }
        }

        decision = create_build_agent(FakeLLM([])).decide(request(state, shared=shared))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "build.event_rule")

    def test_forgotten_altar_shop_removal_cost_is_left_to_llm(self):
        state = build_state(
            "EVENT",
            choices=("sacrifice", "desecrate"),
            details={
                "event_name": "Forgotten Altar",
                "options": (
                    {"choice_index": 0, "text": "[Sacrifice] Gain 5 Max HP. Lose 25 HP."},
                    {"choice_index": 1, "text": "[Desecrate] Become Cursed - Decay."},
                ),
            },
            facts={"current_hp": 56, "max_hp": 71, "gold": 129},
        )
        shared = {
            "run_route": {
                "planned_rooms": ("Event", "Shop", "Event", "Rest", "Elite")
            }
        }

        llm = FakeLLM([response(choice_id=0)])
        decision = create_build_agent(llm).decide(request(state, shared=shared))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "build.llm")
        self.assertEqual(len(llm.requests), 1)

    def test_rest_is_forced_for_a_fresh_at_risk_route(self):
        state = build_state(
            "REST",
            choices=("rest", "smith"),
            facts={"current_hp": 32, "max_hp": 80},
        )
        shared = {
            "run_route": {
                "planned_rooms": ("Rest", "Elite", "Monster", "Boss"),
                "encounter_readiness": {
                    "status": "AVAILABLE",
                    "entry_hp": 32,
                    "groups": {"ELITE": {"status": "AT_RISK"}},
                },
                "rest_readiness": {
                    "status": "AVAILABLE",
                    "entry_hp": 56,
                    "groups": {"ELITE": {"status": "SUPPORTED"}},
                },
            }
        }

        decision = create_build_agent(FakeLLM([])).decide(request(state, shared=shared))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "build.survival_policy")

    def test_rest_is_not_forced_when_healing_does_not_resolve_the_risk(self):
        state = build_state(
            "REST",
            choices=("rest", "smith"),
            facts={"current_hp": 32, "max_hp": 80},
        )
        shared = {
            "run_route": {
                "planned_rooms": ("Rest", "Elite", "Boss"),
                "encounter_readiness": {
                    "entry_hp": 32,
                    "groups": {"ELITE": {"status": "AT_RISK"}},
                },
                "rest_readiness": {
                    "entry_hp": 56,
                    "groups": {"ELITE": {"status": "AT_RISK"}},
                },
            }
        }

        decision = create_build_agent(
            FakeLLM([response(choice_id=1, targets=("Strike",))])
        ).decide(
            request(state, shared=shared)
        )

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "build.llm")

    def test_rest_is_forced_when_it_resolves_inconclusive_route(self):
        state = build_state(
            "REST",
            choices=("rest", "smith", "recall"),
            facts={"act": 3, "current_hp": 30, "max_hp": 71},
        )
        shared = {
            RUN_KEYS_KEY: {"ruby": False, "emerald": True, "sapphire": False},
            "run_route": {
                "future_rests": 6,
                "planned_rooms": ("Rest", "Elite", "Rest", "Boss"),
                "encounter_readiness": {
                    "entry_hp": 30,
                    "groups": {"ELITE": {"status": "INCONCLUSIVE"}},
                },
                "rest_readiness": {
                    "entry_hp": 51,
                    "groups": {"ELITE": {"status": "SUPPORTED"}},
                },
            },
        }

        decision = create_build_agent(FakeLLM([])).decide(request(state, shared=shared))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "build.survival_policy")

    def test_apotheosis_deck_rests_when_simulation_improves_boss_entry(self):
        state = build_state(
            "REST",
            choices=("rest", "smith", "recall"),
            facts={
                "act": 3,
                "floor": 45,
                "current_hp": 45,
                "max_hp": 70,
                "deck": ({"name": "Apotheosis", "upgrades": 1},),
            },
        )
        shared = {
            RUN_KEYS_KEY: {"ruby": False, "emerald": True, "sapphire": True},
            "run_route": {
                "future_rests": 3,
                "planned_rooms": ("Rest", "Elite", "Rest", "Boss"),
                "encounter_readiness": {
                    "entry_hp": 45,
                    "groups": {
                        "ELITE": {
                            "status": "INCONCLUSIVE",
                            "expected_end_hp_on_win": 22.28,
                        }
                    },
                },
                "rest_readiness": {
                    "entry_hp": 70,
                    "groups": {
                        "ELITE": {
                            "status": "INCONCLUSIVE",
                            "expected_end_hp_on_win": 39.05,
                        }
                    },
                },
            },
        }

        decision = create_build_agent(FakeLLM([])).decide(request(state, shared=shared))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "build.survival_policy")

    def test_hexaghost_rest_prompt_explains_divider_hp_scaling(self):
        llm = FakeLLM([response(choice_id=1, targets=("Strike",))])
        state = build_state(
            "REST",
            choices=("rest", "smith"),
            facts={"act_boss": "Hexaghost"},
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertIn("Divider scales", llm.requests[0].messages[-1].content)
        self.assertIn(
            "rank upgrades\nby their concrete marginal impact",
            llm.requests[0].messages[-1].content,
        )

    def test_leaving_shop_continues_past_shop_room(self):
        llm = FakeLLM([])
        agent = create_build_agent(llm)
        shop = build_state("SHOP_SCREEN", commands=("leave",))

        leave = agent.decide(request(shop))

        self.assertEqual(leave.command, "leave")
        self.assertIsNotNone(leave.continuation.value)
        shop_room = build_state(
            "SHOP_ROOM",
            commands=("choose", "proceed"),
            choices=("shop",),
        )
        proceed = agent.decide(request(shop_room, leave.continuation.value))
        self.assertEqual(proceed.command, "proceed")
        self.assertEqual(proceed.source, "build.shop_exit")
        self.assertEqual(proceed.continuation.operation.value, "clear")
        self.assertEqual(llm.requests, [])

    def test_llm_shop_leave_uses_the_same_exit_continuation(self):
        llm = FakeLLM(
            [response(action="leave", choice_id=None, reason="shopping is complete")]
        )
        agent = create_build_agent(llm)
        shop = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("Wild Strike - 28 gold",),
        )

        leave = agent.decide(request(shop))

        self.assertEqual(leave.command, "leave")
        shop_room = build_state(
            "SHOP_ROOM",
            commands=("choose", "proceed"),
            choices=("shop",),
        )
        proceed = agent.decide(request(shop_room, leave.continuation.value))
        self.assertEqual(proceed.command, "proceed")
        self.assertEqual(proceed.source, "build.shop_exit")
        self.assertEqual(len(llm.requests), 1)

    def test_shop_card_policy_falls_back_to_the_unique_approved_card(self):
        llm = FakeLLM(
            [response(choice_id=1, reason="buy the cheap damage card")]
        )
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
                "deck": (
                    {"name": "Strike"},
                    {"name": "Defend"},
                    {"name": "Barricade"},
                )
            },
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 2")
        self.assertEqual(decision.source, "card_reward.shop_veto")
        policy = decision.payload["shop_card_policy"]
        self.assertEqual(policy["allowed_card_choice_ids"], (2,))
        self.assertFalse(policy["llm_approved"])
        self.assertEqual(
            decision.payload[RUN_CONSTRUCTION_KEY]["confirmed_selection"]["card"],
            "True Grit",
        )
        self.assertEqual(decision.payload["llm_proposal"]["choice_id"], 1)
        prompt = prompt_payload(llm.requests[0])
        cards = prompt["card_reward_policy"]["card_choices"]
        self.assertFalse(cards[0]["purchase_allowed"])
        self.assertTrue(cards[1]["purchase_allowed"])

    def test_shop_card_policy_allows_removal_and_shortlisted_card(self):
        for proposal, command in (
            (response(choice_id=0, targets=("Strike",)), "choose 0"),
            (response(choice_id=2), "choose 2"),
        ):
            with self.subTest(command=command):
                llm = FakeLLM([proposal])
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
                        "deck": (
                            {"name": "Strike"},
                            {"name": "Defend"},
                            {"name": "Barricade"},
                        )
                    },
                )

                decision = create_build_agent(llm).decide(request(state))

                self.assertEqual(decision.command, command)
                self.assertEqual(decision.source, "build.llm")
                self.assertTrue(
                    decision.payload["shop_card_policy"]["llm_approved"]
                )

    def test_basic_removal_cannot_forfeit_the_unique_approved_card(self):
        llm = FakeLLM(
            [response(choice_id=0, targets=("Strike",), reason="remove a Strike")]
        )
        state = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("purge", "wild strike", "true grit"),
            details={
                "cards": (
                    {"name": "Wild Strike", "price": 28},
                    {"name": "True Grit", "price": 53},
                ),
                "purge_available": True,
                "purge_cost": 75,
            },
            facts={
                "gold": 80,
                "deck": (
                    {"name": "Strike"},
                    {"name": "Defend"},
                    {"name": "Barricade"},
                ),
            },
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 2")
        self.assertEqual(decision.source, "card_reward.shop_veto")
        self.assertFalse(decision.payload["shop_card_policy"]["llm_approved"])

    def test_dollys_mirror_opens_its_card_selector(self):
        llm = FakeLLM([response(choice_id=0, targets=("Reaper",))])
        state = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("dolly's mirror",),
            details={"relics": ({"name": "Dolly's Mirror", "price": 164},)},
            facts={"gold": 200, "deck": ({"name": "Reaper"},)},
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.continuation.value.data["targets"], ("Reaper",))

    def test_shop_does_not_buy_a_card_rejected_by_winning_path(self):
        llm = FakeLLM([response(choice_id=1, reason="buy Clash")])
        state = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("purge", "clash"),
            details={"cards": ({"name": "Clash", "price": 50},)},
            facts={
                "act": 1,
                "floor": 3,
                "deck": (
                    {"name": "Strike"},
                    {"name": "Defend"},
                    {"name": "Body Slam"},
                    {"name": "Entrench"},
                ),
            },
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "leave")
        self.assertEqual(
            decision.payload["shop_card_policy"]["allowed_card_choice_ids"],
            (),
        )

    def test_shop_keeps_a_confirmed_affordable_purge_commitment(self):
        llm = FakeLLM([response(choice_id=1, reason="buy Anger instead")])
        state = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("purge", "anger"),
            details={
                "cards": ({"name": "Anger", "price": 55},),
                "purge_available": True,
                "purge_cost": 75,
            },
            facts={"gold": 116, "deck": ({"name": "Strike"}, {"name": "Defend"})},
        )
        shared = {
            "build_conversation": {
                "scope_id": state.scope_id,
                "messages": (
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "shop"},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            response(
                                choice_id=2,
                                reason="Buy Shrug It Off; then use the purge.",
                            )
                        ),
                    },
                ),
            }
        }

        decision = create_build_agent(llm).decide(request(state, shared=shared))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "card_reward.shop_veto")
        self.assertEqual(decision.payload["targets"], ("Strike",))

    def test_shop_legalizes_an_unremovable_purge_target(self):
        llm = FakeLLM(
            [response(choice_id=0, targets=("Necronomicurse",))]
        )
        state = build_state(
            "SHOP_SCREEN",
            commands=("choose", "leave"),
            choices=("purge",),
            facts={
                "deck": (
                    {"name": "Strike"},
                    {"name": "Necronomicurse"},
                )
            },
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.payload["targets"], ("Strike",))
        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(
            tuple(decision.payload["llm_proposal"]["targets"]),
            ("Necronomicurse",),
        )

    def test_missing_selection_target_uses_the_visible_grid(self):
        llm = FakeLLM([response(choice_id=0)])
        state = build_state("GRID", choices=("Strike", "Defend"))
        continuation = Continuation(
            AgentKind.BUILD,
            "build_flow",
            state.scope_id,
            expected_screens=("GRID", "HAND_SELECT"),
            data={"flow": "selection", "targets": ("Necronomicurse",)},
        )

        decision = create_build_agent(llm).decide(request(state, continuation))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "build.llm")
        self.assertEqual(decision.continuation.operation.value, "clear")
        self.assertEqual(llm.requests[0].purpose, "build.selector")

    def test_all_legacy_special_events_are_recognized(self):
        cases = {
            "Neow": "neow",
            "The Nest": "the_nest",
            "N'loth": "nloth",
            "The Woman in Blue": "woman_in_blue",
            "Masked Bandits": "masked_bandits",
            "Golden Idol": "golden_idol",
            "Liar's Game": "liars_game",
            "Mind Bloom": "mind_bloom",
            "Dead Adventurer": "dead_adventurer",
            "Knowing Skull": "knowing_skull",
        }
        for name, key in cases.items():
            with self.subTest(name=name):
                state = build_state("EVENT", details={"event_name": name})
                self.assertEqual(event_rule(state), key)

    def test_mind_bloom_prompt_rejects_unprotected_normalities(self):
        llm = FakeLLM([response(choice_id=0)])
        state = build_state(
            "EVENT",
            choices=("i am war", "i am awake", "i am rich"),
            details={"event_id": "MindBloom", "event_name": "Mind Bloom"},
        )

        create_build_agent(llm).decide(request(state))

        payload = prompt_payload(llm.requests[0])
        rule = payload["event_rule"]
        self.assertEqual(rule["key"], "mind_bloom")
        self.assertIn("Never choose I am Rich", rule["prompt"])
        self.assertIn("Normality", rule["prompt"])

    def test_event_prompt_uses_scene_section_and_compact_state(self):
        llm = FakeLLM([response(choice_id=1)])
        state = build_state(
            "EVENT",
            choices=("take", "leave"),
            details={
                "event_id": "Golden Idol",
                "body_text": "A shining idol rests on a pedestal.",
                "options": ({"text": "Take"}, {"text": "Leave"}),
            },
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "build.llm")
        self.assertEqual(decision.metrics["model"], "fake-build-model")
        self.assertEqual(llm.requests[0].purpose, "build.event")
        system = llm.requests[0].messages[0].content
        user = llm.requests[0].messages[1].content
        self.assertNotIn("visible event options", system)
        self.assertIn("visible event options", user)
        self.assertIn("Golden Idol", user)
        self.assertIn("golden_idol", user)
        self.assertNotIn("secret-one", user)

    def test_assets_are_compact_and_entity_effects_are_grounded(self):
        llm = FakeLLM([response(choice_id=1)])
        state = build_state(
            "EVENT",
            choices=("take", "leave"),
            details={"event_name": "Unknown Event"},
            facts={
                "deck": (
                    {"name": "Strike", "upgrades": 0},
                    {"name": "Strike", "upgrades": 1},
                    {"name": "Strike", "upgrades": 0},
                    {"name": "Defend", "upgrades": 0},
                ),
                "relics": ({"name": "Burning Blood"},),
                "potions": (
                    {"name": "Fire Potion"},
                    {"name": "Potion Slot"},
                ),
            },
        )

        create_build_agent(llm).decide(request(state))

        payload = prompt_payload(llm.requests[0])
        self.assertEqual(
            payload["assets"]["deck"],
            [
                {"name": "Strike", "count": 3, "upgrades": 1},
                {"name": "Defend", "count": 1, "upgrades": 0},
            ],
        )
        self.assertEqual(payload["assets"]["potions"], ["Fire Potion", None])
        cards = {row["name"]: row for row in payload["entity_facts"]["cards"]}
        relics = {row["name"]: row for row in payload["entity_facts"]["relics"]}
        potions = {row["name"]: row for row in payload["entity_facts"]["potions"]}
        self.assertEqual(cards["Strike"]["effect"], "Deal 6 damage.")
        self.assertEqual(cards["Strike"]["upgraded_effect"], "Deal 9 damage.")
        self.assertIn("heal 6 HP", relics["Burning Blood"]["effect"])
        self.assertEqual(
            potions["Fire Potion"]["effect"],
            "Deal 20 damage to target enemy.",
        )

    def test_visible_reward_card_effect_is_added_to_entity_facts(self):
        state = build_state(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=("Carnage", "Armaments"),
            details={
                "cards": ({"name": "Carnage"}, {"name": "Armaments"}),
            },
        )

        current = request(state)
        picker = WinningPathCardPicker()
        prompt = build_prompt(
            current,
            PromptLanguage.ENGLISH,
            policy_context=picker.prompt_context(picker.review(current)),
        )

        payload = prompt_payload(prompt)
        names = {row["name"] for row in payload["entity_facts"]["cards"]}
        self.assertIn("Carnage", names)
        self.assertIn("Armaments", names)

    def test_confirmed_llm_exchange_is_reused_as_room_conversation(self):
        first_output = response(choice_id=1, reason="take the event option")
        llm = FakeLLM(
            [
                first_output,
                response(choice_id=0, reason="take an allowed card"),
            ]
        )
        agent = create_build_agent(llm)
        event = build_state(
            "EVENT",
            choices=("take", "leave"),
            details={"event_name": "Unknown Event"},
        )
        first_request = request(event)
        first = agent.decide(first_request)
        reducer = BuildConversationReducer()
        shared = reducer.reduce(
            {},
            ContextEntry(
                1,
                first.command,
                event,
                True,
                scope=first_request.scope,
                decision=first,
            ),
        )

        cards = build_state(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=("Corruption", "Corruption", "Clash"),
            details={
                "cards": (
                    {"name": "Corruption"},
                    {"name": "Corruption"},
                    {"name": "Clash"},
                ),
            },
        )
        agent.decide(request(cards, shared=shared))

        first_messages = llm.requests[0].messages
        second_messages = llm.requests[1].messages
        self.assertEqual(
            [message.role for message in second_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(second_messages[0], first_messages[0])
        self.assertEqual(second_messages[1], first_messages[1])
        self.assertEqual(
            second_messages[2].content,
            json.dumps(first_output, ensure_ascii=False, separators=(",", ":")),
        )
        self.assertIn("card_reward_policy", second_messages[3].content)
        self.assertNotIn("build_conversation", second_messages[3].content)

    def test_rejected_exchange_is_not_committed_and_room_exit_clears_it(self):
        llm = FakeLLM([response(choice_id=1)])
        agent = create_build_agent(llm)
        state = build_state("EVENT", choices=("take", "leave"))
        current_request = request(state)
        decision = agent.decide(current_request)
        reducer = BuildConversationReducer()
        rejected = ContextEntry(
            1,
            decision.command,
            state,
            False,
            scope=current_request.scope,
            decision=decision,
        )
        self.assertEqual(reducer.reduce({}, rejected), {})

        confirmed = ContextEntry(
            1,
            decision.command,
            state,
            True,
            scope=current_request.scope,
            decision=decision,
        )
        shared = reducer.reduce({}, confirmed)
        self.assertIn("build_conversation", shared)
        map_state = GameState(
            AgentKind.MAP,
            "seed:a1:f2:map:map",
            ScreenState("MAP", commands=("choose",), choices=("x=0",)),
        )
        exited = ContextEntry(
            2,
            "proceed",
            map_state,
            True,
            scope=current_request.scope,
            decision=decision,
        )
        self.assertEqual(reducer.reduce(shared, exited), {})

    def test_the_nest_is_forced_without_an_llm_call(self):
        llm = FakeLLM([])
        state = build_state(
            "EVENT",
            choices=("Stay in Line - Ritual Dagger", "Smash and Grab - 99 Gold"),
            details={"event_id": "The Nest"},
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "build.event_rule")
        self.assertEqual(llm.requests, [])

    def test_chinese_neow_prompt_contains_localized_hard_rule(self):
        llm = FakeLLM([response(choice_id=1)])
        state = build_state(
            "EVENT",
            choices=("Transform a Card", "Lose your starting Relic"),
            details={"event_id": "Neow Event", "event_name": "Neow"},
        )

        create_build_agent(
            llm,
            prompt_language=PromptLanguage.CHINESE,
        ).decide(request(state))

        payload = prompt_payload(llm.requests[0])
        self.assertEqual(payload["event_rule"]["key"], "neow")
        self.assertFalse(payload["event_rule"]["prompt"].isascii())

    def test_rest_targets_execute_transactionally_across_grid(self):
        llm = FakeLLM(
            [response(choice_id=1, targets=("Strike", "Defend"))]
        )
        agent = create_build_agent(llm)
        rest = build_state(
            "REST",
            choices=("rest", "smith"),
        )

        parent = agent.decide(request(rest))
        continuation = parent.continuation.value
        self.assertEqual(parent.command, "choose 1")
        self.assertIsNotNone(continuation)

        grid = build_state(
            "GRID",
            commands=("choose", "confirm"),
            choices=("Strike", "Defend", "Bash"),
        )
        first = agent.decide(request(grid, continuation))
        self.assertEqual(first.command, "choose 0")
        second_continuation = first.continuation.value
        self.assertIsNotNone(second_continuation)

        second = agent.decide(request(grid, second_continuation))
        self.assertEqual(second.command, "choose 1")
        self.assertEqual(second.continuation.operation.value, "clear")
        self.assertEqual(len(llm.requests), 1)

    def test_duplicator_executes_the_planned_upgraded_target(self):
        llm = FakeLLM(
            [response(choice_id=0, targets=("Shrug It Off+",))]
        )
        agent = create_build_agent(llm)
        event = build_state(
            "EVENT",
            choices=("Pray", "Leave"),
            details={
                "event_id": "Duplicator",
                "options": (
                    {"text": "[Pray] Duplicate a card in your deck."},
                    {"text": "[Leave] Nothing happens."},
                ),
            },
            facts={
                "deck": (
                    {"name": "Strike", "count": 2},
                    {"name": "Shrug It Off", "count": 2, "upgrades": 1},
                )
            },
        )

        parent = agent.decide(request(event))
        continuation = parent.continuation.value
        self.assertEqual(parent.command, "choose 0")
        self.assertEqual(continuation.data["targets"], ("Shrug It Off+",))

        grid = build_state(
            "GRID",
            commands=("choose",),
            choices=("Strike", "Shrug It Off+", "Shrug It Off"),
        )
        selected = agent.decide(request(grid, continuation))

        self.assertEqual(selected.command, "choose 1")
        self.assertEqual(selected.source, "build.selection")
        self.assertEqual(len(llm.requests), 1)

    def test_rest_retries_before_opening_grid_when_smith_target_is_upgraded(self):
        llm = FakeLLM(
            [
                response(choice_id=1, targets=("Dark Embrace",)),
                response(choice_id=1, targets=("True Grit",)),
            ]
        )
        agent = create_build_agent(llm)
        rest = build_state(
            "REST",
            choices=("rest", "smith"),
            facts={
                "deck": (
                    {"name": "Dark Embrace", "upgrades": 1},
                    {"name": "True Grit", "upgrades": 0},
                )
            },
        )

        decision = agent.decide(request(rest))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.continuation.value.data["targets"], ("True Grit",))
        self.assertEqual(len(llm.requests), 2)
        self.assertIn("no matching unupgraded deck copy", llm.requests[1].messages[-1].content)

    def test_reward_flow_counts_skipped_card_rows(self):
        llm = FakeLLM(
            [response(action="skip", choice_id=None, reason="skip first card")]
        )
        agent = create_build_agent(llm)
        rewards = build_state(
            "COMBAT_REWARD",
            commands=("choose", "proceed"),
            choices=("Gold", "Card", "Card"),
            details={
                "rewards": (
                    {"reward_type": "GOLD", "gold": 20},
                    {"reward_type": "CARD"},
                    {"reward_type": "CARD"},
                )
            },
        )

        gold = agent.decide(request(rewards))
        self.assertEqual(gold.command, "choose 0")

        cards = build_state(
            "COMBAT_REWARD",
            commands=("choose", "proceed"),
            choices=("Card", "Card"),
            details={
                "rewards": (
                    {"reward_type": "CARD"},
                    {"reward_type": "CARD"},
                )
            },
        )
        open_first = agent.decide(request(cards, gold.continuation.value))
        self.assertEqual(open_first.command, "choose 0")

        card_reward = build_state(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=("Test Card A", "Test Card B", "Test Card C"),
            details={
                "cards": (
                    {"name": "Test Card A"},
                    {"name": "Test Card B"},
                    {"name": "Test Card C"},
                )
            },
        )
        skipped = agent.decide(request(card_reward, open_first.continuation.value))
        self.assertEqual(skipped.command, "skip")
        self.assertEqual(
            skipped.continuation.value.data["skipped_card_rows"],
            1,
        )

        open_second = agent.decide(request(cards, skipped.continuation.value))
        self.assertEqual(open_second.command, "choose 1")

    def test_full_potion_reward_is_skipped_without_calling_llm(self):
        llm = FakeLLM([])
        state = build_state(
            "COMBAT_REWARD",
            commands=("choose", "proceed"),
            choices=("Potion",),
            details={"rewards": ({"reward_type": "POTION"},)},
            facts={"potions": ({"name": "Fire Potion"}, {"name": "Block Potion"})},
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "proceed")
        self.assertEqual(decision.source, "build.rewards_done")
        self.assertEqual(llm.requests, [])

    def test_sapphire_key_is_deferred_until_act_three(self):
        llm = FakeLLM([response(choice_id=1)])
        state = build_state(
            "COMBAT_REWARD",
            commands=("choose", "proceed"),
            choices=("Relic", "Sapphire Key"),
            details={
                "rewards": (
                    {"reward_type": "RELIC", "relic": {"name": "Bag of Prep"}},
                    {"reward_type": "SAPPHIRE_KEY"},
                )
            },
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(llm.requests, [])

    def test_sapphire_key_is_forced_in_act_three(self):
        llm = FakeLLM([])
        state = build_state(
            "COMBAT_REWARD",
            commands=("choose", "proceed"),
            choices=("Relic", "Sapphire Key"),
            details={
                "rewards": (
                    {"reward_type": "RELIC"},
                    {"type": "KEY", "value": "SAPPHIRE"},
                )
            },
            facts={"act": 3, "floor": 40},
        )

        decision = create_build_agent(llm).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "build.key_policy")
        self.assertEqual(decision.payload["acquired_key"], "sapphire")

    def test_invalid_llm_output_fails_closed(self):
        state = build_state("EVENT", choices=("one", "two"))
        cases = (
            (response(choice_id=4), "illegal choice"),
            (response(action="skip", choice_id=0), "choice_id=null"),
            (response(choice_id=0, targets=("Strike",)), "targets are only valid"),
        )
        for output, message in cases:
            with self.subTest(output=output):
                with self.assertRaisesRegex(BuildError, message):
                    create_build_agent(FakeLLM([output, output])).decide(request(state))


if __name__ == "__main__":
    unittest.main()
