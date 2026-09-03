from __future__ import annotations

import unittest

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.subagents.map import MapDecisionError, create_map_agent as compose_map_agent
from spire_agent.subagents.build_context import RUN_CONSTRUCTION_KEY
from spire_agent.subagents.llm import (
    LLMOutputError,
    LLMRequest,
    LLMResponse,
    PromptLanguage,
)
from spire_agent.tools.map import DefaultMapTool, MapError, render_map
from spire_agent.tools.map.tool import _needed_families, _route_at_risk


def create_map_agent(llm, *, prompt_language=PromptLanguage.ENGLISH):
    return compose_map_agent(DefaultMapTool(llm, prompt_language))


def map_state(
    *,
    choices=("x=0", "x=2"),
    current_node=None,
    nodes=None,
) -> GameState:
    if nodes is None:
        nodes = [
            {
                "x": 0,
                "y": 0,
                "symbol": "M",
                "children": [{"x": 1, "y": 1}],
            },
            {
                "x": 2,
                "y": 0,
                "symbol": "?",
                "children": [{"x": 1, "y": 1}],
            },
            {
                "x": 1,
                "y": 1,
                "symbol": "E",
                "is_burning": True,
                "children": [{"x": 3, "y": 16}],
            },
        ]
    details = {} if current_node is None else {"current_node": current_node}
    return GameState(
        owner_hint=AgentKind.MAP,
        scope_id="seed:a1:f0:map:map",
        screen=ScreenState(
            type="MAP",
            commands=("choose", "state"),
            choices=choices,
            details=details,
        ),
        facts={
            "class": "IRONCLAD",
            "ascension_level": 20,
            "act": 1,
            "floor": 0,
            "current_hp": 70,
            "max_hp": 80,
            "gold": 99,
            "act_boss": "The Guardian",
            "deck": [{"name": "Strike"}, {"name": "Strike"}, {"name": "Bash"}],
            "relics": [{"name": "Burning Blood"}],
            "potions": [{"name": "Potion Slot"}],
            "map": nodes,
        },
    )


def forced_branch_state() -> GameState:
    return map_state(
        choices=("x=0",),
        nodes=[
            {
                "x": 0,
                "y": 0,
                "symbol": "M",
                "children": [{"x": 0, "y": 1}, {"x": 1, "y": 1}],
            },
            {
                "x": 0,
                "y": 1,
                "symbol": "R",
                "children": [{"x": 3, "y": 16}],
            },
            {
                "x": 1,
                "y": 1,
                "symbol": "$",
                "children": [{"x": 3, "y": 16}],
            },
        ],
    )


def request(state: GameState, *, shared=None) -> DecisionRequest:
    return DecisionRequest(
        state=state,
        scope=DecisionScope(AgentKind.MAP, state.scope_id),
        continuation=None,
        shared=shared or {},
        previous=ContextEntry(
            index=0,
            command=None,
            state=state,
            confirmed=True,
        ),
    )


class FakeLLM:
    def __init__(self, data):
        self.data = data
        self.requests: list[LLMRequest] = []

    def complete(self, current: LLMRequest) -> LLMResponse:
        self.requests.append(current)
        return LLMResponse(
            self.data,
            model="fake-map-model",
            usage={"input_tokens": 10},
        )


class MapViewTests(unittest.TestCase):
    def test_renders_proven_reachable_adjacency_format(self):
        graph, options = render_map(map_state())

        self.assertEqual(
            graph,
            "MAP_GRAPH v1\n"
            "direction: layer increases toward BOSS\n"
            "node_id: L<layer>C<column>\n"
            "legend: M=Combat ?=Unknown(Event/Combat/Shop) E=Elite "
            "E*=BurningElite R=Rest T=Chest $=Shop\n"
            "current: START\n"
            "choices: 0->L00C0 | 1->L00C2\n"
            "L00: C0 M->[L01C1] | C2 ?->[L01C1]\n"
            "L01: C1 E*->[BOSS]\n"
            "BOSS: The Guardian\n",
        )
        self.assertEqual(
            [(item["choice_id"], item["node"], item["room"]) for item in options],
            [(0, "L00C0", "M"), (1, "L00C2", "?")],
        )
        self.assertEqual(
            options[0]["planned_rooms"],
            ["Monster", "Burning Elite", "Boss"],
        )

    def test_recovers_only_unambiguous_stale_previous_act_boss(self):
        graph, _ = render_map(
            map_state(current_node={"x": 3, "y": 15})
        )
        self.assertIn("current: START\n", graph)
        self.assertIn("choices: 0->L00C0 | 1->L00C2\n", graph)

    def test_missing_choice_or_reachable_edge_fails_closed(self):
        with self.assertRaisesRegex(
            MapError,
            "map choice points to missing node L00C6",
        ):
            render_map(map_state(choices=("x=6",)))

        broken_nodes = [
            {
                "x": 0,
                "y": 0,
                "symbol": "M",
                "children": [{"x": 1, "y": 1}],
            }
        ]
        with self.assertRaisesRegex(
            MapError,
            "reachable edge points to missing node L01C1",
        ):
            render_map(map_state(choices=("x=0",), nodes=broken_nodes))

    def test_act_four_accepts_only_the_implicit_heart_after_the_elite(self):
        nodes = [
            {"x": 3, "y": 0, "symbol": "R", "children": [{"x": 3, "y": 1}]},
            {"x": 3, "y": 1, "symbol": "$", "children": [{"x": 3, "y": 2}]},
            {"x": 3, "y": 2, "symbol": "E", "children": [{"x": 3, "y": 3}]},
        ]
        base = map_state(
            choices=("x=3",),
            current_node={"x": -1, "y": 15},
            nodes=nodes,
        )
        state = GameState(
            base.owner_hint,
            base.scope_id,
            base.screen,
            facts={**base.facts, "act": 4, "act_boss": "The Heart"},
        )

        graph, options = render_map(state)

        self.assertIn("L02: C3 E->[BOSS]", graph)
        self.assertIn("BOSS: The Heart\n", graph)
        self.assertEqual(options[0]["forced_segment"][-1]["room"], "E")

        bad = GameState(
            base.owner_hint,
            base.scope_id,
            base.screen,
            facts={
                **base.facts,
                "act": 4,
                "map": [*nodes[:-1], {**nodes[-1], "symbol": "R"}],
            },
        )
        with self.assertRaisesRegex(MapError, "missing node L03C3"):
            render_map(bad)


class ConcreteMapAgentTests(unittest.TestCase):
    def test_burning_elite_requires_distinct_readiness_evidence(self):
        option = {"forced_segment": ({"room": "E*"},)}

        self.assertEqual(_needed_families(1, (option,)), ("BURNING_ELITE",))
        self.assertTrue(_route_at_risk(
            option,
            1,
            {"groups": {"ELITE": {"status": "SUPPORTED"}}},
        ))

    def test_act_one_hallway_does_not_hide_a_later_forced_elite(self):
        options = ({"forced_segment": ({"room": "M"}, {"room": "E"})},)
        self.assertEqual(_needed_families(1, options), ("ELITE",))

    def test_one_fight_readiness_is_not_reused_for_a_second_fight(self):
        class Supported:
            def __init__(self):
                self.families = []

            def evaluate(self, state, families):
                self.families.append(families)
                return {
                    "status": "AVAILABLE",
                    "entry_hp": state.facts["current_hp"],
                    "groups": {
                        "WEAK_HALLWAY": {"status": "SUPPORTED"},
                        "STRONG_HALLWAY": {"status": "SUPPORTED"},
                        "ELITE": {"status": "SUPPORTED"},
                    },
                }

        state = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "M", "children": [{"x": 0, "y": 7}]},
                {"x": 0, "y": 7, "symbol": "E", "children": [{"x": 0, "y": 8}]},
                {"x": 0, "y": 8, "symbol": "R", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 6, "symbol": "M", "children": [{"x": 1, "y": 7}]},
                {"x": 1, "y": 7, "symbol": "R", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "act": 2, "floor": 20},
        )
        llm = FakeLLM({"choice_id": 0, "reason": "reuse stale evidence"})

        readiness = Supported()
        decision = compose_map_agent(
            DefaultMapTool(llm, encounter_readiness=readiness)
        ).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(readiness.families, [("HALLWAY",)])
        self.assertEqual(decision.source, "map.run_policy")
        self.assertEqual(llm.requests, [])
        self.assertTrue(
            _route_at_risk(
                {"forced_segment": ({"room": "M"}, {"room": "E"})},
                1,
                {"groups": {"ELITE": {"status": "SUPPORTED"}}},
            )
        )

    def test_act_two_immediate_elite_requires_supported_simulation(self):
        class AtRisk:
            def evaluate(self, state, families):
                return {"status": "AT_RISK", "estimated_survival": 0.30}

        state = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "E", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 6, "symbol": "M", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "act": 2, "floor": 24, "current_hp": 16},
        )
        llm = FakeLLM({"choice_id": 0, "reason": "take elite"})
        agent = compose_map_agent(DefaultMapTool(llm, encounter_readiness=AtRisk()))

        decision = agent.decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "map.run_policy")
        self.assertEqual(llm.requests, [])

    def test_all_dangerous_routes_use_survival_evidence(self):
        class AtRisk:
            def evaluate(self, state, families):
                return {
                    "status": "AVAILABLE",
                    "weak_hallways_remaining": 1,
                    "groups": {
                        "WEAK_HALLWAY": {
                            "status": "AT_RISK",
                            "estimated_survival": 0.10,
                            "expected_end_hp_on_win": 4,
                        },
                        "ELITE": {
                            "status": "AT_RISK",
                            "estimated_survival": 0.80,
                            "expected_end_hp_on_win": 30,
                        },
                    },
                }

        base = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "M", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 6, "symbol": "E", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            base.owner_hint,
            base.scope_id,
            base.screen,
            facts={**base.facts, "act": 2, "floor": 24, "current_hp": 16},
        )
        llm = FakeLLM({"choice_id": 0, "reason": "ignore survival evidence"})

        decision = compose_map_agent(
            DefaultMapTool(llm, encounter_readiness=AtRisk())
        ).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "map.run_policy")
        self.assertEqual(llm.requests, [])

    def test_healed_readiness_can_allow_an_elite_after_a_rest(self):
        class HpAware:
            def evaluate(self, state, families):
                hp = int(state.facts["current_hp"])
                return {
                    "status": "AVAILABLE",
                    "entry_hp": hp,
                    "groups": {
                        "ELITE": {
                            "status": "SUPPORTED" if hp > 16 else "AT_RISK"
                        },
                        "WEAK_HALLWAY": {"status": "SUPPORTED"},
                        "STRONG_HALLWAY": {"status": "SUPPORTED"},
                    },
                }

        state = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "R", "children": [{"x": 0, "y": 7}]},
                {"x": 0, "y": 7, "symbol": "E", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 6, "symbol": "R", "children": [{"x": 1, "y": 7}]},
                {"x": 1, "y": 7, "symbol": "M", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "act": 2, "floor": 24, "current_hp": 16},
        )
        llm = FakeLLM({"choice_id": 0, "reason": "take elite"})
        agent = compose_map_agent(DefaultMapTool(llm, encounter_readiness=HpAware()))

        decision = agent.decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "map.llm")
        self.assertEqual(len(llm.requests), 1)

    def test_rest_projection_ignores_routes_with_prior_combat(self):
        class HpAware:
            def evaluate(self, state, families):
                hp = int(state.facts["current_hp"])
                return {
                    "status": "AVAILABLE",
                    "entry_hp": hp,
                    "groups": {
                        "ELITE": {"status": "SUPPORTED" if hp > 43 else "AT_RISK"},
                        "WEAK_HALLWAY": {"status": "SUPPORTED"},
                        "STRONG_HALLWAY": {"status": "SUPPORTED"},
                    },
                }

        state = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 3},
            nodes=[
                {"x": 0, "y": 4, "symbol": "M", "children": [{"x": 0, "y": 5}]},
                {"x": 0, "y": 5, "symbol": "R", "children": [{"x": 0, "y": 6}]},
                {"x": 0, "y": 6, "symbol": "E", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 4, "symbol": "$", "children": [{"x": 1, "y": 5}]},
                {"x": 1, "y": 5, "symbol": "R", "children": [{"x": 1, "y": 6}]},
                {"x": 1, "y": 6, "symbol": "E", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "act": 2, "current_hp": 43, "gold": 331},
        )
        llm = FakeLLM(
            {
                "choice_id": 0,
                "path": ["L04C0", "L05C0", "L06C0", "BOSS"],
                "reason": "incorrectly ignore combat damage before the rest",
            }
        )

        decision = compose_map_agent(
            DefaultMapTool(llm, encounter_readiness=HpAware())
        ).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "map.run_policy")
        self.assertEqual(llm.requests, [])
        self.assertEqual(
            decision.payload["run_route"]["rest_readiness"]["entry_hp"], 67
        )

    def test_map_prompt_values_gold_conversion_without_inventing_inventory(self):
        client = FakeLLM(
            {
                "choice_id": 1,
                "path": ["L00C2", "L01C1", "BOSS"],
                "reason": "preserve shop access",
            }
        )

        create_map_agent(client).decide(request(map_state()))

        prompt = client.requests[0].messages[-1].content
        self.assertIn("unspent gold as unrealized power", prompt)
        self.assertIn("at most one funded Shop per Act", prompt)
        self.assertIn("never reject an otherwise safer route", prompt)
        self.assertIn("future gold as uncertain", prompt)

    def test_zero_gold_route_evidence_exposes_two_forced_shops(self):
        nodes = [
            {"x": 0, "y": 0, "symbol": "M", "children": [{"x": 0, "y": 1}]},
            {"x": 0, "y": 1, "symbol": "?", "children": [{"x": 3, "y": 16}]},
            {"x": 1, "y": 0, "symbol": "M", "children": [{"x": 1, "y": 1}]},
            {"x": 1, "y": 1, "symbol": "$", "children": [{"x": 1, "y": 2}]},
            {"x": 1, "y": 2, "symbol": "M", "children": [{"x": 1, "y": 3}]},
            {"x": 1, "y": 3, "symbol": "$", "children": [{"x": 3, "y": 16}]},
            {
                "x": 2,
                "y": 0,
                "symbol": "M",
                "children": [{"x": 2, "y": 1}, {"x": 3, "y": 1}],
            },
            {"x": 2, "y": 1, "symbol": "M", "children": [{"x": 3, "y": 16}]},
            {"x": 3, "y": 1, "symbol": "?", "children": [{"x": 3, "y": 16}]},
        ]
        base = map_state(choices=("x=0", "x=1", "x=2"), nodes=nodes)
        state = GameState(
            base.owner_hint,
            base.scope_id,
            base.screen,
            facts={**base.facts, "gold": 0},
        )
        client = FakeLLM(
            {
                "choice_id": 0,
                "path": ["L00C0", "L01C0", "BOSS"],
                "reason": "avoid unfunded shops",
            }
        )

        _, options = render_map(state)
        create_map_agent(client).decide(request(state))

        self.assertEqual(
            [option["forced_shop_count"] for option in options],
            [0, 2, 0],
        )
        prompt = client.requests[0].messages[-1].content
        self.assertIn('"gold": 0', prompt)
        self.assertIn('"forced_shop_count": 2', prompt)

    def test_coffee_dripper_does_not_hide_an_at_risk_elite_after_a_rest(self):
        class AtRisk:
            def evaluate(self, state, families):
                return {
                    "status": "AVAILABLE",
                    "entry_hp": state.facts["current_hp"],
                    "groups": {
                        "ELITE": {"status": "AT_RISK"},
                        "WEAK_HALLWAY": {"status": "SUPPORTED"},
                        "STRONG_HALLWAY": {"status": "SUPPORTED"},
                    },
                }

        state = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "R", "children": [{"x": 0, "y": 7}]},
                {"x": 0, "y": 7, "symbol": "E", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 6, "symbol": "R", "children": [{"x": 1, "y": 7}]},
                {"x": 1, "y": 7, "symbol": "M", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={
                **state.facts,
                "act": 2,
                "current_hp": 16,
                "relics": ({"name": "Coffee Dripper"},),
            },
        )
        llm = FakeLLM({"choice_id": 0, "reason": "take elite"})

        decision = compose_map_agent(
            DefaultMapTool(llm, encounter_readiness=AtRisk())
        ).decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "map.run_policy")
        self.assertNotIn("rest_readiness", decision.payload["run_route"])
        self.assertEqual(llm.requests, [])

    def test_shop_does_not_hide_two_elites_before_the_next_rest(self):
        class Supported:
            def evaluate(self, state, families):
                return {
                    "status": "AVAILABLE",
                    "entry_hp": 60,
                    "groups": {"ELITE": {"status": "SUPPORTED"}},
                }

        state = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "E", "is_burning": True, "children": [{"x": 0, "y": 7}]},
                {"x": 0, "y": 7, "symbol": "$", "children": [{"x": 0, "y": 8}]},
                {"x": 0, "y": 8, "symbol": "E", "children": [{"x": 0, "y": 9}]},
                {"x": 0, "y": 9, "symbol": "R", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 6, "symbol": "M", "children": [{"x": 1, "y": 7}]},
                {"x": 1, "y": 7, "symbol": "$", "children": [{"x": 1, "y": 8}]},
                {"x": 1, "y": 8, "symbol": "E", "children": [{"x": 1, "y": 9}]},
                {"x": 1, "y": 9, "symbol": "R", "children": [{"x": 3, "y": 16}]},
            ],
        )
        llm = FakeLLM({"choice_id": 0, "reason": "take both elites"})
        agent = compose_map_agent(DefaultMapTool(llm, encounter_readiness=Supported()))

        decision = agent.decide(request(state))

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "map.run_policy")
        self.assertEqual(decision.payload["run_route"]["elites_before_rest"], 1)
        self.assertEqual(llm.requests, [])

    def test_rest_route_records_current_and_healed_survival(self):
        class HpAware:
            def __init__(self):
                self.calls = []

            def evaluate(self, state, families):
                hp = int(state.facts["current_hp"])
                self.calls.append((hp, families))
                return {
                    "status": "AVAILABLE",
                    "entry_hp": hp,
                    "groups": {"ELITE": {"status": "SUPPORTED" if hp > 16 else "AT_RISK"}},
                }

        state = map_state(
            choices=("x=0",),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "R", "children": [{"x": 0, "y": 7}]},
                {"x": 0, "y": 7, "symbol": "E", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "current_hp": 16},
        )
        readiness = HpAware()

        decision = compose_map_agent(
            DefaultMapTool(FakeLLM({}), encounter_readiness=readiness)
        ).decide(request(state))

        self.assertEqual(
            readiness.calls,
            [(16, ("ELITE",)), (40, ("ELITE",))],
        )
        self.assertEqual(decision.payload["run_route"]["rest_readiness"]["entry_hp"], 40)

    def test_boss_entrance_skips_coordinate_rendering_and_llm(self):
        client = FakeLLM({"choice_id": 0, "reason": "unused"})
        state = map_state(
            choices=("boss",),
            current_node={"x": 1, "y": 14},
        )

        decision = create_map_agent(client).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "map.single_choice")
        self.assertEqual(decision.payload["next_node"], "BOSS")
        self.assertEqual(client.requests, [])

    def test_forced_route_skips_llm(self):
        client = FakeLLM({"choice_id": 0, "reason": "unused"})
        state = map_state(choices=("x=0",))

        decision = create_map_agent(client).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "map.single_choice")
        self.assertEqual(client.requests, [])

    def test_forced_current_exit_still_plans_a_later_branch(self):
        state = forced_branch_state()
        client = FakeLLM(
            {
                "choice_id": 0,
                "path": ["L00C0", "L01C0", "BOSS"],
                "reason": "Plan the Rest branch",
            }
        )

        decision = create_map_agent(client).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(
            decision.payload["run_route"]["planned_rooms"],
            ("Monster", "Rest", "Boss"),
        )
        self.assertEqual(len(client.requests), 1)

    def test_empty_llm_output_cannot_block_a_forced_current_exit(self):
        class EmptyLLM:
            def complete(self, request):
                raise LLMOutputError("LLM response content is empty")

        shared = {
            "run_route": {
                "planned_path": ("L00C0", "L01C1", "BOSS"),
            }
        }
        decision = create_map_agent(EmptyLLM()).decide(
            request(forced_branch_state(), shared=shared)
        )

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "map.llm_fallback")
        self.assertEqual(
            decision.payload["run_route"]["planned_rooms"],
            ("Monster", "Shop", "Boss"),
        )

    def test_malformed_llm_data_cannot_block_a_forced_current_exit(self):
        for data in (
            [],
            {},
            {"choice_id": "0", "reason": "wrong type"},
            {"choice_id": 2, "reason": "illegal choice"},
            {"choice_id": 0, "reason": 1},
        ):
            with self.subTest(data=data):
                decision = create_map_agent(FakeLLM(data)).decide(
                    request(forced_branch_state())
                )

                self.assertEqual(decision.command, "choose 0")
                self.assertEqual(decision.source, "map.llm_fallback")

    def test_future_rests_count_only_the_selected_route(self):
        state = forced_branch_state()
        client = FakeLLM(
            {
                "choice_id": 0,
                "path": ["L00C0", "L01C1", "BOSS"],
                "reason": "take the selected Shop branch",
            }
        )

        decision = create_map_agent(client).decide(request(state))

        route = decision.payload["run_route"]
        self.assertEqual(route["planned_rooms"], ("Monster", "Shop", "Boss"))
        self.assertEqual(route["future_rests"], 0)

    def test_multi_choice_calls_llm_and_maps_id_to_command(self):
        client = FakeLLM(
            {
                "choice_id": 1,
                "path": ["L00C2", "L01C1", "BOSS"],
                "reason": "Preserve the more flexible route",
            }
        )
        current = request(
            map_state(),
            shared={
                "current_plan": {"goal": "Find AoE and prepare for an Elite"},
                RUN_CONSTRUCTION_KEY: {
                    "targets": ["The Guardian"],
                    "deficits": [{"type": "SCALING_DAMAGE"}],
                },
            },
        )

        decision = create_map_agent(client).decide(current)

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "map.llm")
        self.assertEqual(decision.payload["next_node"], "L00C2")
        self.assertEqual(
            decision.payload["run_route"]["planned_rooms"],
            ("Event", "Burning Elite", "Boss"),
        )
        self.assertEqual(decision.metrics["model"], "fake-map-model")
        self.assertEqual(len(client.requests), 1)
        llm_request = client.requests[0]
        self.assertEqual(llm_request.purpose, "map.choose_exit")
        prompt = llm_request.messages[1].content
        self.assertIn("MAP_GRAPH v1", prompt)
        self.assertIn("allowed_choice_ids: [0, 1]", prompt)
        self.assertIn("Find AoE and prepare for an Elite", prompt)
        self.assertIn("2x Strike", prompt)
        self.assertIn("DETERMINISTIC ROUTE EVIDENCE", prompt)
        self.assertIn('"run_construction"', prompt)
        self.assertIn('"targets": [\n      "The Guardian"', prompt)
        self.assertIn("At the end of combat, heal 6 HP", prompt)
        self.assertNotIn("# SHARED STRATEGY", prompt)
        self.assertNotIn('"children"', prompt)

    def test_map_prompt_exposes_compact_survival_evidence(self):
        class Supported:
            def evaluate(self, state, families):
                return {
                    "status": "AVAILABLE",
                    "entry_hp": 55,
                    "groups": {
                        "WEAK_HALLWAY": {
                            "status": "SUPPORTED",
                            "estimated_survival": 0.91,
                            "expected_end_hp_on_win": 37,
                        },
                        "STRONG_HALLWAY": {"status": "SUPPORTED"},
                        "ELITE": {
                            "status": "SUPPORTED",
                            "estimated_survival": 0.82,
                            "expected_end_hp_on_win": 28,
                        },
                    },
                }

        state = map_state(
            choices=("x=0", "x=1"),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "M", "children": [{"x": 3, "y": 16}]},
                {"x": 1, "y": 6, "symbol": "E", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "act": 2, "floor": 21, "current_hp": 55},
        )
        client = FakeLLM(
            {"choice_id": 0, "path": ["L06C0", "BOSS"], "reason": "safer"}
        )

        compose_map_agent(
            DefaultMapTool(client, encounter_readiness=Supported())
        ).decide(request(state))

        prompt = client.requests[0].messages[-1].content
        self.assertIn('"estimated_survival": 0.91', prompt)
        self.assertIn('"expected_end_hp_on_win": 28', prompt)
        self.assertIn('"options": [', prompt)

    def test_map_prompt_exposes_coffee_dripper_campfire_constraint(self):
        client = FakeLLM(
            {
                "choice_id": 1,
                "path": ["L00C2", "L01C1", "BOSS"],
                "reason": "avoid combat",
            }
        )
        state = map_state()
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "relics": ({"name": "Coffee Dripper"},)},
        )

        create_map_agent(client).decide(request(state))

        prompt = client.requests[0].messages[-1].content
        self.assertIn('"campfire_rest_healing_available": false', prompt)
        self.assertIn('"campfire_smithing_available": true', prompt)
        self.assertIn(
            '"campfire_rest_healing_blocked_by": "Coffee Dripper"', prompt
        )
        self.assertIn('"rest_site_healing_sources": []', prompt)
        self.assertIn("Coffee Dripper blocks Rest", prompt)

    def test_map_prompt_exposes_fusion_hammer_campfire_constraint(self):
        client = FakeLLM(
            {
                "choice_id": 1,
                "path": ["L00C2", "L01C1", "BOSS"],
                "reason": "avoid combat",
            }
        )
        state = map_state()
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={**state.facts, "relics": ({"name": "Fusion Hammer"},)},
        )

        create_map_agent(client).decide(request(state))

        prompt = client.requests[0].messages[-1].content
        self.assertIn('"campfire_rest_healing_available": true', prompt)
        self.assertIn('"campfire_smithing_available": false', prompt)
        self.assertIn('"campfire_smithing_blocked_by": "Fusion Hammer"', prompt)
        self.assertIn("Fusion Hammer makes R useful only", prompt)

    def test_eternal_feather_still_projects_healing_with_coffee_dripper(self):
        class HpAware:
            def __init__(self):
                self.hp = []

            def evaluate(self, state, families):
                self.hp.append(state.facts["current_hp"])
                return {
                    "status": "AVAILABLE",
                    "entry_hp": state.facts["current_hp"],
                    "groups": {"ELITE": {"status": "SUPPORTED"}},
                }

        state = map_state(
            choices=("x=0",),
            current_node={"x": 3, "y": 5},
            nodes=[
                {"x": 0, "y": 6, "symbol": "R", "children": [{"x": 0, "y": 7}]},
                {"x": 0, "y": 7, "symbol": "E", "children": [{"x": 3, "y": 16}]},
            ],
        )
        state = GameState(
            state.owner_hint,
            state.scope_id,
            state.screen,
            facts={
                **state.facts,
                "current_hp": 16,
                "deck": tuple({"name": "Strike"} for _ in range(15)),
                "relics": (
                    {"name": "Coffee Dripper"},
                    {"name": "Eternal Feather"},
                ),
            },
        )
        readiness = HpAware()

        decision = compose_map_agent(
            DefaultMapTool(FakeLLM({}), encounter_readiness=readiness)
        ).decide(request(state))

        self.assertEqual(readiness.hp, [16, 25])
        self.assertEqual(
            decision.payload["run_route"]["rest_readiness"]["entry_hp"], 25
        )

    def test_prompt_language_is_explicit_and_defaults_to_english(self):
        english_client = FakeLLM(
            {
                "choice_id": 0,
                "path": ["L00C0", "L01C1", "BOSS"],
                "reason": "English",
            }
        )
        chinese_client = FakeLLM(
            {
                "choice_id": 0,
                "path": ["L00C0", "L01C1", "BOSS"],
                "reason": "Localized",
            }
        )

        create_map_agent(english_client).decide(request(map_state()))
        create_map_agent(
            chinese_client,
            prompt_language=PromptLanguage.CHINESE,
        ).decide(request(map_state()))

        english_messages = english_client.requests[0].messages
        chinese_messages = chinese_client.requests[0].messages
        self.assertTrue(english_messages[0].content.isascii())
        self.assertFalse(chinese_messages[0].content.isascii())
        self.assertNotEqual(english_messages, chinese_messages)
        self.assertIn(
            "不得仅因路线经过它们就否定",
            chinese_messages[1].content,
        )

    def test_unsupported_prompt_language_is_rejected_during_wiring(self):
        with self.assertRaisesRegex(ValueError, "unsupported prompt language"):
            create_map_agent(FakeLLM({}), prompt_language="invalid")

    def test_illegal_or_malformed_llm_choice_fails_closed(self):
        for response, message in (
            ({"choice_id": 2, "reason": "invented"}, "illegal map choice 2"),
            ({"choice_id": True, "reason": "bool"}, "must be an integer"),
            ({"choice_id": 0, "reason": ""}, "non-empty string"),
        ):
            with self.subTest(response=response):
                with self.assertRaisesRegex(MapDecisionError, message):
                    create_map_agent(FakeLLM(response)).decide(request(map_state()))

    def test_invalid_explanatory_path_does_not_reject_a_legal_choice(self):
        decision = create_map_agent(
            FakeLLM(
                {
                    "choice_id": 0,
                    "path": ["L00C0", "L00C2", "BOSS"],
                    "reason": "invented edge",
                }
            )
        ).decide(request(map_state()))

        self.assertEqual(decision.command, "choose 0")
        self.assertIn("contains non-edge", decision.metrics["route_error"])


if __name__ == "__main__":
    unittest.main()
