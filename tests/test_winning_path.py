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
from spire_agent.tools.run_keys import RUN_KEYS_KEY, RUN_ROUTE_KEY
from spire_agent.tools.winning_path.analysis import analyze
from spire_agent.tools.winning_path.catalog import (
    CatalogError,
    compile_catalog,
    load_default_catalog,
)
from spire_agent.tools.winning_path.contracts import DecisionState
from spire_agent.tools.winning_path.plan import analyze_deck_plan
from spire_agent.tools.winning_path.parameters import load_parameters, load_policy
from spire_agent.tools.winning_path.protocol import load_protocol
from spire_agent.tools.winning_path.state import project_state


def _request() -> DecisionRequest:
    state = GameState(
        AgentKind.BUILD,
        "seed:a2:f20:reward:build",
        ScreenState(
            "CARD_REWARD",
            commands=("choose", "skip"),
            choices=("lowercase fallback",),
            details={
                "cards": (
                    {"name": "Entrench", "upgrades": 0},
                    {"name": "Shrug It Off", "upgrades": 0},
                ),
                "singing_bowl": True,
            },
        ),
        facts={
            "seed": 2330393804099040876,
            "class": "IRONCLAD",
            "act": 2,
            "floor": 20,
            "ascension_level": 20,
            "act_boss": "The Champ",
            "current_hp": 47,
            "max_hp": 80,
            "gold": 123,
            "room_type": "MonsterRoom",
            "deck": (
                {"name": "Bash", "upgrades": 1, "misc": 7},
                {"name": "Searing Blow+2"},
                {"name": "Barricade", "upgrades": 0},
            ),
            "relics": ({"name": "Burning Blood", "counter": -1},),
            "potions": (
                {"name": "Fire Potion", "can_use": True},
                {"name": "Potion Slot"},
            ),
        },
    )
    scope = DecisionScope(AgentKind.BUILD, state.scope_id)
    return DecisionRequest(
        state,
        scope,
        None,
        {
            RUN_KEYS_KEY: {"ruby": True, "emerald": False, "sapphire": False},
            RUN_ROUTE_KEY: {
                "future_rests": 1,
                "burning_elite_reachable": True,
                "planned_rooms": ("M", "E", "R"),
                "unreviewed_debug_field": "must not leak",
            },
        },
        ContextEntry(0, None, state, True, scope=scope),
    )


def _module(
    module_id: str,
    slots: list[dict],
    *,
    anchors: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    blocked_by: tuple[dict, ...] = (),
    dynamic: bool = False,
) -> dict:
    return {
        "module_id": module_id,
        "name": module_id,
        "aspect": "ENGINE",
        "phase": "BUILDING",
        "candidate_policy": "COMPATIBLE_ONLY",
        "activation": {
            "slots": slots,
            "anchor_slots": list(anchors),
            "requires_capabilities": list(requires),
        },
        "provides": [
            {"aspect": "ENGINE", "capability": capability}
            for capability in provides
        ],
        "blocked_by": list(blocked_by),
        "dynamic_verification": dynamic,
        "energy": {"mode": "NORMAL", "sources": [], "constraints": []},
        "evidence_templates": [],
        "exit_conditions": [{"type": "CAPABILITY_LOST", "capability": "X"}],
        "goals": [{"id": "protect", "type": "PRESERVE_CARD_POOL"}],
        "hard_resource_constraints": {"physical_deck": "MINIMIZE"},
        "mechanism": "test fixture",
        "soft_resource_pressures": [
            {"resource": "card_access", "reason": "find the core"}
        ],
    }


def _card_slot(slot_id: str, name: str) -> dict:
    return {
        "id": slot_id,
        "required": True,
        "all": [{"kind": "CARD", "name": name}],
    }


class WinningPathProtocolTests(unittest.TestCase):
    def test_protocol_points_to_the_single_policy_file(self):
        protocol = load_protocol()
        self.assertEqual(
            protocol["policy_file"], "data/ironclad_policy.json"
        )
        self.assertEqual(
            set(load_policy()["parameters"]),
            {"templates", "transition", "expert", "authority"},
        )
        self.assertEqual(
            set(load_parameters()) & {
                "templates", "transition", "expert", "authority"
            },
            {"templates", "transition", "expert", "authority"},
        )

class WinningPathStateAndPlanTests(unittest.TestCase):
    def test_state_projection_preserves_semantic_assets_and_route(self):
        state = project_state(_request())

        self.assertEqual(state.deck["cards"][0]["misc"], 7)
        self.assertEqual(state.deck["cards"][1]["id"], "Searing Blow")
        self.assertEqual(state.deck["cards"][1]["upgrades"], 2)
        self.assertEqual(state.deck["upgrade_counts"], {"Bash": 1, "Searing Blow": 1})
        self.assertEqual(state.deck["max_upgrades"]["Searing Blow"], 2)
        self.assertEqual(state.assets["relics"][0]["counter"], -1)
        self.assertEqual(tuple(row["id"] for row in state.assets["potions"]), ("Fire Potion",))
        self.assertEqual(
            tuple(row["id"] for row in state.assets["potion_slots"]),
            ("Fire Potion", "Potion Slot"),
        )
        self.assertEqual(state.run["seed"], 2330393804099040876)
        self.assertEqual(state.run["keys"]["ruby"], True)
        self.assertEqual(state.route["planned_rooms"], ("M", "E", "R"))
        self.assertNotIn("unreviewed_debug_field", state.route)
        self.assertEqual(state.reward["offered"], ("Entrench", "Shrug It Off"))
        self.assertEqual(state.reward["offered_cards"][0]["upgrades"], 0)
        self.assertTrue(state.reward["singing_bowl"])
        json.dumps(state.as_dict())

    def test_plan_resolves_fixed_point_commitment_and_blockers(self):
        state = DecisionState(
            run={},
            deck={
                "cards": [],
                "counts": {
                    "Barricade": 1,
                    "Bash": 1,
                    "Inflame": 1,
                    "Demon Form": 1,
                },
                "upgrade_counts": {},
                "physical_size": 4,
            },
            assets={"relics": [{"id": "Burning Blood", "counter": -1}], "potions": []},
            route={},
            reward={"kind": "combat_card_reward", "offered": [], "singing_bowl": False},
        )
        modules = [
            _module(
                "a_consumer",
                [_card_slot("shell", "Bash")],
                requires=("CAP_A",),
                provides=("CAP_B",),
            ),
            _module(
                "b_provider",
                [_card_slot("core", "Bash")],
                provides=("CAP_A",),
            ),
            _module(
                "c_committed",
                [
                    _card_slot("core", "Barricade"),
                    {
                        "id": "density",
                        "required": True,
                        "group": {
                            "cards": ["Flame Barrier", "Impervious"],
                            "minimum_distinct": 2,
                            "minimum_total_copies": 2,
                        },
                    },
                ],
                anchors=("core",),
                dynamic=True,
            ),
            _module(
                "d_blocked",
                [_card_slot("core", "Inflame")],
                blocked_by=({"kind": "CARD", "name": "Demon Form"},),
            ),
        ]
        catalog = {
            "knowledge": {
                "modules": modules,
                "support": {"cards": [{"card": "Bash", "provides": ["VULNERABLE"]}]},
            }
        }

        plan = analyze_deck_plan(state, catalog)

        self.assertEqual(plan.active_modules, ("a_consumer", "b_provider"))
        self.assertEqual(plan.committed_modules, ("c_committed",))
        self.assertEqual(plan.blocked_modules, ("d_blocked",))
        self.assertEqual(plan.capabilities, ("CAP_A", "CAP_B", "VULNERABLE"))
        self.assertEqual(plan.dynamic_verification_required, ("c_committed",))
        self.assertIn(
            "c_committed",
            {row["declaring_module_id"] for row in plan.goals},
        )
        self.assertTrue(
            any(row["resource"] == "card_access" for row in plan.resource_pressures)
        )

    def test_analysis_artifact_is_complete_and_serializable(self):
        catalog = {
            "schema_version": 1,
            "knowledge": {"modules": [], "support": {"cards": []}},
            "derived": {
                "expert_preferences": {
                    "schema_version": 1,
                    "rows": 0,
                    "observations": 0,
                    "pairs": [],
                }
            },
        }

        result = analyze(_request(), catalog)

        self.assertEqual(result["mode"], "LIVE_POLICY")
        json.dumps(result)
        self.assertEqual(len(result["fingerprints"]["state_sha256"]), 64)
        self.assertEqual(
            result["target_plan"]["targets"],
            ["Gremlin Leader", "Slavers", "Book Of Stabbing"],
        )
        self.assertTrue(result["target_plan"]["candidate_independent"])
        self.assertIn("encounter_model_sha256", result["fingerprints"])
        self.assertEqual(len(result["fingerprints"]["implementation_sha256"]), 64)


class WinningPathCatalogTests(unittest.TestCase):
    def test_shipped_catalog_contains_lossless_reviewed_knowledge(self):
        catalog = load_default_catalog()
        modules = {
            row["module_id"]: row for row in catalog["knowledge"]["modules"]
        }

        barricade = modules["barricade_block_storage"]
        self.assertEqual(catalog["model"]["module_count"], 26)
        self.assertEqual(catalog["model"]["signature_count"], 246)
        self.assertEqual(barricade["activation"]["anchor_slots"], ("core",))
        self.assertTrue(barricade["goals"])
        self.assertTrue(barricade["exit_conditions"])
        self.assertTrue(barricade["soft_resource_pressures"])
        self.assertTrue(catalog["knowledge"]["card_policies"][0]["reason"])
        self.assertEqual(
            catalog["knowledge"]["resource_rules"]["skill_lifecycle"][
                "hard_conflicts"
            ][0],
            ("REUSE_SKILLS", "EXHAUST_SKILLS"),
        )
        self.assertEqual(
            catalog["provenance"]["graph_summary"]["reviewed_modules"], 26
        )

    def test_catalog_compiler_validates_but_does_not_duplicate_templates(self):
        module = _module(
            "fixture_module",
            [_card_slot("core", "Barricade")],
            anchors=("core",),
            provides=("BLOCK_STORAGE",),
            dynamic=True,
        )
        module["entry"] = {
            "mode": "EARLY",
            "anchor": "core",
            "late_after_floor": 20,
            "late_entry_behavior": "VERIFY",
            "late_owned_minimum_upgrade": 0,
            "late_requires_active": [],
        }
        graph = {
            "module_catalog": {"fixture_module": module},
            "card_policies": [],
            "certificates": [
                {
                    "certificate_id": "fixture_certificate",
                    "active_modules": [{"module_id": "fixture_module"}],
                }
            ],
        }
        support = {
            "interpretation": "fixture",
            "capabilities": {},
            "density_capabilities": [],
            "verification_support": {},
            "pressure_relief": {},
            "module_aspect_satisfies": {},
            "act_bridges": {},
            "cards": [],
        }
        choice = {
            "decision": {
                "act": 1,
                "floor": 1,
                "choice_index": 0,
                "offered": ["Barricade"],
                "picked": "Barricade",
            },
            "context": {"deck_before_counts": {}},
            "run": {"run_id": "fixture", "heart_kill": True, "ascension": 20},
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            graph_path = directory / "graph.json"
            support_path = directory / "support.json"
            choices_path = directory / "choices.jsonl"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            support_path.write_text(json.dumps(support), encoding="utf-8")
            choices_path.write_text(json.dumps(choice) + "\n", encoding="utf-8")

            catalog = compile_catalog(graph_path, choices_path, support_path)

        self.assertNotIn("knowledge", catalog)
        self.assertEqual(catalog["model"]["module_count"], 1)
        self.assertIn("protocol_sha256", catalog["source"])

    def test_compiler_rejects_new_module_semantics_without_protocol_review(self):
        module = _module("fixture_module", [_card_slot("core", "Barricade")])
        module["secret_score"] = 42
        graph = {
            "module_catalog": {"fixture_module": module},
            "card_policies": [],
            "certificates": [],
        }
        support = {
            "interpretation": "fixture",
            "capabilities": {},
            "density_capabilities": [],
            "verification_support": {},
            "pressure_relief": {},
            "module_aspect_satisfies": {},
            "act_bridges": {},
            "cards": [],
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            graph_path = directory / "graph.json"
            support_path = directory / "support.json"
            choices_path = directory / "choices.jsonl"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            support_path.write_text(json.dumps(support), encoding="utf-8")
            choices_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(CatalogError, "secret_score"):
                compile_catalog(graph_path, choices_path, support_path)


if __name__ == "__main__":
    unittest.main()
