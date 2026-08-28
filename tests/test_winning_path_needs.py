from __future__ import annotations

import unittest

from spire_agent.tools.winning_path.catalog import load_default_catalog
from spire_agent.tools.winning_path.contracts import DecisionState
from spire_agent.tools.winning_path.evidence import analyze_candidate_evidence
from spire_agent.tools.winning_path.needs import (
    analyze_need_profile,
    candidate_need_coverage,
    load_encounter_model,
    plan_targets,
)
from spire_agent.tools.winning_path.resolver import resolve

from tests.test_winning_path_resolver import _catalog, _decisive_core


def _state(
    offered: tuple[str, ...] = ("Flame Barrier", "Cleave"),
    *,
    act: int = 1,
    boss: str = "The Guardian",
    room_type: str = "MonsterRoom",
    route: dict | None = None,
    character: str = "IRONCLAD",
    deck: tuple[str, ...] | None = None,
) -> DecisionState:
    names = deck or ("Strike",) * 5 + ("Defend",) * 4 + ("Bash",)
    cards = [{"id": name, "upgrades": 0} for name in names]
    counts = {name: names.count(name) for name in set(names)}
    return DecisionState(
        run={
            "character": character,
            "act": act,
            "floor": 5 if act == 1 else 20,
            "ascension": 20,
            "boss": boss,
            "room_type": room_type,
        },
        deck={
            "cards": cards,
            "counts": counts,
            "upgrade_counts": {},
            "max_upgrades": {},
            "physical_size": len(cards),
        },
        assets={"relics": []},
        route=route or {},
        reward={
            "kind": "combat_card_reward",
            "offered": list(offered),
            "offered_cards": [
                {"id": name, "upgrades": 0} for name in offered
            ],
            "singing_bowl": False,
        },
    )


class WinningPathNeedTests(unittest.TestCase):
    def test_encounter_model_covers_every_target(self):
        model = load_encounter_model()
        targets = {
            target
            for pool in model["target_pools"].values()
            for family in ("elites", "bosses")
            for target in pool[family]
        }
        requirements = {
            row["encounter"] for row in model["encounter_requirements"]
        }
        self.assertEqual(targets, requirements)

    def test_target_plan_is_candidate_independent(self):
        route = {"planned_rooms": ["M", "E", "R"]}
        left = plan_targets(_state(route=route))
        right = plan_targets(
            _state(("Demon Form", "Offering", "Clash"), route=route)
        )

        self.assertEqual(left.targets, right.targets)
        self.assertEqual(left.groups[0]["rule"], "PLANNED_ELITE_BEFORE_REST")
        self.assertEqual(
            left.targets, ("Gremlin Nob", "Lagavulin", "Three Sentries")
        )

    def test_burning_elite_is_not_mistaken_for_a_boss(self):
        plan = plan_targets(
            _state(route={"planned_rooms": ["Burning Elite", "Rest", "Boss"]})
        )

        self.assertEqual(plan.groups[0]["rule"], "PLANNED_ELITE_BEFORE_REST")

    def test_boss_rewards_look_forward(self):
        act_one = plan_targets(_state(room_type="MonsterRoomBoss"))
        act_three = plan_targets(
            _state(act=3, boss="Time Eater", room_type="MonsterRoomBoss")
        )

        self.assertEqual(act_one.groups[0]["rule"], "NEXT_ACT_BOSS_POOL")
        self.assertEqual(act_one.targets, ("Automaton", "Collector", "Champ"))
        self.assertEqual(act_three.groups[0]["rule"], "HEART_OBJECTIVE")
        self.assertEqual(act_three.targets, ("Shield And Spear", "The Heart"))

    def test_need_profile_and_candidate_coverage_are_discrete(self):
        state = _state(route={"planned_rooms": ["M", "Boss"]})
        catalog = _catalog([], support_cards=[])
        profile = analyze_need_profile(state, catalog)
        coverage = candidate_need_coverage(state, profile)

        self.assertIn("IMMEDIATE_BLOCK", profile.blocking_deficits)
        self.assertIn("SCALING_DAMAGE", profile.blocking_deficits)
        self.assertEqual(coverage[0], ("IMMEDIATE_BLOCK",))
        self.assertEqual(coverage[1], ())

    def test_blocking_need_has_first_authority(self):
        state = _state(
            ("Core Card", "Flame Barrier"),
            route={"planned_rooms": ["M", "Boss"]},
        )
        catalog = _catalog([_decisive_core("Core Card")])

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["frontier_choice_ids"], [1])
        self.assertEqual(result["outcome"], "PICK")

    def test_thunderclap_is_not_aoe_but_cleave_is(self):
        state = _state(
            ("Thunderclap", "Cleave"), boss="Slime Boss",
            deck=("Anger", "Bash") + ("Defend",) * 4 + ("Strike",) * 4,
        )

        evidence = analyze_candidate_evidence(state, load_default_catalog())

        self.assertEqual(evidence["candidates"][0]["transition"]["level"], "NONE")
        self.assertEqual(
            evidence["candidates"][1]["transition"]["level"], "CRITICAL_NEED"
        )

    def test_combust_covers_slime_boss_aoe(self):
        state = _state(
            ("Combust", "Heavy Blade"), boss="Slime Boss",
            deck=("Anger", "Bash") + ("Defend",) * 4 + ("Strike",) * 4,
        )

        evidence = analyze_candidate_evidence(state, load_default_catalog())

        self.assertEqual(
            evidence["candidates"][0]["transition"]["needs"], ["AOE"]
        )
        self.assertEqual(evidence["candidates"][1]["transition"]["level"], "NONE")

    def test_strength_payoffs_need_a_strength_source(self):
        catalog = load_default_catalog()
        unsupported = analyze_need_profile(
            _state(
                boss="Hexaghost",
                deck=("Pummel", "Sword Boomerang", "Bash"),
            ),
            catalog,
        )
        supported = analyze_need_profile(
            _state(
                boss="Hexaghost",
                deck=("Inflame", "Pummel", "Sword Boomerang", "Bash"),
            ),
            catalog,
        )

        def scaling(profile):
            return next(
                row for row in profile.needs if row["type"] == "SCALING_DAMAGE"
            )["status"]

        self.assertEqual(scaling(unsupported), "DEFICIT_WITHIN_MODEL")
        self.assertEqual(scaling(supported), "SATISFIED")

    def test_shockwave_expert_evidence_can_beat_minor_splash_damage(self):
        state = _state(
            ("Clothesline", "Thunderclap", "Shockwave"),
            boss="Slime Boss",
            deck=("Anger", "Bash", "Burning Pact", "Headbutt")
            + ("Defend",) * 4 + ("Strike",) * 4,
        )
        catalog = load_default_catalog()

        result = resolve(state, catalog, analyze_candidate_evidence(state, catalog))

        self.assertEqual(result["policy"], "EXPERT_EXPERIENCE")
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 2})

    def test_completed_body_slam_shell_does_not_overvalue_second_impervious(self):
        deck = (
            ("Strike",) * 4 + ("Defend",) * 4
            + (
                "Bash", "Body Slam", "Disarm", "Impervious", "Pummel",
                "Power Through", "Inflame", "Armaments",
            )
        )
        state = _state(
            ("Impervious", "Reaper", "Barricade"),
            boss="The Guardian",
            room_type="MonsterRoomBoss",
            deck=deck,
        )
        catalog = load_default_catalog()

        result = resolve(state, catalog, analyze_candidate_evidence(state, catalog))

        self.assertEqual(result["policy"], "TRANSITION_NEED")
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 1})

    def test_owned_rupture_without_self_damage_is_not_scaling(self):
        catalog = load_default_catalog()
        unsupported = analyze_need_profile(
            _state(boss="Hexaghost", deck=("Rupture", "Bash")), catalog
        )
        supported = analyze_need_profile(
            _state(
                boss="Hexaghost", deck=("Rupture", "Hemokinesis", "Bash")
            ),
            catalog,
        )

        unsupported_scaling = next(
            row for row in unsupported.needs if row["type"] == "SCALING_DAMAGE"
        )
        supported_scaling = next(
            row for row in supported.needs if row["type"] == "SCALING_DAMAGE"
        )
        self.assertEqual(unsupported_scaling["status"], "DEFICIT_WITHIN_MODEL")
        self.assertEqual(supported_scaling["status"], "SATISFIED")

    def test_foundation_draw_need_selects_early_coolheaded(self):
        starter = (
            ("Strike",) * 4 + ("Defend",) * 4
            + ("Zap", "Dualcast", "Ascender's Bane")
        )
        state = _state(
            ("Coolheaded", "Tempest"), character="DEFECT",
            boss="Hexaghost", deck=starter,
        )
        catalog = _catalog([])
        evidence = analyze_candidate_evidence(state, catalog)
        draw = next(
            row for row in evidence["need_profile"]["needs"]
            if row["type"] == "DRAW_CONSISTENCY"
        )

        self.assertEqual((draw["current_sources"], draw["required_sources"]), (0, 1))
        self.assertEqual(evidence["candidates"][0]["transition"]["level"], "CRITICAL_NEED")
        self.assertEqual(evidence["candidates"][1]["transition"]["level"], "NONE")
        self.assertEqual(
            resolve(state, catalog, evidence)["proposed_action"],
            {"kind": "PICK", "choice_id": 0},
        )

    def test_foundation_draw_requirement_scales_with_deck_size(self):
        base = ("Strike",) * 8 + ("Defend",) * 6 + ("Coolheaded",)
        catalog = _catalog([])
        one = analyze_need_profile(
            _state(character="DEFECT", boss="Hexaghost", deck=base), catalog
        )
        two = analyze_need_profile(
            _state(
                character="DEFECT", boss="Hexaghost",
                deck=base + ("Coolheaded",),
            ),
            catalog,
        )
        one_draw = next(row for row in one.needs if row["type"] == "DRAW_CONSISTENCY")
        two_draw = next(row for row in two.needs if row["type"] == "DRAW_CONSISTENCY")

        self.assertEqual((one_draw["current_sources"], one_draw["required_sources"]), (1, 2))
        self.assertEqual(one_draw["status"], "DEFICIT_WITHIN_MODEL")
        self.assertEqual(two_draw["status"], "SATISFIED")

    def test_completed_draw_engine_satisfies_foundation_density(self):
        deck = (
            ("Strike",) * 8 + ("Defend",) * 8
            + ("Recycle", "TURBO", "Hologram")
        )
        catalog = load_default_catalog("DEFECT")
        profile = analyze_need_profile(
            _state(character="DEFECT", boss="Hexaghost", deck=deck), catalog
        )
        draw = next(row for row in profile.needs if row["type"] == "DRAW_CONSISTENCY")

        self.assertEqual((draw["current_sources"], draw["required_sources"]), (1, 2))
        self.assertEqual(draw["completed_engine_capabilities"], ("DRAW_CONSISTENCY",))
        self.assertEqual(draw["status"], "SATISFIED")


if __name__ == "__main__":
    unittest.main()
