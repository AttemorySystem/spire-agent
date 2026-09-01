from __future__ import annotations

import unittest

from spire_agent.tools.winning_path.preference import PreferenceBuilder
from spire_agent.tools.winning_path.catalog import load_default_catalog
from spire_agent.tools.winning_path.contracts import DecisionState
from spire_agent.tools.winning_path.evidence import analyze_candidate_evidence
from spire_agent.tools.winning_path.resolver import resolve

from tests.test_winning_path import _card_slot, _module


def _state(
    offered: tuple[str, ...],
    *,
    deck: tuple[tuple[str, int], ...] = (),
    bowl: bool = False,
    must_pick: bool = False,
) -> DecisionState:
    counts: dict[str, int] = {}
    upgrade_counts: dict[str, int] = {}
    max_upgrades: dict[str, int] = {}
    cards = []
    for name, upgrades in deck:
        counts[name] = counts.get(name, 0) + 1
        if upgrades > 0:
            upgrade_counts[name] = upgrade_counts.get(name, 0) + 1
        max_upgrades[name] = max(max_upgrades.get(name, 0), upgrades)
        cards.append({"id": name, "upgrades": upgrades})
    return DecisionState(
        run={"character": "IRONCLAD", "act": 1, "floor": 5},
        deck={
            "cards": cards,
            "counts": counts,
            "upgrade_counts": upgrade_counts,
            "max_upgrades": max_upgrades,
            "physical_size": len(cards),
        },
        assets={"relics": [], "potions": []},
        route={},
        reward={
            "kind": "combat_card_reward",
            "offered": list(offered),
            "offered_cards": [
                {"id": name, "upgrades": 0} for name in offered
            ],
            "singing_bowl": bowl,
            "must_pick": must_pick,
        },
    )


def _catalog(
    modules: list[dict],
    *,
    preferences: dict | None = None,
    support_cards: list[dict] | None = None,
    card_policies: list[dict] | None = None,
    dominant_cards: list[dict] | None = None,
    bridges: list[dict] | None = None,
    resource_rules: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "protocol_version": "2.3.0",
        "knowledge": {
            "modules": modules,
            "routes": [],
            "card_policies": card_policies or [],
            "resource_rules": resource_rules or {},
            "forbidden_cards": [],
            "dominant_cards": dominant_cards or [],
            "conditional_cards": [],
            "bridges": bridges or [],
            "candidate_bridges": [],
            "support": {
                "cards": support_cards or [],
                "pressure_relief": {},
            },
        },
        "derived": {
            "expert_preferences": preferences or PreferenceBuilder().payload()
        },
    }


def _preferences(
    offered: tuple[str, ...],
    *,
    picked: str | None = None,
    skipped: bool = False,
    repeats: int = 12,
) -> dict:
    builder = PreferenceBuilder()
    for _ in range(repeats):
        builder.observe(1, offered, picked=picked, skipped=skipped, owned={})
    return builder.payload()


def _decisive_core(card: str, *, dynamic: bool = False) -> dict:
    module = _module(
        f"{card.casefold().replace(' ', '_')}_module",
        [_card_slot("core", card)],
        provides=("IMMEDIATE_BLOCK",),
        dynamic=dynamic,
    )
    module["aspect"] = "DEFENSE"
    module["provides"] = [
        {"aspect": "DEFENSE", "capability": "IMMEDIATE_BLOCK"}
    ]
    return module


class WinningPathEvidenceTests(unittest.TestCase):
    def test_candidate_has_exactly_three_positive_evidence_sources(self):
        state = _state(("Apotheosis",))
        catalog = _catalog(
            [],
            dominant_cards=[{
                "name": "Apotheosis", "acts": [1], "maximum_owned": 0,
            }],
        )

        row = analyze_candidate_evidence(state, catalog)["candidates"][0]

        self.assertEqual(row["template"]["level"], "CORE_ACTIVATION")
        self.assertEqual(
            set(row)
            & {
                "template",
                "transition",
                "expert",
                "construction",
                "resource",
                "plan_delta",
            },
            {"template", "transition", "expert"},
        )

    def test_candidate_that_introduces_resource_conflict_is_rejected(self):
        existing = _decisive_core("Existing Core")
        candidate = _decisive_core("Candidate Core")
        existing["hard_resource_constraints"] = {
            "physical_deck": "MINIMIZE"
        }
        candidate["hard_resource_constraints"] = {
            "physical_deck": "PRESERVE_STRIKE_DENSITY"
        }
        catalog = _catalog(
            [existing, candidate],
            resource_rules={
                "physical_deck": {
                    "hard_conflicts": [
                        ["MINIMIZE", "PRESERVE_STRIKE_DENSITY"]
                    ]
                }
            },
        )
        state = _state(
            ("Candidate Core",), deck=(("Existing Core", 0),)
        )

        row = analyze_candidate_evidence(state, catalog)["candidates"][0]

        self.assertTrue(row["rejected"])
        self.assertEqual(
            row["hard_constraints"][0]["type"], "HARD_RESOURCE_CONFLICT"
        )

    def test_expert_evidence_records_counts_and_score(self):
        preferences = _preferences(("Known Card",), picked="Known Card")
        catalog = _catalog([], preferences=preferences)
        state = _state(("Known Card",))

        expert = analyze_candidate_evidence(state, catalog)["candidates"][0]["expert"]

        self.assertEqual(expert["level"], "DIRECT")
        self.assertEqual((expert["wins"], expert["losses"]), (12, 0))
        self.assertGreater(expert["score"], 1.96)

    def test_rupture_requires_an_owned_self_damage_source(self):
        catalog = load_default_catalog()
        unsupported = _state(("Rupture",))
        supported = _state(("Rupture",), deck=(("Hemokinesis", 0),))

        unsupported_row = analyze_candidate_evidence(unsupported, catalog)["candidates"][0]
        supported_row = analyze_candidate_evidence(supported, catalog)["candidates"][0]

        self.assertEqual(
            unsupported_row["hard_constraints"][0]["type"],
            "MISSING_PREREQUISITE",
        )
        self.assertFalse(supported_row["rejected"])


class WinningPathResolverTests(unittest.TestCase):
    def test_strong_structure_and_take_gate_produce_pick(self):
        catalog = _catalog(
            [],
            dominant_cards=[{"name": "Core Card", "acts": [1], "maximum_owned": 0}],
        )
        state = _state(("Core Card",))
        evidence = analyze_candidate_evidence(state, catalog)

        result = resolve(state, catalog, evidence)

        self.assertEqual(result["outcome"], "PICK")
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 0})
        self.assertEqual(result["policy"], "TEMPLATE_PROGRESS")

    def test_structural_take_conflict_requires_advice(self):
        catalog = _catalog([], dominant_cards=[
            {"name": "Core A", "acts": [1], "maximum_owned": 0},
            {"name": "Core B", "acts": [1], "maximum_owned": 0},
        ])
        state = _state(("Core A", "Core B"))
        evidence = analyze_candidate_evidence(state, catalog)

        result = resolve(state, catalog, evidence)

        self.assertEqual(result["outcome"], "ADVICE_REQUIRED")
        self.assertIsNone(result["proposed_action"])
        self.assertFalse(result["allow_skip"])

    def test_positive_frontier_excludes_skip_and_singing_bowl(self):
        catalog = _catalog([], dominant_cards=[
            {"name": "Core A", "acts": [1], "maximum_owned": 0},
            {"name": "Core B", "acts": [1], "maximum_owned": 0},
        ])
        state = _state(("Core A", "Core B"), bowl=True)

        result = resolve(state, catalog, analyze_candidate_evidence(state, catalog))

        self.assertEqual(result["outcome"], "ADVICE_REQUIRED")
        self.assertEqual(result["allowed_choice_ids"], [0, 1])
        self.assertFalse(result["allow_skip"])
        self.assertIsNone(result["alternative"])

    def test_expert_direct_evidence_can_pick_an_unmodeled_card(self):
        catalog = _catalog(
            [],
            preferences=_preferences(("Unknown Card",), picked="Unknown Card"),
        )
        state = _state(("Unknown Card",))
        evidence = analyze_candidate_evidence(state, catalog)

        result = resolve(state, catalog, evidence)

        self.assertEqual(result["outcome"], "PICK")

    def test_transition_need_excludes_unrelated_direct_expert_alternative(self):
        offer = ("Flame Barrier", "Shockwave")
        catalog = _catalog(
            [], preferences=_preferences(offer, picked="Shockwave")
        )
        state = _state(offer)
        evidence = analyze_candidate_evidence(state, catalog)

        result = resolve(state, catalog, evidence)

        self.assertEqual(evidence["candidates"][0]["transition"]["level"], "CRITICAL_NEED")
        self.assertEqual(evidence["candidates"][1]["expert"]["level"], "DIRECT")
        self.assertEqual(result["frontier_choice_ids"], [0])
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 0})

    def test_transition_need_without_expert_alternative_stays_direct(self):
        state = _state(("Flame Barrier", "Unknown Card"))
        catalog = _catalog([])

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["frontier_choice_ids"], [0])
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 0})

    def test_unmodeled_candidate_without_positive_evidence_skips(self):
        catalog = _catalog(
            [], preferences=_preferences(("Unknown Card",), skipped=True)
        )
        state = _state(("Unknown Card",))
        evidence = analyze_candidate_evidence(state, catalog)

        result = resolve(state, catalog, evidence)

        self.assertEqual(result["outcome"], "SKIP")
        self.assertTrue(result["allow_skip"])
        self.assertEqual(result["allowed_choice_ids"], [])

    def test_confident_take_is_not_vetoed_by_ordinary_unknown(self):
        catalog = _catalog(
            [],
            preferences=_preferences(("Known Take",), picked="Known Take"),
            support_cards=[
                {"card": "Known Take", "provides": []},
                {"card": "Ordinary Unknown", "provides": []},
            ],
        )
        state = _state(("Known Take", "Ordinary Unknown"))

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["outcome"], "PICK")
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 0})

    def test_ordinary_unknowns_without_positive_evidence_skip(self):
        catalog = _catalog(
            [],
            support_cards=[
                {"card": "Ordinary A", "provides": []},
                {"card": "Ordinary B", "provides": []},
            ],
        )
        state = _state(("Ordinary A", "Ordinary B"))

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["outcome"], "SKIP")
        self.assertEqual(result["policy"], "NO_POSITIVE_EVIDENCE")

    def test_all_hard_rejected_uses_bowl_or_skip(self):
        policy = [
            {"card": "Forbidden", "policy": "FORBID", "reason": "fixture"}
        ]
        plain_catalog = _catalog([], card_policies=policy)
        plain = _state(("Forbidden",))
        bowl = _state(("Forbidden",), bowl=True)

        plain_result = resolve(
            plain,
            plain_catalog,
            analyze_candidate_evidence(plain, plain_catalog),
        )
        bowl_result = resolve(
            bowl,
            plain_catalog,
            analyze_candidate_evidence(bowl, plain_catalog),
        )

        self.assertEqual(plain_result["outcome"], "SKIP")
        self.assertEqual(bowl_result["outcome"], "SINGING_BOWL")
        self.assertEqual(bowl_result["proposed_action"]["choice_id"], 1)

    def test_committed_reward_can_choose_the_least_bad_rejected_card(self):
        catalog = _catalog([], card_policies=[
            {"card": "Forbidden A", "policy": "FORBID", "reason": "fixture"},
            {"card": "Forbidden B", "policy": "FORBID", "reason": "fixture"},
        ])
        state = _state(("Forbidden A", "Forbidden B"), must_pick=True)

        result = resolve(state, catalog, analyze_candidate_evidence(state, catalog))

        self.assertEqual(result["outcome"], "ADVICE_REQUIRED")
        self.assertEqual(result["allowed_choice_ids"], [0, 1])
        self.assertFalse(result["allow_skip"])
        self.assertEqual(result["policy"], "COMMITTED_REWARD_ALL_REJECTED")

    def test_worthy_card_is_preferred_to_singing_bowl(self):
        catalog = _catalog(
            [],
            preferences=_preferences(("Modeled Card",), picked="Modeled Card"),
            support_cards=[{"card": "Modeled Card", "provides": []}],
        )
        state = _state(("Modeled Card",), bowl=True)
        evidence = analyze_candidate_evidence(state, catalog)

        result = resolve(state, catalog, evidence)

        self.assertEqual(
            evidence["candidates"][0]["expert"]["level"], "DIRECT"
        )
        self.assertEqual(result["outcome"], "PICK")
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 0})

    def test_reviewed_dominant_card_precedes_other_evidence(self):
        catalog = _catalog(
            [],
            dominant_cards=[{
                "name": "Apotheosis",
                "acts": [1],
                "maximum_owned": 0,
                "reason": "reviewed dominant card",
            }],
        )
        state = _state(("Hand of Greed", "Apotheosis", "Secret Technique"))

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["policy"], "TEMPLATE_PROGRESS")
        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 1})

    def test_reachable_template_is_positive_but_not_direct(self):
        catalog = _catalog(
            [],
            support_cards=[{"card": "Anchor", "provides": []}],
        )
        state = _state(("Anchor",))
        evidence = analyze_candidate_evidence(state, catalog)
        evidence["candidates"][0]["template"] = {
            "level": "REACHABLE_ENTRY",
            "completed_core_gain": 0,
            "anchor_reduction": 1,
            "missing_card_reduction": 1,
            "completion_probability": 0.8,
        }

        result = resolve(state, catalog, evidence)

        self.assertEqual(result["outcome"], "ADVICE_REQUIRED")
        self.assertEqual(result["policy"], "POSITIVE_CONFLICT")

    def test_offer_permutation_changes_only_choice_id(self):
        builder = PreferenceBuilder()
        for _ in range(12):
            builder.observe(1, ["Card A", "Card B"], picked="Card A", owned={})
            builder.observe(1, ["Card B"], picked="Card B", owned={})
        catalog = _catalog(
            [],
            preferences=builder.payload(),
            support_cards=[
                {"card": "Card A", "provides": []},
                {"card": "Card B", "provides": []},
            ],
        )

        selected = []
        for offer in (("Card A", "Card B"), ("Card B", "Card A")):
            state = _state(offer)
            evidence = analyze_candidate_evidence(state, catalog)
            result = resolve(state, catalog, evidence)
            selected.append(offer[result["proposed_action"]["choice_id"]])

        self.assertEqual(selected, ["Card A", "Card A"])

    def test_pairwise_winner_does_not_need_comparisons_between_losers(self):
        builder = PreferenceBuilder()
        for _ in range(12):
            for card in ("Card A", "Card B", "Card C"):
                builder.observe(1, [card], picked=card, owned={})
            builder.observe(1, ["Card A", "Card B"], picked="Card A", owned={})
            builder.observe(1, ["Card A", "Card C"], picked="Card A", owned={})
        state = _state(("Card A", "Card B", "Card C"))
        catalog = _catalog([], preferences=builder.payload())

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["proposed_action"], {"kind": "PICK", "choice_id": 0})

    def test_partial_pairwise_lead_without_full_dominance_needs_advice(self):
        builder = PreferenceBuilder()
        for _ in range(12):
            for card in ("Card A", "Card B", "Card C"):
                builder.observe(1, [card], picked=card, owned={})
            builder.observe(1, ["Card A", "Card B"], picked="Card A", owned={})
        state = _state(("Card A", "Card B", "Card C"))
        catalog = _catalog([], preferences=builder.payload())

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["outcome"], "ADVICE_REQUIRED")

    def test_one_pairwise_observation_cannot_make_a_direct_winner(self):
        builder = PreferenceBuilder()
        for _ in range(12):
            builder.observe(1, ["Card A"], picked="Card A", owned={})
            builder.observe(1, ["Card B"], picked="Card B", owned={})
        builder.observe(1, ["Card A", "Card B"], picked="Card A", owned={})
        state = _state(("Card A", "Card B"))
        catalog = _catalog([], preferences=builder.payload())

        result = resolve(
            state, catalog, analyze_candidate_evidence(state, catalog)
        )

        self.assertEqual(result["outcome"], "ADVICE_REQUIRED")
        self.assertEqual(result["allowed_choice_ids"], [0, 1])

    def test_one_owned_card_observation_does_not_force_a_duplicate(self):
        builder = PreferenceBuilder()
        builder.observe(
            1,
            ["Demon Form"],
            picked="Demon Form",
            owned={"Demon Form": 1},
        )
        state = _state(("Demon Form",), deck=(("Demon Form", 0),))
        catalog = _catalog([], preferences=builder.payload())

        evidence = analyze_candidate_evidence(state, catalog)
        result = resolve(state, catalog, evidence)

        self.assertEqual(evidence["candidates"][0]["expert"]["level"], "NONE")
        self.assertEqual(result["outcome"], "SKIP")


if __name__ == "__main__":
    unittest.main()
