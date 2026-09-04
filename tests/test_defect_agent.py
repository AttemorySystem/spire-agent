from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.subagents.build import create_build_agent
from spire_agent.subagents.llm import LLMResponse
from spire_agent.tools.winning_path import (
    WinningPathCardPicker,
    create_card_picker,
)
from spire_agent.tools.winning_path.defect_data import (
    compile_defect_catalog,
    compile_defect_certificates,
)


class FakeLLM:
    def __init__(self, data):
        self.data = data
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            self.data,
            raw_text=json.dumps(self.data, separators=(",", ":")),
            model="fake-defect",
        )


def state(screen="CARD_REWARD", *, bowl=False):
    choices = ("Defragment", "Cold Snap", "Claw")
    return GameState(
        AgentKind.BUILD,
        "DEFECT:a1:f3:reward:build",
        ScreenState(
            screen,
            commands=("choose", "skip") if screen == "CARD_REWARD" else ("choose", "leave"),
            choices=choices,
            details={
                "cards": tuple(
                    {"name": name, "cost": cost}
                    for name, cost in zip(choices, (1, 1, 0))
                ),
                "singing_bowl": bowl,
            },
        ),
        facts={
            "class": "DEFECT",
            "ascension_level": 20,
            "act": 1,
            "floor": 3,
            "act_boss": "Hexaghost",
            "current_hp": 61,
            "max_hp": 75,
            "gold": 99,
            "room_type": "MonsterRoom",
            "deck": (
                {"name": "Strike_B"},
                {"name": "Defend_B"},
                {"name": "Zap"},
                {"name": "Dualcast"},
            ),
            "relics": ({"name": "Cracked Core"},),
            "potions": ({"name": "Potion Slot"},),
        },
    )


def request(current):
    return DecisionRequest(
        current,
        DecisionScope(AgentKind.BUILD, current.scope_id),
        None,
        {},
        ContextEntry(0, None, current, True),
    )


def response(choice_id=0):
    return {
        "action": "choose",
        "choice_id": choice_id,
        "targets": [],
        "reason": "best current Defect value",
    }


class DefectAgentTests(unittest.TestCase):
    def test_character_factory_never_reuses_ironclad_policy(self):
        self.assertEqual(create_card_picker("IRONCLAD").character, "IRONCLAD")
        self.assertEqual(create_card_picker("defect").character, "DEFECT")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_card_picker("WATCHER")

    def test_defect_makes_dominant_first_copy_deterministically(self):
        current = state()
        details = dict(current.screen.details)
        details["cards"] = ({"name": "Seek"}, {"name": "Claw"}, {"name": "Fusion"})
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Seek", "Claw", "Fusion"),
                details=details,
            ),
            facts=current.facts,
        )

        decision = create_build_agent(
            FakeLLM(response(2)), WinningPathCardPicker("DEFECT")
        ).decide(request(current))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "card_reward.policy")
        review = decision.payload["card_choice_review"]
        self.assertEqual(review["character"], "DEFECT")
        self.assertEqual(review["picker_id"], "defect.winning_path")
        self.assertEqual(len(review["fingerprints"]["catalog_sha256"]), 64)

    def test_defect_template_progress_uses_current_deck(self):
        current = state()
        facts = dict(current.facts)
        facts["deck"] = (*facts["deck"], {"name": "Defragment"}, {"name": "Cold Snap"})
        details = dict(current.screen.details)
        details["cards"] = (
            {"name": "Coolheaded"}, {"name": "Claw"}, {"name": "Fusion"}
        )
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Coolheaded", "Claw", "Fusion"),
                details=details,
            ),
            facts=facts,
        )

        review = WinningPathCardPicker("DEFECT").review(request(current))

        self.assertEqual(review["command"], "choose 0")
        self.assertEqual(review["policy"], "TEMPLATE_PROGRESS")
        self.assertEqual(
            review["candidates"][0]["template"]["route_id"], "focus_frost"
        )

    def test_early_frontload_need_uses_expert_pairwise_ranking(self):
        current = state()
        facts = {
            **current.facts,
            "act_boss": "The Guardian",
            "deck": (*current.facts["deck"], {"name": "Creative AI"}),
        }
        details = {
            **current.screen.details,
            "cards": (
                {"name": "Ball Lightning"},
                {"name": "Leap"},
                {"name": "Darkness"},
            ),
        }
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Ball Lightning", "Leap", "Darkness"),
                details=details,
            ),
            facts=facts,
        )

        review = WinningPathCardPicker("DEFECT").review(request(current))

        self.assertEqual(review["command"], "choose 0")
        self.assertEqual(review["policy"], "TRANSITION_NEED")
        self.assertEqual(
            review["candidates"][0]["transition"]["needs"],
            ["SINGLE_TARGET_FRONTLOAD"],
        )

    def test_defect_uses_expert_supported_artifact_focus_module(self):
        current = state()
        facts = {
            **current.facts,
            "deck": (*current.facts["deck"], {"name": "Biased Cognition"}),
        }
        details = {
            **current.screen.details,
            "cards": (
                {"name": "Core Surge"},
                {"name": "Claw"},
                {"name": "Fusion"},
            ),
        }
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Core Surge", "Claw", "Fusion"),
                details=details,
            ),
            facts=facts,
        )

        review = WinningPathCardPicker("DEFECT").review(request(current))

        self.assertEqual(review["command"], "choose 0")
        self.assertEqual(review["policy"], "TEMPLATE_PROGRESS")
        template = review["candidates"][0]["template"]
        self.assertEqual(template["route_id"], "artifact_focus")
        self.assertEqual(template["level"], "CORE_ACTIVATION")
        self.assertEqual(template["observed_level"], "CORE_ACTIVATION")

    def test_defect_keeps_cooccurrence_only_module_advisory(self):
        current = state()
        facts = {
            **current.facts,
            "deck": (*current.facts["deck"], {"name": "White Noise"}),
        }
        details = {
            **current.screen.details,
            "cards": (
                {"name": "Heatsinks"},
                {"name": "Claw"},
                {"name": "Fusion"},
            ),
        }
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Heatsinks", "Claw", "Fusion"),
                details=details,
            ),
            facts=facts,
        )

        review = WinningPathCardPicker("DEFECT").review(request(current))
        template = review["candidates"][0]["template"]

        self.assertEqual(template["route_id"], "generated_power_payoff")
        self.assertEqual(template["level"], "NONE")
        self.assertEqual(template["observed_level"], "CORE_ACTIVATION")

    def test_defect_does_not_reuse_first_copy_evidence(self):
        current = state()
        facts = {**current.facts, "deck": (*current.facts["deck"], {"name": "Seek"})}
        details = {
            **current.screen.details,
            "cards": ({"name": "Seek"}, {"name": "Claw"}, {"name": "Fusion"}),
        }
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Seek", "Claw", "Fusion"),
                details=details,
            ),
            facts=facts,
        )

        review = WinningPathCardPicker("DEFECT").review(request(current))

        self.assertEqual(review["command"], "skip")
        self.assertEqual(review["policy"], "NO_POSITIVE_EVIDENCE")
        self.assertEqual(review["candidates"][0]["template"]["level"], "NONE")

    def test_focus_frost_deck_accepts_a_second_focus_source(self):
        current = state()
        facts = {
            **current.facts,
            "act": 2,
            "floor": 22,
            "act_boss": "Champ",
            "deck": (
                *current.facts["deck"],
                {"name": "Defragment", "upgrades": 1},
                {"name": "Glacier"},
                {"name": "Cold Snap"},
            ),
        }
        details = {
            **current.screen.details,
            "cards": (
                {"name": "Defragment"},
                {"name": "Hello World"},
                {"name": "Beam Cell"},
            ),
        }
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Defragment", "Hello World", "Beam Cell"),
                details=details,
            ),
            facts=facts,
        )

        review = WinningPathCardPicker("DEFECT").review(request(current))

        self.assertEqual(review["command"], "choose 0")
        self.assertEqual(review["policy"], "TEMPLATE_PROGRESS")
        self.assertEqual(
            review["candidates"][0]["template"]["route_id"], "focus_depth"
        )

    def test_blizzard_deck_accepts_coolheaded_for_frost_and_cycle_depth(self):
        current = state()
        facts = {
            **current.facts,
            "act": 2,
            "floor": 22,
            "act_boss": "Collector",
            "deck": (
                *current.facts["deck"],
                {"name": "Blizzard", "upgrades": 1},
                {"name": "Fission", "upgrades": 1},
                {"name": "Compile Driver"},
                {"name": "Coolheaded", "upgrades": 1},
                {"name": "Glacier", "upgrades": 1},
                {"name": "Cold Snap"},
                {"name": "Chill"},
            ),
        }
        details = {
            **current.screen.details,
            "cards": (
                {"name": "Coolheaded"},
                {"name": "Compile Driver"},
                {"name": "Cold Snap"},
            ),
        }
        current = GameState(
            current.owner_hint,
            current.scope_id,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Coolheaded", "Compile Driver", "Cold Snap"),
                details=details,
            ),
            facts=facts,
        )

        review = WinningPathCardPicker("DEFECT").review(request(current))

        self.assertEqual(review["command"], "choose 0")
        self.assertEqual(review["policy"], "TEMPLATE_PROGRESS")
        template = review["candidates"][0]["template"]
        self.assertEqual(template["route_id"], "blizzard_frost_cycle")
        self.assertEqual(template["missing_card_reduction"], 2)
        for candidate in review["candidates"][1:]:
            self.assertEqual(candidate["template"]["level"], "NONE")
            self.assertEqual(
                candidate["template"]["observed_level"], "COMMITTED_PROGRESS"
            )

        current = GameState(
            current.owner_hint,
            current.scope_id,
            current.screen,
            facts={**facts, "deck": (*facts["deck"], {"name": "Coolheaded"})},
        )
        review = WinningPathCardPicker("DEFECT").review(request(current))
        self.assertEqual(review["command"], "skip")

    def test_defect_catalog_compiler_filters_and_normalizes_expert_runs(self):
        run = {
            "character_chosen": "DEFECT",
            "ascension_level": 20,
            "victory": True,
            "floor_reached": 57,
            "damage_taken": [{"floor": 1, "enemies": "Jaw Worm"}],
            "card_choices": [
                {"floor": 1, "picked": "Gash", "not_picked": ["Steam"]}
            ],
            "master_deck": ["Cold Snap", "Glacier"],
            "relics": ["Data Disk"],
        }
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "runs.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("data/runs/0/1.run", json.dumps(run))
            catalog = compile_defect_catalog(
                archive,
                root / "src/spire_agent/tools/winning_path/data/defect_policy.json",
                root / "src/spire_agent/tools/data/cards.csv",
            )
            certificates = compile_defect_certificates(
                archive,
                root / "src/spire_agent/tools/winning_path/data/defect_policy.json",
                root / "src/spire_agent/tools/data/cards.csv",
            )

        self.assertEqual(catalog["model"]["expert_runs"], 1)
        self.assertEqual(catalog["model"]["winning_runs"], 1)
        self.assertEqual(catalog["model"]["offer_rows"], 1)
        self.assertEqual(catalog["provenance"]["template_support"]["focus_frost"], 1)
        names = {
            name
            for row in catalog["derived"]["expert_preferences"]["pairs"]
            for name in row[1:3]
        }
        self.assertIn("Claw", names)
        self.assertIn("Steam Barrier", names)
        self.assertEqual(
            certificates["summary"],
            {
                "certificate_count": 1,
                "signature_count": 1,
                "module_count": 17,
                "certificates_with_modules": 1,
                "certificates_without_modules": 0,
                "active_module_count_distribution": {"1": 1},
                "module_support": {"focus_frost": 1},
            },
        )
        self.assertEqual(
            certificates["certificates"][0]["active_modules"], ["focus_frost"]
        )
        self.assertEqual(certificates["certificates"][0]["relics"], ["Data Disk"])
        self.assertFalse(certificates["scope"]["runtime_authority"])


if __name__ == "__main__":
    unittest.main()
