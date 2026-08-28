from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.utils.card_choices import compare


def row(choice_id: str, offered: list[str], picked: str) -> dict:
    return {
        "schema_version": 1,
        "choice_id": choice_id,
        "context": {
            "act_boss": "Hexaghost",
            "deck_before_counts": {
                "Ascender's Bane": 1,
                "Bash": 1,
                "Defend": 4,
                "Strike": 5,
            },
            "relics_before_floor_rewards": ["Burning Blood"],
        },
        "decision": {
            "act": 1,
            "floor": 1,
            "kind": "combat_card_reward",
            "offered": offered,
            "picked": picked,
            "skipped": False,
            "used_singing_bowl": False,
        },
    }


class CompareCardChoicesTests(unittest.TestCase):
    def test_owned_singing_bowl_is_in_the_offline_action_set(self):
        current = row(
            "1:f7:c0",
            ["Clash"],
            "Clash",
        )
        current["context"]["relics_before_floor_rewards"] = [
            "Burning Blood",
            "Singing Bowl",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "card_choices.jsonl"
            path.write_text(json.dumps(current) + "\n", "utf-8")

            output = compare(path)

        self.assertIn("[NO_POSITIVE_EVIDENCE]", output)
        self.assertIn("singing bowl", output)

    def test_lists_every_historical_and_current_choice(self):
        rows = [
            row("1:f1:c0", ["Inflame", "Corruption", "Clash"], "Corruption"),
            row("1:f2:c0", ["Carnage", "Armaments"], "Armaments"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "card_choices.jsonl"
            path.write_text(
                "".join(json.dumps(value) + "\n" for value in rows), "utf-8"
            )

            output = compare(path)

        self.assertIn("[1/2] 1:f1:c0", output)
        self.assertIn("Historical: choose 1 (Corruption)", output)
        self.assertIn(
            "Winning Path: choose 0 (Inflame)  [TEMPLATE_PROGRESS]",
            output,
        )
        self.assertIn("[2/2] 1:f2:c0", output)
        self.assertIn("Historical: choose 1 (Armaments)", output)
        self.assertIn(
            "Winning Path: choose 0 (Carnage)  [EXPERT_EXPERIENCE]",
            output,
        )
        self.assertIn("template=", output)
        self.assertIn("DIRECT_DIFFERENT: 2", output)

    def test_boss_reward_uses_the_next_act_targets(self):
        current = row("1:f16:c0", ["Reaper", "Impervious"], "Reaper")
        current["decision"]["kind"] = "boss_card_reward"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "card_choices.jsonl"
            path.write_text(json.dumps(current) + "\n", "utf-8")

            output = compare(path)

        target_line = next(line for line in output.splitlines() if "| Targets:" in line)
        self.assertNotIn("Hexaghost", target_line)


if __name__ == "__main__":
    unittest.main()
