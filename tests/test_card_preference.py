from __future__ import annotations

import unittest

from spire_agent.tools.winning_path.preference import SKIP, PreferenceBuilder, PreferenceTable


class CardPreferenceTests(unittest.TestCase):
    def test_pick_beats_other_cards_and_skip(self):
        builder = PreferenceBuilder()
        builder.observe(1, ["Anger", "Clash", "Cleave"], picked="Cleave")

        result = PreferenceTable(builder.payload()).decide(
            1, ["Clash", "Cleave", "Anger", SKIP]
        )

        self.assertEqual(result["winner"], "Cleave")
        self.assertEqual(result["observed_pairs"], 3)

    def test_skip_beats_every_offered_card(self):
        builder = PreferenceBuilder()
        builder.observe(2, ["Anger", "Clash"], skipped=True)

        result = PreferenceTable(builder.payload()).decide(
            2, ["Anger", "Clash", SKIP]
        )

        self.assertEqual(result["winner"], SKIP)

    def test_evidence_from_another_act_is_not_reused(self):
        builder = PreferenceBuilder()
        builder.observe(1, ["Anger", "Clash"], picked="Anger")

        result = PreferenceTable(builder.payload()).decide(3, ["Anger", "Clash"])

        self.assertIsNone(result["winner"])
        self.assertEqual(result["status"], "NO_EVIDENCE")
        self.assertIsNone(result["comparisons"][0]["bucket"])

    def test_margin_against_other_cards_does_not_break_copeland_tie(self):
        builder = PreferenceBuilder()
        builder.observe(1, ["Alpha", "Beta"], picked="Alpha")
        builder.observe(1, ["Alpha", "Beta"], picked="Beta")
        for _ in range(2):
            builder.observe(1, ["Alpha"], picked="Alpha")
        for _ in range(3):
            builder.observe(1, ["Beta"], picked="Beta")

        result = PreferenceTable(builder.payload()).decide(
            1, ["Alpha", "Beta", SKIP]
        )

        self.assertIsNone(result["winner"])
        self.assertEqual(result["status"], "TIE")
        self.assertEqual(set(result["tied"]), {"Alpha", "Beta"})

    def test_equal_evidence_is_left_unresolved(self):
        builder = PreferenceBuilder()
        builder.observe(1, ["Anger", "Clash"], picked="Anger")
        builder.observe(1, ["Anger", "Clash"], picked="Clash")

        result = PreferenceTable(builder.payload()).decide(1, ["Anger", "Clash"])

        self.assertEqual(result["status"], "TIE")
        self.assertIsNone(result["winner"])

    def test_upgrades_share_the_same_semantic_preference(self):
        builder = PreferenceBuilder()
        builder.observe(1, ["Anger+1", "Clash"], picked="Anger+1")

        result = PreferenceTable(builder.payload()).decide(1, ["Anger", "Clash"])

        self.assertEqual(result["winner"], "Anger")

    def test_owned_copy_context_changes_the_marginal_preference(self):
        builder = PreferenceBuilder()
        for _ in range(5):
            builder.observe(1, ["Spot Weakness", "Carnage"], picked="Spot Weakness")
        for _ in range(3):
            builder.observe(
                1,
                ["Spot Weakness", "Carnage"],
                picked="Carnage",
                owned={"Spot Weakness": 1},
            )
        table = PreferenceTable(builder.payload())

        first_copy = table.decide(1, ["Spot Weakness", "Carnage"])
        duplicate = table.decide(
            1,
            ["Spot Weakness", "Carnage"],
            owned={"Spot Weakness": 1},
        )

        self.assertEqual(first_copy["winner"], "Spot Weakness")
        self.assertEqual(duplicate["winner"], "Carnage")
        self.assertEqual(duplicate["comparisons"][0]["bucket"], "1:01")

    def test_missing_owned_context_does_not_use_first_copy_counts(self):
        builder = PreferenceBuilder()
        for _ in range(5):
            builder.observe(1, ["Spot Weakness", "Carnage"], picked="Spot Weakness")

        result = PreferenceTable(builder.payload()).decide(
            1,
            ["Spot Weakness", "Carnage"],
            owned={"Spot Weakness": 1},
        )

        self.assertEqual(result["status"], "NO_EVIDENCE")
        self.assertIsNone(result["comparisons"][0]["bucket"])

    def test_deck_size_context_precedes_the_broader_owned_context(self):
        builder = PreferenceBuilder()
        builder.observe(
            1, ["Anger", "Clash"], picked="Anger", deck_size=12
        )
        builder.observe(
            1, ["Anger", "Clash"], picked="Clash", deck_size=25
        )
        table = PreferenceTable(builder.payload())

        early = table.decide(1, ["Anger", "Clash"], deck_size=12)
        late = table.decide(1, ["Anger", "Clash"], deck_size=25)

        self.assertEqual(early["winner"], "Anger")
        self.assertEqual(late["winner"], "Clash")
        self.assertEqual(early["comparisons"][0]["bucket"], "1:00:00-14")
        self.assertEqual(late["comparisons"][0]["bucket"], "1:00:25-29")

    def test_confidence_gate_rejects_a_one_vote_majority(self):
        builder = PreferenceBuilder()
        for _ in range(44):
            builder.observe(1, ["Flex"], picked="Flex", deck_size=12)
        for _ in range(43):
            builder.observe(1, ["Flex"], skipped=True, deck_size=12)

        result = PreferenceTable(builder.payload()).decide(
            1,
            ["Flex", SKIP],
            deck_size=12,
            require_confidence=True,
        )

        self.assertEqual(result["status"], "TIE")
        self.assertIsNone(result["winner"])
        self.assertFalse(result["comparisons"][0]["confident"])

    def test_confidence_gate_keeps_strong_evidence(self):
        builder = PreferenceBuilder()
        for _ in range(66):
            builder.observe(1, ["Hemokinesis"], picked="Hemokinesis")
        for _ in range(34):
            builder.observe(1, ["Hemokinesis"], skipped=True)

        result = PreferenceTable(builder.payload()).decide(
            1,
            ["Hemokinesis", SKIP],
            require_confidence=True,
        )

        self.assertEqual(result["winner"], "Hemokinesis")
        self.assertTrue(result["comparisons"][0]["confident"])


if __name__ == "__main__":
    unittest.main()
