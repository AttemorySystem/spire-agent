from __future__ import annotations

import unittest

from spire_agent.adapters.game_seed import SeedMode, SeedRequest


class SeedRequestTests(unittest.TestCase):
    def test_decimal_input_uses_gym_prng_seed(self):
        seed = SeedRequest.parse("00017")

        self.assertEqual(seed.mode, SeedMode.GYM_RANDOM)
        self.assertEqual(dict(seed.reset_kwargs), {"seed": 17})

    def test_alphanumeric_input_uses_exact_sts_seed(self):
        seed = SeedRequest.parse("abc123")

        self.assertEqual(seed.mode, SeedMode.STS_EXACT)
        self.assertEqual(
            {key: dict(value) for key, value in seed.reset_kwargs.items()},
            {"options": {"sts_seed": "ABC123"}},
        )

    def test_replay_can_force_an_all_digit_seed_to_be_exact(self):
        seed = SeedRequest.exact("12345")

        self.assertEqual(seed.mode, SeedMode.STS_EXACT)
        self.assertEqual(
            {key: dict(value) for key, value in seed.reset_kwargs.items()},
            {"options": {"sts_seed": "12345"}},
        )

    def test_empty_or_illegal_seed_is_rejected(self):
        for value in ("", "ABC/O", "HASO"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SeedRequest.parse(value)


if __name__ == "__main__":
    unittest.main()
