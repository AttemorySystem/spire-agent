from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from spire_agent.utils.llm_log import LLMLogViewError, render_llm_log, view_path


def record(user: str, assistant: str) -> dict:
    return {
        "request": {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": user},
            ]
        },
        "response": {
            "raw_text": assistant,
            "data": {"choice_id": 0},
        },
    }


class LLMLogViewTests(unittest.TestCase):
    def test_renders_only_plain_conversation_sections(self):
        rendered = render_llm_log(record("User context", "Assistant output"))

        self.assertEqual(
            rendered,
            "========== SYSTEM ==========\n"
            "System prompt\n\n"
            "========== USER ==========\n"
            "User context\n\n"
            "========== ASSISTANT ==========\n"
            "Assistant output\n",
        )
        self.assertNotIn("choice_id", rendered)

    def test_directory_is_rendered_in_filename_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, user in (("000002-b.json", "second"), ("000001-a.json", "first")):
                (directory / name).write_text(
                    json.dumps(record(user, "answer")),
                    encoding="utf-8",
                )
            output = io.StringIO()

            count = view_path(directory, output=output)

            self.assertEqual(count, 2)
            self.assertLess(output.getvalue().index("first"), output.getvalue().index("second"))
            self.assertIn("000001-a.json", output.getvalue())

    def test_invalid_log_is_rejected(self):
        with self.assertRaisesRegex(LLMLogViewError, "no request object"):
            render_llm_log({})


if __name__ == "__main__":
    unittest.main()
