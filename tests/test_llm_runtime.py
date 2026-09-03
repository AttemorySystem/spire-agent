from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from spire_agent.adapters.openai_llm import LLMSettings, OpenAICompatibleLLMClient
from spire_agent.extensions.llm_recording import (
    LLMAuditError,
    LLMCallRecorder,
    RecordingLLMClient,
)
from spire_agent.extensions.run_directory import (
    RunDirectory,
    RunDirectoryError,
)
from spire_agent.subagents.llm import (
    LLMMessage,
    LLMOutputError,
    LLMRequest,
    LLMResponse,
)


def llm_request() -> LLMRequest:
    return LLMRequest(
        purpose="map.choose_exit",
        messages=(
            LLMMessage("system", "System context"),
            LLMMessage("user", "Complete user context"),
        ),
        response_schema={"type": "object"},
    )


class FakeStructuredClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class SequenceStructuredClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)

    def complete(self, request):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RunDirectoryTests(unittest.TestCase):
    def test_bind_uses_canonical_seed_as_directory_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary) / "runs")
            directory.bind("abc123")

            self.assertEqual(directory.seed, "ABC123")
            self.assertEqual(directory.path.name, "ABC123")
            self.assertTrue(directory.path.is_dir())

    def test_existing_seed_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            RunDirectory(root).bind("ABC123")

            with self.assertRaisesRegex(
                RunDirectoryError,
                "already exists",
            ):
                RunDirectory(root).bind("ABC123")

    def test_existing_directory_requires_explicit_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            created = RunDirectory(root)
            created.bind("ABC123")

            reopened = RunDirectory.open(created.path)

            self.assertEqual(reopened.seed, "ABC123")
            self.assertEqual(reopened.path, created.path)


class LLMSettingsTests(unittest.TestCase):
    def test_loads_three_values_from_process_environment(self):
        environment = {
            "MODEL_URL": "https://example.test",
            "MODEL": "process-model",
            "API_KEY": "process-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = LLMSettings.from_env()

        self.assertEqual(settings.base_url, "https://example.test")
        self.assertEqual(settings.model, "process-model")
        self.assertEqual(settings.api_key, "process-secret")
        self.assertNotIn("process-secret", repr(settings))

    def test_missing_value_fails_at_construction(self):
        with patch.dict(os.environ, {"MODEL": "model"}, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "MODEL_URL, API_KEY",
            ):
                LLMSettings.from_env()

    def test_yaml_values_override_model_environment_without_storing_the_key(self):
        environment = {
            "MODEL_URL": "https://environment.test",
            "MODEL": "environment-model",
            "API_KEY": "process-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = LLMSettings.from_env(
                base_url="https://config.test",
                model="config-model",
            )

        self.assertEqual(settings.base_url, "https://config.test")
        self.assertEqual(settings.model, "config-model")
        self.assertEqual(settings.api_key, "process-secret")


class OpenAICompatibleClientTests(unittest.TestCase):
    @patch("spire_agent.adapters.openai_llm.OpenAI")
    def test_provider_retries_transient_transport_failures(self, provider):
        settings = LLMSettings("https://example.test", "model", "secret")

        OpenAICompatibleLLMClient(settings)

        provider.assert_called_once_with(
            api_key="secret",
            base_url="https://example.test",
            timeout=120.0,
            max_retries=2,
        )

    def test_returns_raw_and_decoded_json_without_hidden_retry_options(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"choice_id": 1, "reason": "route"}'
                    )
                )
            ],
            model="returned-model",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 12}),
        )

        class Completions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return response

        completions = Completions()
        provider = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        client = OpenAICompatibleLLMClient(
            LLMSettings("https://example.test", "configured-model", "secret"),
            client=provider,
        )

        result = client.complete(llm_request())

        self.assertEqual(result.data["choice_id"], 1)
        self.assertEqual(result.raw_text, response.choices[0].message.content)
        self.assertEqual(result.model, "returned-model")
        self.assertEqual(result.usage["total_tokens"], 12)
        self.assertEqual(
            completions.kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn("stream", completions.kwargs)

    def test_streaming_accumulates_the_same_response_and_emits_progress(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content="Checking routes. ",
                        )
                    )
                ],
                model="returned-model",
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content='{"choice_id":1,',
                        )
                    )
                ],
                model="returned-model",
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content='"reason":"route"}')
                    )
                ],
                model="returned-model",
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                model="returned-model",
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 12}),
            ),
        ]

        class Completions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return chunks

        completions = Completions()
        events = []
        client = OpenAICompatibleLLMClient(
            LLMSettings("https://example.test", "configured-model", "secret"),
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            ),
            stream_event=lambda event, value: events.append((event, value)),
        )

        result = client.complete(llm_request())

        self.assertEqual(result.data["choice_id"], 1)
        self.assertEqual(result.reasoning, "Checking routes. ")
        self.assertEqual(result.usage["total_tokens"], 12)
        self.assertTrue(completions.kwargs["stream"])
        self.assertEqual(
            events,
            [
                ("start", "map.choose_exit"),
                ("reasoning", "Checking routes. "),
                ("content", '{"choice_id":1,'),
                ("content", '"reason":"route"}'),
                ("done", ""),
            ],
        )

    def test_invalid_json_preserves_raw_output(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
            model="model",
            usage=None,
        )
        provider = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )
        client = OpenAICompatibleLLMClient(
            LLMSettings("https://example.test", "model", "secret"),
            client=provider,
        )

        with self.assertRaises(LLMOutputError) as raised:
            client.complete(llm_request())
        self.assertEqual(raised.exception.raw_text, "not json")


class LLMRecordingTests(unittest.TestCase):
    def test_structured_output_error_is_retried_and_each_call_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary) / "runs")
            directory.bind("ABC123")
            result = LLMResponse({"choice_id": 0}, raw_text='{"choice_id":0}')
            client = RecordingLLMClient(
                SequenceStructuredClient(LLMOutputError("empty"), result),
                LLMCallRecorder(directory),
            )

            self.assertIs(client.complete(llm_request()), result)
            records = [
                json.loads(path.read_text())
                for path in sorted((directory.path / "llm").iterdir())
            ]
            self.assertEqual([row["status"] for row in records], ["error", "success"])

    def test_success_records_complete_input_and_output_in_one_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary) / "runs")
            directory.bind("ABC123")
            inner = FakeStructuredClient(
                LLMResponse(
                    {"choice_id": 1, "reason": "route"},
                    raw_text='{"choice_id":1,"reason":"route"}',
                    model="test-model",
                    usage={"total_tokens": 12},
                    reasoning="Compared both routes.",
                )
            )
            client = RecordingLLMClient(inner, LLMCallRecorder(directory))

            result = client.complete(llm_request())

            self.assertEqual(result.data["choice_id"], 1)
            files = list((directory.path / "llm").iterdir())
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "000001-map.choose_exit.json")
            record = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "success")
            self.assertEqual(
                record["request"]["messages"][1]["content"],
                "Complete user context",
            )
            self.assertEqual(record["response"]["data"]["choice_id"], 1)
            self.assertEqual(
                record["response"]["raw_text"],
                result.raw_text,
            )
            self.assertEqual(
                record["response"]["reasoning"],
                "Compared both routes.",
            )

    def test_each_failed_retry_records_raw_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary) / "runs")
            directory.bind("ABC123")
            error = LLMOutputError("invalid JSON", raw_text="broken output")
            client = RecordingLLMClient(
                FakeStructuredClient(error=error),
                LLMCallRecorder(directory),
            )

            with self.assertRaises(LLMOutputError):
                client.complete(llm_request())

            files = list((directory.path / "llm").iterdir())
            self.assertEqual(len(files), 2)
            for path in files:
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "error")
                self.assertEqual(record["response"]["raw_text"], "broken output")
                self.assertEqual(record["error"]["type"], "LLMOutputError")

    def test_unbound_directory_prevents_the_model_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            inner = FakeStructuredClient(
                LLMResponse({}, raw_text="{}")
            )
            client = RecordingLLMClient(
                inner,
                LLMCallRecorder(RunDirectory(Path(temporary) / "runs")),
            )

            with self.assertRaises(LLMAuditError):
                client.complete(llm_request())
            self.assertEqual(inner.calls, 0)

    def test_configured_secret_is_redacted_from_every_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = RunDirectory(Path(temporary) / "runs")
            directory.bind("ABC123")
            secret = "never-write-this-secret"
            request = LLMRequest(
                purpose="test.secret",
                messages=(LLMMessage("user", f"accidental {secret}"),),
                response_schema={},
            )
            client = RecordingLLMClient(
                FakeStructuredClient(
                    LLMResponse(
                        {"message": secret},
                        raw_text=secret,
                    )
                ),
                LLMCallRecorder(directory, secrets=(secret,)),
            )

            client.complete(request)

            content = next((directory.path / "llm").iterdir()).read_text()
            self.assertNotIn(secret, content)
            self.assertIn("***REDACTED***", content)


if __name__ == "__main__":
    unittest.main()
