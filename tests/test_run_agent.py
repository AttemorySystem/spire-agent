from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import spire_agent.run_agent as run_agent_module

from spire_agent.contracts import (
    AgentKind,
    Continuation,
    ContextEntry,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.run_agent import (
    AgentConfigError,
    ROOT,
    load_runtime_config,
    parse_args,
    runtime_registry,
)
from spire_agent.tools.map import DefaultMapTool
from spire_agent.tools.mcts import DefaultCombatTool
from spire_agent.tools.mcts import MCTSResult
from spire_agent.subagents import PromptLanguage
from spire_agent.subagents.llm import LLMResponse


class FakeLLM:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if request.purpose == "build.llm":
            data = {"command": "choose 0", "reason": "test"}
        elif request.purpose == "combat.llm":
            data = {"command": "end", "reason": "test"}
        elif request.purpose.startswith("build."):
            data = {
                "action": "choose",
                "choice_id": 0,
                "targets": [],
                "reason": "test",
            }
        else:
            data = {"choice_id": 0, "reason": "test"}
        return LLMResponse(
            data,
            raw_text="{}",
        )


def request(
    kind: AgentKind,
    screen: ScreenState,
    continuation: Continuation | None = None,
) -> DecisionRequest:
    state = GameState(
        owner_hint=kind,
        scope_id=f"runtime-{kind.value}",
        screen=screen,
        facts={
            "floor": 1,
            "sts_seed": "ABC123",
            "deck": [{"name": "Strike"}],
            "relics": [{"name": "Burning Blood"}],
            "potions": [{"name": "Fire Potion"}],
        },
        combat={
            "hand": [{"name": "Strike", "is_playable": True}],
            "monsters": [{"name": "Jaw Worm"}],
        }
        if kind is AgentKind.COMBAT
        else None,
    )
    return DecisionRequest(
        state=state,
        scope=DecisionScope(kind, state.scope_id),
        continuation=continuation,
        shared={},
        previous=ContextEntry(0, None, state, True),
    )


class FakeCombatSearch:
    def __init__(self):
        self.states = []

    def choose(self, state):
        self.states.append(state)
        return MCTSResult(
            command="end",
            follow_up=None,
            metrics={"search_id": "test"},
        )


def registry(llm, search):
    return runtime_registry(
        llm,
        DefaultMapTool(llm),
        DefaultCombatTool(search),
    )


class RuntimeEntryTests(unittest.TestCase):
    def test_default_build_implementation_selects_the_defect_picker(self):
        current_registry = runtime_registry(
            FakeLLM(),
            DefaultMapTool(FakeLLM()),
            DefaultCombatTool(FakeCombatSearch()),
            character="DEFECT",
        )
        current = request(
            AgentKind.BUILD,
            ScreenState(
                "CARD_REWARD",
                commands=("choose", "skip"),
                choices=("Defragment", "Cold Snap", "Claw"),
                details={
                    "cards": (
                        {"name": "Defragment"},
                        {"name": "Cold Snap"},
                        {"name": "Claw"},
                    )
                },
            ),
        )
        current = DecisionRequest(
            GameState(
                current.state.owner_hint,
                current.state.scope_id,
                current.state.screen,
                facts={**dict(current.state.facts), "class": "DEFECT"},
            ),
            current.scope,
            current.continuation,
            current.shared,
            current.previous,
        )

        decision = current_registry.get(AgentKind.BUILD).decide(current)

        self.assertEqual(
            decision.payload["card_choice_review"]["picker_id"],
            "defect.winning_path",
        )

    def test_default_agent_config_selects_winning_path_and_mcts(self):
        config = load_runtime_config(ROOT / "config.yaml")

        self.assertEqual(config.map, "llm")
        self.assertEqual(config.build, "winning_path")
        self.assertEqual(config.combat, "mcts")
        self.assertEqual(config.prompt_language, PromptLanguage.ENGLISH)
        self.assertEqual((config.character, config.ascension), ("IRONCLAD", 20))
        self.assertEqual(config.seed, "random")
        self.assertFalse(config.fullscreen)
        self.assertEqual(config.window_size, (1600, 900))
        self.assertFalse(config.hud)
        self.assertEqual(config.runtime_dir, ROOT / "runtime")
        self.assertEqual(config.log_dir, ROOT)
        self.assertEqual(config.mcts_threads, 12)

    def test_agent_config_validates_size_without_forcing_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "agents:\n  map: llm\n  build: winning_path\n  combat: mcts\n"
                "run:\n  window_size: 1280x800\n",
                encoding="utf-8",
            )
            self.assertEqual(load_runtime_config(path).window_size, (1280, 800))

            path.write_text(
                "agents:\n  map: llm\n  build: winning_path\n  combat: mcts\n"
                "run:\n  window_size: fullscreen\n",
                encoding="utf-8",
            )
            self.assertTrue(load_runtime_config(path).fullscreen)

            path.write_text(
                "agents:\n  map: llm\n  build: winning_path\n  combat: mcts\n"
                "run:\n  window_size: 640x360\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AgentConfigError, "at least"):
                load_runtime_config(path)

    def test_agent_config_accepts_only_registered_implementations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.yaml"
            path.write_text(
                "prompt_language: zh\n"
                "agents:\n  map: llm\n  build: llm\n  combat: llm\n",
                encoding="utf-8",
            )
            config = load_runtime_config(path)
            self.assertEqual(
                (config.map, config.build, config.combat),
                ("llm", "llm", "llm"),
            )
            self.assertEqual(config.prompt_language, PromptLanguage.CHINESE)

            path.write_text(
                "agents:\n  map: unknown\n  build: winning_path\n  combat: mcts\n",
                encoding="utf-8",
            )
            with self.assertRaises(AgentConfigError):
                load_runtime_config(path)

            path.write_text(
                "agents:\n  map: llm\n  build: winning_path\n  combat: mcts\n"
                "run:\n  hud: fullscreen\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AgentConfigError, "hud"):
                load_runtime_config(path)

            path.write_text(
                "prompt_language: de\n"
                "agents:\n  map: llm\n  build: winning_path\n  combat: mcts\n",
                encoding="utf-8",
            )
            with self.assertRaises(AgentConfigError):
                load_runtime_config(path)

            path.write_text(
                "agents:\n  map: llm\n  build: winning_path\n  combat: mcts\n"
                "map_readiness:\n  worlds: 16\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AgentConfigError, "map_readiness"):
                load_runtime_config(path)

            path.write_text(
                "agents:\n  map: llm\n  build: winning_path\n  combat: mcts\n"
                "mcts:\n  threads: 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AgentConfigError, "positive"):
                load_runtime_config(path)

    def test_build_agent_delegates_event_choice_to_llm(self):
        current_registry = registry(FakeLLM(), FakeCombatSearch())
        current = request(
            AgentKind.BUILD,
            ScreenState(
                type="EVENT",
                commands=("choose",),
                choices=("first", "second"),
            ),
        )

        decision = current_registry.get(AgentKind.BUILD).decide(current)

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "build.llm")

    def test_combat_agent_delegates_to_mcts(self):
        search = FakeCombatSearch()
        current_registry = registry(FakeLLM(), search)
        current = request(
            AgentKind.COMBAT,
            ScreenState(type="NONE", commands=("play", "end")),
        )

        decision = current_registry.get(AgentKind.COMBAT).decide(current)

        self.assertEqual(decision.command, "end")
        self.assertEqual(decision.source, "combat.mcts")
        self.assertEqual(len(search.states), 1)

    def test_registry_can_replace_build_and_combat_with_llm(self):
        llm = FakeLLM()
        current_registry = runtime_registry(
            llm,
            DefaultMapTool(llm),
            None,
            build_implementation="llm",
            combat_implementation="llm",
            prompt_language=PromptLanguage.CHINESE,
        )
        build = current_registry.get(AgentKind.BUILD).decide(
            request(
                AgentKind.BUILD,
                ScreenState(
                    type="EVENT",
                    commands=("choose",),
                    choices=("first",),
                    interaction_id="event-1",
                ),
                Continuation(
                    AgentKind.BUILD,
                    "selection",
                    "runtime-build",
                    ("EVENT",),
                    {"step": 2},
                ),
            )
        )
        combat = current_registry.get(AgentKind.COMBAT).decide(
            request(
                AgentKind.COMBAT,
                ScreenState(type="NONE", commands=("play", "end")),
            )
        )

        self.assertEqual(
            (build.command, build.source),
            ("choose 0", "build.llm"),
        )
        self.assertEqual(
            (combat.command, combat.source),
            ("end", "combat.llm"),
        )
        payload = json.loads(llm.requests[0].messages[-1].content)
        self.assertEqual(payload["owner"], "build")
        self.assertEqual(payload["state_owner_hint"], "build")
        self.assertEqual(payload["state_scope_id"], "runtime-build")
        self.assertEqual(payload["screen"]["interaction_id"], "event-1")
        self.assertEqual(payload["continuation"]["data"], {"step": 2})
        self.assertFalse(payload["terminal"])
        self.assertIn("Deal 6 damage", payload["entity_facts"]["cards"][0]["effect"])
        self.assertIn("heal 6 HP", payload["entity_facts"]["relics"][0]["effect"])
        self.assertTrue(
            all(
                "reason in Chinese" in request.messages[0].content
                and "numeric indexes" in request.messages[0].content
                for request in llm.requests
            )
        )

    def test_cli_keeps_seed_as_text_and_reads_yaml_config(self):
        generated = parse_args(["--seed", "0"])
        exact = parse_args(
            [
                "--seed",
                "abc123",
                "--config",
                "agents.yaml",
            ]
        )

        self.assertEqual(generated.seed, "0")
        self.assertEqual(exact.seed, "abc123")
        self.assertEqual(generated.config, ROOT / "config.yaml")
        self.assertEqual(exact.config, Path("agents.yaml"))
        self.assertIsNone(generated.fullscreen)
        self.assertIsNone(generated.log_dir)
        self.assertIsNone(generated.runtime_dir)
        self.assertTrue(parse_args(["--fullscreen"]).fullscreen)

    def test_cli_opens_console_by_default(self):
        with patch("spire_agent.console.main") as console_main:
            run_agent_module.main(["--config", "agents.yaml"])

        console_main.assert_called_once_with(["--config", "agents.yaml"])

    def test_no_tui_uses_the_direct_runner(self):
        with patch.object(run_agent_module, "run", return_value=0) as direct:
            with self.assertRaises(SystemExit) as stopped:
                run_agent_module.main(["--no-tui", "--seed", "0"])

        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(direct.call_args.args[0].seed, "0")


if __name__ == "__main__":
    unittest.main()
