"""Run Spire Agent until completion or a manual interruption."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import os
from pathlib import Path
import secrets
import sys


ROOT = Path(__file__).resolve().parents[2]

from gym_sts import constants as sts_constants
from gym_sts import exceptions as sts_exceptions
from gym_sts.envs.base import SlayTheSpireGymEnv

from spire_agent.adapters import GymStsSession, SeedRequest
from spire_agent.configuration import (
    AgentConfigError,
    load_runtime_config,
)
from spire_agent.context import GameContext
from spire_agent.contracts import ContextEntry, DecisionRequest
from spire_agent.extensions import (
    CardChoiceRecorder,
    LiveOnlyObserver,
    HudObserver,
    ReplayError,
    ReplayJournal,
    ReplayRuntime,
    RunDirectory,
    RunHistoryRecorder,
    WinningPathRecorder,
    create_run_llm_client,
    prepare_display,
    restore_game_rng,
)
from spire_agent.extensions.log_io import jsonable, write_json
from spire_agent.extensions.run_directory import RunDirectoryError
from spire_agent.game_agent import GameAgent
from spire_agent.observers import ObserverHub
from spire_agent.ports import DecisionControl, DecisionProvider, RunObserver
from spire_agent.registry import SubAgentRegistry
from spire_agent.router import RoomScopeRouter, RoutedDecisionProvider
from spire_agent.subagents import (
    BuildConversationReducer,
    CombatTool,
    MapTool,
    PromptLanguage,
    create_build_agent,
    create_combat_agent,
    create_map_agent,
)
from spire_agent.tools.build_flow import build_choice_policy
from spire_agent.tools.map import DefaultMapTool, EncounterReadiness
from spire_agent.tools.mcts import CombatMCTS, DefaultCombatTool, PotionGate
from spire_agent.tools.llm_agents import (
    create_llm_build_agent,
    create_llm_combat_agent,
)
from spire_agent.tools.winning_path import create_card_picker
from spire_agent.validation import AvailableCommandValidator


class ConsoleObserver:
    """Print one compact line for every confirmed or rejected transition."""

    def on_entry(self, entry: ContextEntry) -> None:
        state = entry.state
        command = entry.command or "reset"
        result = "ok" if entry.confirmed else f"rejected: {entry.error}"
        print(
            f"[{entry.index}] {command} -> owner={state.owner_hint.value} "
            f"screen={state.screen.type} floor={state.facts.get('floor', '?')} "
            f"seed={state.facts.get('sts_seed', '?')} ({result})",
            flush=True,
        )


def runtime_registry(
    llm: object,
    map_tool: MapTool,
    combat_tool: CombatTool | None,
    *,
    prompt_language: PromptLanguage | str = PromptLanguage.ENGLISH,
    map_implementation: str = "llm",
    build_implementation: str = "winning_path",
    combat_implementation: str = "mcts",
    character: str = "IRONCLAD",
    choice_policy: Callable[
        [DecisionRequest], Mapping[str, object] | None
    ] = build_choice_policy,
) -> SubAgentRegistry:
    if map_implementation != "llm":
        raise AgentConfigError(f"cannot compose map agent {map_implementation!r}")
    map_agent = create_map_agent(map_tool)
    if build_implementation == "llm":
        build_agent = create_llm_build_agent(llm, prompt_language)
    elif build_implementation == "winning_path":
        build_agent = create_build_agent(
            llm,
            create_card_picker(character),
            prompt_language=prompt_language,
            choice_policy=choice_policy,
        )
    else:
        raise AgentConfigError(f"cannot compose build agent {build_implementation!r}")
    if combat_implementation == "llm":
        combat_agent = create_llm_combat_agent(llm, prompt_language)
    elif combat_implementation == "mcts" and combat_tool is not None:
        combat_agent = create_combat_agent(combat_tool)
    else:
        raise AgentConfigError(f"cannot compose combat agent {combat_implementation!r}")
    return SubAgentRegistry(
        (
            build_agent,
            map_agent,
            combat_agent,
        )
    )


def run(
    args: argparse.Namespace,
    *,
    hud: bool | None = None,
    control: DecisionControl | None = None,
    wrap_decisions: Callable[[DecisionProvider], DecisionProvider] | None = None,
    extra_observers: Sequence[RunObserver] = (),
    emit: Callable[[str], object] | None = print,
    show_transitions: bool = True,
) -> int:
    config = load_runtime_config(args.config)
    runtime_dir = args.runtime_dir or config.runtime_dir
    log_dir = args.log_dir or config.log_dir
    mcts_binary = args.mcts_binary or config.mcts_binary
    card_eval_binary = args.card_eval_binary or config.card_eval_binary
    communication_timeout = (
        args.communication_timeout
        if args.communication_timeout is not None
        else config.communication_timeout
    )
    replay_action_delay = (
        config.replay_action_delay_seconds
        if args.replay_action_delay is None
        else args.replay_action_delay
    )
    if replay_action_delay < 0:
        raise AgentConfigError("replay action delay must be non-negative")
    fullscreen = config.fullscreen if args.fullscreen is None else args.fullscreen
    hud = config.hud if hud is None else hud
    lib_dir, mods_dir, out_dir = (
        runtime_dir / "lib",
        runtime_dir / "mods",
        runtime_dir / "out",
    )
    if fullscreen or hud:
        error = prepare_display()
        if error is not None:
            raise AgentConfigError(error)
    if hud:
        visualizer = mods_dir.resolve() / "AgentVisualizer.jar"
        if not visualizer.is_file():
            raise AgentConfigError(
                f"hud requires {visualizer}; run "
                "game_mods/agent_visualizer/build.sh"
            )
        os.environ["AGENT_OVERLAY_STATE"] = str(
            (out_dir / "agent_overlay.json").resolve()
        )
    if args.replay is None:
        run_directory = RunDirectory(log_dir.expanduser() / "runs")
        replay = ReplayJournal(run_directory)
        seed = config.seed if args.seed is None else str(args.seed)
        if seed.casefold() == "random":
            seed = str(secrets.randbelow(2**63 - 1))
        seed_request = SeedRequest.parse(seed)
        character = str(args.character or config.character).strip().upper()
        ascension = config.ascension if args.ascension is None else args.ascension
    else:
        run_directory = RunDirectory.open(args.replay)
        replay = ReplayJournal(run_directory, resume=True)
        if replay.seed != run_directory.seed:
            raise ReplayError(
                f"replay seed {replay.seed!r} does not match directory "
                f"{run_directory.seed!r}"
            )
        seed_request = SeedRequest.exact(replay.seed)
        character = replay.character.strip().upper()
        ascension = replay.ascension
    if character not in {"IRONCLAD", "DEFECT"}:
        raise AgentConfigError(f"unsupported character {character!r}")
    hud_observer = HudObserver(
        run_directory,
        replay,
        out_dir / "agent_overlay.json",
        display=hud,
    )
    llm = create_run_llm_client(
        run_directory,
        base_url=config.llm_base_url,
        model=config.llm_model,
        stream_event=hud_observer.on_llm_event,
    )
    readiness = EncounterReadiness(card_eval_binary, run_directory)
    map_tool = DefaultMapTool(llm, config.prompt_language, readiness)
    combat_tool = (
        DefaultCombatTool(
            CombatMCTS(
                mcts_binary,
                run_directory,
                simulations=config.mcts_simulations,
                threads=config.mcts_threads,
                max_time_ms=config.mcts_max_time_ms,
                adaptive_time_ms=config.mcts_adaptive_time_ms,
                adaptive_simulations=config.mcts_adaptive_simulations,
            ),
            PotionGate(run_directory),
        )
        if config.combat == "mcts"
        else None
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    env = SlayTheSpireGymEnv(
        str(lib_dir),
        str(mods_dir),
        str(out_dir),
        character=character,
        ascension=ascension,
        required_mods=(
            sts_constants.HUD_MODS
            if hud
            else sts_constants.REQUIRED_MODS
        ),
        communication_timeout=communication_timeout,
        fullscreen=fullscreen,
        game_dir=str(runtime_dir / "tmp"),
        window_size=config.window_size,
        game_language=(
            "ZHS"
            if config.prompt_language is PromptLanguage.CHINESE
            else "ENG"
        ),
        player_name="Agent",
    )

    def bind_run(sts_seed: str) -> object:
        path = run_directory.bind(sts_seed)
        if args.replay is None:
            effective = jsonable(asdict(config))
            effective.pop("fullscreen", None)
            effective.update(
                character=character,
                ascension=ascension,
                seed=seed_request.input_value,
                window_size=(
                    "fullscreen"
                    if fullscreen
                    else f"{config.window_size[0]}x{config.window_size[1]}"
                ),
                hud=hud,
                runtime_dir=str(runtime_dir),
                log_dir=str(log_dir),
                mcts_binary=str(mcts_binary),
                card_eval_binary=str(card_eval_binary),
                llm_base_url=config.llm_base_url
                or os.environ.get("MODEL_URL", ""),
                llm_model=config.llm_model or os.environ.get("MODEL", ""),
                communication_timeout=communication_timeout,
                replay_action_delay_seconds=replay_action_delay,
            )
            write_json(
                path / "run_config.json",
                {
                    "schema_version": 1,
                    "source": str(args.config),
                    "settings": effective,
                },
                sort_keys=True,
            )
        return path

    live_session = GymStsSession(
        env,
        reset_kwargs=seed_request.reset_kwargs,
        on_sts_seed=bind_run,
        rejected_exceptions=(sts_exceptions.StSError,),
        fatal_exceptions=(sts_exceptions.StSTimeoutError,),
    )
    live_decisions: DecisionProvider = RoutedDecisionProvider(
        RoomScopeRouter(),
        runtime_registry(
            llm,
            map_tool,
            combat_tool,
            prompt_language=config.prompt_language,
            map_implementation=config.map,
            build_implementation=config.build,
            combat_implementation=config.combat,
            character=character,
            choice_policy=lambda request: build_choice_policy(
                request,
                lambda state, groups: readiness.evaluate(state, groups=groups),
            ),
        ),
    )
    if wrap_decisions is not None:
        live_decisions = wrap_decisions(live_decisions)
    replay_runtime = ReplayRuntime(
        live_session,
        live_decisions,
        replay,
        lambda rng, key: restore_game_rng(env._do_action, rng, key),
        action_delay_seconds=replay_action_delay,
    )
    observers: list[RunObserver] = [RunHistoryRecorder(run_directory, replay)]
    observers.append(hud_observer)
    if show_transitions:
        observers.append(ConsoleObserver())
    observers.extend(extra_observers)
    observers.extend(
        (
            LiveOnlyObserver(WinningPathRecorder(run_directory), replay),
            LiveOnlyObserver(CardChoiceRecorder(run_directory), replay),
        )
    )
    game = GameAgent(
        session=replay_runtime,
        context=GameContext(BuildConversationReducer()),
        decisions=replay_runtime,
        validator=AvailableCommandValidator(),
        observers=ObserverHub(observers),
        control=control,
    )

    if emit is not None:
        emit(
            f"Starting Spire Agent: input_seed={seed_request.input_value} "
            f"seed_mode={seed_request.mode.value} character={character} "
            f"ascension={ascension} map_agent={config.map} "
            f"build_agent={config.build} combat_agent={config.combat} "
            f"prompt_language={config.prompt_language.value} "
            f"fullscreen={fullscreen} hud={hud}"
        )
    try:
        game.run()
    except KeyboardInterrupt:
        print("Manual interruption received; closing the game.", flush=True)
        return 130
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="local root containing lib/, mods/, tmp/, and out/",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="base directory containing runs/",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="runtime configuration YAML",
    )
    parser.add_argument(
        "--mcts-binary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--card-eval-binary",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed")
    parser.add_argument(
        "--replay",
        type=Path,
        help="resume an explicitly selected new-format run directory",
    )
    parser.add_argument(
        "--replay-action-delay",
        type=float,
        help="seconds to show each settled historical state before replaying",
    )
    parser.add_argument(
        "--character", choices=("IRONCLAD", "DEFECT")
    )
    parser.add_argument("--ascension", type=int, choices=range(21))
    parser.add_argument("--communication-timeout", type=float)
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=None,
        help="launch the game in fullscreen mode",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--no-tui" not in values:
        from spire_agent.console import main as console_main

        console_main(values)
        return
    values.remove("--no-tui")
    try:
        code = run(parse_args(values))
    except (AgentConfigError, ReplayError, RunDirectoryError) as error:
        print(f"Agent stopped: {error}", flush=True)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()


__all__ = [
    "AgentConfigError",
    "ConsoleObserver",
    "load_runtime_config",
    "main",
    "parse_args",
    "run",
    "runtime_registry",
]
