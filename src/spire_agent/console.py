"""Interactive terminal console for starting and supervising Spire Agent runs."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
import curses
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys
from threading import Event, Lock, Thread
import textwrap

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    ContextView,
)
from spire_agent.configuration import AgentConfigError, load_runtime_config
from spire_agent.ports import DecisionProvider
from spire_agent.router import RoomScopeRouter
from spire_agent.run_agent import (
    ROOT,
    parse_args as parse_agent_args,
    run,
)


CHARACTERS = frozenset({"IRONCLAD", "DEFECT"})
BUILD_RESULT_SCREENS = frozenset(
    {"BOSS_REWARD", "CARD_REWARD", "CHEST", "EVENT", "REST", "SHOP_SCREEN"}
)
ROOM_NAMES = {
    "M": "Monster", "?": "Event", "E": "Elite", "E*": "Burning Elite",
    "R": "Rest", "T": "Chest", "$": "Shop", "BOSS": "Boss",
}


class ConsoleCommandError(ValueError):
    pass


ConsoleCommand = tuple[str, ...]


def parse_command(value: str) -> ConsoleCommand:
    try:
        parts = shlex.split(value)
    except ValueError as error:
        raise ConsoleCommandError(str(error)) from error
    if not parts:
        raise ConsoleCommandError("empty command")
    words = [part.casefold() for part in parts]
    if words[0] == "run":
        if len(parts) == 1:
            return ("run",)
        if len(parts) != 4:
            raise ConsoleCommandError("usage: run [character aN seed]")
        character = parts[1].upper()
        if character not in CHARACTERS:
            raise ConsoleCommandError(f"unsupported character: {parts[1]}")
        ascension = words[2]
        if not ascension.startswith("a") or not ascension[1:].isdigit():
            raise ConsoleCommandError("ascension must look like a17")
        level = int(ascension[1:])
        if not 0 <= level <= 20:
            raise ConsoleCommandError("ascension must be between a0 and a20")
        return ("run", character, str(level), parts[3])
    if words[0] == "replay":
        if len(parts) != 2:
            raise ConsoleCommandError("usage: replay seed")
        return ("replay", parts[1].upper())
    if words[0] == "agent":
        if len(parts) != 2 or words[1] not in {"on", "off"}:
            raise ConsoleCommandError("usage: agent on|off")
        return ("agent", words[1])
    if words in (["window"], ["setting", "window"]):
        return ("view", "window")
    if words in (["fullscreen"], ["full", "screen"], ["setting", "fullscreen"]):
        return ("view", "fullscreen")
    if len(parts) == 1 and words[0] in {"hud=on", "hud=off"}:
        return ("hud", words[0].split("=", 1)[1])
    if len(parts) == 1 and words[0] in {"help", "clear", "quit", "exit"}:
        return (words[0],)
    raise ConsoleCommandError(f"unknown console command: {parts[0]}")


class ConsoleBuffer:
    """Thread-safe display lines shared by the game and TUI threads."""

    def __init__(self, limit: int = 5000) -> None:
        self._lines: deque[tuple[str | None, str]] = deque(maxlen=limit)
        self._lock = Lock()

    def add(self, message: object) -> None:
        lines = str(message).splitlines() or [""]
        with self._lock:
            self._lines.extend((None, line) for line in lines)

    def write(self, message: str) -> int:
        for line in message.splitlines():
            if line:
                self.add(line)
        return len(message)

    def flush(self) -> None:
        pass

    def set(self, key: str, message: object) -> None:
        value = str(message).replace("\n", " ")
        with self._lock:
            for index in range(len(self._lines) - 1, -1, -1):
                if self._lines[index][0] == key:
                    self._lines[index] = (key, value)
                    return
            self._lines.append((key, value))

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    def wrapped(self, width: int) -> list[str]:
        width = max(1, width)
        with self._lock:
            lines = tuple(value for _, value in self._lines)
        result: list[str] = []
        for line in lines:
            result.extend(
                textwrap.wrap(
                    line,
                    width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
                or [""]
            )
        return result


class AgentControl:
    """Pause only after the current continuation has completed."""

    def __init__(self, emit: Callable[[str], object]) -> None:
        self._enabled = Event()
        self._enabled.set()
        self._emit = emit
        self._lock = Lock()
        self._paused = False
        self._continuation_notice = False

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def enable(self) -> None:
        self._enabled.set()

    def disable(self) -> None:
        self._enabled.clear()

    def before_decision(self, context: ContextView) -> bool:
        if self.enabled:
            self._continuation_notice = False
            return False
        if context.continuation is not None:
            if not self._continuation_notice:
                self._emit("Agent off is pending until the active continuation completes")
                self._continuation_notice = True
            return False
        with self._lock:
            self._paused = True
        self._emit("Agent paused; the game window is under human control")
        self._enabled.wait()
        with self._lock:
            self._paused = False
        self._continuation_notice = False
        return True


class DecisionActivity:
    """Show the in-flight live decision in a row replaced on confirmation."""

    def __init__(self, provider: DecisionProvider, output: ConsoleBuffer) -> None:
        self._provider = provider
        self._output = output
        self._router = RoomScopeRouter()

    def decide(self, context: ContextView):
        scope = self._router.route(context)
        if not _show_decision(scope.owner, context.state):
            return self._provider.decide(context)
        action = {
            AgentKind.MAP: "Map Agent is thinking ...",
            AgentKind.BUILD: "Build Agent is thinking ...",
            AgentKind.COMBAT: "Combat Agent is searching ...",
        }[scope.owner]
        key = _decision_key(
            context.entry_count, scope.owner, scope.id, context.state.screen.type
        )
        previous = (
            context.last_entry.decision
            if context.last_entry.scope == scope else None
        )
        self._output.set(key, _activity_line(context.state, action, previous))
        routed = self._provider.decide(context)
        if scope.owner is AgentKind.COMBAT and any(
            value is not None for value in _search_stats(routed.decision)
        ):
            self._output.set(
                key,
                _activity_line(context.state, action, routed.decision),
            )
        return routed


class DecisionDisplayObserver:
    """Replace an in-flight row with one concise confirmed result."""

    def __init__(self, output: ConsoleBuffer, run_root: Path | None = None) -> None:
        self._output = output
        self._run_root = run_root
        self._previous = None
        self._build_actions: dict[tuple[str, str], list[str]] = {}

    def on_entry(self, entry: ContextEntry) -> None:
        previous = self._previous
        self._previous = entry.state
        if previous is None:
            seed = entry.state.facts.get("sts_seed")
            if seed:
                suffix = f" | log: {self._run_root / str(seed)}" if self._run_root else ""
                self._output.add(f"RUN | canonical seed={seed}{suffix}")
            return
        if entry.decision is None:
            self._output.add(
                f"{_room_id(entry.state)} | Window takeover synchronized; "
                "continuation cleared; replay disabled"
            )
            return
        owner = entry.scope.owner if entry.scope is not None else previous.owner_hint
        if not _show_decision(owner, previous):
            return
        scope_id = entry.scope.id if entry.scope is not None else previous.scope_id
        key = _decision_key(
            entry.index,
            owner,
            scope_id,
            previous.screen.type,
        )
        result = (
            self._build_sequence_result(scope_id, previous, entry)
            if owner is AgentKind.BUILD
            and previous.screen.type in {"EVENT", "SHOP_SCREEN"}
            else _decision_result(owner, previous, entry)
        )
        if not entry.confirmed:
            result += f" (rejected: {entry.error})"
        self._output.set(key, f"{_room_id(previous)} | {result}")

    def _build_sequence_result(
        self, scope_id: str, state, entry: ContextEntry
    ) -> str:
        choices = [_choice_name(item) for item in state.screen.choices]
        selected = _selection(entry.command or "", choices)
        screen = state.screen.type
        actions = self._build_actions.setdefault((screen, scope_id), [])
        if entry.confirmed:
            actions.append(selected)
        result = f"Build Agent | {'Shop' if screen == 'SHOP_SCREEN' else 'EVENT'}: "
        result += " -> ".join(actions or [selected])
        if screen == "SHOP_SCREEN":
            gold = entry.state.facts.get("gold", state.facts.get("gold", "?"))
            result += f" | Gold: {gold}"
        return result


def _decision_key(
    index: int, owner: AgentKind, scope_id: str, screen_type: str = ""
) -> str:
    if owner is AgentKind.COMBAT:
        return f"combat:{scope_id}"
    if owner is AgentKind.BUILD and screen_type in {"EVENT", "SHOP_SCREEN"}:
        return f"{screen_type.casefold()}:{scope_id}"
    return f"decision:{index}"


def _show_decision(owner: AgentKind, state) -> bool:
    if owner is not AgentKind.BUILD:
        return True
    return state.screen.type in BUILD_RESULT_SCREENS and (
        state.screen.type != "REST" or bool(state.screen.choices)
    )


def _room_id(state) -> str:
    floor = state.facts.get("floor")
    return f"Room {str(floor if floor is not None else '?'):>2}"


def _decision_result(owner: AgentKind, state, entry: ContextEntry) -> str:
    if owner is AgentKind.MAP:
        route = entry.decision.payload.get("run_route")
        rooms = route.get("planned_rooms") if isinstance(route, Mapping) else ()
        path = " -> ".join(str(room) for room in rooms or ())
        if not path:
            segment = route.get("forced_segment") if isinstance(route, Mapping) else ()
            known = [
                ROOM_NAMES.get(str(row.get("room")), str(row.get("room")))
                for row in segment or ()
                if isinstance(row, Mapping)
            ]
            room = str(entry.decision.payload.get("room") or "")
            path = " -> ".join(known) + (" -> ... -> Boss" if known else "")
            path = path or ROOM_NAMES.get(room, room or entry.command or "none")
        return f"Map Agent | Path: {path}"
    if owner is AgentKind.COMBAT:
        current = entry.state.facts.get("current_hp")
        maximum = entry.state.facts.get("max_hp")
        current = current if current is not None else "?"
        maximum = maximum if maximum is not None else "?"
        return f"Combat Agent | HP: {current} / {maximum}"
    choices = [_choice_name(item) for item in state.screen.choices]
    selected = _selection(entry.command or "", choices)
    if state.screen.type == "CARD_REWARD":
        return (
            f"Build Agent | Card reward: [{', '.join(choices)}]"
            f" -> {selected}"
        )
    if state.screen.type == "BOSS_REWARD":
        return f"Build Agent | Boss reward: [{', '.join(choices)}] -> {selected}"
    if state.screen.type == "REST":
        return f"Build Agent | Campfire: {selected}"
    return f"Build Agent | {state.screen.type}: {selected}"


def _activity_line(state, action: str, decision=None) -> str:
    rate, best_hp = _search_stats(decision)
    parts = []
    if rate is not None:
        parts.append(f"win-rate {rate:.0%}")
    if best_hp is not None:
        hp = f"{best_hp:.1f}".removesuffix(".0")
        parts.append(f"best HP {hp}")
    suffix = " | " + " | ".join(parts) if parts else ""
    return f"{_room_id(state)} | {action}{suffix}"


def _search_stats(decision) -> tuple[float | None, float | None]:
    metrics = decision.metrics if decision is not None else {}
    risk = metrics.get("risk") if isinstance(metrics, Mapping) else None
    risk = risk if isinstance(risk, Mapping) else {}
    rate = risk.get("winSampleRate")
    rate = (
        min(1.0, max(0.0, float(rate)))
        if isinstance(rate, (int, float)) and not isinstance(rate, bool)
        else None
    )
    hp = risk.get("meanBestWinEndHp", risk.get("expectedEndHpOnWin"))
    hp = (
        max(0.0, float(hp))
        if isinstance(hp, (int, float)) and not isinstance(hp, bool)
        else None
    )
    return rate, hp


def _choice_name(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "id", "label"):
            if value.get(key):
                return _display_name(value[key])
    return _display_name(value)


def _selection(command: str, choices: list[str]) -> str:
    parts = command.split()
    if len(parts) >= 2 and parts[0] == "choose" and parts[1].isdigit():
        index = int(parts[1])
        if index < len(choices):
            return choices[index]
    return _display_name(command) if command else "none"


def _display_name(value: object) -> str:
    return " ".join(
        word[:1].upper() + word[1:]
        for word in str(value).split(" ")
    )


@dataclass(slots=True)
class ConsoleSettings:
    log_dir: Path
    config: Path
    character: str = "IRONCLAD"
    ascension: int = 20
    seed: str = "random"
    fullscreen: bool = False
    window_size: tuple[int, int] = (1600, 900)
    hud: bool = False
    model: str = ""

    @property
    def run_root(self) -> Path:
        return self.log_dir / "runs"


class ConsoleController:
    def __init__(self, settings: ConsoleSettings, output: ConsoleBuffer) -> None:
        self.settings = settings
        self.output = output
        self.control = AgentControl(output.add)
        self._thread: Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def execute(self, value: str) -> bool:
        try:
            command = parse_command(value)
            name = command[0]
            if name == "run":
                self._start(
                    self.settings.character if len(command) == 1 else command[1],
                    self.settings.ascension if len(command) == 1 else int(command[2]),
                    self.settings.seed if len(command) == 1 else command[3],
                )
            elif name == "replay":
                self._replay(command[1])
            elif name == "agent":
                enabled = command[1] == "on"
                if enabled:
                    self.control.enable()
                else:
                    self.control.disable()
                self.output.add(
                    "Agent resume requested; synchronizing before its next decision"
                    if enabled else
                    "Agent off requested; takeover starts after any continuation completes"
                )
            elif name == "view":
                if self.active:
                    raise ConsoleCommandError("display mode applies to the next run")
                self.settings.fullscreen = command[1] == "fullscreen"
                self.output.add(
                    "Game display: fullscreen" if self.settings.fullscreen else
                    "Game display: window"
                )
            elif name == "hud":
                if self.active:
                    raise ConsoleCommandError("HUD setting applies to the next run")
                self.settings.hud = command[1] == "on"
                self.output.add(f"HUD: {command[1]}")
            elif name == "clear":
                self.output.clear()
            elif name in {"quit", "exit"}:
                if self.active:
                    raise ConsoleCommandError(
                        "a run is active; stop the process with Ctrl+C to close it"
                    )
                return False
            else:
                self.output.add(HELP)
        except (ConsoleCommandError, ValueError) as error:
            self.output.add(f"Error: {error}")
        return True

    def _start(self, character: str, ascension: int, requested_seed: str) -> None:
        if self.active:
            raise ConsoleCommandError("a run is already active")
        args = self._agent_args(
            "--seed", requested_seed,
            "--character", character,
            "--ascension", str(ascension),
        )
        self.output.add(
            f"Run requested: {character} A{ascension} seed={requested_seed}"
        )
        self._launch(args)

    def _replay(self, seed: str) -> None:
        if self.active:
            raise ConsoleCommandError("a run is already active")
        path = self.settings.run_root / seed
        if not path.is_dir():
            raise ConsoleCommandError(f"run directory does not exist: {path}")
        self.output.add(f"Replay requested: {seed}")
        self._launch(self._agent_args("--replay", str(path)))

    def _agent_args(self, *values: str) -> argparse.Namespace:
        common = [
            "--log-dir", str(self.settings.log_dir),
            "--config", str(self.settings.config),
            *values,
        ]
        if self.settings.fullscreen:
            common.append("--fullscreen")
        args = parse_agent_args(common)
        args.fullscreen = self.settings.fullscreen
        return args

    def _launch(self, args: argparse.Namespace) -> None:
        observer = DecisionDisplayObserver(self.output, self.settings.run_root)

        def target() -> None:
            try:
                with redirect_stdout(self.output), redirect_stderr(self.output):
                    code = run(
                        args,
                        hud=self.settings.hud,
                        control=self.control,
                        wrap_decisions=lambda provider: DecisionActivity(
                            provider, self.output
                        ),
                        extra_observers=(observer,),
                        emit=self.output.add,
                        show_transitions=False,
                    )
                self.output.add(f"Run finished with exit code {code}")
            except Exception as error:
                self.output.add(f"Run failed: {type(error).__name__}: {error}")

        self._thread = Thread(target=target, name="spire-agent-run", daemon=True)
        self._thread.start()


HELP = """Commands:
  run
  run ironclad a17 SEED
  replay SEED
  agent on | agent off
  window | fullscreen
  hud=on | hud=off
  help | clear | quit
  Ctrl+D exits the console
  Up/Down browse command history; PageUp/PageDown scroll output
When the agent is off, control the game directly in its window. Turning the
agent on refreshes settled state before automatic decisions resume."""


def _draw(screen, controller: ConsoleController, text: str, scroll: int) -> int:
    height, width = screen.getmaxyx()
    screen.erase()
    if height < 6 or width < 30:
        screen.addnstr(0, 0, "Terminal is too small", max(1, width - 1))
        screen.refresh()
        return scroll
    view_height = height - 4
    lines = controller.output.wrapped(width - 1)
    maximum = max(0, len(lines) - view_height)
    scroll = min(maximum, max(0, scroll))
    end = len(lines) - scroll
    start = max(0, end - view_height)
    for row, line in enumerate(lines[start:end]):
        screen.addnstr(row, 0, line, width - 1)
    width_px, height_px = controller.settings.window_size
    mode = (
        "[fullscreen]"
        if controller.settings.fullscreen
        else f"[{width_px}x{height_px}] fullscreen"
    )
    agent = (
        "on" if controller.control.enabled else
        "off" if controller.control.paused else "off (pending)"
    )
    screen.hline(height - 4, 0, curses.ACS_HLINE, width - 1)
    screen.addnstr(
        height - 3,
        0,
        f"setting: {mode} | hud={'on' if controller.settings.hud else 'off'} "
        f"| agent={agent}",
        width - 1,
        curses.A_BOLD,
    )
    screen.addnstr(
        height - 2,
        0,
        f"default: model={controller.settings.model or '<unset>'} "
        f"character={controller.settings.character.casefold()} "
        f"ascension={controller.settings.ascension} seed={controller.settings.seed} "
        f"log-dir={controller.settings.log_dir}",
        width - 1,
    )
    prompt = "> " + text
    screen.addnstr(height - 1, 0, prompt, width - 1)
    screen.move(height - 1, min(width - 1, len(prompt)))
    screen.refresh()
    return scroll


def _tui(screen, controller: ConsoleController) -> None:
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(100)
    value = ""
    history: list[str] = []
    history_index: int | None = None
    draft = ""
    scroll = 0
    controller.output.add("Slay the Spire console ready. Type 'help' for commands.")
    running = True
    while running:
        scroll = _draw(screen, controller, value, scroll)
        try:
            key = screen.get_wch()
        except curses.error:
            continue
        if key == "\x04":
            running = False
        elif key in ("\n", "\r", curses.KEY_ENTER):
            command, value = value.strip(), ""
            if command:
                if not history or history[-1] != command:
                    history.append(command)
                controller.output.add(f"> {command}")
                running = controller.execute(command)
                scroll = 0
            history_index, draft = None, ""
        elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
            value = value[:-1]
        elif key == "\x15":
            value = ""
        elif key == curses.KEY_PPAGE:
            scroll += max(1, screen.getmaxyx()[0] - 5)
        elif key == curses.KEY_NPAGE:
            scroll = max(0, scroll - max(1, screen.getmaxyx()[0] - 5))
        elif key == curses.KEY_UP:
            if history:
                if history_index is None:
                    draft, history_index = value, len(history) - 1
                elif history_index > 0:
                    history_index -= 1
                value = history[history_index]
        elif key == curses.KEY_DOWN:
            if history_index is not None:
                history_index += 1
                if history_index < len(history):
                    value = history[history_index]
                else:
                    value, history_index = draft, None
        elif key == curses.KEY_END:
            scroll = 0
        elif isinstance(key, str) and key.isprintable():
            value += key


def parse_console_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run without the terminal UI: spire-agent --no-tui [options]",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="override config paths.log_dir",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_console_args(argv)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit("spire-agent requires an interactive terminal; use --no-tui")
    config_path = args.config.resolve()
    try:
        config = load_runtime_config(config_path)
    except AgentConfigError as error:
        raise SystemExit(f"Console stopped: {error}") from error
    settings = ConsoleSettings(
        log_dir=(args.log_dir or config.log_dir).expanduser(),
        config=config_path,
        character=config.character,
        ascension=config.ascension,
        seed=config.seed,
        fullscreen=config.fullscreen,
        window_size=config.window_size,
        hud=config.hud,
        model=config.llm_model or os.environ.get("MODEL", "").strip(),
    )
    controller = ConsoleController(settings, ConsoleBuffer())
    try:
        curses.wrapper(_tui, controller)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


__all__ = [
    "AgentControl",
    "ConsoleBuffer",
    "ConsoleCommandError",
    "ConsoleController",
    "ConsoleSettings",
    "DecisionActivity",
    "DecisionDisplayObserver",
    "main",
    "parse_command",
    "parse_console_args",
]
