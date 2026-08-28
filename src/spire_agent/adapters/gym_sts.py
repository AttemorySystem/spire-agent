"""Normalize gym-sts observations and expose the frozen ``GameSession`` port.

This module is deliberately independent of gym-sts imports.  Tests, replay,
and alternate bridge versions can pass either a raw CommunicationMod mapping
or any object exposing that mapping as ``.state``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
from typing import Any

from spire_agent.contracts import (
    AgentKind,
    ExecutionResult,
    GameState,
    ScreenState,
    SessionRefresh,
)
from spire_agent.tools.game_stability import (
    StabilityPolicy,
    settle_game_state,
    stable_boundary_key,
)


class GymStsAdapterError(ValueError):
    """A gym-sts value cannot be represented by the Spire Agent contracts."""


class GymStsSessionError(RuntimeError):
    """The live gym-sts session cannot produce a usable state."""


_NON_COMBAT_SCREENS = frozenset(
    {
        "BOSS_REWARD",
        "CHEST",
        "COMBAT_REWARD",
        "EVENT",
        "GAME_OVER",
        "MAP",
        "REST",
        "SHOP_ROOM",
        "SHOP_SCREEN",
    }
)

_FACT_EXCLUSIONS = frozenset(
    {"choice_list", "combat_state", "current_action", "screen_state", "screen_type"}
)


def _raw_state(observation: object) -> Mapping[str, Any]:
    raw = getattr(observation, "state", observation)
    if not isinstance(raw, Mapping):
        raise GymStsAdapterError(
            "expected a CommunicationMod mapping or an object with mapping .state"
        )
    return raw


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _screen_type(game: Mapping[str, Any]) -> str:
    value = game.get("screen_type")
    if value not in (None, ""):
        return str(value).upper()
    return "NONE" if game else "MAIN_MENU"


def _living_enemy(combat: Mapping[str, Any]) -> bool:
    enemies = combat.get("enemies", combat.get("monsters", ()))
    if not isinstance(enemies, Sequence) or isinstance(enemies, (str, bytes)):
        return False
    for raw_enemy in enemies:
        enemy = _mapping(raw_enemy)
        nested = _mapping(enemy.get("enemy"))
        if nested:
            enemy = nested
        if enemy.get("is_gone"):
            continue
        hp = enemy.get("current_hp", enemy.get("hp"))
        if hp is None:
            return True
        try:
            if int(hp) > 0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _is_active_combat(game: Mapping[str, Any], screen: str) -> bool:
    """Ignore stale combat snapshots retained on post-combat screens."""

    combat = _mapping(game.get("combat_state"))
    phase = str(game.get("room_phase") or "").upper()

    if screen == "CARD_REWARD":
        return phase == "COMBAT" and _living_enemy(combat)
    if screen in _NON_COMBAT_SCREENS:
        return False
    if phase == "COMBAT":
        return True

    combat_shaped = any(
        key in combat for key in ("player", "monsters", "enemies", "hand", "turn")
    )
    return screen == "NONE" and combat_shaped


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _commands(raw: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(raw.get("available_commands")))


def _scope_id(game: Mapping[str, Any], owner: AgentKind) -> str:
    seed = str(game.get("seed") or "unknown")
    act = str(game.get("act") or "?")
    floor = str(game.get("floor") or "?")
    room = str(game.get("room_type") or "unknown").casefold()
    return f"{seed}:a{act}:f{floor}:{room}:{owner.value}"


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]


class GymStsObservationAdapter:
    """Convert one raw gym-sts observation into an immutable ``GameState``."""

    def command_error(self, observation: object) -> str | None:
        error = _raw_state(observation).get("error")
        return None if error in (None, "") else str(error)

    def adapt(
        self,
        observation: object,
        *,
        sts_seed: str | None = None,
    ) -> GameState:
        raw = _raw_state(observation)
        error = self.command_error(raw)
        if error is not None:
            raise GymStsAdapterError(f"gym-sts rejected the command: {error}")

        game = _mapping(raw.get("game_state"))
        screen = _screen_type(game)
        active_combat = _is_active_combat(game, screen)
        if screen == "MAP":
            owner = AgentKind.MAP
        elif active_combat:
            owner = AgentKind.COMBAT
        else:
            owner = AgentKind.BUILD

        scope_id = _scope_id(game, owner)
        commands = _commands(raw)
        choices = _sequence(game.get("choice_list"))
        details = _mapping(game.get("screen_state"))
        current_action = str(game.get("current_action") or "")
        interaction = {
            "screen": screen,
            "commands": commands,
            "choices": choices,
            "details": details,
            "current_action": current_action,
        }

        facts = {
            key: value for key, value in game.items() if key not in _FACT_EXCLUSIONS
        }
        facts["replay_boundary_key"] = stable_boundary_key(observation)
        if sts_seed is not None:
            facts["sts_seed"] = str(sts_seed)
        facts["bridge"] = {
            "in_game": raw.get("in_game"),
            "ready_for_command": raw.get("ready_for_command"),
        }
        terminal = screen == "GAME_OVER" or (
            str(game.get("room_type") or "").casefold() == "truevictoryroom"
        )

        return GameState(
            owner_hint=owner,
            scope_id=scope_id,
            screen=ScreenState(
                type=screen,
                commands=commands,
                choices=choices,
                interaction_id=f"{scope_id}:{_fingerprint(interaction)}",
                current_action=current_action,
                details=details,
            ),
            terminal=terminal,
            facts=facts,
            combat=_mapping(game.get("combat_state")) if active_combat else None,
        )


class GymStsSession:
    """Thin live-game adapter implementing the frozen ``GameSession`` port."""

    def __init__(
        self,
        env: object,
        *,
        adapter: GymStsObservationAdapter | None = None,
        stability_policy: StabilityPolicy | None = None,
        reset_kwargs: Mapping[str, Any] | None = None,
        on_sts_seed: Callable[[str], object] | None = None,
        rejected_exceptions: tuple[type[BaseException], ...] = (),
        fatal_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._env = env
        self._adapter = adapter or GymStsObservationAdapter()
        self._stability_policy = stability_policy or StabilityPolicy()
        self._reset_kwargs = dict(reset_kwargs or {})
        self._on_sts_seed = on_sts_seed
        self._rejected_exceptions = rejected_exceptions
        self._fatal_exceptions = fatal_exceptions
        self._current_observation: object | None = None
        self._sts_seed: str | None = None
        self._closed = False

    def reset(self) -> GameState:
        self._ensure_open()
        reset = getattr(self._env, "reset", None)
        if not callable(reset):
            raise GymStsSessionError("gym-sts environment has no reset()")
        result = reset(**self._reset_kwargs)
        observation, reset_info = self._reset_result(result)
        sts_seed = reset_info.get("sts_seed", getattr(self._env, "sts_seed", None))
        if sts_seed in (None, ""):
            raise GymStsSessionError("gym-sts reset did not return sts_seed")
        self._sts_seed = str(sts_seed)
        if self._on_sts_seed is not None:
            self._on_sts_seed(self._sts_seed)
        observation = self._settle(None, observation, "reset")
        self._current_observation = observation
        return self._adapt(observation)

    def execute(self, command: str) -> ExecutionResult:
        self._ensure_open()
        command = str(command).strip()
        if not command:
            raise GymStsSessionError("command must not be empty")
        before = self._current_observation
        if before is None:
            raise GymStsSessionError("gym-sts session must be reset before execute()")

        try:
            observation = self._send(command)
        except self._fatal_exceptions:
            # A transport timeout can mean the command was accepted but its
            # acknowledgement was lost. Retrying or refreshing it as a normal
            # rejection would make a replay log ambiguous.
            raise
        except self._rejected_exceptions as error:
            return self._rejected(command, str(error))

        error = self._adapter.command_error(observation)
        if error is not None:
            return self._rejected(command, error)

        observation = self._settle(before, observation, command)
        self._current_observation = observation
        return ExecutionResult(
            command=command,
            state=self._adapt(observation),
            confirmed=True,
        )

    def refresh(self) -> SessionRefresh:
        """Settle state after optional direct interaction with the game UI."""

        self._ensure_open()
        before = self._current_observation
        if before is None:
            raise GymStsSessionError("gym-sts session must be reset before refresh()")
        observation = self._settle(before, self._refresh(), "state")
        before_game = _mapping(_raw_state(before).get("game_state"))
        after_game = _mapping(_raw_state(observation).get("game_state"))
        changed = (
            stable_boundary_key(observation) != stable_boundary_key(before)
            or before_game.get("replay_rng_state")
            != after_game.get("replay_rng_state")
        )
        self._current_observation = observation
        return SessionRefresh(self._adapt(observation), changed)

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
        self._closed = True

    def _send(self, command: str) -> object:
        send = getattr(self._env, "_do_action", None)
        if not callable(send):
            raise GymStsSessionError("gym-sts environment has no string command API")
        return send(command)

    def _refresh(self) -> object:
        observe = getattr(self._env, "observe", None)
        if callable(observe):
            return observe()
        return self._send("state")

    def _wait_frames(self, frames: int) -> object:
        return self._send(f"wait {int(frames)}")

    def _settle(
        self,
        before: object | None,
        after: object,
        command: str,
    ) -> object:
        return settle_game_state(
            before,
            after,
            command,
            read_state=self._refresh,
            wait_frames=self._wait_frames,
            policy=self._stability_policy,
        )

    def _rejected(self, command: str, error: str) -> ExecutionResult:
        observation = self._settle(None, self._refresh(), "state")
        self._current_observation = observation
        return ExecutionResult(
            command=command,
            state=self._adapt(observation),
            confirmed=False,
            error=error,
        )

    def _reset_result(
        self,
        result: object,
    ) -> tuple[object, Mapping[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            info = _mapping(result[1])
            candidate = info.get("observation", result[0])
        else:
            info = {}
            candidate = result
        try:
            raw = _raw_state(candidate)
        except GymStsAdapterError:
            return self._refresh(), info
        if "game_state" not in raw and "available_commands" not in raw:
            return self._refresh(), info
        return candidate, info

    def _adapt(self, observation: object) -> GameState:
        if self._sts_seed is None:
            raise GymStsSessionError("gym-sts session has no canonical sts_seed")
        return self._adapter.adapt(observation, sts_seed=self._sts_seed)

    def _ensure_open(self) -> None:
        if self._closed:
            raise GymStsSessionError("gym-sts session is closed")


__all__ = [
    "GymStsAdapterError",
    "GymStsObservationAdapter",
    "GymStsSession",
    "GymStsSessionError",
]
