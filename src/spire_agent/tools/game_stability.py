"""Deterministic convergence gate for raw CommunicationMod observations.

This is infrastructure, not Agent policy. It is the only module that knows
about queued effects, command barriers, and semantic decision boundaries.
Callers provide only read-only STATE and bounded WAIT operations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import time
from typing import Any


class GameStabilityError(RuntimeError):
    """The game did not reach a safe decision boundary."""

    def __init__(self, reason: str, observation: object):
        self.reason = str(reason)
        self.observation = observation
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class StabilityPolicy:
    command_wait_frames: int = 10
    poll_interval: float = 0.1
    timeout: float = 5.0
    max_refreshes: int = 50
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.command_wait_frames < 0:
            raise ValueError("wait frames must be non-negative")
        if self.poll_interval < 0 or self.timeout < 0:
            raise ValueError("stability timings must be non-negative")
        if self.max_refreshes < 0:
            raise ValueError("max_refreshes must be non-negative")


_INFRASTRUCTURE_COMMANDS = frozenset({"click", "key", "state", "wait"})
_GAMEPLAY_COMMANDS = frozenset(
    {
        "cancel",
        "choose",
        "confirm",
        "end",
        "leave",
        "play",
        "potion",
        "proceed",
        "return",
        "skip",
    }
)


def _raw_state(observation: object) -> Mapping[str, Any]:
    raw = getattr(observation, "state", observation)
    if not isinstance(raw, Mapping):
        raise TypeError("game observation must be a mapping or expose mapping .state")
    return raw


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _game(observation: object) -> Mapping[str, Any]:
    return _mapping(_raw_state(observation).get("game_state"))


def _screen(observation: object) -> str:
    game = _game(observation)
    value = game.get("screen_type")
    if value not in (None, ""):
        return str(value).upper()
    return "NONE" if game else "MAIN_MENU"


def _commands(observation: object) -> set[str]:
    return {
        str(command).casefold()
        for command in _sequence(
            _raw_state(observation).get("available_commands")
        )
    }


def _choices(observation: object) -> tuple[Any, ...]:
    return _sequence(_game(observation).get("choice_list"))


def _command_family(command: str) -> str:
    return str(command or "").strip().split(" ", 1)[0].casefold()


def _terminal(observation: object) -> bool:
    game = _game(observation)
    return _screen(observation) == "GAME_OVER" or (
        str(game.get("room_type") or "").casefold() == "truevictoryroom"
    )


def _logical_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _logical_value(item)
            for key, item in value.items()
            if str(key).casefold() not in {"uuid", "card_uuid", "body_text"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_logical_value(item) for item in value]
    return value


def _card_signature(cards: object) -> list[tuple[object, ...]]:
    return [
        (
            card.get("id"),
            card.get("name"),
            card.get("upgrades"),
            card.get("cost"),
            card.get("combat_cost"),
            card.get("misc"),
            card.get("base_damage"),
            card.get("base_block"),
        )
        for card in _sequence(cards)
        if isinstance(card, Mapping)
    ]


def _enemy_signature(enemies: object) -> list[tuple[object, ...]]:
    result = []
    for enemy in _sequence(enemies):
        if not isinstance(enemy, Mapping):
            continue
        gone = bool(enemy.get("is_gone"))
        half_dead = bool(enemy.get("half_dead"))
        result.append(
            (
                enemy.get("id"),
                enemy.get("name"),
                enemy.get("intent"),
                enemy.get("move_id"),
                enemy.get("current_hp"),
                enemy.get("block"),
                gone,
                half_dead,
                ()
                if gone and not half_dead
                else _logical_value(enemy.get("powers") or ()),
            )
        )
    return result


def _transition_signature(observation: object) -> str:
    raw = _raw_state(observation)
    game = _mapping(raw.get("game_state"))
    combat = _mapping(game.get("combat_state"))
    player = _mapping(combat.get("player"))
    value = {
        "screen_type": _screen(observation),
        "screen_state": _logical_value(game.get("screen_state")),
        "choices": _logical_value(_choices(observation)),
        "commands": sorted(_commands(observation)),
        "act": game.get("act"),
        "floor": game.get("floor"),
        "room_phase": game.get("room_phase"),
        "action_phase": game.get("action_phase"),
        "current_action": game.get("current_action"),
        "transition_pending": game.get("transition_pending"),
        "event_state": _logical_value(raw.get("event_state")),
        "seed": game.get("seed"),
        "class": game.get("class"),
        "ascension": game.get("ascension_level"),
        "gold": game.get("gold"),
        "current_hp": game.get("current_hp"),
        "max_hp": game.get("max_hp"),
        "room_type": game.get("room_type"),
        "act_boss": game.get("act_boss"),
        "map": _logical_value(game.get("map") or ()),
        "deck": _card_signature(game.get("deck")),
        "relics": _logical_value(game.get("relics") or ()),
        "potions": _logical_value(game.get("potions") or ()),
        "combat": {
            "turn": combat.get("turn"),
            "cards_played": combat.get("cards_played_this_turn"),
            "attacks_played": combat.get("attacks_played_this_turn"),
            "skills_played": combat.get("skills_played_this_turn"),
            "powers_played": combat.get("powers_played_this_combat"),
            "cards_discarded": combat.get("cards_discarded_this_turn"),
            "times_damaged": combat.get("times_damaged"),
            "lightning_channeled": combat.get("lightning_channeled_this_combat"),
            "frost_channeled": combat.get("frost_channeled_this_combat"),
            "emotion_chip_pending": combat.get("emotion_chip_pending"),
            "centennial_puzzle_used": combat.get(
                "centennial_puzzle_used_this_combat"
            ),
            "energy": player.get("energy"),
            "player_hp": player.get("current_hp"),
            "player_block": player.get("block"),
            "facing_left": player.get("facing_left"),
            "orb_slots": player.get("orb_slots"),
            "orbs": _logical_value(player.get("orbs") or ()),
            "player_powers": _logical_value(player.get("powers") or ()),
            "hand": _card_signature(combat.get("hand")),
            "draw": _card_signature(combat.get("draw_pile")),
            "discard": _card_signature(combat.get("discard_pile")),
            "exhaust": _card_signature(combat.get("exhaust_pile")),
            "enemies": _enemy_signature(combat.get("monsters")),
        },
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _settle_signature(observation: object) -> str:
    """Include visible event progress without changing replay boundary keys."""

    screen_state = _mapping(_game(observation).get("screen_state"))
    event_id = str(screen_state.get("event_id") or "").casefold()
    body = (
        screen_state.get("body_text")
        if _screen(observation) == "EVENT" and event_id != "spire heart"
        else None
    )
    return _transition_signature(observation) + json.dumps(
        body, ensure_ascii=False, default=str
    )


def stable_boundary_key(observation: object) -> str:
    """Return the UUID-independent key used by new replay recordings."""

    value = _transition_signature(observation).encode("utf-8")
    return sha256(value).hexdigest()


def _deck_size(observation: object) -> int:
    return len(_sequence(_game(observation).get("deck")))


def _card_count(observation: object, identity: str) -> int:
    wanted = str(identity or "").casefold()
    return sum(
        1
        for card in _sequence(_game(observation).get("deck"))
        if isinstance(card, Mapping)
        and wanted
        in {
            str(card.get("id") or "").casefold(),
            str(card.get("name") or "").casefold(),
        }
    )


def _choice_index(command: str) -> int | None:
    parts = str(command or "").split()
    if len(parts) != 2 or parts[0].casefold() != "choose":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _selected_permanent_card(before: object | None, command: str):
    if before is None or _screen(before) != "CARD_REWARD":
        return None
    game = _game(before)
    if str(game.get("room_phase") or "").upper() == "COMBAT":
        return None
    index = _choice_index(command)
    cards = _sequence(_mapping(game.get("screen_state")).get("cards"))
    if (
        index is None
        or not 0 <= index < len(cards)
        or not isinstance(cards[index], Mapping)
    ):
        return None
    card = cards[index]
    identity = str(card.get("id") or card.get("name") or "")
    if not identity:
        return None
    return {
        "identity": identity,
        "name": str(card.get("name") or identity),
        "before_count": _card_count(before, identity),
        "before_size": _deck_size(before),
    }


def _choice_name(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("id") or "").strip()
    return str(value or "").strip()


def _selected_external_grid_card(before: object | None, command: str):
    if before is None or _screen(before) != "GRID":
        return None
    game = _game(before)
    if str(game.get("room_phase") or "").upper() == "COMBAT":
        return None
    screen_state = _mapping(game.get("screen_state"))
    if any(
        bool(screen_state.get(flag))
        for flag in ("for_purge", "for_transform", "for_upgrade")
    ):
        return None
    index = _choice_index(command)
    choices = _choices(before)
    if index is None or not 0 <= index < len(choices):
        return None
    names = tuple(_choice_name(choice) for choice in choices)
    external_offer = any(
        name and _card_count(before, name) == 0 for name in names
    )
    name = names[index]
    if not external_offer or not name:
        return None
    return {
        "identity": name,
        "name": name,
        "before_count": _card_count(before, name),
        "before_size": _deck_size(before),
    }


def _grid_expected_deck_size(
    before: object | None,
    after: object,
    command: str,
) -> int | None:
    if before is None or _screen(before) != "GRID" or _screen(after) == "GRID":
        return None
    screen_state = _mapping(_game(before).get("screen_state"))
    family = _command_family(command)
    relic_names = {
        str(relic.get("name") or "").casefold()
        for relic in _sequence(_game(after).get("relics"))
        if isinstance(relic, Mapping)
    }
    confirmation_cards = _sequence(screen_state.get("cards"))
    pandora_confirmation = (
        family == "confirm"
        and bool(screen_state.get("confirm_up"))
        and "pandora's box" in relic_names
        and bool(confirmation_cards)
        and not any(
            bool(screen_state.get(flag))
            for flag in ("for_upgrade", "for_purge", "for_transform")
        )
        and not _sequence(screen_state.get("selected_cards"))
    )
    if pandora_confirmation:
        expected = _deck_size(before) + len(confirmation_cards)
        return expected if _deck_size(after) < expected else None
    if family != "choose" or screen_state.get("for_purge"):
        return None
    before_size = _deck_size(before)
    return before_size if _deck_size(after) < before_size else None


def _deck_growth_expected_size(before: object | None, command: str) -> int | None:
    """Return a permanent card addition that can lag behind its UI response."""

    if before is None or _command_family(command) != "choose":
        return None
    game = _game(before)
    if _screen(before) == "CHEST":
        relics = [
            item
            for item in _sequence(game.get("relics"))
            if isinstance(item, Mapping)
        ]
        names = {_choice_name(item).casefold() for item in relics}
        omamori_blocks_curse = any(
            _choice_name(item).casefold() == "omamori"
            and isinstance(item.get("counter"), int)
            and item["counter"] > 0
            for item in relics
        )
        if (
            "cursed key" in names
            and not omamori_blocks_curse
            and "boss" not in str(game.get("room_type") or "").casefold()
        ):
            return _deck_size(before) + 1
    if _screen(before) != "SHOP_SCREEN":
        return None
    index = _choice_index(command)
    choices = _choices(before)
    if index is None or not 0 <= index < len(choices):
        return None
    selected = _choice_name(choices[index]).casefold()
    cards = _sequence(_mapping(game.get("screen_state")).get("cards"))
    if any(
        isinstance(card, Mapping)
        and _choice_name(card).casefold() == selected
        for card in cards
    ):
        return _deck_size(before) + 1
    return None


def _monster_groups(combat: Mapping[str, Any]):
    monsters = [
        item
        for item in _sequence(combat.get("monsters"))
        if isinstance(item, Mapping)
    ]
    living = [
        monster
        for monster in monsters
        if not monster.get("is_gone")
        and not monster.get("half_dead")
        and (monster.get("current_hp") or 0) > 0
    ]
    reviving = [monster for monster in monsters if monster.get("half_dead")]
    return monsters, living, reviving


def _semantic_issue(
    observation: object,
    *,
    selected_card: Mapping[str, Any] | None,
    grid_expected_deck_size: int | None,
    deck_growth_expected_size: int | None,
) -> str | None:
    raw = _raw_state(observation)
    if _terminal(observation):
        return None
    if raw.get("ready_for_command") is False:
        return "CommunicationMod is not ready for a command"

    commands = _commands(observation)
    decision_commands = commands - _INFRASTRUCTURE_COMMANDS
    if not decision_commands:
        return "no decision command is available"
    if (
        _screen(observation) == "NONE"
        and not _choices(observation)
        and decision_commands
        and decision_commands <= {"choose", "play", "potion"}
    ):
        return "parameterized commands are available without their choices"

    if selected_card is not None:
        if (
            _card_count(observation, str(selected_card["identity"]))
            < int(selected_card["before_count"]) + 1
            or _deck_size(observation) < int(selected_card["before_size"]) + 1
        ):
            return f"selected card {selected_card['name']!r} is not in the deck"

    if (
        grid_expected_deck_size is not None
        and _deck_size(observation) < grid_expected_deck_size
    ):
        return (
            "GRID deck update is incomplete: "
            f"deck has {_deck_size(observation)}, expected {grid_expected_deck_size}"
        )

    if (
        deck_growth_expected_size is not None
        and _deck_size(observation) < deck_growth_expected_size
    ):
        return (
            "permanent card addition is incomplete: "
            f"deck has {_deck_size(observation)}, expected "
            f"{deck_growth_expected_size}"
        )

    game = _game(observation)
    if (
        str(game.get("room_phase") or "").upper() != "COMBAT"
        and game.get("transition_pending") is True
    ):
        names = ", ".join(
            str(item) for item in _sequence(game.get("pending_effects"))[:5]
        )
        return "game transition effects are pending" + (
            f": {names}" if names else ""
        )

    combat = _mapping(game.get("combat_state"))
    monsters, living, reviving = _monster_groups(combat)
    if (
        monsters
        and not living
        and not reviving
        and str(game.get("room_phase") or "").upper() == "COMBAT"
        and _screen(observation) == "NONE"
    ):
        return "combat victory has not reached the reward screen"

    combat_ready = (
        str(game.get("room_phase") or "").upper() == "COMBAT"
        and _screen(observation) == "NONE"
        and bool(commands & {"play", "end"})
    )
    if combat_ready:
        if game.get("transition_pending") is True:
            return "combat actions or effects are still pending"
        missing_signals = [
            name
            for name in ("action_phase", "current_action")
            if name not in game
        ]
        if missing_signals:
            return "required combat stability signals are missing: " + ", ".join(
                missing_signals
            )
        current_action = str(game.get("current_action") or "").strip()
        action_phase = str(game.get("action_phase") or "").upper()
        if current_action:
            return f"combat action queue is executing {current_action}"
        if action_phase and action_phase != "WAITING_ON_USER":
            return f"combat action phase is {action_phase}"
        missing_moves = [
            str(monster.get("name") or monster.get("id") or "UNKNOWN")
            for monster in living
            if type(monster.get("move_id")) is not int
        ]
        if missing_moves:
            return "combat move is not ready for " + ", ".join(missing_moves)

    debug_intents = [
        str(monster.get("name") or monster.get("id") or "UNKNOWN")
        for monster in living
        if str(monster.get("intent") or "").upper() == "DEBUG"
    ]
    if debug_intents:
        return "combat intent is DEBUG for " + ", ".join(debug_intents)
    return None


def settle_game_state(
    before: object | None,
    after: object,
    command: str,
    *,
    read_state: Callable[[], object],
    wait_frames: Callable[[int], object],
    policy: StabilityPolicy | None = None,
) -> object:
    """Return the first safe post-command decision boundary.

    The same function must be used for live play and new replay. It never
    executes a gameplay decision; only the supplied STATE and WAIT operations
    are reachable from this module.
    """

    policy = policy or StabilityPolicy()
    current = after
    game = _game(current)
    if game and not _terminal(current) and "transition_pending" not in game:
        raise GameStabilityError(
            "required stability signal transition_pending is missing; "
            "AgentStateFixes must be installed",
            current,
        )
    baseline = _settle_signature(before) if before is not None else None
    family = _command_family(command)
    selected_card = _selected_permanent_card(before, command)
    if selected_card is None:
        selected_card = _selected_external_grid_card(before, command)
    grid_deck_size = _grid_expected_deck_size(before, after, command)
    deck_growth_size = _deck_growth_expected_size(before, command)

    def advance(operation: Callable[[], object], label: str) -> object:
        try:
            return operation()
        except Exception as error:
            raise GameStabilityError(f"{label} failed: {error}", current) from error

    barrier_frames = (
        policy.command_wait_frames if family in _GAMEPLAY_COMMANDS else 0
    )
    game = _game(current)
    # Generated combat rewards suspend the action queue until a choice is made.
    # Waiting after CommunicationMod exposes that boundary can deadlock it.
    if (
        (baseline is None or _settle_signature(current) != baseline)
        and str(game.get("room_phase") or "").upper() == "COMBAT"
        and _screen(current) == "CARD_REWARD"
        and bool(_choices(current))
        and bool(_commands(current) & {"choose", "skip"})
        and bool(str(game.get("current_action") or "").strip())
        and _semantic_issue(
            current,
            selected_card=selected_card,
            grid_expected_deck_size=grid_deck_size,
            deck_growth_expected_size=deck_growth_size,
        )
        is None
    ):
        return current
    if not _terminal(current) and barrier_frames and "wait" in _commands(current):
        current = advance(
            lambda: wait_frames(barrier_frames),
            "transition barrier",
        )

    deadline = policy.clock() + policy.timeout
    issue = "state has not been checked"
    for attempt in range(policy.max_refreshes + 1):
        issue = _semantic_issue(
            current,
            selected_card=selected_card,
            grid_expected_deck_size=grid_deck_size,
            deck_growth_expected_size=deck_growth_size,
        )
        changed = (
            baseline is None
            or family in {"reset", "state"}
            or _settle_signature(current) != baseline
        )
        if changed and issue is None:
            return current
        if attempt == policy.max_refreshes or policy.clock() >= deadline:
            break

        commands = _commands(current)
        if "wait" in commands:
            if attempt > 0 and policy.poll_interval:
                policy.sleep(policy.poll_interval)
            current = advance(
                lambda: wait_frames(policy.command_wait_frames),
                "transition wait",
            )
        elif "state" in commands:
            if policy.poll_interval:
                policy.sleep(policy.poll_interval)
            current = advance(read_state, "state refresh")
        else:
            issue = f"{issue}; neither STATE nor WAIT is available"
            break

    if baseline is not None and _settle_signature(current) == baseline:
        issue = f"effect of {command!r} is still not observable"
    raise GameStabilityError(
        f"transition {command!r} did not settle: {issue}",
        current,
    )


__all__ = [
    "GameStabilityError",
    "StabilityPolicy",
    "settle_game_state",
    "stable_boundary_key",
]
