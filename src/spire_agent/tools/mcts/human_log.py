"""Compact append-only MCTS log for live human inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any


def append_search_log(
    run_path: Path,
    search_id: str,
    request: Mapping[str, Any],
    settings: Mapping[str, Any],
    raw: object,
    elapsed_ms: float,
    error: Exception | None,
) -> None:
    game = _map(request.get("game_state"))
    combat = _map(game.get("combat_state"))
    player = _map(combat.get("player"))
    monsters = _items(combat.get("monsters"))
    orbs = _items(player.get("orbs"))
    result = _map(raw)
    lines = [
        f"==== MCTS {search_id} | Act {game.get('act', '?')} "
        f"Floor {game.get('floor', '?')} Turn {combat.get('turn', '?')} | "
        f"{settings.get('search_role', 'baseline')} | {elapsed_ms:.0f} ms ====",
        "Monsters:",
        *(_monster(index, value) for index, value in enumerate(monsters)),
        "Player:",
        f"  HP={player.get('current_hp', '?')}/{player.get('max_hp', '?')} "
        f"block={player.get('block', 0)} energy={player.get('energy', '?')} "
        f"powers={_powers(player.get('powers'))}",
        *((f"  Orbs: {_assets(orbs)}",) if orbs else ()),
        f"  Relics: {_assets(game.get('relics'), counters=True)}",
        f"  Potions: {_assets(game.get('potions'))}",
        "  Counters: " + " ".join(
            f"{name.removesuffix('_this_turn')}={combat.get(name, '?')}"
            for name in (
                "cards_played_this_turn",
                "attacks_played_this_turn",
                "skills_played_this_turn",
                "cards_discarded_this_turn",
                "times_damaged",
            )
        ),
        "CardManager:",
        f"  Hand: {_pile(combat.get('hand'), True)}",
        f"  Draw: {_pile(combat.get('draw_pile'))}",
        f"  Discard: {_pile(combat.get('discard_pile'))}",
        f"  Exhaust: {_pile(combat.get('exhaust_pile'))}",
    ]
    if error is not None:
        lines.extend(("Search: ERROR", f"  {type(error).__name__}: {error}"))
    else:
        chosen = str(result.get("rootCommand") or "")
        lines.append(
            f"Actions: policy={result.get('rootSelectionPolicy', '?')} "
            f"stop={result.get('searchStopReason', '?')}"
        )
        for value in _items(result.get("rootActions")):
            row = _map(value)
            action = str(row.get("action") or "?")
            lines.append(
                f" {'*' if action == chosen else ' '} {_action(action, combat)}"
                f" | win={_pct(row.get('winSampleRate'))}"
                f" low={_pct(row.get('lowerQuartileWinSampleRate'))}"
                f" value={_num(row.get('selectionValue', row.get('value')), 4)}"
                f" endHP={_num(row.get('expectedEndHpOnWin'), 1)}"
                f" visits={_num(row.get('visits'), 0)}"
            )
        lines.append(f"Chosen: {_action(chosen, combat)}")
    with (run_path / "mcts.log").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n\n")
        stream.flush()
        os.fsync(stream.fileno())


def _monster(index: int, value: object) -> str:
    monster = _map(value)
    damage = monster.get("move_adjusted_damage")
    flags = ",".join(
        name for name in ("half_dead", "is_gone") if monster.get(name)
    )
    attack = (
        f" damage={damage}x{monster.get('move_hits', 1)}"
        if isinstance(damage, (int, float)) and damage >= 0
        else ""
    )
    return (
        f"  [{index}] {monster.get('name', '?')} "
        f"HP={monster.get('current_hp', '?')}/{monster.get('max_hp', '?')} "
        f"block={monster.get('block', 0)} intent={monster.get('intent', '?')} "
        f"move={monster.get('move_id', '?')}{attack} "
        f"powers={_powers(monster.get('powers'))} flags=[{flags}]"
    )


def _pile(value: object, indexed: bool = False) -> str:
    cards = _items(value)
    labels = []
    for index, value in enumerate(cards, 1):
        card = _map(value)
        label = str(card.get("name") or card.get("id") or "?")
        if card.get("cost") is not None:
            label += f"({card['cost']})"
        labels.append(f"[{index}] {label}" if indexed else label)
    return f"{len(cards)} [" + ", ".join(labels) + "]"


def _powers(value: object) -> str:
    return "[" + ", ".join(
        f"{power.get('name') or power.get('id')}:{power.get('amount', '?')}"
        for item in _items(value)
        if (power := _map(item)).get("name") or power.get("id")
    ) + "]"


def _assets(value: object, counters: bool = False) -> str:
    result = []
    for item in _items(value):
        asset = _map(item)
        name = str(asset.get("name") or asset.get("id") or "?")
        counter = asset.get("counter")
        if counters and isinstance(counter, int) and counter >= 0:
            name += f":{counter}"
        result.append(name)
    return "[" + ", ".join(result) + "]"


def _action(action: str, combat: Mapping[str, Any]) -> str:
    parts = action.split()
    if len(parts) < 2 or parts[0] != "play" or not parts[1].isdigit():
        return "end turn" if action == "end" else action
    hand, monsters = _items(combat.get("hand")), _items(combat.get("monsters"))
    index = int(parts[1]) - 1
    card = _map(hand[index]) if 0 <= index < len(hand) else {}
    label = f"{action} {card.get('name', '?')}"
    if len(parts) >= 3 and parts[2].isdigit():
        target = int(parts[2])
        monster = _map(monsters[target]) if 0 <= target < len(monsters) else {}
        label += f" -> [{target}] {monster.get('name', '?')}"
    return label


def _map(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%" if isinstance(value, (int, float)) else "?"


def _num(value: object, precision: int) -> str:
    return f"{float(value):.{precision}f}" if isinstance(value, (int, float)) else "?"


__all__ = ["append_search_log"]
