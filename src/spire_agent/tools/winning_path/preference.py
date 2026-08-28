"""Compile and query expert card preferences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
import math
import re
from typing import Any


SKIP = "__SKIP__"
CONFIDENCE_Z = 1.96


class PreferenceBuilder:
    """Collect auditable pairwise outcomes from expert reward choices."""

    def __init__(self) -> None:
        self.rows = 0
        self.observations = 0
        self._counts: Counter[tuple[str, str, str]] = Counter()

    def observe(
        self,
        act: object,
        offered: Iterable[object],
        *,
        picked: object | None = None,
        skipped: bool = False,
        used_singing_bowl: bool = False,
        owned: object = (),
        deck_size: int | None = None,
    ) -> None:
        self.rows += 1
        if used_singing_bowl:
            return
        cards = tuple(dict.fromkeys(normalize_card(value) for value in offered))
        cards = tuple(card for card in cards if card)
        selected = SKIP if skipped else normalize_card(picked)
        if not cards or (selected != SKIP and selected not in cards):
            return
        self.observations += 1
        owned_cards = _owned_cards(owned)
        size_band = _deck_size_band(
            deck_size if deck_size is not None else _owned_count(owned)
        )
        opponents = cards if selected == SKIP else (*cards, SKIP)
        for opponent in opponents:
            if opponent != selected:
                self._record(str(int(act or 0)), selected, opponent)
                left, right = sorted((selected, opponent))
                context = f"{int(left in owned_cards)}{int(right in owned_cards)}"
                self._record(
                    f"{int(act or 0)}:{context}", selected, opponent
                )
                self._record(
                    f"{int(act or 0)}:{context}:{size_band}", selected, opponent
                )

    def payload(self) -> dict[str, Any]:
        pairs: dict[tuple[str, str, str], list[int]] = {}
        for (bucket, winner, loser), count in self._counts.items():
            left, right = sorted((winner, loser))
            values = pairs.setdefault((bucket, left, right), [0, 0])
            values[0 if winner == left else 1] += count
        return {
            "schema_version": 1,
            "rows": self.rows,
            "observations": self.observations,
            "pairs": [
                [bucket, left, right, wins[0], wins[1]]
                for (bucket, left, right), wins in sorted(pairs.items())
            ],
        }

    def _record(self, bucket: str, winner: str, loser: str) -> None:
        self._counts[(bucket, winner, loser)] += 1


class PreferenceTable:
    """Resolve an offer with pairwise majority votes and a logged tie-break."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        context_order: Sequence[str] = (
            "act+owned-pair+deck-size-band",
            "act+owned-pair",
            "act when both cards are unowned",
        ),
        deck_size_limits: Sequence[int] = (15, 20, 25, 30),
    ) -> None:
        self._context_order = tuple(context_order)
        self._deck_size_limits = tuple(map(int, deck_size_limits))
        self._pairs: dict[tuple[str, str, str], tuple[int, int]] = {}
        for row in payload.get("pairs") or ():
            if not isinstance(row, Sequence) or len(row) != 5:
                continue
            bucket, left, right, left_wins, right_wins = row
            self._pairs[(str(bucket), str(left), str(right))] = (
                int(left_wins), int(right_wins)
            )

    def decide(
        self,
        act: object,
        options: Iterable[object],
        *,
        owned: object = (),
        deck_size: int | None = None,
        require_confidence: bool = False,
        confidence_z: float = CONFIDENCE_Z,
    ) -> dict[str, Any]:
        names = tuple(dict.fromkeys(normalize_option(value) for value in options))
        names = tuple(name for name in names if name)
        owned_cards = _owned_cards(owned)
        scores = {name: [0, 0] for name in names}
        evidence = []
        for first, second in combinations(names, 2):
            left, right = sorted((first, second))
            counts, bucket = self._lookup(
                act, left, right, owned_cards, deck_size
            )
            left_wins, right_wins = counts
            confident = _confident_majority(left_wins, right_wins, confidence_z)
            if left_wins == right_wins or (require_confidence and not confident):
                majority = None
            else:
                majority = left if left_wins > right_wins else right
                minority = right if majority == left else left
                scores[majority][0] += 1
                scores[minority][0] -= 1
                margin = abs(left_wins - right_wins)
                scores[majority][1] += margin
                scores[minority][1] -= margin
            evidence.append(
                {
                    "left": left,
                    "right": right,
                    "left_wins": left_wins,
                    "right_wins": right_wins,
                    "bucket": bucket,
                    "majority": majority,
                    "confident": confident,
                }
            )
        observed = sum(
            row["left_wins"] + row["right_wins"] > 0 for row in evidence
        )
        ranking = sorted(names, key=lambda name: (-scores[name][0], name))
        best = tuple(
            name
            for name in ranking
            if scores[name][0] == scores[ranking[0]][0]
        ) if ranking else ()
        winner = best[0] if observed and len(best) == 1 else None
        return {
            "winner": winner,
            "status": "WINNER" if winner else "TIE" if observed else "NO_EVIDENCE",
            "scores": {
                name: {"copeland": scores[name][0], "margin": scores[name][1]}
                for name in ranking
            },
            "tied": list(best) if observed and not winner else [],
            "observed_pairs": observed,
            "comparisons": evidence,
        }

    def compare(
        self,
        act: object,
        first: object,
        second: object,
        *,
        owned: object = (),
        deck_size: int | None = None,
    ) -> dict[str, Any]:
        """Return the contextual pair counts without applying a threshold."""

        first, second = normalize_option(first), normalize_option(second)
        left, right = sorted((first, second))
        counts, bucket = self._lookup(act, left, right, _owned_cards(owned), deck_size)
        left_wins, right_wins = counts
        return {
            "left": left,
            "right": right,
            "left_wins": left_wins,
            "right_wins": right_wins,
            "bucket": bucket,
        }

    def _lookup(
        self,
        act: object,
        left: str,
        right: str,
        owned: set[str],
        deck_size: int | None,
    ) -> tuple[tuple[int, int], str | None]:
        act_bucket = str(int(act or 0))
        context = f"{int(left in owned)}{int(right in owned)}"
        for mode in self._context_order:
            if mode == "act+owned-pair+deck-size-band" and deck_size is not None:
                bucket = f"{act_bucket}:{context}:{_deck_size_band(deck_size, self._deck_size_limits)}"
            elif mode == "act+owned-pair":
                bucket = f"{act_bucket}:{context}"
            elif mode == "act when both cards are unowned" and context == "00":
                bucket = act_bucket
            else:
                continue
            counts = self._pairs.get((bucket, left, right), (0, 0))
            if sum(counts):
                return counts, bucket
        return (0, 0), None


def _confident_majority(
    left_wins: int, right_wins: int, confidence_z: float = CONFIDENCE_Z
) -> bool:
    total = left_wins + right_wins
    return bool(
        total
        and abs(left_wins - right_wins) / math.sqrt(total) >= confidence_z
    )


def normalize_card(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("id") or value.get("value")
    name = re.sub(r"\+\d*$", "", str(value or "").strip())
    return {
        "AscendersBane": "Ascender's Bane",
        "Adaptation": "Rushdown",
        "All For One": "All for One",
        "Auto Shields": "Auto-Shields",
        "BootSequence": "Boot Sequence",
        "Conserve Battery": "Charge Battery",
        "ClearTheMind": "Tranquility",
        "Crippling Poison": "Crippling Cloud",
        "Defend_R": "Defend",
        "Defend_B": "Defend",
        "Gash": "Claw",
        "Ghostly": "Apparition",
        "Fasting2": "Fasting",
        "Judgement": "Judgment",
        "Lockon": "Bullseye",
        "Night Terror": "Nightmare",
        "PathToVictory": "Pressure Points",
        "Redo": "Recursion",
        "Steam": "Steam Barrier",
        "Steam Power": "Overclock",
        "Strike_R": "Strike",
        "Strike_B": "Strike",
        "Turbo": "TURBO",
        "Undo": "Equilibrium",
        "Underhanded Strike": "Sneaky Strike",
        "Vengeance": "Simmering Fury",
        "Wireheading": "Foresight",
        "Wraith Form v2": "Wraith Form",
    }.get(name, name)


def _owned_cards(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            normalize_card(name)
            for name, count in value.items()
            if int(count or 0) > 0
        }
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return {normalize_card(item) for item in value if normalize_card(item)}
    return set()


def _owned_count(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(max(0, int(count or 0)) for count in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return sum(1 for _ in value)
    return 0


def _deck_size_band(size: int, limits: Sequence[int] = (15, 20, 25, 30)) -> str:
    lower = 0
    for limit in limits:
        if size < limit:
            return f"{lower:02d}-{limit - 1:02d}"
        lower = limit
    return f"{lower}+"


def normalize_option(value: object) -> str:
    return SKIP if value == SKIP else normalize_card(value)


__all__ = [
    "SKIP",
    "PreferenceBuilder",
    "PreferenceTable",
    "normalize_card",
]
