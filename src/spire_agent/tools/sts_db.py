"""Small read-only database of authoritative Slay the Spire entity text."""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources
import re


class StsDB:
    """Query only the card, relic, and potion facts needed by BuildAgent."""

    def card(self, name: object) -> dict[str, object] | None:
        base = re.sub(r"\+\d*$", "", str(name or "").strip())
        row = _table("cards.csv").get(_key(base))
        if row is None:
            return None
        result: dict[str, object] = {
            "name": row["Name"],
            "rarity": row["Rarity"],
            "type": row["Type"],
            "cost": _cost(row["Cost"]),
            "upgraded_cost": _cost(row["CostUpgrade"]),
            "effect": row["Desc"],
            "upgraded_effect": row["DescUpgrade"],
        }
        return result

    def relic(self, name: object) -> dict[str, str] | None:
        return _simple_fact("relics.csv", name)

    def potion(self, name: object) -> dict[str, str] | None:
        return _simple_fact("potions.csv", name)

    def mentions(self, *texts: object) -> dict[str, list[dict[str, object]]]:
        """Find explicitly named entities in screen details and choice labels."""

        corpora = [str(value or "") for value in texts]
        result: dict[str, list[dict[str, object]]] = {}
        for output, filename, query in (
            ("cards", "cards.csv", self.card),
            ("relics", "relics.csv", self.relic),
            ("potions", "potions.csv", self.potion),
        ):
            facts = []
            for row in _table(filename).values():
                name = row["Name"]
                pattern = rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])"
                if not any(
                    re.search(pattern, text, flags=0 if index == 0 else re.IGNORECASE)
                    for index, text in enumerate(corpora)
                ):
                    continue
                fact = query(name)
                if fact is not None:
                    facts.append(fact)
            if facts:
                result[output] = facts
        return result


@lru_cache(maxsize=3)
def _table(filename: str) -> dict[str, dict[str, str]]:
    path = resources.files("spire_agent.tools").joinpath("data", filename)
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {
            _key(row.get("Name")): dict(row)
            for row in csv.DictReader(stream)
            if row.get("Name")
        }


def _simple_fact(filename: str, name: object) -> dict[str, str] | None:
    row = _table(filename).get(_key(name))
    if row is None:
        return None
    return {"name": row["Name"], "effect": row["Desc"]}


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _cost(value: str) -> int | str:
    return int(value) if value.isdigit() else value


__all__ = ["StsDB"]
