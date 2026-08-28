"""Compile reviewed construction modules into the runtime catalog."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from .preference import PreferenceBuilder


BRIDGES = (
    {
        "id": "act1_damage",
        "act": 1,
        "latest_floor": 15,
        "cards": [
            "Anger", "Bludgeon", "Carnage", "Clothesline", "Hemokinesis",
            "Immolate", "Hand of Greed", "Whirlwind",
        ],
    },
    {
        "id": "act1_defense",
        "act": 1,
        "latest_floor": 15,
        "cards": [
            "Clothesline", "Disarm", "Flame Barrier", "Impervious",
            "Power Through", "Second Wind", "Shockwave", "True Grit",
        ],
    },
    {
        "id": "act1_aoe",
        "act": 1,
        "latest_floor": 15,
        "cards": ["Cleave", "Immolate", "Dramatic Entrance", "Whirlwind"],
    },
)

CANDIDATE_BRIDGES = (
    {
        "id": "act1_consistency",
        "act": 1,
        "latest_floor": 15,
        "cards": ["Armaments", "Battle Trance", "Pommel Strike", "Shrug It Off"],
    },
)

DOMINANT_CARDS = (
    {
        "name": "Apotheosis",
        "acts": [1],
        "maximum_owned": 0,
        "reason": "Act 1 decks still have substantial combat upgrade debt",
    },
)

CONDITIONAL_CARDS = (
    {
        "name": "Reckless Charge",
        "requires_any_owned": [
            "Dark Embrace", "Evolve", "Feel No Pain", "Fire Breathing",
            "Medical Kit",
        ],
    },
    {
        "name": "Wild Strike",
        "requires_any_owned": ["Evolve", "Fire Breathing", "Medical Kit"],
    },
)


def compile_data(
    graph_path: Path,
    choices_path: Path,
    support_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic catalog; certificates remain diagnostic evidence."""

    graph = json.loads(graph_path.read_text("utf-8"))
    support_path = support_path or graph_path.with_name("support_capabilities.json")
    support = json.loads(support_path.read_text("utf-8"))
    modules = [_module(row) for _, row in sorted(graph["module_catalog"].items())]
    module_by_id = {row["id"]: row for row in modules}
    support_cards = [_support_card(row) for row in support.get("cards") or ()]
    forbidden = [
        {
            "name": str(row["card"]),
            "acts": [int(act) for act in row.get("acts") or ()],
        }
        for row in graph.get("card_policies") or ()
        if row.get("policy") == "FORBID"
    ]

    signatures: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for certificate in graph["certificates"]:
        signature = tuple(
            sorted(str(row["module_id"]) for row in certificate["active_modules"])
        )
        signatures[signature].append(str(certificate["certificate_id"]))
    routes = [
        {
            "id": examples[0],
            "modules": list(signature),
            "cards": sorted(
                {
                    card
                    for module_id in signature
                    for card in _module_cards(module_by_id[module_id])
                }
            ),
            "examples": examples,
        }
        for signature, examples in sorted(signatures.items())
    ]

    relevant = {
        card for module in modules for card in _module_cards(module)
    } | {
        str(row["card"]) for row in support_cards
    } | {
        card for row in BRIDGES for card in row["cards"]
    }
    offers: Counter[str] = Counter()
    preferences = PreferenceBuilder()
    reward_count = 0
    winning: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for raw in choices_path.read_text("utf-8").splitlines():
        row = json.loads(raw)
        decision = row.get("decision") or {}
        context, run = row.get("context") or {}, row.get("run") or {}
        preferences.observe(
            decision.get("act"),
            decision.get("offered") or (),
            picked=decision.get("picked") or decision.get("selected_card"),
            skipped=bool(decision.get("skipped")),
            used_singing_bowl=bool(decision.get("used_singing_bowl")),
            owned=context.get("deck_before_counts") or {},
        )
        offered = set(map(str, decision.get("offered") or ()))
        if not offered:
            continue
        reward_count += 1
        offers.update(offered & relevant)
        if run.get("heart_kill") and int(run.get("ascension") or 0) >= 20:
            winning[str(run.get("run_id"))].append(
                (
                    int(decision.get("choice_index") or 0),
                    int(decision.get("act") or 0),
                    int(decision.get("floor") or 0),
                )
            )
    horizons: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sequence in winning.values():
        sequence.sort()
        for index, (_, act, floor) in enumerate(sequence):
            horizons[(act, floor)].append(len(sequence) - index - 1)

    return {
        "schema_version": 2,
        "model": {
            "meaning": "reviewed construction-module progress",
            "module_count": len(modules),
            "certificate_count": len(graph["certificates"]),
            "signature_count": len(routes),
            "offer_rows": reward_count,
        },
        "source": {
            "graph_sha256": _sha256(graph_path),
            "choices_sha256": _sha256(choices_path),
            "support_sha256": _sha256(support_path),
        },
        "forbidden_cards": forbidden,
        "dominant_cards": DOMINANT_CARDS,
        "conditional_cards": CONDITIONAL_CARDS,
        "bridges": BRIDGES,
        "candidate_bridges": CANDIDATE_BRIDGES,
        "support_cards": support_cards,
        "expert_preferences": preferences.payload(),
        "offer_rates": {
            card: round(offers[card] / reward_count, 8) for card in sorted(relevant)
        },
        "horizons": {
            str(act): [
                [floor, statistics.median(values)]
                for (candidate_act, floor), values in sorted(horizons.items())
                if candidate_act == act
            ]
            for act in range(1, 5)
        },
        "modules": modules,
        "routes": routes,
    }


def _module(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(raw["module_id"]),
        "aspect": str(raw.get("aspect") or ""),
        "phase": str(raw.get("phase") or ""),
        "candidate_policy": str(raw.get("candidate_policy") or ""),
        "activation": raw.get("activation") or {},
        "provides": [
            str(row["capability"])
            for row in raw.get("provides") or ()
            if row.get("capability")
        ],
    }


def _support_card(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: raw[key]
        for key in (
            "card", "provides", "upgraded_provides", "bridge_provides",
            "requires_any_owned",
        )
        if key in raw
    }


def _module_cards(module: Mapping[str, Any]) -> set[str]:
    activation = module.get("activation") or {}
    clauses = [
        clause
        for slot in activation.get("slots") or ()
        for clause in (slot.get("any") or (slot,))
    ]
    return {
        str(fact["name"])
        for clause in clauses
        for fact in clause.get("all") or ()
        if fact.get("kind") == "CARD"
    } | {
        str(card)
        for clause in clauses
        for card in (clause.get("group") or {}).get("cards") or ()
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = compile_data(args.graph, args.choices, args.support)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {payload['model']['module_count']} modules and "
        f"{payload['model']['signature_count']} diagnostic signatures with "
        f"{len(payload['expert_preferences']['pairs'])} preference pairs to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()


__all__ = ["compile_data", "main"]
