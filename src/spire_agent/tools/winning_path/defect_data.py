"""Compile the Defect catalog from reviewed templates and expert runs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
import re
import statistics
from typing import Any
import zipfile

from .preference import PreferenceBuilder, normalize_card
from .protocol import PROTOCOL_VERSION, protocol_sha256


def compile_defect_catalog(
    archive_path: Path, parameters_path: Path, cards_path: Path
) -> dict[str, Any]:
    parameters = _parameters(parameters_path)
    canonical = _card_names(cards_path)
    preferences, offers, sequences = PreferenceBuilder(), Counter(), []
    winning_assets: list[tuple[Counter[str], set[str]]] = []
    expert_runs = winning_runs = reward_rows = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".run"):
                continue
            run = json.loads(archive.read(member))
            if run.get("character_chosen") != "DEFECT" or int(
                run.get("ascension_level") or 0
            ) != 20:
                continue
            expert_runs += 1
            rows = _reward_rows(run, canonical)
            reward_rows += len(rows)
            sequence = []
            for act, floor, offered, picked in rows:
                preferences.observe(
                    act, offered, picked=picked,
                    skipped=picked == "SKIP",
                    used_singing_bowl=picked == "Singing Bowl",
                    owned=(),
                )
                offers.update(set(offered))
                sequence.append((act, floor))
            if _heart_win(run):
                winning_runs += 1
                sequences.append(sequence)
                winning_assets.append((
                    Counter(
                        _canonical(card, canonical)
                        for card in run.get("master_deck") or ()
                    ),
                    set(map(str, run.get("relics") or ())),
                ))
    knowledge = dict((parameters.get("templates") or {}).get("knowledge") or {})
    modules = list(knowledge.get("modules") or ())
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "model": {
            "character": "DEFECT",
            "meaning": "reviewed construction-module progress",
            "module_count": len(modules),
            "expert_runs": expert_runs,
            "winning_runs": winning_runs,
            "offer_rows": reward_rows,
            "preference_context": "ACT_ONLY",
        },
        "source": {
            "archive_sha256": _sha256(archive_path),
            "parameters_sha256": _sha256(parameters_path),
            "cards_sha256": _sha256(cards_path),
            "protocol_sha256": protocol_sha256("DEFECT"),
        },
        "derived": {
            "expert_preferences": preferences.payload(),
            "offer_rates": {
                card: round(count / reward_rows, 8)
                for card, count in sorted(offers.items())
            },
            "horizons": _horizons(sequences),
        },
        "provenance": {
            "expert_filter": {
                "character": "DEFECT",
                "ascension": 20,
                "preference_runs": "all",
                "template_runs": "A20 Heart wins",
                "standard_rewards": "combat floors and Act 1/2 boss rewards",
            },
            "template_support": {
                module["module_id"]: sum(
                    _module_complete(deck, module, relics)
                    for deck, relics in winning_assets
                )
                for module in modules
            },
        },
    }


def compile_defect_certificates(
    archive_path: Path, parameters_path: Path, cards_path: Path
) -> dict[str, Any]:
    """Map A20 Heart wins to current modules without granting runtime authority."""

    parameters = _parameters(parameters_path)
    modules = list((parameters["templates"]["knowledge"]).get("modules") or ())
    canonical = _card_names(cards_path)
    certificates = []
    signatures: dict[tuple[str, ...], list[str]] = defaultdict(list)
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".run"):
                continue
            run = json.loads(archive.read(member))
            if (
                run.get("character_chosen") != "DEFECT"
                or int(run.get("ascension_level") or 0) != 20
                or not _heart_win(run)
            ):
                continue
            certificate_id = str(run.get("play_id") or Path(member).stem)
            if certificate_id in seen:
                raise ValueError(f"duplicate Defect certificate {certificate_id!r}")
            seen.add(certificate_id)
            deck = Counter(
                _canonical(card, canonical) for card in run.get("master_deck") or ()
            )
            active = tuple(
                sorted(
                    str(module["module_id"])
                    for module in modules
                    if _module_complete(
                        deck, module, set(map(str, run.get("relics") or ()))
                    )
                )
            )
            signature_id = _signature_id(active)
            signatures[active].append(certificate_id)
            certificates.append(
                {
                    "certificate_id": certificate_id,
                    "archive_member": member,
                    "signature_id": signature_id,
                    "active_modules": list(active),
                    "deck_counts": dict(sorted(deck.items())),
                    "relics": sorted(map(str, run.get("relics") or ())),
                }
            )
    module_support = Counter(
        module for row in certificates for module in row["active_modules"]
    )
    distribution = Counter(len(row["active_modules"]) for row in certificates)
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scope": {
            "character": "DEFECT",
            "ascension": 20,
            "outcome": "HEART_WIN",
            "runtime_authority": False,
        },
        "source": {
            "archive_sha256": _sha256(archive_path),
            "parameters_sha256": _sha256(parameters_path),
            "cards_sha256": _sha256(cards_path),
            "protocol_sha256": protocol_sha256("DEFECT"),
        },
        "summary": {
            "certificate_count": len(certificates),
            "signature_count": len(signatures),
            "module_count": len(modules),
            "certificates_with_modules": sum(
                bool(row["active_modules"]) for row in certificates
            ),
            "certificates_without_modules": sum(
                not row["active_modules"] for row in certificates
            ),
            "active_module_count_distribution": {
                str(count): frequency
                for count, frequency in sorted(distribution.items())
            },
            "module_support": dict(sorted(module_support.items())),
        },
        "signatures": [
            {
                "signature_id": _signature_id(active),
                "active_modules": list(active),
                "example_count": len(examples),
                "certificate_ids": sorted(examples),
            }
            for active, examples in sorted(
                signatures.items(), key=lambda row: (-len(row[1]), row[0])
            )
        ],
        "certificates": sorted(
            certificates, key=lambda row: str(row["certificate_id"])
        ),
        "limitations": [
            "A module signature proves final-deck co-occurrence, not causality.",
            "Only the currently reviewed Defect modules are evaluated.",
            "The archive does not provide exact deck-before state for every reward.",
            "This artifact is offline evidence and is not loaded by the runtime picker.",
        ],
    }


def _reward_rows(
    run: Mapping[str, Any], canonical: Mapping[str, str]
) -> list[tuple[int, int, list[str], str]]:
    combat_floors = {
        int(row.get("floor") or 0)
        for row in run.get("damage_taken") or ()
        if isinstance(row, Mapping)
    }
    result = []
    for row in run.get("card_choices") or ():
        if not isinstance(row, Mapping):
            continue
        floor = int(row.get("floor") or 0)
        if floor not in combat_floors and floor not in (16, 33):
            continue
        picked = _canonical(row.get("picked"), canonical)
        offered = [_canonical(card, canonical) for card in row.get("not_picked") or ()]
        if picked not in {"SKIP", "Singing Bowl"}:
            offered.append(picked)
        if offered:
            result.append((_act(floor), floor, offered, picked))
    return result


def _horizons(sequences: Sequence[Sequence[tuple[int, int]]]) -> dict[str, list[list[float]]]:
    values: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sequence in sequences:
        for index, (act, floor) in enumerate(sequence):
            values[act, floor].append(len(sequence) - index - 1)
    return {
        str(act): [
            [floor, statistics.median(samples)]
            for (row_act, floor), samples in sorted(values.items())
            if row_act == act
        ]
        for act in range(1, 5)
    }


def _module_complete(
    deck: Counter[str], module: Mapping[str, Any], relics: set[str] | None = None
) -> bool:
    for slot in module["activation"]["slots"]:
        if not bool(slot.get("required", True)):
            continue
        clauses = slot.get("any") or (slot,)
        if not any(_clause_complete(deck, relics or set(), row) for row in clauses):
            return False
    return True


def _clause_complete(
    deck: Counter[str], relics: set[str], clause: Mapping[str, Any]
) -> bool:
    for fact in clause.get("all") or ():
        name = str(fact.get("name") or "")
        if fact.get("kind") == "RELIC" and name not in relics:
            return False
        if fact.get("kind") == "CARD" and deck[name] < int(fact.get("count") or 1):
            return False
    group = clause.get("group") or {}
    cards = group.get("cards") or ()
    return (not cards) or (
        sum(deck[card] > 0 for card in cards) >= int(group["minimum_distinct"])
        and sum(deck[card] for card in cards) >= int(group["minimum_total_copies"])
    )


def _card_names(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            _compact(row["Name"]): row["Name"]
            for row in csv.DictReader(handle)
            if row.get("Name")
        }


def _canonical(value: object, names: Mapping[str, str]) -> str:
    normalized = normalize_card(value)
    return normalized if normalized in {"SKIP", "Singing Bowl"} else names.get(
        _compact(normalized), normalized
    )


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _act(floor: int) -> int:
    return 1 if floor <= 16 else 2 if floor <= 33 else 3 if floor <= 50 else 4


def _heart_win(run: Mapping[str, Any]) -> bool:
    return bool(run.get("victory")) and int(run.get("floor_reached") or 0) >= 57


def _signature_id(modules: Sequence[str]) -> str:
    if not modules:
        return "defect-signature-empty"
    digest = sha256("\n".join(modules).encode()).hexdigest()[:12]
    return f"defect-signature-{digest}"


def _parameters(path: Path) -> Mapping[str, Any]:
    parameters = _json_object(path)
    if (
        parameters.get("schema_version") != 1
        or parameters.get("policy_id") != "defect.winning_path"
        or (parameters.get("scope") or {}).get("character") != "DEFECT"
    ):
        raise ValueError("Defect parameters have an incompatible identity")
    return parameters


def _json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--certificates-output", type=Path)
    args = parser.parse_args(argv)
    if not args.output and not args.certificates_output:
        parser.error("provide --output and/or --certificates-output")
    if args.output:
        payload = compile_defect_catalog(args.archive, args.parameters, args.cards)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"Wrote {payload['model']['offer_rows']} expert choices to {args.output}"
        )
    if args.certificates_output:
        certificates = compile_defect_certificates(
            args.archive, args.parameters, args.cards
        )
        args.certificates_output.write_text(
            json.dumps(certificates, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        summary = certificates["summary"]
        print(
            f"Wrote {summary['certificate_count']} certificates and "
            f"{summary['signature_count']} signatures to "
            f"{args.certificates_output}"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "compile_defect_catalog", "compile_defect_certificates", "main"
]
