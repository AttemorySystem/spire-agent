"""Compile derived expert evidence for Winning Path."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from importlib import resources
import json
from pathlib import Path
from typing import Any

from spire_agent.contracts import frozen_mapping

from .parameters import load_parameters
from .protocol import (
    PROTOCOL_VERSION, canonical_sha256, normalize_character, protocol_sha256,
)


class CatalogError(ValueError):
    """Reviewed knowledge is malformed or uses an unreviewed field."""


_MODULE_FIELDS = frozenset(
    {
        "activation",
        "aspect",
        "blocked_by",
        "bottleneck_requires_prerequisites",
        "candidate_policy",
        "convergence",
        "dynamic_verification",
        "energy",
        "entry",
        "evidence_templates",
        "exit_conditions",
        "goals",
        "hard_resource_constraints",
        "mechanism",
        "module_id",
        "name",
        "phase",
        "provides",
        "soft_resource_pressures",
        "strategic_role",
    }
)
_REQUIRED_MODULE_FIELDS = frozenset(
    {
        "activation",
        "aspect",
        "candidate_policy",
        "dynamic_verification",
        "energy",
        "evidence_templates",
        "exit_conditions",
        "goals",
        "hard_resource_constraints",
        "mechanism",
        "module_id",
        "name",
        "phase",
        "provides",
        "soft_resource_pressures",
    }
)
_CANDIDATE_POLICIES = frozenset(
    {"ADVISORY_ONLY", "COMPATIBLE_ONLY", "DEFAULT_SKIP", "MAINLINE_ONLY"}
)
_KNOWLEDGE_FIELDS = frozenset(
    {
        "modules", "routes", "card_policies", "resource_rules", "graph_scope",
        "graph_semantics", "forbidden_cards", "dominant_cards",
        "conditional_cards", "bridges", "candidate_bridges", "support",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "interpretation",
        "capabilities",
        "density_capabilities",
        "verification_support",
        "pressure_relief",
        "module_aspect_satisfies",
        "act_bridges",
        "cards",
    }
)
_SUPPORT_CARD_FIELDS = frozenset(
    {
        "card",
        "provides",
        "upgraded_provides",
        "advisory_provides",
        "bridge_provides",
        "requires_any_owned",
        "copy_evidence",
        "resource_costs",
    }
)


def compile_catalog(
    graph_path: Path,
    choices_path: Path,
    support_path: Path | None = None,
    parameters_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic catalog without duplicating policy knowledge."""

    from .build_data import compile_data

    support_path = support_path or graph_path.with_name("support_capabilities.json")
    graph = _load_object(graph_path, "graph")
    support = _load_object(support_path, "support")
    modules_raw = _object(graph.get("module_catalog"), "graph.module_catalog")
    for catalog_id, raw in sorted(modules_raw.items()):
        module = _object(raw, f"graph.module_catalog.{catalog_id}")
        _validate_module(str(catalog_id), module)
    _validate_support(support)

    compiled = compile_data(graph_path, choices_path, support_path)
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "model": deepcopy(compiled["model"]),
        "source": {
            **deepcopy(compiled["source"]),
            "protocol_sha256": protocol_sha256(),
        },
        "derived": {
            "expert_preferences": deepcopy(compiled["expert_preferences"]),
            "offer_rates": deepcopy(compiled["offer_rates"]),
            "horizons": deepcopy(compiled["horizons"]),
        },
        "provenance": {
            "graph_source": deepcopy(graph.get("source") or {}),
            "graph_summary": deepcopy(graph.get("summary") or {}),
        },
    }
    if parameters_path is not None:
        parameters = _load_object(parameters_path, "parameters")
        payload["source"]["parameters_sha256"] = canonical_sha256(parameters)
    return payload


@lru_cache(maxsize=None)
def load_default_catalog(character: object = "IRONCLAD") -> Mapping[str, Any]:
    """Load the shipped, fingerprinted catalog."""

    character = normalize_character(character)
    filename = (
        "ironclad_catalog.json" if character == "IRONCLAD" else "defect_catalog.json"
    )
    path = resources.files("spire_agent.tools.winning_path").joinpath(
        "data", filename
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CatalogError("default catalog must be an object")
    expected = {
        "schema_version",
        "protocol_version",
        "model",
        "source",
        "derived",
        "provenance",
    }
    if set(payload) != expected:
        raise CatalogError("default catalog has an unexpected top-level schema")
    if payload["schema_version"] != 1 or payload["protocol_version"] != PROTOCOL_VERSION:
        raise CatalogError("default catalog version mismatch")
    if (payload.get("source") or {}).get("protocol_sha256") != protocol_sha256(character):
        raise CatalogError("default catalog was built with another protocol")
    payload["knowledge"] = _plain(
        load_parameters(character)["templates"]["knowledge"]
    )
    knowledge = _object(payload.get("knowledge"), "default catalog knowledge")
    if set(knowledge) != _KNOWLEDGE_FIELDS:
        raise CatalogError("default knowledge has an unexpected schema")
    for raw in _array(knowledge.get("modules"), "default catalog modules"):
        module = _object(raw, "default catalog module")
        _validate_module(str(module.get("module_id") or ""), module)
    _validate_support(_object(knowledge.get("support"), "default catalog support"))
    return frozen_mapping(payload)


@lru_cache(maxsize=None)
def default_catalog_sha256(character: object = "IRONCLAD") -> str:
    return canonical_sha256(load_default_catalog(character))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _validate_module(catalog_id: str, module: Mapping[str, Any]) -> None:
    actual = set(map(str, module))
    missing = _REQUIRED_MODULE_FIELDS - actual
    unknown = actual - _MODULE_FIELDS
    if missing or unknown:
        raise CatalogError(
            f"module {catalog_id!r} fields differ: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if _text(module["module_id"], f"module {catalog_id}.module_id") != catalog_id:
        raise CatalogError(
            f"module catalog key {catalog_id!r} does not match module_id"
        )
    for field in ("name", "aspect", "phase", "mechanism"):
        _text(module[field], f"module {catalog_id}.{field}")
    if module["candidate_policy"] not in _CANDIDATE_POLICIES:
        raise CatalogError(
            f"module {catalog_id} has unknown candidate_policy "
            f"{module['candidate_policy']!r}"
        )
    if not isinstance(module["dynamic_verification"], bool):
        raise CatalogError(
            f"module {catalog_id}.dynamic_verification must be boolean"
        )

    activation = _object(module["activation"], f"module {catalog_id}.activation")
    unknown_activation = set(activation) - {
        "slots",
        "anchor_slots",
        "requires_capabilities",
    }
    if unknown_activation:
        raise CatalogError(
            f"module {catalog_id}.activation has unknown fields "
            f"{sorted(unknown_activation)}"
        )
    slots = _array(activation.get("slots"), f"module {catalog_id}.activation.slots")
    slot_ids = []
    for index, raw in enumerate(slots):
        slot = _object(raw, f"module {catalog_id}.activation.slots[{index}]")
        unknown_slot = set(slot) - {"id", "required", "all", "any", "group"}
        if unknown_slot:
            raise CatalogError(
                f"module {catalog_id} slot has unknown fields {sorted(unknown_slot)}"
            )
        slot_ids.append(_text(slot.get("id"), f"module {catalog_id} slot id"))
        if "required" in slot and not isinstance(slot["required"], bool):
            raise CatalogError(f"module {catalog_id} slot.required must be boolean")
        if not any(key in slot for key in ("all", "any", "group")):
            raise CatalogError(f"module {catalog_id} slot has no satisfaction clause")
    if len(slot_ids) != len(set(slot_ids)):
        raise CatalogError(f"module {catalog_id} has duplicate slot IDs")
    anchors = _strings(
        activation.get("anchor_slots", ()),
        f"module {catalog_id}.activation.anchor_slots",
        allow_empty=True,
    )
    if not set(anchors).issubset(slot_ids):
        raise CatalogError(f"module {catalog_id} anchors reference unknown slots")
    _strings(
        activation.get("requires_capabilities", ()),
        f"module {catalog_id}.activation.requires_capabilities",
        allow_empty=True,
    )
    provides = _array(module["provides"], f"module {catalog_id}.provides")
    for index, raw in enumerate(provides):
        row = _object(raw, f"module {catalog_id}.provides[{index}]")
        if set(row) != {"aspect", "capability"}:
            raise CatalogError(
                f"module {catalog_id}.provides[{index}] must contain aspect and capability"
            )
        _text(row["aspect"], f"module {catalog_id}.provides[{index}].aspect")
        _text(row["capability"], f"module {catalog_id}.provides[{index}].capability")


def _validate_support(support: Mapping[str, Any]) -> None:
    actual = set(map(str, support))
    if actual != _SUPPORT_FIELDS:
        raise CatalogError(
            "support fields differ: "
            f"missing={sorted(_SUPPORT_FIELDS - actual)}, "
            f"unknown={sorted(actual - _SUPPORT_FIELDS)}"
        )
    seen: set[str] = set()
    for index, raw in enumerate(_array(support["cards"], "support.cards")):
        row = _object(raw, f"support.cards[{index}]")
        unknown = set(row) - _SUPPORT_CARD_FIELDS
        if unknown or "card" not in row or "provides" not in row:
            raise CatalogError(
                f"support.cards[{index}] fields differ: unknown={sorted(unknown)}"
            )
        name = _text(row["card"], f"support.cards[{index}].card")
        if name in seen:
            raise CatalogError(f"duplicate support card {name!r}")
        seen.add(name)
        for field in (
            "provides",
            "upgraded_provides",
            "advisory_provides",
            "bridge_provides",
            "requires_any_owned",
        ):
            if field in row:
                _strings(row[field], f"support.cards[{index}].{field}", allow_empty=True)


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"cannot load {label} from {path}: {error}") from error
    return _object(payload, label)


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CatalogError(f"{path} must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{path} must be a non-empty string")
    return value.strip()


def _strings(value: object, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    values = tuple(_text(item, f"{path}[]") for item in _array(value, path))
    if not allow_empty and not values:
        raise CatalogError(f"{path} must not be empty")
    if len(set(values)) != len(values):
        raise CatalogError(f"{path} must contain unique values")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = compile_catalog(
        args.graph, args.choices, args.support, args.parameters
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {payload['model']['offer_rows']} expert choices to {args.output}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CatalogError",
    "compile_catalog",
    "default_catalog_sha256",
    "load_default_catalog",
    "main",
]
