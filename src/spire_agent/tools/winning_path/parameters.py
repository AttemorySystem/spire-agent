"""Load the one reviewed parameter file for each character."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
import json
from typing import Any

from spire_agent.contracts import frozen_mapping

from .protocol import (
    PROTOCOL_VERSION,
    canonical_sha256,
    load_protocol,
    normalize_character,
    policy_id,
)


_TEMPLATE_LEVELS = ("NONE", "REACHABLE_ENTRY", "COMMITTED_PROGRESS", "CORE_ACTIVATION")
_TRANSITION_LEVELS = ("NONE", "OPEN_NEED", "CRITICAL_NEED", "BLOCKING_NEED")
_FIXED = {
    "resolver_order": [
        "hard constraints",
        "blocking survival",
        "template progress",
        "transition need; expert ranks only its best frontier",
        "standalone expert experience",
        "take or skip",
        "bounded LLM",
    ],
    "template_levels": list(_TEMPLATE_LEVELS),
    "transition_levels": list(_TRANSITION_LEVELS),
    "no_positive_evidence": "SKIP",
    "multiple_unranked_positive": "LLM",
    "singing_bowl": "replace skip only",
    "online_mcts_authority": False,
}
_CHANGE_CONTROL = {
    "parameter_families": ["templates", "transition", "expert", "authority"],
    "new_family_or_field_requires_review": True,
    "resolver_change_requires_review": True,
    "one_knowledge_mutation_per_candidate": True,
}


@lru_cache(maxsize=None)
def load_parameters(character: object = "IRONCLAD") -> Mapping[str, Any]:
    """Return the complete decision-tuning surface for one character."""

    character = normalize_character(character)
    protocol = load_protocol(character)
    path = resources.files("spire_agent.tools.winning_path").joinpath(
        *str(protocol["policy_file"]).split("/")
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {
        "schema_version", "protocol_version", "policy_id", "scope",
        "templates", "transition", "expert", "authority",
    }:
        raise ValueError(f"{path.name} has an unexpected schema")
    if (
        value["schema_version"] != 1
        or value["protocol_version"] != PROTOCOL_VERSION
        or value["policy_id"] != policy_id(character)
        or value["scope"] != {"character": character, "screen": "CARD_REWARD"}
    ):
        raise ValueError(f"{path.name} identity mismatch")
    if set(value["templates"]) != {"distance", "knowledge"}:
        raise ValueError("template parameters must contain distance and knowledge")
    if set(value["templates"]["distance"]) != {
        "algorithm", "state_order", "progress_order", "future_choices",
        "offer_probability",
    }:
        raise ValueError("unreviewed template distance field")
    if set(value["transition"]) != {
        "capabilities", "target_rules", "target_pools", "capability_aliases",
        "card_capabilities", "encounter_requirements", "foundation_needs",
    }:
        raise ValueError("unreviewed transition parameter field")
    if set(value["expert"]) != {
        "context_order", "deck_size_limits", "positive_z", "direct_z",
    }:
        raise ValueError("unreviewed expert parameter field")
    if set(value["authority"]) != {"template", "transition"}:
        raise ValueError("unreviewed authority parameter field")
    expert, authority = value["expert"], value["authority"]
    if not 0 <= float(expert["positive_z"]) < float(expert["direct_z"]):
        raise ValueError("expert positive_z must be below direct_z")
    if authority["template"] not in _TEMPLATE_LEVELS:
        raise ValueError("unknown template authority")
    if authority["transition"] not in _TRANSITION_LEVELS:
        raise ValueError("unknown transition authority")
    return frozen_mapping(value)


@lru_cache(maxsize=None)
def load_policy(character: object = "IRONCLAD") -> Mapping[str, Any]:
    """Expose parameters plus fixed kernel semantics to consumers."""

    character = normalize_character(character)
    value = load_parameters(character)
    catalog = "ironclad_catalog" if character == "IRONCLAD" else "defect_catalog"
    return frozen_mapping({
        "schema_version": 1,
        "policy_id": value["policy_id"],
        "parameters": {
            "templates": {
                "data": f"{catalog}.knowledge",
                "editable": ["knowledge", "distance"],
                "distance": value["templates"]["distance"],
            },
            "transition": {
                "data": "parameters.transition",
                "editable": [
                    "target pools", "encounter requirements",
                    "capability aliases", "card capabilities", "foundation needs",
                ],
                "capabilities": value["transition"]["capabilities"],
                "target_rules": value["transition"]["target_rules"],
            },
            "expert": {
                "data": f"{catalog}.derived.expert_preferences",
                "observation": (
                    "picked beats other offers and skip; skip beats all offers; "
                    "singing bowl excluded"
                ),
                "score": "(wins-losses)/sqrt(wins+losses)",
                **dict(value["expert"]),
            },
            "authority": value["authority"],
        },
        "fixed": _FIXED,
        "change_control": _CHANGE_CONTROL,
    })


@lru_cache(maxsize=None)
def parameters_sha256(character: object = "IRONCLAD") -> str:
    return canonical_sha256(load_parameters(character))


@lru_cache(maxsize=None)
def policy_sha256(character: object = "IRONCLAD") -> str:
    return canonical_sha256(load_policy(character))


__all__ = [
    "load_parameters", "load_policy", "parameters_sha256", "policy_sha256",
]
