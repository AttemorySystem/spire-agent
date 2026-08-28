"""Build one complete Winning Path decision artifact."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spire_agent.contracts import DecisionRequest

from .catalog import default_catalog_sha256, load_default_catalog
from .evidence import analyze_candidate_evidence
from .needs import encounter_model_sha256
from .parameters import parameters_sha256, policy_sha256
from .protocol import (
    PROTOCOL_VERSION,
    canonical_sha256,
    implementation_sha256,
    normalize_character,
    protocol_sha256,
)
from .resolver import resolve
from .state import project_state


def analyze(
    request: DecisionRequest,
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = project_state(request)
    character = normalize_character(state.run.get("character"))
    if catalog is None:
        catalog = load_default_catalog(character)
        catalog_hash = default_catalog_sha256(character)
    else:
        catalog_hash = canonical_sha256(catalog)
    evidence = analyze_candidate_evidence(state, catalog)
    resolution = resolve(state, catalog, evidence)
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "mode": "LIVE_POLICY",
        "fingerprints": {
            "protocol_sha256": protocol_sha256(character),
            "catalog_sha256": catalog_hash,
            "encounter_model_sha256": encounter_model_sha256(character),
            "policy_sha256": policy_sha256(character),
            "parameters_sha256": parameters_sha256(character),
            "implementation_sha256": implementation_sha256(),
            "state_sha256": state.sha256,
        },
        "state": state.as_dict(),
        "deck_plan": _mapping(evidence.get("skip")).get("deck_plan"),
        "target_plan": evidence.get("target_plan"),
        "need_profile": evidence.get("need_profile"),
        "candidate_evidence": evidence,
        "resolution": resolution,
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["analyze"]
