"""Version and fingerprint the Winning Path executable contract."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from hashlib import sha256
from importlib import resources
import json
from typing import Any

from spire_agent.contracts import frozen_mapping


PROTOCOL_VERSION = "2.3.0"
POLICY_ID = "ironclad.winning_path"
_PROFILES = {
    "IRONCLAD": ("ironclad_protocol.json", "ironclad_policy.json", POLICY_ID),
    "DEFECT": (
        "defect_protocol.json",
        "defect_policy.json",
        "defect.winning_path",
    ),
}
_IMPLEMENTATION_FILES = (
    "analysis.py",
    "card_policy.py",
    "catalog.py",
    "contracts.py",
    "evidence.py",
    "needs.py",
    "parameters.py",
    "picker.py",
    "plan.py",
    "preference.py",
    "resolver.py",
    "state.py",
    "template_search.py",
    "templates.py",
)


class ProtocolError(ValueError):
    pass


@lru_cache(maxsize=None)
def load_protocol(character: object = "IRONCLAD") -> Mapping[str, Any]:
    character = normalize_character(character)
    path = resources.files("spire_agent.tools.winning_path").joinpath(
        "data", _PROFILES[character][0]
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(value, character)
    return frozen_mapping(value)


@lru_cache(maxsize=None)
def protocol_sha256(character: object = "IRONCLAD") -> str:
    return canonical_sha256(load_protocol(character))


@lru_cache(maxsize=1)
def implementation_sha256() -> str:
    root, digest = resources.files("spire_agent.tools.winning_path"), sha256()
    for name in _IMPLEMENTATION_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(root.joinpath(name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_protocol(
    value: Mapping[str, Any], character: object = "IRONCLAD"
) -> None:
    character = normalize_character(character)
    _, policy_file, policy_id = _PROFILES[character]
    expected = {
        "schema_version", "protocol_version", "policy_id", "scope",
        "policy_file",
    }
    if set(value) != expected:
        raise ProtocolError("protocol has an unexpected schema")
    if value["schema_version"] != 1 or value["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError("protocol version mismatch")
    if (
        value["policy_id"] != policy_id
        or value["policy_file"] != f"data/{policy_file}"
    ):
        raise ProtocolError("protocol policy mismatch")
    if value["scope"] != {"character": character, "screen": "CARD_REWARD"}:
        raise ProtocolError("protocol scope mismatch")


def normalize_character(character: object) -> str:
    value = str(character or "").strip().upper()
    if value not in _PROFILES:
        raise ProtocolError(f"unsupported character {character!r}")
    return value


def policy_id(character: object = "IRONCLAD") -> str:
    return _PROFILES[normalize_character(character)][2]


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(encoded).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    return value


__all__ = [
    "POLICY_ID", "PROTOCOL_VERSION", "ProtocolError", "canonical_sha256",
    "implementation_sha256", "load_protocol", "normalize_character",
    "policy_id", "protocol_sha256", "validate_protocol",
]
