"""Adapt one CLI seed to the two reset modes supported by gym-sts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


_STS_SEED_CHARACTERS = frozenset("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")


class SeedMode(str, Enum):
    GYM_RANDOM = "gym_random"
    STS_EXACT = "sts_exact"


@dataclass(frozen=True, slots=True)
class SeedRequest:
    input_value: str
    mode: SeedMode
    reset_kwargs: Mapping[str, Any]

    @classmethod
    def parse(cls, value: object) -> "SeedRequest":
        raw = str(value).strip()
        if not raw:
            raise ValueError("seed must not be empty")
        if raw.isascii() and raw.isdecimal():
            return cls(
                input_value=raw,
                mode=SeedMode.GYM_RANDOM,
                reset_kwargs=MappingProxyType({"seed": int(raw)}),
            )

        return cls.exact(raw)

    @classmethod
    def exact(cls, value: object) -> "SeedRequest":
        """Build an exact in-game seed request, including all-digit seeds."""

        raw = str(value).strip()
        if not raw:
            raise ValueError("seed must not be empty")
        canonical = raw.upper()
        invalid = sorted(set(canonical) - _STS_SEED_CHARACTERS)
        if invalid:
            raise ValueError(
                "Slay the Spire seed contains illegal character(s): "
                + ", ".join(repr(item) for item in invalid)
            )
        return cls(
            input_value=raw,
            mode=SeedMode.STS_EXACT,
            reset_kwargs=MappingProxyType(
                {"options": MappingProxyType({"sts_seed": canonical})}
            ),
        )


__all__ = ["SeedMode", "SeedRequest"]
