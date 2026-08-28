"""Framework-level validation that is independent of screen semantics."""

from __future__ import annotations

from .contracts import Decision, GameState
from .errors import ActionValidationError


class AvailableCommandValidator:
    """Require the returned command family to be exposed by the game."""

    def validate(self, state: GameState, decision: Decision) -> None:
        available = set(state.screen.commands)
        if decision.command_family not in available:
            raise ActionValidationError(
                f"SubAgent command {decision.command!r} is unavailable on "
                f"{state.screen.type}; available families={sorted(available)!r}"
            )


__all__ = ["AvailableCommandValidator"]
