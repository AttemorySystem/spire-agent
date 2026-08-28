"""Framework boundary errors.

Domain failures should define their own errors beside their implementation.
These errors only describe violations of the frozen system contracts.
"""


class FrameworkError(RuntimeError):
    """Base class for Spire Agent framework contract violations."""


class ContextError(FrameworkError):
    """A command/state transaction violated GameContext ordering."""


class RegistryError(FrameworkError):
    """The SubAgent registry is missing or duplicates an owner."""


class ActionValidationError(FrameworkError):
    """A SubAgent returned a command unavailable in the current state."""


__all__ = [
    "ActionValidationError",
    "ContextError",
    "FrameworkError",
    "RegistryError",
]
