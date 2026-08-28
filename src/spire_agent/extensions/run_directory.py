"""Bind durable run artifacts to the canonical Slay the Spire seed."""

from __future__ import annotations

from pathlib import Path
import re
from threading import Lock

class RunDirectoryError(RuntimeError):
    """A canonical run directory cannot be created or resolved safely."""


class RunDirectory:
    """A run path that becomes available after gym-sts returns ``sts_seed``."""

    __slots__ = ("_lock", "_path", "_root", "_seed")

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = Lock()
        self._seed: str | None = None
        self._path: Path | None = None

    @classmethod
    def open(cls, path: str | Path) -> "RunDirectory":
        """Open an existing run only when replay was requested explicitly."""

        resolved = Path(path)
        if not resolved.is_dir():
            raise RunDirectoryError(f"replay run directory does not exist: {resolved}")
        result = cls(resolved.parent)
        seed = resolved.name.strip().upper()
        if not re.fullmatch(r"[0-9A-Z]+", seed):
            raise RunDirectoryError(
                f"invalid replay run directory seed: {resolved.name!r}"
            )
        result._seed = seed
        result._path = resolved
        return result

    @property
    def seed(self) -> str:
        with self._lock:
            if self._seed is None:
                raise RunDirectoryError("run directory has not been bound to sts_seed")
            return self._seed

    @property
    def path(self) -> Path:
        with self._lock:
            if self._path is None:
                raise RunDirectoryError("run directory has not been bound to sts_seed")
            return self._path

    def bind(self, sts_seed: object) -> Path:
        seed = str(sts_seed).strip().upper()
        if not re.fullmatch(r"[0-9A-Z]+", seed):
            raise RunDirectoryError(f"unsafe canonical sts_seed: {sts_seed!r}")

        with self._lock:
            if self._seed is not None:
                if self._seed != seed:
                    raise RunDirectoryError(
                        f"run directory is already bound to {self._seed}, not {seed}"
                    )
                if self._path is None:
                    raise AssertionError("bound run directory has no path")
                return self._path

            self._root.mkdir(parents=True, exist_ok=True)
            path = self._root / seed
            try:
                path.mkdir()
            except FileExistsError as error:
                raise RunDirectoryError(
                    f"run directory already exists for sts_seed {seed}: {path}"
                ) from error
            self._seed = seed
            self._path = path
            return path

__all__ = ["RunDirectory", "RunDirectoryError"]
