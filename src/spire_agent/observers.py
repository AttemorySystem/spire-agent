"""Non-fatal fan-out for logs and optional observers."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ContextEntry
from .ports import ObserverCollection, ObserverErrorSink, RunObserver


@dataclass(frozen=True, slots=True)
class ObserverFailure:
    observer_type: str
    error: str


class ObserverHub:
    """Publish immutable entries; observer failures never affect gameplay."""

    def __init__(
        self,
        observers: ObserverCollection = (),
        *,
        error_sink: ObserverErrorSink | None = None,
    ) -> None:
        self._observers = tuple(observers)
        self._error_sink = error_sink
        self._failures: list[ObserverFailure] = []

    @property
    def failures(self) -> tuple[ObserverFailure, ...]:
        return tuple(self._failures)

    def publish(self, entry: ContextEntry) -> None:
        for observer in self._observers:
            try:
                observer.on_entry(entry)
            except Exception as exc:  # Observers are explicitly non-critical.
                self._failures.append(
                    ObserverFailure(type(observer).__name__, str(exc))
                )
                if self._error_sink is not None:
                    try:
                        self._error_sink(observer, exc)
                    except Exception:
                        pass


__all__ = ["ObserverFailure", "ObserverHub", "RunObserver"]
