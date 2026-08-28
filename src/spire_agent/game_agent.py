"""Small non-LLM application root for one complete game run."""

from __future__ import annotations

from .context import GameContext
from .contracts import ContextView
from .observers import ObserverHub
from .ports import ActionValidator, DecisionControl, DecisionProvider, GameSession


class GameAgent:
    """Execute the fixed state -> route -> decide -> command transaction.

    It contains no page, card, map, combat, reward, fast-path, MCTS, or prompt
    logic.  Those decisions belong entirely to the selected SubAgent.
    """

    def __init__(
        self,
        *,
        session: GameSession,
        context: GameContext,
        decisions: DecisionProvider,
        validator: ActionValidator,
        observers: ObserverHub | None = None,
        control: DecisionControl | None = None,
    ) -> None:
        self._session = session
        self._context = context
        self._decisions = decisions
        self._validator = validator
        self._observers = observers or ObserverHub()
        self._control = control

    def run(self) -> ContextView:
        """Run until the normalized game state declares itself terminal."""

        if self._context.started:
            raise RuntimeError("GameAgent instances may run only once")
        try:
            initial = self._context.start(self._session.reset())
            self._observers.publish(initial)
            while not self._context.current_state.terminal:
                if self._control is not None:
                    if self._control.before_decision(self._context.view()):
                        refreshed = self._session.refresh()
                        if refreshed.changed:
                            self._observers.publish(
                                self._context.resync(refreshed.state)
                            )
                            if refreshed.state.terminal:
                                break
                view = self._context.view()
                routed = self._decisions.decide(view)
                self._validator.validate(view.state, routed.decision)
                self._context.stage(routed)
                result = self._session.execute(routed.decision.command)
                entry = self._context.confirm(result)
                self._observers.publish(entry)
            return self._context.view()
        finally:
            self._session.close()


__all__ = ["GameAgent"]
