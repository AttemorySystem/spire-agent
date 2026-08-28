from __future__ import annotations

import unittest

from spire_agent.contracts import (
    AgentKind,
    ContextEntry,
    Continuation,
    ContinuationOperation,
    DecisionRequest,
    DecisionScope,
    GameState,
    ScreenState,
)
from spire_agent.subagents.combat import create_combat_agent as compose_combat_agent
from spire_agent.tools.mcts import (
    DefaultCombatTool,
    MCTSResult,
    resolve_selection,
)


def create_combat_agent(search):
    return compose_combat_agent(DefaultCombatTool(search))


class FakeSearch:
    def __init__(self, result: MCTSResult):
        self.result = result
        self.states = []

    def choose(self, state):
        self.states.append(state)
        return self.result


def combat_state(
    *,
    screen: str = "NONE",
    commands: tuple[str, ...] = ("play", "end"),
    choices: tuple[object, ...] = (),
    details: dict | None = None,
    current_action: str = "",
) -> GameState:
    return GameState(
        owner_hint=AgentKind.COMBAT,
        scope_id="combat-1",
        screen=ScreenState(
            type=screen,
            commands=commands,
            choices=choices,
            details=details or {},
            current_action=current_action,
        ),
        facts={"room_phase": "COMBAT"},
        combat={
            "hand": [{"id": "Strike_R", "name": "Strike"}],
            "monsters": [{"id": "JawWorm", "current_hp": 40}],
        },
    )


def request(
    state: GameState,
    continuation: Continuation | None = None,
) -> DecisionRequest:
    return DecisionRequest(
        state=state,
        scope=DecisionScope(AgentKind.COMBAT, "combat-1"),
        continuation=continuation,
        shared={},
        previous=ContextEntry(0, None, state, True),
    )


class CombatAgentTests(unittest.TestCase):
    def test_normal_combat_delegates_one_root_action_to_mcts(self):
        search = FakeSearch(
            MCTSResult("end", None, {"score": 0.0})
        )
        decision = create_combat_agent(search).decide(request(combat_state()))

        self.assertEqual(decision.command, "end")
        self.assertEqual(decision.source, "combat.mcts")
        self.assertEqual(len(search.states), 1)

    def test_echo_form_hologram_replans_its_second_selector(self):
        search = FakeSearch(MCTSResult("choose 1", None, {}))
        state = combat_state(
            screen="GRID",
            commands=("choose",),
            choices=("Strike", "Hologram+"),
            current_action="BetterDiscardPileToHandAction",
        )
        continuation = Continuation(
            AgentKind.COMBAT,
            "mcts_card_selection",
            "combat-1",
            expected_screens=("GRID",),
            data={
                "kind": "card_selection",
                "task": "HOLOGRAM",
                "cards": [],
                "completionCommand": None,
            },
        )

        decision = create_combat_agent(search).decide(
            request(state, continuation)
        )

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "combat.mcts")
        self.assertEqual(search.states, [state])

    def test_multi_card_follow_up_executes_one_confirmed_step_at_a_time(self):
        plan = {
            "kind": "card_selection",
            "task": "EXHAUST_MANY",
            "cards": [
                {"id": "Strike_R", "name": "Strike", "upgrades": 0, "sourceIndex": 0},
                {"id": "Defend_R", "name": "Defend", "upgrades": 0, "sourceIndex": 1},
            ],
            "completionCommand": "confirm",
        }
        search = FakeSearch(
            MCTSResult("play 1", plan, {})
        )
        agent = create_combat_agent(search)

        root = agent.decide(request(combat_state()))
        self.assertEqual(root.command, "play 1")
        self.assertEqual(root.continuation.operation, ContinuationOperation.SET)
        self.assertEqual(
            root.continuation.value.data["cards"][1]["sourceIndex"],
            1,
        )

        first_state = combat_state(
            screen="HAND_SELECT",
            commands=("choose", "confirm"),
            choices=("Strike", "Defend"),
            details={
                "hand": [
                    {"id": "Strike_R", "name": "Strike", "upgrades": 0},
                    {"id": "Defend_R", "name": "Defend", "upgrades": 0},
                ]
            },
        )
        first = agent.decide(request(first_state, root.continuation.value))
        self.assertEqual(first.command, "choose 0")
        self.assertEqual(first.continuation.operation, ContinuationOperation.SET)

        second_state = combat_state(
            screen="HAND_SELECT",
            commands=("choose", "confirm"),
            choices=("Defend",),
            details={
                "hand": [
                    {"id": "Defend_R", "name": "Defend", "upgrades": 0}
                ]
            },
        )
        second = agent.decide(request(second_state, first.continuation.value))
        self.assertEqual(second.command, "choose 0")
        self.assertEqual(second.continuation.operation, ContinuationOperation.SET)

        confirm_state = combat_state(
            screen="HAND_SELECT",
            commands=("confirm",),
        )
        confirm = agent.decide(request(confirm_state, second.continuation.value))
        self.assertEqual(confirm.command, "confirm")
        self.assertEqual(confirm.continuation.operation, ContinuationOperation.CLEAR)

    def test_single_card_follow_up_confirms_when_live_screen_requires_it(self):
        plan = {
            "kind": "card_selection",
            "task": "EXHAUST_ONE",
            "cards": [
                {
                    "id": "Offering",
                    "name": "Offering+",
                    "upgrades": 1,
                    "sourceIndex": 0,
                }
            ],
            "completionCommand": None,
        }
        search = FakeSearch(MCTSResult("play 1", plan, {}))
        agent = create_combat_agent(search)

        root = agent.decide(request(combat_state()))
        select_state = combat_state(
            screen="HAND_SELECT",
            commands=("choose",),
            choices=("Offering+",),
            details={
                "max_cards": 1,
                "selected": [],
                "hand": [
                    {"id": "Offering", "name": "Offering+", "upgrades": 1}
                ],
            },
        )
        selected = agent.decide(request(select_state, root.continuation.value))
        self.assertEqual(selected.command, "choose 0")
        self.assertEqual(selected.continuation.operation, ContinuationOperation.SET)
        self.assertEqual(selected.continuation.value.data["cards"], ())

        confirm_state = combat_state(
            screen="HAND_SELECT",
            commands=("confirm",),
            details={
                "max_cards": 1,
                "selected": [
                    {"id": "Offering", "name": "Offering+", "upgrades": 1}
                ],
                "hand": [],
            },
        )
        confirmed = agent.decide(
            request(confirm_state, selected.continuation.value)
        )

        self.assertEqual(confirmed.command, "confirm")
        self.assertEqual(
            confirmed.continuation.operation,
            ContinuationOperation.CLEAR,
        )

    def test_seek_plus_follow_up_selects_two_distinct_cards(self):
        plan = {
            "kind": "card_selection",
            "task": "SEEK",
            "cards": [
                {"id": "Defend_B", "name": "Defend", "sourceIndex": 0},
                {"id": "Dualcast", "name": "Dualcast", "sourceIndex": 1},
            ],
            "completionCommand": None,
        }
        agent = create_combat_agent(FakeSearch(MCTSResult("play 1", plan, {})))
        root = agent.decide(request(combat_state()))
        cards = (
            {"id": "Defend_B", "name": "Defend", "uuid": "defend-1"},
            {"id": "Dualcast", "name": "Dualcast", "uuid": "dualcast-1"},
        )
        first_state = combat_state(
            screen="GRID",
            commands=("choose",),
            choices=("defend", "dualcast"),
            details={"cards": cards, "selected_cards": [], "num_cards": 2},
        )

        first = agent.decide(request(first_state, root.continuation.value))
        self.assertEqual(first.command, "choose 0")

        second_state = combat_state(
            screen="GRID",
            commands=("choose",),
            choices=("defend", "dualcast"),
            details={
                "cards": cards,
                "selected_cards": [cards[0]],
                "num_cards": 2,
            },
        )
        second = agent.decide(request(second_state, first.continuation.value))

        self.assertEqual(second.command, "choose 1")
        self.assertEqual(
            second.continuation.value.data["cards"],
            (),
        )

    def test_echo_form_recycle_confirms_between_two_selectors(self):
        plan = {
            "kind": "card_selection",
            "task": "RECYCLE",
            "cards": [
                {"id": "Creative AI", "name": "Creative AI", "sourceIndex": 5},
            ],
            "completionCommand": None,
        }
        state = combat_state(
            screen="HAND_SELECT",
            commands=("confirm",),
            details={"selected": [{"id": "Loop", "name": "Loop"}]},
        )
        command, remaining = resolve_selection(state, plan)

        self.assertEqual(command, "confirm")
        self.assertEqual(remaining, plan)

    def test_generated_card_page_passes_explicit_task_to_mcts(self):
        search = FakeSearch(
            MCTSResult("choose 2", None, {})
        )
        state = combat_state(
            screen="CARD_REWARD",
            commands=("choose", "skip"),
            choices=("A", "B", "C"),
            current_action="DiscoveryAction",
        )

        decision = create_combat_agent(search).decide(request(state))

        self.assertEqual(decision.command, "choose 2")
        self.assertEqual(search.states[0].screen.current_action, "DiscoveryAction")

    def test_changed_generated_candidates_are_searched_again(self):
        continuation = Continuation(
            AgentKind.COMBAT,
            "mcts_card_selection",
            "combat-1",
            expected_screens=("CARD_REWARD",),
            data={
                "kind": "card_selection",
                "task": "DISCOVERY",
                "cards": [{"id": "PanicButton", "name": "Panic Button"}],
                "completionCommand": None,
            },
        )
        search = FakeSearch(MCTSResult("choose 1", None, {}))
        state = combat_state(
            screen="CARD_REWARD",
            commands=("choose",),
            choices=("Madness", "Master of Strategy", "Mayhem"),
            current_action="DiscoveryAction",
        )

        decision = create_combat_agent(search).decide(
            request(state, continuation)
        )

        self.assertEqual(decision.command, "choose 1")
        self.assertEqual(decision.source, "combat.mcts")
        self.assertEqual(search.states, [state])

    def test_single_generated_card_is_a_fast_path(self):
        search = FakeSearch(
            MCTSResult("choose 0", None, {})
        )
        state = combat_state(
            screen="CARD_REWARD",
            commands=("choose",),
            choices=("only",),
            current_action="DiscoveryAction",
        )

        decision = create_combat_agent(search).decide(request(state))

        self.assertEqual(decision.command, "choose 0")
        self.assertEqual(decision.source, "combat.single_choice")
        self.assertEqual(search.states, [])

    def test_fresh_gambling_chip_selection_is_owned_by_mcts(self):
        search = FakeSearch(MCTSResult("confirm", None, {}))
        state = combat_state(
            screen="HAND_SELECT",
            commands=("choose", "confirm"),
            choices=("Strike",),
            details={"can_pick_zero": True, "max_cards": 10},
            current_action="GamblingChipAction",
        )

        decision = create_combat_agent(search).decide(request(state))

        self.assertEqual(decision.command, "confirm")
        self.assertEqual(decision.source, "combat.mcts")
        self.assertEqual(search.states, [state])


if __name__ == "__main__":
    unittest.main()
