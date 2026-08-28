from __future__ import annotations

from copy import deepcopy
import unittest

from spire_agent.tools.game_stability import (
    GameStabilityError,
    StabilityPolicy,
    settle_game_state,
    stable_boundary_key,
)


def observation(
    screen: str,
    *,
    commands: tuple[str, ...] = ("choose", "wait", "state"),
    choices: tuple[object, ...] = (),
    phase: str = "COMPLETE",
    deck: list[dict] | None = None,
    screen_state: dict | None = None,
    combat: dict | None = None,
    **game_facts,
) -> dict:
    game = {
        "screen_type": screen,
        "room_phase": phase,
        "choice_list": list(choices),
        "screen_state": screen_state or {},
        "deck": deepcopy(deck or []),
        "transition_pending": False,
        **game_facts,
    }
    if combat is not None:
        game["combat_state"] = combat
    if phase == "COMBAT":
        game.setdefault("action_phase", "WAITING_ON_USER")
        game.setdefault("current_action", None)
    return {
        "available_commands": list(commands),
        "ready_for_command": True,
        "in_game": True,
        "game_state": game,
    }


class Driver:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def _next(self):
        if not self.responses:
            raise AssertionError("unexpected stability refresh")
        return self.responses.pop(0)

    def read_state(self):
        self.calls.append("state")
        return self._next()

    def wait_frames(self, frames: int):
        self.calls.append(f"wait {frames}")
        return self._next()


class GameStabilityToolTests(unittest.TestCase):
    def setUp(self):
        self.policy = StabilityPolicy(
            poll_interval=0,
            timeout=10,
            max_refreshes=4,
        )

    def settle(self, before, after, command, driver):
        return settle_game_state(
            before,
            after,
            command,
            read_state=driver.read_state,
            wait_frames=driver.wait_frames,
            policy=self.policy,
        )

    def test_unchanged_acknowledgement_waits_for_visible_transition(self):
        before = observation("REST", choices=("rest", "smith"))
        stale = deepcopy(before)
        grid = observation(
            "GRID",
            choices=("Strike", "Bash"),
            screen_state={"for_upgrade": True},
        )
        driver = Driver(grid)

        result = self.settle(before, stale, "choose 1", driver)

        self.assertIs(result, grid)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_room_exit_confirms_map_command_handoff(self):
        before = observation(
            "COMBAT_REWARD", commands=("proceed", "wait", "state")
        )
        map_screen = observation(
            "MAP",
            commands=("choose", "return", "wait", "state"),
            choices=("x=4",),
        )
        settled = deepcopy(map_screen)
        map_screen["game_state"]["transition_pending"] = True
        driver = Driver(map_screen, settled)

        result = self.settle(before, map_screen, "proceed", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])

    def test_combat_queue_requires_barrier_and_semantic_completion(self):
        before = observation(
            "CARD_REWARD",
            phase="COMBAT",
            combat={"turn": 1},
        )
        executing = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={
                "turn": 1,
                "monsters": [
                    {"name": "Guardian", "current_hp": 200, "move_id": 3}
                ],
            },
            action_phase="EXECUTING_ACTIONS",
            current_action="DiscoveryAction",
        )
        settled = deepcopy(executing)
        settled["game_state"]["action_phase"] = "WAITING_ON_USER"
        settled["game_state"]["current_action"] = None
        driver = Driver(executing, settled)

        result = self.settle(before, executing, "choose 2", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])

    def test_waiting_phase_is_not_stable_while_combat_actions_are_pending(self):
        before = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={
                "turn": 1,
                "player": {"energy": 3},
                "monsters": [{"current_hp": 40, "move_id": 1}],
            },
        )
        pending = deepcopy(before)
        pending["game_state"]["transition_pending"] = True
        settled = deepcopy(before)
        settled["game_state"]["combat_state"]["player"]["energy"] = 2
        driver = Driver(pending, settled)

        result = self.settle(before, pending, "play 1", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])

    def test_lethal_play_waits_until_reward_screen(self):
        before = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={
                "turn": 8,
                "monsters": [
                    {"name": "Lagavulin", "current_hp": 3, "move_id": 1}
                ],
            },
            action_phase="WAITING_ON_USER",
        )
        reward = observation("COMBAT_REWARD", choices=("Gold",))
        driver = Driver(reward)

        result = self.settle(before, deepcopy(before), "play 1 0", driver)

        self.assertIs(result, reward)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_terminal_state_needs_no_barrier_or_ready_signal(self):
        before = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={"turn": 14, "monsters": [{"current_hp": 34}]},
        )
        terminal = observation(
            "GAME_OVER",
            phase="COMBAT",
            commands=("proceed", "wait", "state"),
            combat={"turn": 14, "player": {"current_hp": 0}},
            screen_state={"victory": False},
        )
        terminal["ready_for_command"] = False
        driver = Driver()

        result = self.settle(before, terminal, "end", driver)

        self.assertIs(result, terminal)
        self.assertEqual(driver.calls, [])

    def test_pending_permanent_effect_is_not_a_decision_boundary(self):
        before = observation("EVENT", choices=("accept", "leave"))
        pending = observation("EVENT", choices=("leave",))
        pending["game_state"]["transition_pending"] = True
        pending["game_state"]["pending_effects"] = ["ShowCardAndObtainEffect"]
        committed = deepcopy(pending)
        committed["game_state"]["transition_pending"] = False
        committed["game_state"]["deck"] = [{"id": "Decay", "name": "Decay"}]
        driver = Driver(committed)

        result = self.settle(before, pending, "choose 0", driver)

        self.assertIs(result, committed)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_card_reward_waits_until_card_enters_deck(self):
        deck = [{"id": "Bash", "name": "Bash"}]
        before = observation(
            "CARD_REWARD",
            choices=("Berserk", "Impervious"),
            deck=deck,
            screen_state={
                "cards": [
                    {"id": "Berserk", "name": "Berserk"},
                    {"id": "Impervious", "name": "Impervious"},
                ]
            },
        )
        early = observation("COMBAT_REWARD", deck=deck)
        committed = observation(
            "COMBAT_REWARD",
            deck=[*deck, {"id": "Impervious", "name": "Impervious"}],
        )
        driver = Driver(committed)

        result = self.settle(before, early, "choose 1", driver)

        self.assertIs(result, committed)

    def test_new_recording_waits_for_library_card_before_exposing_event(self):
        deck = [{"id": "Bash", "name": "Bash"}]
        before = observation(
            "GRID",
            choices=("Flex", "Shrug It Off"),
            deck=deck,
            screen_state={"num_cards": 1},
        )
        early = observation("EVENT", choices=("leave",), deck=deck)
        committed = observation(
            "EVENT",
            choices=("leave",),
            deck=[*deck, {"id": "Shrug It Off", "name": "Shrug It Off"}],
        )
        driver = Driver(committed)

        result = self.settle(before, early, "choose 1", driver)

        self.assertIs(result, committed)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_transform_waits_until_deck_size_is_restored(self):
        deck = [{"name": f"Card {index}"} for index in range(11)]
        before = observation(
            "GRID",
            deck=deck,
            screen_state={"num_cards": 2, "for_transform": True},
        )
        pending = observation("EVENT", deck=deck[:9])
        still_pending = deepcopy(pending)
        committed = observation("EVENT", deck=deck)
        driver = Driver(still_pending, committed)

        result = self.settle(before, pending, "choose 0", driver)

        self.assertIs(result, committed)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])

    def test_neow_transform_ignores_incorrect_grid_flag(self):
        deck = [{"name": f"Card {index}"} for index in range(11)]
        before = observation(
            "GRID",
            deck=deck,
            room_type="NeowRoom",
            screen_state={"num_cards": 2, "for_transform": False},
        )
        pending = observation("EVENT", deck=deck[:9])
        committed = observation("EVENT", deck=deck)
        driver = Driver(committed)

        result = self.settle(before, pending, "choose 1", driver)

        self.assertIs(result, committed)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_grid_purge_allows_deck_to_shrink(self):
        deck = [{"name": "Strike"}, {"name": "Defend"}]
        before = observation(
            "GRID", deck=deck, screen_state={"for_purge": True}
        )
        purged = observation("EVENT", deck=deck[1:])
        driver = Driver(purged)

        result = self.settle(before, purged, "choose 0", driver)

        self.assertIs(result, purged)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_open_combat_selection_survives_the_command_barrier(self):
        before = observation(
            "NONE",
            phase="COMBAT",
            commands=("potion", "wait", "state"),
            combat={"turn": 1, "monsters": [{"current_hp": 40, "move_id": 1}]},
        )
        grid = observation(
            "GRID",
            phase="COMBAT",
            choices=("Shrug It Off", "Carnage"),
            combat={"turn": 1, "monsters": [{"current_hp": 40, "move_id": 1}]},
            action_phase="EXECUTING_ACTIONS",
            current_action="BetterDiscardPileToHandAction",
        )
        driver = Driver(grid)

        result = self.settle(before, grid, "potion use 0", driver)

        self.assertIs(result, grid)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_single_combat_grid_choice_waits_for_automatic_close(self):
        before = observation(
            "GRID",
            phase="COMBAT",
            choices=("Defend+", "Uppercut+", "Wound"),
            combat={"turn": 10, "monsters": [{"current_hp": 290, "move_id": 5}]},
            action_phase="EXECUTING_ACTIONS",
            current_action="DiscardPileToTopOfDeckAction",
        )
        transient = observation(
            "GRID",
            phase="COMBAT",
            choices=("Defend+", "Wound", "Headbutt+"),
            combat={"turn": 10, "monsters": [{"current_hp": 290, "move_id": 5}]},
            action_phase="EXECUTING_ACTIONS",
            current_action="DiscardPileToTopOfDeckAction",
        )
        settled = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={"turn": 10, "monsters": [{"current_hp": 290, "move_id": 5}]},
            action_phase="WAITING_ON_USER",
            current_action="",
        )
        driver = Driver(settled)

        result = self.settle(before, transient, "choose 1", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_multi_card_combat_grid_remains_a_decision_boundary(self):
        before = observation(
            "GRID",
            phase="COMBAT",
            choices=("Strike", "Defend", "Bash"),
            combat={"turn": 3, "monsters": [{"current_hp": 40, "move_id": 1}]},
        )
        remaining = observation(
            "GRID",
            phase="COMBAT",
            choices=("Defend", "Bash"),
            combat={"turn": 3, "monsters": [{"current_hp": 40, "move_id": 1}]},
        )
        waited = deepcopy(remaining)
        driver = Driver(waited)

        result = self.settle(before, remaining, "choose 0", driver)

        self.assertIs(result, waited)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_committed_combat_selection_keeps_temporary_cost_boundary(self):
        before = observation(
            "GRID",
            phase="COMBAT",
            choices=("Carnage",),
            combat={"turn": 1, "monsters": [{"current_hp": 40, "move_id": 1}]},
        )
        committed = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={
                "turn": 1,
                "hand": [{"name": "Carnage", "cost": 0}],
                "monsters": [{"current_hp": 40, "move_id": 1}],
            },
            action_phase="WAITING_ON_USER",
            current_action="",
        )
        driver = Driver(committed)

        result = self.settle(before, committed, "choose 0", driver)

        self.assertIs(result, committed)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_pandora_confirmation_waits_for_all_replacements(self):
        reduced = [{"name": f"Card {index}"} for index in range(21)]
        replacements = [{"name": f"Replacement {index}"} for index in range(9)]
        before = observation(
            "GRID",
            deck=reduced,
            screen_state={
                "cards": replacements,
                "selected_cards": [],
                "confirm_up": True,
                "for_transform": False,
            },
            relics=[{"name": "Pandora's Box"}],
        )
        pending = observation(
            "CHEST", deck=reduced, relics=[{"name": "Pandora's Box"}]
        )
        settled = observation(
            "CHEST",
            deck=[*reduced, *replacements],
            relics=[{"name": "Pandora's Box"}],
        )
        driver = Driver(pending, settled)

        result = self.settle(before, pending, "confirm", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])

    def test_cursed_key_chest_waits_for_curse_in_deck(self):
        deck = [{"name": "Strike"}] * 10
        before = observation(
            "CHEST",
            choices=("open",),
            deck=deck,
            room_type="TreasureRoom",
            relics=[{"name": "Cursed Key"}],
        )
        early = observation("COMBAT_REWARD", deck=deck)
        settled = observation("COMBAT_REWARD", deck=[*deck, {"name": "Doubt"}])
        driver = Driver(settled)

        result = self.settle(before, early, "choose 0", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_cursed_key_does_not_expect_a_curse_from_boss_chest(self):
        deck = [{"name": "Strike"}] * 10
        before = observation(
            "CHEST",
            choices=("open",),
            deck=deck,
            room_type="BossTreasureRoom",
            relics=[{"name": "Cursed Key"}],
        )
        reward = observation("BOSS_REWARD", deck=deck)
        driver = Driver(reward)

        result = self.settle(before, reward, "choose 0", driver)

        self.assertIs(result, reward)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_omamori_charge_prevents_cursed_key_deck_growth(self):
        deck = [{"name": "Strike"}] * 10
        before = observation(
            "CHEST",
            choices=("open",),
            deck=deck,
            room_type="TreasureRoom",
            relics=[
                {"name": "Cursed Key", "counter": -1},
                {"name": "Omamori", "counter": 1},
            ],
        )
        reward = observation("COMBAT_REWARD", deck=deck)
        driver = Driver(reward)

        result = self.settle(before, reward, "choose 0", driver)

        self.assertIs(result, reward)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_depleted_omamori_still_waits_for_cursed_key_curse(self):
        deck = [{"name": "Strike"}] * 10
        before = observation(
            "CHEST",
            choices=("open",),
            deck=deck,
            room_type="TreasureRoom",
            relics=[
                {"name": "Cursed Key", "counter": -1},
                {"name": "Omamori", "counter": 0},
            ],
        )
        early = observation("COMBAT_REWARD", deck=deck)
        settled = observation("COMBAT_REWARD", deck=[*deck, {"name": "Doubt"}])
        driver = Driver(settled)

        result = self.settle(before, early, "choose 0", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_shop_card_purchase_waits_for_card_in_deck(self):
        deck = [{"name": "Strike"}] * 10
        before = observation(
            "SHOP_SCREEN",
            choices=("Feel No Pain", "Anchor", "purge"),
            deck=deck,
            screen_state={
                "cards": [{"name": "Feel No Pain"}],
                "relics": [{"name": "Anchor"}],
            },
        )
        early = observation("SHOP_SCREEN", choices=("Anchor", "purge"), deck=deck)
        settled = observation(
            "SHOP_SCREEN",
            choices=("Anchor", "purge"),
            deck=[*deck, {"name": "Feel No Pain"}],
        )
        driver = Driver(settled)

        result = self.settle(before, early, "choose 0", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_shop_relic_purchase_does_not_expect_deck_growth(self):
        deck = [{"name": "Strike"}] * 10
        before = observation(
            "SHOP_SCREEN",
            choices=("Feel No Pain", "Anchor", "purge"),
            deck=deck,
            screen_state={
                "cards": [{"name": "Feel No Pain"}],
                "relics": [{"name": "Anchor"}],
            },
        )
        after = observation(
            "SHOP_SCREEN", choices=("Feel No Pain", "purge"), deck=deck
        )
        driver = Driver(after)

        result = self.settle(before, after, "choose 1", driver)

        self.assertIs(result, after)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_pandora_does_not_turn_upgrade_confirmation_into_transforms(self):
        deck = [{"name": f"Card {index}"} for index in range(20)]
        before = observation(
            "GRID",
            deck=deck,
            screen_state={
                "cards": deck,
                "selected_cards": [{"name": "Bash"}],
                "confirm_up": True,
                "for_upgrade": True,
            },
            relics=[{"name": "Pandora's Box"}],
        )
        rest = observation(
            "REST", deck=deck, relics=[{"name": "Pandora's Box"}]
        )
        driver = Driver(rest)

        result = self.settle(before, rest, "confirm", driver)

        self.assertIs(result, rest)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_combat_entry_waits_for_numeric_move_ids(self):
        before = observation("MAP", choices=("x=1",))
        transient = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={
                "turn": 1,
                "monsters": [{"name": "Cultist", "current_hp": 50}],
            },
            action_phase="WAITING_ON_USER",
        )
        settled = deepcopy(transient)
        settled["game_state"]["combat_state"]["monsters"][0]["move_id"] = 3
        driver = Driver(transient, settled)

        result = self.settle(before, transient, "choose 0", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])

    def test_half_dead_enemy_is_a_valid_combat_boundary(self):
        before = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={
                "turn": 6,
                "hand": [{"id": "Strike_R", "name": "Strike"}],
                "monsters": [
                    {
                        "name": "Awakened One",
                        "current_hp": 1,
                        "move_id": 3,
                    }
                ],
            },
            action_phase="WAITING_ON_USER",
        )
        dormant = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end", "wait", "state"),
            combat={
                "turn": 6,
                "monsters": [
                    {
                        "name": "Awakened One",
                        "current_hp": 0,
                        "is_gone": True,
                        "half_dead": True,
                        "move_id": 3,
                    }
                ],
            },
            action_phase="WAITING_ON_USER",
        )
        driver = Driver(dormant)

        result = self.settle(before, dormant, "play 1 0", driver)

        self.assertIs(result, dormant)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_boundary_key_ignores_object_uuid_but_not_gameplay_state(self):
        first = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end"),
            combat={"hand": [{"id": "Strike_R", "uuid": "one"}]},
        )
        rebuilt = deepcopy(first)
        rebuilt["game_state"]["combat_state"]["hand"][0]["uuid"] = "two"
        changed = deepcopy(rebuilt)
        changed["game_state"]["combat_state"]["hand"][0]["cost"] = 0

        self.assertEqual(stable_boundary_key(first), stable_boundary_key(rebuilt))
        self.assertNotEqual(
            stable_boundary_key(rebuilt),
            stable_boundary_key(changed),
        )

        damaged = deepcopy(rebuilt)
        damaged["game_state"]["current_hp"] = 42
        self.assertNotEqual(
            stable_boundary_key(rebuilt),
            stable_boundary_key(damaged),
        )

    def test_boundary_key_covers_defect_orbs_history_and_dynamic_cards(self):
        first = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end"),
            combat={
                "turn": 2,
                "powers_played_this_combat": 1,
                "lightning_channeled_this_combat": 2,
                "frost_channeled_this_combat": 1,
                "emotion_chip_pending": False,
                "hand": [
                    {"id": "Gash", "name": "Claw", "base_damage": 5}
                ],
                "player": {
                    "energy": 3,
                    "orb_slots": 3,
                    "orbs": [
                        {"id": "Dark", "evoke_amount": 12},
                        {"id": "Empty"},
                        {"id": "Empty"},
                    ],
                },
            },
        )
        mutations = []
        for path, value in (
            (("player", "orb_slots"), 4),
            (("player", "orbs", 0, "evoke_amount"), 18),
            (("lightning_channeled_this_combat",), 3),
            (("frost_channeled_this_combat",), 2),
            (("powers_played_this_combat",), 2),
            (("emotion_chip_pending",), True),
            (("hand", 0, "base_damage"), 7),
        ):
            changed = deepcopy(first)
            target = changed["game_state"]["combat_state"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            mutations.append(changed)

        key = stable_boundary_key(first)
        self.assertTrue(all(stable_boundary_key(row) != key for row in mutations))

    def test_combat_screen_waits_until_play_or_end_is_available(self):
        before = observation("MAP", choices=("x=1",))
        transient = observation(
            "NONE",
            phase="COMBAT",
            commands=("potion", "wait", "state"),
            combat={
                "turn": 1,
                "monsters": [
                    {"name": "Lagavulin", "current_hp": 82, "move_id": 5}
                ],
            },
        )
        settled = deepcopy(transient)
        settled["available_commands"] = ["play", "end", "potion", "wait", "state"]
        driver = Driver(transient, settled)

        result = self.settle(before, transient, "choose 0", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])

    def test_boundary_key_ignores_late_power_cleanup_on_gone_enemy(self):
        gone = observation(
            "NONE",
            phase="COMBAT",
            commands=("play", "end"),
            combat={
                "monsters": [
                    {
                        "id": "FuzzyLouseDefensive",
                        "current_hp": 0,
                        "is_gone": True,
                        "half_dead": False,
                        "powers": [{"id": "Vulnerable", "amount": 1}],
                    }
                ]
            },
        )
        cleaned = deepcopy(gone)
        cleaned["game_state"]["combat_state"]["monsters"][0]["powers"] = []

        self.assertEqual(stable_boundary_key(gone), stable_boundary_key(cleaned))

        dormant = deepcopy(gone)
        dormant["game_state"]["combat_state"]["monsters"][0]["half_dead"] = True
        cleaned_dormant = deepcopy(dormant)
        cleaned_dormant["game_state"]["combat_state"]["monsters"][0]["powers"] = []
        self.assertNotEqual(
            stable_boundary_key(dormant),
            stable_boundary_key(cleaned_dormant),
        )

    def test_boundary_key_ignores_cosmetic_event_body_text(self):
        first = observation("EVENT", choices=("continue",))
        first["game_state"]["screen_state"] = {
            "event_id": "Spire Heart",
            "body_text": "You deal 1837 damage!",
        }
        replay = deepcopy(first)
        replay["game_state"]["screen_state"]["body_text"] = (
            "You deal 1862 damage!"
        )

        self.assertEqual(stable_boundary_key(first), stable_boundary_key(replay))

    def test_event_choice_advances_before_accepting_an_idle_response(self):
        before = observation("EVENT", choices=("obtain a curse",))
        early = observation("EVENT", choices=("leave",))
        early["game_state"]["deck"] = [{"id": "Bash"}]
        committed = deepcopy(early)
        committed["game_state"]["deck"].append({"id": "Clumsy"})
        driver = Driver(committed)

        result = self.settle(before, early, "choose 0", driver)

        self.assertIs(result, committed)
        self.assertEqual(driver.calls, ["wait 10"])

    def test_identical_event_choice_waits_for_visible_animation_to_finish(self):
        before = observation("EVENT", choices=("watch",))
        before["game_state"]["screen_state"] = {
            "event_id": "The Joust",
            "body_text": "Give me strength, Noodles!",
        }
        pending = deepcopy(before)
        pending["game_state"]["screen_state"]["body_text"] = "CRASH! KLANG! POW!"
        pending["game_state"]["transition_pending"] = True
        settled = deepcopy(pending)
        settled["game_state"]["transition_pending"] = False
        driver = Driver(pending, settled)

        result = self.settle(before, pending, "choose 0", driver)

        self.assertIs(result, settled)
        self.assertEqual(driver.calls, ["wait 10", "wait 10"])
        self.assertEqual(stable_boundary_key(before), stable_boundary_key(settled))

    def test_unstable_state_fails_closed(self):
        state = observation("EVENT", choices=("leave",))
        driver = Driver(state, state)
        policy = StabilityPolicy(poll_interval=0, timeout=10, max_refreshes=1)

        with self.assertRaisesRegex(GameStabilityError, "not observable"):
            settle_game_state(
                state,
                deepcopy(state),
                "choose 0",
                read_state=driver.read_state,
                wait_frames=driver.wait_frames,
                policy=policy,
            )

    def test_missing_state_fixes_signal_fails_immediately(self):
        state = observation("EVENT", choices=("leave",))
        state["game_state"].pop("transition_pending")
        driver = Driver()

        with self.assertRaisesRegex(GameStabilityError, "AgentStateFixes"):
            self.settle(None, state, "reset", driver)
        self.assertEqual(driver.calls, [])


if __name__ == "__main__":
    unittest.main()
