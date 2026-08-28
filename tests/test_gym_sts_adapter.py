from __future__ import annotations

import json
from pathlib import Path
import unittest

from spire_agent.adapters.gym_sts import (
    GymStsAdapterError,
    GymStsObservationAdapter,
    GymStsSession,
    GymStsSessionError,
)
from spire_agent.contracts import AgentKind
from spire_agent.tools.game_stability import StabilityPolicy


ROOT = Path(__file__).resolve().parents[1]


def raw_state(
    screen: str,
    *,
    phase: str = "COMPLETE",
    room: str = "EventRoom",
    commands: list[str] | None = None,
    choices: list[object] | None = None,
    combat: dict | None = None,
    ready: bool = True,
) -> dict:
    game = {
        "seed": "TESTSEED",
        "act": 1,
        "floor": 6,
        "room_type": room,
        "room_phase": phase,
        "screen_type": screen,
        "screen_state": {"prompt": screen},
        "choice_list": choices if choices is not None else [],
        "deck": [{"name": "Strike"}],
        "transition_pending": False,
    }
    if combat is not None:
        game["combat_state"] = combat
    return {
        "available_commands": (
            commands if commands is not None else ["choose", "state"]
        ),
        "ready_for_command": ready,
        "in_game": True,
        "game_state": game,
    }


class ObservationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = GymStsObservationAdapter()

    def test_accepts_a_real_gym_sts_observation(self):
        from gym_sts.spaces.observations.observations import Observation

        payload = json.loads(
            (ROOT / "tests" / "fixtures" / "combat.json").read_text()
        )
        state = self.adapter.adapt(Observation(payload))

        self.assertEqual(state.owner_hint, AgentKind.COMBAT)
        self.assertEqual(state.screen.type, "NONE")
        self.assertEqual(state.facts["floor"], 7)
        self.assertIsNotNone(state.combat)
        self.assertNotIn("combat_state", state.facts)

    def test_routes_by_room_and_ignores_stale_post_combat_data(self):
        combat = {"monsters": [{"current_hp": 0, "is_gone": True}], "turn": 4}
        post_combat = self.adapter.adapt(
            raw_state("CARD_REWARD", phase="COMBAT", combat=combat)
        )
        combat_grid = self.adapter.adapt(
            raw_state("GRID", phase="COMBAT", combat={"hand": [], "turn": 2})
        )
        event_grid = self.adapter.adapt(raw_state("GRID"))
        map_state = self.adapter.adapt(raw_state("MAP", room="MapRoom"))

        self.assertEqual(post_combat.owner_hint, AgentKind.BUILD)
        self.assertIsNone(post_combat.combat)
        self.assertEqual(combat_grid.owner_hint, AgentKind.COMBAT)
        self.assertEqual(event_grid.owner_hint, AgentKind.BUILD)
        self.assertEqual(map_state.owner_hint, AgentKind.MAP)

    def test_live_combat_card_choice_stays_with_combat(self):
        combat = {"monsters": [{"current_hp": 18}], "turn": 2}
        state = self.adapter.adapt(
            raw_state("CARD_REWARD", phase="COMBAT", combat=combat)
        )
        self.assertEqual(state.owner_hint, AgentKind.COMBAT)

    def test_scope_is_stable_across_nested_screens(self):
        event = self.adapter.adapt(raw_state("EVENT", choices=["accept", "leave"]))
        grid = self.adapter.adapt(raw_state("GRID", choices=["Strike", "Defend"]))

        self.assertEqual(event.scope_id, grid.scope_id)
        self.assertNotEqual(
            event.screen.interaction_id,
            grid.screen.interaction_id,
        )

    def test_terminal_and_command_errors_are_explicit(self):
        terminal = raw_state("COMPLETE", room="TrueVictoryRoom", commands=[])
        self.assertTrue(self.adapter.adapt(terminal).terminal)

        rejected = raw_state("EVENT")
        rejected["error"] = "invalid choice"
        with self.assertRaisesRegex(GymStsAdapterError, "invalid choice"):
            self.adapter.adapt(rejected)


class FakeEnv:
    def __init__(self, initial: dict):
        self.current = initial
        self.sts_seed = "TESTSTSSEED"
        self.reset_calls: list[dict] = []
        self.commands: list[str] = []
        self.closed = 0
        self.next_response: object | None = None

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return {
            "serialized": True
        }, {"observation": self.current, "sts_seed": self.sts_seed}

    def _do_action(self, command: str):
        self.commands.append(command)
        if self.next_response is not None:
            response, self.next_response = self.next_response, None
            return response
        return self.current

    def observe(self):
        return self.current

    def close(self):
        self.closed += 1


class GymStsSessionTests(unittest.TestCase):
    def policy(self) -> StabilityPolicy:
        return StabilityPolicy(max_refreshes=2, poll_interval=0)

    def test_reset_unpacks_the_real_gym_return_shape(self):
        env = FakeEnv(raw_state("EVENT"))
        canonical_seeds: list[str] = []
        session = GymStsSession(
            env,
            stability_policy=self.policy(),
            reset_kwargs={"seed": 17},
            on_sts_seed=canonical_seeds.append,
        )

        state = session.reset()

        self.assertEqual(state.screen.type, "EVENT")
        self.assertEqual(state.facts["sts_seed"], "TESTSTSSEED")
        self.assertEqual(env.reset_calls, [{"seed": 17}])
        self.assertEqual(canonical_seeds, ["TESTSTSSEED"])
        self.assertEqual(env.commands, [])

    def test_sts_seed_hook_failure_aborts_reset(self):
        env = FakeEnv(raw_state("EVENT"))

        def reject_existing(seed: str) -> None:
            raise RuntimeError(f"duplicate run {seed}")

        session = GymStsSession(
            env,
            stability_policy=self.policy(),
            on_sts_seed=reject_existing,
        )

        with self.assertRaisesRegex(RuntimeError, "duplicate run TESTSTSSEED"):
            session.reset()
        self.assertEqual(env.commands, [])

    def test_execute_settles_then_confirms_the_command(self):
        stable = raw_state("MAP", room="MapRoom")
        unstable = raw_state("EVENT", commands=["wait", "state"])
        env = FakeEnv(raw_state("EVENT"))
        session = GymStsSession(env, stability_policy=self.policy())
        session.reset()
        env.current = stable
        env.next_response = unstable

        result = session.execute("choose 0")

        self.assertTrue(result.confirmed)
        self.assertEqual(result.command, "choose 0")
        self.assertEqual(result.state.owner_hint, AgentKind.MAP)
        self.assertEqual(result.state.facts["sts_seed"], "TESTSTSSEED")
        self.assertEqual(env.commands, ["choose 0", "wait 10"])

    def test_rejected_command_returns_a_fresh_uncommitted_state(self):
        env = FakeEnv(raw_state("EVENT"))
        rejected = raw_state("EVENT")
        rejected["error"] = "choice out of range"
        session = GymStsSession(env, stability_policy=self.policy())
        session.reset()
        env.next_response = rejected

        result = session.execute("choose 99")

        self.assertFalse(result.confirmed)
        self.assertEqual(result.error, "choice out of range")
        self.assertEqual(result.state.screen.type, "EVENT")

    def test_refresh_settles_direct_game_window_changes(self):
        env = FakeEnv(raw_state("EVENT", choices=["accept"]))
        session = GymStsSession(env, stability_policy=self.policy())
        session.reset()

        unchanged = session.refresh()
        env.current = raw_state("MAP", room="MapRoom", choices=["x=1"])
        changed = session.refresh()

        self.assertFalse(unchanged.changed)
        self.assertTrue(changed.changed)
        self.assertEqual(changed.state.screen.type, "MAP")
        self.assertEqual(env.commands, [])

    def test_refresh_detects_hidden_rng_changes_at_the_same_boundary(self):
        initial = raw_state("EVENT", choices=["accept"])
        initial["game_state"]["replay_rng_state"] = {"event": [1, 2, 3]}
        env = FakeEnv(initial)
        session = GymStsSession(env, stability_policy=self.policy())
        session.reset()
        changed = raw_state("EVENT", choices=["accept"])
        changed["game_state"]["replay_rng_state"] = {"event": [1, 2, 4]}
        env.current = changed

        refreshed = session.refresh()

        self.assertTrue(refreshed.changed)

    def test_transport_timeout_is_fatal_and_is_not_refreshed(self):
        class Rejected(Exception):
            pass

        class Timeout(Rejected):
            pass

        class TimeoutEnv(FakeEnv):
            def _do_action(self, command: str):
                self.commands.append(command)
                raise Timeout("acknowledgement lost")

        env = TimeoutEnv(raw_state("EVENT"))
        session = GymStsSession(
            env,
            stability_policy=self.policy(),
            rejected_exceptions=(Rejected,),
            fatal_exceptions=(Timeout,),
        )
        session.reset()

        with self.assertRaisesRegex(Timeout, "acknowledgement lost"):
            session.execute("choose 0")
        self.assertEqual(env.commands, ["choose 0"])

    def test_close_is_idempotent_and_prevents_more_commands(self):
        env = FakeEnv(raw_state("EVENT"))
        session = GymStsSession(env, stability_policy=self.policy())
        session.close()
        session.close()

        self.assertEqual(env.closed, 1)
        with self.assertRaisesRegex(GymStsSessionError, "closed"):
            session.execute("choose 0")


if __name__ == "__main__":
    unittest.main()
