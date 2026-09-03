# Agent State Fixes

This minimal ModTheSpire support mod adds state fields that the agent needs but
CommunicationMod does not expose.

It also makes the combat queue protocol explicit: every game-state response
contains `action_phase` and `current_action`, with an empty `current_action`
meaning that no action is executing.

It currently adds:

- `is_burning: true` to the map node containing the Emerald Key, so map
  prompts render that node as `E*` (Burning Elite).
- `cards_played_this_turn`, `attacks_played_this_turn`, and
  `skills_played_this_turn` to combat state, so a fresh `sts_lightspeed`
  search resumes the current turn instead of resetting per-turn relic and
  card limits.
- `powers_played_this_combat`, so Force Field keeps its combat-wide cost
  reduction across searches.
- each combat card's `misc` value, preserving accumulated Rampage damage and
  the state of other cards whose value changes during combat.
- each combat card's persistent `combat_cost`, distinct from CommunicationMod's
  existing current-turn `cost`, so retained Snecko costs and temporary cost
  modifiers are reconstructed separately.
- Defect orb slots, Lightning/Frost channel counts, Emotion Chip's pending
  trigger, and dynamic Claw/Steam Barrier state, including cards held by
  Bronze Automaton's Stasis power.
- Time Eater's private `usedHaste` flag as `miscBool`, preventing a mid-combat
  search from granting the boss a second Haste heal.
- the already-rolled `move_id` even when Runic Dome hides the rendered intent,
  so combat readiness and native simulation do not wait forever for a field
  that CommunicationMod intentionally omits.
- `centennial_puzzle_used_this_combat`, preventing a mid-combat search from
  triggering Centennial Puzzle more than once.
- the player's current facing direction, and restores the base game's facing
  update for CommunicationMod card commands during the Shield and Spear fight.
- every seeded dungeon RNG's two internal state words and counter. The harness
  records them before each live action and restores them before the matching
  replay action, so replay does not depend on animation timing or machine
  speed.
- `transition_pending` and diagnostics for queued permanent-state effects, so
  live runs and replay do not advance past an event while a selected card,
  potion, key, or other persistent result is still being committed. Decorative
  effects such as speech bubbles do not block decisions; card movement, timed
  event animations, player escape, and battle completion do because their
  completion can change game state.
- a language-independent `grid_operation`, including ambiguous Neow remove and
  transform rewards, so stability checks survive localization and manual
  takeover.
- discard-to-hand cost overrides remain atomic across bridge frames, so effects
  such as Liquid Memories keep their temporary cost in live play and replay.

Build it after placing the game jars under `runtime/lib/`:

```bash
./game_mods/agent_state_fixes/build.sh
```

The output is `runtime/mods/AgentStateFixes.jar`. The `agentstatefixes` mod
must be included in ModTheSpire's `--mods` list.
