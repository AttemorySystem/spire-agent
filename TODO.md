# TODO

- Add offline MCTS labels to the frozen evaluation cases. They must evaluate
  policy changes only and must never enter the runtime card decision path.
- Validate the three finite Act 1 stage needs against full runs. Add or change a
  need only as a reviewed data change, never as a per-card policy exception.
- Replace independent marginal offer probabilities with empirical joint offer
  scenarios only if evaluation shows the approximation changes decisions.
- Model relics in Winning Path after the card templates stabilize. Cover all
  common, Ironclad, and Defect Boss relics plus ordinary relics that provide or
  constrain template capabilities (especially energy, draw, Focus, orb slots,
  orb triggers, defense, and scaling). Derive Boss-relic preference evidence
  from expert history, expose the relic model through character policy data for
  opt-agent tuning, and require historical plus combat regression before it
  receives runtime authority.
- Remove hidden-RNG strategy fusion from MCTS. Independent determinized worlds
  can currently choose incompatible future policies: a random optional action
  such as White Noise is deferred in worlds with an unfavorable future result
  and used later in favorable worlds, although the live agent cannot know that
  result before playing it. This overvalues deferral and delayed White Noise for
  24 turns in the DEFECTXX97 Champ fight. Implement a shared observable-state
  policy across RNG worlds: decisions before a random result is revealed must
  share action statistics, while distinct visible outcomes may branch after the
  reveal. Do not use card-name rules, timing thresholds, or replay RNG state.
  Also make `applyRobustContinuationDominance()` a true conservative override:
  its challenger must be no worse at the comparable replanning boundary in
  reachability, horizon, effective HP, potion count, and state value. Add a
  synthetic hidden-RNG information-set test and DEFECTXX97 Champ fixtures, then
  require the complete historical battle-case regression with no previously
  winning case becoming a loss before merging.
