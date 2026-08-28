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
- Rework MCTS robust-continuation dominance. It currently lets rare high-quality
  continuations replace much more reliable root actions. Keep it out of this
  fix; later either remove the override or restrict it to statistically tied
  root actions, with the existing Burning Pact regression retained.
