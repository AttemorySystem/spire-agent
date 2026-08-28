# Card reward evaluation data

This directory is the frozen input layer for evaluating card-reward policy
changes. Runtime agents do not import it.

- `cases.jsonl` contains independent decision snapshots.
- `runs.jsonl` groups the snapshots into ordered runs and records fixed deck
  changes between them. These changes cover removals, upgrades, transforms,
  and cards obtained outside normal card rewards.
- `manifest.json` records counts and SHA-256 hashes of every source journal.
- `report.md` is a human-readable data quality summary.
- `expert_actions.jsonl`, when present, contains immutable historical expert
  actions used for comparison.

`observed_action` is what the old agent did. It is useful for behavior
regression and provenance, but it is not a correctness label. Deaths are kept
because removing them would hide bad construction trajectories. Incomplete
runs are kept with an explicit quality flag.

Build the dataset from the repository root:

```bash
uv run sts-card-reward-dataset --source ../remote \
  --output src/spire_agent/tools/winning_path/data/evaluation
```

Build the primary expert dataset from all Baalorlord records. Every case keeps
its evidence quality; `modern_verified` is the highest-confidence subset:

```bash
uv run sts-card-reward-dataset \
  --expert-source ../card_choice/expert_data/ironclad_card_choices.jsonl \
  --output src/spire_agent/tools/winning_path/data/evaluation/expert
```

The generated files contain no timestamp or absolute path, so identical
source journals produce byte-identical output.

Snapshot agreement alone cannot evaluate deck growth. The sequential evaluator
starts from the first snapshot, applies the candidate policy, then replays each
`fixed_deck_delta`. It reports impossible removals instead of hiding them.

## Evaluate the current policy

```bash
uv run sts-card-reward-eval \
  --dataset src/spire_agent/tools/winning_path/data/evaluation \
  --output src/spire_agent/tools/winning_path/data/evaluation/current_policy
```

The evaluator does not call the game, LLM, or MCTS. Its snapshot pass compares
direct decisions exactly and checks whether an observed action falls inside an
advice shortlist. Its conservative sequential pass executes direct policy
decisions and uses Skip whenever advice would be required. This makes deck
growth deterministic, but it is not a win-rate prediction.

`differences.jsonl` contains every conflict and `verified_differences.jsonl`
contains only the highest-quality subset. There is no hidden score or cutoff.

Use the highest-confidence historical subset after every policy change:

```bash
uv run sts-card-reward-eval \
  --dataset src/spire_agent/tools/winning_path/data/evaluation/expert \
  --output src/spire_agent/tools/winning_path/data/evaluation/expert/current_policy \
  --review-quality modern_verified \
  --compact
```

Run the full historical comparison without compact output with:

```bash
uv run sts-card-reward-eval \
  --dataset src/spire_agent/tools/winning_path/data/evaluation/expert \
  --output /tmp/winning-path-candidate \
  --check
```

## Recorded combat evaluation

The combat evaluator rebuilds the candidate deck in run order, extracts every
recorded passed Act boss and fatal encounter, and compares the historical and
candidate decks with independent paired battles. Advice-required choices use
the historical action by default, so the result isolates deterministic policy
changes without introducing LLM variance.

Prepare and inspect checkpoints before spending MCTS time:

```bash
uv run sts-winning-path-combat-eval \
  --dataset src/spire_agent/tools/winning_path/data/evaluation --source ../remote \
  --output /tmp/winning-path-combat --prepare-only
```

Run a screening budget, then reuse the prepared checkpoints for a larger
confirmation budget:

```bash
uv run sts-winning-path-combat-eval --output /tmp/winning-path-combat \
  --simulate-only --worlds 16 --simulations 500
uv run sts-winning-path-combat-eval --output /tmp/winning-path-combat \
  --simulate-only --worlds 64 --simulations 3000 --max-time-ms 500
```

`checkpoints.jsonl` contains exact deck construction and counterfactual
warnings. `combat_results.jsonl` retains paired per-world outcomes;
`regressions.jsonl` and `improvements.jsonl` contain only results confirmed by
at least 16 complete worlds. This is a fixed-state deck-swap regression test,
not an estimate of whole-run win rate.

For a single historical run, generate one review document containing every
card choice, every checkpoint deck difference, the battle summary, and each
paired RNG world:

```bash
uv run sts-winning-path-run-report ../remote/216 \
  --output /tmp/winning-path-run-216
```
