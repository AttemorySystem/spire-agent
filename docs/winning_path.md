# Winning Path

Status: the only Winning Path runtime for Ironclad and Defect card rewards

Protocol: `2.3.0`

## 1. Motivation

Slay the Spire permits an enormous number of decks, but successful Ironclad
decks tend to converge on a much smaller set of coherent structures: strength
plus survival, block plus Body Slam, exhaust engines, status engines, and
energy/draw engines. Winning Path treats card selection as a search from the
current deck toward those reviewed winning structures.

V1 showed that this approach works, but its optimization surface was too open.
A surprising choice could lead to a new rule, threshold, or exception in a
different layer. It then became difficult to answer three basic questions:

- which input caused a decision;
- which other decisions a change could affect;
- whether a local fix improved card quality overall.

Making the LLM responsible for every card reward is not an acceptable escape.
Its variance makes failures hard to reproduce, and unconstrained advice tends
to grow the deck. Training a policy model was also rejected: it hides the
domain model, needs a trustworthy target that does not exist, and makes causal
review harder.

Winning Path therefore has two goals:

1. Keep the complete tuning surface fixed, explicit, and reviewable.
2. Improve toward expert-level play through reproducible expert and combat
   evaluation instead of accumulating special cases.

It is deliberately a small card picker. It does not replace BuildAgent,
MapAgent, or CombatAgent, and it does not run MCTS while the live game is
waiting for a card choice.

## 2. Policy model

There are exactly three sources of positive card evidence. Hard constraints
may veto a card, and Skip is the result when no card has enough positive
evidence; neither is a fourth positive source.

### 2.1 Winning templates and distance

The catalog describes reviewed deck modules and routes. A module contains
slots, alternatives, anchors, prerequisites, compatibility, capabilities, and
resource constraints. A route is a compatible set of modules that can form a
complete deck.

For every current deck and every candidate deck, Winning Path computes a fixed
lexicographic certificate distance:

1. missing capabilities;
2. missing relics;
3. missing required cards;
4. completion probability;
5. completed core modules;
6. active route modules.

The future reward horizon and observed offer probabilities are part of the
distance calculation. This matters when two offered cards advance different
templates: completing a route whose remaining cards are realistically
findable is better evidence than merely reducing a raw slot count.

Candidate progress is classified as:

- `NONE`;
- `REACHABLE_ENTRY`;
- `COMMITTED_PROGRESS`;
- `CORE_ACTIVATION`.

The module definitions, routes, and distance inputs are parameters. The
lexicographic comparison algorithm is fixed policy code.

### 2.2 Foundation and transition needs

A deck may need to survive the next elite or boss before its final structure
is complete. Winning Path projects a candidate-independent target plan from
the current Act, boss, route facts, and Act 4 objective. The encounter model then measures
deficits in a small reviewed capability vocabulary:

- single-target frontload;
- AOE;
- immediate block;
- Weak or Strength reduction;
- scaling damage;
- scaling defense;
- draw consistency;
- energy;
- sustain.

The same layer also maintains a reviewed minimum density for foundation
capabilities. Draw consistency currently requires one source through deck size
14, two through 22, and three thereafter; a completed draw engine satisfies the
need directly. This is construction readiness, not a separate evidence source.

Candidate coverage is classified as `NONE`, `OPEN_NEED`, `CRITICAL_NEED`, or
`BLOCKING_NEED`. Target pools, encounter requirements, aliases, and card
capabilities are parameters. Offered cards never influence which encounter is
selected as the target.

### 2.3 Contextual expert experience

Expert card-choice records are converted to pairwise observations:

- a picked card beats every other offered card and Skip;
- Skip beats every offered card;
- Singing Bowl choices are excluded because Bowl is not a card preference.

Lookup proceeds from specific to general:

1. Act, owned/unowned status of both cards, and deck-size band;
2. Act and owned/unowned status of both cards;
3. Act-only, when both cards are unowned.

The signed confidence score is `(wins - losses) / sqrt(wins + losses)`. During
cross-validation, all choices from the tested run are excluded from the table
used to score that run. Expert evidence is a reproducible prior, not a claim
that every historical action is uniquely optimal.

## 3. Fixed resolver

The resolver has one reviewed order:

1. reject hard constraints;
2. resolve blocking survival needs;
3. resolve authorized template progress;
4. resolve the best authorized transition frontier, using expert comparisons
   only to rank cards within that frontier;
5. resolve standalone direct expert evidence;
6. Skip when no eligible card has positive evidence;
7. ask the LLM only for an unresolved positive frontier.

Within a frontier, pairwise expert comparisons may select a card only when it
is the unique sufficiently supported Condorcet winner: it must beat every
other card in that frontier. Otherwise it returns the frontier as a bounded
shortlist. The LLM cannot select outside the shortlist and cannot override a
direct result.

Singing Bowl is intentionally simple. If Winning Path selects a card, the card
wins. If it selects Skip or exposes Skip as the non-card alternative, Bowl
replaces that alternative. Two maximum HP never outrank a useful card.

Winning Path has no legacy runtime fallback and no online MCTS authority.

## 4. Architecture

The system separates source data, compiled knowledge, the deterministic
runtime kernel, and evaluation:

```text
expert records      winning-deck graph      support capabilities
      |                       |                       |
      +---------------- catalog compiler ------------+
                              |
                  ironclad_catalog.json + policy data
                              |
DecisionRequest -> canonical state -> plan/needs/evidence -> resolver
                                                           |
                                      direct command or bounded shortlist
                                                           |
                                      unchanged CardPicker interface

historical journals -> frozen datasets -> expert CV + deck-swap checkpoints
                                                   |
                                             sts_lightspeed MCTS
                                                   |
                                          baseline/candidate gate
```

The composition root selects only the character-specific picker. BuildAgent,
the LLM approval path, observers, replay, and the `card_choices.jsonl` schema
continue to use the existing `CardPicker` contract.

### 4.1 Runtime modules

The online path is intentionally linear:

| Module | Single responsibility |
|---|---|
| `state.py` | Project a `DecisionRequest` into an immutable, replayable state. |
| `plan.py` | Reconstruct active, committed, and blocked deck modules. |
| `templates.py` | Compare current and candidate template certificates. |
| `needs.py` | Select encounter targets and calculate transition deficits. |
| `evidence.py` | Produce hard, template, transition, and expert facts per card. |
| `resolver.py` | Apply the fixed authority order and bound unresolved choices. |
| `analysis.py` | Assemble one complete trace artifact. |
| `picker.py` | Adapt the result to the unchanged live `CardPicker` interface. |

Live runs, evaluation, and comparison utilities all call the same analysis and
resolver path. There is no shadow policy or benchmark-only adapter.

### 4.2 Offline modules

The larger modules are business and evaluation tools, not agent framework:

| Module | Responsibility |
|---|---|
| `catalog.py` | Validate source contracts and compile deterministic expert evidence. |
| `parameters.py` | Load and validate the complete parameter index. |
| `protocol.py` | Version and fingerprint the policy kernel. |
| `evaluation.py` | Run the shared snapshot and expert cross-validation pass. |
| `combat_eval.py` | Build historical deck-swap checkpoints and run paired battles. |

The runtime does not import evaluation datasets or invoke the optimizer.

## 5. Data contracts and build flow

### 5.1 Source data

Ironclad currently uses three reviewed source files outside this package:

- `../card_choice/winning_deck_graph.json`: modules, routes, card policies,
  resource rules, and graph semantics;
- `../card_choice/expert_data/ironclad_card_choices.jsonl`: expert reward
  observations;
- `../card_choice/support_capabilities.json`: capability and prerequisite
  annotations for support cards.

The graph is the domain model. The expert JSONL supplies observed preferences
and offer statistics. The support file translates individual cards into the
capability vocabulary used by templates and encounter needs.

### 5.2 Compiled artifacts

The shipped artifacts under `src/spire_agent/tools/winning_path/data/` are:

- `ironclad_policy.json`: the sole hand-edited Ironclad parameter file, containing
  templates and distance, transition needs, expert context, and authority;
- `ironclad_catalog.json`: generated, read-only expert preferences, offer rates,
  reward horizons, and provenance; template knowledge is not duplicated here;
- `ironclad_protocol.json`: policy identity, scope, and kernel version.

Defect follows the same split with `defect_policy.json`,
`defect_catalog.json`, and `defect_protocol.json`. Fixed resolver
semantics live in code and are deliberately outside the tuning surface.

Rebuild the catalog from the repository root after reviewing source changes:

```bash
uv run sts-winning-path-build \
  --graph ../card_choice/winning_deck_graph.json \
  --choices ../card_choice/expert_data/ironclad_card_choices.jsonl \
  --support ../card_choice/support_capabilities.json \
  --parameters src/spire_agent/tools/winning_path/data/ironclad_policy.json \
  --output src/spire_agent/tools/winning_path/data/ironclad_catalog.json
```

The compiler rejects unknown module/support fields, invalid slot references,
duplicate definitions, and protocol mismatches. Output ordering is
deterministic. Generated catalogs are never edited by hand.

### 5.3 Evaluation datasets

There are two distinct frozen datasets:

1. `data/evaluation/expert` is built from expert records and supplies preference
   cross-validation.
2. `data/evaluation` is built from `remote/*/card_choices.jsonl` run journals and
   supplies sequential deck reconstruction and combat checkpoints.

Build them with:

```bash
uv run sts-card-reward-dataset \
  --expert-source ../card_choice/expert_data/ironclad_card_choices.jsonl \
  --output src/spire_agent/tools/winning_path/data/evaluation/expert

uv run sts-card-reward-dataset \
  --source ../remote \
  --output src/spire_agent/tools/winning_path/data/evaluation
```

Each dataset stores independent choice snapshots in `cases.jsonl`, ordered run
trajectories and non-reward deck changes in `runs.jsonl`, and source hashes in
`manifest.json`. Historical choices are reproducible comparison behavior;
downstream combat simulation supplies the quality signal for changed decks.

## 6. Runtime, logging, and replay

Winning Path is the only Winning Path implementation. Runtime configuration
selects `agents.build: winning_path`; `create_card_picker(character)` chooses
the character knowledge behind the same `CardPicker` methods.

Every decision writes the existing Winning Path and card-choice records.
The compact card-choice row identifies `ironclad.winning_path` and keeps
the state needed to rebuild the historical dataset. Its linked Winning Path
record contains:

- canonical decision state;
- offered candidates and allowed choice IDs;
- selected command and whether an LLM proposal was approved;
- protocol, policy, catalog, encounter, implementation, and state hashes;
- deck plan, target plan, need profile, candidate evidence, and resolution.

This is enough to explain a decision without rerunning the LLM. Replay uses
the confirmed command journal exactly as before; Winning Path adds no alternate
replay path and no direct game input. The record shape is covered by a replay
round-trip test.

## 7. Benchmark pipeline

One metric cannot prove a card picker is better. Winning Path uses three layers.

### 7.1 Focused tests

Small tests express the semantic invariant behind a change and include both a
positive case and a nearby case that must remain unchanged. They should assert
the policy, shortlist, or command rather than incidental explanation text.

### 7.2 Leakage-safe expert cross-validation

The expert evaluator uses five folds grouped by run. All observations from the
test run are excluded from its preference table. It reports:

- deterministic coverage;
- direct agreement with the expert action;
- `PICK_VS_NO_CARD`, `NO_CARD_VS_PICK`, and `PICK_VS_PICK` directions;
- those directions by deck-size band and policy label.

Run it directly with:

```bash
uv run sts-card-reward-eval \
  --dataset src/spire_agent/tools/winning_path/data/evaluation/expert \
  --output /tmp/winning-path-expert \
  --preference-folds 5 \
  --preference-only
```

The accepted Ironclad baseline covers 86.3539% of 2,323 expert decisions
deterministically. Agreement within direct decisions is 56.0319%. Coverage is
a hard ownership goal; exact agreement is diagnostic because expert actions
are not assumed to be uniquely optimal.

### 7.3 Historical deck-swap combat regression

For each selected historical run, the evaluator rebuilds the candidate deck
in choice order. It extracts every recorded passed Act boss and fatal
encounter. `sts_lightspeed` then runs paired battles for the historical and
candidate decks using the same encounter state and RNG worlds.

The frozen default protocol uses:

- runs `210` through `239`;
- 16 paired worlds per checkpoint;
- 500 MCTS simulations per decision;
- 100 ms maximum search time per decision;
- 300 decisions per battle;
- eight independent evaluator processes;
- potions disabled;
- the historical action when Winning Path returns advice.

Using history for advice isolates deterministic policy changes from LLM
variance. Cache keys include the complete battle input, search budget, and
simulator hash, so unchanged checkpoints reuse identical results.

This benchmark is a fixed-state counterfactual, not a whole-run replay. It
changes reward-selected cards while keeping recorded route, HP, relics,
encounter, and other state fixed. A candidate deck may therefore contain a
card that would have changed an earlier fight or route; such warnings are
retained in the checkpoint artifacts rather than hidden.

## 8. Fixed tuning surface

`ironclad_policy.json` and `defect_policy.json` are the only hand-edited,
decision-affecting parameter files for their characters. Each has exactly four
families:

- `templates`: module/route definitions and certificate distance inputs;
- `transition`: encounter targets, requirements, aliases, and card
  capabilities;
- `expert`: observation construction, context order, deck-size bands, score,
  and confidence thresholds;
- `authority`: the minimum template and transition evidence levels permitted
  to decide directly.

The loader rejects unknown families, fields, and distance fields. A new
parameter family or field, evidence source, or resolver semantic change
requires explicit review; it must not be hidden in a card-, floor-, seed-, or
boss-specific exception.

## 9. Adding another character

The reusable part is the kernel: canonical state, evidence categories,
resolver order, bounded-LLM contract, logging/replay boundary, and benchmark
machinery. The game knowledge is character-specific and must never borrow
Ironclad card or template evidence.

Migrate one character in this order:

1. Collect character-specific expert `card_choices.jsonl` records with stable
   card names, run IDs, deck snapshots, Act/floor, boss, and final action.
2. Build and review its winning modules: anchors, slots, alternatives,
   prerequisites, capabilities, resource conflicts, and compatible routes.
3. Define its transition capability vocabulary and encounter requirements.
   Reuse a capability name only when its semantics genuinely match.
4. Compile separate catalog, policy, encounter, and protocol artifacts with a
   character-specific `policy_id`; never add character branches inside an
   Ironclad data file.
5. Build an expert dataset and establish run-grouped cross-validation. Review
   deterministic coverage and asymmetric deck-growth counters before combat
   tuning.
6. Build a historical run dataset. Confirm that every candidate card and
   character mechanic needed by the checkpoints is implemented correctly in
   `sts_lightspeed` before treating MCTS output as a label.
7. Establish a frozen character-specific 30-run baseline and acceptance gate.
   Benchmark run IDs and simulator support are part of that character's
   protocol.
8. Add focused hard-constraint and evidence tests for unique character
   mechanics.
9. Add the character profile to `create_card_picker`; all characters use the
   same `WinningPathCardPicker`. Do not modify BuildAgent, replay, or the
   decision contract.
10. Run live contract and replay round-trip tests, then register the picker
    only after its benchmark passes.

This separation allows later character knowledge to evolve independently
while keeping the small policy kernel and external architecture unchanged.
The first completed migration is documented in
[defect_winning_path.md](defect_winning_path.md).

## 10. Limits and maintenance rules

- Static capability labels do not prove that a deck beats an encounter.
- Expert agreement measures behavior, not perfect play.
- Fixed-budget MCTS is reproducible regression evidence, not perfect play.
- Deck-swap checkpoints do not reproduce exact draws or whole-run route
  consequences.
- Missing or conflicting evidence must reduce Winning Path authority; it must
  not create a hidden fallback or special rule.
- Framework and agent changes require separate justification. Normal Winning
  Path work stays under `src/spire_agent/tools/winning_path` and its data/tests/docs.
- Generated baseline snapshots belong in an output directory such as `/tmp`,
  not in source control. The repository stores source data contracts, policy
  artifacts, code, and tests.

The intended improvement loop is:

```text
expert experience -> templates and parameters -> deterministic policy
                  -> historical combat checkpoints -> paired MCTS labels
                  -> accept or reject one reviewed change -> next hypothesis
```

In that limited and auditable sense, Winning Path is a self-improving agent:
MCTS provides repeatable outcome labels, the fixed gate rejects regressions,
and successive small knowledge changes can approach expert card-picking
quality without surrendering determinism or maintainability.
