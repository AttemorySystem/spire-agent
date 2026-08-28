# Defect Winning Path

## Goal

The first Defect picker reuses the Winning Path decision kernel without
reusing Ironclad knowledge. Its job is to make most card rewards deterministic,
keep every decision traceable to reviewed evidence, and provide a stable place
to improve Defect knowledge as replayable runs accumulate.

The external `CardPicker` interface is unchanged.
`create_card_picker("DEFECT")` selects `WinningPathCardPicker("DEFECT")`; there
is no alternate Defect card-reward policy in the runtime.

## Architecture

The shared kernel still has three evidence sources and one fixed resolver:

1. Template distance: progress toward a reviewed winning mechanism.
2. Transition need: foundation density and immediate readiness for the next
   elite, boss, or Heart.
3. Expert experience: contextual pairwise card preferences.
4. Resolver: hard constraints, survival, templates, transition needs, expert
   evidence, take-or-skip, then a bounded LLM only for unresolved conflicts.

Character knowledge is selected from `state.run.character`. The Defect profile
has one hand-edited parameter file, `defect_policy.json`, containing all four
parameter families: templates and distance, transition needs, expert context,
and authority thresholds. `defect_catalog.json` is generated, read-only
expert evidence: pairwise preferences, offer rates, horizons, and provenance.

Ironclad keeps its existing catalog, certificate distance, and decisions. The
shared loader became character-aware; no BuildAgent or framework contract
changed.

## Initial knowledge

The initial expert source is Baalorlord's public profile-0 archive:

- archive: `https://baalorlord.tv/archive/0/runs.zip`
- SHA-256: `5beee3a24715c0d562e30094d364be242cbd5b46ea5522c16cbcbe63dd72ee93`
- 661 Defect A20 runs
- 14,959 standard combat or Act 1/2 boss card rewards
- 347 A20 Heart wins used to check template support

Expert preferences use all A20 runs; templates use only Heart-winning final
decks. Raw run history does not contain a reliable deck-before snapshot for
every reward, so archive preferences are deliberately Act-only. In particular,
first-copy evidence is not reused for an already-owned card. New replayable run
logs can later add exact owned-card and deck-size contexts without changing the
resolver.

The reviewed catalog has 17 small mechanistic templates. Each support count is
the number of 347 expert A20 Heart-winning final decks that complete the exact
module slots:

| Template | Heart-win support |
| --- | ---: |
| Focus and Frost defense | 260 / 347 |
| Focus depth | 186 / 347 |
| Power engine | 171 / 347 |
| Draw and card-access engine | 159 / 347 |
| Dark orb control | 137 / 347 |
| Orb capacity scaling | 122 / 347 |
| Recycle energy loop | 114 / 347 |
| Genetic Algorithm scaling block | 107 / 347 |
| Electrodynamics lightning | 92 / 347 |
| Self Repair sustain | 83 / 347 |
| Generated Power payoff | 71 / 347 |
| Meteor Strike energy engine | 66 / 347 |
| Buffer amplification | 62 / 347 |
| Artifact-supported Focus | 60 / 347 |
| All for One recursion | 43 / 347 |
| Reprogram physical scaling | 27 / 347 |
| Blizzard Frost cycle | 18 / 347 |

These supports validate that a mechanism occurs in winning decks. They are not
card scores and are not compared across templates as win rates.

The original seven modules retain direct construction authority. Of the ten
new modules, Focus depth, Blizzard Frost cycle, and Artifact-supported Focus
have direct authority. The other seven are `ADVISORY_ONLY`. Their
completed level is preserved as `observed_level` for certificates and decision
logs, but their candidate level is `NONE`; final-deck co-occurrence alone cannot
override contemporaneous expert choices or create an LLM conflict.

Transition cards are modeled separately from winning mechanisms. This avoids
inventing a fake end-state template for a card whose expert value is primarily
stage-dependent:

| Card | Reviewed role | Expert card-vs-skip evidence |
| --- | --- | --- |
| Charge Battery | immediate block and energy | favored 134:63 in Act 1; not treated as a late-game template |
| Sunder | Act 1 frontload | favored 104:9 in Act 1; not treated as permanent scaling |
| Reinforced Body | immediate block | favored in Acts 1-3 |
| Equilibrium | immediate block and retained setup | favored in Acts 1-3 |
| Machine Learning | draw consistency | 128:14 overall |
| Capacitor | orb-capacity scaling component | 146:57 overall |
| Genetic Algorithm | early scaling block | 170:85 overall, with its win-deck module kept advisory |
| White Noise | generated-Power support | 197:60 overall; 71 wins contain White Noise plus Heatsinks or Storm |

## Template distance

Each module has one anchor slot and a small number of required support slots.
For an offered card, Winning Path compares module progress before and after
adding exactly that card:

- `CORE_ACTIVATION`: the card completes a module.
- `COMMITTED_PROGRESS`: an owned anchor exists and the card completes another
  required slot.
- `REACHABLE_ENTRY`: the card supplies the first anchor.
- `NONE`: no verified slot progress.

Only `COMMITTED_PROGRESS` and above have direct template authority, and an
`ADVISORY_ONLY` module is never promoted to that threshold. Future
completion probability is a tie-break derived from expert offer rates and the
remaining reward horizon. Cards that merely look synergistic do not receive
template evidence.

Dominant cards are limited to reviewed first copies and Act ranges. Owning the
card removes this evidence, which prevents first-copy statistics from silently
authorizing extra copies.

## Rebuilding the catalog

The raw archive is intentionally not committed. Rebuild the generated catalog
from the Spire Agent repository root:

```bash
uv run python -m spire_agent.tools.winning_path.defect_data \
  --archive /path/to/profile-0-runs.zip \
  --parameters src/spire_agent/tools/winning_path/data/defect_policy.json \
  --cards src/spire_agent/tools/data/cards.csv \
  --output src/spire_agent/tools/winning_path/data/defect_catalog.json
```

The compiler filters character and ascension, normalizes legacy internal card
IDs, extracts only standard rewards, builds pairwise preferences, derives offer
rates and horizons, and records source fingerprints and per-template support.

Generate the separate certificate analysis without rewriting the runtime
catalog:

```bash
uv run python -m spire_agent.tools.winning_path.defect_data \
  --archive /path/to/profile-0-runs.zip \
  --parameters src/spire_agent/tools/winning_path/data/defect_policy.json \
  --cards src/spire_agent/tools/data/cards.csv \
  --certificates-output \
    src/spire_agent/tools/winning_path/data/defect_certificates.json
```

`defect_certificates.json` maps every A20 Heart-winning final deck to the
currently completed modules and groups identical module sets into signatures.
It retains final deck counts and relics for diagnosis. This is offline evidence:
it is not loaded by the picker, and it does not populate the runtime `routes`
parameter until the signatures and missing modules have been reviewed.

The current 17 modules produce:

- 347 final-deck certificates and 311 exact module signatures;
- all 347 certificates with at least one completed module;
- support ranging from 18 wins for Blizzard Frost cycle to 260 wins for
  Focus and Frost defense.

The separate Focus Depth module is supported by 186 winning decks. It lets a
Frost deck recognize progress from a second Focus source without redefining a
single Focus source as incomplete.

The Blizzard Frost cycle is supported by 18 exact winning decks. It requires
an already-owned Blizzard, five Frost-generation card copies, and four
substantial draw or card-access sources. A candidate has direct authority only
when it completes both density requirements; partial progress remains visible as
`observed_level` but cannot take over the decision. This lets Coolheaded express
its combined draw and Frost value without adding a card-specific rule or
authorizing the first Blizzard from final-deck co-occurrence alone.

The 311 exact combinations are deliberately not converted into 311 runtime
routes. They show that the expanded modules explain every winning deck, but
also that exact final decks are too fragmented to serve as policy rules. Future
route work must merge signatures by reviewed mechanism and validate the merged
route against reward decisions before granting authority.

## Archive benchmark

Run the reproducible archive alignment check with:

```bash
uv run python -m spire_agent.tools.winning_path.defect_benchmark \
  --archive /path/to/profile-0-runs.zip \
  --cards src/spire_agent/tools/data/cards.csv
```

The initial catalog produced:

- 14,959 choices
- 84.26% deterministic coverage
- 54.90% agreement within deterministic decisions
- 46.26% direct agreement over all choices

After adding the reviewed draw-consistency density and keeping unrelated
expert cards outside a transition frontier, the same archive produces 89.29%
deterministic coverage, 50.79% agreement within deterministic decisions, and
45.35% direct agreement over all choices. The coverage increase is expected;
the agreement decrease is retained as an explicit regression signal rather
than interpreted as proof of worse play.

With the 15-module catalog, advisory authority enforced, and only
Artifact-supported Focus promoted after case-level review, deterministic
coverage remains 89.29%. Deterministic matches increase from 6,784 to 6,797:
31 prior disagreements become matches and 18 prior matches regress, for a net
gain of 13. This is a positive but small archive signal, not a claim that the
new templates are optimal.

After restricting Reprogram to a supported physical plan and making committed
Frost construction conflict with Focus reduction, deterministic coverage is
89.65% and deterministic matches increase to 6,913. The direct match rate is
46.21%; the change also rejects both observed orb-deck Reprogram failures.

After recognizing Auto-Shields as existing block, requiring non-starter orb
supply before Consume, and admitting Data Disk as a Focus anchor, deterministic
coverage is 89.67% and deterministic matches increase to 6,939. Of 131 changed
archive decisions, 48 become matches and 22 regress, for a net gain of 26. The
Data Disk case is additionally covered by an exact relic-aware regression test
because the archive replay cannot reconstruct relic acquisition.

Genetic Algorithm is restricted to Act 1 because a late copy has too few
combats left to become reliable scaling block. This intentionally diverges from
49 observed Act 2/3 expert picks: 88 archive decisions change, all involving
Genetic Algorithm, and deterministic coverage becomes 89.63% with 6,909
matches. The lower historical agreement is retained as an explicit cost of the
reviewed timing constraint rather than hidden by another heuristic.

The Blizzard Frost cycle keeps deterministic coverage unchanged at 89.8656%
(13,443 of 14,959 choices). Deterministic expert matches increase from 6,943 to
6,944. It changes only 11 archive states after requiring Blizzard to be owned
before density progress receives authority; the first Blizzard remains an
expert or bounded-LLM decision.

The benchmark replays reward picks in order but cannot reconstruct every shop,
event, removal, or transform from the archive. It is therefore a regression and
root-cause signal, not a gameplay-quality verdict. Expert agreement is also not
treated as optimality.

## Improvement loop

Future changes follow the same bounded Winning Path loop:

1. Record new Defect choices with exact deck-before context and replay identity.
2. Add stable historical cases and combat checkpoints to the regression set.
3. Group deterministic disagreements by evidence source and reason code.
4. State one root-cause hypothesis and change one reviewed knowledge item:
   template definition, template distance data, transition capability, expert
   evidence, or authority threshold.
5. Regenerate fingerprints and run unit tests, archive alignment, protected
   historical cases, and MCTS combat checkpoints.
6. Accept only changes with a positive signal and no protected regression.

This is the same self-improving process used by Ironclad: MCTS supplies
combat-grounded labels, deterministic replay detects regressions, and the
picker gradually approaches expert quality without training an opaque model or
expanding the resolver rule space.
