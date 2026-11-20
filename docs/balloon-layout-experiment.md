# Experiment: original-enclosure lettering retries

## Scope and provenance

This experiment follows the capability-gap audit, not an upstream implementation.
Design inputs are Tralens's own `automatic.py`, `preparation.py`, `typography.py`,
saved failed layouts, and original synthetic fixtures. No upstream code, model,
font, dependency or new service is acquired. Prior research exposure is not erased;
this is not a certified clean-room process. Existing model-rights restrictions
remain unchanged.

User-owned staged documentation deletions and changes to `.gitignore`,
`pyproject.toml` and `uv.lock` are outside this experiment and must remain intact.

## Missing capability

Native cleanup can succeed while English lettering fails because the available
frame is only a small expansion of a narrow Japanese column. For example,
`ubunchu01_06.png` has a cleanable `すげぇな` balloon whose saved translation is
not placed. Broadening cleanup or shrinking lettering is not the solution.

This is **layout only**, not a new semantic detector, grouping repair, OCR fix,
translation rewrite or artwork reconstruction method. In particular it will not
resolve the Synaptic/network balloons' separate cleanup failures.

## Independently designed hypothesis

After all existing successful edits have been reserved, retry layout failures
only when an existing native flat cleanup plan is available. Propose larger
lettering space from an original closed light region corroborated with 4- and
8-connected geometry, bounded area, complete source-ink containment, measured
flat pixels, outline support and exclusion of other detections/unexplained ink.
Connectivity alone is not semantic balloon ownership.

Derive layout obstacles from the original pixels and the **unchanged** native
erase permission. Existing committed alpha/erase reservations remain protected.
No additional pixels may be erased, no source or target strings may change, and
the requested minimum font size remains authoritative. Record both rejected and
accepted hypotheses. Keep this behind an explicit opt-in; default output must
not change.

## Cases frozen before implementation

Development: the six original pages `ubunchu01_06`, `ubunchu04_0002`,
`ubunchu04_0005`, `ubunchu06_0002`, `ubunchu03_0002`, `ubunchu05_0002`, with their
existing source-bound OCR/translation previews. Preserve their current successful
edits as protected comparisons. Do not manufacture shorter target strings.

Portable controlled positives: a closed curved balloon with narrow source text;
a nonrectangular balloon; an off-white flat surface. Negatives: adjacent balloons,
open/page-connected regions, a shared region containing another detection or
annotation, unexplained art, boundary-touching ink, genuine reservation overlap,
and stale/foreign geometry. Synthetic source/target strings are test fixtures,
not native-page accuracy references.

After freezing the implementation, evaluate the previously used 15-page corpus
subset without using it for tuning. It is a reused evaluation set, **not a new
independent held-out accuracy benchmark**.

## Acceptance / rejection criteria

- Recover at least one previously skipped full dialogue balloon at >=16px with
  the exact saved target, or reject the production experiment as ineffective.
- Every previous successful source ID, foreground, erase mask and lettering tile
  remains unchanged. Retry only after those edits, never ahead of strict work.
- No changed cleanup pixels; no new alpha over borders, art, neighbors or prior
  reservations. Independent composite/mask checks must pass.
- Negatives must reject expansion rather than clip source ink, connect regions,
  waive annotation ownership or guess missing background.
- Inspect original/before/after images visually at readable size; counts and
  containment alone do not establish quality or full-page completion.
- Run focused/full regressions and document actual gains, failures and limits.

## Separate routing experiment: explicit semantic evidence on coverage inputs

The audit also identifies a concrete wiring defect: the existing explicit
`text_mask_model` fallback can be consumed by strict units but not uncertain
coverage units. Test an additional bounded fallback route after the existing
native attempts, without changing segmentation thresholds, cleanup rules,
translation eligibility or defaults. Reserve prior strict outputs first and
share the existing single page-level inference/cache and work limits.

Acceptance is an end-to-end synthetic coverage fixture that reaches the existing
mask-association code with an injected probability array, preserves a prior
strict edit, and rejects unsupported/all-zero evidence. No actual CTD weights
will be run: their deployment rights remain unresolved. This tests routing and
source/pixel contracts only, not real learned-mask quality or artifact clearance.
Without an explicit configured mask model, no backend may be constructed.

## Implemented interfaces

- `layout_enclosures.py` derives original flat-space hypotheses from the page,
  not from a text-painted image. Four/eight-connected **usable cores** must agree
  exactly. Tiny differences at an excluded boundary fringe are recorded; an
  actual connection to another pocket/page still rejects. All pixels in the
  usable core must be explained by the measured flat background or the existing
  native erase mask. Other source/annotation geometry is never waived.
- `automatic.py` retries only previously failed minimum-size layouts, after all
  prior successful operations are reserved. Native foreground and erase masks
  are copied unchanged. It first tries comfortable typography, then zero extra
  leading inside the already inset enclosure, never a lower font floor.
- The main pipeline and offline renderer expose `--layout-policy enclosed-retry`.
  This requires `--replacement-policy confidence-gated` and best-effort rendering.
  The default remains `source-local`. The development web viewer does not yet
  expose this new option; the comparison gallery shows explicit CLI results.
- An explicitly supplied semantic mask model now has a fallback route on coverage
  inputs as well as strict inputs. Existing lazy initialization/failure caching,
  flat/periodic mask guards, source bindings and strict-first reservations are
  reused. No segmentation threshold or model-rights decision changed.

Example renderer-only invocation (no OCR/translation inference):

```sh
.venv/bin/python tools/render_preview.py auto-compose \
  --preview test-results/quality50-baseline-v1/ubunchu01_06/preview/preview.json \
  --image images/quality50-v1/ubunchu01_06.png \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --replacement-policy confidence-gated --layout-policy enclosed-retry \
  --output test-results/my-fresh-enclosure-trial
```

The output must be a fresh path. Nothing grants pixel-edit permission merely
because a translation exists, nor promotes a region to reviewed/full-utterance
truth.

## Results and decision

| Comparison | Before edits | After edits | Newly placed balloons |
| --- | ---: | ---: | ---: |
| Six development pages | 57 | 58 | 1 |
| Fifteen reused evaluation pages | 122 | 123 | 1 |

Development evidence: `test-results/enclosure-layout-development-v1/`. Only
`ubunchu01_06` changes (14 -> 15): its saved `すげぇな` target fits at **17px**
when the larger closed original space is supplied. The other five pages retain
their previous 7, 10, 8, 9 and 9 edits. A neighboring/shared region and a
4/8-inconsistent region remain rejected rather than being merged or clipped.

Evaluation evidence: `test-results/enclosure-layout-reused-evaluation-v1/` freezes
the actual source files before inspecting outcomes. Its source-set digest is
`67f8b56e7706076d1f5b7cabb0059fc56eae54f136732de5194bd4fe39f5c01b`.
Only `ubunchu08_0007` changes (5 -> 6): the exact saved `めでたしめでたし` target
fits at **17px**. The remaining fourteen composites are unchanged. This set was
already inspected in the previous loop: it supplies reused regression evidence,
not a fresh independent accuracy benchmark. No evaluation outcomes guided tuning.

The two changed original/before/after crops and full composites were visually
inspected. Borders/art appear retained, but both targets contain multiple slash-
separated translation alternatives, so the new lettering remains cluttered and
linguistically awkward. No wording was shortened, selected or rewritten to make
the result fit. Large cleanup-blocked dialogue on the user's example remains
unfixed, as do dark/toned text, many annotations and fragmented context.

`test-results/enclosure-layout-independent-audit-v1/` independently rechecks all
21 source bindings, protected pixels, outside-mask locality and alpha composition.
Every previous successful foreground/erase/text PNG is byte-identical, and both
new edits match independently recomputed **existing native** foreground/erase
masks. The global erase union naturally adds each newly committed operation;
no individual cleanup permission is expanded by the layout retry.

The semantic coverage routing fix passes end-to-end synthetic tests for accepted
and unsupported probability arrays, one backend shared across multiple uncertain
units, cached backend failure, retained translations, protected prior strict edits,
and absence of model construction when unconfigured. These are **not real CTD
quality results**. No actual CTD model was run or acquired.

Keep the layout capability **experimental and opt-in**, not a default promotion
or a full-page quality claim. Keep the explicit-evidence routing fix, but model
clearance and actual precise-mask quality are still separate unmet requirements.
The two improvements address narrow layout capacity and an integration defect;
they do not complete the audit's broader page-structure, OCR, reconstruction or
contextual-translation objectives.

## Verification

- **884 tests pass in each existing environment** (`.venv`, `.venv-paddle6`).
- Source-only verification passes with 39 optional local-evidence skips.
- All 444 default coverage and 129 default strict-only regression PNGs remain
  byte-identical; no model/setting is enabled implicitly.
- 16 new portable tests exercise enclosure positives/negatives, pipeline opt-in,
  strict-output preservation, and explicit semantic coverage routing.
- User changes to `.gitignore`, dependency files, and staged documentation
  deletions remain untouched. No license/visibility change, push, new dependencies,
  downloaded weights/fonts, reference translations or upstream code acquisition.

Local read-only comparison gallery: `http://100.121.133.88:7862/` on the existing
private Tailscale interface. It serves only allowlisted evaluation images, not
repository contents, logs or credentials.
