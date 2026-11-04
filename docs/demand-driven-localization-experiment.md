# Demand-driven localization checkpoint

## Frozen hypothesis (before tuning)

At `c4c37e1`, 237 crop predictions on 21 reused pages include 151 predictions
for already-placeable regions. Only 41 reach cleanup; 23 attempt records are
rejected by its separate budget. This checkpoint targets wasted inference, not
linguistic accuracy or a new erasure capability.

Render natively first. Request source-bound crop evidence only for attempted,
unresolved cleanup failures that pass the existing model-free closed-enclosure
prerequisites. Allocate the existing four-retry / 1,024,000,000 raster-work limit
and crop limits before inference. Preserve vertical/multiline/length ordering.
The coordinator alone initializes the explicitly configured existing reader for
a crop-only pass. Do not rerun translation, full-page OCR, or rewrite old runs.
Keep failed, unrequested and budget-missed records visible.

Freeze originals, previews, baseline layers and approved font assets for
`ubunchu01_06`, `ubunchu04_0002`, `ubunchu04_0005`, `ubunchu06_0002`, and
`ubunchu08_0007` before implementation. They include protected dialogue, fragments,
furigana, mixed scripts, neighboring/shared balloons, dark text and effects.
Replay saved predictions on all 21 reused pages before fresh crop inference.

Acceptance: no request for an already-placed unit; preserve all 182 successful
source replacements, their exact targets, masks and expected pixels. Report calls,
predictions reaching cleanup, budget misses, execution failures and measured time
separately from visible improvements. Additional recoveries are not required or
inferred from fewer calls. Inspect original/baseline/candidate pages and crops at
reading size. No gray/art/source/pixel guard changes. References are evaluation
only. No new models, dependencies, fonts, providers, upstream implementations,
publication or push. Typography and viewer integration are separate checkpoints.

## Implementation

`localized_strokes.prerequisites()` extracts the existing light-region dominance,
closed bounds, same-enclosure, 4/8 usable-core agreement, extent and dark-outline
checks. Only member polygons need rasterizing at this stage. Cleanup still checks
all competing source polygons, whole components, exact cropped OCR agreement,
gray interiors, neighboring ink, reservations and actual pixel/alpha integrity.
No threshold or erasure permission changed.

`run_automatic(plan_localization=True)` records a `localization_request_plan` after
native/enclosure routes. It contains an exact preview revision (excluding only
optional localization evidence), source, font manifest hash, font/style/render
options, per-unit decisions and fixed call/raster/crop allocations. Prerequisite
checks have a separate bounded work allowance; failures and budget exclusions
stay visible. Planning does not read old crop outputs or construct any model.

For rendered `--source-localization` jobs, `src/main.py` now:

1. Runs ordinary OCR/translation, closing the OCR engine before translation.
2. Publishes `rendering-native/` and its demand plan.
3. Rechecks original source/font preflight, then opens the same explicitly
   configured local medium reader for **crop-only** inference. Empty plans never
   initialize a second engine. It does not call `process_images()` or translate.
4. Writes a separate `localization/preview.json`, preserving all original units,
   requests, results and raw OCR, and rechecks source/fonts before retry rendering.
5. Writes final `rendering/`; the renderer reconstructs and validates the demand
   plan before consuming predictions. Failed optional inference retains native
   successes and is reported as partial, not a quality success.

The optional pass reinitializes the reader and repeats native rendering before
cleanup retry. This is intentionally conservative about lifetime and provenance,
**not an optimized end-to-end execution path**. Timing includes this overhead;
fewer crop calls alone do not establish a page-level speedup. Preview-only legacy
localization remains an explicit evidence-gathering API, not the rendered path.

Saved-evidence replay is a separate explicit mode of `execute_requests()`. It
records zero new calls, reused prediction counts and exact origin revisions.
Missing predictions do not trigger inference. Unrequested, failed and skipped
records are retained. Old predictions remain supported by offline rendering.

The pipeline reports requested/failed/predicted/consumed counts, budget misses,
native/planning/crop/retry timings and both preview bindings. Translation is never
rerun to compare masks or typography.

## Replay and regression evidence

Before editing, five representative pages, originals, baseline PNG/JSON layers,
approved font and license bytes, and application sources were frozen under
`test-results/demand-localization-freeze-v1/` (hash-addressed blobs and manifest).
They are local evaluation artifacts, not production corrections or new assets.

Reproduce the model-free replay in a fresh directory:

```sh
.venv/bin/python tools/demand_localization_evaluation.py \
  --corpus test-results/localized-dialogue-corpus-v1/report.json \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --output test-results/my-fresh-demand-replay
```

The first 21-page replay (`demand-localization-replay-v1`) selected **23 requests**
versus the prior **237 crop calls**, with **zero already-placed requests**. Twenty
had saved predictions; all twenty reached cleanup. Three lacked saved predictions
(`ubunchu07_0005`, `ubunchu08_0010`, `ubunchu08_0011`); no inference was substituted.
Eleven otherwise eligible unresolved regions remained retry-budget exclusions.
These counts describe scheduling/replay, not 23 fresh calls or successful masks.

All **182** prior successful source replacements, exact bindings/targets and
foreground/erase/text PNGs were preserved. All **21 full-page composites were
byte-identical**. No new recovery, OCR correction, wording change or visible
typographic improvement is claimed. Existing Japanese omissions, awkward wording,
printed menus and cramped lettering remain. The network balloon still rejects
at `localized_gray_interior`; gray/art protection was not relaxed.

Five original/baseline/candidate full pages were inspected at ~620px reading
width, with native-resolution crops, under
`test-results/demand-localization-reading-review-v1/`. Candidate/baseline equality
means no new source remnants, artwork/border damage or readability changes, not
that the existing page is satisfactory. Furigana, dark text and effects stay in
the protected comparison rather than disappearing from its denominator.

The first replay took 161.12s wall time: 75.76s native rendering (including 3.82s
planning), 0.66s saved replay, and 84.05s retry rendering. Unit tests ran concurrently;
this is not a controlled performance comparison, crop-inference timing or speedup
claim. A final-source repeat and separate fresh subset are recorded below.

Default regression comparisons retain all **444 coverage** and **129 strict**
PNGs byte-for-byte. The existing focused menu comparison also retains all **249**
PNGs; this scheduling checkpoint does not promote or change its translation route.

## Tests and boundaries

**923 tests pass** in both `.venv` and `.venv-paddle6`. Source-only passes with
39 optional local-evidence skips. Nine new portable tests exercise demand-driven
coordinator execution, no redundant translations/page OCR, empty requests, budget
exclusions before inference, failed prerequisites, failed predictions and reader
initialization, stale source/preview/font/style/request plans, and explicit replay
with missing predictions. Existing localization tests now expect only the one
unresolved crop and inspect the separate derived preview.

No learned semantic mask was configured. This remains flat/native-stroke cleanup
with uncertain OCR agreement. No new model, font, dependency or provider was used;
dependency/ignore edits and staged older-document deletions remain user-owned.
Nothing is pushed or published. The experimental opt-in flag stays explicit.

This checkpoint does **not** improve transcription, choose translations, remove
slash menus, reletter successful dialogue, or add viewer controls. The subsequent
typography-only comparison must hold cleanup masks/targets fixed, allow text-layer
changes, and retain successful coverage. Viewer options and a real viewer run
remain a later checkpoint after those visible changes are validated.

## Final-source repeat and fresh subset

`test-results/demand-localization-replay-v2/` repeats all 21 reused pages against
the final application sources. Its frozen source hashes match the implementation.
Results are unchanged from v1: **23 requests, 20 reused predictions/cleanup
attempts, three unavailable saved predictions, 11 budget exclusions, 182 retained
replacements, and 21 byte-identical composites**. No new crop inference occurred.
Measured wall time was 170.21s (native 80.13s including 4.10s planning, saved replay
0.72s, retry rendering 88.65s). Unit tests/regression comparisons ran concurrently;
this is scheduling evidence, not a controlled renderer performance benchmark.

After replay passed, `test-results/demand-localization-fresh-v1/` used only the
existing explicitly configured local Paddle v6 medium reader on the five frozen
pages. Source and font preflights were checked before/after each crop pass. No
full-page OCR or translation was run. The runner and per-unit raw predictions,
crop PNGs, source-bound request plans, timings and rejection records are retained.

| Reused page | Fresh crop calls | Reaching cleanup | Crop phase seconds¹ | Retry render seconds |
| --- | ---: | ---: | ---: | ---: |
| `ubunchu01_06` | 2 | 2 | 4.52 | 8.90 |
| `ubunchu04_0002` | 1 | 1 | 0.84 | 2.15 |
| `ubunchu04_0005` | 1 | 1 | 0.99 | 3.06 |
| `ubunchu06_0002` | 2 | 2 | 1.99 | 3.46 |
| `ubunchu08_0007` | 1 | 1 | 1.43 | 2.92 |
| Total | **7** | **7** | **9.75** | **20.49** |

¹ Includes reader initialization/release and crop evidence persistence. The first
case includes cold import/initialization cost. Existing background services/load
and caches are not controlled; no old inference timing matched these conditions.
These five pages previously spent **58** crop calls. This is not a measured 58-call
versus 7-call runtime A/B, nor a fresh held-out linguistic benchmark.

All **47** existing replacements and all five composites remain byte-identical.
There were no crop-provider/runtime exceptions. One inference produced an
empty/structurally unusable prediction (`localized prediction association
contract`); it reached cleanup validation but did **not** derive a mask. Here and
in the replay totals, “reaching cleanup/consumed” means a prediction was supplied
to the bounded cleanup attempt, including validation rejection—not necessarily
a mask or a successful replacement. The remaining rejections retain exact-line,
extra-text and ownership checks. The network still rejects gray interiors.

**Decision:** accept this as a narrower scheduling improvement with output
preservation. Reject any claim of visible/OCR/wording quality gain from this
checkpoint alone. It makes no change to the motivating page's remaining Japanese,
screen text, effects, poor terminology, slash menus or narrow successful lettering.
No mask or font policy was weakened to recover them.
