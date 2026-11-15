# Corroborated source extents v7

Continues `e92f516`. The previously surviving **Man in APT-Man is removed** using
a separately bound crop prediction. **The title background is still artificial,
and serious OCR/translation errors remain.** Accepted local output:
`test-results/production-best-effort-v7/render-v3/`.

## Change

The original title detection read `APT-` and stopped inside M. Neither its source
polygon nor page CTD probabilities included the rest of Man. The existing opt-in
`source_localization=True` flow now requests wider original-image context for
eligible large display lines after closed-stroke prerequisites fail. Eligibility
uses source geometry, not page names, known answers or corrected readings.

The coordinator still dispatches crop OCR; rendering consumes saved evidence.
Rendering accepts only a unique, aligned prediction that overlaps the source and
matches or continues its reading. Competing source regions, ambiguous matches,
crop-edge truncation and extents exceeding native reconstruction scope reject.
The observed reading remains an alternative. Raw OCR, source polygons, selected
English and translation bindings are never rewritten by this cleanup route.

Both aggressive and final-removal passes use the **whole corroborated polygon**,
with two-pixel edge completion and protected output subtracted. Merely extending
beyond the old source support left part of M in the controlled trial. Extra
cleanup context does not change original ownership or English placement geometry.
LaMa receives native crops no larger than 512 pixels per side; source pixels are
not resized to fit it. Inference, commit, protection and layout masks stay separate.

Visual review also exposed a fallback-routing defect: a more precise localized
cleanup changed a later collision, allowing an eight-call-budget-exhausted
classical smear to commit before final reconstruction. Such candidates now remain
pending for the existing final-removal pass and its separate budget. That pass
builds a fresh bound request; if reconstruction is unavailable, it still removes
source ink with a local surface. Failed English placement cannot cancel removal.

Policies: `unresolved_cleanup_crop_demand_v2_extent`,
`crop_corroborated_display_extent_v1`,
`crop_local_aggressive_v10_corroborated_extents`, and
`remove_then_source_centered_patch_v7_corroborated_extents`.
Legacy crop-demand planning, including its budget-check order, remains supported.
Source localization remains opt-in. TG27, its prompt, selected translations and
the existing reconstruction limits retain their behavior.

## Experiments and visual review

- Three local CTD context trials still missed most of Man; that route was rejected.
- Two local Paddle trials supplied different extents: 192px context read APT-Man,
  while 384px read APT-Ma. Both predictions and native crops are retained.
- Two LaMa trials compared outer-edge-only and full-extent cleanup. Only the latter
  removed the diagnosed title. It also removed chest detail and produced a pale patch.
- Four native controls requested seven coordinator crop calls: five existing
  closed-stroke candidates and two large-line extent candidates. All seven calls
  completed. A wrapper error between pages was corrected without repeating calls.
- The title extent was accepted. The large sound-effect candidate on
  `ubunchu07_0002` lacked a unique aligned extension and was rejected. Its bad
  reading `1941Pチ1P4` remains unchanged.
- All four first-replay composites/cleanplates and nine comparison cards were
  inspected. The title improved, but saved-example's Wisdom region regressed to
  conspicuous classical smears; that replay was rejected despite passing audits.
- After the routing correction, both saved-example outputs and ten comparison
  cards were inspected. Wisdom's regression and cable-edge blur were removed.
  Smears around Success and Three also disappeared. Success now uses fresh
  reconstruction; Three uses a plain final surface after the final budget is spent.
- Final replay cleanplates, English layers and composites are pixel-identical to
  those reviewed results: saved-example equals the corrected second replay, and
  the other three pages equal the first replay. Thirteen final cards are retained.

| Page | Exact replacements | Final removals | LaMa calls | Changed clean pixels vs v6 |
| --- | ---: | ---: | ---: | ---: |
| saved-example | 35 | 6 | 12 | 8,475 |
| ubunchu08_0007 | 19 | 2 | 7 | 51,437 |
| ubunchu07_0002 | 20 | 4 | 3 | 0 |
| ubunchu03_0006 | 24 | 4 | 9 | 0 |

All **98 replacements** retain their exact selected wording. The title's English
layer is unchanged. Saved-example has a slightly repositioned seven-line Synaptic
balloon and changed Success/Three placement. Overlapping cards and changed-pixel
counts are not independent quality successes.

## Remaining defects

The title has lost hatching and a conspicuous pale reconstruction with irregular
boundary marks. Success's backing cuts a border; Three's flat patch still cuts
sound-effect/art details. Japanese interface lettering, source fragments,
intrusive captions, white backings and other damaged artwork remain. The exact
target `Try adding “Of” after “aこPー”.` has its source removed but remains undrawn
because the configured font lacks coverage.

Language failures including “Wgetz, make,” “P-code,” “Wisdom,” “Top—,”
“APT (Advanced Persistent Threat),” “the art of coffee,” and “1941P Chi 1P4” remain.
Crop evidence is not automatically promoted into a corrected translation.
Remote request `b85a0d050d8a4cbf940b9a245cbcf7fc` remains unresolved; remote
inference stays paused and its durable incident bytes are unchanged.

## Verification and cost

- Working-tree suite: **1,081 tests passed**, 119.941 seconds, isolated scheduling
  state. Nine new tests cover extent association/rejection, stale bindings,
  legacy planning, prior-edit protection, native crops, placement-independent
  removal, budget deferral and the real coordinator with mocked engines.
- Independent final/control audit: **2,051 artifact hashes**, seven crop receipts,
  two controlled reconstruction trials, 63 protected requests, 31 raw learned
  page outputs and **55 cleanup/placement transactions**. Exact recomposition and
  zero outside-support changes pass. Complete accepted extent inclusion is checked.
- Earlier accepted **137,916 cleanup pixels and 141,539 lettering pixels** remain
  exact. Original images, fonts, raw OCR/alternatives and selected bindings remain
  unchanged. The 28-path working/index baseline and **19,270 historical evidence
  files** are checked before committing the task-only tree.
- All diagnostics, rejected runs, controls and final replay: **74 LaMa calls**,
  zero failures, **352.802 worker seconds / 168.641 inference seconds**;
  **16 CTD calls**, **18.005 seconds**; **nine crop OCR calls** (two diagnostic,
  seven coordinator), zero inference failures. Final replay alone used 31 LaMa
  calls. No page OCR, translation, remote calls or new asset acquisition occurred.
  Existing local CPU runtimes were used; no currency or energy rate was supplied.

The ignored evidence README contains reproduction commands. `visual-review.md`,
`costs.json`, `render-v3-audit.json` and `render-v3-comparison.json` retain the
observations and receipts. Isolated task-only tests and tested/committed-tree
equality are recorded in `verification.json` and `commit.json`. Independent
bilingual review, untouched-page validation, sustained live soak, memory plateau
and distribution provenance remain open.
