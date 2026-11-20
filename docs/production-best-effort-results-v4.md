# Source-bound cleanup continuation v4

**Partial improvement, not a faithful full-page translation result.** This continues
`7b69cbe`. Evidence is retained under the ignored local directory
`test-results/production-best-effort-v4/`; the previous evidence and originals remain.

## Retained approach

- Explicit CTD probabilities can narrow aggressive cleanup to source-associated
  strokes. Native connected components complete bounded outlines and their holes;
  original detection geometry, inference support, commit support and layout remain
  separate. The strict semantic/periodic route retains its existing guards.
- Coverage-only pages now load the explicitly configured text-mask model even when
  no strict semantic route was attempted. The viewer accepts `--text-mask-model`
  and passes that path through its isolated worker to `src/main.py`.
- A narrow source column with derived stroke support can receive a source-centered
  readable rectangle retry before reconstruction is abandoned for lack of placement. Exact wording, font
  coverage, the requested font floor, and existing committed pixels remain binding.
- Final cleanup can perform fresh, source-bound LaMa inference after excluding
  earlier edits. It uses its own four-call/40-second budget. Failed placement still
  commits source removal; a model failure still reaches local surface cleanup.
  Rejected broad-mask cleanplates are not reused. Native LaMa inputs never resize.
- New learned requests retain protected masks and raw backend crops as well as
  inference masks, commit masks and composited candidates, permitting independent
  verification of worker hashes and exact composition.
- Incomplete model-server responses are retained in the durable incident before
  callers can fail during teardown. Subsequent dispatch stays blocked, and explicit
  acknowledgement archives the response. The visual diagnostic also retains its
  failure and avoids issuing an unload while completion remains unknown.

These are source-bound heuristics, not proof that every predicted stroke is text
or that reconstructed artwork is authentic. No source wording or translation was
silently edited. OCR, translation and rendering remain separate stages.

## Visual trials and remaining defects

The replay series uses the same four source-bound previews as the previous v3
checkpoint. All translations are unchanged, allowing cleanup differences to be
inspected without language changes hiding regressions. The v4 trials use
`replacement_policy=confidence-gated`; the historical v3 replay used `all`.
`without-mask-final/` is the matching v4 control. A separate `without-mask-matched/`
run uses `all` and reproduces the v3 output pixels exactly on all four pages.

- `render-v1/`: tighter masks preserve more panel structure than broad detector
  polygons, but large final fallbacks remain flat patches.
- `render-v2/`: a wider readable layout retry helps narrow columns; collisions
  still send some reconstructed regions to the final independent removal pass.
- `render-v3/`: fresh final inference improves several large fill-shaped areas.
- `render-v4/`–`render-v6/`: trial native-component cuts and background routing
  exposed regressions. Mostly white collars could erase panel borders; forcing
  reconstruction also produced new glyph-like marks in small regions. These trials
  are retained as failures, not inferred successful from replacement counts.
- The additional `without-mask/` comparison showed that widening a layout before
  broad-mask reconstruction could introduce more fake lettering. That retry now
  requires successfully derived source-bound stroke support. The expanded native
  cuts and their background-routing changes were reverted together.
- **Retained: `render-v7/`**, with `source_bound_semantic_strokes_v1`,
  `crop_local_aggressive_v6_source_supported_layout`, and
  `remove_then_source_centered_patch_v4_semantic_cleanup`. All four composites and
  cleanplates, plus all 14 final-cleanup crop comparisons, were visually inspected.

| Page | Exact replacements drawn | Final cleanups committed | LaMa calls, both passes |
| --- | ---: | ---: | ---: |
| saved-example | 35 | 4 | 11 |
| ubunchu08_0007 | 19 | 2 | 6 |
| ubunchu07_0002 | 20 | 4 | 3 |
| ubunchu03_0006 | 24 | 4 | 9 |

The **98 drawn replacements** are a coverage count, not an accuracy score. Fourteen
other targets exactly match the source and retain the original lettering; one
proposal has no selected translation, and one bound target cannot be drawn.
The final 29 LaMa calls include eight calls in the independent final-removal pass.

The output still has serious OCR/translation errors: “Wgetz, make,” “P-code,”
“Wisdom,” “Top—,” “APT (Advanced Persistent Threat),” “the art of coffee,” and
“1941P Chi 1P4.” Some stylized lettering lies outside reliable source support;
small remnants, invented texture, cropped titles, intrusive captions and local art
loss remain. One exact target contains Japanese glyphs missing from the configured
English font; cleanup is committed, but it remains unavailable for exact drawing.
The blank balloon is visible in `ubunchu07_0002`. The crop comparisons show that
large rectangular occlusions became smaller reconstructed areas in several places,
but `ubunchu08_0007` still has a broken partial title and artificial texture.
`ubunchu03_0006` still has remnants around “from” and “It’s already…”, and displaced
lettering crosses artwork/panel edges. No general cleanplate-quality selector was
established. The removal requirement is therefore still not completely satisfied.

## Reader diagnostics

- Two Manga OCR presentations: **109 source proposals, 218 reads**, with agreement
  on 60 proposals. Agreement included wrong text, including high-confidence controls.
- Local Paddle crop comparison: **65 disagreements, 130 predictions**, native and
  2× presentations using the existing v6-medium weights. Some readings improve,
  others repeat errors or include neighboring text. Predictions remain diagnostic.
- Qwen visual comparison: **21 successful retained responses**, followed by one
  incomplete response. Confident errors and crop contamination persist. The original
  incomplete response was lost before the teardown fix; its duration and tokens are
  unknown. No new reader-selection or translation-prompt policy was justified.

Remote request `b85a0d050d8a4cbf940b9a245cbcf7fc` remains unresolved at
`http://100.121.133.88:11434`. The user answered **“Still running or unknown.”**
Remote inference remains paused. Local tests use a separate scheduling state root;
they do not acknowledge or bypass this incident.

## Reproduction and verification

The existing pinned checkpoint can be supplied to either renderer CLI or viewer:

```text
--text-mask-model /tmp/opencode/tralens-learned-eval-v1/models/comictextdetector.pt.onnx
```

Learned background reconstruction additionally needs the existing explicit LaMa
configuration and aggressive-removal/LaMa selection. The restored CTD bytes match
the previously approved SHA256
`1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`.
`asset-restoration.json` records the exact 94,669,756-byte acquisition. No new
checkpoint family or upstream Python implementation was introduced.

- Final working-tree suite: **1,060 passed**, 121.575 seconds, including the
  pre-existing local work. The isolated task-only candidate ran **1,015 tests**:
  975 passed and 40 skipped, 38.265 seconds. The isolated archive omits ignored
  local evidence. Both logs and the tested code hashes are retained.
- Independent artifact audit: **13,994 artifact hashes across 43 page renders**,
  including rejected trials and controls. Exact wording, unique displayed
  membership, request/mask bindings, composition and zero changes outside declared
  support passed. The final four pages include 63 checked protected requests and
  29 checked raw learned outputs. These checks do not establish bilingual accuracy
  or complete source-text removal.
- Native controls: **137,916 cleanup pixels and 141,539 lettering pixels** from
  earlier accepted routes are preserved exactly by the retained candidate.
- Matching no-CTD control: **zero pixel differences** against v3 in all 32 compared
  source/output/mask images. The initial mismatched-policy comparison is retained.
- The renderer code matches the retained replay snapshot. The task-only candidate
  preserves all 28 baseline dirty/deleted/untracked paths, with the two shared
  files reconstructed exactly. The remote-incident bytes remain unchanged.
- Viewer forwarding is covered by integration tests. **No fresh v4 browser upload
  or live translation run occurred.**

Local evidence is indexed by `test-results/production-best-effort-v4/README.md`.
`audit.py`, `compare_pixels.py`, `costs.py`, `verification.json`, and the retained
visual-review ledger contain the checks and their scope.

## Recorded inference cost

Across every completed trial/control page: **246 LaMa calls**, 1,155.757 worker
wall-seconds and 553.224 model-inference seconds; **35 CTD calls**, 41.057 seconds.
The reader diagnostics used 218 Manga calls (42.738 seconds) and 130 Paddle crop
calls (31.639 seconds). The 21 completed Qwen replies report 19.707 daemon seconds,
2,030 prompt tokens and 455 generated tokens. One unresolved request has unknown
duration/tokens. There were **zero new translation calls**. Reader times include
bookkeeping; no currency/energy price was supplied for the private runtimes.

Model provenance/distribution clearance, independent bilingual review, untouched
title validation, a sustained live soak and embedded memory plateau remain open as
documented in the preceding report. This continuation does not establish release
quality or completion of all requested quality fixes.
