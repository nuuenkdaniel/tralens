# Fixed-commit context experiment: mixed; do not promote blanket masking

The frozen comparison supports **neighbor-letter contamination as one failure
mechanism**, but not a default hide-all-detected-neighbors policy. It removes copied
glyphs in the motivating balloon and some dense labels, while losing useful panel
structure elsewhere. Production `aggressive.recover` retains its current inference
mask. No candidate, language, layout or selector improvement is advertised as a
validated production quality gain.

## Evidence and scope

Local evidence: `test-results/production-best-effort-v2/`. Start with
[the inspection index](../test-results/production-best-effort-v2/README.md) or
[native sheets and complete pages](../test-results/production-best-effort-v2/context-matrix-worker-fixed/inspection/index.html).

- Eight preselected categories, from **three reused pages and one constructed tone
  image**. These are not eight independent pages, new titles or bilingual gold.
- Target-only A versus A plus detected-neighbor polygons expanded by 3px (B).
  Two margins, 16px/48px around the fixed commit support, at native **1:1** scale.
  All crops fit 512². Image **and** inference mask use symmetric bottom/right
  padding; every transform/padding amount is recorded. No rescaling before inference.
- Identical page-coordinate commit support across each case's four slots. Original
  pixels, binding, model and preprocessing are fixed within matched pairs. English,
  backing layers and other accepted cleanup are reused, not regenerated.
- White/tone have no added neighbors: four B slots reuse their identical A request.
  **32 slots, 28 actual CPU forwards**, not 32 independent observations.
- The user approved 28 forwards, <=30 seconds each, <=840 seconds total, no retries,
  OCR, translation, review calls or downloads. The first dispatch failed before
  ONNX import/model loading due to worker helper shadowing: **zero forwards**.
  Fix `71cca44` and an explicit one-time restart authorization are retained. The
  revised freeze has identical `files`, `requests`, `cases` and `model` fields.
- Native cleanplates were inspected before English, followed by every unique full
  final page at up to 800px reading width. Reading resampling is display-only.
  Raw cleanplates use original pixels outside commit; full pages retain prior edits
  there. The retained baseline and A/B therefore differ outside the crop's target
  in the cleanplate sheet, but not in full-page composition.

The model remains pinned Carve LaMa ONNX SHA256
`1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`,
CPU ONNX Runtime 1.23.2, private noncommercial evaluation only.

## Per-case outcomes

These are provisional visual judgments, not semantic OCR scores or independent
artwork references. A smooth predicted region is not proof of recovered artwork.

| Frozen case | 16px A -> B | 48px A -> B | Remaining problem / reading-size consequence |
| --- | --- | --- | --- |
| Copied glyphs: `ubunchu08_0007`, row 2 | Copied dark columns largely disappear. | Strong removal of the copied columns; A has conspicuous repeated glyphs. | The left source column is outside commit and survives in both B pages. English is more readable, but source membership/removal is incomplete and narrow wrapping persists. |
| Native white control: `ubunchu04_0002`, row 8 | Identical A/B request, reused. | Identical A/B request, reused. | Clean/readable at page size; faint learned tonal variation is not an improvement over the accepted exact flat fill. This is no evidence for B specifically. |
| Border/balloon: `saved-example`, row 13 | Both broadly clean; no meaningful B gain. | Both broadly clean with balloon/panel continuity retained in the inspected region. | Existing narrow five-line English remains. Tiny plate differences do not establish a useful policy gain. |
| Constructed tone | Identical A/B, uneven reconstructed pattern. | Identical A/B, somewhat lower reference error, still irregular. | Mask MAE **22.26 -> 19.52** for 16px -> 48px on known clean tone (0–255). Repeated texture phase is not restored faithfully; no B treatment effect. |
| Outlined title: `ubunchu08_0007`, row 20 | Fewer glyph-like copies, but a conspicuous invented gray crosshatched patch. | Fewer copied strokes, with a pale/washed speculative patch. | Letter bases and the rest of the title lie outside support. Wording remains the saved erroneous APT expansion. Neither complete-page version is acceptable as faithful translated art. |
| Dense labels: `saved-example`, row 19 | B removes A's invented tab/glyph fragments; useful local gain. | A already much cleaner; B is similar, with small diagonal/seam artifacts. | Source below the committed mask remains. Exact tiny single-line English is unchanged. Both margin and neighbor policy matter; they are not interchangeable. |
| Artwork contact: `ubunchu08_0007`, row 9 | Both invent white shapes over dark fabric. | Both still invent a white patch; B changes its shape. | No useful reconstruction fix. Remaining protected glyph fragments and outlined English persist. Contamination is not the only failure mechanism. |
| Protected-neighbor case: `saved-example`, row 23 | B reduces noisy invented strokes but loses useful boundary information and cuts through the bottom border. | A continues the vertical panel border; B erases much of it and interrupts neighboring structure. | Prior accepted English/cleanup pixels stay exact, but B harms structural continuity inside commit. Both learned variants beat the retained Telea smear in some respects; that does not make B better than matched A. |

No whole-page output was retuned, and no best-of-four oracle was installed. In
particular the protected-neighbor case was originally classical under the page's
learned quota. This direct candidate experiment does **not** validate the page
scheduler's choices, quota distribution or an automatic context selector.

## Costs and independent checks

| Case | Actual forwards | Load s | Inference s | Worker wall s | Maximum RSS KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copied glyphs | 4 | 8.626 | 8.888 | 18.421 | 1,188,428 |
| White control | 2 | 4.374 | 4.437 | 9.289 | 1,089,404 |
| Border | 4 | 8.584 | 8.820 | 18.318 | 1,199,108 |
| Tone | 2 | 4.329 | 4.415 | 9.188 | 1,155,676 |
| Outlined title | 4 | 8.702 | 8.834 | 18.488 | 1,220,456 |
| Dense labels | 4 | 8.658 | 8.887 | 18.476 | 1,221,820 |
| Artwork contact | 4 | 8.632 | 8.832 | 18.374 | 1,156,240 |
| Protected neighbor | 4 | 8.593 | 8.883 | 18.442 | 1,231,108 |
| **Total** | **28** | **60.497** | **61.998** | **128.995** | **peak 1,231,108** |

Maximum individual wall time: 4.676s. RSS is per-process peak (maximum about
1.174 GiB), not cumulative memory. Wall excludes evaluator setup and inspection.
All 28 calls succeeded after the separately retained zero-forward startup failure.

`tools/inspect_reconstruction_context.py` independently checked **220 frozen file
hashes**, every native crop/commit transform, all 32 slots including reuse, exact
full-page recomposition and bindings. Every slot has **zero changed pixels outside
commit and zero changed protected prior-edit pixels**. Final English uses exactly
the retained text layer. These locality checks establish pixel ownership, not
complete source removal or faithful art. Output hashes and numerical pairwise
differences are in `inspection/artifact-hashes.json` and `verification.json`.

## Failure attribution and next work

1. **Membership/extent:** uncommitted dialogue columns and title continuations remain
   irrespective of B. Inference masking cannot authorize their removal or bind new
   English. Preserve raw polygons separately from any future derived extent.
2. **Context:** dense-label A improves with 48px, while copied-glyph A does not.
   More context can add evidence and contamination. Neither margin wins globally.
3. **Scale:** held at native 1:1; the matrix establishes no preferred global scale.
4. **Reconstruction:** dark-fabric shapes and tone distortion persist with B. Larger
   masks can hide useful lines and do not guarantee a better prediction.
5. **Selection:** the existing residual/footprint/backend heuristic is not an
   aesthetic selector. This trial evaluates candidates only; it does not validate
   choosing the visually best mask/context automatically.
6. **Presentation/language:** unchanged English layers expose narrow wrapping,
   alternative menus, wrong scope/terms and source-left-behind. None are fixed by
   a different cleanplate.

Next technically justified reconstruction step is an offline design for
structure-aware neighbor exclusion, with line/border controls and independent
source-extent evidence, before any new frozen inference allocation. Do not repeat
these 28 calls or tune a case-specific mask. The approved allocation is exhausted.

## Rejected presentation trial (zero inference)

Failure addressed: aggressive frames can wrap exact English too narrowly despite
nearby unused white space. Candidate assumptions: a closed light interior is a
presentation hypothesis, not a balloon identity or permission to erase.

The trial intersected 4/8-connected light envelopes, rejected open/cropped or
ambiguous regions, eroded the envelope by 4px, and excluded foreign protected ink.
It compared the existing frame with this alternative using the existing bounded
typography score (font size, line balance, orphan penalties). Wording, font floor,
cleanup, ownership and reservations were fixed. Cost: two local component passes
and at most two typography routes per <=512² crop. Expected failures: connected
white artwork, open balloons, dense neighboring English, and typography scores
that favor a slightly larger font without better reading flow.

Frozen classical-only baseline/candidate outputs on seven reused pages retained
all native successes. Replacement counts stayed **25, 8, 13, 14, 13, 11, 19**.
Six final pages are pixel-identical. Only saved-example row 13 changed **18px to
19px**, with the same five lines; 3,889 final pixels changed. Source, cleanplate,
erase mask and exact transcript files are identical on all seven pages. Full-page
review showed no useful coverage or wider-flow gain. Rejected for production;
code retained in `tools/experimental/recovery_layout.py`, with the exact evaluated
source snapshot and all artifacts in `layout-candidate-code/` and `layout-*`.

Recorded aggressive-pass time totals were about 7.46s baseline / 8.45s candidate;
these are single-run observations, not a benchmark. Whole-run/RSS timing was not
instrumented for this offline trial. No models were invoked.
