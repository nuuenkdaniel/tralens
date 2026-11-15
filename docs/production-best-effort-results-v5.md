# Native-line cleanup and inference context v5

Continues `f49bd79`. The retained result improves source removal and one intrusive
caption. **Source remnants, damaged backgrounds and serious language errors still
remain.** Local evidence is under `test-results/production-best-effort-v5/`.

## Changes

- Complete bounded native characters within narrow source lines when CTD misses
  disconnected trailing characters. Measure slanted lines using their rotated
  geometric thickness; all source pixels and masks stay in native coordinates.
  Broad detection rectangles still require nearby learned support.
- Check a native component's physical source containment before subtracting
  protected pixels. Protection cutting across one part of a character no longer
  prevents completion of its remaining outline and interior.
- Hide nearby predicted lettering from LaMa using an **inference-only mask**.
  The separately bound commit mask still controls every output pixel. Neighboring
  text and earlier edits can be hidden from inference while remaining protected
  from composition. Both reconstruction passes retain the separate masks and raw
  backend output for independent verification.
- Retry a successful but tall, automatically wrapped recovery layout using the
  existing source-centered rectangle search. Accept it only with fewer lines,
  shorter ink height, no smaller font, and bounded source-centered displacement.
  Explicit line breaks are respected. This retry requires derived stroke support;
  failed retries retain the original successful layout.

Exact selected wording, glyph coverage, font floors, unique source membership and
independent removal after failed placement remain binding. No OCR reading,
translation, prompt, source polygon, model family or reconstruction budget changed.

## Experiments and visual review

Twenty native-crop reconstructions compared physical completion, narrow-line
completion, inference-only text masks and wider native context on five difficult
regions. All five comparison cards were inspected. Wider context added little and
was not adopted. The retained implementation uses one native-component pass.

Two subsequent full-page replays used the four unchanged, source-bound previews
and `replacement_policy=confidence-gated`, matching the v4 baseline. Retained:
**`render-v2/`**. All four final composites and cleanplates were inspected, along
with all 36 changed-region v1 comparison cards and the changed v2 caption crops.
The other three v2 composites, all four cleanplates, and all four erase masks are
pixel-identical to v1. Rotation-aware measurement did not change these four
cleanplates; its missed slanted-character case is covered by a synthetic regression.

Observed improvements:

- `ubunchu07_0002`: removed the disconnected Japanese tail below “Congratulations!”.
- `saved-example`: fewer copied strokes around “Okay.” and the wired-connection
  label. Small boundary marks remain.
- `ubunchu03_0006`: reduced fragments around “from” and “It’s already…”. The
  “First, let's run a security scan.” caption now occupies four lines instead of
  six and no longer crosses the top panel border. It still overlaps some artwork.
- `ubunchu08_0007`: fewer dark remnants in the removed title and subtitle. The
  gray/white reconstructed block remains conspicuous, with thin boundary marks
  and a partial original title.

| Page | Exact replacements drawn | Final removals | LaMa calls |
| --- | ---: | ---: | ---: |
| saved-example | 35 | 4 | 11 |
| ubunchu08_0007 | 19 | 2 | 7 |
| ubunchu07_0002 | 20 | 4 | 3 |
| ubunchu03_0006 | 24 | 4 | 9 |

The total remains **98 replacements**. Fourteen targets equal the source, one
proposal lacks a selected translation, and one exact mixed-script target lacks
font coverage. Its source is removed but its balloon remains blank. Replacement
counts and dark-pixel counts do not establish complete removal or accuracy.

## Remaining defects

Small source/protected-boundary fragments persist. Unreliably detected lettering
and larger source extents still survive, including some titles and interface text.
Opaque backings, smoothed patches, partial strokes, lost local art and captions
crossing artwork remain. The new completion can remove bounded art within a source
line; the sparkle near “1941P Chi 1P4” is smaller and book/label detail is further
reduced. The evidence retains these losses. No automatic cleanplate-quality
selector or proof of complete source removal was established.

The existing language errors, including “Wgetz, make,” “P-code,” “Wisdom,” “Top—,”
“APT (Advanced Persistent Threat),” “the art of coffee,” and “1941P Chi 1P4,” remain.
TG27 and the existing OCR/translation coordination are unchanged. Remote request
`b85a0d050d8a4cbf940b9a245cbcf7fc` remains unresolved; remote inference stays paused.
Its durable incident has the same SHA256 as the v5 baseline. No acknowledgement,
remote inference, or new translation request occurred.

## Verification and cost

- Working-tree suite: **1,069 tests passed**, 119.426 seconds, with an isolated
  `TRALENS_STATE_ROOT`. Nine new regressions cover missed native characters,
  partial protection, slanted geometry, broad-box exclusions, both inference-mask
  call paths, exact native origins, compact layouts and explicit line breaks.
- Independent page audit: **2,631 artifact hashes across eight renders**, exact
  selected wording and recomposition, valid request/mask bindings, and zero changes
  outside declared support. The final four pages include 63 protected requests and
  30 retained raw learned outputs.
- Earlier accepted **137,916 cleanup pixels and 141,539 lettering pixels** remain
  exact. Existing user work and staged deletions are checked against the 28-path
  baseline before creating the task-only commit.
- Total experiment cost: **80 LaMa calls**, zero failures, 376.374 worker seconds
  and 180.446 model-inference seconds; **eight CTD calls**, 8.620 seconds. Zero new
  OCR, remote or translation calls. These use the existing private CPU models;
  no currency or energy rate was supplied.
- The exact previously approved CTD/LaMa checkpoints and ten pinned runtime wheels
  were restored after their temporary directories disappeared. Sizes and hashes
  match the retained acquisition records. The failed copied-Python bootstrap is
  retained; the working isolated environment uses a symlinked interpreter.

Reproduction uses `test-results/production-best-effort-v4/replay.py` with a fresh
output directory and the existing explicit reconstruction/CTD paths. Each replay
retains its rendering implementation snapshot. `audit.py`, `costs.py`,
`visual-review.md`, and `README.md` in the v5 evidence directory describe the local
verification. Independent bilingual review, untouched-page validation, sustained
live soak, memory plateau and distribution provenance remain open.
