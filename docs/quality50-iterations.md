# Fifty-page current-model development loop

The user requested autonomous run/inspect/plan/fix/retest loops with tested
checkpoint commits and a stop when useful gains become small or unreliable.
Only the existing Paddle v6 medium exports, Manga OCR snapshot,
`qwen3.8:latest`, `translategemma:27b` and licensed local fonts are used.

## Dataset and frozen baseline

Exactly **50 additional unique Japanese original PNGs**, excluding existing
basenames and content hashes, were acquired from eight Ubunchu chapters at source
revision `f3c6473206a8a8b6f1a946e8c60b4cdaff5ea71b`. Selection was deterministic
SHA256-ranked round-robin by chapter, excluding credit pages, not selected on
output quality. Original URLs, hashes, sizes, eight English license records and
the acquisition runner are in ignored `images/quality50-v1/`. Total acquisition:
22,500,726 bytes. No implementations, translation references, fonts, models or
dependencies were downloaded.

Declared artwork license: CC BY-NC 3.0; attribution Hiroshi Seo / ASCII MEDIA
WORKS Inc. Retained for local noncommercial evaluation, not training or
redistribution. This does not select/change the project license. All pages are
from one series: more pages are not equivalent to broad domain diversity.

The acquisition-time split reserves 35 development pages and 15 initially
uninspected evaluation pages in `split.json`. Evaluation outputs must not guide
tuning before the implementation is frozen; inspecting/tuning them retires that
status. There are no independent Japanese/English references, so execution,
coverage and visual observations are not accuracy measurements.

`test-results/quality50-baseline-v1/` freezes the actual source code from commit
`e69dc1c` in `frozen-code/`, records its archive hash and runs every original
serially through the real CLI. Active working-tree changes cannot alter that
baseline. Every page retains its command, log, original binding, preview,
translations and any clean/composite/intermediate rendering layers. The shared
Ollama reader/translator runs are not parallelized.

## Loop 1: a detector-edge integration failure

The expanded development run exposed a genuine execution failure on
`ubunchu01_11.png`: source ID 29 (`おしまい。`) has a polygon vertex at exactly
`y=1635`, the continuous bottom edge of the 1635-pixel page. OCR's source contract
accepts that coordinate, but rendering incorrectly required `< height` and
rejected the entire page before any image edit.

`natural.polygon_mask(..., evidence=True)` now accepts exactly width/height as
continuous original-image edges. It retains the original polygon unchanged and
rasterizes into the existing canvas. Coordinates beyond the edge, negatives,
booleans, nonfinite values, self intersections and degenerate polygons remain
rejected. Reviewed edit/mask geometry still requires actual pixel coordinates
strictly below width/height. This is not broad polygon clipping or a cleanup
permission change.

A fresh real renderer run using the frozen OCR/translation preview produces
**8/19 replacements**, instead of an execution failure:
`test-results/quality50-edge-rerender-v1/`. Model visual inspection against the
full original found legible balloon edits, preserved faces/frame/art and a clean
“The end.” footer. Difficult text over artwork and several other regions remain
Japanese; this is a partial candidate, not a finished page.

Portable regressions cover exact-edge rasterization, unchanged source geometry,
true out-of-page rejection, and an end-to-end pipeline with an edge detection
that must not block two unrelated successful replacements.
The exact staged checkpoint passed 56 renderer/pipeline tests in an isolated
source snapshot (`test-results/edge-checkpoint-staged-tests-v1/`).

Color-cover preflight rejections are separate from this bug. Some new originals
contain embedded ICC profiles and one has genuinely nonopaque alpha (220–255).
They are retained unchanged and reported as unsupported inputs, not silently
stripped of profiles, flattened against a guessed matte or counted as successful
pipeline runs.

## Loop 2: reserve committed output, not unused layout space

A same-model, same-preview ablation found that a valid native cleanup on
`ubunchu05_0002.png` was being deferred because a previous operation reserved its
entire unused placement region. The remaining mask could fit only 14px lettering,
below the requested 16px floor. The actual previous lettering and cleanup occupy
far less space than that mask; its whitespace is not semantic ownership.

Confidence-gated fallback placement now reserves **actual committed text alpha
and native erase pixels plus a 2px collar**. Complete source foreground, erase
masks, original obstacles and requested minimum size remain unchanged. Real
cleanup/ink overlap still rejects; previously unused whitespace can be considered
for a later exact-glyph layout. The default route and initial symmetric strict
collision rejection remain unchanged. This does not grant new erasure permission.

The recovered paragraph fits at **17px**, with unchanged target wording and native
cleanup, increasing this page from 8 to 9 replacements and covering four more
bound main IDs (24–27). Visual inspection found retained balloon/frame edges,
hair/face/art and separate lower-left lettering without overlap. The result is
still dense, and two annotation-bearing dialogue units remain untranslated in
this renderer-only experiment.

All five existing development previews were rerendered with exact OCR and
translation results held constant in `test-results/quality50-layout-five-v1/`.
No previously edited source ID was lost. Existing foreground/erase PNGs are
byte-identical on all five pages. Two composites are entirely byte-identical;
two others have small safe placement shifts at unchanged font sizes, visually
inspected against the old output. The fifth gains the paragraph above. The
default-route 444 coverage and 129 strict-only PNGs remain byte-identical.

Portable tests cover exact occupancy/collar pixels, reuse of unused whitespace,
unchanged foreground/erase, real committed overlap rejection, invalid text tiles
and exclusion from the default route.
The exact staged checkpoint passed 69 renderer/pipeline tests in an isolated
source snapshot (`test-results/occupancy-checkpoint-staged-tests-v1/`).

## Loop 3: paired reread composition and a high-score script audit

Primary paired OCR previously inserted a newline between every detected visual
line, even when Japanese text continued directly across columns. It now composes
the exact separately retained visual strings without artificial newlines, like
the existing raw/mixed-line input. Old saved newline-based evidence still
validates without rewriting its text. This does not infer word spacing between
Latin fragments or certify full sentences.

A narrow routing hint, `[A-Za-z]ー[A-Za-z]` after comparison normalization,
requests a paired audit even when the raw score is high. It is not an expected
word list or automatic glyph substitution. Existing enclosure, membership,
reader agreement and stage budgets remain necessary. If the primary block fails,
the mixed fallback must independently reread the suspicious line rather than
passing its high raw score through. Changed text never inherits that score.

In the real browser run on the existing chapter-06 development page, Paddle's
`Koaーaの` is corroborated as visual `Koalaの` / Manga `Ｋｏａｌａの`, alongside
three other paired main lines. The saved target and visible lettering change
from “KarmicKoa-a” to “KarmicKoala”. The missing word space remains a limitation;
the code does not silently supply the expected software name. Original Paddle
strings, rejected alternatives, exact visual lines and source-pixel bindings
remain saved. Portable tests cover composition drift, old evidence compatibility,
high-score paired routing, disagreement, exhausted budgets and raw-score
separation. This route is still inside opt-in quality rereads, not the default.

Five fresh real browser jobs in `test-results/current-loop-viewer-v1/` completed
as partial candidates with zero browser/pipeline errors or translation-budget
exhaustion. All five composites were visually inspected. Their source/request/
result consistency was checked offline, separately from judging their quality.

## Rejected loop: narrow reading-annotation associations

A bounded experiment required an already unique tentative parent, the same
original enclosure, separate small original kana components, raw score >=0.70,
and agreeing native/isolated Manga readings. It preserved original proposals and
only enabled uncertain translation, never strict promotion or coarse erasure.
On the existing chapter-05 development page it corroborated `おもちゃ` and
`あきは`, producing two additional translation previews, 15 -> 17. But neither
became a new image edit: native cleanup still rejected unsupported gray interior
pixels. The target for `相変わらずね秋波のヤツっ` was “Still flirting, huh?”,
misinterpreting the character name rather than using the supported reading.
Execution/translation counts alone would have overstated this result.

Across **all 31 available new development previews**, the experiment recovered
**zero additional associations**. Tiny, non-kana and ambiguous cases remained
blocked. The extra code and weakened ownership restriction were not justified
by broader visible gains. The experiment and its six tests were selectively
removed from production, retaining source snapshots, the combined patch,
originals, probes and actual browser evidence under
`test-results/annotation-experiment-retired-v1/` and the earlier probe folders.
No user work was reverted. Existing annotation ownership guards remain intact.
Its old previews require that retired implementation for replay; they are not
claimed as final-checkpoint translations or renders.

## Frozen 50-page renderer comparison

All 50 baseline attempts finished: **42 partial candidates, 4 rendering execution
failures and 4 rejected originals**, with no translation-budget exhaustion.
The 46 accepted originals produced 965 proposed units and 927 saved translations
(256 strict and 671 uncertain coverage previews). There are 38 untranslated
proposed units, plus the four unsupported whole inputs; neither should be
confused with the much larger number of translated regions left unedited.

`test-results/quality50-renderer-eval-v1/` freezes committed **`2e8c779`**, archive
SHA256 `378ffbcef95b1f3c3db8d523341e4fef1a50fdb9071ce8a76e8fd3b0b9f5a935`.
It reuses the exact original files and baseline OCR/translation previews. This
is a renderer-only comparison, **not** a second 50-page end-to-end inference run
of loop 3. Evaluation outputs were inspected only after that renderer checkpoint
was frozen, and did not guide later renderer tuning.

| Split | Originals | Renderer inputs | Rejected inputs | Replacements before -> after | Identical composites |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 35 | 31 | 4 | 251 -> 287 | 17 |
| Evaluation | 15 | 15 | 0 | 122 -> 122 | 9 |
| Total | 50 | 46 | 4 | **373 -> 409** | 26 |

All four execution failures recover partial candidates: chapter 01 page 11
gets 8 replacements, chapter 04 page 11 gets 9, and chapter 05 pages 8/11 get
10/7. The occupancy change adds only two further replacements in the new corpus:
`ubunchu01_06.png` (13 -> 14, “Oops.”) and `ubunchu06_0006.png` (9 -> 10,
“By the way... / Speaking of something else...”). Both are visually checked
against originals. These are small development gains; the evaluation subset
shows **no replacement-count gain**.

The independent artifact audit in `test-results/quality50-final-audit-v1/`
recomputes source hashes, exact preview bindings, cleanup/text mask containment,
protected-pixel preservation and final alpha composition for every accepted page.
**No previously edited source ID is lost. All previously committed foreground
and applied-erase PNGs are byte-identical.** Every accepted output passes those
pixel checks; this does not prove semantic ownership or linguistic correctness.

Model visual inspection covered all changed-region original/baseline/final
comparison sheets, the four restored full pages, the five fresh development
composites, and additional evaluation full-page samples (07/0001, 07/0007 and
08/0006). Placement changes remain contained and legible, but the samples expose
untranslated stylized effects, screen/blackboard text, leftover detached glyphs,
split balloons, wrong names and awkward/incomplete English. The accepted color
cover `07/0001` has **zero edits**, not a successfully translated page. Even some
restored pages retain Japanese next to translated fragments or a footer glyph.
No page is certified as a finished, fully translated image.

### Per-original outcomes

Units/translations are baseline counts, unchanged in this renderer comparison.
“Recovered execution” still means a partial output. A dash denotes rejected
original input, not zero text or a successful unchanged page.

| Original | Split | Units | Translations | Before edits | After edits | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ubunchu01_01.png | dev | — | — | — | — | rejected input |
| ubunchu01_03.png | dev | 26 | 23 | 8 | 8 | partial |
| ubunchu01_06.png | dev | 40 | 39 | 13 | 14 | partial |
| ubunchu01_08.png | dev | 21 | 20 | 11 | 11 | partial |
| ubunchu01_09.png | dev | 39 | 39 | 10 | 10 | partial |
| ubunchu01_11.png | dev | 19 | 19 | 0 | 8 | recovered execution |
| ubunchu02_01.png | dev | — | — | — | — | rejected input |
| ubunchu02_05.png | dev | 23 | 21 | 10 | 10 | partial |
| ubunchu02_06.png | dev | 30 | 30 | 12 | 12 | partial |
| ubunchu02_07.png | dev | 16 | 16 | 8 | 8 | partial |
| ubunchu02_08.png | dev | 19 | 19 | 10 | 10 | partial |
| ubunchu02_10.png | dev | 52 | 52 | 4 | 4 | partial |
| ubunchu02_11.png | dev | 10 | 10 | 6 | 6 | partial |
| ubunchu03_0003.png | dev | 28 | 25 | 15 | 15 | partial |
| ubunchu03_0006.png | dev | 25 | 25 | 12 | 12 | partial |
| ubunchu03_0007.png | dev | 18 | 16 | 9 | 9 | partial |
| ubunchu03_0009.png | dev | 21 | 20 | 9 | 9 | partial |
| ubunchu03_0010.png | dev | 24 | 23 | 5 | 5 | partial |
| ubunchu03_0011.png | dev | 19 | 18 | 12 | 12 | partial |
| ubunchu04_0001.png | dev | — | — | — | — | rejected input |
| ubunchu04_0003.png | dev | 26 | 24 | 9 | 9 | partial |
| ubunchu04_0006.png | dev | 19 | 18 | 11 | 11 | partial |
| ubunchu04_0008.png | dev | 23 | 22 | 12 | 12 | partial |
| ubunchu04_0011.png | dev | 14 | 14 | 0 | 9 | recovered execution |
| ubunchu05_0003.png | dev | 27 | 23 | 10 | 10 | partial |
| ubunchu05_0006.png | dev | 16 | 16 | 8 | 8 | partial |
| ubunchu05_0008.png | dev | 18 | 17 | 0 | 10 | recovered execution |
| ubunchu05_0009.png | dev | 13 | 12 | 8 | 8 | partial |
| ubunchu05_0010.png | dev | 15 | 15 | 9 | 9 | partial |
| ubunchu05_0011.png | dev | 20 | 20 | 0 | 7 | recovered execution |
| ubunchu06_0001.png | dev | — | — | — | — | rejected input |
| ubunchu06_0003.png | dev | 18 | 18 | 12 | 12 | partial |
| ubunchu06_0006.png | dev | 15 | 15 | 9 | 10 | partial |
| ubunchu06_0009.png | dev | 13 | 13 | 7 | 7 | partial |
| ubunchu06_0010.png | dev | 3 | 3 | 2 | 2 | partial |
| ubunchu01_07.png | eval | 19 | 19 | 10 | 10 | partial |
| ubunchu04_0009.png | eval | 14 | 14 | 6 | 6 | partial |
| ubunchu06_0004.png | eval | 7 | 7 | 6 | 6 | partial |
| ubunchu07_0001.png | eval | 13 | 12 | 0 | 0 | partial |
| ubunchu07_0002.png | eval | 22 | 22 | 9 | 9 | partial |
| ubunchu07_0003.png | eval | 23 | 21 | 11 | 11 | partial |
| ubunchu07_0005.png | eval | 24 | 23 | 13 | 13 | partial |
| ubunchu07_0006.png | eval | 17 | 17 | 11 | 11 | partial |
| ubunchu07_0007.png | eval | 21 | 20 | 6 | 6 | partial |
| ubunchu08_0003.png | eval | 16 | 15 | 8 | 8 | partial |
| ubunchu08_0004.png | eval | 11 | 10 | 7 | 7 | partial |
| ubunchu08_0006.png | eval | 34 | 31 | 7 | 7 | partial |
| ubunchu08_0007.png | eval | 27 | 25 | 5 | 5 | partial |
| ubunchu08_0010.png | eval | 21 | 20 | 11 | 11 | partial |
| ubunchu08_0011.png | eval | 26 | 26 | 12 | 12 | partial |

## Retained-checkpoint verification

After removing the annotation experiment, two fresh actual browser runs in
`test-results/retained-final-viewer-v1/` confirm the retained chapter-06 paired
reading and chapter-05 17px paragraph. Both outputs were visually inspected and
pass offline source/request/result checks in `test-results/retained-final-audit-v1/`.
They have zero browser/pipeline errors and no translation-budget exhaustion.
Chapter 06 retains 20 translations / 8 edits; chapter 05 intentionally returns
to 15 translations / 9 edits with its two unsupported annotation units blocked.

The retained source passes **868 tests in both `.venv` and `.venv-paddle6`**.
The source-only check also passes (868 discovered, 39 optional-local-evidence
skips). All 50 downloaded original hashes were rechecked unchanged. User-owned
dependency files remain exactly as supplied, including their existing changes:

- `pyproject.toml`: `a6985c3110ac6a6c23e6126751327f89c4dda770856b39ae1b114ffb8125efa7`
- `uv.lock`: `a9a9ac53bb3592b9199e7d10c8cc2f98704128e9097171e87b0e12a26d481185`

The exact staged source snapshot additionally passes 89 focused OCR, confidence,
pipeline and offline-review checks (3 optional saved-evidence skips), retained
in `test-results/reread-final-checkpoint-staged-tests-v1/`.

Checkpoints `03c3801` (edge contract) and `2e8c779` (actual occupancy) are local
commits. Nothing is pushed; project license/visibility are unchanged.

## Stop decision

Keep the tested edge/occupancy fixes and paired composition/script audit; reject
the annotation production change. Native cleanup, requested minimum lettering
size, source uncertainty, exact identity preservation and default policies remain
unchanged. Do not chase the gray-interior failures by increasing erase permission
or waiving artwork/ownership guards.

This pass stops at diminishing returns: 34 of the 36 extra corpus replacements
come from repairing execution, only two are new placement recoveries, the
evaluation subset gains no replacements, and the annotation trial brings no
additional edits or new-corpus associations. The remaining translation/context,
detection and cleanup failures are substantial. More same-series threshold
tuning would risk overfitting rather than establish better general quality.
Future work needs broader licensed material and independent transcription/
translation references, not another accuracy claim from this development sample.
Prior upstream research exposure also prevents a formal clean-room claim.
