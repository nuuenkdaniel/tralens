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
