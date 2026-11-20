# Rendering readiness review

## Outcome

The automatic path is better hardened and substantially faster on repeated,
overfull text. It still **is not production-ready for arbitrary artwork or
unrestricted full pages**. Best-effort placement is not proof of clean source
removal, readable lettering, or reconstructed hidden art.

The review retained three production changes:

- Validate exact target/glyph support before allocating cleanup geometry; build
  component/ownership snapshots only when an eligible target will be attempted.
- Treat a failed candidate pixel audit as an integrity failure. Do not retry it
  as artistic uncertainty. Distinguish resource failures from input/layout errors.
- Cache up to 1,024 identical measurement strings **per font size**. Preserve
  search ordering, logical work limits, exact line breaks and raster output.

An additional input fix accepts fully opaque `LA`/`RGBA` originals in automatic
rendering/preparation. Every alpha sample must be 255; RGB/grayscale transparency
keys and any nonopaque sample are rejected. No matte is guessed and no image is
silently resized. Original file bytes and canonical RGB fingerprints are retained.
Manual candidate composition remains RGB/L-only and also rejects transparency
keys rather than silently dropping them.

Checkpoint commits: `dea5875` (retry integrity, lazy validation, measurement cache)
and `273f284` (opaque-alpha input support).

## Additional examples and rights

Acquired six untouched originals, with source URLs, hashes, native dimensions,
attribution and saved license evidence under ignored `images/render-review-v1/`:

| Source | Native size | Terms / attribution |
| --- | --- | --- |
| Ubunchu! chapter 04, pages 0002 and 0005 | 1246×1635 | CC BY-NC 3.0; Hiroshi Seo / ASCII MEDIA WORKS Inc. |
| Ubunchu! chapter 06, pages 0002 and 0005 | 1246×1635 | Same; the supplied LA images have entirely opaque alpha |
| Little Nemo, 1905-10-15 | 1740×2376 | Public-domain evidence retained from Wikimedia Commons; Winsor McCay |
| Little Nemo, 1905-10-22 | 1740×2376 | Same |

Ubunchu! source tree: `ubunchu-translators/ubunchu` at
`f3c6473206a8a8b6f1a946e8c60b4cdaff5ea71b`, chapter-specific `license_en.txt`
files. The NC examples are local noncommercial evaluation material, **not
commercial product assets or redistribution clearance**. No acquired images or
derived output were added to git. No provider calls, cloud uploads, training,
dependency installation or new model downloads were involved.

Nemo evidence revisions:

- [1905-10-15](https://commons.wikimedia.org/w/index.php?title=File:Little_Nemo_1905-10-15.jpg&oldid=709615093)
- [1905-10-22](https://commons.wikimedia.org/w/index.php?title=File:Little_Nemo_in_Slumberland_(1905-10-22).jpeg&oldid=1215792021)

The second page's prose description inconsistently mentions 1908, while its date,
filename and visible comic identify the 1905 strip. Both dates are old enough for
the retained public-domain rationale; this is not evidence about later works by
Winsor McCay Jr. Acquired hashes are local byte pins, not publisher signatures.

`tools/download_render_examples.py` explicitly downloads only this bounded set,
requires a fresh directory, retains license evidence, and checks pinned artwork
and Ubunchu! license digests. It is never invoked by the pipeline. License HTML
is recorded with its acquired digest, not assumed byte-stable across requests.

## Frozen tests: what they measure

`test-results/render-review-frozen-v1/` contains **48 fixed cases**, frozen before
algorithm experiments:

- **24 native-text crops**: four per acquired page, with evaluation-only source
  boxes and English placeholder targets. These are not OCR outputs or claimed
  translations. Real hidden backgrounds and complete source-text coverage are
  unknown; imperfect boxes can leave source remnants.
- **24 controlled overlays** on native artwork crops: six backgrounds × short,
  neighboring text, long-word and deliberately overfull targets. These alone
  have clean-background and alpha references. References are scoring-only and
  are never passed to cleanup/layout.

`tools/render_review_trials.py` exercises cleanup, layout, composition, exact
character offsets and pixel audits. It is explicitly a **lower-level quality
trial**, not a substitute for the production binding/selection pipeline or a
full-page coverage benchmark. No caller-supplied erase masks enter the algorithms.
All crops preserve native pixels; only ordinary review thumbnails may be scaled.

The renderer uses the existing explicit, digest-bound Liberation Sans Bold face
and its installed OFL license evidence. No font lookup/download was introduced.

### Results

| Cases | Conservative placement | Best-effort placement |
| --- | ---: | ---: |
| Native text | 9/24 | 24/24 |
| Controlled overlays | 0/24 | 18/24 |

The six intentionally overfull targets remain unplaced at bounded resource
limits. This is an execution/readability limit, not permission to truncate text.
For the 18 placed overlays, the coarse footprint covers all target alpha pixels,
but mean clean-background target MAE is **29.07/255** and a mean **1,762 unoccluded
pixels** differ from the reference. Those are visible-damage warnings, not a
clean restoration result. Repeated short/long-word inputs share backgrounds;
these counts must not be presented as independent cleanplate samples.

Visual inspection found smeared speech-balloon fills in historical color scans,
blurred art beneath coarse OCR-footprint erasure, and remnants with underbounded
source boxes. Emergency 8px lettering over art can be difficult to read. Native
placement counts **do not** certify removal, readability or artistic quality.

### Retained performance improvement

Compare `render-review-baseline-v2/` with `render-review-cached-v3/`:

- Median of six overfull best-effort cases: **11.13s → 1.36s** (about 8.2× faster).
- Aggregate trial elapsed times: **71.02s → 12.09s** on this local CPU run.
- All decisions, layout metadata, exact characters and logical search budgets
  are identical; **all 204 generated PNGs are byte-identical**.
- No wall-clock timing assertion was added to unit tests. The cache regression
  instead compares pixels, full layout metadata and actual measurement calls.

These are local development measurements, not a hosted-service latency SLA.
`render-review-baseline-v1/` is retained but **not used** for performance/production
comparisons: its first runner relaxed typography even after a conservative fit.
Version 2 corrected that mismatch before the cache comparison. The frozen input
cases did not change.

### Rejected mask experiment

`tools/compare_local_strokes.py` tests a fixed 21×21 morphological top/black-hat
mask, threshold 24, before local inpainting. The trial is **evaluation-only** and
is not imported or routed by production rendering.

On 12 short/neighbor overlay contexts, coarse-footprint mean target MAE was
**29.07**, versus **49.57** for the narrower mask. Unoccluded changed pixels fell
from 1,762 to 1,178 on average, but worst visible-target support coverage fell to
**79.61%**. Some real color cleanplates looked less smeared, but the controlled
evidence demonstrated missed source strokes. **Do not adopt this route.** These
are development trials, not fresh held-out confirmation after inspection.

## Integrity and scale verification

- Final full suites pass **756 tests** in both project environments. The isolated
  source-only suite passes the same 756 tests with 37 optional-evidence skips.
- Added regressions for a one-off candidate compositor corruption, fallback
  corruption, unsupported glyphs before cleanup, untranslated-page allocation,
  truthful resource reasons, and exact cached/uncached output equivalence.
- A genuinely bound synthetic **32-unit page** exercises the automatic router,
  not just primitives. It publishes the allowed **16 units**, reports the other
  16 as resource-limited, and stays within artifact-count and pixel-audit limits.
- The saved real-page end-to-end render still places **8/8 existing strict
  translations**, including two marked best-effort. All **129 page/unit PNGs**
  match `coverage-first-parent-v2/` byte-for-byte. Its other 21 saved units remain
  blocked/untranslated. No extra OCR or translation calls were made.
- All four newly acquired Ubunchu! originals now pass automatic preflight. Both
  native Nemo full pages exceed the 4M-pixel page budget and are explicitly
  rejected; only their native crops were used for quality trials.

## Remaining deployment blockers

1. **Artwork reconstruction:** coarse fallback remains visibly destructive in
   some cases; no reliable general cleanplate model or real ground truth has been
   validated. Keep quality warnings and original/intermediate layers visible.
2. **Full-page scale:** at most 4M page pixels / 4096px side, 16 attempted slots,
   32M aggregate unit pixels and 512M raster-work allowance. At 1246×1635 the
   aggregate unit-pixel cap permits only 15 attempts. Dense and larger pages need
   a crop-local/streaming redesign, not arbitrary limit increases or resizing.
3. **Readability and geometry:** 8px emergency type, source-local art overlays,
   underbounded OCR, rotated lettering, unusual scripts and multicolor effects
   are not solved by pixel locality. The baseline supports horizontal English
   targets with explicit glyph coverage, not arbitrary multilingual typesetting.
4. **Translation availability:** fragment previews are not full translations or
   erase authorization. Missing strict translations remain explicit failures.
5. **Learned-artifact rights:** CTD artifact-specific commercial/distribution
   clearance and the original LaMa checkpoint chain remain unresolved. Neither
   is silently acquired or required by this review's default trials.
6. **Service isolation:** current bounds are algorithmic limits, not a CPU/RAM
   sandbox or wall-clock deadline. If exposing this local tool as a service, use
   constrained workers, trusted explicit font manifests, private job directories
   and output quotas. Do not expose arbitrary filesystem-path manifests as an
   untrusted public API.

## Reproduction and evidence

```sh
.venv/bin/python tools/download_render_examples.py --output images/render-review-new
.venv/bin/python tools/render_review_trials.py freeze \
  --examples images/render-review-new \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --output test-results/render-review-frozen-new
.venv/bin/python tools/render_review_trials.py run \
  --frozen test-results/render-review-frozen-new \
  --output test-results/render-review-new
```

Every output path must be fresh. Local fonts/evidence must exist; the command does
not install dependencies or regenerate translations. Source-only tests do not
require these ignored examples.

Recorded SHA256 evidence:

| Evidence | Digest |
| --- | --- |
| Acquired-source ledger | `7d457d45bc6580cd2f18c7231d6dc42135d8743d42565a3e44a3b8443192da23` |
| Frozen case manifest | `3d162db963a03b51ca3548b8e159683e1f0347a5e2c201fd639a89ef3ed44bd0` |
| Uncached report v2 | `39ddde9416bd04156b3b0064fbdaa9a79056d2b829bf4bccc212337f578575a1` |
| Cached report v3 | `bd0483454139896982511d603c6191e0df6e70efd4fe6f9546a450f7b9f3f6e2` |
| Rejected stroke trial report | `68181ef7d4b75ac425dd7cb88a8c7203cb6665b6ca254cb2a078ec38c71eff15` |
| Final report v4 | `2c9b767ecc104bd09dde9e082ea98cbd3f25c83c0c98a2516bebc7862077d16d` |

Final reruns are in `test-results/render-review-final-v4/`; the current saved-page
review is `test-results/render-review-parent-v4/review.html`. Reports retain
implementation hashes and the corresponding frozen/runner snapshots.
