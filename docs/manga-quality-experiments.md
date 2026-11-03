# Manga quality: frozen failures and independent experiments

## Checkpoint and scope

This checkpoint adds an **offline evaluation baseline**, not an OCR/cleanup fix.
No production defaults, models, dependencies, fonts, licenses or repository
visibility changed. No OCR/translation inference, downloads or pushes were made.
The running development viewer was left alone.

Design inputs are Tralens's own code, saved failures and the
[methods handoff](handoffs/manga-quality-methods.md). No upstream implementation
was fetched for this work. This is **not** a formal clean-room claim: the prior
research session inspected upstream code. Code, weights, fonts and example-image
rights remain separate; this checkpoint grants no distribution clearance.

## What the available evidence actually shows

The upload, saved preview, both coverage render bundles and original strict-only
bundle are available. Every declared artifact hash checked successfully: 448
artifacts in each coverage bundle and 133 in the original strict-only bundle.
The coverage bundles contain 444 PNGs each; the strict bundle contains 129 PNGs.
These checks verify retained files, not a new rerender or artistic quality.

The selected final coverage bundle is bound to the unchanged upload SHA256
`bc5a90b6952e45f11f9d6de80195cef48d2c0dee211e6071521001dd0848abe3`
and preview SHA256
`ced1ce13cbc56c5d14c99966199b1977f138dde72c7349064694e7d7e0360a05`.
It retains 29 translations/replacements, 23 best-effort replacements and zero
outside-cleanup/protected change counters. Those are **not** 29 quality successes.

Specific observations from the source, cleanplate and composite:

| Case | Saved evidence / concern |
| --- | --- |
| Double-click instructions | IDs 25–27 read `でこのアコンをダブンルプリッ7`; ID 29, `で`, is a separate unit. The target becomes “And this aconite, Dabunlupri 7.” plus “with”. Both OCR and grouping need evaluation. |
| LiveCD balloon | Mixed Latin/Japanese IDs 13, 18, 20 are grouped, but selected `LiveCD,てゆーらいいですよ` is not a checked transcription. |
| Desktop dialogue | IDs 14, 19, 21, 22 contain erroneous wording; a fluent target about “hilarious desktop wallpaper” does not validate it. |
| Broad sound effect | ID 24 reads `九月九A朔月内加`, becomes a calendar-like translation, and removes visible artwork. |
| Small source gloss | Small lettering near the bottom of the help balloon remains without a dedicated detected unit. Its role is unverified; 29/29 detected-unit replacement does not mean all page text was detected. |
| Short interjection / joined lobes | Preserve the separate `ねえ` and test joined-balloon grouping independently; proximity or connected whitespace is not utterance proof. |
| Label / frame-clipped effects | Broad cleanup visibly damages the laptop label surroundings, hair and line art. Dark/outlined/rotated lettering needs its own evidence, not a Japanese-only filter. |

These are provisional visual observations, **not checked Japanese references**.
The instruction crop includes neighboring lettering on purpose. Its current
division is an experiment target, not an instruction to merge the two units.

### Relevant causes and constraints in our code

- `src/tralens/ocr/automatic_units.py`: reading crops are detection unions.
  Barrier/annotation/companion guards already exist; blanket nearest-neighbor
  grouping would regress them. Full-utterance completeness remains unverified.
- `src/tralens/ocr/automatic_recognition.py`: `prepare_automatic_crop` requires
  isolated crops, known orientation/ownership and Japanese script evidence.
  Mixed/Latin inputs are not routed to the Japanese recognizer. These are
  candidate-route constraints, not reasons to block coverage-first translation.
  **This saved run has no automatic reread results**; it does not establish that
  any alternative recognizer was attempted and failed.
- `src/tralens/rendering/automatic.py`: the strict flat route already measures
  whole dark/contrast components and checks antialias coverage and clear collars.
  It supports light neutral backgrounds. Cleanup planning and local layout are
  coupled: an unusable layout can reject an otherwise useful removal hypothesis.
- `src/tralens/rendering/best_effort.py`: footprint fallback dilates source
  polygons and repairs their interiors. Seam/color scoring selects a bounded
  classical candidate; it does not measure source removal or art preservation.
- `src/tralens/rendering/segmentation.py`: semantic association and periodic
  repair already exist as opt-in experiments. They are not new fixes, and their
  model provenance does not justify default deployment.

## Frozen cases and reproducibility

`docs/quality/manga-cases-v1.json` defines **14 native development cases**:
11 detected-text contexts, one missed-text observation and two negative contexts.
It pins source bytes, typed detection anchors and native-pixel diagnostic windows.
Windows are **evaluation metadata**, not manual production masks or corrected
wording. No native transcription, lettering-mask or hidden-background truth is
invented. Quality scores remain `null`.

`tools/manga_quality_cases.py`:

- Checks original file/RGB, raw/proposal/unit, request/result, rendering-record,
  job-revision and every declared artifact binding. Recomputes page pixel audits.
- Copies the complete saved rendering bundle, including original bytes, exact
  preview/readings/alternatives, all intermediate layers and font metadata.
  It never follows model/font paths, response sidecars or URLs from JSON.
- Publishes native source/clean/composite/text/erase/protected crops, source-bound
  case metadata, a script-free review page, manifest digest and freezer snapshot.
- Keeps available translations visible even when rendering was skipped. Crop
  change counters describe the **page result**, including neighboring units;
  they are exposure/locality diagnostics, not source-remnant or art-damage scores.
- Requires fresh output directories. Verification is self-contained; original
  absolute paths need not remain accessible. Optional external digest pins detect
  manifest replacement; local hashes do not authenticate publishers or rights.

Run using existing local evidence only:

```sh
.venv/bin/python tools/manga_quality_cases.py freeze \
  --preview test-results/translation-coverage-live-v1/preview/preview.json \
  --image test-results/dev-frontend/7d151a7fa33e473e9901d4e709cb3912/input.png \
  --rendering test-results/translation-coverage-live-final-rerender-v1 \
  --cases docs/quality/manga-cases-v1.json \
  --output test-results/manga-quality-frozen-new
.venv/bin/python tools/manga_quality_cases.py verify \
  --frozen test-results/manga-quality-frozen-new
```

Images, exact saved OCR/translation snapshots and generated reports stay ignored
and local. The tracked case spec contains no corrected transcription or target
reference. Raw-response sidecars remain in the original run, unmodified; this
freezer preserves their metadata but does not certify or copy their payloads.

### Split and reference gaps — blockers to promoting a quality change

All crops of this demonstrated page belong to **development**. Overlapping crops
are not independent samples. Chapters 1–3 and the acquired chapter 4/6/Nemo pages
have prior evaluation exposure too. Do not relabel them untouched held-out data.
There are currently **zero native held-out cases in this new freeze**.

Before tuning a production candidate:

1. Freeze newly generated controlled overlays with separate development and
   unopened confirmation background/lettering families. Retain exact background,
   alpha, transform and seed references for scoring only. Include punctuation,
   faint edges, white/outlined/rotated text, mixed scripts, small annotation-like
   lettering, art crossing a broad detection, empty art and neighboring text.
   Existing render-review overlays are development comparisons, not fresh tests.
2. Reserve fresh native pages/cases before inspection/tuning, recording exposure
   and rights. No acquisition is authorized or performed by this checkpoint.
   Include at least eight independently checked text blocks per split and at
   least four fresh pages across two sources for native confirmation. This is a
   minimum experiment size, not a generalization guarantee.
3. Keep independently checked exact transcription/grouping/annotation-role
   references in a scoring-only ledger with source/crop hashes and reviewer notes.
   Mark unresolved characters/roles unassessed. Evaluation references must never
   enter recognition/translation prompts or production cleanup geometry.

These are evaluation activities, **not required user corrections, masks or
per-region approval in the product**. Missing references block an accuracy claim
or default promotion, not ongoing translation availability.

## Independent experiments, in priority order

The numerical gates below are **proposed acceptance criteria**, fixed before
these experiments, not observed results or values inherited from another project.
Record every attempt, failure, resource skip, latency and peak memory. Compare
paired inputs; freeze implementation/configuration digests before confirmation.

### 1. Source-bound block and smaller-region rereads

**Hypothesis:** appropriate block context plus detector-line rereads can recover
hand-lettered text without treating ownership uncertainty as unreadability.

Design a separate bounded candidate route using existing approved local OCR
assets only. Reuse unchanged source pixels, keep full-block and line crops
separate, and retain orientation alternatives with invertible coordinate mapping.
Unknown ownership is recorded, not resolved by assertion. Conflicting neighboring
ink stays visible in the crop evidence; do not hide it with a manual mask. Keep
Latin/mixed-language Paddle candidates; do not send everything to a Japanese-only
decoder. Any unavailable/unapproved recognizer is explicitly not run.

Initially **store candidates only**, preserving raw selected strings and coverage
requests. Candidate ranking/selection is a separate experiment; do not choose by
fluency, OCR confidence, agreement alone, or hard-coded expected words. Each
candidate binds original bytes/RGB, all raw evidence, typed members, exact crop,
transform, neighbors, recognizer identity/settings and consumed attempt budget.
Use at most one block and one reread per main detector line, with a fixed page
budget; deduplicate identical crop/transform/engine attempts and report omissions.
Do not request translations for every alternate during recognition trials.

**Pass before promoting selection:**

- Exact-Unicode character edit count falls at least 20% on the checked paired
  subset in **both** development and fresh confirmation; publish per-case and
  micro-averaged errors. Report normalized scores separately, never as replacements
  for punctuation/width-sensitive exact scores. Missing/failed outputs count as
  empty predictions, not removed cases.
- The complete instruction block, including its terminal line, improves without
  duplicated/missing characters or a case-specific substitution. No degradation
  on checked Latin/mixed labels or the short interjection; unresolved references
  stay out of accuracy scores with explicit assessed/all denominators.
- No candidate from the art-only negatives is automatically promoted to source
  truth. Count nonblank hallucinated candidates even when not selected.
- Original evidence and all 29 available coverage translations remain retained;
  no automatic lexical correction is inferred from a better-sounding translation.

### 2. Grouping hypotheses, not destructive regrouping

**Hypothesis:** a bounded local relationship graph can connect compatible lines
and short terminal runs without crossing balloons/panels or annexing annotations.

Represent same-block membership, within-block order, orientation, balloon support
and annotation role as separate evidence. Test alignment, scale, intervening
source ink and barriers across the complete candidate span, not only the closest
pair. Existing barrier/chain-bridge/companion negatives remain mandatory. Joined
lobes and singleton interjections must allow competing hypotheses. Retain the
original unit partition and every source ID; never merge merely because a reread
produced a plausible sentence. Link alternative groups to original units so
translation reuse cannot accidentally apply one group's target to another mask.

**Pass:** at least 25% fewer checked split/missing-membership pair errors, **zero
new false-merge pairs across checked balloon/panel or annotation boundaries**, and
no order regression on the checked subset in either split. Report false merges,
false splits, order inversions and role mistakes separately; every detection must
remain assigned exactly once per hypothesis or explicitly unassigned. An
unassessed relationship must not be scored as correct.

### 3. Lettering support and background-dependent cleanup

**Hypothesis:** local surface evidence plus whole-stroke support can remove
lettering without repairing the entire OCR reading footprint.

Separate read crops, proposed lettering/erase support and layout geometry. Test
removal-only candidates before typography so layout failure cannot hide a good or
bad mask. Start with independently measured simple light **and dark** surfaces,
both contrast polarities, full antialias/outline support and punctuation. Use
source-bound line/glyph evidence to constrain association; do not classify every
same-color component as text. Ambiguous art-connected strokes remain ambiguous.
Do not adopt the previously rejected morphological mask or a learned restorer as
the presumed solution. Keep native evidence and any separately approved learned
evidence identifiable; neither constitutes semantic truth.

Compare mask candidates under the **same** repair backend first. Then compare
measured fill versus localized classical repair as a separate factor. Preserve
all masks, background evidence, candidate cleanplates and failure reasons. Seam
error, mask size and pixel equality outside a mask are not acceptance metrics for
the mask's artistic appropriateness.

**Pass on controlled references, every case included:**

- At least 99.5% support coverage of nonzero source alpha **per case**, with 100%
  coverage of checked high-contrast punctuation/outline cores. Report uncovered
  alpha and residual source pixels as well as aggregate recall; no mean may hide
  a missed-letter case.
- No worse target-region reconstruction error per case and at least 20% lower
  aggregate target-region MAE than the paired footprint baseline. Measure native
  cleanplate pixels, not translated composites that can conceal remnants.
- At least 30% fewer changed unoccluded pixels in aggregate, with **no per-case
  increase**; protected artwork/neighbor/border references remain pixel-identical.
  No-text negatives must remain unchanged. Report each background family separately.
- Zero outside-mask/protected changes, unchanged original bytes, exact source
  bindings and explicit resource/error accounting. Native visual review must show
  no new remnants or damaged borders at 1:1; hidden native art remains unknown.

A route failing either removal **or** preservation stays experimental. Do not
reduce attempted translation coverage, silently omit SFX, or call text-only or
overlay output a clean replacement to improve these scores. If a safe image edit
cannot be established, retain the translation and explicitly report the failed
or degraded replacement; do not erase evidence to obtain a higher count.

### 4. Balloon-aware typography after cleanup

Only after a complete cleanup candidate exists, derive a separate supported
balloon/layout component with readable margins and nearby-output protection.
Compare whole-balloon versus current source-local fitting with the **same saved
translations**, no new translation calls. Test balanced breaks and actual glyph
containment rather than expanding erasure to make room for English.

**Pass:** exact supplied characters/explicit breaks preserved; zero new glyph,
border or neighboring-output collisions; no newly sub-minimum lettering; and at
least 80% of paired checked layouts preferred at native reading size by a blinded
review with ties/failures reported. Both splits must pass, and no case previously
readable may become unreadable. Layout scores cannot offset bad OCR or remnants.

## Common release gates and next implementation slice

Keep OCR, translation and rendering separate under `src/main.py`. Add analysis
sidecars and rerender from saved translations; do not repeat translation for a
mask/layout trial. No new route becomes default from these development crops.

Every comparison reports detected/missed text, attempt/results, strict/uncertain
translations, actual replacements, failure/skip reasons, source remnants, artwork
damage, readability and runtime separately. Do not collapse them into a quality
score or hide failed cases. Preserve strict-only 129-PNG and coverage 444-PNG
regressions when the experimental route is disabled. Coverage-first stays default;
none of the existing integrity, exact-text or resource guards is waived.

The next implementation slice should be the **candidate-only reread plan plus
its provenance/budget tests**, alongside a fresh controlled-removal fixture
generator. Then run the recognition and mask trials independently. Automatic
source selection, broad cleanup fallback replacement and balloon typography
changes wait for the corresponding evidence gates. Do not add/download models,
fonts, dependencies or redistribute the local images without separate approval.

## This checkpoint's local verification

- Frozen review: `test-results/manga-quality-frozen-v1/review.html` (local only).
- Frozen manifest SHA256:
  `92c32bb6e72d1b7e74489dc071f54740a641e8a8629cf02f14807176b7ece2eb`.
  All 536 declared snapshot artifacts and their source/result bindings verify.
- **807 tests pass** in each of `.venv` and `.venv-paddle6`.
- Isolated source-only verification runs the same 807 tests successfully with
  **39 explicit optional-local-evidence skips**. Portable tests cover exact
  native crops, retained/skipped coverage translations, hash and semantic drift,
  source immutability, no-network verification, safe paths and script-free HTML.

No new OCR/translation model was run. These checks establish regression and
snapshot integrity, **not improved transcription, removal or visual quality**.
The pre-existing edits to `pyproject.toml` and `uv.lock` remain untouched.
