# Handoff: independently improving manga OCR, cleanup and layout

Prepared 2026-11-03. Read this as a research brief and set of product requirements,
not as an implementation to reproduce.

## Purpose and provenance

The user wants Tralens to become open source eventually, but not yet. They want
to learn from other projects without copying their implementation. Do not change
the project's license, publish/push further work, copy upstream code, port its
functions, or mechanically translate its implementation without user approval.

This brief contains original prose about general methods, source references,
Tralens's own requirements and proposed experiments. It intentionally excludes
upstream code, pseudocode, function signatures, exact tuning constants and
function-by-function instructions. Implementation choices should be designed
against Tralens's tests and requirements, not reconstructed from upstream code.

**This is not a certified clean-room specification.** Earlier research in the
preceding session inspected upstream implementation files. A fresh session and
different wording do not erase that history or guarantee copyright independence.
Do not claim otherwise. If formal clean-room assurance is required, stop and
arrange a genuinely separated research/specification/implementation process with
appropriate review. This document is not legal advice or artifact clearance.

The next session can use this brief and Tralens's own source without fetching
upstream implementation files. Public papers and product documentation are
references, not instructions to copy code or acquire models.

## The problem we are solving

Tralens originally translated/replaced only eight of 29 saved text units because
of conservative spatial eligibility. Coverage-first requests now attempt all 29
on that saved page, but the result remains poor. The local run exposed:

- Incorrect OCR entering translation, including garbled double-click instructions.
- Text belonging together being treated as separate units, or grouping uncertain
  regions without sufficiently strong evidence about their relationship.
- Broad cleanup regions destroying artwork, balloon borders and nearby detail.
- Placement driven by source footprints rather than comfortable balloon geometry.
- High replacement counts concealing bad translations, smeared art and awkward type.

The user still prefers broad translation attempts over undertranslation. Do not
solve quality by silently dropping difficult text, filtering all sound effects,
or restoring the old ownership gate for translation. Translation availability
and permission for a particular image edit are different questions.

## Other projects: methods and scope

These are observations, not claims that any project reliably solves arbitrary
pages or outperforms Tralens on our evidence. No comparative inference benchmark
was run for this handoff. Several projects include substantial manual editing.

### 1. Manga/Image Translator

Reference: [project documentation at the reviewed revision][mit-readme].

This is an end-to-end detection, recognition, translation, removal and lettering
pipeline. Its relevant methods include forming coherent text blocks from detected
lines, retaining writing direction, refining removal regions using text evidence,
and using balloon shape when fitting translated dialogue.

The useful distinction is between the area that supplies OCR context and the
pixels that should disappear. A readable crop may contain surrounding artwork;
that does not make the entire crop appropriate to erase. The project offers
multiple detection/removal/rendering choices, so its output cannot be attributed
to a single universally safe mask technique.

**Lesson for Tralens:** evaluate grouping and pixel-level removal separately.
Use the full available balloon for comfortable lettering when its geometry is
supported, rather than assuming the original narrow Japanese column is the best
English text frame. Do not inherit upstream scoring rules or tuning values.

### 2. BallonsTranslator

Reference: [English project documentation at the reviewed revision][ballons-readme].

This is both an automated pipeline and an editing application. It estimates
source formatting, uses extracted balloon regions for layout, and supports
different OCR and reconstruction systems. Its cleanup approach distinguishes
simple backgrounds that can be filled from backgrounds that need reconstruction.
It also provides manual mask and image repair, so polished examples need not
represent unattended output.

Its documented hybrid detection approach illustrates another distinction: text
locations can come from one detector while lettering masks come from another.
Its fallback can become coarser when precise evidence is unavailable. That is a
quality tradeoff to expose, not proof that fallback rectangles preserve artwork.

**Lesson for Tralens:** decide the cleanup treatment from observed background and
mask evidence, not merely from the presence of a translation. Keep fallback
quality visible. Do not introduce required manual repair into our default route.

### 3. Koharu

References: [project overview][koharu-readme], [vision models][koharu-vision],
[cleanup guidance][koharu-cleanup], [typesetting guidance][koharu-typesetting].

Koharu treats text regions, balloon/page structure and removal masks as distinct
information. It has interchangeable recognition and inpainting stages, layered
editing, and automatic text fitting within a presentation frame. Its cleanup
guidance emphasizes precise removal regions and checking for remnants, damaged
borders and smeared textures. It recommends separate text layers rather than
asking a reconstruction model to paint the translation itself.

Earlier implementation research also observed simple-background filling before
localized masked reconstruction. The transferable idea is background-dependent
treatment with bounded context and localized composition, not its code or exact
decision rules. Public documentation and model defaults evolve; do not combine
older and newer descriptions into an invented single configuration.

**Lesson for Tralens:** retain distinct reading, cleanup and layout geometry.
Preserve orientation as data, inspect at actual reading size, and avoid counting
a tiny or poorly positioned translation as a satisfactory layout.

### 4. Manga OCR

Reference: [recognizer documentation at the reviewed revision][mocr-readme].

This Japanese recognizer can read multiline text blocks, including vertical text
and furigana. Its documentation also notes that smaller subregions can help when
longer passages fail. Crucially, it warns that its generative decoder can produce
plausible text even from unsuitable or empty input.

**Lesson for Tralens:** compare block-level context with smaller line-level
rereads on suspicious regions. A second result is a candidate, not a correction
certified by fluency. Keep the original string, exact crop/source association,
alternative strings and uncertainty. English-looking OCR on a Japanese page is
not automatically wrong: mixed-language labels and logos need fair treatment.

### 5. Mokuro

Reference: [project documentation at the reviewed revision][mokuro-readme].

Mokuro detects and recognizes Japanese manga, then saves text and spatial
metadata for a separate reading interface. It uses manga-oriented detection and
recognition, but is a reading aid rather than an automatic cleanplate translator.
Its valuable architectural choice is that extracted text can remain inspectable
without destructively rewriting the source image, and analysis can outlive a UI.

**Lesson for Tralens:** keep analysis and display/rendering separate. Preserve
available translations even when a particular cleanup/layout attempt fails.
An inspection overlay or text-only result must be labelled as such; it is not a
replacement and must not inflate replacement counts. This is an architectural
lesson, not a decision to make overlays the default final output.

### 6. MAGI / The Manga Whisperer

References: [research project overview][magi-readme] and
[The Manga Whisperer paper, version 3][magi-paper], especially the discussion of
transcript generation and evaluation.

MAGI studies page understanding and transcription, not artwork reconstruction.
It treats detected page objects and their relationships as separate information.
It considers panel order and text order within panels rather than assuming that
one global coordinate sort captures the intended reading sequence. It also
distinguishes uncertain associations from confident ones.

**Lesson for Tralens:** page order, within-block order, balloon ownership and
speaker identity are separate hypotheses. Panel structure may help avoid merging
nearby but unrelated dialogue. Unknown speakers must not prevent translating
readable text, and should not be invented to make a sentence feel complete.
Full character identification is not a prerequisite for the next quality fix.

## Cross-project method summary

| Concern | General method worth testing | Mistake to avoid |
| --- | --- | --- |
| Text grouping | Combine compatible lines using spatial, directional and typographic evidence; retain uncertainty about group membership. | Concatenating nearby text across balloons or panels. |
| OCR quality | Compare appropriate whole-block and smaller-region readings while preserving the source evidence. | Selecting the most fluent-looking string as truth. |
| Removal masks | Estimate lettering support separately from reading crops and layout space. | Erasing all pixels inside a detection rectangle. |
| Cleanup | Use measured fill for genuinely simple backgrounds and localized masked reconstruction for harder ones. | Assuming a stronger inpainter can fix an oversized or incorrect mask. |
| Layout | Fit exact translated text to supported balloon geometry with margins, direction and sensible line breaks. | Maximizing replacement count using unreadable size or arbitrary wrapping. |
| Text categories | Consider dialogue, captions, embedded labels and sound effects as potentially different treatments. | Discarding non-dialogue text merely to make the output look cleaner. |
| Presentation | Store source, analysis, cleanup and lettering separately and support inspection/re-rendering. | Re-running OCR/translation just to inspect a different layout, or calling an overlay a replacement. |

The table describes design objectives, not an upstream implementation recipe.
Derive our own algorithms, tradeoffs and parameter choices through controlled
experiments. Do not copy upstream data structures or interface signatures just
because the methods sound appropriate.

## Proposed experiments for Tralens — not upstream results

These proposals are derived from our failures and requirements. None is already
validated simply because another project uses a related method.

### A. Freeze quality cases before changing defaults

Include the garbled instruction balloon, LiveCD dialogue, the separate short
interjection, joined balloons, text on dark/toned backgrounds, mixed-language
labels, outlined/rotated sound effects, furigana, and broad regions crossing art.
Keep difficult no-text/art-only and neighboring-text negatives too.

Separate development and held-out cases before tuning. Use licensed/local images
and synthetic controlled examples. Evaluation annotations/references must not
become user-supplied production masks or source wording given to the model.

### B. Compare recognition strategies without silently changing source truth

Compare current raw readings with source-bound block/line rereads. Measure actual
transcription errors on an explicitly checked reference subset, grouping errors,
missed/duplicated characters, punctuation and annotation-role mistakes. Record
unassessed regions; OCR confidence, agreement and fluency are not accuracy.

Do not introduce reference-specific word substitutions or let the translator
repair missing source wording. Candidate selection needs its own documented,
testable policy and binding, not silent replacement of original OCR evidence.

### C. Compare removal evidence independently of inpainting

Keep a lettering mask distinct from the read crop and the balloon/layout region.
Test native evidence and legally appropriate learned evidence without assuming
either is semantic truth. Compare source remnants, protected borders, punctuation
and nearby artwork before adding a larger reconstruction system.

A smaller mask is not automatically better: it can miss letters and leave source
text visible. A lower seam error or fewer changed pixels is not enough to accept
it. Source-removal coverage and artwork preservation must improve together.

### D. Evaluate layout after a complete cleanup candidate exists

Compare local footprint-based fitting with supported balloon-aware fitting.
Evaluate normal-size readability, margins, line breaks, collisions and writing
direction while preserving exact supplied target characters. Keep translations
available when replacement fails; do not disguise skipped edits as successful
page reconstruction or drop words to make a fit.

### E. Report quality and coverage separately

Track detected text, translation attempts/results, actual replacements, remaining
source text, damaged visible artwork, readability, execution failures, latency
and memory. Distinguish strict and uncertain results and keep denominators honest.
Real hidden backgrounds are generally unknown: use controlled overlays for
ground-truth reconstruction tests and label native-page visual review honestly.
Pixel equality outside a declared mask is an integrity check, not proof that the
mask itself was artistically appropriate.

## Project constraints and restart context

- Work is on `ocr-cleanup-quality`, based on committed coverage-first work.
  The preceding research checkpoint is `1e62724`; the last full quality-related
  implementation checkpoint is `fc9b21b` on `translation-coverage`.
- The frontend was pushed on `development-frontend`. Do not push new work or
  change repository visibility/licensing without explicit user approval.
- Preserve the user's pre-existing edits to `pyproject.toml` and `uv.lock`.
- OCR, translation and rendering remain separate; `src/main.py` coordinates them.
- Preserve originals, exact strings, source bindings, alternatives, intermediate
  layers, honest uncertainty and explicit resource/failure reports.
- No required manual masks, source correction or per-region confirmation.
  Use automatic bounded attempts; make incomplete/degraded outcomes inspectable.
- Do not acquire models/fonts, add dependencies or enable remote providers
  automatically. Reuse approved local assets only after checking their terms.
- Reuse the saved upload for retries in a fresh run. Do not overwrite old evidence.
- The prior implementation passed 797 tests in both existing environments;
  source-only verification had 38 optional-evidence skips. That is a historical
  checkpoint, not a test run performed for this documentation-only handoff.
- Preserve successful existing rendering while testing new routes. Previously
  rejected morphological masks missed visible letters, and prior LaMa experiments
  did not establish a quality improvement. Do not repeat them as adopted fixes.

Relevant Tralens entry points: `src/tralens/ocr/automatic_units.py`,
`src/tralens/ocr/automatic_recognition.py`, `src/tralens/translation/contracts.py`,
`src/tralens/translation/coverage.py`, `src/tralens/rendering/automatic.py`,
`src/tralens/rendering/segmentation.py`, `src/tralens/rendering/best_effort.py`,
and `src/main.py`. These are our own implementation context, not upstream recipes.

Local evidence, if still available:

- `test-results/translation-coverage-live-v1/`: fresh OCR/translation/rendering;
  29 attempted translations and replacements, with visibly poor regions.
- `test-results/translation-coverage-live-final-rerender-v1/`: same saved
  translations rerendered without additional translation calls.
- `test-results/dev-frontend/7d151a7fa33e473e9901d4e709cb3912/run/`: prior
  eight-translation baseline for the same saved upload.
- `docs/translation-coverage.md` and `docs/rendering-production-review.md`:
  Tralens's own outcomes, limitations and rejected quality experiments.

These ignored local paths are not portable assets or proof of rights to publish
their images. Confirm availability and bindings before using them.

## Licensing boundaries for the next session

Code, weights, fonts, training data and example images have separate terms. This
brief recommends no new dependency or checkpoint as commercially cleared.

- Manga/Image Translator, BallonsTranslator and Mokuro list GPLv3. Do not copy or
  port their implementations and assume attribution alone satisfies the license.
- Koharu offers its application code under MIT or Apache-2.0. That does not clear
  every dependency or model. Its referenced layout model declares custom terms
  associated with Manga109 academic-use conditions.
- Manga OCR code and the referenced base model declare Apache-2.0; retain notices
  and examine exact artifact provenance if reused. Metadata is not a warranty.
- MAGI's README limits the provided models/datasets to academic research.
  Treat the paper as a conceptual reference, not permission to deploy its weights.
- Existing Tralens CTD/LaMa artifacts have unresolved deployment/provenance
  questions. Keep evaluation-only/opt-in status unless separately resolved.
- Private development now does not authorize relicensing other people's work
  later. Maintain a provenance inventory and obtain review before distribution.

## Sources and version notes

Research references below are attributed so findings can be checked. They do not
authorize source copying, model downloads or execution of README commands. This
handoff does not quote or reproduce their sample code. Live documentation can
change; the repository references use pinned revisions where available.

| Project | Reviewed repository revision | Source used in this brief |
| --- | --- | --- |
| Manga/Image Translator | `d6de6ab62a7be6afc0d07a3184149cdefbcd32f2` | README plus high-level findings from earlier implementation research. |
| BallonsTranslator | `7fe9979f1233eb8edb5f431c060d2493efbc995b` | English README plus high-level findings from earlier implementation research. |
| Koharu | `cd31ad2c7016b64d0ff0400a09e86cd13fb5df13` | README, public workflow/vision/cleanup/typesetting documentation and earlier high-level findings. |
| Manga OCR | `c333b5d36e88d539d6b040b4c4cf90ad5ecd4f69` | README capabilities and limitations. |
| Mokuro | `9f79b1281066f953ec6e5d1e7086c401fa8da159` | README workflow and metadata/reader separation. |
| MAGI | `2a45bf09b43adc80778270a366372aaa148e2291` | Research overview, stated model/data terms and the v1 paper. This is not a claim about every later model version. |

[mit-readme]: https://github.com/zyddnys/manga-image-translator/blob/d6de6ab62a7be6afc0d07a3184149cdefbcd32f2/README.md
[ballons-readme]: https://github.com/dmMaze/BallonsTranslator/blob/7fe9979f1233eb8edb5f431c060d2493efbc995b/README_EN.md
[koharu-readme]: https://github.com/koharu-rs/koharu/blob/cd31ad2c7016b64d0ff0400a09e86cd13fb5df13/README.md
[koharu-vision]: https://koharu.rs/models/vision
[koharu-cleanup]: https://koharu.rs/guides/cleanup
[koharu-typesetting]: https://koharu.rs/guides/typesetting
[mocr-readme]: https://github.com/kha-white/manga-ocr/blob/c333b5d36e88d539d6b040b4c4cf90ad5ecd4f69/README.md
[mokuro-readme]: https://github.com/kha-white/mokuro/blob/9f79b1281066f953ec6e5d1e7086c401fa8da159/README.md
[magi-readme]: https://github.com/ragavsachdeva/magi/blob/2a45bf09b43adc80778270a366372aaa148e2291/readme.md
[magi-paper]: https://arxiv.org/html/2401.10224v3

## Suggested opening request for the new session

Read `docs/handoffs/manga-quality-methods.md` as a high-level research brief.
First inspect Tralens's own source and saved failures, then propose independent
OCR/grouping and precise-cleanup experiments with explicit acceptance criteria.
Do not fetch upstream implementation files, copy/port code, acquire models,
change licenses or push changes without asking. Keep broad translation coverage,
preserve evidence, and do not describe this as a certified clean-room process.
