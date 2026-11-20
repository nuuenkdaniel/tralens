# Optional source-bound crop reader checkpoint

The optional local Hayai reader now participates in the coordinator, translation
bindings and render-time pixel validation. TG27 remains the default translator.
This is a useful OCR checkpoint, **not zero-error translation**.

## Use and provenance

Add `--crop-reader-model /absolute/path/to/existing/snapshot` to the rendered,
coverage-first pipeline. It is mutually exclusive with the other built-in
refiners. `src/main.py` owns loading/closing the reader; OCR, translation and
rendering remain separate. Import and `--help` load no model runtime.

The reader requires the existing pinned Hayai v2.7 Supernova snapshot documented
in `qwen-repetition-diagnosis.md`. Every configuration, custom module, tokenizer,
vision configuration and weight file has a fixed SHA256 in
`src/tralens/ocr/hayai_adapter.py`. No automatic asset acquisition occurs.
The upstream implementation is loaded from that external snapshot, not copied
into this repository. The model card declares Apache-2.0.

The exercised local CPU environments contain Torch 2.14.1+cpu, Transformers
4.57.6, tokenizers 0.22.2 and safetensors 0.7.0. Paddle v6 runs in the existing
isolated `.venv-paddle6` environment (PaddleOCR 3.7.0/PaddleX 3.7.2). Hayai uses
float32, four threads, 256 patches, and at most 128 new tokens. Thread and
environment settings are restored when the reader closes.

## Selection and bindings

- Native, member-polygon-isolated and padded crops must agree after whitespace
  and NFKC comparison. The actual first reading is retained exactly.
- Unknown orientation, Latin-only spelling changes, strong raw readings and
  inconsistent views receive additional protection. Existing selected rereads
  take precedence. Agreement is a single-reader hypothesis, not independent proof.
- Every attempted reading and presentation is retained. Only a selected changed
  candidate enters an utterance; failed, unchanged and deferred trials preserve
  the previous utterance and translation revisions.
- Raw detections and earlier alternatives remain immutable. Derived translation
  inputs remain explicitly uncertain. Saved geometry, scores, membership,
  ordering, model identity, selection and original/view pixels are revalidated.
- Rendering runs no OCR. Source cleanup remains independent of target placement.

## Optional projective reader evidence

Add `--reader-presentations` alongside `--crop-reader-model` to retain three
source-bound projective readings of **every saved detector polygon**. The
coordinator reuses its already-open Hayai reader and closes it before translation.
The option defaults off. It adds no call quota: each tight, axial and contextual
view is planned, including empty/rejected OCR records and artwork controls.
Exact input-pixel deduplication keeps every source association. Non-quadrilateral,
oversized and otherwise unavailable views have explicit failure records.

`src/tralens/ocr/reader_presentations.py` stores the plan, unique PNG inputs, every
response and final evidence under `preview/reader-presentations/`. Failed,
overlength and empty responses remain available; one failed read does not cancel
later views. Controlled interruption preserves the complete plan and completed
responses. Existing candidates, selected readings and utterance revisions remain
untouched. These extra readings are diagnostic and have no erase permission.

The workflow snapshots this evidence into `preview.json` and replays source,
geometry, pixel, association and artifact bindings before translation. The public
`validate_saved` function supports the same checks without a reader. The saved
preview travels through rendering as evidence; rendering dispatches no OCR.
Transforms, interpolation and white off-page reader borders follow
`reader_geometry.py`, rather than changing native source or cleanup geometry.

The inspected projective trial in `source-removal-completion-results.md` explains
why these additional readings remain unselected: corrected tilt recovers some
text, but different readers also agree on omissions and wrong grouping. Nine
additional offline tests cover prior-selection preservation, exact-pixel
deduplication, empty/rejected/nonquad records, failures, interruption, tampering,
coordinator lifetime, workflow rejection and option validation. The combined
reader/geometry/grouping/workflow/pipeline check passes 88 tests.

The completed two-page live integration adds 348 reader calls over 116 detector
polygons, with no failed/unavailable views. All 68 selected-source/target cards and
the full source/clean/composite pages were inspected. Of 67 translations, 59 are
displayed and eight are marked unchanged; one further unit has no translation.
Wrong selected meanings, false unchanged cases and source remnants persist. See
the live-pipeline section of `source-removal-completion-results.md` for the runtime
correction, bound evidence and explicit failures. The option is complete as an
evidence collector; it does not claim improved selection or complete removal.

## Four-page evidence

Ignored local evidence is retained under `test-results/production-best-effort-v8/`:

- `crop-reading-trial-01/`: 339 calls, 17 initial consistency hypotheses.
- `crop-reading-trial-02/`: 339 calls, 13 new hypotheses after retaining four
  previously selected readings and their existing translation bindings.
- `crop-translation-01/`: 13 freshly bound TG27 translations; all original
  requests/results and prior previews retained. Two target strings stay identical.
- `crop-render-01/`: fresh localization and four native-resolution render trials.
  All 13 changed-source outputs were inspected against original/prior/clean/final
  cards. The audit checks 1,321 artifact hashes, exact recomposition and strings,
  font floors, and zero changed pixels outside declared support. There are 99
  displayed replacements and zero available-but-undisplayed targets. One old
  source remains without a translation. Counts do not measure language accuracy.

Clear useful changes include “The master of copy and paste,” “Super Cow,”
“Update,” removal of a generated tone explanation, and “Hurry up!” Some targets
remain misleading: the clapping sound becomes `*crackling sound*`; `いいの` becomes
“Is that okay?”; split open-source dialogue reads like a heading; merged UI
labels become invented sentences. Clipped source glyphs and untranslated remnants
also persist. Reconstruction changes remain visible and are not called exact
preservation of the previous cleanplate.

`context-translation-01/` retains a 13-target nearby-context Qwen comparison.
It does not justify switching translators: the clap becomes a transliteration,
the reassurance remains a question, and “Super Cow” becomes `[Name]`.

The earlier upscaled Qwen OCR matrix completed 65/65 with no incomplete replies,
but still confidently misread ACPI, copy-paste and other passages. Completion
and reported confidence are not accuracy evidence.

Eight offline tests exercise preservation, failed trials, prior selections,
tamper rejection, exact crop pixels, the coordinator/renderer path, reader
closure and model-free imports. Follow-on work is original-ink column grouping,
line-framing recovery, context-sensitive translation and untouched-page review.
