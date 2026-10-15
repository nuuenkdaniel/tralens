# OCR review

`OCR(image_path)` keeps the default manga grouping path. Use
`OCR(image_path, media_type=MediaType.OTHER)` for nonmanga input. Import
`MediaType`, `OCRStatus`, and `OCRReviewPolicy` from `tralens.ocr` alongside `OCR`.

## Source layout and imports

```text
src/tralens/
  ocr/          types.py, detection.py, grouping.py, review.py,
                recognition_units.py, transcription_selection.py, automatic_units.py,
                automatic_evidence.py, automatic_recognition.py, manga_adapter.py
  translation/  translator.py
  rendering/    renderer.py
tools/          diagnostics.py
tests/          ocr/, translation/, rendering/, tools/
```

Each package has an `__init__.py`; unittest recursively discovers the test
packages. All 299 existing cases are retained, plus four local-evidence guard
regressions (303 total). Saved-original integrations are optional as described below.
For source-checkout Python use, set `PYTHONPATH=src` (no installation required):

```python
from tralens.ocr import OCR, MediaType, OCRStatus, OCRReviewPolicy, Text_Box, Text_Group
from tralens.ocr.recognition_units import snapshot_detections, plan_recognition_units
from tralens.translation import Translate
from tralens.rendering import cover_text, draw_text
```

`OCR` combines detection with grouping/review mixins; existing default calls,
algorithms and provenance are unchanged. Paddle's import-time host check is
deferred until explicit OCR construction, through `detection.create_paddle_ocr`.
The OCR facade loads detection only when `OCR` is requested, so unit diagnostics
do not import an engine. No legacy source-path wrappers remain. The empty
`src/main.py` remains untouched; there is no speculative pipeline module.

Paddle text, confidence, polygon and box ID remain in `paddle_*` fields.
`selected_text` (also available as legacy `text`), `selected_engine`, `status`,
`reason`, and `ocr_metadata` describe selection separately. Raw detections remain
in `ocr.raw_detections` and groups/debug views; only accepted nonblank boxes reach
automatic translation. Debug colors: green accepted, orange review, red rejected.

Low-confidence nonblank detections default to review. Blank detections default to
rejected, but trustworthy enclosed-region evidence can make them review eligible.
Configure `OCRReviewPolicy(accept_confidence=.85, reject_low_confidence=False,
blank_region_review=True, crop_padding=2)`; padding is limited to 0–4 pixels.
These defaults are heuristics, not calibrated accuracy claims.

Manga OCR is optional and disabled by default. Explicit runtime enablement is
`OCR(image_path, enable_manga_ocr=True, manga_model_path="/existing/local/model")`.
The separate adapter imports `manga_ocr` only on enablement—an intentional exception
to top-level imports so baseline Paddle works without this dependency. Enablement
fails explicitly if the package or usable local weights are unavailable; no
installation or download is attempted. Installed versions verified from the
existing venv's package metadata: `manga-ocr==0.1.16`, `torch==2.14.1+cpu`, and
`transformers==4.57.6`. The supplied local model snapshot can be selected with:

```python
ocr = OCR(
    image_path,
    enable_manga_ocr=True,
    manga_model_path="/home/Danuu/.cache/huggingface/hub/models--kha-white--manga-ocr-base/snapshots/aa6573bd10b0d446cbf622e29c3e084914df9741",
)
```

Model usability and live recognition have not been verified by these offline tests.

For offline tests, inject `manga_recognizer=callback`. The callable receives an
unaltered RGB PIL crop plus keyword arguments `source_box_id` and `crop_bounds`
(half-open original-image coordinates), and must return text only. At most one
attempt is made per eligible box; bounds are clipped, padding avoids neighbors,
and contaminated/invalid crops are not recognized. Isolation uses intersections
of valid convex OCR polygons, then checks the actual axis-aligned crop too:
disjoint polygons alone do not make a rectangular crop safe. Padding may shrink,
but the visible source footprint is never cut. No rectification is applied.
Text-corroborated contained fragments may occupy at most 10% of source polygon
area in total; full-text near-duplicates require at least 95% overlap of both
polygons. These conservative geometry/text hypotheses never use confidence as
evidence of noise. Decisions and neighbor IDs are recorded in `crop_isolation`.
Different/unknown annotations, substantial conflicts, and malformed polygons
remain blocked for review (empty crops are rejected); contamination not represented by OCR polygons
cannot be ruled out. No rotation, contrast change,
masking or upscaling is applied. Normalized agreement can accept; disagreement
retains both candidates and normalized differing spans for review. Nonblank
recovery from blank is also review, not established truth. Scores from different
engines are never compared. Test callbacks are not live Manga OCR recognition.

Offline checks: `.venv/bin/python -m unittest discover -s tests -v`.
Also run `.venv/bin/python -m compileall -q src/tralens tools tests` and
`git diff --check`. Test package markers include source-path setup; the diagnostic
test package extends its search path to the checkout's real `tools/` directory
because `discover -s tests` gives it the same top-level name. This is test-only,
not a production compatibility wrapper.

## Portable QA checkout

Use the existing project venv; these checks do not initialize OCR models or fetch
dependencies:

```sh
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m compileall -q src tools tests
git diff --check
```

Without local images/manifests, discovery runs the synthetic/core tests and reports
explicit skips for **21 individual saved-evidence methods**. With the authorized
originals and saved artifacts present, those methods run their full existing
assertions automatically. There are no class-wide OCR skips. Detector-cache tests
use temporary offline fixtures, not the user's cached model or a fixed checkout.

`tests/local_evidence.py` checks availability at test execution, not collection.
Skip reasons name exact missing paths. Manifests referencing another checkout are
skipped rather than following stale absolute paths into private evidence there.
Malformed JSON, failed hash/provenance checks and assertion failures are not
converted into availability skips. Optional evaluation requires the original
assets and hash-bound manifests generated for that same checkout.

Prove the source package works without evidence, using the same installed venv:

```sh
.venv/bin/python -m unittest tests.source_only_check -v
```

This separate packaging check copies only Python source/tests/tools plus README
and `.gitignore` into a temporary checkout. It copies no images, `test-results`,
models, dependency manifests or venv; the child uses isolated Python mode and
checks that project imports come from the temporary checkout. It then runs full
discovery: **303 tests, 21 optional-local-evidence skips, zero failures/errors**.
It is excluded from normal `test*.py` discovery to avoid recursive copy tests.

### Local image licensing and privacy

`images/` (including `images/ubunchu-ja-test/` license/sample metadata) and
`test-results/` are ignored local evidence, not part of the source package.
Saved comparisons/manifests may contain original manga text, image paths, crops,
overlays and model outputs; do not include them in source-only QA packaging.
The six Ubunchu pages are a CC BY-NC, single-series evaluation subset; that does
not grant redistribution rights for other original manga pages. Keep authorized
originals and their local metadata together outside the published source snapshot.
Synthetic test images are created in temporary directories and require no original
page assets. Ignore rules do not remove files or untrack anything already tracked;
the parent performs final staging review.

## Standalone diagnostics

Visualization is diagnostic-only and has moved out of the pipeline. There are
no `OCR.visualize_groups*` methods or `Text_Box.debug_color` property now:

```python
from tools.diagnostics import debug_color, visualize_groups, visualize_groups_polygon

visualize_groups(image_path, groups, "rectangles.png")
visualize_groups_polygon(image_path, groups, "polygons.png")
color = debug_color(box.status)  # or debug_color(box)
```

Helpers accept explicit paths/groups, save new PNG files, never show a GUI, and
do not mutate groups/boxes. Hardcoded OCR/provider/GUI demos were removed from
pipeline entrypoints; production `tralens.rendering.cover_text`/`draw_text` are unchanged.
`tools/diagnostics.py` does not import the pipeline or initialize OCR/models, including
on import. Its CLI reads existing comparison JSON (`input` + `boxes`, or
`summary.image` + `boxes`) and original images only.

Prepare bounded, headless review artifacts for the seven existing saved pages:

```sh
.venv/bin/python tools/diagnostics.py \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . --output-dir test-results/package-refactor-relationships-preflight-v1 \
  --max-cases 28 --max-requests 0
```

For the parent's approved live diagnostics, explicitly select the Ollama origin
and **exact already-installed Qwen3.8 model tag** (replace the example tag below
with the configured local tag; there is no model discovery/pulling/download):

```sh
.venv/bin/python tools/diagnostics.py \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . --output-dir test-results/package-refactor-relationships-live-v1 \
  --max-cases 28 --max-requests 28 \
  --run-model --endpoint http://127.0.0.1:11434 --model qwen3.8
```

Without `--run-model`, no request is made. Failed requests consume the request
bound, with no retries. Hard limits: 256 cases, 64 requests, timeout 1–300 seconds,
context padding 0–128 pixels. Defaults: 28 cases, 12 requests, timeout 120 seconds,
padding 24; model options are temperature 0, seed 0, num_predict 256, num_ctx 4096.
Output directories must not already exist, so inputs/prior artifacts cannot be
overwritten. Endpoint credentials, proxies and redirects are not used.

Optional repeatable selection: `--case lmgJZ-comparison:8:7` (comparison filename
stem and two source IDs in order). Integer IDs, including negative IDs, and string
IDs retain their types. Otherwise nearest bbox pairs are sampled round-robin
across pages, without confidence/status/overlap filtering. This is bounded
diagnostic sampling, not a relationship classifier or representative evaluation;
explicitly select independent/negative cases as needed.

Each case saves original and annotated contextual PNG crops; review sheets use
thumbnails only. `review.json` and `requests.jsonl` retain source snapshots/IDs,
input hashes, prompts, image paths/bounds, model/endpoint/options, raw responses,
timings and errors. Valid labels are `furigana`, `duplicate`, `same_text_block`,
`independent`, `uncertain`. IDs must match the supplied pair exactly, including
order/type. Malformed JSON, missing/foreign IDs, invalid labels and request errors
remain unresolved; negative/independent results are retained. All results are
**advisory only**: nothing updates pipeline text/status/groups, erases or translates.
No live diagnostic calls are exercised by the offline tests.

## Optional reviewed recognition units

`tralens.ocr.recognition_units` provides `snapshot_detections`, `plan_recognition_units`,
`prepare_unit_crop` and `recognize_unit`. It does not import OCR engines or
diagnostics. `RawDetection`/`PageProvenance` are frozen snapshots: original Paddle
text, confidence, polygon, ID, current status/reason and metadata remain separate
from mutable unit results. Existing `Text_Box` objects are never changed by this
API. Main readings and explicitly reviewed companion columns form ordered units;
reading annotations have explicit child-to-parent mappings. Small size, confidence
or a Qwen label does not make a detection furigana.

Each immutable `UnitPlan` has a stable page-bound ID, member/main IDs, roles,
annotation parents, direction, granularity and reviewed bubble ownership. Every
relationship needs `reviewed: true`, a nonempty reviewer, existing source IDs and
matching page provenance. Duplicate records, multiple parents, cycles, conflicting
unit membership and uncertain relationships remain unresolved. Qwen diagnostic
outputs are advisory and are **not** accepted as reviewed unit input.

Unit crops include reviewed members without masking/rotation/contrast changes.
Unrelated glyphs inside the final rectangular crop, malformed geometry, clipped
members and mixed/unknown ownership block recognition; only padding can shrink.
File hashes and decoded RGB pixel fingerprints bind reviews/crops to the original
page. Unknown ownership is not inferred from the previous grouping/bubble signal.

Unit callbacks receive `(crop, *, source_unit_id, member_ids, crop_bounds)` and
return text only. Candidates, disagreements and crop provenance stay on units,
never copied into member boxes. One primary attempt and at most one explicitly
requested alternate are allowed (unpadded for legacy crops; identical extent for
reviewed utterance crops). Conflicting alternate candidates remain
review. A block candidate is compared only with joined **main** readings in an
explicitly reviewed order/granularity; annotation readings are excluded. Missing
or mismatched granularity/blank main readings cannot establish agreement.
Normalized agreement is consistency, not independent transcription truth.
Units never automatically translate/erase or suppress per-box processing;
`fallback_source_ids` retains all raw IDs until a separate unit rendering contract
exists. The default OCR path is unchanged.

### Optional reviewed full-utterance extent

Detection polygons sometimes omit trailing strokes or punctuation. An optional
`ReviewedUtteranceExtent(unit_id, member_ids, page, bounds, source, reviewed=False,
reviewer="")` supplies a separate recognition rectangle. Pass records through
`plan_recognition_units(..., utterance_extents=[extent])`; existing calls and
legacy padding/neighbor trimming are unchanged when no extent is supplied.
This is **not** a detection polygon change or a translation/erasure/render region.

Each reviewed page in the diagnostic input may have an `utterance_extents` list.
Each record must include these fields:

```text
unit_id: exact plan.unit_id from a model-free reviewed-unit preflight
member_ids: exact ordered plan.member_ids, including approved annotations
page: full exact page object (page_id, image_path and all three SHA256 fields)
bounds: [left, top, right, bottom] in original-image, half-open pixel coordinates
source: nonempty description of the original-page review evidence
reviewed: true
reviewer: nonempty human reviewer identity
```

Bounds must be finite integer pixel edges, ordered, within the original image,
and contain every included polygon. Fractional edges are rejected, not rounded.
Stale page hashes, foreign unit/member bindings, duplicate declarations, missing
review, malformed geometry, excluded members and mixed ownership stay unresolved.
An unmatched unit binding blocks the page's units rather than falling back to
legacy crops. All nonmember polygons are checked against the **actual rectangle**,
including rotated glyphs: any text intersection blocks the extent, even within
the same bubble. A bbox overlap alone is not a polygon intersection.

Approved extents are used exactly: no extra padding, trimming, whitening, masking,
rotation or contrast changes. Primary and bounded alternate crops both preserve
the entire reviewed extent; changing/removing it between attempts blocks the
alternate. Crop provenance records `detection_union_bounds`, `reviewed_extent`
(supplied bounds/source/reviewer/page/unit/member evidence), actual `bounds`, and
`comparison_granularity`. Diagnostic original/annotated and attempt artifacts use
that rectangle; unsafe requests are exported as `requested_utterance_extents`
with unresolved errors, never recognized or silently replaced.

Full-utterance candidates have
`comparison_granularity: "full_utterance_without_complete_reference"` and remain
review with `full_utterance_reference_completeness_unverified`, even if normalized
text equals joined main readings. Those readings may omit punctuation. Differences
are retained as advisory spans when a joined reference exists, not as proof of
correctness. No independent full-utterance transcription/reference API is implied.

#### Parent completeness experiment

Keep `test-results/unit-reviewed-ownership-v2.json` unchanged. First export stable
unit/member bindings and existing original crops without any model:

```sh
.venv/bin/python tools/diagnostics.py --unit-experiment \
  --comparison test-results/crop-isolation-v2/*-comparison.json --image-root . \
  --unit-review test-results/unit-reviewed-ownership-v2.json \
  --output-dir test-results/utterance-binding-preflight-v1
```

Copy that review input to `test-results/unit-reviewed-utterance-v1.json` and add
explicit extent records after reviewing the **original image**. Copy exact
`unit.plan.unit_id`, `unit.plan.member_ids` and `unit.plan.page` from the
corresponding record in `units.json`. The warning-context evidence under
`test-results/transcription-review-v1/warning-context.png` motivates checking
trailing strokes/exclamations, but supplies neither approved bounds nor automatic
authorization. The parent chooses bounds; there are no sample-ID rules.

Validate the new review with another model-free run using
`--unit-review test-results/unit-reviewed-utterance-v1.json` and a fresh output
directory, inspect original/annotated artifacts, then explicitly run:

```sh
.venv/bin/python tools/diagnostics.py --unit-experiment \
  --comparison test-results/crop-isolation-v2/*-comparison.json --image-root . \
  --unit-review test-results/unit-reviewed-utterance-v1.json \
  --output-dir test-results/utterance-completeness-live-v1 \
  --run-unit-model --max-unit-attempts 12 --unit-alternate \
  --manga-model-path /home/Danuu/.cache/huggingface/hub/models--kha-white--manga-ocr-base/snapshots/aa6573bd10b0d446cbf622e29c3e084914df9741
```

### Parent saved-page experiment

First export exact page provenance and an empty review template, with no model:

```sh
.venv/bin/python tools/diagnostics.py --unit-experiment \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . --output-dir test-results/package-refactor-unit-preflight-v1
```

Copy/fill `review-template.json` into an explicit reviewed-input file. Keep each
`page` object exactly unchanged. Each page has these lists (example IDs below are
external parent-review evidence only, **not production rules**):

```json
{
  "relationships": [
    {"source_id": 8, "parent_id": 7, "relationship": "furigana", "reviewed": true, "reviewer": "Daniel"}
  ],
  "ownerships": [
    {"source_id": 7, "bubble_id": "manually-reviewed-bubble-A", "reviewed": true, "reviewer": "Daniel"},
    {"source_id": 8, "bubble_id": "manually-reviewed-bubble-A", "reviewed": true, "reviewer": "Daniel"}
  ],
  "units": [
    {"main_ids": [7], "direction": "vertical", "granularity": "column", "reviewed": true,
     "order_reviewed": true, "granularity_reviewed": false, "reviewer": "Daniel"}
  ]
}
```

Set ownership/order only after original-page review. Leave
`granularity_reviewed: false` unless each main detection's reading granularity is
confirmed; this still permits a unit candidate but keeps it in review. Companions
need a reviewed `same_text_block` relation and both IDs in ordered `main_ids`, with
`granularity: "block"`. The supplied external annotation associations are
8→7, 12→11, 18→17, 21→14, 22→19; 16/13 are companions. They do not independently
establish bubble ownership, reading order or transcription truth. Evidence remains
in `test-results/blocked-manual-review-v1` and `test-results/crop-isolation-v2`.

Validate the filled file without models by adding
`--unit-review test-results/unit-reviewed-input-v1.json` and a fresh output
directory. Then the **parent**, not the agent, can explicitly run the installed
local MangaOCR model:

```sh
.venv/bin/python tools/diagnostics.py --unit-experiment \
  --comparison test-results/crop-isolation-v2/*-comparison.json --image-root . \
  --unit-review test-results/unit-reviewed-input-v1.json \
  --output-dir test-results/package-refactor-unit-live-v1 \
  --run-unit-model --max-unit-attempts 12 --unit-alternate \
  --manga-model-path /home/Danuu/.cache/huggingface/hub/models--kha-white--manga-ocr-base/snapshots/aa6573bd10b0d446cbf622e29c3e084914df9741
```

The unchanged optional adapter is loaded only when a validated crop actually
requests recognition. `--max-cases` bounds units (1–256); `--max-unit-attempts`
bounds all primary/alternate calls together (0–64), including failures. Unit
planning defaults to no recognition. `units.json` retains raw snapshots, reviewed
input hash, unit plans/candidates/status/reason/disagreements, original attempt
crops, model settings and timings; review sheets are diagnostic-only. Existing
output directories are refused. Offline fixtures exercise this API with test-only
callbacks; real saved-page preflight leaves unreviewed ownership unresolved.

## Explicit checked OCR transcription selection

`tralens.ocr.transcription_selection` is a small model-free production API:

- `bind_candidate(unit_id, member_ids, page, candidate_index, text, engine, crop)`
  returns immutable `CandidateEvidence`. Pass a `CropProvenance` or its dictionary.
  The SHA256 `candidate_id` binds exact text, engine, attempt index, full crop
  provenance, unit/member IDs and exact original page/image provenance.
- `TranscriptionReview` describes human checks and selection/correction.
- `select_transcription(candidates, reviews)` handles **one unit atomically** and
  returns `SelectionResult(selected=SelectedTranscription(...))` or unresolved
  `errors`. Submit exactly one review per unit. Duplicate/conflicting records,
  incomplete checks and stale bindings produce no selected text for that unit.

Selection is **separate OCR output**, not a box update, acceptance-status change,
translation permit or render region. Raw detections, every original candidate,
engine statuses and crop extents remain unchanged. No confidence, normalized
agreement, previous status, PM suggestion or Qwen label establishes review/truth.
Production selection performs no metrics, visualization, file I/O or model calls.

## Opt-in automatic manga structure proposals

No manual records are needed to **generate** these proposals. The production
`tralens.ocr.automatic_units` API is:

```python
from tralens.ocr.automatic_units import propose_automatic_units

# Frozen RawDetection snapshots plus the exact original RGB PIL image:
batch = propose_automatic_units(detections, image, max_detections=512)
units = batch.proposals
coverage = batch.coverage
```

For the existing OCR workflow, explicitly request proposals:

```python
from tralens.ocr import OCR

ocr = OCR(image_path)  # Existing explicit Paddle initialization/use unchanged.
groups = ocr.process_images(propose_units=True)
proposal_batches = ocr.automatic_unit_proposals  # One batch per result page; empty page -> None.
```

`propose_units=False` is the default. The existing return value remains per-page
`Text_Group` lists; original boxes/groups are not replaced, filtered or mutated by
proposal generation. Manga mode is required for opt-in proposals. No additional
recognizer/model is called. `propose_page_from_boxes(boxes, image_path)` provides
the same concrete file/snapshot coordination without initializing OCR. Its
`comparison_sha256` is explicitly a canonical original detection-snapshot hash,
not a saved comparison-file hash or a human review claim.

### Automatic evidence and safety boundaries

`grouping.analyze_manga_regions` and `grouping.group_manga_boxes` are engine-free
helpers reusing the existing enclosed-white-region and complete-link grouping
algorithms. Automatic proposals analyze all valid original detections, including
blank/uncertain ones, without using confidence to decide structural roles.
Reliable region labels constrain merges. Alignment, orientation, scale, gap and
complete-link anti-bridging constrain the fallback; even a shared region does not
join distant incompatible columns. Order reuses manga right-to-left vertical
columns and top-to-bottom horizontal rows, with fragment order within bands.

Policy `automatic_manga_structure_v2` allows short/square reading rectangles:
the **parent** defines the axis, rather than requiring the child's aspect ratio
to match. Smaller relative thickness, bounded side adjacency, along-axis overlap
and containment still constrain candidates. Right-of-vertical/above-horizontal
placement can overlap the parent's outer footprint, since that footprint may
already include annotation ink. No polygon is resized or replaced.

`automatic_evidence.OriginalInkEvidence` measures original-image component size,
flow and positions in separate Otsu analysis masks. Smaller side-adjacent glyphs
can support a tentative annotation even when its OCR is blank/ASCII or the main
reading is bad. Missing kana/Kanji corroboration is reported as word uncertainty,
not proof of independent dialogue. Explicitly conflicting original glyph scale,
flow or boundary evidence rejects the relation and retains the detection as main.
Without sufficient pixels or wording, the candidate relationship remains
unresolved. Kana/Kanji is corroboration, not truth; dash-only text does not provide
wording corroboration. Different reliable region labels reject the association. Multiple
plausible parents or nested/dual roles remain unresolved; the detection stays a
main, rather than disappearing or gaining two parents. Small ordinary dialogue,
square/unsupported SFX and misplaced kana are not annotations merely because of
size or confidence. Comparable original glyph sizes help distinguish small normal
dialogue even when the detection rectangle is relatively narrow. Some real small
dialogue can still mimic annotation geometry:
all such assignments remain **tentative**, never reviewed truth.

Every valid in-page raw detection is assigned exactly once as a main or tentative
annotation. Invalid/clipped polygons are explicitly `unassigned`; duplicate IDs,
foreign provenance and excessive detection counts reject the page, with no
silent truncation. IDs/order are deterministic for the same original provenance
and geometry; source IDs/polygons and every original detection are preserved.
JSON records expose possible parents, unresolved reasons, original region labels
and containment-vote fractions. These fractions are measured geometry support,
**not calibrated confidence in bubble ownership or OCR accuracy**.

A shared connected white region is **not** sufficient permission to merge.
Automatic grouping supplies an optional `merge_guard` to the reused grouping
helper; default/legacy grouping never supplies it and remains unchanged. The
guard looks for long original dark strokes in facing main-glyph/column corridors,
including bounded endpoint probes near balloon tails/connectors. It sees original
grayscale pixels—not OCR-erased region masks—and vetoes a merge across an observed
outline/panel boundary even when both detections share one white component.
These analysis probes do not expand recognition crops. Missing/open corridors
are inconclusive, not ownership proof. Unsupported blank/noisy mains remain in
the exact-once partition, but cannot mediate a merge into a larger group.

`batch.rejected_merges` exports refused candidate unions, reasons, unsupported
source IDs, actual scan/corridor bounds and measured dark-stroke spans/thresholds.
Relationship records include child/parent glyph evidence and rejected boundary,
scale or flow reasons. Coverage includes the number of distinct refused unions,
not a model call count. External reviewed sample pairs appear only in offline
evaluation tests; neither old PM labels nor sample IDs are production inputs.

`AutomaticUnitProposal` is deliberately NOT a reviewed `UnitPlan`. Proposal IDs
have a `proposal-` prefix; `reviewed` is always false and structural status is
`proposal_unverified`. Automated metadata is not accepted in place of human
ownership/order/extent/transcription review, and does not bypass the existing
reviewed-unit crop/recognition API. No translation, erasure or automatic transcription
selection is enabled. Proposal generation alone never enables recognition; the
separate candidate handoff below requires explicit opt-in.

Structural reasons and `word_reasons` are separate. A blank or uncertain reading
does not prevent an otherwise compatible structural proposal. Unknown ownership,
orientation and annotation-role conflicts remain exposed. Preview bounds are
the original member polygon rectangle, with no masking, rotation, whitening,
contrast change or padding increase. `crop_safe` means only valid geometric
isolation in that rectangle; it does NOT verify ownership or utterance coverage.
Every crop reports `unverified_full_utterance` and warns that punctuation outside
detected polygons has not been ruled out. Invalid nonmember geometry or actual
polygon intersection (including rotated glyphs) makes isolation unresolved.
Unsafe previews may be shown for diagnosis but never authorize recognition.

### Parent saved-page proposal command (zero model calls)

```sh
.venv/bin/python tools/diagnostics.py --automatic-proposals \
  --comparison test-results/crop-isolation-v2/*-comparison.json --image-root . \
  --output-dir test-results/automatic-unit-diagnosis-parent-v3 \
  --max-cases 7 --max-detections 512
```

Use a fresh output directory. This mode needs no PM labels/review JSON and cannot
be combined with model/review/selection modes. `--max-cases` bounds input pages
(1–256); `--max-detections` bounds each page (1–2048). `automatic-proposals.json`
contains raw snapshots, ordered mains, tentative annotation mappings/conflicts,
region evidence, structural/word/crop/completeness reasons, per-page and total
coverage (raw/main/annotation/unassigned counts, units/conflicts/unsafe crops), and
page errors. Original/annotated unit crops, page overlays with IDs/roles/order and
headless review sheets are diagnostic-only. Counts describe proposal coverage on
successfully loaded pages, **not accuracy or recognition performance**. No model
is loaded and recognition-call count is zero in this default model-free mode.

### Exact unsafe/split diagnosis

Every proposal now records `crop_conflicts`: exact nonmember source IDs/polygons,
the polygon clipped against the **actual final rectangle**, intersection area,
nonmember area, and overlap areas against each included member polygon. Invalid
geometry is identified explicitly instead of being treated as no intersection.
`rejected_merge_evidence` attaches relevant refused unions and original boundary/
unsupported-bridge reasons to each affected proposal, not just a page-level list.
Raw evidence is bound by `batch.evidence_sha256`.

Tools classify observed footprint overlap versus rectangular-gap contamination,
report neighbor roles/proposal IDs and candidate relation records, layout/script
evidence, and other proposals sharing a region. None of those labels asserts
balloon ownership or a shared utterance. Mixed scripts, horizontal narration,
ordinary dialogue and SFX-like layouts can all be unsafe; unsafe count is **not**
furigana count. Diagnosis categories can overlap and their counts are not accuracy
metrics. This handoff slice makes **no grouping/merge/barrier change**: available
examples did not justify blindly relaxing fragmentation or crop isolation rules.

### Separate automatic candidate recognition handoff

`tralens.ocr.automatic_recognition` is an engine-free production API:

- `prepare_automatic_crop(proposal, batch, image, original_image_bytes=...)`
  revalidates the original file SHA256, RGB pixel fingerprint, page binding, raw
  evidence hash, member geometry/partition and full regenerated automatic
  proposal/region/relationship evidence. It checks the actual rectangle against
  **all** nonmembers again; a saved `crop_safe` flag is never sufficient permission.
- `recognize_automatic_proposal(result, proposal, batch, image, callback, ...)`
  permits one primary attempt, including failed initialization/callback/output.
- `recognize_automatic_batch(batch, image, callback, original_image_bytes=...,
  model_identity=..., model_settings=..., max_attempts=20, results=None)` applies
  a bounded page budget and returns separate `AutomaticRecognitionResult` records.
  Reuse `results` to retain history and prevent retries; starting a new results
  collection is a separate explicitly authorized experiment, not an automatic retry.

Callbacks receive `(original_rgb_crop, *, source_proposal_id, member_ids,
crop_bounds)` and return text only. Inputs and all candidates stay separate from
raw boxes, `Text_Group`s, reviewed recognition units and checked selections.
For explicit production coordination after `process_images(propose_units=True)`:

```python
results = ocr.recognize_automatic_proposals(
    callback, model_identity="explicit-installed-recognizer-identity",
    model_settings={"local_only": True}, max_attempts=20,
)
```

This separate method stores `ocr.automatic_recognition_results`; the existing
`process_images` return/default path does not change. It reuses retained results
on subsequent calls and enforces a global 0–20 attempt budget across pages.
Production coordination needs an explicitly supplied callback, never an implicit
model/provider. It does not load the optional adapter or call anything on import.

Eligibility is deliberately conservative: invalid/clipped geometry, stale or
mutated evidence, duplicate/foreign membership, unsafe rectangles, unknown
orientation, missing reliable automated region support for **any included
member**, mixed regions, and ambiguous/conflicting relationships block recognition.
Region evidence is still automated—not proof of ownership—so all its structural
and full-utterance uncertainties accompany any generated candidate. Tentative
annotations with stable structural support can be included, but their readings
are excluded from `reference_main_text`.

Horizontal Japanese can be eligible. Clearly Latin/other non-Japanese or mixed
Japanese/non-Japanese script readings are not routed to MangaOCR. Blank/punctuation-
only evidence is skipped; Han-only readings are language-ambiguous and explicitly
flagged if otherwise eligible. Script detection is a routing hint, not a language
truth claim. No hallucinated translation or universal-recognizer assumption is
introduced. The exact unnormalized main reading is retained for before/after text
and differing spans (including punctuation/small kana); these are consistency
observations, not accuracy metrics.

Every candidate remains `automated_unverified`, `reviewed: false`, with original
crop/page/member/raw/proposal hash provenance, model identity/settings and
`unverified_full_utterance`. Even exact text agreement cannot grant human-check
or accepted status. No reviewed `UnitPlan` is manufactured; no unit translation,
erasure, per-box write or selection is enabled. There is no alternate/retry path,
mask, contrast change, rotation or padding inflation. Errors consume the attempt
bound and are retained alongside skipped reasons.

After inspecting the model-free diagnosis above, the **parent** may explicitly run
at most 20 local MangaOCR attempts with already-existing weights:

```sh
.venv/bin/python tools/diagnostics.py --automatic-proposals \
  --comparison test-results/crop-isolation-v2/*-comparison.json --image-root . \
  --output-dir test-results/automatic-unit-candidates-parent-v1 \
  --max-cases 7 --max-detections 512 \
  --run-automatic-model --max-automatic-attempts 20 \
  --manga-model-path /home/Danuu/.cache/huggingface/hub/models--kha-white--manga-ocr-base/snapshots/aa6573bd10b0d446cbf622e29c3e084914df9741
```

The optional unchanged local-only adapter is imported/initialized lazily only
after a safe eligible crop reaches recognition. Default proposal/diagnosis runs
perform zero model calls. Output remains `automatic-proposals.json` plus original/
annotated crops and sheets: each record includes diagnosis, preflight eligibility,
attempt/skip/error state, actual supplied bounds and members, main before text,
candidate after text/spans, and complete unverified candidate provenance. Settings
record model identity/path and crop policy. `recognition_calls` counts callback
attempts including failures; `live_model_calls` counts actual adapter invocations
after initialization (so initialization failures do not pretend a model ran).
Injected offline callbacks are marked `test_only` and never count as live calls.
Use a fresh output directory; the agent performs only offline callback tests.

Remaining limits: isolated crops can still omit undetected punctuation or
misidentify connected-white-region ownership. Unknown region support is skipped
rather than approved. These conservative gates may produce fewer than 20 calls
and leave SFX, narration, fragments and mixed-script cases unresolved. Existing
proposal completeness/ownership limitations are not repaired by a model output.

Limitations: component scale/flow can be distorted by touching strokes, fragmented
glyphs, screentone or outlines; boundary probes can conservatively split true
same-balloon columns or miss faint/open dividers. The reused region mask can miss open/large/stylized balloons or
misinterpret small enclosed panel regions; geometric fallback cannot establish
unknown balloon ownership. Tilted, mixed-orientation, heavily fragmented or
kana-only mains may remain separate/unresolved. Completeness outside OCR polygons
requires later evidence/review; previews do not invent a full-utterance extent.

## Checked transcription review schema

The diagnostic document is:

```text
units_artifact_sha256: SHA256 of the exact existing units.json bytes
reviews: [one TranscriptionReview per submitted unit]
```

Each record requires:

```text
unit_id, member_ids: exact unit and ordered included member IDs
page: full exact original PageProvenance object (all IDs/paths/file/pixel hashes)
candidate_id: exact hash exported in candidate-options.json
candidate_text: exact saved candidate text (not the correction)
crop_bounds: exact ACTUAL candidate crop bounds, not a planned render region
mode: "existing_candidate" or "corrected_text"
review_kind: "provisional_visual" or "independent_reference"
reviewer, source: nonempty human identity and original-page review evidence
reviewed, wording_checked, order_checked, full_utterance_checked: true
wording_uncertain, order_uncertain, coverage_uncertain, punctuation_uncertain: false
corrected_text: null for existing_candidate; exact reviewed string for corrected_text
independence_confirmed: false for provisional_visual
reference_source: "" for provisional_visual
```

All flags must be actual JSON booleans. Template checks default false and
uncertainties true; do not clear uncertainty merely to make a selection pass.
The wording/full-utterance/punctuation checks include exact dash representation
and count. A candidate-specific selection does **not** verify a model's elongated
dash automatically. If count/representation is uncertain, retain
`punctuation_uncertain: true` and leave that unit unresolved. Earlier PM visual
preferences are provisional, not independently annotated references. No supplied
sample wording or IDs are encoded into production rules or generated approvals.

An `independent_reference` additionally requires `mode: "corrected_text"`,
`independence_confirmed: true`, and a nonempty `reference_source` describing a
separately annotated original-page transcription. Establish the reference
independently of the candidate, then bind it to the saved case for evaluation.
Choosing or correcting a model output by visual comparison alone is **not** an
independent reference, even when its text happens to be correct. Missing/false
independence claims cannot qualify for CER. To retain both provisional and
independent review histories for one unit, use separate review documents/runs;
multiple records in the same run are ambiguous, not silently applied in sequence.

### Saved-artifact workflow (parent, no models)

Export an unapproved template, all candidate identities, original/annotated crops
at their actual saved extents, and a headless review sheet:

```sh
.venv/bin/python tools/diagnostics.py --selection-report \
  --units-artifact test-results/utterance-complete-live-v2/units.json \
  --output-dir test-results/transcription-selection-preflight-v1 --max-cases 28
```

Copy `selection-review-template.json` to
`test-results/transcription-selection-reviewed-v1.json`. The default record binds
the first candidate but is **not** approved. To select another saved candidate,
copy its exact identity/text/crop/page/member binding from `candidate-options.json`.
For reviewer correction, keep the candidate binding unchanged, use
`mode: "corrected_text"` and supply `corrected_text`. Fill checks only after
original-page review; leave the uncertain warning dash unresolved. Omitted cases
remain unresolved, never auto-selected.

```sh
.venv/bin/python tools/diagnostics.py --selection-report \
  --units-artifact test-results/utterance-complete-live-v2/units.json \
  --selection-review test-results/transcription-selection-reviewed-v1.json \
  --output-dir test-results/transcription-selection-report-v1 --max-cases 28
```

Outputs are new-only: `selection-report.json`, `selection-review-template.json`,
`candidate-options.json`, diagnostic original/annotated PNGs and review sheets.
The report retains submitted evidence, separate selected text, unresolved errors,
reviewed/unresolved/provisional/independent counts, and coverage claims versus
successfully complete checks. Original file/pixel hashes and candidate bounds are
verified against the original image. A stale whole-artifact hash or foreign review
unit blocks all selections in that run; malformed/duplicate/stale records for a
known unit block that entire unit, without partial application. Saved plan errors
remain unresolved. Source page errors are reported, not hidden from coverage.

CER/edit distance lives **only in tools**. It compares the original bound candidate
(never reviewer-corrected output) to each explicitly independent reference.
Normalization is **none**: Unicode codepoints, retaining whitespace, full/half-width
punctuation, small kana, elongated strokes and combining-mark distinctions. CER is
sum of Levenshtein edits divided by sum of independent reference codepoints; the
report states that denominator, eligible case count and unit coverage over all
saved units. Cases over 4096 codepoints are explicitly excluded from metrics.
With no eligible independent references, denominator is zero and CER is `null`;
provisional visual selections never contribute. This is a measured reference
**subset**, not a whole-dataset accuracy claim. No metrics run in production OCR.

## Deterministic saved structural evaluation (tools only)

`tools/structural_evaluation.py` consumes an existing rich saved proposal artifact;
it never imports production OCR, regenerates grouping, recognizes text or calls a
provider. The default selection is at most 21 cases, capped at three per page:
observable strata first, then balanced unsafe/unknown-region/relationship-conflict
cases, with stable page/proposal-ID ordering. These tags are saved observations,
not ground truth: horizontal layout is not automatically narration, and unknown
orientation is only an SFX **candidate**. Tentative annotation/multi-main cases and
unverified crop completeness are included without declaring relationships approved.

```sh
.venv/bin/python tools/structural_evaluation.py \
  --artifact test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json \
  --output-dir test-results/structural-evaluation-v1 --max-cases 21 --per-page 3
```

Outputs under the new-only directory:

- `evaluation.json`: full saved-artifact counts, overlapping risk categories,
  saved eligibility blockers and representative proposal/member IDs. The dominant
  blocker is a recorded gate, **not measured model error**. Completeness uncertainty
  is reported separately rather than counted as an exclusive failure everywhere.
- `evidence.json`: hash-bound selected cases with the entire original source
  record (statuses, candidates, conflicts, crop bounds and uncertainty unchanged),
  plus full raw/page batch snapshots. Original file/RGB hashes are verified.
- `case-XX-original.png`: exact saved crop pixels, never enlarged for recognition.
  `case-XX-context-{original,overlay}.png`: diagnostic-only context; overlays mark
  main/annotation/nonmember footprints and the unchanged actual crop rectangle.
- `structural-reference.json`: initially **unannotated** structural observations.
  After inspecting original/context evidence, fill only provisional visual
  observations, name the observer (including AI if applicable), and describe the
  source/uncertainty. Narration, Latin/mixed text, stylized SFX, ownership ambiguity,
  tentative furigana/companions and completeness can be recorded explicitly.

Reference records bind case ID, original page/hash, proposal/source-record hashes,
member IDs and crop bounds. They cannot set `candidate_reviewed` or
`independent_transcription_reference` true, claim full completeness, or silently
override source OCR/status. `state` is `unannotated` or
`provisional_visual_observation`, never a human OCR-check status. The format is
not a production review/selection input. Preserve the generated bindings when
copying/filling the reference file; validate a filled file in another fresh run
by adding `--reference path/to/structural-reference.json` with identical selection
settings and the same saved artifact. Stale/foreign/duplicate bindings fail closed.

The harness emits no CER or whole-dataset accuracy claim. Selected-case visual
observations are a small structural reference subset, separate from full-artifact
gate counts. Sparse categories may remain uncovered under a small per-page cap;
both selected and full category counts are reported. No grouping fixes follow
automatically from any classification or count.

### Component-shape regression evidence

Run `.venv/bin/python -m unittest tests.ocr.test_component_shapes -v` for
model-free shape and saved-page regressions. Synthetic images exercise the actual
production grayscale/inverted-Otsu mask, OCR-polygon ink removal and 8-connected
components. Cases include compact/rounded balloons, diamonds, a long tail, a
slanted artwork pocket, open/page-edge regions, oversized panels, connected
balloons and a detection that erases part of a divider. Original-pixel boundary
vetoes remain necessary for both column merges and annotation candidates.

The two saved `ubunchu02_04` components reproduce rectangular fills
`0.5042107958067211` and `0.5450255404079699`, with full polygon overlap and the
other production predicates passing. Their original component overlays also show
white connectivity into surrounding artwork. Neither lowering rectangular fill
nor substituting convex-hull fill establishes balloon ownership: the slanted
artwork fixture has hull fill above `.95` while failing rectangular fill. A
compact enclosed artwork pocket can even pass the existing region predicate.
These tests document the heuristic's limits, not new positive ownership labels.

Saved-page regressions compare all seven pages against
`automatic-unit-diagnosis-parent-v3/automatic-proposals.json`: immutable raw
evidence, ordered memberships, annotation relationships, regions, rejected merges,
crop conflicts and model-free eligibility. They also verify exact original crop
pixels and retain Latin/mixed/inconclusive routing. This comparison performs no
recognition and writes no artifacts. It establishes reproducibility, not improved
OCR accuracy or full-utterance completeness. The six Ubunchu pages remain a
single-series CC BY-NC evaluation subset.

### Opt-in complementary-connectivity ownership

`propose_automatic_units(raw, image, complementary_ownership=True)` enables a
narrow fallback after grouping and tentative annotation assignment. The default
remains `False`. The same optional keyword is available on `propose_page_from_boxes`;
the default per-box OCR API and legacy grouping/region analysis are unchanged.

An 8-connected dark outline can be crossed diagonally by 8-connected white
components. Four-connected white components respect those diagonal outline
contacts without changing any pixels. The fallback is limited to an already
grouped multi-main dialogue whose original glyph flow, geometric compatibility
and original-pixel boundary checks agree. All members must have enclosed unmasked
original support, and the original production component must fail **only** fill.
Every full detection footprint must overlap the same enclosed four-connected
component by at least `.9`; that component must still pass rectangular fill
`>= .55` and the original area limits. A resolvable white neck is not cut. No
morphological closure/opening, hull fill, local crop fill or image alteration is used.

Existing positive region IDs retain their eight-connected meaning. Negative IDs
identify supplemental four-connected components. `AutomaticProposalBatch` adds
`ownership_policy` and `ownership_repairs`, retaining original production,
unmasked and complementary component statistics, original glyphs and barrier
checks. Enabled proposal IDs bind the supplemental policy. Preflight recomputes
the exact selected policy and rejects evidence/policy drift; all-neighbor crop
isolation, routing, wording uncertainty and unverified completeness remain separate.

Generate model-free proposals into a fresh directory:

```sh
.venv/bin/python -m tools.diagnostics --automatic-proposals \
  --complementary-ownership \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . \
  --output-dir test-results/automatic-unit-diagnosis-session-ec273496-complementary-v1
```

The historical v1 seven-page regression has 268 detections, 136 proposals and
24 unsafe crops with either policy. Ordered memberships, annotation relationships,
crop bounds, crop conflicts and original evidence are identical. Supplemental
ownership supports five proposals; four gain preflight eligibility (`52 -> 56`).
The fifth retains its script-routing block. This is an automated eligibility
improvement, not measured recognition accuracy or a full-utterance guarantee.
The second reported `ubunchu02_04` component remains unsupported; ordinary
four-connected gaps and artwork semantics are not solved by this fallback.

### Opt-in bounded vertical companion regrouping

Ownership evidence does not establish full-utterance completeness. An additional
`regroup_companions=True` keyword on the automatic proposal APIs can join contiguous
vertical companion groups that exceed the default complete-link span cap. It
requires `complementary_ownership=True`; both keywords default to `False`.
The normal OCR facade exposes both options without changing its returned legacy
groups or mutating raw boxes:

```python
groups = ocr.process_images(
    propose_units=True,
    complementary_ownership=True,
    regroup_companions=True,
)
batches = ocr.automatic_unit_proposals
```

The repair requires two multi-main groups in the same supplemental component,
compatible vertical orientation and original glyph flow, aligned reading extents,
bounded total span, and a local non-interleaving column seam. Its original-ink
corridor must be at least 98% white, with at least half of its columns white over
the entire facing height. The original endpoint-probed boundary veto must also
pass, and the union rectangle must be isolated from **every** nonmember polygon.
Same-region ownership alone cannot trigger a merge. Existing annotations retain
their parents and never become main columns. No horizontal regrouping is inferred.

The batch exports a separate `companion_policy`, `companion_regroupings` and
`companion_rejections`, including previous groups, seam/corridor evidence, bounds
and conflicts. Enabled proposal IDs bind the companion policy. Fresh preflight
recomputes that policy and rejects changed evidence. Regrouped proposals remain
`unverified_full_utterance`, with an explicit
`companion_regrouping_not_full_utterance_proof` reason; no crop gains verified
completeness or accepted text from regrouping.

```sh
.venv/bin/python -m tools.diagnostics --automatic-proposals \
  --complementary-ownership --regroup-companions \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . --output-dir test-results/ownership-companions-parent-v1
```

The original multi-main branch retains the supported `ubunchu02_04` five-column
regrouping. Under the historical companion-v1 policy with single-glyph roles,
that branch consolidated two eligible fragments: 134 to 133 proposals and 57 to
56 eligible proposals, with the same eligible source IDs. The guarded singleton
extension below is separate; it does not globally permit singleton or positive-
region merges. Compare ordered memberships/annotation parents, not proposal IDs:
IDs bind the selected policy. The remaining low-fill component is not forced
through either repair.

On the saved Ubunchu target page, the repair adds one local original-boundary scan and no
additional full-page component passes (four remain per fresh proposal calculation).
Each preflight still performs full fresh validation; no cross-crop cache is used.
Local recognition and visual completeness assessment remain separate evaluation.

Each validated `process_images()` call starts a new pass. It invalidates previous
automatic proposals, automatic recognition results and the previous latest raw
page set before prediction. New proposals are published only when the entire pass
finishes successfully, so failed prediction or later-page proposal generation
cannot expose stale or partial units. A call without `propose_units=True` leaves
no automatic proposals available for recognition. Invalid arguments are rejected
before invalidating a valid pass. Previously returned records are not mutated;
callers can retain them as history. Within one successful pass, retained automatic
results still prevent duplicate primary attempts.

### Opt-in single leading-glyph reading proposals

`complementary_ownership=True` now selects the versioned
`enclosed_complementary_connectivity_v2` policy. Alongside the existing component
fallback, it can propose one noisy single-character reading as a **tentative**
annotation of a uniquely supported vertical main. Default per-box APIs and
`complementary_ownership=False` retain their previous behavior. No exact kana
transcription or reviewed relationship is manufactured.

This is not a reduction of the existing multi-component requirement. The separate
single-glyph path requires all of the following:

- A single letter/number OCR hypothesis (not punctuation or a modifier mark), a
  leading kanji hypothesis on the parent, and an already supported parent region.
- A whole original connected child component inside its polygon, not a clipped
  main stroke or a component also contained by another detection.
- Small, nontrivial two-axis ink structure; a simple filled dot, ring, line or
  connected hatching is insufficient. Hole filling is confined to a tiny analysis
  silhouette to prevent a ring's hole from supplying spurious shape complexity.
- A contained, interior main-glyph core spanning a vertical run; the child must
  align with the original leading glyph, with majority child ink in the parent's
  overlapping detection margin but separate actual ink and a clear original gap.
- The same enclosed **unmasked** white background, no original boundary veto,
  and exactly one remaining parent with no ambiguous parent/annotation chain.

Evidence is retained in each relevant relationship's `image_evidence` under
`single_glyph_position`, including original bounds, shape/run measures, leading
glyph alignment, white support and derived parent-region support. Unsupported
cases remain unresolved. All-nonmember crop isolation is rechecked unchanged;
ownership support never makes a contaminated crop eligible. No confidence score
or sample ID selects the role. Child wording remains raw and uncorroborated in
the unit's wording reasons, and main-only script routing remains independent.

```sh
.venv/bin/python -m tools.diagnostics --automatic-proposals \
  --complementary-ownership \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . --output-dir test-results/single-glyph-roles-only-v1
```

On identical saved inputs, only the two provisional `lmgJZ` roles change:
`38 -> 37` and `43 -> 42`. Raw evidence, the five existing known associations,
ordinary-small-dialogue controls, and all other memberships/order are retained.
With companion regrouping disabled, the first union is isolated; the second still
conflicts with nonmember `44` and remains blocked. Compared with the historical
complementary-only replay: 268 raw detections remain, units are `136 -> 134`,
tentative annotations `14 -> 16`, relationship conflicts `13 -> 11`, unsafe units
`24 -> 21`, and eligible units `56 -> 57`. The unsafe-unit
drop includes consolidation of former child-only units, not three new safe crops.

Enabled proposal IDs bind v2, so compare ordered main/annotation membership rather
than old IDs. v1 artifacts/candidates remain historical evidence; regenerate the
opt-in batch before recognition because fresh validation rejects the old policy.
Original dark/white labels are calculated at most once per fresh page calculation
when a noisy single-glyph candidate needs them (two additional component passes).
They are not cached across crops or passes. Completeness remains unverified, all
new roles are tentative, and no recognition accuracy improvement is established
by these model-free checks or provisional visual role observations.

### Guarded annotated-singleton companion extension

`regroup_companions=True` now binds `bounded_vertical_companions_v2`. Its separate
singleton branch handles reciprocal conflicts between a one-main group and an
existing multi-column group. It requires a shared **positive** main region and
image-supported, unique tentative reading relationships on both groups. The
original multi-main/negative-region branch keeps its previous limits.

Additional requirements are distinct, aligned original main-ink columns with
compatible scales and a complete-span bound; reciprocal conflicts involving a
main from each group; shared enclosed unmasked background for every member; and
a clear original seam between the visible ink envelopes, including assigned
readings. Annotation ink is not removed from the original corridor or the full
main-to-main endpoint-probed boundary scan. Thus oversized OCR polygons can
overlap while the actual neighboring columns remain visibly separated.

The final detection-union rectangle must still have **zero nonmember conflicts**.
A singleton with multiple structurally plausible destination groups remains
ambiguous rather than being assigned to the nearest group. An accepted singleton
extension is terminal: it cannot become a host for a singleton chain. Unknown
annotation region support is derived only after all acceptance/isolation checks
pass, and is recorded with original white-background evidence in
`companion_regroupings`; existing known region assignments are not overwritten.

```sh
.venv/bin/python -m tools.diagnostics --automatic-proposals \
  --complementary-ownership --regroup-companions \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . --output-dir test-results/singleton-companion-parent-v1
```

The identical-input regression changes only `[44]` plus `[42,41]` into ordered
mains `[44,42,41]`, retaining parents `43 -> 42`, `45 -> 44`, `46 -> 44`. The
adjacent `[40,39]` balloon and the Ubunchu five-column regroup remain separate and
unchanged. The new union is `(518,979,610,1083)` with no nonmember intersection.
Against `single-glyph-parent-v1`: raw detections/main roles/annotations remain
`268/252/16`, units are `133 -> 132`, unsafe units `21 -> 19`, relationship
conflicts remain 11, and eligible units `56 -> 57`. Two unsafe proposals become
one eligible union; the new eligible source set adds precisely six member IDs.

The trace retains the original parser's rejected nonadjacent-column merge and
the alternative adjacent seam evidence. Previously known labels and thresholds,
original pixels, raw
readings, annotation roles and all-neighbor isolation are preserved. Proposals
remain unverified full utterances, not accepted text or exact Japanese references.
Fresh preflight recomputes the selected policy and rejects old companion-v1
batches or changed evidence. On the saved lmgJZ page, original component context
is reused only within that fresh calculation: six full-page passes remain six,
with no cross-crop cache. Parent visual/model verification remains separate.

### External diagnostic boundary-ink observations

`--observe-boundary-ink` records **ambiguous ink crossing a crop rectangle** in
the existing diagnostic command. Speed lines and artwork can satisfy the same
pixel predicates while all main text is contained. This is not a validated
omitted-glyph detector and does not improve or assess punctuation completeness.
The observer lives only in `tools/diagnostics.py`, outside the production pipeline.

```sh
.venv/bin/python -m tools.diagnostics --automatic-proposals \
  --complementary-ownership --regroup-companions --observe-boundary-ink \
  --comparison test-results/crop-isolation-v2/*-comparison.json \
  --image-root . --output-dir test-results/boundary-ink-semantics-parent-v1
```

Each unit gains a separate `boundary_ink_observation`. The original pixel-selection
predicates are unchanged: small original dark components must overlap a main OCR
polygon and cross the final crop with at least two dark pixels outside. Original
background/glyph-flow checks and nonmember exclusion limit the observations but
do **not** distinguish glyphs from speed lines or artwork. Context remains bounded
to two glyph scales, capped at 96 pixels per side. No glyph mask is subtracted.

Statuses are `ambiguous_boundary_crossing_ink`, `no_boundary_crossing_observed`,
and `not_assessed` (with a reason). A negative observation does not mean the crop
is complete. Every record explicitly sets `is_completeness_assessment: false`,
`glyph_ownership: unresolved`, `artwork_contamination_possible: true`, and retains
`unverified_full_utterance`. Disconnected exclamation dots and other undetected
punctuation remain unobservable by this method. Geometry alone cannot establish
whether a crossing component belongs to text.

Records retain component bounds, pixel counts, and white-background support.
Positive observations add `*-boundary-ink-context-{original,annotated}.png` paths
to `image_paths`: original views use exact source pixels; magenta boxes in the
annotated view mark ambiguous components. The manifest summarizes
`boundary_ink_observation_counts`. Proposal IDs, memberships, crops, candidates,
eligibility and routing are unchanged. The observer is not run by default or by
production preflight; its two page-level component passes are diagnostic only.

With companion-v2 the seven-page observer has four ambiguous crossings,
76 without an observed crossing, and 52 unassessed (before the singleton extension:
4/75/54; historical pre-single-glyph v1: 4/74/57).
Parent visual inspection found the positives
mainly involved speed lines/artwork rather than missing main text. These counts
are not validated omitted-glyph findings, completeness improvement, warning
precision, or OCR accuracy. Full batches remain identical to the saved lifecycle
replay apart from the explicitly tested structural repairs (currently 132 units,
57 eligible, 19 unsafe). The observer itself does not change any batch.

The former `--warn-clipped-ink` spelling remains a CLI compatibility alias, but
new output uses only the corrected observation names above. Existing saved
`clipped-ink-parent-v1` artifacts are preserved with their historical schema; their
`potential_clipped_ink` / `no_supported_clipping_detected` labels must not be
interpreted as omitted-glyph or crop-completeness conclusions.

## Paddle detector/recognizer result contract

The installed PaddleX OCR pipeline keeps all `dt_polys`, but appends
`rec_texts`, `rec_scores` and `rec_polys` together only when recognition passes
`text_rec_score_thresh`. Iterating the filtered scores against `dt_polys[i]`
therefore drops detector-only boxes and can put later readings on earlier polygons.

`paddle_result.associate_paddle_result` now retains detector-list indices as box
IDs and preserves every representable native detector polygon. Recognition binds
only by exact `rec_polys` geometry, allowing cyclic vertex starts/reversed winding
without rounding, nearest-box or IoU guesses. Duplicate/foreign/malformed geometry,
invalid scores/text and mismatched recognition lengths remain blank and unresolved.
Detector-only confidence `0` is an explicitly marked unresolved sentinel, not a
measurement. Matched native scores/polygons are preserved; blank text cannot gain
acceptance from a high confidence score.

Legacy results without `rec_polys` use positional alignment only when detector,
text and score lengths agree fully and detector geometry is valid and unique.
Present-but-malformed `rec_polys` never triggers this fallback. Normal balanced
outputs keep the existing box/list return contract and review defaults; no
recognition threshold is changed to work around the association contract.

`OCR.prediction_contract_reports` retains selective raw fields, lengths, row
indices and failures for each page. Each box has `ocr_metadata.paddle_association`;
one detached complete page report is stored on the first detector box under
`paddle_page_contract`, so raw/orphan recognition evidence survives automatic
snapshots without duplicating the whole page on every box. Alignment status is
not transcription truth. Structurally unrepresentable detector geometry fails the
page with `PaddleResultContractError.report`, rather than repairing or discarding
it; the failed-pass report remains accessible and prior automatic state is invalidated.

Contract-fix test summary: `tests/ocr/test_paddle_result.py` has 18 engine-free
fixtures for filtered first/middle lines, reordered/cyclic/winding variants,
NumPy arrays, detector-only boxes, duplicate/foreign/malformed geometry, invalid
text/scores, high-score blanks, proven legacy alignment, retained raw evidence,
automatic snapshots and failed-pass lifecycle state. These validate association
and uncertainty handling, not recognition accuracy. No `text_rec_score_thresh=0`
workaround is used. Focused verification (no Paddle/model initialization):

```sh
.venv/bin/python -m unittest tests.ocr.test_paddle_result tests.tools.test_detector_scale_experiment -q
```

Selective before-edit source snapshot:
`test-results/ocr-contract-before-session-ec273496-v1/detection.py.before` with
`snapshot.json`. The earlier rollback archive and all prior work are preserved.

## Bounded cached detector scale experiment

`tools/detector_scale_experiment.py` is a standalone detector-only comparison for
the seven existing evaluation images. It does not run grouping, recognition,
translation or erasure and does not replace historical comparisons. Help and the
default planning mode do not import or initialize Paddle. No research reports
are embedded in the tool.

First create a model-free plan in a new directory:

```sh
.venv/bin/python -m tools.detector_scale_experiment \
  --model-dir /home/Danuu/.paddlex/official_models/PP-OCRv5_server_det \
  --variant 1280:max --variant 1600:max \
  --output-dir test-results/detector-scale-preflight-parent-v3
```

The parent can explicitly execute the same bounded experiment in another new
directory (the agent has not exercised the live detector):

```sh
.venv/bin/python -m tools.detector_scale_experiment \
  --run-local-detector \
  --model-dir /home/Danuu/.paddlex/official_models/PP-OCRv5_server_det \
  --variant 1280:max --variant 1600:max \
  --output-dir test-results/detector-scale-parent-v2
```

The cached `inference.yml`, `inference.json` and `inference.pdiparams` must already
exist as regular static-model files at that exact pinned directory. Preflight
validates the export/configuration and hashes those files and all seven images.
The installed `TextDetPredictor` registers `DetLabelEncode` and `KeepKeys` as
builders returning `(None, None)`; `_build` excludes those operations from inference.
The validator admits only the verified inert export forms `DetLabelEncode: null`
and `KeepKeys: {keep_keys: [image, shape, polys, ignore_tags]}`, at most once each.
Their data is retained in `model.ignored_training_transforms`; other fields,
unknown transforms and arbitrary scale expressions remain rejected. The cached
BGR decode, standard normalization, `resize_long: 960`, and DB unclip `1.5` are
supported without rewriting the cache or forcing a different postprocessor value.
No model-name/path fallback or dependency/model download is provided. Live mode
uses one installed `paddleocr.TextDetection` instance on CPU, one inference at a
time, batch size one and CPU threads one. Network, model-registry lookup and child-
process guards are active around initialization/inference/cleanup. Source images
are passed as original file paths, never externally resized, masked or enhanced.

Installed `paddle/base/core.py` queries Linux libc using a fixed `ldd --version`
pipeline and AVX using a fixed `/proc/cpuinfo` pipeline during import. The local
guard emulates only those exact query/call shapes **without starting a process**:
GNU libc comes from `os.confstr("CS_GNU_LIBC_VERSION")` and AVX-query presence from
reading `/proc/cpuinfo`. Requested values are recorded under
`read_only_capability_probes`. Arbitrary `Popen`/shell/child-process calls remain
blocked, as do network access and model-registry fallback. Probe failures do not
manufacture a libc version or positive AVX result. No executable discovery,
compiler, downloader or arbitrary shell command is admitted by this compatibility path.

The installed `cuda/pathfinder/_dynamic_libs/load_dl_linux.py` is imported through
Torch even on this CPU path. Its `find_library("dl")` lookup otherwise invokes
stdlib linker-cache discovery via `/sbin/ldconfig -p`. Only the `"dl"` lookup is
intercepted: `ctypes.CDLL("libdl.so.2")` must first load successfully and expose
`dlinfo` and `dlerror`, exactly as required by that installed module. The verified
soname is recorded as `read_only_capability_probes.libdl_soname`. No linker-cache
process is started; other library names retain their original lookup subject to
the existing subprocess/network deny rules. Missing library/symbols fail rather
than enabling a broader lookup or fallback. The Linux system-library test performs
this real, non-model load; detector runtime tests remain fake.

Preserve the failed `detector-scale-parent-v1` manifest. Retry only in the fresh
`detector-scale-parent-v2` directory above. This is a narrow correction for the
reported import chain, not permission for additional discovery commands; any
further substantive blocker should be assessed from its traceback before another run.

Installed source verification: `paddleocr/_models/_text_detection.py` exposes
`limit_side_len`/`limit_type`; `paddlex/inference/models/text_detection/predictor.py`
accepts those per prediction and uses fixed DB postprocessing. Resize semantics
are in `text_detection/processors.py`: `max` caps the long side, `min` raises the
short side, with SDK stride rounding and a fixed maximum-side limit. The script
supports lengths 32..2048 and this max/min subset, at most two distinct overrides.
It records initialized effective resize/postprocess settings, not just YAML hints
(PP-OCRv5 wrapper defaults can differ from declared export resize values).
For PP-OCRv5, `build_resize` uses an explicit constructor override or
`resize_long` (default 960), with default type `max`; the experiment still reads
the initialized processor as the authoritative effective baseline. Per-call
`limit_side_len`/`limit_type` then select only the requested scale variants.

The fresh baseline is the initialized cached detector's settings. Thresholds,
unclip ratio, dilation, other DB settings and runtime remain fixed across variants.
Historical saved OCR data lacks runtime settings: `historical_baseline_reproduced`
is explicitly false. A fresh detector baseline is not a claim to reproduce the
historical 268-box OCR output. Native detector indices are local to each page/
condition and are not cross-scale or integrated-OCR box identities.

Outputs are an exclusive `detector-scale-plan.json`, native polygon/score sidecars
per page/condition, and `detector-scale-results.json` with model identity/hashes,
software/runtime versions, effective parameters, timestamps and wall timing.
Native input dimensions/type/pixel digest are recorded and checked across scales;
original file hashes are rechecked. Baseline plus two variants is at most **21**
page-conditions. Errors stop the run with no retries, preserving completed outputs
and available failed native evidence. Missing cache/configuration or unsupported
installed API fails closed; it does not install, fetch or switch models. Output
must be a fresh directory under this checkout's `test-results`.

Only fake runtimes are exercised by the offline experiment tests. Actual cached
model usability, effective baseline values, detection deltas and runtime cost
remain for parent execution/assessment. Detection-count changes are not measured
OCR accuracy or annotated coverage. Six of the seven pages remain one CC BY-NC
evaluation series.
