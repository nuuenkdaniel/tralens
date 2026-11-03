# Learned mask / background restoration feasibility

This is a **local evaluation, not a production fallback**. The working automatic
renderer remains flat fill followed by classical restoration. No translation,
training, cloud inference, image upload, dependency-file edit or push is required.

## Approved acquisition and provenance

The user explicitly approved both checkpoints, a separate CPU runtime, and bounded
local comparisons after reviewing the licensing/provenance caveats below.

| Artifact | Pinned origin | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `comictextdetector.pt.onnx` | `zyddnys/manga-image-translator` release `beta-0.2.1`, asset `55024546` | 94,669,756 | `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f` |
| `lama_fp32.onnx` | `Carve/LaMa-ONNX@c3c0c9e468934d62e79c329e35d82dd09ff8c444` | 208,044,816 | `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` |

CTD's hash records the downloaded bytes: the release API provided **no publisher
digest**. It is not independent authentication. LaMa's bytes matched the pinned
Hugging Face LFS SHA256.

Source pins:

- CTD: `dmMaze/comic-text-detector@440b978563c71b758e31aaa315d100faba1efa2f`.
  Refinement, image processing and text-block source files are hash-checked before
  loading. The source license is GPLv3; separate checkpoint terms are unresolved.
- Original LaMa: `advimman/lama@786f5936b27fb3dacd2b1ad799e4de968ea697e7`.
  Source license Apache-2.0.
- Conversion source:
  `Carve-Photos/lama@f5fb39a18022c34a71bf9a47a6ec393c804b49ca`.
  Its notebook uses an unpinned `smartywu/big-lama` mirror checkpoint. The model
  card declares Apache-2.0, but the exact original checkpoint chain is not
  independently digest-verified. The converter replaces Fourier operations; its
  claim of equivalence has not been locally validated against original PyTorch.

These records **do not grant distribution or commercial clearance**. The models,
third-party source and wheels are outside the repository under
`/tmp/opencode/tralens-learned-eval-v1/`. No pickle checkpoint is loaded, and the
conversion notebook is inspected, not executed. ONNX still requires trusting a
native model parser; process separation is not a security or licensing exemption.

The separate Python 3.10 environment uses ONNX Runtime 1.23.2 CPU, OpenCV headless
4.10.0.84, NumPy 2.2.6, Pillow 10.4.0 and Shapely 2.1.2. Its binary dependencies
were pinned and checked against PyPI wheel hashes before offline installation.
The wheel download total was below the approved 150 MB limit. Existing `.venv`,
`.venv-paddle6`, `pyproject.toml` and `uv.lock` were not changed.

## Frozen cases

`tools/freeze_learned_evaluation.py` generated
`test-results/learned-evaluation-frozen-v1/manifest.json` **before learned
inference**. Its SHA256 is
`29f1572f856551979db1613b23aa186a75121e6214fee2844de6d2442ddd9706`.

- 38 real source-unit cases: first four and last two saved proposal memberships
  from each of six Ubunchu! pages, plus the two unresolved saved translations.
- 16 known-overlay synthetic cases: white, off-white/vertical, touching line,
  punctuation, neighboring text, gradient, periodic dots and rotated line crossing;
  separate development/held-out sizes, pattern phase and rotation.
- Chapters 1–2 are development; chapter 3 is held out from this learned experiment.
  Chapter 3 was used in earlier geometry-only tests, so it is not a never-seen
  product test set. One title and one synthetic Latin font are not broad coverage.
- Ubunchu! by Hiroshi Seo (Seotch), ASCII MEDIA WORKS Inc., CC BY-NC 3.0: local
  noncommercial evaluation only. Original bytes, source ledger and chapter
  licensing evidence remain in `images/ubunchu-ja-test/`.

Fixture audit after inference found that the two cases named `line_crossing`
actually have zero dark-reference pixels beneath their fixed erase masks. They
test rotated text **near** lines, not interrupted-line reconstruction. The
separate `outline_touch` cases do cross a known art line. The frozen cases were
not silently changed; this is a coverage limitation, not evidence of success on
general crossing line art.

Synthetic masks are **evaluation ground truth**, not required user masks or a
production workflow. Real pages have neither clean-background nor pixel-mask
ground truth. Their OCR polygons are hypotheses, not semantic certification.

## Mask experiment

A standalone real CTD mask was obtained first, in
`test-results/ctd-standalone-v1/`, before creating the experiment runner.

The pinned ONNX path consumes RGB NCHW float32 in 0..1, aspect-preserving linear
resize into 1024×1024 with black **bottom/right** padding. Output names/shapes were
observed as `blk [1,64512,7]`, `det [1,2,1024,1024]`, and
`seg [1,1,1024,1024]`. The segmentation output is selected by shape, not position.
Match upstream by quantizing to uint8, removing padding and linearly mapping back
to native resolution. The raw native mask is **grayscale**, not binary.

The frozen raw threshold is 0.3. Save the grayscale separately. Compare current
foreground/erase masks, thresholded CTD and pinned upstream INPAINT refinement.
Refinement receives the **same frozen source-unit union box**, not CTD-regrouped
utterances, and does not recover undetected blocks. This is a source-bound
refinement variant, not a claim to benchmark the complete upstream detector.

No mask is accepted because it is nonempty. Report both ink precision/recall and
dilated-support precision/recall on synthetic truth, misses, and nonmember polygon
overlap. Refinement adds dilation; comparing it only against undilated ink is not
a fair erase-support precision measure, so both measurements are retained.
Tiny Lanczos antialias/ringing pixels count in the synthetic reference and make
exact-recall requirements stricter than thumbnail inspection.

Zero overlap with another OCR polygon does **not** establish that neighboring
unannotated text or art is safe. In particular, the standalone unresolved-unit
crop visibly includes some nearby unassigned lettering in its refined mask.

### Mask findings

The existing conservative methods produced plans for **22/38 real units** (14/26
development, 8/12 originally held-out), and abstained on all 16 synthetic
challenges. These are mask/restoration plans, **not eligible translations or
completed English replacements**. Synthetic failures include faint disconnected
antialias pixels, touching lines, nonwhite support and texture; abstention is
retained, not scored as a precise successful mask.

CTD gave promising visual masks on several source units rejected by components,
including much of both unresolved saved-page units, but it is not ready for
automatic erase permission:

| Frozen synthetic split | Raw ink precision / recall | Refined ink precision / recall |
| --- | --- | --- |
| Development (8 cases) | 0.893 / 0.629 | 0.838 / 0.701 |
| Originally held-out (8 cases) | 0.871 / 0.723 | 0.845 / 0.786 |

These are micro-aggregates against every nonzero overlay-alpha pixel, not manga
segmentation accuracy claims. Refined dilation trades precision for coverage;
separate dilated-support metrics are in the machine report. Repeated dots expose
missed text, including a large miss in one development specimen. Both CTD variants
overlap nonmember OCR polygons in **4/38 real cases**. Additional unannotated
neighbor/art errors are not captured by that count. Native visual inspection
remains provisional, not pixel-level ground truth.

Artifacts: `test-results/learned-mask-comparison-v1/report.json`, per-case native
contact sheets, and the combined local review generated below.

## Fixed-mask restoration experiment

Every candidate in a comparison receives identical source pixels, context and
binary mask. Compare median-ring fill (diagnostic, not production flat permission),
Telea, Navier–Stokes and the acquired LaMa conversion.

- Synthetic support: known overlay alpha >0, dilated once by 3×3.
- Real support: only a successful current automatic plan's erase mask. Unaccepted
  CTD masks are **not** promoted just to generate learned cleanplates.
- Two frozen context margins, 16px and 64px; native coordinates, **no resizing**.
  Skip crops exceeding the conversion's fixed 512×512 input or lacking clearance.
- Image and mask padded symmetrically together on bottom/right. LaMa consumes
  float32 RGB NCHW 0..1 and N1HW mask where **1 means remove**. Its output is
  0..255, clipped and truncated to uint8 as in the converter's example.
- Record the native crop and exact padding. Unpad output and copy candidate pixels
  **only through the fixed mask**; no feathering. Audit exact outside-mask equality.
  Raw model output differences are separately diagnostic: this export already
  blends source pixels and may introduce unmasked rounding differences.
- Save cleanplates before lettering. There is no translated text hiding defects.

**Preprocessing correction:** restoration v1 hit the command timeout; v2 repeated
the same zero-mask-padding protocol. Both are invalid quality evidence: symmetric
image padding can mirror hidden target letters into artificial context while zero
mask padding presents those mirrored copies as known image pixels. The corrected
v3 experiment reflects the image and removal mask together (consistent with the
converter's image-and-mask padding helper), so reflected targets remain hidden.
The original native erase masks, cases, threshold and two context extents remain
frozen. Synthetic reference pixels never enter model input. Earlier evidence is
retained rather than overwritten, but must not be counted as valid LaMa results.
The correction was motivated after inspecting both splits. The original
`held_out` labels remain for traceability, but **v3 is a corrected feasibility
rerun, not untouched confirmatory validation**. New confirmatory pages are needed
before adoption. The correction plan is saved before v3 inference as
`test-results/learned-restoration-padding-correction-v1.json`.

Synthetic results have masked MAE/RMSE/PSNR and dark-reference continuity signals.
Real results have no invented pixel-error scores. Comparing LaMa against the
lowest-error classical method is an **oracle diagnostic**, not a deployed selector.
Fixed-mask restoration is not end-to-end automatic replacement coverage.

Per-stage process peak RSS and individual wall times are retained. Feasibility
timings under mixed load, including concurrent test runs, are not controlled speed
claims. No GPU performance claim follows from these CPU runs.

### Corrected restoration findings and decision

v3 completed **76 context comparisons**: 22 real units and 16 synthetic units,
each at two context extents. The remaining 16 real units had no accepted current
mask and were explicitly skipped. All 304 composited candidate images passed
exact outside-mask equality. This is locality evidence, **not mask correctness**.

At the 64px context extent, known-reference masked MAE (8-bit channel levels):

| Synthetic specimen | Best classical inpainter | LaMa |
| --- | ---: | ---: |
| Development periodic dots | 39.50 | 19.81 |
| Originally held-out periodic dots | 15.93 | 7.26 |
| Development thin line touching text | 6.90 | 1.17 |
| Originally held-out thin line touching text | 14.84 | 1.31 |

LaMa retained all 70/77 dark reference pixels under the two line masks at this
extent; classical methods blurred/missed many of them. That is one very simple
straight-line structure, not broad manga line-art validation. The dot specimens
still have substantial defects: only about 50%/64% of originally dark masked dot
pixels remained below the dark threshold. Lower average error is not an invisible
edit, nor proof of correct dot phase, shape or density.

Across both context sizes LaMa had lower masked MAE than the better classical
inpainter in **8/32 synthetic comparisons** (the dots and touching-line cases).
Classical restoration was better in the other 24, including plain backgrounds
and gradients. This supports retaining cheap routes, not replacing everything
with LaMa. The “best classical” choice is an oracle comparison, not a model-free
automatic quality selector.

The native-scale [local review](../test-results/learned-evaluation-review-v1/index.html)
contains source/mask overlays, cleanplate comparisons, skips and per-case metrics.
The exact executed runner sources are archived beside their reports and match the
recorded implementation hashes. Acquisition URLs, wheel/model digests and byte
counts are retained in `test-results/learned-evaluation-acquisition-v1/`.

**Decision: no production adoption yet.** CTD still misses text and crosses
neighbor evidence; difficult real backgrounds without safe masks were not
restored. The saved translation page remains 6/8 eligible replacements. The next
bounded step is source-unit mask association/refinement with explicit neighboring
text/art protection, calibrated on development cases, then a fresh untouched
confirmatory set. Do not expand model families or loosen erasure just to produce
more cleanplates. Checkpoint distribution/commercial rights remain unresolved.

### Small process-cold / warm timing sample

After the quality runs, models were timed sequentially in fresh CPU processes
with four threads, without concurrent agent-launched tests/inference. OS file
cache was warm; these are **not disk-cold or broad throughput benchmarks**.

| Model/input | Load | First forward | Three warm forwards | Process peak RSS |
| --- | ---: | ---: | --- | ---: |
| CTD, one 1246×1635 page letterboxed to 1024² | 0.045s | 1.037s | 0.894 / 0.904 / 0.872s | 1.24 GiB |
| LaMa, one native dot crop padded to 512² | 2.108s | 1.379s | 1.373 / 1.366 / 1.358s | 1.14 GiB |

These timings exclude preprocessing and final composition, and are not GPU or
end-to-end page speeds. Full samples/input hashes are in the local review's
`process-timings.json`.

## Running the local experiment

The runner never acquires models or installs packages. Existing source paths,
hashes, fixture artifacts and frozen manifest hash are checked before inference.
Output directories must be new, to avoid mixing runs.

```sh
.venv/bin/python tools/freeze_learned_evaluation.py \
  --output test-results/learned-evaluation-frozen-v1 \
  --fonts test-results/render-parent-input-v1/fonts.json

/tmp/opencode/tralens-learned-eval-v1/runtime/bin/python \
  tools/learned_restoration_experiment.py masks \
  --frozen test-results/learned-evaluation-frozen-v1 \
  --runtime /tmp/opencode/tralens-learned-eval-v1 \
  --output test-results/learned-mask-comparison-v1

/tmp/opencode/tralens-learned-eval-v1/runtime/bin/python \
  tools/learned_restoration_experiment.py restoration \
  --frozen test-results/learned-evaluation-frozen-v1 \
  --runtime /tmp/opencode/tralens-learned-eval-v1 \
  --masks test-results/learned-mask-comparison-v1 \
  --output test-results/learned-restoration-fixed-v3

.venv/bin/python tools/summarize_learned_evaluation.py \
  --masks test-results/learned-mask-comparison-v1 \
  --restoration test-results/learned-restoration-fixed-v3 \
  --output test-results/learned-evaluation-review-v1
```

Use new versioned paths when repeating a run; do not regenerate a held-out fixture
or tune thresholds in place. The linked image artifacts and provenance reports
remain ignored local evidence, not redistribution of manga or checkpoints.
