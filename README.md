# Tralens

## Minimal development frontend

No frontend framework, npm, build step or new dependencies. Using the existing
validated runtime and this checkout's saved licensed font manifest:

```sh
.venv-paddle6/bin/python tools/dev_frontend.py \
  --fonts test-results/render-parent-input-v1/fonts.json
```

Open **http://127.0.0.1:7860**, choose a PNG/JPEG, select how far to run, and click
**Run**. See raw OCR boxes, grouped units, source/translated text, actual stage
status, then the cleanplate/text/composite layers and masks. Click a unit to
highlight its source boxes. Blocked, budget-limited and failed units remain visible.

The image-first stage strip lets you switch between **Original → Detected text →
Translation → Cleaned page → Final**. Each stage has a thumbnail and a large
preview, with **Compare with original** and **Actual size** controls. Clicking a
stage pauses automatic following so it stays selected while processing continues.
Translation is a text-only phase: the source page highlights available previews,
with the English text alongside it; only Final shows actual replaced lettering.
Stages without output explain why instead of showing a fake or unchanged final.
Advanced settings, masks and raw artifacts are tucked into expandable sections.

The run card has just two choices:

- **Output**: Translated page (default), Text only, or Detect text.
- **Replacement**: **Replace all text** (default) or **Protect artwork**.

**Replace all text** is the pipeline's own default profile: broad replacement,
aggressive local removal, and *always show available translations*. Where clean
removal fails, the English is drawn on an opaque patch or a nearby caption, so
artwork behind text may be smeared or covered—a translated page with some
background loss is preferred over untranslated text. **Protect artwork** uses
native cleanup + OCR checks and preserves uncertain artwork; translations it
cannot place safely stay in the side list only.

Every run uses Coverage first translation and Best effort rendering. Advanced
holds the call budget (default 64), LaMa background reconstruction (when started
with `--reconstruction-config`) and paired OCR rereads (when configured). Retries
send the full current settings, so experimental options from older saved runs
(enclosure retry, crop localization, fragment previews) are not carried over;
those remain available from the CLI. See [translation coverage](docs/translation-coverage.md)
and [always show available translations](docs/visible-translations-experiment.md).

For the optional paired OCR rereads, start the viewer with existing local assets:

```sh
.venv-paddle6/bin/python tools/dev_frontend.py \
  --fonts PATH/fonts.json --manga-reread-model PATH/existing-manga-ocr-snapshot
```

This enables the **Paired OCR rereads** checkbox under Advanced, not automatic inference.
It uses the installed `qwen3.8:latest` vision tag (override with `--reread-model`)
on the configured private Ollama endpoint, plus the supplied local Manga OCR
snapshot. Only explicit rendering runs send crop presentations; nothing is
downloaded. The route requires Coverage first translation and remains experimental:
reader agreement does not prove correct source text.

Saved outputs that left translations off the page (Protect artwork, or older
broad-replacement runs without the visible-translation fallback) show a notice
and **Retry and replace all text**. This makes a fresh job, retains earlier
results and labels the policy actually used by each output.
Reread-selected sources are shown alongside their retained original OCR. A link
with `?job=JOB_ID` opens that exact saved job rather than another recent output.
See [quality iteration evidence and limitations](docs/ocr-rendering-quality-iterations.md).

After any completed, partial or failed run, **Run pipeline** becomes **Retry
pipeline**. It reuses the saved upload with the current Run settings—no file
selection is needed, including after refresh. A retry reruns the pipeline in a
fresh directory with a fresh call budget; it does not resume or overwrite the
previous run. Choosing a different image switches the button back to Run pipeline.

Under **Advanced → Translation model**, choose the configured local model, GPT-6 Luna,
GPT-6.1 Terra, or GPT-6.2 Sol. The three frontier options use the signed-in Codex CLI. Visual modes send the current page and the same-manga pages you select to
OpenAI. Visual full-page runs use one budgeted call to find missed Japanese lettering
when at least two calls are available. New regions enter the regular grouping,
translation and replacement stages. The default frontier mode sends numbered crops with the page and rechecks
skipped regions while the call budget allows. Choose **OCR text per region** for
the text-only comparison. The CLI accepts `--model gpt-6-luna`, `--model gpt-6.1-terra`, or `--model gpt-6.2-sol` with
optional repeated `--related-page PATH` arguments. The local model remains the
viewer default. See the [four-page Luna comparison](docs/luna-translation-trial.md).

The default OCR export root is `~/.cache/tralens/paddle-v6-models`; override with
`--model-root PATH` if needed. Local translation uses the existing Ollama service
and `translategemma:27b`; no models are downloaded. The viewer respects
`OLLAMA_HOST` (including a bare local IP, with port 11434); `--endpoint http://IP:PORT`
overrides it. Otherwise it uses `http://127.0.0.1:11434`. The selected endpoint is
printed at startup and shown in Run settings. A working OCR stage does not prove
Ollama is reachable: connection failures leave no translation to render.
Without `--fonts`, OCR and translation still work, but rendering is disabled.
For another checkout, supply
your own explicit licensed font manifest using the schema below.

The translation call budget defaults to **64**. Existing rendering/eligibility limits still apply. No
inference happens on server startup, page load or refresh—only on **Run** or **Retry**.

Status comes from real saved checkpoints, polled once a second. OCR/grouping
details appear when that phase publishes evidence, translations appear as units
finish, and rendering layers appear when published. There is no fake percentage
progress, token streaming, mask editing or manual source correction in this viewer.

Jobs keep their uploads and pipeline artifacts under `test-results/dev-frontend/`.
One job runs at a time, with eight new runs per server session. Refresh preserves
the run list; restarting restores up to 24 recent saved runs for inspection and
retry, including older pipeline runs without viewer metadata. Restore never starts
inference, resumes a running checkpoint, or rewrites old evidence. Interrupted
runs are marked as such and can be retried explicitly. Ctrl+C stops the server;
it does not guarantee a running native OCR/model operation finished or its last
checkpoint is final.

**Local development only:** binds to loopback, rejects cross-origin requests and
path traversal, and accepts only bounded PNG/JPEG uploads (32 MiB, 4M pixels,
4096px side; opaque source required). Do not expose this Python development server
as a public upload service. Native engine setup/model failures are shown in the UI.

Verification: offline viewer tests cover HTTP upload through the real pipeline
with injected engines/providers, checkpoint visibility, mode dispatch, errors,
source retention, retries, saved-run recovery and request/path guards. A headless
Firefox smoke test also exercised browser file upload, five stage views,
comparison, zoom, highlighting, page refresh and retry without re-uploading;
it used the same synthetic pipeline fixtures, not live inference.

## One-command image pipeline

The stage implementations stay separate; **`src/main.py` connects them**:

```text
src/
  main.py                    CLI and job coordination only
  tralens/
    ocr/                     detection, grouping, recognition and source evidence
    translation/             bound translation previews and provider runtime
    rendering/               cleanup, typography, composition and pixel checks
tools/                       standalone diagnostics and saved-artifact commands
```

The coordinator reuses the existing image-to-OCR/translation preview workflow,
then passes its exact saved preview to the automatic renderer. No OCR/grouping,
translation eligibility, font selection or erasure algorithm is implemented in
`main.py`. Neither the OCR nor translation package imports the renderer.

Use the validated `.venv-paddle6` runtime, existing local OCR exports, an installed
translation model, and an explicit licensed font manifest (schema below). Choose a
fresh output directory whose parent already exists:

```sh
.venv-paddle6/bin/python src/main.py \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --output test-results/pipeline-page-v1 \
  --device cpu --model-root "$HOME/.paddlex/official_models" \
  --run-local-ocr --run-local-model \
  --render --fonts test-results/render-parent-input-v1/fonts.json \
  --max-attempts 20
```

With multiple manifest faces, also supply one global `--font-id`. The default
translator remains `translategemma:27b` on `http://127.0.0.1:11434`; existing
`--model`, `--profile` and explicit private `--endpoint` overrides still apply.
No model/font download, provider/model-name fallback, mask drawing or per-balloon
confirmation occurs. Background-method fallbacks are automatic as described below.

Outputs are:

* `pipeline.json`: job status, original source/font hashes, bound preview/render
  manifest hashes, coverage, upstream blockers, translation budget skips and
  rendering skip reasons.
* `preview/`: unchanged OCR/translation evidence, `preview.json`, journal and any
  bounded provider/review sidecars.
* `rendering/`: `composite.png`, original/clean/text layers, masks, `manifest.json`
  and `review.html`, using the existing automatic rendering policy.

**Replacement now defaults to `--render-policy best-effort`.** After the existing
methods fail, it tries risk-labelled native/footprint cleanup and relaxed exact-text
lettering instead of immediately leaving a translated unit unchanged. Edits may
damage artwork or use smaller type; originals and all layers remain available.
Use `--render-policy conservative` for the former skip-on-uncertainty policy and
strict font minimum. See [coverage-first behavior and limits](docs/coverage-first-rendering.md).
The [rendering readiness review](docs/rendering-production-review.md) records
additional native/color-page tests, retained fixes, rejected experiments and
remaining production blockers; best-effort placement is not an art-quality guarantee.

Rendering uses `--background-method auto`: flat fill first, then a
separately validated classical-inpainting fallback for remaining strict translated
candidates. No per-region manual selection is needed. `--background-method flat`
preserves white/off-white-only behavior in conservative mode without the learned-model opt-in below. This option is also available on the
standalone `auto-compose` command. Missing source-bound full translations remain
explicit skips. Uncertain masks/backgrounds now receive a marked best-effort
attempt by default; they still skip under `--render-policy conservative`.

Image format/dimensions, fonts/licenses, shaping availability and rendering options
are checked **before OCR**. The pipeline uses the renderer's narrower limit of
4M pixels/4096px per side. Original image/font bindings are checked again before
rendering. OCR is closed before translation; rendering makes no model calls.

Budgets remain explicit: CLI translation defaults to **2 calls**, allowed 0–64;
`--max-attempts 20` above requests a broader bounded pass, not guaranteed full-page
coverage. Rendering attempts at most 32 units, possibly fewer under raster-work
budgets. `--max-units N` is an optional rendering evaluation cap only. Rendering
size bounds default to 16–28px (`--min-size`/`--max-size`). Guards are not relaxed
to make a page fit.

Exit **0** means every saved unit was composed and the preview pass completed;
**1** means partial coverage or a failed job; **130** means interrupted; **2** means
CLI/rendering preflight rejection. These are processing outcomes, not proof of
language quality or complete detection. Inspect `pipeline.json` for the exact
status/reasons. Failed/interrupted previews never render. Successful strict results
from a pass with individual translation failures may still render as partial output.
With no renderable bound translations (including empty OCR), rendering is explicitly skipped
and **no composite is produced**. A report can also contain an unchanged composite
if every attempted rendering was safely rejected. All outputs remain unapproved
candidates, and skipped units keep original lettering.

Existing output paths and symlink inputs/output ancestors are rejected. Jobs never
resume or overwrite earlier evidence. Failures after job reservation retain their
checkpoints; abrupt termination/storage failure can leave the last `running`
checkpoint or partial artifacts. Use a new output name to retry.

Without `--render`, the original OCR-only/translation-preview CLI and output layout
remain unchanged. The standalone saved-preview rendering tools below also remain
available. Offline integration verification uses synthetic pages/fonts and injected
OCR/providers only:

```sh
.venv/bin/python -m unittest tests.test_pipeline -v
```

## Automatic saved-preview rendering: input → output

`auto-compose` is a separate **machine-selection policy**, not automatic operator
confirmation of the assisted/manual tools. It takes the original raster, saved
translation preview and explicit local licensed font manifest and writes a fresh
partial candidate output directly. No selection JSON, proposal IDs, mask drawing,
operator identity or per-balloon review is required or consumed.

```sh
.venv/bin/python tools/render_preview.py auto-compose \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --output test-results/automatic-tight-render-parent-v1
```

The sole explicitly supplied manifest face is used if the manifest contains one
font. With several approved faces, supply one global `--font-id`; there is no
implicit system-font lookup or learned font matching. Global size bounds default
to `--min-size 16 --max-size 28` (allowed 8–144). Optional `--max-units N` is an
explicit evaluation cap, not per-unit selection; without it all current strict
translated candidates are machine-selected in saved order and attempted within
declared resource budgets. Every saved unit remains in the report, including
blocked/untranslated/fragment/alternate-only entries and resource skips. No OCR,
translation, provider, model initialization or network request occurs.

### Automatic tight-mask policy

The first stage, `automatic_original_tight_flat_v3`, does not need a complete white-balloon partition.
Its erasure/background checks are unchanged from v1; v2 added a bounded irregular
layout fallback, and v3 adds comfortable typesetting. Neither grants more erasure
permission. The default `automatic_flat_then_classical_v1` coordinator tries the
separate restoration policy below only when this flat-stage mask cannot be derived:

1. Seed original **whole 8-connected dark components** from member polygons using
   RGB-to-gray `<180`. Every component must remain within the original member
   geometry plus 2px. Frame contacts, excessive extents, spills and nonmember
   polygon intersections reject; OCR rectangles are never paint masks.
2. Measure original RGB background from a local collar 2–4px outside the dark
   silhouette. Require >=32 samples, light neutral white/off-white (minimum
   channel 180, maximum channel spread 16), and maximum channel deviation **2**.
   Colored/gradient/texture/neighbor contamination never receives a looser flatness
   threshold; any restoration must independently establish its own source mask.
3. Re-label full original RGB contrast `>2` relative to that measured background,
   retaining whole components seeded by the dark ink. This audits antialias
   coverage instead of assuming a one-pixel dilation of the 180-threshold mask
   is sufficient. Contrast-component spill and unexplained/isolated gray marks
   in the source neighborhood reject. Colored foreground is unsupported.
4. Dilate that complete foreground by **1px** for erasure. Its independent **2px
   collar** must be original flat background and it must not intersect any
   nonmember polygon or reach the source frame. Fill only this mask with the
   measured RGB color. Outside it the cleanplate remains exact canonical source.
5. Separately derive a bounded source-local layout rectangle using original flat
   whitespace and 2px clearance from all unrelated original contrast and all
   nonmember geometry. Expansion is limited to the source bbox plus <=32px
   horizontally/24px vertically. The previous rectangular result remains the
   preferred path, with identical pixels when it succeeds.
6. If **only the seed's rectangular box** is obstructed, but the actual complete
   foreground plus 2px collar is unobstructed and wholly inside that same bounded
   window, use the single **4-connected safe whitespace component** containing
   every seed pixel. Unrelated contrast/nonmember clearance remains excluded,
   including holes. No hull, hole filling, trimmed seed, diagonal bridge or
   disconnected pocket is permitted. The existing renderer checks every final
   alpha pixel against the resulting irregular mask; it may still overflow.
   Shared white regions can have disjoint local layouts; this is not semantic
   ownership, a Voronoi partition or a claimed speech boundary. Actual obstructed
   or disconnected complete-ink seeds, colliding layouts/erasures or RAQM overflow
   skip the unit. Both colliding units reject symmetrically.

All effects/full-resolution alpha checks use the existing measured RAQM renderer.
No text rewrite, clipping, squashing or size below the supplied bound is allowed.
Cleanup is committed only after successful typography and per-unit pixel checks;
skips leave source ink untouched. The manual/assisted flat-interior cleanup guard
is unchanged. The flat stage does not use inpainting; its rejection is not erase
permission for the separate fallback. Learned segmentation is not used by default;
the explicit native-mask option and historical CTD/LaMa experiment are described below.

Original source masks are raster-counted once per job; exact per-pixel counts
preserve coincident/contained nonmember annotations when subtracting a member's
contribution. Whole-component/color/collar checks still run per unit. No global
white ownership or shared-outline condition is required for tight erasure.

`mask_evidence.layout_derivation` records rectangle versus connected-safe-mask
placement. The fallback retains the original rectangle rejection, connectivity,
required seed count, zero discarded seed pixels and local component/pixel counts.
The additional labeling is bounded to the existing source-local window and 8192
components. Text placement changes do not change the source-derived erasure mask.

The irregular-layout regression improved **5/8 to 6/8 strict translated candidates** on
the same `ubunchu01_04.png` preview. The newly fitted long balloon previously had
two obstructed *empty box-corner* pixels and zero obstructed actual seed pixels;
its complete original lettering is removed. V2 fitted it at 27px within the existing
16–28px bounds, preserving all five previous erasure/layout/text rasters. V3 keeps
all six erasures and the cleanplate identical while improving the text layers.
The remaining `component_spill` and
`unassigned_contrast` units stay skipped; fragments/blocked units are not promoted.
This is still partial output (6 of 29 saved units), not full-page coverage.
Fresh typeset output: `test-results/automatic-typography-v3/review.html` and
`composite.png`; the earlier tight-mask/irregular-layout baselines are retained.
Runtime and independent
pixel comparisons find zero changes outside erasure/translated alpha or inside
protected pixels. These are locality checks, not semantic or aesthetic approval.

### Comfortable automatic typesetting

`rendering/typography.py` owns `comfortable_dialogue_v1`; no typesetting algorithm
is added to the main coordinator, OCR or translation. Automatic text now uses:

* **Inner margins:** shrink only the text-placement mask by 2–8px, scaled to its
  smaller dimension and the supplied maximum size. Constant-zero erosion preserves
  holes and source/page-edge clearance. No erased pixel or outer layout permission
  changes; `unit-NNN-content-mask.png` retains the actual inset mask.
* **A shared preferred size:** 90% of the explicit maximum, clamped to the explicit
  minimum (25px with default 16–28px bounds). Narrow regions may require smaller
  sizes; the policy does not force uniform text or infer a source font.
* **Readable line breaks:** presentation scores penalize isolated 1–2-character
  words in longer utterances, a disproportionately short final line, ragged line
  widths and excess lines. Existing explicit newlines and short/name inputs are
  not treated as accidental short-word orphans. No character, punctuation, space
  or original hyphen is added/removed except legal inserted line breaks.
* **Bounded size comparison:** at most four descending caps (100%, 90%, 80%, 65%
  of the supplied maximum, clamped to the minimum). Each probe finds a contained
  RAQM fit; the best presentation/size score wins. Repeated caps/fits are avoided.
  All probes share the existing 65,536 measurement, 1M DP-transition and 64M local
  raster-pixel limits, not fresh budgets per attempt. Resource, coverage, shaping
  and containment failures stay failures; no retry with weaker margins or bounds.

The review displays chosen/preferred size, margin, line count and a padded-mask
link. Manifest `layout.typography` retains all probe scores, chosen probe, content
bounds and margins. Search/raster work records the total spent, including probes
that were not selected. This is a deterministic bounded aesthetic heuristic, not
linguistic assessment, a globally optimal layout or exact source-font matching.
Manual/assisted `natural.layout(...)` calls keep their existing size-first behavior
unless the optional shared-work/ranking API is explicitly supplied.

On the saved page, all **6/8 eligible** units still compose with an identical
cleanplate. Four use the shared 25px preference; the narrower units use 19/18px.
In particular, `I don't need it.` changes from four stacked lines to two while
remaining within the original size bounds and padded mask.

Two additional saved originals have no strict translated previews. Their tests
therefore use the literal QA string **`Sample dialogue.`**, never a fabricated
translation or composed page: `ubunchu01_02.png` has 6 safe mask plans/5 sample
fits from 15 saved groups (one explicit overflow), and `ubunchu02_04.png` has 8/8
from 14 groups. Uncertain geometry remains skipped; no source pixels are erased.
These 29 cases broaden geometry/typesetting checks, **not** end-to-end translated
page coverage. See `tests/rendering/test_typography.py` and local diagnostic
specimens in `test-results/automatic-typography-multipage-v1/`.

### Automatic nonwhite background restoration

`rendering/restoration.py` implements `source_bound_classical_inpaint_v1`. Both
`src/main.py --render` and `tools/render_preview.py auto-compose` automatically try
it after flat-mask rejection, under the same strict translation bindings and job
budgets. Existing white fills remain the first choice. Typography failures do not
trigger unnecessary inpainting or silently reduce the minimum font size.

Supported initial cases are **strongly contrasting lettering on solid colors,
dark backgrounds, smooth RGB gradients, and mild low-amplitude texture**. This is
not universal restoration of screentones, photographs, or detailed manga artwork:

1. Reuse original member/nonmember geometry and exact overlapping-annotation counts.
   Fit a local affine RGB background from an original 2–6px collar around the member
   geometry, using every sample (no outlier discarding). Require at least 64 and at
   most 65,536 samples, maximum channel deviation 12, RMSE 6, bounded slopes and
   plausible RGB predictions. Neighboring detections or clipped support reject.
2. Seed high-contrast foreground inside source polygons and trace **whole original
   contrast components**, including supported antialias pixels. The residual floor
   is 2–14 channel levels according to measured background variation; the strong
   seed threshold is at least 48. Require complete component containment within the
   member neighborhood, bounded glyph extents, no unassigned contrast, and an
   independent clear collar outside the 1px erasure dilation. Foreground and its
   clearance cannot touch nonmembers. Never paint the OCR rectangle or cut a
   connected art stroke to manufacture a text mask.
3. Run bounded OpenCV **Navier–Stokes** inpainting for smooth support, or **Telea**
   for mild texture, with radius 3. If its result fails background-model consistency,
   try the other method once. Original RGB/BGR conversions are explicit. The raw
   backend result must leave every unmasked pixel exact; backend drift rejects the
   operation rather than being concealed by a subsequent masked copy.
4. Derive placement from original obstacles and nonmember geometry, not from the
   invented cleanplate. Fit English using the same comfortable typography checks.
   Dark restored regions use white text with a black 1px outline; lighter regions
   use black text with a white 1px outline. Only after typography, collisions and
   independent pixel audits succeed are restored mask pixels and text committed.

Every unit records the flat/restoration attempts, skip reasons, actual cleanup
method, foreground/model evidence and restoration consistency metrics. The review
shows those attempts. `background_summary` counts actual flat/restored outputs and
attempts; the main `pipeline.json` includes it in the rendering summary. On failure,
both stages leave source pixels untouched. A restoration failure never promotes
fragments, visual alternates, structurally blocked or untranslated entries.

Additional bounds: local restoration <=262,144 pixels/1024px per side, <=8192
contrast components, <=64 seeded components, <=96px component sides, <=20,000 ink
pixels/component, component ink occupancy <=half the member-mask area (64px floor),
and <=one-third masked local pixels. Joined-letter bounding boxes are not treated
as solid ink; whole foreground components are still retained. At most two backend calls
per attempted unit within the existing <=16-unit job/raster budgets. No provider,
model weights, downloads or dependency changes are required.

**Restoration quality stays unverified.** A consistent background model is not
ground truth for hidden artwork. Texture below the measured noise floor cannot
be distinguished perfectly from faint antialiasing; oversized OCR regions can
also contain artwork. Dense/complex texture, low-contrast text, crossing strokes
and insufficient evidence remain unsupported rather than bypassing the guards.
Truly general artwork restoration still needs stronger text segmentation and a
learned inpainting backend; none is silently downloaded or substituted here.

Known-background synthetic regressions verify color, dark, gradient and mild-noise
cases against the actual pre-lettering pixels, not merely plausible output. They
also verify automatic fallback, dark-text contrast selection, both backend methods,
explicit flat-only mode, locality, nonmember protection and no erasure when typography
fails. The existing saved white page still renders 6/8 eligible translations; its
remaining ownership/ink problems are not solved by changing background methods.
See `test-results/classical-restoration-benchmark-v2/` for labelled synthetic
specimens and measured reconstruction errors; these are not real translated-page
quality claims.

### Learned-mask and background-restoration experiment

The optional local CTD/LaMa experiment is separate from production routing.
It freezes source hashes and development/held-out memberships, compares raw and
source-bound refined CTD masks, and compares classical methods against a pinned
LaMa ONNX conversion **using identical masks and native-resolution contexts**.
That first pass did not add a production adapter. A real standalone CTD output
was obtained before the experiment runner was built. The separately validated
native-component follow-up now has the explicit opt-in route below.

See [the evaluation/provenance record](docs/learned-restoration-evaluation.md) for
exact artifacts, licensing uncertainty, preprocessing corrections, reproducible
commands, results and limitations. Models/dependencies require separate scoped
acquisition; these tools never download them automatically. They do not need
provider calls, training, user editing or changes to existing environments.

### Automatic outputs and provenance

Follow-up native-mask trials recovered **8/8 eligible saved translations**, while
preserving the previous six units' pixels. Source-bound mask association, connected
safe-space layout and phase-locked periodic repair are described in
[the follow-up trial record](docs/native-background-trials.md). These results do
not imply that the other untranslated/blocked units or arbitrary artwork are solved.

To use the retained fallback automatically, add **`--text-mask-model PATH`** to
`auto-compose` or `src/main.py --render`, pointing to the already acquired,
digest-pinned `comictextdetector.pt.onnx`. It tries the cheap existing routes first,
then source-bound native masks with measured flat fill or supported phase-locked
periodic repair. This is one global model setting, not user masks or per-region
confirmations. No downloads or new project dependencies occur. Without the option,
the previous behavior is unchanged. Checkpoint deployment rights remain unresolved;
the current approval covers local evaluation, not distribution/commercial clearance.

The output purpose is `automatic_candidate_output`, separate from operator-selected
`candidate_only` jobs. It includes:

* `original-file.bin`, `source.png`, exact `evidence-preview.json` and
  `evidence-fonts.json`.
* `clean.png`, transparent accepted `text.png`, `composite.png`, applied
  `erase-mask.png`, `alpha-mask.png`, `protected-mask.png` and binary diffs.
* Bounded per-unit foreground/proposed-erasure/applied-erasure/layout masks and
  text layers where derived. Skipped/colliding units have empty applied erasure;
  any saved text is diagnostic, not applied.
* Responsive, escaped, CSP-constrained, script-free `review.html` with original,
  clean, text, composite and mask specimens. No browser/editor/server is launched.
* `manifest.json` with exact source/file/RGB/preview/font hashes, implementation
  and engine versions, strict unit/request/result/text bindings, mask/antialias/
  background evidence, resolved layout/font metadata, artifact hashes, job revision,
  machine selection, coverage and per-unit skip reasons.

`operator_confirmation` stays null, `linguistic_approval` stays `not_assessed`, and
`approved_export` is false. Nothing fabricates a human review, silently calls
confirmation, or promotes fragments/blocked/VLM alternatives. The shared strict
result validator is used under an explicit `automatic_candidate_only` binding
contract with no geometry-intent/review assertion. Existing manual commands keep
their original default binding contract and confirmation behavior.

Per-unit `candidate_pixel_audit` describes the isolated proposed operation before
collision rejection; final global `pixel_audit` and applied masks describe actual
committed pixels. Runtime outside-cleanup/outside-alpha/protected change counts
must be zero. Failed final locality rolls output back to source and records the
rejected counters. Inputs and local font/license hashes are rechecked before
exclusive publication. CLI exit 0 means a report/partial candidate was written,
not that every unit rendered or the language/visual quality is correct.

Bounds: <=4M source pixels/4096px per side, <=512 source units/detections, <=32
processed mask slots, <=128M aggregate unit-page pixels with bit-packed retained
masks and <=512M source-binding/
neighbor raster work. Labelings allow <=8192 components; <=64 seeded components
per unit, max component side 96px/20,000 ink pixels and bbox area no more than half
member-mask area (with a 64px floor). Existing font/license, RAQM search and local
raster budgets apply. At most 512 artifacts/128 MiB total, HTML <=4 MiB, manifest
<=8 MiB, and specimens <=1000px per side. Budget exhaustion is explicit partial
skipping or preflight rejection, never implicit truncation or unsafe rendering.

Remaining uncertainty: source polygons and strict candidates can be wrong. This
policy does not certify text-versus-art identity inside an oversized source box,
utterance completeness/punctuation outside detections, language truth or balloon
ownership. Unassigned nearby contrast is a skip; unknown outside obstacles are
preserved and constrain layout. The parent independently checks actual generated
page outputs; product generation has no operator confirmation gate.

```sh
.venv/bin/python -m unittest tests.rendering.test_automatic tests.rendering.test_automatic_layout tests.rendering.test_typography tests.rendering.test_restoration -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/tralens/rendering tests/rendering tools/render_preview.py
git diff --check
```

## Candidate natural rendering (approved Phases 0–3)

`tools/render_preview.py` is an offline, separate rendering workstream. It reads
an existing translation preview and explicit operator-supplied selection, font
manifest and original image. It never initializes OCR, translation, models or a
provider. Legacy `tralens.rendering.cover_text` / `draw_text` remain compatible.

```sh
.venv/bin/python tools/render_preview.py plan \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --selection /explicit/local/selection.json --fonts /explicit/local/fonts.json \
  --output test-results/render-plan-parent-v1
.venv/bin/python tools/render_preview.py compose \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --selection /explicit/local/selection.json --fonts /explicit/local/fonts.json \
  --output test-results/render-compose-parent-v1
```

Both output directories must be fresh and their parents must already exist.
Existing outputs, symlink inputs/outputs/ancestors and overlapping selected regions
are rejected. A failed run may retain its fresh partial directory as evidence;
choose another output name for a subsequent run. Source files are never written.

### Explicit font manifest

Supply installed/local **single-face, static sfnt TTF/OTF** fonts and local license
evidence. No font lookup, fallback, download, collection, variable-font support or
font redistribution is performed. Each font is manually assigned a dialogue role
and regular/bold weight. SHA256 values are over exact file bytes. Example schema:

```json
{"schema_version":1,"fonts":[{
  "id":"dialogue-regular", "role":"dialogue", "weight":"regular",
  "path":"/explicit/local/dialogue.ttf", "sha256":"<font-file-sha256>",
  "license":"<actual license identifier or description>",
  "license_path":"/explicit/local/font-license.txt",
  "license_sha256":"<license-file-sha256>"
}]}
```

Only font/license paths explicitly listed in this manifest are read; rendering
geometry cannot name JSON or mask sidecars. Font bytes and licenses are not copied
into outputs. Hash/license evidence records supplied provenance, not a legal
permission determination. Unicode cmap formats 4/12 verify nonzero glyph mappings;
unknown/malformed coverage fails closed. Coverage is not shaping-quality proof.

### Source-bound selection generation and geometry

This milestone produces **candidates only**, never approved exports. Schema 1/2/3
strict preview results are not selected automatically. The operator chooses exact
strict unit IDs/text; choosing a candidate does not claim Daniel checked every
word. Fragments, VLM visual alternates, blocked units and modified wording are
rejected. No reviewed-override path is implemented; such cases require a separately
specified, real operator decision contract. Do not relabel a fragment as strict.

The following parent-side recipe creates a bound selection after the operator has
chosen `UNIT_ID`, a font and actual source-pixel polygons. It is documentation,
not an instruction to run on private page data during coding. `GEOMETRY` must be
filled from original-page inspection; OCR rectangles alone are not erase permission.
No approval flags or sample-specific production rules are generated.

```python
from pathlib import Path
from tralens.rendering.natural import document, digest, read_explicit
from tralens.translation.contracts import canonical, revision

preview_path = Path("test-results/workflow-fragments-priority-parent-v1/preview.json")
fonts_path = Path("/explicit/local/fonts.json")
UNIT_ID = "<operator-chosen strict unit ID>"
FONT_ID = "dialogue-regular"
# Each value is a list of polygons, each polygon a list of [x,y] vertices.
# Fill all four using original source-pixel inspection before running this recipe.
GEOMETRY = {"interior": [], "layout": [], "cleanup_permission": [], "protected": []}

preview_bytes = read_explicit(preview_path)
p = document(preview_bytes)
r = next(r for r in p["units"] if r["unit"]["unit_id"] == UNIT_ID)
assert r["status"] == "preview" and r.get("input_kind", "strict_utterance") == "strict_utterance"
assert r["result"]["status"] == "preview"
translations = dict(r["result"]["translations"])
unit = {
    "unit_id": UNIT_ID, "member_ids": r["unit"]["member_ids"], "order": r["unit"]["order"],
    "unit_revision": r["unit_revision"], "request_revision": r["request_revision"],
    "result_sha256": revision(r["result"]), "text": translations[UNIT_ID],
    "font_id": FONT_ID, "style": {"min_size": 16, "max_size": 30, "outline": 0},
    "geometry_intent": "<operator's actual original-page geometry review intent/evidence>",
    **GEOMETRY,
}
selection = {
    "schema_version": 1, "purpose": "candidate_only", "source": p["input"],
    "preview_sha256": digest(preview_bytes), "fonts_sha256": digest(read_explicit(fonts_path)),
    "units": [unit],
}
# Exclusive creation, never overwrite previous evidence:
with Path("/explicit/local/selection.json").open("x", encoding="utf-8") as stream:
    stream.write(canonical(selection) + "\n")
```

`PYTHONPATH=src` is sufficient for this recipe. Input `source.path` must equal the
explicit original image's absolute resolved path. `source.sha256` binds original
file bytes; `pixels_sha256` is SHA256 of `RGB:<width>:<height>:` followed by decoded
RGB bytes, and `size` is `[width,height]`. These are different identities. Saved
source/member order, raw geometry/text, unit/request/result revisions, chosen text,
preview bytes and font-manifest bytes are checked again; original preview JSON is
retained in full. Only supplied files are opened, never paths in a preview.

All masks use **original source pixel coordinates**, with inclusive polygon
vertices bounded by `0..width-1`, `0..height-1`; they are not preview-thumbnail,
normalized or half-open rectangle coordinates. Supplied mask vertices are integers;
retained OCR polygons may have finite fractional coordinates. Polygons must be
simple/nondegenerate; masks are inline unions. Supply:

* `interior`: explicitly inspected supported flat balloon interior.
* `layout`: approved text placement region, inside interior, excluding protection.
* `cleanup_permission`: narrowly supplied glyph-search permission, inside interior,
  excluding protection. This is **not** the final erased mask.
* `protected`: border/art/other pixels that must remain exact original pixels.
  An empty list means the caller supplies no additional protected pixels.

Nonmember OCR polygons intersecting an interior block the selection, including
annotations. Undetected art/annotations still require original-page inspection.
Cleanup estimates a median RGB background from the interior outside permission
and protection (at least 32 samples), requires all sampled pixels within 2 channel
levels, and requires a light background (all channels >=180). Off-white is valid.
Pixels differing by more than 2 levels form a limited glyph mask, whose area must
be at most one-third of the interior and whose one-pixel collar stays within
permission, interior and outside protection. Unexplained ink, sampled texture,
gradients or crossing strokes block cleanup. This conservative test is not a
general art/text separator: supplied permissions must describe actual glyphs.
Only glyph-mask pixels receive the sampled color; no whole OCR rectangle is painted.

Cleanup failures now carry bounded, actionable `cleanup_diagnostics` on the blocked
unit. Codes distinguish `insufficient_sample`, `nonflat_sample`,
`unsupported_dark_background`, `unpermitted_ink`, `no_cleanup_ink`,
`excessive_ink_area`, and `boundary_touch`. These describe observed constraints,
not semantic identification of a border, texture, gradient, annotation or art.
`semantic_classification` stays `not_inferred`.

Each diagnostic records the actual median/floored RGB background estimate (null
when no samples exist; provisional when samples are insufficient), sample/interior
counts, unchanged thresholds, observed maximum sample deviation when measured,
glyph count when reached, offending-pixel count and exact bounding rectangle.
Bounds are **half-open source pixels** `[left, top, right, bottom]`. At most 32
coordinates `[x,y]` are retained in deterministic row-major order, with an explicit
truncation flag. Coordinate extraction uses bounded row chunks instead of creating
an unbounded full coordinate array. Boundary-touch coordinates identify the
unsupported one-pixel glyph collar; they need not be dark pixels themselves.

The static report visibly explains the rejection, background/sample/offending
counts and next safe action. Inspect the **original source**, then supply a fresh
explicitly reviewed geometry revision or leave the unit blocked. A nonflat sample
does not authorize erasing the reported marks or relaxing the variance guard.
Blocked units apply neither cleanup nor translated alpha, and their applied
cleanup specimen remains empty. Successful/plan units have
`cleanup_diagnostics: null`. Candidate/review verification re-runs cleanup for
blocked layouts and checks exact diagnostics and rejection reasons, even if a
modified report has had its artifact hashes recomputed. Implementation changes
require fresh jobs; earlier artifacts and review records are retained.

### Typography, artifacts and parent review

English horizontal/LTR dialogue uses the same Pillow RAQM engine for measurement
and drawing (`textlength`, ink `multiline_textbbox`, kerning, language `en`). Latin
letters/combining marks and punctuation are supported if explicitly covered by the
font. Explicit newlines, punctuation and whitespace are retained; additional
linebreaks can be inserted after ordinary spaces or after a single existing
ASCII/U+2010 hyphen between letters/digits. The existing hyphen stays on the first
line; no hyphen is added or removed. Nonbreaking U+2011 hyphens, dash runs and
punctuation-only suffixes do not create internal break sites. Search is
deterministic: descending integer size within operator bounds (8–144), then
bounded balanced line partitions (minimum squared visible RAQM line advances for each
line count) with six legal greedy alternatives for irregular regions. At a given
size fewer added breaks are preferred, then balanced advances and stable source
offsets. A generic hyphenated utterance can therefore fit as two larger lines in
a tall region, rather than forcing a small single line.

Drawing first uses a padded **local glyph canvas**, independent of source-page
edges. Touching that canvas's guard blocks output. Actual complete ink is cropped,
centered and checked against the full-resolution layout region before page
placement; nonzero alpha counts are compared again after placement. Actual ink,
including a caller-selected white outline (0–8 pixels), must fit. Black text/white outline
are the baseline. No truncation, rewriting, squashing, clipping or fallback font
is used. Overflow/missing glyphs/unavailable RAQM become explicit blocked units.

Fit screening ignores only the advance of existing trailing ASCII spaces before
line ends; those spaces remain in exact drawn text. Otherwise an invisible blank
advance can falsely reject a larger complete-ink fit in a narrow region. The
multiline metric width is not a final rejection criterion: drawing on the bounded
padded canvas and checking complete final alpha remain authoritative. Layout
provenance records full `line_advances`, separate `line_fit_advances`, and policy
`balanced_legal_hyphen_visible_space_v2`. Size bounds and all containment, outline,
rotation and resource checks remain unchanged. This bounded search does not promise
a globally optimal fit: unbreakable words, supplied width/height and irregular
regions still legitimately limit font size. Geometry is never expanded automatically.

`plan` draws transparent layers and a no-erase composite; `clean.png` stays exact
canonical RGB source. `compose` independently validates cleanup before applying a
unit's text; blocked units do not alter the cleanplate or composite. Outside the
cleanup mask the cleanplate is exact canonical source; outside translated alpha
the composite is exact cleanplate. The original file bytes remain untouched and
are retained as `original-file.bin`. Only RGB/L, single-frame rasters are accepted;
embedded ICC profiles and nontrivial EXIF orientation are rejected. There is no
color-managed/profile-conversion claim.

Output schema 1 contains:

* `manifest.json`: `purpose: candidate_only`, `approved_export: false`, mode,
  immutable canonical selection/source/preview/font bindings, engine versions,
  per-unit status/reason, exact chosen/drawn text, ink bounds, origin, advances,
  size/outline/spacing, sampled background and cleanup count; artifact SHA256 map.
  Layout provenance also records `inserted_break_offsets` in the **original text**,
  `line_break_policy`, local canvas size, complete `local_alpha_pixels`, cumulative
  raster work and line-search work. Reconstruct `draw_text` by inserting newlines
  at those offsets in reverse order; original explicit newlines stay in place.
* `evidence-preview.json`, `evidence-selection.json`, `evidence-fonts.json`,
  `original-file.bin`, `source.png`, `clean.png`, `composite.png`.
* `unit-NNN-{interior,layout,permission,protected}.png`, transparent
  `unit-NNN-text.png` when layout succeeds, `unit-NNN-glyph-cleanup.png` when
  composition succeeds; `cleanup-diff.png` and `composite-diff.png` binary maps.
* Static, script-free `review.html` with source/clean/composite comparisons,
  diffs, unit statuses and a provenance link. No browser is launched.

Per-unit and final `pixel_audit` fields count actual RGB comparisons at runtime:
clean changed/unchanged pixels, unchanged and changed pixels outside cleanup,
protected unchanged/changed pixels, translated alpha pixels, composite changes
outside alpha, and `untouched_source_pixels`. Outside-cleanup, outside-alpha and
protected change counts must be zero; per-unit `outside_layout_alpha_pixels` and
`outside_cleanup_permission_pixels` must also be zero. A failed per-unit check
blocks that unit before applying its pixels. A failed final check retains
`rejected_pixel_audit`, marks all units blocked and writes canonical original
clean/composite images instead. These are pixel invariants, never human approval.

Resource checks precede large coverage-set or raster allocations: at most 16M
source/page pixels, 32M aggregate unit-page pixels, 4M pixels/8192 pixels per side
for each padded local canvas, and 64M cumulative local raster pixels per unit.
Fonts are limited to 32 MiB per file/64 MiB aggregate, 128 tables, 65,536 expanded
Unicode cmap points per font and 131,072 aggregate coverage points. Table bytes
use bounded views rather than duplicated slices. Search allows at most 128
paragraph/break endpoints, 32 total lines/candidates, 16 lines per paragraph,
65,536 substring measurements and 1M DP transitions per unit across sizes.
The manifest records major `resource_limits`. Per-font and per-unit resource
failures become `blocked` with `failure_kind: resource_limit`; page/aggregate
preflight failures reject the command before reserving output. Conservative
background variance, protected-pixel and contamination guards are unchanged.

Statuses are `candidate_layout_no_erase`, `candidate_composed`, or `blocked`.
CLI success means the candidate report was produced, not that every unit rendered;
inspect statuses. Hashes bind evidence, not translation correctness or approval.
The parent must supply real-page selections/fonts/geometry, generate fresh outputs
and visually inspect the original-resolution results before any subsequent phase.
No real-page quality, semantic correctness or human approval is claimed here.

Offline synthetic verification (includes an original synthetic sfnt font fixture,
actual subprocess CLI plan/compose, and both legacy renderer tests):

```sh
.venv/bin/python -m unittest tests.rendering.test_layout_repairs tests.rendering.test_natural tests.rendering.test_renderer -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/tralens/rendering tests/rendering tools/render_preview.py
git diff --check
```

### Phase 3: explicit role styles and aesthetic review loop

Existing schema-1 selections and `plan`/`compose` commands still work. Omitting
`preset` keeps the legacy black/white, centered, unrotated rendering behavior.
An operator can copy a selection to a **fresh JSON file** and edit its per-unit
`style` object, retaining exact chosen text and all source/request/result bindings:

```json
{"preset":"shout", "min_size":16, "max_size":28,
 "fill":[25,25,25], "outline":2, "outline_fill":[255,255,255],
 "leading":0.2, "align":"center", "rotation":-5}
```

| Preset | Default RGB fill | Outline | Leading (em spacing) | Alignment | Requested weight |
| --- | --- | --- | --- | --- | --- |
| `dialogue` | `[0,0,0]` | 0 | 0.20 | center | regular |
| `narration` | `[25,25,25]` | 0 | 0.30 | left | regular |
| `shout` | `[0,0,0]` | 2 | 0.15 | center | bold |
| `thought` | `[55,55,70]` | 1 | 0.30 | center | regular |

The operator must explicitly choose `font_id` from the supplied, hash/license-bound
font manifest. Presets never select a system font or synthesize a bold face. The
same already-approved local font ID can be explicitly used for any presentation
role; manifest `role`/`weight` are declarations, not measured face certification.
`resolved_style.actual_face` records FreeType-reported family/subfamily and the
chosen font's OS/2 `weight_class`, or null when unavailable. `requested_weight`
is separately recorded, with `synthetic_weight: false`. `style_approximation`
always says no exact source-font match is asserted and explains unknown or
mismatched weights. To change actual weight, select another explicit licensed
font ID/face; changing only the preset does not change the selected face.

Controls are bounded: RGB integer triplets 0–255, outline 0–8 px, leading 0–1 em,
alignment `left`/`center`/`right`, and finite group rotation −20° to +20°. The
explicit 8–144 px size bounds remain mandatory. Alignment aligns the lines inside
the ink-centered group. Rotation transforms the entire multiline group, including
outline, on its guarded local canvas using bicubic resampling. Both untransformed
and transformed ink must pass padding/crop guards; OpenCV's raster bounding box
independently checks the complete post-transform alpha crop against Pillow's
ink bounds. The complete transformed alpha
must fit the full-resolution approved layout/protection masks. Transformed canvas
allocations and cumulative raster work are budgeted before rotation. Overflow and
missing glyphs stay blocked; text is never rewritten or reduced below its bound.
Rotated layouts record pretransform alpha count, posttransform canvas size and
complete final alpha count; their page `origin` is null because a single unrotated
drawing origin no longer describes placement.

#### Static specimens and fresh revisions

Every new job produces a CSP-constrained, script-free `review.html`. Each unit has
side-by-side original/clean/text/composite specimens, interior/layout/permission/
protected-mask and actual applied glyph-cleanup views, blocked reasons, exact text, resolved settings, approximation
notes and locality counters. `unit-NNN-specimen-*.png` thumbnails are bounded to
1000×1000 pixels, with original source bounds and scale recorded; full-resolution
evidence is retained separately. Output `review.html` and specimen images are
included in the manifest's artifact hashes.

The report wraps long unit IDs/job hashes and uses four desktop columns, two at
700px and below, and one at 420px and below. Figures have no browser-default
margins/minimum width; content stays readable without globally hiding overflow.
The applied glyph-cleanup specimen is empty for plan/blocked units, which perform
no erasure. It is hash-bound and reconstructed during candidate/review verification.

`job_revision` deterministically binds exact selection bytes, original/preview/font
bindings, mode, rendering implementation hashes, shaping-engine versions and any
explicit parent/comparison provenance. Equal inputs/implementation produce equal
revisions and artifacts across fresh output names. Edited jobs always write fresh
directories; neither source evidence nor earlier jobs/reviews are overwritten.

Parent-side example using the supplied installed-font manifest (agent verification
uses synthetic rasters/fonts only):

```sh
# Phase-3 baseline; the Phase-2 parent is retained as a hash-checked snapshot.
.venv/bin/python tools/render_preview.py compose \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --selection test-results/render-parent-input-v1/selection-broader-v3.json \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --parent test-results/render-broader-parent-v3 \
  --output test-results/render-phase3-baseline-parent-v1

# Copy/edit selection-broader-v3.json to selection-phase3-thought-v1.json first.
# Set the desired units' preset/controls and explicitly keep/choose font IDs.
.venv/bin/python tools/render_preview.py compose \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --selection test-results/render-parent-input-v1/selection-phase3-thought-v1.json \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --parent test-results/render-phase3-baseline-parent-v1 \
  --compare test-results/render-phase3-baseline-parent-v1 \
  --output test-results/render-phase3-thought-parent-v1
```

Use distinct fresh names if any destination exists. `--parent` links the exact
prior manifest/job/source hashes without inheriting its review state. Historical
Phase-2 parents can be linked as hash-checked snapshots. `--compare` accepts at
most four **explicit CLI directory inputs** for current Phase-3 jobs; their
source/preview/font bindings, masks, re-rendered layers, reconstructed output
pixels, specimens and hashes are validated before copying a comparison image.
Selections may differ. Preview/image/font inputs must be the same current ones.
No candidate/parent/comparison paths are opened from selection JSON or arbitrary
sidecars. A changed font manifest requires updating `selection.fonts_sha256` to
its exact bytes; changed translated wording still requires a correspondingly
source-bound strict preview/result, not a free-text rendering override.

#### Explicit aesthetic review records

Candidates always start with `aesthetic_state: not_assessed` and
`linguistic_approval: not_assessed`. After actual operator inspection, a separate
explicit action can record `not_assessed`, `accepted`, `rejected` or `rework` for a
successfully composed unit. Blocked, foreign and plan-only targets are rejected.
There is no inferred/model-generated acceptance or final export.

```sh
# Replace the ID/operator/note with the real chosen unit and actual review action.
.venv/bin/python tools/render_preview.py record-review \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --selection test-results/render-parent-input-v1/selection-phase3-thought-v1.json \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --candidate test-results/render-phase3-thought-parent-v1 \
  --unit-id '<exact composed unit ID>' --state rework \
  --operator '<actual operator>' --note '<actual aesthetic observations>' \
  --output test-results/render-parent-input-v1/aesthetic-thought-v1.json

.venv/bin/python tools/render_preview.py verify-review \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --selection test-results/render-parent-input-v1/selection-phase3-thought-v1.json \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --candidate test-results/render-phase3-thought-parent-v1 \
  --review test-results/render-parent-input-v1/aesthetic-thought-v1.json
```

Production API equivalents are `record_review(...)`, `verify_review(...)` and
`verify_candidate(...)` in `tralens.rendering.review`; `natural.run(..., parent=...,
compare=[...])` creates fresh jobs. Records have `scope: aesthetic_only`, an explicit
operator/note/state, `identity_verified: false`, `linguistic_approval: not_assessed`,
`candidate_only: true`, an exact binding and a self-consistency revision hash.
Record creation/verification re-reads the supplied current source, preview,
selection and font/license bytes, re-layouts composed text and reconstructs actual
clean/composite pixels. It checks all mask and specimen pixels, layout/settings,
engine/implementation revisions, counters and closed-name artifact hashes.

The review binding includes exact manifest/job/source/preview/selection/font hashes,
chosen unit settings/layout, all mask/output/report artifact hashes. Any relevant
edit—including source-file bytes, text, geometry, font/license, style, rendered
output, HTML or implementation—invalidates an old acceptance. A new revision does
not carry acceptance forward. Verification checks unsigned provenance consistency;
it does **not** authenticate a person's identity, prevent a person from forging a
new unsigned record, or establish translation truth. Aesthetic acceptance never
promotes the candidate to a final export.

Additional bounds: selection JSON ≤4 MiB, font manifest ≤256 KiB, manifest ≤8 MiB,
HTML ≤4 MiB, review record ≤16 KiB, at most 1200 closed-name artifacts and 128 MiB
aggregate artifact bytes per job/verification. Reports contain no scripts,
external assets, editor/server, or save controls. Existing flat-interior variance,
glyph-mask and protected-border guards are unchanged. Installed-font/real-page
appearance remains parent visual verification.

Phase-3 checks include actual synthetic CLI compose/revision/record/verify runs:

```sh
.venv/bin/python -m unittest tests.rendering.test_refinement tests.rendering.test_phase3 tests.rendering.test_layout_repairs tests.rendering.test_natural tests.rendering.test_renderer -v
.venv/bin/python -m unittest discover -s tests -q
.venv-paddle6/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest tests.source_only_check -v
.venv-paddle6/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/tralens/rendering tests/rendering tools/render_preview.py
git diff --check
```

### Assisted flat-balloon preparation: proposals, then explicit confirmation

`tools/render_preview.py prepare` reduces manual polygon transcription for a
conservative subset of closed flat balloons. It consumes the **original** image,
saved preview, explicit licensed font manifest/ID, size bounds and explicitly
requested strict unit IDs. It imports no OCR facade, initializes no OCR/provider,
and never changes recognition/translation evidence. A successful proposal is
geometry evidence only, not erase permission or linguistic approval.

The installed OpenCV/Pillow path:

1. Labels original light pixels (all RGB channels >=180) with both 4- and
   8-connectivity. Member background support must identify the same closed,
   non-frame-connected enclosure in both labelings. No OCR polygons are removed
   from the source raster to manufacture an enclosure.
2. Traces whole original dark components touching source members. Their complete
   extents must fit a bounded member neighborhood and remain separate from
   outline/art. Every other dark component and every nonmember OCR polygon inside
   the enclosure rejects the proposal, including annotations and unassigned marks.
   Missing punctuation is not silently eroded away or granted ownership.
   Touching/seam checks are local: re-label original outline ink in a fixed 8px
   exterior band and test source-supported peer contacts there. A distant common
   black panel/frame is not by itself touching-balloon evidence. Peer support
   uses independent original detection geometry (at least half light support,
   90% of light votes in one region), never confidence or translated wording.
   Enclosing/four-edge surrounding light regions are recorded as ambient context;
   unassigned exterior art pockets/counters are recorded without declaring them
   balloons or granting any erase permission. This does not certify semantic
   independence or detect every unlabeled neighboring balloon.
3. Accounts for holes only when their dark pixels belong to those supported
   components (e.g. a glyph counter). This fills an **analysis silhouette**, not
   source pixels. There is no hull mask or OCR-rectangle whitening.
4. Insets the original enclosure for boundary clearance, preserving all required
   glyph neighborhoods. The usable core must have single row/column spans.
   Small irregular empty edge scanlines can be excluded only after the **entire
   current core** passes unchanged cleanup/flatness checks; at most the supplied
   clearance count per edge is trimmed and no required ink/collar pixel can be
   discarded. Internal gaps, excessive irregularity, shared source regions and
   clipped cases still reject. Bounded polygonization
   is accepted only after checking every rasterized pixel stays inside the safe
   mask and all required ink/collar pixels remain represented. Unrepresentable
   holes or excessive geometry complexity reject.
5. Proposes independent interior, inset layout and limited glyph-neighborhood
   cleanup permission. The unchanged flatness/collar cleanup guard runs as an
   offline preflight. It is not broadened or retried with weaker thresholds.
   A nonflat sample confined to the original edge can qualify for one additional
   global inward geometry step (maximum 3px): every offending pixel must connect
   to the original dark outline through original non-background pixels in the
   bounded boundary band, avoid source-member neighborhoods, and preserve **all**
   original owned ink/permission/collar pixels. The inset is measured from exact
   edge distances, not searched until a sample passes. The resulting complete
   core must pass the same cleanup guard or remain blocked. Isolated marks,
   interior contamination, gradients, nonmembers and narrow-clearance failures
   do not qualify. Tracing 1/2-level antialias bridges is original-pixel evidence,
   never a change to cleanup's maximum allowed sample deviation of 2.

`protected` is empty in this initial assisted path: no art-hole permission is
inferred. Boundary pixels are excluded by the interior clearance, and the existing
renderer preserves all pixels outside cleanup/translated alpha. This does **not**
claim a separately inspected complete border/art mask. Unknown enclosed ink is
rejected rather than covered by an outer contour or invented protection decision.

The feasibility suite supports rectangles/ellipses, white/off-white, accounted
glyph counters, and a synthetic eight-border-pixel geometry-error case. It rejects
open/page/frame-connected, shared/touching, nonmember/furigana, unexplained ink,
gradient/texture and clipped-component cases. This is not universal balloon,
ownership, text-versus-art or full-utterance detection: art completely inside an
oversized source member may be indistinguishable from glyph evidence. Operator
inspection of the original source remains necessary. Real pages may yield many
honest rejects; existing manual selection remains available.

The local-boundary revision records exact global/local contact counts and capped
adjacent-label samples (bounds, area, source IDs and distance), rather than treating
every closed light pocket connected through page-wide black artwork as a balloon.
Its saved-page regression reads only the explicitly authorized original/preview/
font manifest and `render-preparation-parent-v1/proposals.json`; it creates no
real-page artifact. The later boundary-inset revision reads the authorized
`render-preparation-parent-v2/proposals.json` request and produces two geometry
positives in memory: Risa and the long unit. Risa's three 3-level edge pixels
qualify for a 1px additional inset with zero required pixels discarded; the same
variance limit of 2 then passes. The two units sharing white component 719 still
reject their mutual nonmember geometry.
These are measured candidate/guard outcomes, not visual or linguistic approval.

`additional_boundary_inset` retains the original failure, connectivity evidence,
exact proposed inset, refusal reason where applicable, excluded-pixel count and
required-pixel preservation. Confirmation regenerates this evidence and geometry;
it is not an editable approval flag. No permission grows and no source pixels are
changed by proposal generation.

Shared-region diagnostics now probe original enclosure clearance radii 1–8 while
checking each source group's complete ink plus collar. The actual source 8/10
region remains a single component at every radius, without losing either group's
required pixels. There is no defensible narrow separator within the supported
range: no bbox/Voronoi cut, invented line, translation merge or ownership split is
exported. A synthetic thin-corridor counterexample confirms the diagnostic can
detect distinct complete-ink cores, but it remains diagnostic, not automatic
permission to split a joined speech region.

Additional-page evaluation is **geometry-only**: the discovered 25 preview-named
artifacts (including retained copies/baseline) all identify `ubunchu01_04.png`.
No valid different-page strict translated preview was found. The existing
`automatic-unit-diagnosis-parent-v3/automatic-proposals.json` supplies hash-bound
originals/raw polygons for separate evaluations: all 15 saved groups on
`ubunchu01_02.png` yield 1 proposed/14 rejected; all 14 on `ubunchu02_04.png`
reject. Image byte/RGB hashes and saved raw/proposal revisions are validated; no
strict preview, translated result, selection or confirmation is manufactured.
These older groups include real ambiguity/nonmember/clipped/irregular cases and
are not an end-to-end translation benchmark. Tests run at most two bounded
evaluation batches (<=16 groups total) per page and preserve explicit skips when
the local evidence is unavailable.

For pixel contours that touch their own vertex/edge, exact clipped strips of
32/16/8 pixels in two orientations can be decomposed into simple integer cycles.
Every reconstructed mask must remain a subset of the original allowed mask and
retain all required pixels; no permission is broadened. The existing 128 polygons,
128 vertices/polygon and 4096 aggregate vertices/unit limits remain. Splitting is
bounded to 256 pending/output cycles and 4M edge tests per strip trial. This fixes
raster representation only; it does not split shared balloon interiors. Preview
windows include the original enclosure so excluded background stays inspectable.

#### CLI workflow (parent-generated real-page artifacts)

```sh
common=(
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json
  --image images/ubunchu-ja-test/ubunchu01_04.png
  --fonts test-results/render-parent-input-v1/fonts.json
)

.venv/bin/python tools/render_preview.py prepare "${common[@]}" \
  --unit-id '<explicit saved strict unit ID>' \
  --font-id dialogue-bold --min-size 16 --max-size 28 \
  --output test-results/render-preparation-parent-v1

# Inspect review.html, original/overlay images, reasoning and exact geometry first.
# Repeat --proposal-id only for the specific inspected, proposed rows to select.
.venv/bin/python tools/render_preview.py confirm-preparation "${common[@]}" \
  --proposal-bundle test-results/render-preparation-parent-v1 \
  --proposal-id '<exact geometry-... ID from a proposed row>' \
  --operator '<actual inspecting operator, including honest AI-PM identity>' \
  --intent '<actual original-source geometry inspection/selection intent>' \
  --output test-results/render-parent-input-v1/selection-assisted-v1.json

.venv/bin/python tools/render_preview.py compose "${common[@]}" \
  --selection test-results/render-parent-input-v1/selection-assisted-v1.json \
  --output test-results/render-assisted-compose-parent-v1
```

`--unit-id` can repeat during preparation; there is no implicit all-units mode.
Only strict source/request/result bindings accepted by the existing renderer can
produce confirmable geometry. Blocked strict records, fragments, VLM alternates
and other unrequested evidence stay in the exact retained preview. Choosing a
strict input for proposal generation does not claim its wording was human checked.

Preparation output has **`purpose: flat_geometry_proposals`**, which compose
rejects as a selection. The exclusively created bundle contains:

* `proposals.json`: deterministic proposal revision/IDs, source/file/RGB/preview/
  font/implementation bindings, settings, `proposed`/`rejected`/`unsupported` rows,
  exact selected-result bindings where valid, geometry, measured reasoning,
  uncertainty and artifact hashes. `operator_confirmation` remains null.
* `evidence-preview.json`, `evidence-fonts.json`, `evidence-request.json`,
  `original-file.bin`, `source.png`: original evidence, never an erased page.
* Bounded `unit-NNN-source.png` / `unit-NNN-overlay.png` views and responsive,
  escaped, CSP-constrained, script-free `review.html`. Green is interior, blue
  layout, orange cleanup permission; rejected source footprints are red.

No cleaned image, composite or translated text layer is written by preparation.
The cleanup preflight is evaluated in memory and discarded. No source image is
overwritten. Multiunit proposed-region collisions reject both affected proposals.

Confirmation requires explicit unique proposal IDs, operator identity and intent;
no generation flag can confirm all units. It re-reads the current explicit source,
preview and font/license inputs, revalidates bindings, and **regenerates** geometry,
measurements, IDs, overlays and report bytes under the current implementation.
Every stored artifact must match. Tampered geometry/output/report, stale source,
font/license/implementation changes, invented approval flags and unsupported IDs
reject. No filenames from supplied manifest JSON are followed: only regenerated,
closed artifact names are opened. The result is a fresh `candidate_only` selection
with separate `preparation_confirmation` provenance (`scope: geometry_candidate_only`,
`identity_verified: false`, `linguistic_approval: not_assessed`). It does not modify
the bundle or authenticate a person; it is an unsigned, operator-supplied action.
The normal plan/compose cleanup, font coverage, full-alpha and pixel-locality
guards still run independently and can block confirmed geometry/typography.

Production APIs are `propose_geometry`, `prepare`, and `confirm_preparation` in
`tralens.rendering.preparation`; production never imports `tools`. Confirmation
outputs cannot overwrite files or be written inside the original proposal bundle.
Repeat generation/confirmation uses fresh paths and revisions, retaining evidence.

Preparation bounds are intentionally narrower than manual rendering: <=4M source
pixels and 4096 per side, <=16 requested units, <=32M aggregate unit-page pixels,
<=512 raw detections, <=512M source-binding/neighbor raster work, <=32 member masks
and <=32M member-mask pixels per unit. Each original labeling allows <=8192
components; each unit <=64 owned ink and <=64 outline components. Enclosure area must be >=64 and
<=1M pixels, <=60% of the page and <=30 times member-mask area. Geometry allows
<=128 polygons/mask, <=128 vertices/polygon and <=4096 aggregate vertices/unit.
Defaults: clearance 3px (allowed 2–8), member padding 2px (0–4), glyph padding 2px
(2–4), layout inset 1px (0–4). Only these four integer settings are configurable;
there is no variance/brightness bypass. Source preview <=16 MiB, image <=32 MiB,
font manifest <=256 KiB; manifest/HTML <=4 MiB each; aggregate artifact bytes
<=128 MiB; overlay sides <=1000px. Existing local-font/license budgets also apply.
Resource/invalid-input failures reject before output publication; valid but
unsupported unit cases are preserved explicitly in the proposal report.
The additional inset has a fixed 3px cap; the separator diagnostic tests at most
8 radii and 8192 components per labeling. Neither exposes a tolerance/ownership
bypass option. `limits` in each new proposal bundle records these bounds.

Offline checks (synthetic assets/fonts; parent owns saved-page generation/inspection):

```sh
.venv/bin/python -m unittest tests.rendering.test_preparation tests.rendering.test_refinement tests.rendering.test_phase3 tests.rendering.test_layout_repairs tests.rendering.test_natural tests.rendering.test_renderer -v
.venv/bin/python -m unittest tests.rendering.test_preparation_boundaries -v
.venv/bin/python -m unittest tests.rendering.test_geometry_coverage -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/tralens/rendering tests/rendering tools/render_preview.py
git diff --check
```

## Opt-in Paddle model profiles (isolated 3.7 runtime)

The production default is explicitly pinned to **`v6_medium`**, including both
model names, by the approved default-selection change. It requires the validated
**PaddleOCR 3.7.0 / PaddleX 3.7.2** runtime in `requirements-paddle6.txt`.
`OCR(image_path)` keeps its call shape; its selected model now changes to medium.
Grouping/review/candidate policies remain unchanged. Counts alone are not an
accuracy justification. **`v5_server` remains an explicit compatibility option**
for the original runtime: `OCR(image_path, model_profile="v5_server")`.

```python
from tralens.ocr import OCR

ocr = OCR(image_path, device="gpu:0",  # Default: v6_medium.
          model_root="~/.paddlex/official_models")
# Equivalent explicit local export pair (detector first, recognizer second):
ocr = OCR(image_path, device="cpu", cpu_threads=1, model_dirs=(
    "~/.paddlex/official_models/PP-OCRv6_medium_det",
    "~/.paddlex/official_models/PP-OCRv6_medium_rec",
))
```

| Profile | Explicit detector / recognizer |
| --- | --- |
| `v5_server` (explicit compatibility option) | `PP-OCRv5_server_det` / `PP-OCRv5_server_rec` |
| `v6_tiny` | `PP-OCRv6_tiny_det` / `PP-OCRv6_tiny_rec` |
| `v6_small` | `PP-OCRv6_small_det` / `PP-OCRv6_small_rec` |
| `v6_medium` (production default) | `PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec` |

V6 needs the isolated runtime below. **Tiny is restricted experimental and blocked
by default for `media_type=MediaType.MANGA` (the default media type).** Parent
inspection of the official recognizer export found 6,904 dictionary characters
and **zero hiragana/katakana** for tiny, versus 18,708 characters including 86
hiragana and 94 katakana for small/medium. Tiny does not provide general Japanese
kana recognition and is not preferred for Japanese manga. Dictionary coverage
alone does not establish recognition quality for small/medium either.

Only an explicit boolean override permits an experimental tiny manga run:

```python
ocr = OCR(image_path, model_profile="v6_tiny", device="cpu",
          model_root="~/.paddlex/official_models", allow_restricted_paddle=True)
```

This emits a `RuntimeWarning` describing missing kana and records the override,
media type and restriction warning in `paddle_provenance`, page reports and raw
box model metadata (also on `PaddleInitializationError.provenance` if initialization
fails). `allow_restricted_paddle` defaults to `False` and must be a real boolean;
invalid flags/media/profile selections are rejected before either engine starts.
Explicit `v6_tiny` selection with `media_type=MediaType.OTHER` is allowed without
the manga override, still warns about absent kana and makes no general Japanese
support claim. The override is a facade permission, not a Paddle runtime parameter.
Switching a nonmanga tiny instance to `process_images(media_type=MediaType.MANGA)`
also requires the permission supplied at construction; it cannot bypass the guard.

All profiles require parent-run geometry/visual quality validation before adoption.
No OCR accuracy claim follows from detection counts, agreement, or timing.

`device=None` preserves legacy upstream automatic device selection. Explicit
selection accepts only `cpu` or one `gpu:N`; both initialized predictors must
report that exact device or construction fails, with no CPU retry. There is no
implicit GPU preference setting. Use `device="gpu:0"` to request GPU explicitly.

Every profile explicitly sets **`enable_mkldnn=False`** (oneDNN disabled) and
**`cpu_threads=1`** by default, including automatic/GPU device selection. Set
`OCR(..., cpu_threads=3)` to request another positive integer thread count;
booleans, zero, negative values and non-integers fail before either engine
initializes. The exact thread count and disabled oneDNN flag are retained in
profile settings and model provenance. This is an explicit configuration, with
no retry or runtime fallback.

**Known CPU incompatibility:** parent production probes of v6 small/medium with
upstream oneDNN settings initialized successfully but failed during prediction
with `ConvertPirAttribute2RuntimeAttribute ... pir::ArrayAttribute<pir::DoubleAttribute>`
at `onednn_instruction.cc:116`. The benchmark's eight profile/device configurations
(120 calls) passed with oneDNN disabled and one CPU thread; production now uses
those same two flags. The parent will rerun CPU/GPU facade and integrated proposal
checks. Historical resource evidence is preserved. Profile/facade source hashes
change for the medium default and validation fixes; the benchmark source hash
also changes to correct custom-thread identity metadata. Default effective
benchmark engine flags remain identical. Other benchmark flags remain as
documented below.

`model_root` expands `~` and appends the exact model names above; alternatively
provide a two-element **tuple** in `model_dirs`. These options are mutually
exclusive. Both exports must contain nonempty `inference.yml`, `inference.json`,
and `inference.pdiparams`, with matching `Global.model_name` in the YAML. Bad
profiles/devices/paths fail before Paddle or the optional manga engine initializes.
Omitting local paths retains legacy upstream weight resolution; the benchmark
always requires local paths and blocks lookup/download fallback.
After native initialization, the facade verifies requested model names and local
directories as well as explicit devices; a substitution fails instead of running.

V6 construction checks installed package metadata **before importing PaddleOCR**.
Only the pinned validated package pair above is accepted for v6; missing, older,
or other unvalidated versions raise a clear `PaddleInitializationError` naming
the required/installed versions and the explicit v5 option. No install, automatic
v5 substitution or retry occurs. Explicit v5 does not require the v6 package pins.
Importing the facade or resolving a profile alone does not import an OCR engine.

`ocr.paddle_provenance`, per-page `prediction_contract_reports[*].model_identity`,
and each raw box's `ocr_metadata["paddle_model"]` retain the requested profile,
device policy, exact constructor settings, installed package versions, local-path
selection and initialization status. `PaddleInitializationError.provenance`
retains failed initialization identity and chains the original error. These are
selection records, not independent weight authenticity or quality certification;
the benchmark adds local weight hashes and checks observed predictor identities.
Original pixel input, source IDs, raw polygons/text/scores and association repair
are preserved; profile selection does not alter annotation/grouping semantics.

### Parent-managed environment preparation

`requirements-paddle6.txt` pins `paddleocr==3.7.0` and `paddlex[ocr]==3.7.2`.
Keep the original `.venv`, `pyproject.toml`, and `uv.lock` intact. The parent can
copy the existing GPU environment to a **new** `.venv-paddle6` and upgrade only
that copy (the agent does not run these commands). For a fresh destination:

```sh
test ! -e .venv-paddle6 && cp -a .venv .venv-paddle6
.venv-paddle6/bin/python -m pip install -r requirements-paddle6.txt
```

Use the copied environment's **Python module invocation**, since copied entrypoint
shebangs may still reference the original environment. If `.venv-paddle6` already
exists, use the parent's prepared copy. Parent verifies Paddle GPU/package
compatibility and obtains the six official v6 exports and the v5 server pair
separately; this code never performs benchmark downloads. No v5 mobile profile
is introduced. The isolated environment is Git-ignored.

### Parent-only serial comparison CLI

The benchmark deliberately retains **`--profile v5_server` as its reference
default**, independently of the production medium default. Always pass `--profile`
when comparing models. A nondefault `--cpu-threads` value is reflected in both
effective constructor parameters and profile identity.

Run each profile/device in a **separate fresh process**, serially. For example,
after weights exist under `~/.paddlex/official_models/<exact-model-name>`:

```sh
.venv-paddle6/bin/python tools/paddle_model_benchmark.py \
  --run-local-ocr --profile v6_small --device gpu:0 \
  --model-root "$HOME/.paddlex/official_models" \
  --manifest test-results/detector-scale-parent-v2/detector-scale-results.json \
  --output test-results/paddle-v6-small-gpu0.json
```

Repeat separately with `--profile v5_server`, `v6_tiny`, or `v6_medium`, and
`--device cpu` or `gpu:0`, using a fresh output filename each time. The output's
parent directory must exist; exclusive creation refuses existing files. Replace
`--manifest ...` with `--images existing/page1.png existing/page2.png` for explicit
raster inputs. Default manifest input verifies all seven `plan.images` file/RGB
hashes and sizes. No page substitution or image manipulation is performed.

Defaults: one process-cold inference on the first page, then **two full serial
warm-process passes over all pages** (15 calls for the seven-page manifest).
`--warm-passes 2`, `--cpu-threads 1`, and `--sample-interval 0.2` are explicit
controls. The first encounter of later pages is labeled warm-process, not
per-image warm-cache. OS filesystem/GPU driver caches are not flushed. Constructor
timing excludes package import and initial timing synchronization; GPU inference
timings synchronize the requested device before and after each call. Runtime
uses recognition batch size 1, with HPI, MKLDNN, TensorRT and CINN disabled.
Upstream preprocessing/postprocessing defaults are retained and exported as the
merged public pipeline configuration and observed effective pipeline parameters;
this is not a reproduction claim for historical runs with unknown settings.

JSON retains package versions, requested/observed device and model names/paths,
exact constructor parameters, source/weight/image/manifest SHA256 hashes,
initialization and first-inference timing, every cold/warm per-page wall/CPU time,
process user/system seconds, and `resource.getrusage` lifetime peak RSS with units.
Each record includes the complete selective raw `dt_polys`, `rec_polys`, texts,
scores (and detector scores when present) in `contract.raw`, plus the existing
association contract report. Invalid detector contracts remain explicit evidence.
Failures preserve a failed JSON report and raise; there is no retry/substitution.

GPU telemetry samples `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory`
and filters to this process PID. It records per-GPU process memory, timestamps,
sampling interval, sampled peak and errors. Missing PID rows, unavailable tools,
unsupported/N/A memory and short-lived peaks are reported honestly; no whole
desktop allocation or zero-memory assumption is used. Sampling has overhead and
can miss peaks. Parent performs the actual CPU/GPU matrix and visual review before
adoption; the offline suite only uses fake exports and mocked engines.

Offline checks (no engine initialization):

```sh
.venv/bin/python -m unittest tests.ocr.test_paddle_profiles tests.tools.test_paddle_model_benchmark -v
.venv/bin/python -m unittest discover -s tests -q
.venv-paddle6/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/tralens tools tests
git diff --check
```

`OCR(image_path)` keeps the default manga grouping path. Use
`OCR(image_path, media_type=MediaType.OTHER)` for nonmanga input. Import
`MediaType`, `OCRStatus`, and `OCRReviewPolicy` from `tralens.ocr` alongside `OCR`.

## Source layout and imports

```text
src/tralens/
  ocr/          types.py, detection.py, grouping.py, review.py,
                recognition_units.py, transcription_selection.py, automatic_units.py,
                automatic_evidence.py, automatic_recognition.py, manga_adapter.py
  translation/  contracts.py, workflow.py, runtime.py, ocr_review.py, translator.py
  rendering/    automatic.py, best_effort.py, restoration.py, segmentation.py, typography.py, natural.py, preparation.py,
                review.py, styles.py, renderer.py
src/main.py     preview CLI and optional end-to-end rendering coordinator
tools/          standalone OCR/translation/rendering diagnostics
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
do not import an engine. No legacy source-path wrappers remain. `src/main.py`
connects the existing stages; there is no separate pipeline package.

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

## Immutable utterance translation previews (Phases 1–2; 3–4 foundation)

`tralens.translation` now exposes `Reading`, `Utterance`, `Style`,
`TranslationRequest`, `TranslationResult`, `automatic_units`, and
`PreviewTranslator`. These are preview contracts: no image erasure or rendering.
Frozen requests/results bind the exact page provenance, member IDs, main reading
order, raw evidence, selected candidate revision, alternative readings, target vs
context IDs, languages, glossary and style. Results bind the full request revision.
Keep the request manifest with each result; changing any bound input invalidates
reuse. Schema validity does **not** establish semantic translation quality.

`automatic_units(batch, candidates)` accepts the existing automatic OCR dataclasses
or their saved JSON representation. It snapshots inputs without changing OCR
statuses or requiring human review. The default reading is the ordered raw main
text; optional automatic candidates remain explicit alternatives, never silently
promoted to source truth. Select another reading explicitly with
`dataclasses.replace(unit, selected_revision=unit.readings[index].revision)`.
The adapter checks saved raw/candidate evidence consistency. It does not re-open
private images, rerun OCR, certify pixel provenance or infer source corrections.

Lexical uncertainty, tentative annotations, and generic unverified completeness
remain visible and permit automated previews. Known ownership, relationship,
orientation, isolation and blank-main faults produce an explicit `blocked`
disposition. Context cannot bypass those blocks. One utterance yields one stable
translation, not a copy distributed over every OCR box. Japanese → English is
the default; `Style(source_language=..., target_language=...)` is configurable.
No guessed speakers, genders, plot, or reconstructed source text is requested.

`PreviewTranslator(runtime, model="qwen3.8:latest", max_attempts=1, timeout=60)`
uses an injected provider-neutral `generate(model=..., payload=..., timeout=...)`
runtime. Construction performs no load/inference/network. `OllamaRuntime` is a
lazy standard-library HTTP implementation with explicit origin, no ambient proxy
or credential lookup, redirects disabled, bounded response size, deterministic
sampling, bounded output tokens, nonstreaming generation and thinking disabled.
Generalists receive a schema and must return a complete JSON object with exactly
one typed outcome per target ID; only explicit abstention permits empty translation.
Fences, prose, duplicate keys,
extra/missing/duplicate IDs, nonfinite constants and malformed values fail closed.
Results distinguish structural blocks, capability failures, invalid output,
provider errors, timeouts and cancellation; failed results carry no translations.
At most three attempts share one generation deadline. Cancellation is cooperative
between calls; a socket timeout cannot guarantee a hard wall-clock deadline or
stop generation already running on a remote server. Injected runtimes must honor
timeouts themselves. There is no background retry or provider fallback.

`GENERALIST` supports schema JSON and bounded text context, with images disabled.
Mode A is utterance text; B adds bounded same-page text; C adds a hash-bound
page presentation to the same bounded text condition, requiring an explicitly
image-capable profile and explicitly supplied original image paths.
Do not infer model image/context capabilities from its name.
`TRANSLATEGEMMA` uses the parent-verified official **single-user chat prompt**
from <https://ollama.com/library/translategemma> (Prompt Guide). Its instruction
ends with `Please translate the following Japanese text into English:` followed
by **three newline characters**, then the exact selected source text. There is
one user message, no system message, no JSON instruction, and no custom style or
context. `/api/chat` applies the installed Ollama Gemma3 user/model turn template;
no raw turn/control tokens are invented. Native CLI use is simply
`--profile translategemma --model translategemma:4b` (or `:12b`, `:27b`), without a
specification file. These three explicit tags are the parent-verified native
profile scope; a generalist model cannot use this specialist profile, and known
TranslateGemma tags cannot silently use the arbitrary-schema generalist profile.
Native specialist mode accepts A only, one utterance, no context/images/custom
glossary or style conditioning, and wraps one nonempty bounded plaintext response.
Vision metadata alone does not extend this tested specialist prompt contract.

The explicit name/code registry is Japanese/`ja`, English/`en`, French/`fr`,
German/`de`, Spanish/`es`, Korean/`ko`, Chinese/`zh`. Names and codes normalize to
those pairs; unknown names, typo codes and locale strings are rejected. The parent
verified Japanese→English execution; this registry is not a quality certification
for other language pairs. Template revision, official source, parent-verification
provenance and the cited local acquisition/pilot evidence are stored in native
plans and results. The agent did not fetch those sources or call a model.

Specialist plaintext does not expose uncertainty to the model through unsupported
extra instructions. Instead, result `execution_metadata_json` explicitly reports
`primary_reading_only`, `alternatives_supplied_to_model=false`, alternative reading
revisions, uncertain/unverified source interpretation, lack of context/style
conditioning, and `honorific_policy_not_enforced`. Full alternatives and raw
evidence remain bound in the request manifest. A fluent output is not evidence
that an ambiguous/slang source reading was interpreted correctly. No source
correction, semantic score or quality gain is inferred from previews.
No embedded Hugging Face runtime or dependencies are introduced.

The legacy `Translate(...).translate_groups(groups, image_path)` remains an
accepted-OCR-only compatibility API with lazy construction, injectable chat client,
strict full-JSON parsing, complete ID matching and no fabricated `Failed` output.
Its legacy per-box results preserve existing OCR provenance; unresolved/rejected
boxes cannot become renderable output. It does not consume `image_path` or send
images. New utterance previews use the immutable API above. An injected legacy
client controls its own timeout policy.

### Saved-evidence preflight and bounded parent-only Ollama execution

Use a parent-selected existing `automatic-proposals.json` exported by diagnostics.
For A/B, the tool reads the explicitly supplied evidence file and optional parent
selection/model/template records, checks saved binding
consistency, preserves caller selection order (or saved page/proposal order), and creates a **new**
output directory whose parent must already exist. It refuses overwrites. No model
runtime is constructed without `--run-local-model`. Input is capped at 16 MiB,
selection at 20 units, each compact serialized target at 32,768 characters, and context
at 8,000 compact serialized characters. Explicit `--unit-id` flags or a saved
`--selection` JSON ID list pin comparisons while preserving their order.
Errors consume the global attempt budget; every completed batch is appended and
flushed to `results.jsonl`. `requests.json` retains full immutable request evidence
and revisions, and `summary.json` records dispositions/attempt counts. Input page
errors are carried forward; preflight is not a completeness or accuracy claim.

### Ordered compact context and matched A/B/C diagnostics

The model-facing compiler sends only IDs, page positions, selected reading revision,
source reading alternatives/origins, lexical/structural uncertainty, completeness,
explicit unknown speaker/gender labels, and style. Full raw geometry, OCR metadata,
candidate blobs and source provenance remain in `requests.json`, outside the prompt.
The same compact compiler determines context budget usage; no alternatives are
silently dropped to fit. Nearby crop-centre distance prioritizes useful same-page
neighbours, with saved page order breaking ties. Selected neighbours are presented
in that saved order, including the target at its page position. The saved proposal
order is explicitly **unverified**, not a claim of definitive manga reading order.
An optional per-page `translation_page_order` record can supply `unit_ids` as an
exact proposal permutation and a descriptive `status`; both are bound to requests.

`--selection test-results/translation-parent-v1/selection.json --max-units 7`
reuses the parent's seven targets in identical order for A, B and C. Do not compare
the old three-target A run against a different seven-target condition as matched
evidence. `--model-record test-results/translation-parent-v1/model.json` reads the
explicit parent-saved identity/capability record, validates model/full-digest
consistency and records it; it performs no daemon lookup. The observed parent
digest is `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643`.

Parent matched-seven preflight and bounded B generation (set `MANIFEST` to the
same saved automatic-proposals evidence used by the parent A trial; fresh outputs):

```sh
.venv-paddle6/bin/python tools/translation_diagnostics.py \
  --manifest "$MANIFEST" --output test-results/translation-parent-v1/preflight-B7-v2 \
  --selection test-results/translation-parent-v1/selection.json --max-units 7 \
  --model-record test-results/translation-parent-v1/model.json \
  --mode B --max-context-chars 4000 --max-attempts 0

.venv-paddle6/bin/python tools/translation_diagnostics.py \
  --manifest "$MANIFEST" --output test-results/translation-parent-v1/qwen-B7-v2 \
  --selection test-results/translation-parent-v1/selection.json --max-units 7 \
  --model-record test-results/translation-parent-v1/model.json \
  --run-local-model --endpoint http://100.121.133.88:11434 \
  --mode B --max-context-chars 4000 --max-attempts 8 --timeout 90 --preload
```

Repeat with **new output names** and `--mode A` for the matched A7 control. For C,
use `--mode C --vision-capable --image-max-side 1536` and all explicit
`--image 'PAGE_ID=ORIGINAL_PATH'` mappings described below; model-free C preflight
uses `--max-attempts 0` without live/preload flags. Keep B/C context budgets equal
to isolate the image condition. Seven one-call batches plus preload are at most
eight calls; errors count. The parent runs these commands, not the coding agent.

C additionally requires `--vision-capable` and explicit mappings such as
`--image 'PAGE_ID=/absolute/path/to/original.png'`, one per selected safe page.
No image path is implicitly opened from OCR metadata. Original file and RGB pixel
SHA256 must match the saved page; structurally blocked units do not trigger image
reads or provider calls. Raw-main geometry must be nondegenerate and inside the
original image. Only a copied RGB presentation is optionally resized, defaulting
to a 1536-pixel longest side with LANCZOS. `--image-max-side 0` disables resizing
(4096-pixel maximum side). Original inputs remain byte-identical. Original bytes
are limited to 32 MiB/40 million pixels; presentations to 8 MiB. The exact PNG,
transformation/size evidence and original/presentation hashes are saved in the
new output directory and bound to the request. API execution rechecks the supplied
presentation hash. Vision capability is a parent assertion; it is also checked
against `--model-record` when supplied. Images do not relax structural safety.

Runtime `generate` may return text or immutable `Generation(text, response_json)`.
Ollama returns the latter, preserving load/total/prompt-eval/eval nanosecond durations,
prompt/eval token counts, completion reason and raw response. Diagnostics retains
each response (also malformed/incomplete/late responses) in exclusive sidecars,
with full canonical response SHA256, original/retained byte counts, truncation flag
and extracted metadata in `results.jsonl`. Default retention is 1 MiB per attempt;
`--max-response-bytes` accepts 1024..4194304 bytes. Truncated sidecars are explicitly
`.txt` prefixes, not valid full JSON claims. Preload raw evidence and timings are
separate. Timeouts without a response record that absence and still spend a call.
The global attempt bound includes preload and errors. Socket/remote-cancel
limitations remain unchanged. `semantic_quality_assessed` is always false: schema
successes and token/timing evidence are not semantic quality scores.

### Optional custom raw specialist encoder

The standard native TranslateGemma chat profile above needs **no template file**.
For an exceptional parent-verified raw encoder, the same specialist CLI accepts
an explicit JSON file via `--specialist-template /absolute/path/to/verified.json`
with these fields:

- `schema_version`: `1`; `encoding`: `literal_substitution_v1`.
- Exact `model` and full `model_digest` matching the run.
- Nonempty `verified_by` and `verification_source` recording parent verification.
- `template`: the verified raw prompt, using `${source_lang_code}`,
  `${target_lang_code}`, and exactly one `${text}` substitution.
- `template_sha256`: SHA256 of the exact UTF-8 template string.

Only literal substitution is supported, with no recursive interpretation of source
text and no execution of Python, Go/Jinja templates, or arbitrary code. If the
official template cannot be represented by that method, the CLI rejects it;
the existing injected Python runtime encoder remains the source-level extension.
Identity, hash and placeholder mismatches fail before model loading. No actual
custom raw template or model usability is claimed by the synthetic tests. This
explicit override switches to the separately recorded raw execution path; it does
not replace the standard installed-template chat path by default.

### Comparable bounded generation controls

Both generalists and specialists accept `--num-ctx` (512..131072, default 8192),
`--num-predict` (1..8192, default 2048), and `--keep-alive` (integer seconds
0..3600, default 300). These controls apply to preload and generation and are
recorded in the plan and result execution metadata. Temperature is fixed at 0.
`--keep-alive 0` unloads after the request and changes subsequent load costs;
keep it identical between conditions and do not treat an arbitrary first call as
proven cold. Tokenizer differences mean equal token budgets are controlled
settings, not equal text lengths. Generalists retain their own structured source
and schema prompt; the specialist prompt is never forced onto them.

`--max-output-chars` (1..131072, default 32768) independently bounds accepted
translation text. Empty, nontext, oversized or truncated specialist outputs fail
without renderable translations. Malformed response types retain raw diagnostic
evidence. Raw HTTP response and retention bounds described above also apply.
Using the same controls still does not make schema/plaintext acceptance a semantic
quality comparison; ground truth and source ambiguity require independent assessment.

Parent specialist-seven command (set `MANIFEST`, `MODEL_DIGEST` and `MODEL_DETAILS`
from the parent's verified saved evidence; use a fresh output directory):

```sh
.venv-paddle6/bin/python tools/translation_diagnostics.py \
  --manifest "$MANIFEST" --output test-results/translation-gemma4b-A7-parent-v3 \
  --selection test-results/translation-parent-v1/selection.json --max-units 7 \
  --profile translategemma --model translategemma:4b \
  --model-digest "$MODEL_DIGEST" --model-details "$MODEL_DETAILS" \
  --mode A --num-ctx 8192 --num-predict 2048 --keep-alive 300 \
  --max-output-chars 32768 --max-attempts 8 --timeout 90 --preload \
  --run-local-model --endpoint http://100.121.133.88:11434
```

For model-free preflight, use a fresh output name, `--max-attempts 0`, and omit
`--run-local-model --preload`. Repeat separately for `:12b` and `:27b` with each
model's own verified digest/details. At most seven generations plus preload are
requested; errors spend the budget. Parent comparison against generalist controls
uses the same explicit numeric settings and target selection, retaining the
generalist profile/schema prompt. Earlier A7/B7 previews alone established no clear
semantic quality gain, and are not model-generated ground truth.

Parent model-free preflight (replace only the saved manifest path; use fresh output
names for each run):

```sh
.venv-paddle6/bin/python tools/translation_diagnostics.py \
  --manifest /absolute/path/to/automatic-proposals.json \
  --output test-results/translation-preflight-A-01 \
  --mode A --max-units 3 --max-attempts 0
```

Parent-only bounded live command, after verifying the endpoint, installed model,
full digest, model/template metadata and selected evidence:

```sh
.venv-paddle6/bin/python tools/translation_diagnostics.py \
  --manifest /absolute/path/to/automatic-proposals.json \
  --output test-results/translation-qwen38-A-01 \
  --run-local-model --endpoint http://100.121.133.88:11434 \
  --model qwen3.8:latest --model-digest 22130167c4c2 \
  --model-details 'parent-reported 27.3B Q4_K_M vision/thinking; verify before execution' \
  --mode A --max-units 3 --max-attempts 4 --timeout 90 --preload
```

The digest/details are explicitly recorded as **parent assertions**, not daemon
verification. Replace the reported short digest with the parent-verified full
digest when available. Preload is a separate explicit empty generation probe,
consumes one attempt (including on error), and records wall/load timing separately
from translation generation. It is a cold-load measurement only if the parent has
independently established a cold initial state; this tool does not unload models.
Three one-unit batches plus that probe use at most four calls and no retries.
For B use a new output directory, `--mode B --max-context-chars 4000` and the same
explicit `--unit-id` selections from preflight. A larger context is not assumed
better. Ground truth must be independent of model outputs. TranslateGemma
4B/12B/27B, Qwen3.8 and smaller-control comparisons remain future parent-verified
experiments; the historical `qwen3-vl:32b` default is not assumed installed.

Focused offline checks:

```sh
.venv/bin/python -m unittest tests.translation.test_translator tests.translation.test_previews tests.translation.test_specialist_chat tests.translation.test_hardening tests.tools.test_translation_diagnostics tests.tools.test_translation_conditions -v
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/tralens/translation tests/translation tests/tools/test_translation_diagnostics.py tests/tools/test_translation_conditions.py tools/translation_diagnostics.py
git diff --check
```

### Final hardening: semantic dispositions and release-readiness

The generalist response contract is now **`typed_unit_disposition_v2`**. Each
`translations` item requires `id`, `translation`, `disposition`, `reasons`, and
`source_interpretation`. Disposition is `preview`, `review`, or `abstained`;
interpretation is `unverified`, `ambiguous`, or `unreadable`. There is no
`confident`, `accepted`, or quality-score label. Review/abstention needs nonblank
reasons; abstention requires exactly empty translation, and unreadable sources
cannot carry translated text. At most 12 model reasons of 512 characters are
accepted. Full-JSON, duplicate-key, finite-number and exact target-ID checks still
apply. An older text-only schema response is historical evidence, **not silently
upgraded** to this contract.

Immutable `UnitOutcome` records preserve every target, including abstentions, and
bind each to its exact selected reading revision. Partial batches retain the
explicit abstained outcomes while exposing translation pairs only for units with
text. All-abstained batches return `status="abstained"`, not an empty successful
preview or provider failure. Known lexical uncertainty/differing readings force
an application-visible `review` outcome and `review_needed=true`, even if the
model claims a plain preview. Model interpretations/reasons are unverified model
reports, not independent facts. Generic unverified completeness alone does not
block a call or impose a mandatory review step. Review flags are informational;
all results remain preview-only, and this code never erases or renders images.

Specialist output remains exact-prompt plaintext, with a visible `review` outcome
and `specialist_reduced_capability` reason. No uncertainty instructions are added
to its unsupported prompt. The raw source alternatives remain in the request;
plaintext fluency does not resolve ambiguous OCR. Blank selected/main readings
remain explicit blocked units with reasons and zero calls rather than disappearing.
The legacy per-box compatibility path now also marks copied outputs as unverified
translation previews in metadata; it retains OCR status and source provenance.

`result.assert_current(request)` rejects reuse after any bound source, candidate,
style, order, context or image revision changes. Hash binding establishes saved
input consistency, not correctness or authenticity. Model digests/details remain
explicit **parent assertions**, with no daemon/weight authentication claim.
The request's `utterance_preview_v2` contract version also participates in its
revision, so historical text-only requests/results do not share the new revision.

Endpoint validation now enforces **address scope**, not merely a hostname label:
HTTP origins must use literal loopback, RFC1918 private IPv4, shared-address
`100.64.0.0/10` overlay, or IPv6 ULA addresses. `localhost` is pinned to
`127.0.0.1`; arbitrary DNS names, public/link-local/unspecified addresses,
credentials, paths and redirects are rejected. This is not proof that a private
server is trustworthy or physically on the same host. The normalized origin and
the absence of server authentication are recorded before execution. The approved
`http://100.121.133.88:11434` origin satisfies the overlay scope rule.

Direct immutable requests also have a 65,536-character compact-source ceiling,
20-target/84-unit ceiling and pre-decode image-encoding bound. The legacy API is
bounded to 20 groups/84 boxes per group and 32,768 source/response characters.
Explicit evidence files are read with `limit+1` bounded reads, not just a prior
file-size check. Conflicting duplicate ownership records and nonboolean crop
safety flags are rejected. Existing original-image/pixel/presentation hashes,
frame/dimension checks and safe-source geometry checks remain enforced. Images
and file paths from source/model text are never implicitly opened.

Each retry can retain its own raw evidence. If an evidence sink fails after a
provider call, the result is `failed/evidence_retention` with that attempt counted,
no output, and no retry; diagnostics appends that batch and charges the global
budget. If storage cannot append the journal itself, execution fails rather than
claiming successful persistence; previously flushed records remain. Results and
summary distinguish per-unit review/abstention from provider/timeout/budget errors.

**Readiness assessment:** Phases 1–2 contracts and the bounded Phase 3–4 preview/
diagnostics paths are ready for parent final two-environment/source-only and live
acceptance checks. They are not a validated autonomous translation/rendering
pipeline. There is no independent semantic reference and no automatic model
promotion. The parent-reported six-model × fourteen application previews are
transport/schema/plaintext evidence, not 84 verified translations. Earlier A7/B7
schema successes likewise establish no semantic quality gain.

Experimental comparison profiles (separate from the user-selected TG27B workflow default):

- **Qwen3.8 generalist schema:** structured, uncertainty-capable A reference;
  evaluate B/C only against matched A targets. Context/vision quality gains remain
  unproven. The new disposition contract needs fresh parent live verification.
- **TranslateGemma 12B/27B native chat:** comparison candidates with mandatory
  visible reduced-capability review flags, not automatically promoted by scores. Parent notes
  suggest different pragmatic/idiomatic behavior, but noisy OCR defeated all
  models and there is no independently scored winner.
- **TranslateGemma 4B:** retain as a low-cost control; parent-observed major meaning
  errors do not support recommendation for translation quality.
- **Qwen3:4B residency/plain-output trial:** capability-limited under the tested
  runtime/template. Parent reported three of three `done_reason="length"` outputs
  with reasoning-like content despite `think=false`. The application rejects
  length-terminated responses and retains their evidence. These are failures,
  not translations; no guessed thinking/template workaround is implemented.

Repeated identical-prompt warm measurements can exploit prefix caching and are
cached **lower bounds**, not real multi-page throughput. Keep load, prompt-eval,
generation and wall timings separate; compare matched distinct targets with
identical controls and explicitly recorded residency. Human/independent reference
evaluation must address source ambiguity, fidelity, omissions and pragmatics,
not fluency or model-generated ground truth. Embedded HF remains deferred.

## External clean-source quality comparison helper

`tools/translation_quality.py` runs a bounded, **text-only Japanese→English**
comparison using the current immutable translation API and existing prompts. It
does not tune prompts, select a winner, change defaults, acquire a corpus, or use
OCR/image fixtures. The parent supplies a licensed, independently human-referenced
JSON list at `test-results/translation-quality-corpus-v1/cases.json`. Every item
has exactly `id`, `source`, `reference`, `split`, and `origin` string fields;
IDs must be unique. Origin/license/reference independence are parent assertions,
not certifications made by the helper. Keep sources short enough for 4096 context
tokens; the helper does not estimate model-specific tokenization.

Only source text and an opaque case key enter a `Reading`/`Utterance` request.
`page_json` explicitly says `synthetic_text_only_benchmark_not_OCR` and names the
benchmark; no synthetic polygon, image hash or OCR evidence is manufactured.
**References, split and origin never enter model prompts or requests.** References
are used only in the separate reviewer artifact or optional post-generation
metric. Changing a reference does not change the source request/prompt, but does
change the frozen corpus-file hash used to validate paired comparisons.

The helper selects the first `--max-cases` entries in **file order**, default/hard
cap **150**. It stores the exact requests, model-facing payloads and their hashes
in `run.json`. Same-profile model runs use identical prompts/options; generalists
and TranslateGemma retain their different supported prompt formats. All runs in
a comparison must have identical selected IDs/order/counts, corpus bytes and
generation options. Frozen source/prompt/result bindings are revalidated. Missing,
duplicate, reordered or stale records fail comparison rather than shrinking the
denominator.

Default execution is model-free. `--run-local-model` enables one serial attempt per
case, **no retries or preload calls**, with timeout fixed at **120 seconds**, context
4096, default `--num-predict 1024` and `--keep-alive 60` seconds. Prediction and
residency overrides obey the existing bounded runtime ranges (1..8192 and 0..3600).
Model identity must match an explicit parent-saved `--model-record` with `model`
and full 64-character `digest`. This records a parent assertion, not a daemon or
weight verification. Existing private/overlay endpoint restrictions apply.

Each completed case is appended and flushed to `results.jsonl`, including status,
typed outcomes, source/request/prompt hashes, model/digest/options, wall/generation
durations and raw-response metadata. Raw evidence uses the existing observer and
`retain_response`, retaining up to the runtime's full 4 MiB bound per attempt
(at most 600 MiB for 150 calls); truncation, if any, is explicitly labeled.
Provider failures, structural blocks, abstentions and preflight cases have blank
candidates and remain in every selected-case denominator. A failed case is not
retried or silently discarded. Storage failures stop rather than claim persistence;
previously flushed results remain available. Output directories are exclusively
created and their parent must already exist. Corpus input is capped at 16 MiB,
model metadata at 64 KiB, and saved comparison JSONL at 64 MiB per run.

### Parent commands

Model-free preflight (the named parent model record already exists; output must
be new):

```sh
.venv-paddle6/bin/python tools/translation_quality.py \
  --cases test-results/translation-quality-corpus-v1/cases.json \
  --output test-results/translation-quality-qwen-preflight-v1 \
  --model qwen3.8:latest \
  --model-record test-results/translation-parent-v1/model.json \
  --profile generalist --max-cases 150 --num-predict 1024 --keep-alive 60
```

Parent-only live run:

```sh
.venv-paddle6/bin/python tools/translation_quality.py \
  --cases test-results/translation-quality-corpus-v1/cases.json \
  --output test-results/translation-quality-qwen-v1 \
  --model qwen3.8:latest \
  --model-record test-results/translation-parent-v1/model.json \
  --profile generalist --endpoint http://100.121.133.88:11434 \
  --run-local-model --max-cases 150 --num-predict 1024 --keep-alive 60
```

Run each additional model separately with its own fresh output and verified model
record. Standard specialists use `--profile translategemma --model translategemma:12b`
(or verified `:4b`/`:27b`) and their own records. Keep corpus/order/options fixed.
No corpus references or generated reference substitutes are accepted as context.

Offline paired review after the parent has produced both named runs:

```sh
.venv-paddle6/bin/python tools/translation_quality.py \
  --cases test-results/translation-quality-corpus-v1/cases.json \
  --output test-results/translation-quality-paired-review-v1 --max-cases 150 \
  --compare test-results/translation-quality-qwen-v1 test-results/translation-quality-gemma12-v1
```

Both single-run and comparison outputs contain **`review.json`** with per-case
source, human reference, split/origin, blank failures and opaque candidate labels;
candidate display order is shuffled. Model identities/paths are stored separately
in **`model-mapping.json`**. Distribute only `review.json` to a blinded reviewer,
without run directory names, manifests, summaries or mapping. Tool metadata does
not reveal identities there, but candidate text itself can self-identify a model;
this is reviewer blinding, not an anonymity guarantee. Review fields invite
fidelity, omissions and pragmatics notes. Failures stay blank rather than receiving
model/provider error text as a candidate translation.

`--score` optionally calls **already-installed** `sacrebleu.corpus_bleu` after
generation or during offline comparison. It uses every selected case, including
blank failures, and records package version/default settings and denominator.
Without that package it reports **metric unavailable** and never installs it;
without `--score` it only checks availability through `importlib`. Preflight-only
outputs are not scored. No custom overlap metric is mislabeled chrF. Neither
BLEU, schema acceptance nor plaintext fluency constitutes a semantic-quality or
accuracy claim, and no metric triggers model promotion. Serial wall/load/eval
evidence is not a controlled cold-load test or real-page throughput benchmark.

Source-only verification:

```sh
.venv/bin/python -m unittest tests.tools.test_translation_quality -v
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q tools/translation_quality.py tests/tools/test_translation_quality.py
```

## Normal image-to-unit preview workflow

Without `--render`, `src/main.py` is the normal **preview-only** image entrypoint. It uses the current
OCR facade with `propose_units=True`, `complementary_ownership=True`, and
`regroup_companions=True`, then snapshots automatic utterances and optionally
translates them. It never imports a renderer/erasure implementation, writes on the
input image, transforms translations into OCR boxes, or imports diagnostic tools.
Importing the entrypoint or requesting `--help` loads no OCR/model engines or
provider clients. Image processing requires `--run-local-ocr`; translation requires
the separate `--run-local-model` flag. **Daniel selected `translategemma:27b` with
the native `translategemma` profile as this workflow's default pair**, in both
`src/main.py` and `run_preview`. Omitting model/profile with model opt-in uses that
pair. OCR-only mode still constructs no model runtime and makes zero model calls.
There is no implicit model acquisition, download or fallback if the configured tag
is unavailable. This is an explicit user choice, not an inferred quality score.

### Parent-ready CPU OCR preview

Set `IMAGE` to an existing single raster image and `MODEL_ROOT` to the existing
local export root. The root must contain complete, identity-matching
`PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec` exports (nonempty `inference.yml`,
`inference.json`, and `inference.pdiparams`). The workflow pins the production
medium profile, explicit device, one CPU thread and disabled oneDNN through the
existing facade. It validates paths/arguments before constructing an engine; there
is no download, missing-weight lookup, profile/device retry or fallback. Use the
parent-prepared `.venv-paddle6` with the validated Paddle runtime.

```sh
IMAGE=/absolute/path/to/page.png
MODEL_ROOT="$HOME/.paddlex/official_models"

.venv-paddle6/bin/python src/main.py \
  --image "$IMAGE" --output test-results/workflow-ocr-parent-v1 \
  --run-local-ocr --device cpu --model-root "$MODEL_ROOT"
```

This performs OCR/proposals only: zero translation calls and no translation runtime
construction. `preview.json` still contains original OCR polygons/text/confidence,
current status/selection metadata, prediction reports, model provenance, complete
proposal evidence, candidates if present, all proposed units, immutable requests,
source-binding hashes and per-unit dispositions. Unassigned detections remain
visible in raw/proposal evidence; they are not guessed into utterances. Human
checking of every readable unit is not a prerequisite for automatic previews.

### Parent-ready default TG27B preview with at most two translation calls

Use a fresh output directory. This second command performs fresh CPU OCR before
translation; it does not silently reuse a previous OCR snapshot.

```sh
.venv-paddle6/bin/python src/main.py \
  --image "$IMAGE" --output test-results/workflow-tg27-default-parent-v1 \
  --run-local-ocr --device cpu --model-root "$MODEL_ROOT" \
  --run-local-model \
  --endpoint http://100.121.133.88:11434 \
  --max-attempts 2 --timeout 120 --keep-alive 60 --num-predict 1024
```

The default specialist uses the documented single-user text prompt and installed
Ollama chat template. Overrides are explicit:

- TG12: add `--model translategemma:12b` (the default native profile still applies),
  optionally spelling out `--profile translategemma`.
- Qwen generalist: add **both** `--model qwen3.8:latest --profile generalist`;
  the existing strict typed schema applies.
- `run_preview(..., model="qwen3.8:latest", profile="generalist")` and
  `run_preview(..., model="translategemma:12b")` provide the same API overrides.

Specifying Qwen without its generalist profile, or a generic profile with the
inherited TG27B model, is rejected before OCR construction. No compatible model or
profile is guessed. The legacy per-box `Translate` API retains its generalist
default/behavior and is not redirected to a specialist. This
workflow sends **utterance text only**, not page images/context. A model tag is
configuration, not weight/digest authentication or a quality recommendation.
The existing explicit private/overlay endpoint restriction and server-authentication
limitations apply. These commands are for the parent; the coding agent exercises
only fake OCR/providers in offline tests.

### Artifacts, budgets and errors

- `--output` must be a **new directory** with an existing parent. Existing files,
  directories and symlinks are rejected before engine creation. The output directory
  is private (`0700`); `preview.json` is published exclusively, then atomically
  replaced only as this run's own checkpoint. JSON snapshots have a 64 MiB cap.
- Original input is capped at 32 MiB, one frame and 40 million pixels. Original
  file SHA256, RGB pixel SHA256 and size are bound to the page/proposal evidence.
  Bytes are rechecked after OCR, before/after model calls and on completion. A
  changed, removed or unreadable image, or mismatched raw/proposal snapshot, fails closed. Any retained
  translations on a changed file are explicitly marked `stale_source` with
  `binding_current=false`, not presented as current-source output.
- Raw OCR and **every proposed unit** are checkpointed before model execution.
  Unit records retain membership/order, all alternatives, selected-reading and
  full request revisions, typed result/outcomes, blocking reasons and raw evidence.
  `ocr_only`, `not_selected`, `budget_exhausted`, `blocked`, failed/abstained and
  interrupted/not-executed units remain visible. Generic uncertainty does not cause
  blanket blocking; genuine structural/blank-source faults remain explicit.
- Default `--max-attempts 2` is a **global 0..64 provider-call budget**, with one
  attempt per selected unit and no retries/preload calls. Errors and interrupted
  dispatched calls spend the budget. Blocks and capability/input rejection that
  never dispatch a call do not. Strict inputs run before uncertain coverage inputs;
  repeated `--unit-id ID` flags optionally select units and their call order while
  keeping unchosen units in the artifact. No source correction or candidate
  promotion is performed. [Coverage-first translation](docs/translation-coverage.md)
  derives separately bound text-region inputs without modifying original OCR units.
  With the optional fragment mode below, strict inputs run first; automatic
  fragments use stable script-hint tiers, while explicit selection retains its order.
- Context is fixed at 4096; default `--num-predict 1024` and `--keep-alive 60`
  use existing runtime bounds. `--timeout` is 0..300 seconds per utterance (default
  120), not an OCR timeout or hard remote-generation kill. Ctrl-C preserves a
  reviewable interrupted checkpoint when Python can handle the interruption;
  remote generation may continue. Abrupt process/storage failure can leave the
  last valid `running` checkpoint and partial journal, not a completion claim.
- `results.jsonl` is exclusively created, appended, flushed and fsynced per visited
  nonblocked selected unit. Raw-response sidecars retain at most 1 MiB per attempt,
  with full response SHA256/size, retained size, truncation flag and available
  load/eval/token metadata. Truncated prefixes are `.txt`, not full-JSON claims.
  Model/provider errors cannot become translated `Failed` text. Evidence-sink
  failures use the existing typed failure path and still spend dispatched calls.
- The workflow snapshots OCR evidence and closes the facade/underlying Paddle
  engine in a `finally` path **before translation**, including prediction failures.
  Injected translation runtimes remain caller-owned; the built-in runtime closes
  each HTTP response. Ollama residency follows the explicit keep-alive setting.
- Exit code **0** means the bounded pass completed (including OCR-only, blocked,
  abstained or budget-limited units); **1** means unit translation failures,
  interruption or a failed pipeline, with a reviewable artifact when storage permits;
  **2** means rejected arguments/paths before engine creation. Inspect unit outcomes,
  not just the exit code. Preview completion is not semantic quality certification.

Source-only checks for this workflow:

```sh
.venv/bin/python -m unittest tests.translation.test_workflow -v
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/main.py src/tralens/translation/workflow.py tests/translation/test_workflow.py
```

Production rendering/erasure, OCR algorithms/adapters, dependency manifests,
provider prompts and legacy `Translate` behavior/defaults are outside this
workflow change. The TG27B/native workflow default is user-selected; it does not
establish semantic quality or enable automatic model promotion.

## Explicit unverified text-only fragment previews

The image workflow's **strict default is unchanged**. An explicit
`--allow-fragment-previews` (API `allow_fragment_previews=True`) additionally permits
derived raw-text fragment inputs for a narrow subset of blocked sources. It does
not make the original utterance strictly eligible or change its blockers.
OCR and model opt-ins remain separate; this flag alone enables neither engine.
The API requires an actual boolean, not `1`, a string or another truthy value.

Policy `unverified_raw_singleton_fragment_v1` supports **singletons only**:

- Exactly one original main/source ID, with consistent singleton order and full
  raw-detection/page/reading provenance. The selected reading must be the original
  raw main reading; no alternative is silently selected or corrected.
- Only `unknown_region_ownership`, `unsafe_crop` and `crop_isolation_unresolved`
  may prevent strict eligibility. Other structural faults, unknown orientation,
  mixed ownership, annotation-role/relationship conflicts, unknown structural
  reasons, missing/inconsistent raw evidence and unresolved source association
  retain the block. Explicit rejected/blank raw readings are not dispatched.
- Combined units are **not split in this first policy**. Their existing contract
  does not certify independent fragment role/order; they retain the explicit
  `requires_independent_singleton_raw_input` reason. No merge or reading order is
  inferred, and no manual-review requirement is added to normal readable previews.
- Raw confidence is retained as evidence, not used as a readability threshold or
  proof. Nonempty garbled text can still be unrecoverable. Generalists can abstain;
  specialist plaintext may refuse or mistranslate and is not semantically certified.

### Distinct contracts and source binding

`FragmentInput` embeds the immutable **original `Utterance`**, original-unit
revision, source reading revision, source IDs and policy revision. All original
readings/candidates, ownership/crop evidence and blocked reasons remain intact.
Its new fragment identifier and derived-reading revision are distinct from the
original unit/reading; neither pretends to be a new OCR detection.

`FragmentRequest` is not `TranslationRequest`. Strict `translate()` and
`make_payload()` reject fragment requests; the original strict request still
returns blocked. Execution uses `translate_fragment(...,
allow_fragment_previews=True)` and a separate `FragmentResult`:

```python
from tralens.translation import fragment_plan, FragmentRequest

inputs, rejected = fragment_plan(original_unit, allow_fragment_previews=True)
if inputs:
    request = FragmentRequest(inputs[0])
    result = translator.translate_fragment(request, allow_fragment_previews=True)
    result.assert_current(request, original_unit)
```

Fragment results expose **`fragments`**, not ordinary `translations`; their status
is `fragment_preview`, `fragment_abstained`, `fragment_failed`, or another explicit
fragment-prefixed disposition. `output_kind="unverified_text_fragment"`,
`review_needed=true`, `preview_only=true` and `complete_utterance_claimed=false`
are mandatory/recorded. Source readability is not verified by the wrapper. Model
outputs and reasons remain unverified data. Replacing original evidence, selected
reading, source IDs or policy invalidates the derived input/result binding.

TG27 remains the workflow default. Its **native single-user prompt is unchanged**;
uncertainty and reduced capability are recorded outside the unsupported prompt.
No page images/context, recognition cropping, rendering or erasure permission is
granted. Even an image-capable profile cannot send images through the fragment API.

### Workflow artifacts and shared budget

Opt-in output uses workflow schema2. Every original unit/request remains present;
an original blocked unit remains `status="blocked"` with its ordinary `result=null`.
Derived requests/results live only in its `fragment_previews` list, alongside
`fragment_rejection_reasons`. Schema1/strict mode derives nothing.

`strict_eligible_units`, `strict_preview_count`, `fragment_candidate_inputs`,
`fragment_preview_count`, `fragment_status_counts`, `strict_attempts` and
`fragment_attempts` are separate. A fragment must never be counted as a complete
utterance translation. `attempts` is the single global budget across both paths.
Selected strict inputs run first in selection/saved order. For **automatic**
selection (no `--unit-id` / empty `selected_ids`), singleton fragments containing
any Kana/Han letter run first, stably within the two tiers. There is no minimum
length: short kana, kanji-only text and Japanese-script SFX are preserved.
NFKC is used only in a temporary inspection view (including halfwidth kana);
raw text, fullwidth Latin, alternatives, hashes and model prompts are unchanged.
Han is shared with other languages; this hint certifies neither Japanese language,
readability, dialogue role nor quality. Kana-containing noise can rank first;
Latin/marker inputs remain candidates, not discarded sources.

Nonempty explicit `selected_ids` bypass script reordering and preserve requested
order within the strict-first contract. `--unit-id` continues to select original proposal IDs,
not derived fragment IDs; unchosen/budget-skipped derived inputs remain visible.

Scheduling policy `strict_then_kana_han_stable_v1` is recorded in settings and each
fragment's `scheduling` data, with reason, tier, Unicode version and planned position.
`input_schedule` is checkpointed before provider entry; budget-skips retain those
reasons. Admission policy/provenance and original blocked reasons are unchanged.

Both paths retain bounded raw responses, source/file drift checks, cancellation,
exclusive output and atomic checkpoints. Fragments get exactly one attempt even
if a standalone translator permits explicit strict retries; workflow retries and
preload calls remain disabled. Errors and interrupted dispatched fragment calls
consume the same global budget. OCR closes before either translation path.

Parent-ready command, using the existing image/local medium exports and a **new**
output directory (the agent does not run live OCR/models):

```sh
.venv-paddle6/bin/python src/main.py \
  --image "$IMAGE" --output test-results/workflow-fragments-priority-parent-v1 \
  --run-local-ocr --device cpu --model-root "$MODEL_ROOT" \
  --run-local-model --allow-fragment-previews \
  --endpoint http://100.121.133.88:11434 \
  --max-attempts 20 --timeout 120 --keep-alive 60 --num-predict 1024
```

Omit `--run-local-model` for an OCR-only candidate inventory with zero translation
calls; omit `--allow-fragment-previews` for the original strict behavior. The flag
waives only text-input permission for the specified policy, not source ownership,
crop safety, source certainty, utterance completeness or downstream approval.

### Frozen current-page input comparison

`test-results/translation-fragment-preview-v1/strict-vs-fragment.json` binds the
accepted current page and lists every original unit, unchanged blockers, derived
source IDs/revisions and rejection reasons. Strict eligibility stays **8 of29**;
the same **21 original units remain blocked**. The opt-in policy permits15
additional unverified singleton inputs, source IDs:

`0,1,2,4,7,9,11,12,23,24,29,33,34,35,41`.

Singletons3/32 remain blocked by orientation/structural faults. Combined groups
14/19/21/22,13/18/20,25/26/27 and28/30 are not derived. This is **23 possible
inputs, not23 complete/correct translations**. With a20-call cap, strict attempts
take priority and remaining fragment candidates can be budget-exhausted. No model
was called to produce this comparison and no coverage/accuracy claim follows.

Scheduling-only replay of the saved live page is in
`test-results/translation-fragment-priority-v1/before-after.json`. Strict inputs
stay8 and fragments stay15. At the same20-call budget, projected fragment slots
gain34/35/41 (including `ねえ`) and budget-skip11/23/33 instead. This is neither a
model rerun nor a quality gain claim; old outputs are not reassigned to new inputs.

Focused verification:

```sh
.venv/bin/python -m unittest tests.translation.test_fragments -v
.venv/bin/python -m unittest tests.translation.test_fragment_priority -v
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/main.py src/tralens/translation tests/translation/test_fragments.py
```

## Optional source-bound visual OCR review (off by default)

`--review-ocr` adds a **sequential** optional phase: local OCR/proposals → visual
reread proposals → reviewer release → normal TG27B/text previews. Without the flag
there is no reviewer construction/call, image-view generation or release, and the
original paths/prompts are retained. OCR-only plus `--review-ocr` is rejected before
engine initialization: image review requires the separate `--run-local-model` opt-in.

The default reviewer is the parent-installed **`qwen3.8:latest`**, configurable via
`--review-model`. The existing explicit private/overlay `--endpoint` is used for both
models. A configurable reviewer must support vision/schema JSON; unsupported
providers fail visibly without model fallback/download. Tags are configuration,
not weight/digest authentication. Same reviewer/translator tags (including implicit
`:latest`) are rejected before OCR rather than silently unloading the translator.

`--max-review-attempts 4` is an independent **0..20 generation budget**, with one
attempt per target, no retry or preload. Review options are fixed at8192 context,
1024 prediction tokens, temperature0,60-second residency and `think=false` on the
built-in chat adapter. `--review-timeout` defaults90 and is capped120 seconds.
These are requested settings; the parent smoke must verify actual installed model
behavior. Raw response/load/eval/token evidence is retained, including truncation
and invalid-output failures. Existing1 MiB retention prefixes are explicitly marked.

### Attribution and immutable suggestions

This bounded version reviews **attributable singletons only**. Combined main/order
or annotation/role conflicts are skipped, while their original translation path
remains intact. A blocked source is reviewable for alternate translation only when
`--allow-fragment-previews` is also explicitly enabled and its existing singleton
fragment policy permits it. Review never clears a structural block or enables
fragment mode implicitly. Every unit has a review status/skip reason.

Views are independently constructed from current raw polygons/member IDs and the
exact original file/RGB hashes, not from a trusted `crop_safe` shortcut. Positive
target/nonmember bbox overlap or invalid geometry is conservatively skipped.
The model receives an **unmodified RGB context crop** and a **separate blue-target
localization overlay**. Context-only neighboring IDs, global/local polygon,
context bounds, original/view file and pixel hashes, unit/reading/model/request
revisions and requested-body SHA256 are saved. No resize, rotation, enhancement,
erasure or recognition-crop permission is introduced. View limits:1536 maximum
side,2 million pixels and1 MiB per PNG; larger views are visibly skipped.

Strict full JSON requires exact target/source IDs and original-text echo, one of
`unchanged`, `proposed_correction`, `unreadable`, `uncertain`, bounded reasons and
`unverified`/`ambiguous` uncertainty. Only a correction may carry nonempty different
proposed text (at most8192 characters). Duplicate keys, extra confidence/approval
fields, foreign/duplicate IDs, wrong types, empty corrections, oversized or
length-truncated generation fail validation. No confidence threshold accepts text.
Wider context cannot prove missing glyph membership. Neighbor text must never be
treated as target source; actual semantic attribution still needs parent checking.

Original OCR boxes, units, readings/candidates and `selected_revision` are **never
overwritten**. A validated proposal creates separate `AlternateInput`/request/result
records, with `output_kind="unverified_visual_alternate"`, `alternate_*` statuses,
mandatory review, exact original/proposed text and source/image/review provenance.
It exposes `alternates`, not ordinary `translations`. For blocked originals its
base must remain the already-enabled fragment request. Strict APIs reject alternate
requests; a suggestion cannot promote an original or fragment to strict eligibility.
TG27B uses its unchanged native user template with the proposed text, never arbitrary
JSON/image conditioning. Original text and original translation jobs remain retained.

### Phase, fallback and residency rules

- All review calls complete before any translation. Selected/saved parent-unit
  order is used for review, subject to attribution skips and its independent budget.
  `--unit-id` selects original units for both stages; no model result chooses new IDs.
- Original strict jobs run first. Strict-source alternates follow them. In the
  selected/script-ordered fragment queue, a proposed alternate precedes its own
  original fragment, retaining the immediate original fallback where budget permits.
  Both original and alternate translation attempts spend the same existing global
  `--max-attempts` budget; alternate counts are separate from strict/fragment counts.
- Reviewer provider/output failure is best-effort: visible failure/raw evidence,
  no proposal, original translation path intact. Unreadable/uncertain verdicts are
  unverified model reports, not source mutation or automatic vetoes. Cancellation,
  source drift and changed view files are fatal phase stops, not swallowed fallback.
- A workflow-created reviewer gets **one explicit tag-scoped** `/api/generate`
  empty-prompt `keep_alive=0` control call after review, also on failure/interruption
  when a generation was dispatched. It is not a service stop or wildcard unload.
  `control_calls` is separately bounded to1; release timeout is at most30 seconds.
  A failed release stops translation to avoid silently overlapping the large models.
  The acknowledgment is recorded; actual residency/memory behavior needs parent smoke.
- Injected caller-owned reviewers are **never unloaded or closed** by the workflow;
  caller residency responsibility is explicit in the artifact. Zero review calls
  initialize/load/release no default review provider.

Review mode writes workflow schema3. `ocr_review` records all unit dispositions,
requests, view paths/hashes, verdict revisions, generation timings/raw evidence and
residency/control costs. Each unit also has `visual_alternates`; original strict and
fragment fields remain separate. Source/request/crop changes invalidate proposals.
Proposals, unchanged judgments and fluent alternate outputs are not human review,
OCR truth, semantic-quality evidence or rendering/erasure authorization. The workflow
does not invoke a renderer or the separate offline HTML report tool below.

### Parent local smoke (four review calls maximum)

Use existing `IMAGE` / `MODEL_ROOT` and a fresh output directory:

```sh
.venv-paddle6/bin/python src/main.py \
  --image "$IMAGE" --output test-results/workflow-visual-review-parent-v1 \
  --run-local-ocr --device cpu --model-root "$MODEL_ROOT" \
  --run-local-model --review-ocr --review-model qwen3.8:latest \
  --max-review-attempts 4 --review-timeout 90 \
  --endpoint http://100.121.133.88:11434 \
  --max-attempts 20 --timeout 120 --keep-alive 60 --num-predict 1024
```

Add `--allow-fragment-previews` explicitly to permit proposals on supported blocked
singletons. Saved-page **input-only** preflight is in
`test-results/translation-visual-review-v1/saved-page-preflight.json`: without that
flag the first4 attributable targets are8/10/31/42; with it they are0/4/7/8.
This preflight called no provider and established no reread or translation quality.
The coder runs only mocked/offline tests; the parent runs this actual model/OCR smoke.

```sh
.venv/bin/python -m unittest tests.translation.test_ocr_review -v
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m unittest tests.source_only_check -v
.venv/bin/python -m compileall -q src/main.py src/tralens/translation tests/translation/test_ocr_review.py
```

## Offline translation HTML report

`tools/translation_review.py` turns a **saved** workflow preview into a static,
self-contained `review.html` and `coverage.json`. It supports schema1 (strict),
schema2 (separate fragment previews), schema3 (optional visual review and
separate proposed-reading alternate translations), and schema4 (separately bound
uncertain OCR-region translations). It performs no OCR, inference,
provider/network requests, engine initialization or browser launch, including on
import. It needs only existing Python dependencies and the explicit original image.

```sh
.venv/bin/python tools/translation_review.py \
  --preview test-results/workflow-fragments-priority-parent-v1/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --output test-results/translation-fragments-html-v1

.venv/bin/python tools/translation_review.py \
  --preview test-results/workflow-visual-review-parent-v2/preview.json \
  --image images/ubunchu-ja-test/ubunchu01_04.png \
  --output test-results/translation-visual-html-v1
```

The output directory must be **new**, with an existing parent and no symlink
components. Existing directories and symlinks are refused. The saved `input.path`
and batch page `image_path` must exactly equal the resolved explicit `--image` path;
there is no relocation/substitution or automatic path following. Only `--preview`
and `--image` are read as data inputs. Paths inside the preview are never opened;
raw-response/view sidecars are not required. Traversal/URL-like sidecar names are
rejected. Source files are never updated, and no reading is automatically selected
or accepted. The tool does not follow the saved model endpoint.

The report embeds the full original RGB page, with a checkbox for numbered
source-unit geometry and separate unassigned-source labels. Geometry is diagnostic
only. Unit cards show original OCR → unverified proposed correction → original
strict/fragment and alternate translation outputs, with their distinct statuses,
blockers, uncertainties, typed outcomes, failures and retained provenance. All
outputs require review. Counters show translation/review budgets and retained
review-release information; acknowledgment is not measured model residency.
`--crops` optionally embeds conservative crops derived from the explicit image;
blocked, unsafe or neighbor-intersecting crops are omitted in favor of the page.
These crops do not establish ownership, completeness, rendering or erasure rights.

Validation recomputes original byte/RGB hashes, detection snapshot/batch evidence,
proposal membership identities, original units/readings and exact workflow requests.
Strict, fragment and alternate results are checked against their own immutable
contracts, including source/reading/request revisions and execution metadata.
Visual review crops/overlays and requested-body hashes are regenerated from the
original pixels, then compared with retained view/request/verdict/alternate metadata.
**Raw HTTP responses and existing view files are not opened or authenticated.**
Matching claimed response digests establishes internal consistency only, not that
a model actually produced the text. No data hash proves semantic/model truth,
accuracy, human approval or transcription fidelity. Regenerating encoded view
hashes requires compatible Pillow/font/PNG behavior; an encoder mismatch is rejected
rather than silently certified.

Failed/interrupted artifacts with consistent retained source evidence can be
reported. An OCR-stage failure before request construction shows derived source
cards explicitly as unexecuted, with source-only coverage. Missing/inconsistent
provenance, stale sources, unsupported schemas, malformed geometry/types, duplicate
JSON keys and contradictory counters/statuses are rejected without publishing a
report. Empty no-detection pages are supported. Limits: 64MiB JSON, 32MiB original
image, one raster frame/40 million pixels, 2048 detections, 512 units, bounded JSON
depth/items/text, 32MiB unit-card HTML and 64MiB total HTML. HTML/SVG text is escaped;
the report has no scripts or external resources and a restrictive no-network CSP.

Offline verification (portable synthetic fixtures plus explicit optional saved-page
schema1/2/3 CLI integrations, skipped when those local artifacts are unavailable):

```sh
.venv/bin/python -m unittest tests.tools.test_translation_review -v
```

## License

Copyright (C) 2026 Daniel Lew

Tralens is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. It is
distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE) for the full terms.

Third-party packages, and models or fonts you supply yourself, keep their own
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
