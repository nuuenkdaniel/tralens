# Iterative OCR and rendering quality route

This is an **opt-in development route**, not a replacement for the default
pipeline or a claim that arbitrary manga is solved. It was designed against
Tralens's source and local failures, without fetching related applications'
implementations. Prior research exposure still prevents a formal clean-room claim.

## Rendering: keep complete cleanup, share only unused layout space

The confidence-gated route initially preserved a useful `Hey` replacement because
its layout window overlapped whitespace reserved for another accepted dialogue.
`reserve_layout` now subtracts reserved pixels from **layout only**, adding them
to obstacles. The complete foreground/erase masks stay unchanged. Exact-glyph
fitting, requested minimum size, collision checks and pixel audits still apply.
It never clips a source stroke or cleanup mask to force a fit.

The saved-translation rerender in
`test-results/replacement-confidence-render-v2/` recovers that interjection:
11/29 replacements rather than 10/29, with all 29 translations retained. It does
not restore coarse footprint erasure. This is a layout-support improvement, not
evidence that the OCR or translation is correct.

## OCR: preserve raw evidence, propose a better bounded reading

`src/tralens/ocr/quality_reread.py` implements:

1. **Terminal-row grouping hypothesis.** A multi-row horizontal block can acquire
   one short terminal row when alignment/extent/gap checks and the original
   enclosed light-background component agree. Ambiguous hosts, different
   enclosures and chained extensions are rejected. Current source polygons and
   crop conflicts stay recorded. This does not certify complete utterances.
2. **Separate original and derived proposals.** All Paddle detections, scores and
   strings remain unchanged. Original proposal batches are retained alongside the
   new grouping hypotheses. No manually supplied source wording or masks enter
   the production route.
3. **Explicit visual reread.** Original block/line crops and digest-bound 4×
   LANCZOS presentations go to an explicitly selected installed vision model.
   Upscaling changes presentation, not original evidence or source coordinates.
   Each target line has its own bounded response field. Model uncertainty and
   malformed, truncated or overgenerated answers do not authorize a selection.
4. **Separate-reader corroboration.** A candidate must match native Manga OCR
   line rereads under a fixed bounded set of padding/scale presentations. NFKC and
   whitespace removal are used **only for agreement diagnostics**. The exact
   selected visual string, including supplied spaces and line breaks, is what
   enters translation. No expected-word substitutions or fluency selection exist.
5. **Unverified derived translation input.** A preferred candidate is separately
   bound to the original raw unit, proposal, image and line evidence. The original
   unit still selects raw OCR and retains its structural blocks and alternatives.
   The existing coverage input records the separate selected-reading revision.
   This route cannot waive hard structural/association/annotation faults.
6. **Independent edit requirements.** Corroborated rereads do not inherit raw
   Paddle scores or claim an accuracy probability. They remain explicitly
   unverified; confidence-gated image replacement still requires native cleanup,
   adequate lettering size, exact text and all existing integrity checks.

Agreement is not truth: two recognizers can agree on a wrong reading. This route
also requires bounded source geometry, original ink and enclosed-background
evidence, but those are not semantic text certification. It remains opt-in until
fresh independent references and held-out pages establish its tradeoffs.

## Local assets and runtime boundaries

The live experiments reuse existing local Paddle v6 exports, the already-present
Manga OCR snapshot `aa6573bd10b0d446cbf622e29c3e084914df9741`, the configured
Ollama `qwen3.8:latest` vision model and native `translategemma:27b`. Model cards/
daemon metadata declare Apache-2.0 for the reread artifacts; this is not a full
training-data rights audit or distribution clearance. No weights, fonts or
dependencies were downloaded or bundled.

Unlike the offline confidence comparison, **this explicitly enabled route sends
reading-crop presentations to the configured private Ollama server** (the existing
Tailscale endpoint in the local runs). It does not enable a new public/cloud
provider. Reread responses, native/transformed crops and attempts remain local
evidence. The visual model is explicitly unloaded before translation; failure to
release it is reported rather than silently loading another large model beside it.

## Reproduction

Use a fresh output directory every time. No old evidence is overwritten.

```sh
.venv-paddle6/bin/python src/main.py \
  --image PATH/original.png --output NEW_RUN \
  --device cpu --model-root /home/Danuu/.cache/tralens/paddle-v6-models \
  --run-local-ocr --run-local-model --model translategemma:27b \
  --endpoint http://100.121.133.88:11434 --max-attempts 64 \
  --render --fonts PATH/fonts.json --render-policy best-effort \
  --replacement-policy confidence-gated --quality-reread \
  --reread-model qwen3.8:latest \
  --manga-reread-model /home/Danuu/.cache/huggingface/hub/models--kha-white--manga-ocr-base/snapshots/aa6573bd10b0d446cbf622e29c3e084914df9741
```

`--quality-reread` requires coverage-first translation and explicit existing local
Manga OCR assets. Default runs perform no new rereads or terminal regrouping.
The existing viewer is not restarted or silently switched to this route.

## Development observations and limitations

Early local Paddle rectangle rereads improved some characters but remained poor
on handwriting. Manga OCR block reads also hallucinated plausible text on art;
line reads sometimes helped, and pixel-scale/context changes sometimes hurt.
The installed visual model confidently misread several tiny native crops.
Explicit higher-resolution presentations improved the instruction probe, but
whole-block JSON strings could truncate or repeat a terminal line. Those outputs
were rejected, not cut into apparently successful transcripts. The later response
contract uses bounded per-line fields and separate-reader checks.

Live iterations and failure evidence remain ignored under
`test-results/quality-iteration-live-v*/`. The first run failed during an optional
module import; later runs retained full previews/partial composites and reported
failed/uncorroborated rereads. These development failures are retained, not erased
from the experiment history. This is still one previously inspected page, not a
held-out generalization benchmark.

Counts must distinguish raw detections, original units, derived grouped units,
translations and actual image replacements. Joining two original units can reduce
the derived-unit count without dropping either source detection. Preserve all
available translations even when their source text remains on the image. Do not
call a partial translated page complete or use fewer edits as proof of OCR success.

## Successful development checkpoint and repeat run

`quality-iteration-live-v4` finally produced an automatically corroborated
instruction reading:

```text
で この
アイコンを
ダブルクリック
で
```

The exact saved candidate also retains its trailing space. The native translator
produced **“Double-click this icon.”** The original garbled raw readings and all
alternatives remain saved. No expected-word replacement or source correction was
injected into either recognizer or translator.

Visual inspection of the full composite, native cleanplate, foreground and
translated crop found the source instruction lettering removed, the balloon
border intact, nearby face/sound-effect/shorthand artwork retained and readable
English placement. This is a real development improvement over the earlier
“aconite / Dabunlupri 7” output and broad smear. It is not independent native
hidden-background ground truth.

A second fresh full pipeline run, `quality-iteration-live-v5-confirm`, produced
the same instruction and **byte-identical final composite**. Both successful
runs have:

- 43/43 raw detections represented exactly once in the derived partition.
- The original 29-unit proposal batch retained separately; 28 derived units
  after joining the instruction block with its terminal line.
- 28 translation results from 28 translation calls, plus eight visual reread
  attempts and five Manga OCR line attempts. No translation-budget exhaustion.
- 12 actual replacements, including all eight original strict units and the
  recovered `Hey` and instruction replacements. Five identical-string cases are
  explicitly preserved without repainting; nine low-score and two unsupported
  cleanup edits are deferred. All 28 translations remain available.
- Zero outside-cleanup, outside-alpha or protected changed pixels.
- The page explicitly reported **partial**, with no full-page translation or
  export approval. The CLI's partial-output exit status is intentional, not a
  concealed execution failure.

The confirmation run retains 46 reread artifacts, 991,374 measured bytes and
1,373,764 accounted raster pixels. These are local measurements, not deployment
latency/memory guarantees. All reread artifact hashes checked, and the offline
translation reviewer verified the retained source/request/result bindings.

Remaining visible limitations matter: LiveCD and desktop dialogue are still not
reliably reread, several effects remain Japanese, a small gloss is not separately
translated, and the `Okay` target for a short interjection is questionable. The
guard deliberately rejected other hallucinated or uncorroborated model outputs.
This checkpoint improves one important OCR/grouping/removal failure and avoids
known destructive edits; it does **not** solve all text or arbitrary artwork.

Local final review:
`test-results/quality-iteration-live-v5-confirm/rendering/review.html`.

| Retained evidence | SHA256 |
| --- | --- |
| Confirmation preview | `4ab7bf946393e524c3e214c5206040b6f64f6262fa4d0f18b25ac5b0c6963683` |
| Confirmation rendering manifest | `6e3eece6dccd6458097e4f995ec1783fd62543aec345461304c141c18b79b820` |
| Composite in both successful runs | `6fe4a61685d18c937770147efe56f5abbdb15c55c2a79d782838ffbf0cb8f5e4` |

Default routes were separately rerendered: all **444 coverage PNGs** and all
**129 strict-only PNGs** remain byte-identical to their saved baselines. Originals,
previous failed/successful runs, dependency edits and the running viewer remain
untouched. No repository push, visibility change or license choice was made.

## Final checkpoint verification

- `.venv/bin/python -m unittest discover -s tests -t . -q`: **835 tests passed**.
- `.venv-paddle6/bin/python -m unittest discover -s tests -t . -q`:
  **835 tests passed**.
- `.venv/bin/python -m unittest tests.source_only_check -q`: source-only
  verification passed; the 835-test suite reports **39 optional-local-evidence
  skips** without the ignored local assets.
- Portable regressions cover original/derived source bindings, paired-reader
  disagreement, uncertainty, malformed/truncated output retention, pixel-evidence
  drift, empty original crops, budgets, terminal grouping, layout-only reservation
  and offline translation review.
- `git diff --check` passed. User-owned `pyproject.toml` and `uv.lock` content
  hashes still match their pre-iteration values; neither is part of this checkpoint.

These checks establish reproducibility and regression coverage, not general OCR
accuracy or independent approval of linguistic/artistic quality.

## Viewer integration after the screenshot feedback

The user's browser screenshot exposed a delivery gap: `execute_job` in the local
viewer did not pass the experimental replacement/reread options. Its fresh
best-effort runs therefore still used coarse fallback erasure and raw OCR. The
separate CLI quality reports were not what the viewer displayed. That mismatch
was an integration failure, not a reason to dismiss the visible quality problems.

The viewer now exposes the same native-cleanup confidence gate and optional
paired rereads. The operator supplies existing local reader assets at startup;
configuring them alone does not run models or enable rereads. Client requests may
choose only bounded options, not arbitrary model tags or filesystem paths.
Original/default runs retain their settings; broad-replacement outputs show a
warning and a fresh **Retry with quality checks** action. The displayed policy
comes from the executed pipeline/manifest rather than unsaved form controls.
Selected reread input and original raw OCR remain separately inspectable, and
`?job=...` identifies the exact saved run to display.

The idle local viewer was restarted with the existing Manga OCR snapshot enabled.
A real headless Firefox session opened the user's saved broad-replacement job,
clicked the quality retry, waited for the real pipeline and reloaded the result.
This was **not** an injected-provider run. Its saved run is:

`test-results/dev-frontend/a0651eb1907c45d9862cc2ebd8c2dc2c/`

Exact browser view:
<http://127.0.0.1:7860/?job=a0651eb1907c45d9862cc2ebd8c2dc2c>

The browser exposed all 43 source detections, 28 grouped translations and 12
image replacements. Its composite is byte-identical to the earlier confirmed
quality image (`6fe4a61685d18c937770147efe56f5abbdb15c55c2a79d782838ffbf0cb8f5e4`).
Visual inspection of both the actual served composite and browser screenshot
confirmed that the large hair/panel smears and garbled instruction replacement
were gone; difficult effects and dialogue were preserved, not magically solved.
The bad raw translations remain available in the inspector with deferral reasons.
The result is still partial, with the previously documented OCR and lettering
limitations. It is now labeled **Rendered candidate**, not presented as a
polished final page. All earlier outputs remain untouched.

Browser screenshots, served PNGs and the run snapshot are retained under
`test-results/viewer-quality-live-v1/`. A separate injected-provider browser smoke
test also passed upload, five-stage switching, comparison, zoom, highlighting,
refresh and retry without re-upload. The new offline viewer integration test
uses real source/reread/translation/rendering contracts with synthetic readers;
it verifies that the controls reach the coordinator and display the actual
selected source instead of pairing the new target with old garbled raw OCR.

The new legacy-restoration regression initially caught a missing fallback to the
executed pipeline's translation policy when preview metadata is unavailable.
Restoration now retains that policy rather than silently excluding the saved
quality run. All **24 viewer tests** pass, and source-only verification passes
the **839-test suite with 39 optional-local-evidence skips**. After a server
restart, a second real-browser inspection restored the exact new run and its
quality settings without dispatching any additional job. Offline translation
review and all 46 reread artifact hashes also checked for this viewer-produced
run. User-owned dependency files remain excluded from the checkpoint.

Final full reruns after the restoration fix: **839 tests passed in `.venv` and
839 in `.venv-paddle6`**. JavaScript syntax, Python compilation and
`git diff --check` also passed. These remain regression/integration checks, not
a claim that the remaining image or linguistic quality is satisfactory.
