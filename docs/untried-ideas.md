# Untried ideas: detailed plans

Everything here is marked NEW in `docs/improvement-ideas.md` (no sign of it in the
repo as of 2026-11-19), plus new ideas marked 💡. Every item keeps the existing
rules:

- OCR, translation and rendering stay separate, coordinated by `src/main.py`.
- Everything is opt-in until it's measured.
- New readings or masks never silently gain erase permission.
- No evaluation answers go into model inputs.
- Pixels outside declared support stay exact.
- Licences are checked before any model or dataset is downloaded.

VRAM budget (RTX 5090, 32 GB): load one heavy model at a time and unload between
stages, as the coordinator already does.

| Model size | Approx. VRAM |
| --- | --- |
| 32B LLM/VLM at Q4_K_M | ~20–22 GB, plus context |
| 24–27B at Q4 | ~16–18 GB |
| 12B diffusion at fp8 | ~13–16 GB |
| Detector / LaMa / OCR models | < 2 GB each |

Each item lists: **Targets** (which known failure), **How** (concrete steps),
**Measure** (how we'd know it worked) and **Cost** (effort and risk).

---

## A. Evaluation first (makes everything else measurable)

### A1 💡 ★ Pepper&Carrot as a real gold set
Pepper&Carrot (CC BY 4.0) publishes source files with text on separate layers, an
artwork-only version, and the same pages in many languages, including English.
That gives, for free and legally:
- the exact text pixels (a perfect erase mask),
- the exact clean background (perfect inpainting ground truth),
- the exact Japanese strings (OCR ground truth),
- a human English translation (translation reference).

**How**
1. Download the episode source bundles: Krita/ORA files, plus text-free and
   English renders for the episodes already cited in
   `test-results/colored-comics-peppercarrot-01/provenance.json`. Record URLs and
   SHA-256 hashes the same way.
2. Write `tools/build_peppercarrot_gold.py`. It exports per page:
   - `original.png` (Japanese composite),
   - `clean.png` (artwork only),
   - `text_mask.png` (alpha of the text layers, thresholded),
   - `units.json` (Japanese strings and boxes from the text layers or the SVG
     source),
   - `reference_en.json` (English strings matched to units by layer or position).
3. Keep it strictly as evaluation data: never a pipeline input, never in prompts.
   Store it under ignored `test-results/gold-peppercarrot/`.

**Measure**
- Removal recall: fraction of `text_mask` pixels that differ from `original` in
  our cleanplate.
- Inpainting error inside the mask against `clean.png`: PSNR/LPIPS.
- Collateral: changes outside the mask.
- OCR character error rate.
- Translation: chrF/COMET against the English edition, read as a soft signal,
  since it's one valid translation among many.

**Cost**: 1–2 days. Very high value. The caveat is that the comic is Japanese-translated
Western art, not manga, so it needs a manga gold set beside it (A2).

### A2 💡 ★ Synthetic paste-and-remove gold
Take text-free regions of real manga pages (e.g. panels with no detections, or
manually confirmed blank areas), and render real Japanese strings onto them.
Render with licensed Japanese fonts, vertical and horizontal, with outlines,
random sizes and rotations, over flat, tone and painted backgrounds. The ground
truth (mask, string, clean background) is then known exactly.

**How**
1. Write `tools/synth_text_overlay.py`. Inputs: page, region, string source (any
   public-domain Japanese text such as Aozora Bunko), font list. Output: composite,
   mask, clean crop, string.
2. Generate ~500 crops in difficulty tiers. Also include SFX-like ones: large,
   rotated, stroked, hand-drawn-looking fonts.
3. Run detection, masking and cleanup only (no translation needed) and score as
   in A1.

**Measure**: residual-ink recall per tier, collateral, and OCR character error
rate per tier. This becomes the regression suite for every mask/inpaint change.

**Cost**: 1–2 days. Also doubles as training data for B2.

### A3 Residual-Japanese metric on real outputs
**How**: after rendering, run PaddleOCR detection plus the CTD mask on the
*cleanplate* (not the composite), restricted to declared support. Count detections
whose reading contains kana or kanji. Write `residual-japanese.json` per page and
show it as a red overlay in the dev frontend ("Leftover source" layer).
**Measure**: residual count per page over time. **Cost**: half a day.

### A4 A–B comparison view in the dev frontend
**How**: add `/compare?a=<job>&b=<job>` to `devui/server.py`. Show units side by
side (source crop, translation A versus B, clean crop A versus B). Keys:
`1`/`2`/`=`, plus `x` for "both wrong". Save `preferences.jsonl`.
Optional, never required for the pipeline to run.
**Cost**: 1 day. Turns your inspection time into reusable data.

### A5 Soak test
**How**: `tools/soak.py` loops 100–300 pages through the full pipeline. It
logs `nvidia-smi --query-gpu=memory.used`, host RSS, Ollama `/api/ps`, the time
per stage and any failures to CSV, then plots memory against page index.
**Measure**: memory plateau (no upward trend after warm-up), zero orphan processes.
**Cost**: half a day, plus overnight runtime.

---

## B. Detection and masks

### B1 ★ Test-time augmentation and ensemble masks
**Targets**: missed strokes, which are the main cause of leftover source text.

**How**
1. Run the CTD mask at 3 scales (0.75×, 1×, 1.5×) and with a horizontal flip.
   Map back to native coordinates.
2. Add a second segmenter: a YOLO11-seg model trained in B2, or SAM 2 prompted
   with every Paddle detector polygon as a box.
3. Vote per pixel: take a pixel only if ≥2 sources agree *and* it's inside a
   detector polygon dilated by k × stroke width. Outside polygons, require ≥3
   votes plus the "ink-like" test you already have in `source_support.py`.
4. Feed the result as a new candidate mask that goes through the existing
   source-bound erase-permission logic. Don't bypass it.

**Measure**: A2 recall and collateral at matched collateral. **Cost**: 2 days.

### B2 💡 Train our own text-pixel segmenter on synthetic data
**How**: generate 20–50k crops with A2's tool, using backgrounds from licence-clean
art (Pepper&Carrot artwork-only pages, your own pages). Train a small
U-Net or YOLO11-seg (Ultralytics is AGPL, so check it fits GPL-3.0; a
plain PyTorch U-Net avoids the question). A few hours on a 5090.
**Why it's interesting**: removes the unresolved CTD distribution licence and is
tuned to *our* failure cases (outlined, coloured and SFX text).
**Measure**: A1/A2 recall against CTD. **Cost**: 3–5 days.

### B3 SFX and stylised-lettering class
**How**
1. Candidate generator: large connected ink components with stroke width well
   above the dialogue median, outside balloons, often with outlines.
2. Ask a VLM per candidate: "Is this hand-lettered Japanese sound effect text?
   Answer JSON {is_sfx, reading, confidence}", with Ollama `format` set to a JSON
   schema.
3. SFX units get their own path: erase with a wider mask plus a learned inpainter
   (art underneath), and render with an SFX font plus outline and the original
   angle (see E2/E3).

**Measure**: A2 SFX tier, plus manual review of the "Top—/Wisdom/Three" pages.
**Cost**: 2–3 days.

### B4 💡 Ask the VLM for text boxes as an extra detector
Qwen-VL-family models can return bounding boxes. Prompt: "List every region of
Japanese text, including tiny annotations and sound effects, as JSON boxes."
Use the boxes only to *propose* missed regions, which are then checked by
CTD/Paddle and the ink test.
**Measure**: new true regions found on A1/A2 and real pages. **Cost**: 1 day.

---

## C. OCR

### C1 ★ Script-mismatch re-read
**Targets**: `TP`, `G`, `To`, `2` staying as unchanged targets while Japanese
remains on the page.

**How**
1. For each unit whose reading is mostly ASCII, compute a "looks Japanese" score
   from the crop. Options: CJK probability from the existing recogniser's
   per-character logits, a tiny classifier on glyph aspect and stroke density, or
   a VLM yes/no.
2. If the crop looks Japanese but the reading isn't, mark it
   `script_mismatch` and re-read with Hayai, manga-ocr and the VLM. If any reader
   returns kana or kanji, record it as an alternative reading. It is still not
   auto-selected; use C2.
3. In rendering, never treat raw == target as "nothing to do" when
   `script_mismatch` is set. Erase if the source is supported.

**Measure**: count of unchanged-but-Japanese units on reviewed pages. **Cost**: 1 day.

### C2 ★ Consensus plus language-model plausibility for choosing readings
**How**
1. Candidates: Paddle, Hayai (tight, axial and contextual projective views),
   manga-ocr and the VLM.
2. Score each candidate on:
   - (a) character-level agreement with the others (character error rate to each
     peer, weighted by the reader's known accuracy on A2),
   - (b) Japanese plausibility: mean token log-probability from a small Japanese
     LM (e.g. a 1–3B model run through `llama.cpp` with logprobs, or Ollama
     logprobs if available),
   - (c) **analysis by synthesis** (C3).
3. Learn the weights on A2 plus reviewed real units; never per test page.
4. Promote only if the margin over the current selection exceeds a threshold
   tuned on A2. Record why it was promoted.

**Measure**: OCR character error rate on A1/A2; downstream translation changes on
reviewed pages. **Cost**: 2–3 days.

### C3 💡 Analysis by synthesis: render the reading and compare it to the glyphs
For a candidate reading, render it in a few Japanese fonts at the detected size and
direction, align it to the source crop (rigid plus scale search), and compute
glyph-shape similarity (chamfer distance on skeletons, or IoU of dilated masks).
- Wrong readings (missing characters, `G` for a kanji) align badly.
- The aligned render also gives an *expected* glyph mask, and source ink outside
  it is a hint for residual strokes the mask missed.

**Measure**: separation between correct and wrong readings on A2. **Cost**: 2–3 days.

### C4 Test-time augmentation for OCR
Re-read each crop at 2–3 scales, with ±1–2° rotation and slight contrast
stretching, and vote per position. Cheap with manga-ocr and Hayai.
**Cost**: half a day.

---

## D. Grouping, order and unit type

### D1 ★ Unit-type classification
**How**
1. One VLM call per page (Qwen VL 32B at Q4) with the page image and a numbered
   overlay of unit boxes. The overlay is drawn in a copy, never the original.
2. Ask for JSON: `{id, type ∈ [dialogue, thought, narration, annotation,
   furigana, sfx, ui_label, ui_body, sign, title, credit], panel_id,
   speaker_hint, attaches_to}`.
3. Store it as a separate `unit-types.json` evidence file bound by hash.
4. Use it to pick the translation prompt (F3), font/style (E2), and merge rules:
   `ui_label` never merges with `ui_body`; `annotation` attaches to its target.

**Measure**: accuracy against labels for A1 and reviewed pages; fewer UI/label
merges and fewer split annotations. **Cost**: 2 days.

### D2 Graph-based grouping with hard constraints
**How**: nodes are detected lines. Edge weight combines direction match,
font-size ratio, gap relative to line height, alignment, same balloon (from the
balloon experiment or SAM 2), same unit type, and VLM "same utterance?" for
ambiguous pairs. Hard cut edges: different panel, different balloon, a
`ui_label`/body pair. Group with connected components above a threshold, tuned
on reviewed pages.
**Measure**: grouping F1 against reviewed membership. **Cost**: 3 days.

### D3 Panel segmentation without Magi
Classic approach: binarise, find the white gutters, then recursive XY-cut, plus a
line-segment detector (`cv2.createLineSegmentDetector`) for panel borders.
Alternatively train a small YOLO on panel boxes from a licence-clean source.
Use panels as hard constraints in D2 and for clipping layout boxes.
**Cost**: 1–2 days.

---

## E. Typesetting

### E1 Text colour and outline sampling
Inside the erase mask, cluster the source ink into fill, outline and background
(k-means in Lab). Render the translation with the fill colour, the outline colour
and an outline width proportional to the source's. Important for colour pages.
**Cost**: 1 day.

### E2 Fonts and style per unit type
Add licensed faces to `fonts.json` for SFX (heavy display), narration (serif or
italic), UI (sans) and thought (lighter). Choose by D1 type.
**Cost**: half a day plus a licence check for each font.

### E3 SFX rendering
Keep the original SFX's angle (from the min-area rect of its ink), use large text
with an outline, and allow it to follow a curve. 💡 Optionally render the
translated SFX *small beside* stylised art, rather than erasing art-heavy
lettering, as a switchable "translator-note" mode.
**Cost**: 1–2 days.

### E4 💡 Balloon capacity in the translation prompt
Before translating, compute how many English characters fit in the balloon at the
minimum font size and pass "≤ N characters" to the translator. This shapes the
first draft instead of rewriting later, so it doesn't break the exact-wording rule.
**Cost**: 1 day.

---

## F. Translation

### F1 ★ Head-to-head model comparison
**Candidates**:
- Qwen3-32B text (thinking off),
- Qwen3-30B-A3B (fast mixture-of-experts),
- Qwen3-VL-32B (with the image),
- Mistral Small 24B,
- Aya Expanse 32B,
- a Japanese-tuned model (Shisa family),
- TranslateGemma 27B as baseline.

**How**
1. Freeze unit inputs from ~10 reviewed pages plus the A1 English edition.
2. Same JSON-schema output, greedy decoding, same context.
3. Blind A4 view and COMET/chrF.

Record digests the way TG27 already is.
**Cost**: 1 day of setup, then inference time.

### F2 ★ Whole-page structured translation
**How**: one request per page containing:
- the units in reading order: `{id, type, source, alt_readings, speaker_hint,
  capacity}`,
- the previous page's translations,
- the series glossary (F5),
- optionally the page image (VLM route).

Output a JSON array keyed by id (Ollama `format` schema). Validate that every
input id appears exactly once with a non-empty English string. On failure, retry
just the missing ids per unit.
This works with generalist models, not the TranslateGemma template, so run it
under F1.
**Targets**: から → "from", cut-off endings, inconsistent names.
**Cost**: 2 days.

### F3 Type-specific prompts
Instructions per D1 type:
- SFX: one short English onomatopoeia.
- UI label: literal, title case.
- Annotation: short gloss, no sentence.
- Dialogue: natural, keep the speaker's tone.

**Cost**: half a day after D1.

### F4 💡 Minimum Bayes risk (MBR) selection across samples and models
**How**
1. For each page, collect N = 6–10 candidates: TG27 greedy, 2–3 other models
   greedy, and a few sampled at temperature 0.5.
2. Pick the candidate with the highest average similarity to all the others,
   measured with COMET (if licensed) or chrF plus multilingual embeddings.
3. Tie-break on fit within capacity.

MBR reliably beats single greedy outputs and needs no reference.
**Cost**: 2 days, about 6–10× translation compute (fine locally).

### F5 💡 Auto-built series glossary and story memory
After each page, a small LLM call extracts proper nouns, recurring terms and
character names with their chosen English. Store them in
`series/<id>/glossary.json`, review in the dev frontend if wanted (optional), and
feed them to F2.
A "story so far" summary of ≤200 words is carried across pages.
**Targets**: "Co-Co Deck" for codec, inconsistent names.
**Cost**: 1–2 days.

### F6 Back-translation drift check
Translate the English back to Japanese with a different model, then compare
meaning with multilingual sentence embeddings (e.g. multilingual-e5, BGE-M3) and
ask an LLM: "List information in B that isn't in A, and vice versa." Flag
additions ("without any permission", "user-friendly") and omissions. Flagged
units get re-translated by the next-best MBR candidate.
**Cost**: 1–2 days.

### F7 💡 Onomatopoeia lexicon
Build a lookup table of common Japanese onomatopoeia and mimetic words
(ドキドキ, ゴゴゴ, シーン, …) with English renderings, written from general
knowledge or a licence-clean dictionary. This is a general resource, not tuned to
evaluation answers. Give it to the translator as a hint for `sfx`-typed units.
**Cost**: 1 day.

### F8 Fine-tune (later)
QLoRA on a 24–32B model (Unsloth or PEFT; fits in 32 GB at 4-bit) using:
- licence-clean Japanese–English comic pairs (Pepper&Carrot across episodes,
  OpenMantra if its licence permits),
- A4 preference pairs, used for DPO.

Only after A1/A2/A4 exist, so we can show it helps.
**Cost**: 1–2 weeks.

---

## G. Inpainting

### G1 ★ Manga-tuned LaMa, tiled at native resolution
**How**
1. After a licence check, convert `mayocream/lama-manga` (or similar) to ONNX in
   the same way as the existing LaMa.
2. Instead of fitting a region into 512², crop native-resolution 512 tiles with
   ≥128 px overlap around each mask component, inpaint, and blend with
   feathering *only inside the mask*. Pixels outside the mask are copied from the
   original exactly.
3. For large masks, use LaMa's high-resolution refinement idea (multiscale
   feature refinement), implemented ourselves from the paper rather than copied.

**Measure**: A1 LPIPS inside the mask; review of tone and line pages. **Cost**: 2 days.

### G2 ★ Region-type router
Per mask component, compute features:
- background variance in a ring around the mask,
- periodicity (FFT peak strength),
- edge density,
- colour count,
- D1 type.

Route by those features:

| Region | Route |
| --- | --- |
| Flat | Colour or Laplace fill (exists) |
| Periodic tone | Phase-locked repair (exists), else G4 |
| Lines crossing | G3, then LaMa |
| Painted/colour | G5 diffusion |
| Everything else | LaMa |

Tune thresholds on A1/A2.
**Cost**: 1–2 days.

### G3 💡 Redraw panel lines and strokes before inpainting
Detect line segments in a ring around the mask (LSD or Hough), keep those that
enter on one side and exit on the other with matching angle and width, and draw
them through the mask with anti-aliasing at the measured width and colour. Then
inpaint what's left with that line locked.
**Targets**: damaged panel rules, hair strands and UI borders. **Cost**: 2 days.

### G4 Rotated and nonstationary screentone
Extend the phase-locked repair: estimate the dominant 2-D lattice from the FFT of
the surrounding ring (handles rotation), fit the dot size as a slowly varying
field (gradient tones), and synthesise. Fall back to exemplar patch synthesis
(PatchMatch-style, written ourselves) using only the same tone region.
**Cost**: 3 days.

### G5 Diffusion inpainting for colour and painted art
**Candidates**: FLUX.1 Fill [dev] (fp8; non-commercial licence, so private
evaluation only), an SDXL inpainting model (OpenRAIL++), BrushNet/PowerPaint on
SD1.5/SDXL, Qwen-Image-Edit (check licence).

**How**: crop around the component with generous context, prompt empty or
"background, no text", run 3 seeds, score each with G6, and paste back *only
inside the mask*. Keep it off by default; route only G2 "painted" regions to it.
**Cost**: 2–3 days. Slow (seconds per region), which is fine for evaluation.

### G6 💡 Multi-inpainter selection by score
For each region, produce candidates from flat fill, LaMa, manga-LaMa and
diffusion. Score each on:
- (a) seam continuity: gradient mismatch along the mask border,
- (b) residual text: CTD and OCR on the result (must be ~0),
- (c) texture statistics against the surrounding ring (Gram/LPIPS distance),
- (d) hallucination: new edges or faces in the mask (edge density above the ring).

Choose the best one and save every candidate for review.
**Cost**: 2 days.

### G7 💡 Reuse matching background from the same page or chapter
Manga often repeats backgrounds and screens across panels. Search other panels
for a patch matching the ring around the mask (ORB/SIFT features plus a
homography). If an aligned match is found with a low ring error, copy its pixels
into the mask. That's real art, not hallucinated. Especially good for the
Ubuntu pages' repeated UI screenshots.
**Cost**: 2 days.

### G8 💡 Colour-page layer decomposition
On colour pages, lettering is usually flat black or white with an outline on top
of painted art. Use colour-difference keying: estimate the art underneath as a
smooth field from the ring (low-order polynomial or bilateral fill in Lab), find
pixels well away from that field in lettering colours as text, and peel them off
with soft alpha. Small residual gaps go to LaMa.
**Cost**: 2 days. Pepper&Carrot gold (A1) measures it directly.

---

## H. Pipeline-level

### H1 Verified erase loop
After the first cleanup:
1. A3 finds residual Japanese,
2. widen the mask by one stroke width around the hits,
3. re-route through G2,
4. repeat at most twice,
5. record each iteration.

This closes the loop on "ALWAYS REMOVE THE TEXT".
**Cost**: 1 day after A3.

### H2 Erase-only fallback for blocked units
Units blocked on ownership (e.g. `saved-example` #19): if the ink is confidently
Japanese (C1/C3) and inside a supported region, erase it and render a small
"[untranslated]" marker or the best-effort translation in a distinct style,
instead of leaving the source.
**Cost**: 1 day; needs a policy decision on what to display.

### H3 Model scheduler
A small `scheduling.py` extension that orders stages by model across a whole
batch (all OCR, then all VLM type calls, then all translation, then all
inpainting). That keeps each model loaded once per batch rather than once per
page. Big speed-up for soak tests and MBR.
**Cost**: 1–2 days.

---

## Suggested order

1. **A1 + A2 + A3**: gold sets and the leftover-Japanese metric.
2. **C1 + H1**: directly attack retained source text.
3. **D1 + F2 + F1**: unit types plus page-level translation model comparison.
4. **G2 + G1 + G3**: inpainting router, tiled manga LaMa, line redraw.
5. **F4 + F5 + F6**: MBR, glossary, drift check.
6. **B1/B2, G5–G8, C2/C3**: bigger bets once the metrics exist.
