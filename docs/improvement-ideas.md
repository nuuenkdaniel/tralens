# Improvement ideas: translation and inpainting (local, single RTX 5090)

Status is based on a keyword search of `src/`, `tools/` and `docs/` on 2026-11-19,
not a full code read. Verify before relying on a status.

Legend:
- **DONE**: in the default pipeline
- **PARTIAL**: some of it exists, or it's opt-in or diagnostic only
- **TRIED**: experimented with and not adopted (reason given)
- **NEW**: no sign of it in the repo
- ★ = suggested first

## 1. Pipeline architecture

| Status | Idea |
| --- | --- |
| TRIED | ★ Balloon/panel-first backbone: detect balloons, captions and SFX regions, then read and erase inside them. Exists as the `balloons.py` / `balloon_*` experiment (`docs/balloon-first-experiment.md`); not the production backbone. |
| PARTIAL | ★ Vision-language model for understanding plus specialist models for pixels. Vision models (`qwen3.8:latest` = Qwen3.5 27B, `gemma4:26b`) are used for re-reading and diagnostics. The production translator, TranslateGemma 27B, is text-only. |
| NEW | Whole-page translation in one structured JSON call. Today: per-unit calls, optionally with `--translation-context nearby`. |
| TRIED | Draft, critic, revise loop. The language-review foundation was rejected (`docs/language-review-foundation-results.md`). A critic focused narrowly on additions and omissions is untested. |
| PARTIAL | Erase separately from translating. Cleanup is independent of lettering failure, but units blocked on ownership (e.g. `saved-example` #19) are still neither erased nor translated. |
| DONE | Layered output: original, preview, cleanplate and render layers are stored separately. |

## 2. Text detection and masks

| Status | Idea |
| --- | --- |
| DONE | Comic-text-detector (CTD) ONNX text mask (`--text-mask-model`). |
| NEW | ★ Ensemble masks: CTD plus a YOLO segmentation model fine-tuned on manga text, voted. |
| TRIED | Magi v2 for panels, text and speaker links. Researched; released models are academic-use only (`docs/handoffs/manga-quality-methods.md`). |
| NEW | SAM 2 prompted by text or balloon boxes for balloon and SFX segmentation. |
| NEW | Dedicated SFX and stylised-lettering detection pass and class (only mentioned in `tools/structural_evaluation.py`). |
| PARTIAL | Stroke-width-aware dilation including outlines and halos. Lots of dilation and native-stroke logic exists; not checked whether it scales with measured stroke width. |
| PARTIAL | ★ Post-erase verification: re-run the detector and OCR on the cleaned plate and widen the mask on hits. Residual-stroke checks exist; no full detector/OCR re-run on the cleanplate found. |

## 3. OCR

| Status | Idea |
| --- | --- |
| DONE | PaddleOCR v6 detection and recognition. |
| DONE | Hayai crop reader plus projective reader presentations (opt-in evidence; readings stay unselected). |
| DONE | manga-ocr as a re-read model (`--manga-reread-model`). |
| PARTIAL | Vision-language OCR per crop (`qwen3.8` re-read; `gemma4:26b` reading trials were diagnostic only). Qwen3-VL-32B and PaddleOCR-VL not tried. |
| NEW | ★ Pick readings by consensus plus a Japanese language-model plausibility score. Would let projective/Hayai readings be promoted on evidence, not agreement alone. |
| NEW | ★ Script-mismatch check: kana/kanji-looking glyphs read as `TP`/`G`/`2` are re-read rather than accepted as unchanged targets. |
| PARTIAL | Crop plus surrounding context (contextual projective views and native-plus-scene presentations exist). |

## 4. Grouping, reading order and unit type

| Status | Idea |
| --- | --- |
| NEW | ★ Unit-type classification (dialogue, narration, annotation, SFX, UI label, sign, credit), each with its own prompt and render style. |
| PARTIAL | Reading order: heuristic reading order exists; no learned or vision-model ordering. |
| PARTIAL | Panel boundaries as hard constraints (some panel logic in `grouping.py` / `aggressive.py`, part of it uncommitted). |
| NEW | Graph-based grouping with balloon membership as a hard constraint. |
| PARTIAL | Furigana/annotation attachment (furigana handling in `recognition_units.py` / `automatic_units.py`; gaze annotation still splits). |

## 5. Translation

| Status | Idea |
| --- | --- |
| DONE | TranslateGemma 27B, official prompt, greedy decoding, slash-alternative retry. |
| PARTIAL | ★ Model comparison. Tried: TG27, Qwen3.5 27B (`qwen3.8`), Gemma 4 26B, Gemma 3 4B/12B. Not tried: Qwen3-32B text, Qwen3-30B-A3B, Mistral Small 24B, Aya Expanse 32B, Shisa/Japanese-tuned models. |
| PARTIAL | ★ Image-aware translation with a vision-language model (tried via the generalist route in `docs/cropped-localization-context-experiment.md`; not production). |
| PARTIAL | Glossary, character list, previous-page context. A `glossary` field exists in `contracts.py`, but the TranslateGemma template rejects custom glossaries. No auto-built series memory. |
| NEW | Generate N candidates or use several models, then rerank with a quality-estimation model (CometKiwi-style; check licence) plus fit. |
| NEW | Back-translation drift check for additions and omissions ("without any permission", "user-friendly"). |
| NEW | Fit-constrained shorter rewrite when text overflows. Conflicts with the current exact-target-wording rule; would need its own source-bound revision. |
| NEW | QLoRA fine-tune of a ~24–32B model on licence-clean Japanese–English manga pairs (after an evaluation set exists). |

## 6. Inpainting

| Status | Idea |
| --- | --- |
| DONE | Flat fill plus classical restoration; aggressive / best-effort routes. |
| PARTIAL | LaMa ONNX (512) via the reconstruction config: evaluation and private use only, licence unresolved. |
| NEW | ★ Manga-tuned LaMa (`mayocream/lama-manga` is in the licensing doc, not in code), run at native resolution with overlapping tiles. |
| PARTIAL | ★ Route by region type. Exists: flat fill, Laplace-style fill (`aggressive.py`), phase-locked periodic screentone repair (axis-aligned only). Missing: rotated/nonstationary tone, and a learned route for painted colour art. |
| TRIED | Diffusion inpainting (RORem/SDXL deferred for cost and licence). Candidates: FLUX.1 Fill dev (fp8), SDXL inpaint, BrushNet/PowerPaint, Qwen-Image-Edit. Matters for colour pages such as Pepper&Carrot. |
| NEW | MI-GAN / MAT as a fast learned route for easy regions. |
| NEW | Redraw panel rules and long lines geometrically before inpainting. Targets the panel-rule and hair collateral damage. |
| NEW | Run 2–3 inpainters and pick by score (seam continuity, residual text, texture statistics). |

## 7. Typesetting

| Status | Idea |
| --- | --- |
| PARTIAL | Font by unit type. Only `dialogue-bold` is configured; no SFX, narration or UI faces. |
| PARTIAL | Shape-aware layout (`layout_enclosures.py`, `balloon_flow.py` experiment); hyphenation in `natural.py`. |
| NEW | Copy original text colour, outline and rotation (matters for colour pages and SFX). |
| NEW | Use the shorter rewrite (section 5) instead of failing below the minimum font size. |

## 8. Evaluation

| Status | Idea |
| --- | --- |
| PARTIAL | ★ Gold set. `quality50`, structural evaluation and provisional references exist; no independent, untouched gold set with per-unit masks and accepted English. |
| PARTIAL | Automatic metrics. Outside-support pixel checks and replay audits are done. Missing: residual-Japanese recall on cleanplates, OCR character error rate against gold, grouping F1, COMET-style scores. |
| NEW | A–B comparison view in the dev frontend with accept/reject keys. |
| NEW | Soak test (100+ pages) with VRAM plateau logging. |

## Suggested order

1. Gold set plus residual-Japanese metric (section 8).
2. Script-mismatch check plus consensus reading selection (section 3): cheap, and directly hits the unchanged `TP`/`G` failures.
3. Unit-type classification and panel/balloon hard constraints (sections 1 and 4).
4. Page-level translation comparison: TG27 versus a Qwen VL model with the image, plus a back-translation check (section 5).
5. Inpainting: manga LaMa tiled at native resolution, line redraw, diffusion for colour art (section 6).
