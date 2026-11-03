# Coverage-first replacement

The default is now **`--render-policy best-effort`** for `src/main.py --render`,
`tools/render_preview.py auto-compose`, and the automatic rendering API. This
implements the requested preference: if conservative cleanup/layout fails, try a
lower-confidence replacement instead of immediately retaining the original text.

Use **`--render-policy conservative`** to retain the previous behavior, including
strict minimum font sizes and skipping uncertain masks/backgrounds. Manual and
assisted rendering workflows are unchanged.

## Automatic fallback order

1. Keep the existing measured flat fill, classical restoration, and optional
   digest-pinned CTD/native-mask/periodic routes. Successful replacements are not
   changed by the new fallback.
2. If cleanup succeeded but typography failed, first reuse that cleanup with less
   whitespace padding. Next try smaller type, down to **8px**, without changing
   target words or discarding characters.
3. Try a tighter **native contrast hint** with whole source-associated strokes and
   measured flat fill. This is explicitly a grayscale heuristic, **not** learned
   text probability or proof that a component is text. Ambiguous same-color art
   can still be erased, so these outputs are marked best-effort.
4. If necessary, treat the OCR polygons plus 1px as an **uncertain edit footprint**,
   excluding known neighboring detections and already accepted output. Compare
   bounded Navier–Stokes and Telea restoration using the same mask/context. A
   seam/color heuristic chooses a candidate; it does not certify hidden artwork.
   If both backends fail, use a measured/guessed color backing rather than leave
   a known translation unplaced. `--background-method flat` uses backing instead
   of classical inpainting in this stage.
5. Prefer locally measured low-contrast placement space. If it cannot fit the
   complete translation, permit a bounded source-local **art overlay**, with
   contrasting outlined text. Emergency line breaks may split long words, but
   never remove characters, punctuation, original whitespace or original line
   breaks. Their exact insertion offsets are recorded and checked.

The fallback never commits cleanup until lettering, containment and pixel audits
succeed. Conservative collision rejection remains unchanged; afterward, the
coverage-first pass reserves accepted outputs and processes remaining failures
in saved order. It does not sacrifice an already accepted replacement to make
space for a weaker one.

## Explicit quality tradeoff

**Best-effort edits can damage artwork, balloon outlines, texture or hidden line
structure. They can use smaller type, awkward wrapping, or lettering over art.**
This is a coverage-first approximation, not invisible-background reconstruction.
In particular, an OCR polygon is not a certified lettering mask.

Every degraded unit has `render_quality: "best_effort"`, its original failure,
fallback attempts, actual cleanup method, warnings and typography decisions.
The HTML review prominently marks these edits. `quality_summary` separately
counts conservative and best-effort replacements, with `has_degraded_output`;
`pipeline.json` includes the same summary in its rendering section. Nothing is
declared approved, artistically verified or linguistically verified.

## Requirements that remain hard

- A current, source-bound **full translation** must exist. Missing translations,
  abstentions and text-only review fragments are not invented or promoted into
  full dialogue. This rendering policy does not add provider calls or override
  translation budgets.
- Exact supplied target characters, original file bytes, licensed font bytes and
  source/preview/geometry bindings are preserved. Missing glyphs or invalid/stale
  input still fail explicitly; there is no silent font lookup or text truncation.
- Known neighboring text and accepted output remain protected. A coincident
  detection with no independent source support may still prevent replacement.
- Pixel changes outside the declared erase support and actual lettering alpha
  remain forbidden. The allowed edit support is larger/weaker for risky edits,
  so outside-mask equality is **not** an artwork-preservation certificate.
- Resource caps and explicit evaluation caps remain enforced. Best-effort local
  restoration is <=262,144 pixels / 1024px per side; placement expansion is at most
  64px horizontally and 40px vertically around source geometry. At most two
  best-effort classical backend calls share the same mask. Typography retries
  share a bounded work budget. There is no full-page generative redraw.

Thus this is not a promise of replacement for corrupt input, unavailable
translations/fonts or exhausted budgets. It removes *artistic uncertainty* as an
automatic reason to stop trying, not fundamental execution requirements.

## Saved-page evidence

`test-results/coverage-first-parent-v2/` renders **8/8 existing strict translations
without a CTD option**: six conservative replacements and two marked best-effort
native-hint replacements. The previous six unit PNGs remain unchanged. The page
still contains 21 blocked/untranslated saved units; this is not a full-page
translation claim. No new OCR/translation/provider calls were made for this test.

- [Current review](../test-results/coverage-first-parent-v2/review.html)
- `manifest.json`: exact words, source/model/font evidence, masks, failures and
  quality flags.
- Original file, source/clean/text/composite layers and per-unit masks are kept.

The earlier `coverage-first-parent-v1/` uses the coarse footprint route and is
retained as lower-quality trial evidence. Tighter native contrast hints and
compact placement were kept because they preserved more of the visible source.
Neither real run has a clean-background reference, so no invented reconstruction
error or universal art-preservation claim is made.
