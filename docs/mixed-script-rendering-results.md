# Exact mixed-script replacement checkpoint

Local evidence: `test-results/production-best-effort-v8/font-render-v3/`.
This fixes a display omission, not the incorrect source reading behind it.

The previously undrawn target `Try adding “Of” after “aこPー”.` is now rendered
exactly at **17px**, above the requested 16px floor, with its source text removed.
The original, previous composite, new cleanplate and new composite were visually
inspected together in `mixed-target-review.png`. The source actually refers to
ACPI/off; correcting that language remains separate work.

## Implementation

- Licensed, hash-bound font manifests may explicitly select a TTC `face_index`.
  A bounded sfnt copy keeps the selected face's tables/glyphs and repairs offsets
  and checksums. No system font search or font acquisition occurs.
- The manifest's optional `fallbacks` map defines an ordered list per requested
  face, for example `{"dialogue-bold": ["cjk-bold"]}`. A whole target uses the
  requested font whenever it has complete glyph coverage. Otherwise only these
  explicitly supplied faces are considered.
- Latin/CJK horizontal targets retain their exact characters. Missing glyphs,
  controls and unsupported characters remain failures; failed layout cannot
  cancel source removal.
- The selected face follows the target through native layout, best-effort layout,
  aggressive cleanup, final removal/backing and visibility fallback. Each row
  records its selected face and extracted font hash.

The local replay uses installed Noto Sans CJK JP Bold, collection index 0, with
its installed OFL license. Existing English targets keep Liberation Sans Bold.
Fonts and licenses are hashed in the retained manifest.

## Verification and limitations

- Six new offline tests exercise real synthetic TTC/sfnt bytes, selected-face
  glyph masks/metrics, checksums, malformed input, exact target layout, fallback
  policy, mixed-unit pipeline behavior and cleanup after missing-glyph failure.
- The working-tree suite passed **1,092 tests**. The earlier full-suite command
  was interrupted by the shell's 120-second timeout; that log is retained, and
  the completed run uses no shell time limit.
- Four pages were replayed with fresh font-bound localization requests. The
  stale-plan rejection and wrong-Paddle-environment trial remain retained.
  Crop OCR subsequently ran in the existing validated `.venv-paddle6` environment.
- Independent audit checked **1,327 artifact hashes**, exact clean/patch/text
  recomposition, exact target strings/font floors, and **zero changed pixels
  outside declared support**.
- Three final composites are pixel-identical to V7. On `ubunchu07_0002`, only the
  newly displayed target area changes (2,870 composite pixels). Every previously
  displayed glyph pixel is preserved; 967 pixels of its earlier blank cleanup
  differ in this fresh reconstruction. These are reported, not counted as exact
  preservation of the old cleanplate.
- Visible translated replacements rise from **98 to 99** across the four pages;
  no available target remains transcript-only. Original-identical targets remain
  explicitly unchanged. These counts are not a language-accuracy metric.

Serious reading/grouping/translation errors remain, including the newly displayed
ACPI line, copy-paste, clap sound effects and clipped software labels. The next
step is source-bound reader improvements and fresh TG27 translation, with all
earlier wording and failed trials retained.
