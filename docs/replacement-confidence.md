# Experimental replacement confidence

`--replacement-policy confidence-gated` is an **opt-in image-edit policy**, not a
translation filter. Default `all` retains the existing best-effort behavior.
Coverage-first translation, original OCR records, source bindings, exact targets,
alternatives and saved intermediate layers are unchanged.

The policy uses the minimum bound raw **main-line** OCR score, not a mean that can
hide a bad line. Only raw readings can use those scores; other recognizer readings
do not inherit Paddle confidence. Missing, malformed or out-of-range scores defer
the edit. Annotation scores are not treated as main-word accuracy.

Passing the score threshold is not sufficient:

- Native stroke cleanup/layout routes must succeed. Coarse OCR-footprint repair
  and its art-overlay fallback are disabled, even at an OCR score near 1.0.
- Lettering must fit at the requested minimum size; emergency tiny type is not
  accepted by this policy.
- If target text exactly equals the selected source string, preserve the original
  lettering. No normalization or language/script-based exclusion is used. This is
  a no-edit outcome, not proof that OCR or the target is correct.
- Existing integrity, neighboring-output and resource guards remain mandatory.

Every deferred edit has a reason and retains its bound translation. The manifest,
HTML review and pipeline checkpoint report image replacements separately from
available translations and preserved source text. Partial output stays partial.
This policy does not establish linguistic correctness or semantic text masks.

## Provisional calibration

The user authorized direct model visual review. The original/composite and crop
pairs on the previously exposed saved page were visually assessed by GPT-6.1 Sol.
The local scoring-only ledger has 29 source/baseline-bound records, including
uncertain translation/remnant cases. It is **not an independent reference** and
is never imported by production or supplied to OCR/translation.

`tools/calibrate_replacement_confidence.py` compares a fixed threshold grid. The
selection rule is the lowest grid threshold reaching 0.90 agreement precision on
assessed OCR visual labels; it reports mismatches, deferred matches, uncertain
labels and all-unit denominators. That development objective is not a calibrated
accuracy probability or statistically validated reliability level.

**0.70** was the first passing grid value: 17 visually matching readings and one
visually mismatching reading among 18 score-passing units. **0.70–0.85 made the
same selections on this page**; an exact optimum cannot be inferred. Increasing
to 0.95 deferred more useful text without removing the high-scoring bad label.
The cleanup constraint catches cases that a numeric OCR cutoff cannot.

Actual rerender, with no new OCR/translation calls:

| Outcome | Existing best effort | Confidence gated |
| --- | ---: | ---: |
| Saved translations | 29 | 29 |
| Actual replacements | 29 | 10 |
| Best-effort replacements | 23 | 4 |
| Original-lettering identity no-edits | 0 | 5 |
| Low-score edit deferrals | 0 | 10 |
| Native cleanup/layout unsupported | 0 | 4 |

Against the **baseline visual operation labels**, the gate selects eight judged
acceptable and two uncertain operations, and none of the 15 judged unacceptable
operations. It also defers one judged useful `Hey` replacement. This is a
selection comparison on development evidence, not independent approval of the
new page. The final rerender was visually inspected too: large smears are absent,
but the instructions, LiveCD dialogue and several effects remain in Japanese.
The selected `Okay` translation is still semantically questionable, and a small
source gloss remains. **Fewer edits are not an OCR fix or complete translation.**

All 50 retained per-unit text/foreground/applied-erase/layout/content PNGs are
byte-identical to those operations in the baseline. Default-policy rerendering
also retains all 444 baseline PNGs. Pixel audits still report zero outside-mask
and protected changes; they are integrity checks, not hidden-art truth.

Local ignored evidence:

- `test-results/replacement-confidence-vision-v1/observations.json`
- `test-results/replacement-confidence-grid-v1/report.json`
- `test-results/replacement-confidence-render-v1/review.html`
- `test-results/replacement-confidence-comparison-v1/review.html`
- `test-results/replacement-confidence-all-regression-v1/`

## Use on an existing saved preview

```sh
.venv/bin/python tools/render_preview.py auto-compose \
  --preview PATH/preview.json --image PATH/original.png --fonts PATH/fonts.json \
  --render-policy best-effort --replacement-policy confidence-gated \
  --min-ocr-confidence 0.70 --output NEW_OUTPUT_DIRECTORY
```

The same options are available with `src/main.py --render` and the Python
coordinator. The running viewer was not restarted or changed to enable this
experiment. No provider/model/font/dependency acquisition or remote image upload
was added. Before any default promotion, use fresh held-out pages and independent
transcription/quality checks; a threshold from this one Paddle-v6 page is not
portable calibration for other recognizers.

Checkpoint verification: 823 tests pass in `.venv`; the new portable tests cover
weakest-line binding, exact boundaries/identity, alternatives, malformed scores,
unchanged defaults, coverage retention, cleanup vetoes, readability, offline
comparison and labels staying out of production. The existing user-owned
`pyproject.toml` and `uv.lock` edits are excluded from this checkpoint.
