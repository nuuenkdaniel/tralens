# Always show available translations

## Product policy and pre-implementation hypothesis

Explicit opt-in `always-show` visibility prioritizes readable supplied English
over untouched artwork. Existing replacement policy choices remain available and
defaults/old runs do not change. Clean replacements retain their native/flat/gray
guards. Afterwards, a separate bounded pass may place opaque backing over a local
source footprint, or a nearby caption with a leader when source placement fails.
Neither fallback requires a native cleanup mask or a raw-score threshold.

No reconstruction, extra inference, translated wording changes, slash-menu
selection or font-floor relaxation. Source/result bindings remain mandatory.
Keep all existing clean edits first, protect other detections and edited lettering,
and declare every fallback patch/leader pixel. Captioned Japanese stays visible;
displaying English is not a clean source replacement or semantic approval.

Freeze `test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/` successful
layers before editing. Renderer-only evaluation must display the saved network
translation and the saved title in the partly translated balloon at >=16px,
without changing any old successful text/foreground/erase PNG. These are evaluation
inputs, not production text/ID/coordinate rules. Compare full pages at reading size
and crops; include reused pages with annotations, effects and textured art.

Explicitly report clean replacements, opaque patches, nearby captions, unchanged
text, available translations still undisplayed, and missing translations. Keep
source visibility, artwork occlusion, placement failures and separate budgets
inspectable. No claim of guaranteed 100% display or linguistic accuracy.

The existing broad route is not reused as this policy: its two classical repair
trials and below-requested-size emergency fitter are not deliberate opaque backing.
Reuse source contracts, approved fonts, legal-break typography and pixel checks,
not guessed background reconstruction. OCR/translation remain unchanged.

## Implemented policy

`visibility_policy='always-show'` is independent of `replacement_policy`. It is
not another name for the old broad-replacement/inpainting route. The default is
`replacement-only`; existing conservative choices and old jobs keep their policy.
The source-confidence threshold still controls clean replacement, **not** this
display fallback. Only valid source-bound primary translations are consumed;
missing outputs are never invented, and exact source/target identity is kept
intentionally unchanged. Fragment/alternative metadata remains untouched.

1. Existing native/validated cleanup routes run first. In the new mode the old
   coarse-footprint inpainting fallback is not attempted and below-requested-size
   fits are not accepted. Normal clean guards remain unchanged.
2. After clean composition and audit, `visible_translations.py` tries an opaque
   white text-sized patch within the source-local allowance. It does not need a
   foreground mask, flat interior, antialias agreement, or minimum raw OCR score.
3. If placement/readability/collisions prevent a source patch, up to four bounded
   caption shapes and eight nearby positions each are tested. Captions have an
   opaque warm backing, border, visible leader and source anchor. Their original
   source remains visible. Geometry is association, not reading/speaker inference.
4. Preserve every native erase/lettering reservation, every other known source
   polygon (with clearance), page bounds and previously accepted fallback. Text
   uses existing legal-break layout and the exact supplied characters. Neither
   emergency smaller type nor manual phrase selection is used.

The separate budget is 32 fallback attempts, 3,072,000,000 bounded raster-work
units, 262,144 pixels per local layout, and 65,536 pixels per backing/leader
footprint. It also reserves publication slots for all native artifacts before
allowing fallback artifacts. Layout-failed regions are scheduled before remaining
valid translations; raw scores only prioritize scheduling, never authorize/deny a
display. Failed fits/budgets keep their original translation and a reason. An
optional display-pass failure rolls back overlays only, never native successes.

`clean.png` remains the native cleanup layer. `patch.png` contains opaque backing,
borders and leaders; `text.png` contains native plus fallback English. Their alpha
composition exactly reconstructs `composite.png`. `native-text.png`,
`native-composite.png`, `fallback-text.png`, `fallback-mask.png` and per-unit
fallback patch/text/mask PNGs retain intermediate evidence. Outside the union of
declared fallback pixels, the result must equal the native composite exactly.
Native alpha and erase pixels must remain unchanged. Original contrast covered
and footprint pixels outside source polygons are recorded as **occlusion
observations**, not a semantic count of damaged artwork.

The old `coverage.rendered` stays a clean-replacement count; new `display_summary`
and `coverage.displayed_translations` report display separately. Captions never
increment clean replacements. Per-region `display_status`, `source_visibility`
and `display_fallback` retain native rejection, exact target/font, backing bounds,
leader, footprint size, source coverage and unresolved reasons. Clean pixel audits
are explicitly scoped before overlays; final alpha/locality audits describe the
additional display layers. No human/linguistic or reconstruction approval.

## Viewer and coordination

The viewer exposes **Always show available translations** with the exact warning:
“Prioritizes visible translations; fallback patches may cover artwork.” It is an
explicit rendering-only option. It persists through upload, retry, metadata,
legacy pipeline-based history restoration and `src/main.py` execution. It does
not silently enable source localization, switch models, or change clean policies.
Final-stage badges distinguish opaque patches and captions with retained source;
inspection links show each footprint, backing and text. Translation and display
counts remain separate. The mode is also available to offline rendering and CLI.

No live OCR or translation was rerun to evaluate this task. HTTP/coordinator tests
use synthetic readers/providers; renderer-only acceptance uses saved real inputs.
The existing viewer was checked for active jobs (none), then restarted on the
same **0.0.0.0:7860**, with the same model/font/endpoint/readers and history. Its
old saved example still restores as `replacement-only`, without inference.
The pre-task network-binding functions and network tests are unchanged. A first
inspection helper lacked `os.pidfd_open` in the project Python and sent no signal;
the existing system Python then waited for a clean idle shutdown before restart.

## Frozen renderer-only evaluation

`test-results/visible-translations-freeze-v1/` holds the pre-task successful
layers, previews, originals, fonts/license evidence, application sources and
unrelated working-tree snapshot. Final-source evaluation and its code snapshot:
`test-results/visible-translations-evaluation-v2/`. All are local/ignored evidence;
none was force-added. `tools/evaluate_visible_translations.py` consumes explicit
JSON case paths/options and always writes fresh artifacts.

| Reused case | Clean | Patch | Caption | Unchanged | Available, not displayed | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact supplied saved example | 16 | 12 | 3 | 4 | 4 | 1 |
| `ubunchu04_0002` (annotations/furigana) | 7 | 2 | 4 | 2 | 1 | 3 |
| `ubunchu04_0005` (effects/mixed scripts) | 10 | 4 | 0 | 1 | 0 | 1 |
| `ubunchu06_0002` (screen text/effects) | 8 | 4 | 4 | 3 | 1 | 0 |
| `ubunchu08_0007` (dark/textured art) | 6 | 3 | 10 | 3 | 3 | 2 |
| Total | **47** | **25** | **21** | **13** | **9** | **7** |

All **47 prior clean replacements**, their exact bindings/target strings and
foreground/erase/text PNGs remain unchanged. Clean pages are byte-identical.
With the new mode off, **all 750 baseline PNGs** across these cases are identical.
All 46 degraded displays meet the requested minimum (16px in these cases).
All nine unresolved available translations failed bounded collision/placement,
not a cleanup-confidence veto. Missing and unchanged inputs stay in the table.

### The two acceptance examples

- Network: exact saved `It also automatically connects to the network.` is
  visible on an opaque patch at **17px**, despite `localized_gray_interior` on
  clean replacement. Japanese remains above/below the text-sized patch; this is
  reported as partly covered source, **not** a clean balloon replacement.
- Title: exact saved `Chairman` is visible at **16px** in an associated caption
  beside the narrow original column, with a leader to it. The Japanese remains.
  The successful `I think even I could use this.` text/mask stays identical.

Original/baseline/candidate full pages were inspected at ~620px reading width,
and both acceptance crops were inspected at native resolution. The backing
clearly improves access to those saved English strings, but is intentionally
not seamless. It obscures some artwork/effect strokes, panel/screen details and
drawings inside declared footprints. Some Japanese protrudes around patches;
captions retain it deliberately. Caption placement can be awkward or cross into
a nearby margin. Known neighboring text and all old clean output are protected;
undetected text/art inside the allowed footprint is not certified safe.

The new mode also exposes pre-existing bad/awkward wording (for example `Wisdom.`,
`Fifth grade.`, `Hehe, poop.` and `APT (Advanced Persistent Threat)` in these saved
cases) and unedited slash alternatives. This is a display-policy gain, **not** an
OCR correction, linguistic improvement or independent held-out accuracy result.
No new mask, source confirmation or model approval was required.

Renderer-only work made **zero inference calls**. The final evaluation took 55.02s
including both off/on renderer passes for five pages; the additional display pass
totaled 2.34s. Tests ran concurrently, caches/load were uncontrolled. These are
execution observations, not a general speedup or throughput claim. Footprints
total 260,481 pixels across the five pages; this is not an artwork-damage score.

## Verification and use

**945 tests pass** in each existing environment; source-only passes with 39
optional-local-evidence skips. Sixteen new portable tests cover opaque fallback
without cleanup, narrow-column caption association, exact target preservation,
neighbors/prior edits/page bounds, no smaller font, mode-off identity, stale source,
separate-budget exhaustion and rollback, low-score and conservative-clean display,
layer composition, separate counts and HTTP upload/retry/history persistence.
Existing network-binding tests remain user-owned and pass unchanged. Node syntax
and view-state notices pass too.

Saved-example reproduction (fresh output; no OCR/translation):

```sh
.venv/bin/python tools/render_preview.py auto-compose \
  --preview test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/run/rendering/evidence-preview.json \
  --image test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/input.png \
  --fonts test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/run/rendering/evidence-fonts.json \
  --output test-results/my-fresh-visible-run \
  --replacement-policy confidence-gated --layout-policy enclosed-retry \
  --cleanup-policy localized --visibility-policy always-show
```

Those clean-route options reproduce this baseline; they are **not prerequisites**
for opaque fallback. A normal pipeline uses `--render --visibility-policy
always-show`. The live viewer is `http://100.121.133.88:7860/`; the read-only
five-page original/baseline/candidate comparison is
`http://100.121.133.88:7863/`. Gallery routes allowlist only comparison images,
not repository files. Old runs, raw OCR, alternatives and translations are intact.
