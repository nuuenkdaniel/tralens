# Balloon-first dialogue experiment

## Hypotheses frozen before implementation

Baseline: `afe40c2`. Keep the committed visible-translations safety net and all
user-owned ignore/dependency/network/staged-removal changes untouched. Do not
restart the viewer. Saved real evidence is reused, not held-out accuracy data.

1. Detect bounded light interiors from original pixels and source geometry before
   asking for any native glyph mask. A closed boundary and a measured, uniform
   background may support a new cleanup hypothesis where the old glyph-first
   route fails. This is not the existing layout-only enclosure retry, a changed
   gray-antialias cutoff, or permission to fill a whole balloon.
2. Maintain separate records for raw boxes, balloon candidates, utterance
   membership/order/annotation roles, pixel ownership and English flow frames.
   White connectivity alone cannot join utterances. Connected lobes, detached
   addresses and separate interjections need bounded linguistic proposals as well
   as geometry; uncertain proposals do not establish truth or speaker identity.
3. Changed groups use new source-bound immutable requests through the translation
   stage. Old units/results/raw evidence remain immutable. Activate a new group
   only atomically with a valid translation, supported cleanup and readable layout;
   otherwise keep the old edits/translation displays. Do not render old and new
   membership twice. No English manipulation in rendering.
4. Own only independently bounded original contrast inside assigned source
   geometry, with surrounding background, boundary clearance and glyph-run
   support. Reject gradients, illustrations, unassigned ink, shared detections,
   open/page-connected interiors and ambiguous geometry. Record exact footprints
   and outside-pixel equality; no arbitrary blanket fill or art reconstruction.

Freeze motivating and representative saved pages before tuning: ordinary light
dialogue, adjacent/joined balloons, detached address, interjections, furigana,
effects/illustrations, screen art and nonuniform surfaces. Synthetic negatives
must include art inside detector boxes, not only outside them. Inspect original,
native and candidate full pages and native-resolution motivating crops. Report
grouping, remaining Japanese, boundary/art damage, typography and inference cost
separately. A failed hypothesis is a reportable result, not a reason to weaken a
previously rejected rule. Do not silently change defaults or old runs.

## Implemented opt-in experiment

Entry point: `tools/balloon_experiment.py`, coordinated by
`src/main.py:run_balloon_experiment`. It accepts a saved preview, original image,
approved fonts and native rendering directory. It never reruns page OCR or
changes the normal viewer pipeline. Source/request/result bindings, baseline
artifact hashes and layer composition are revalidated. Outputs must be fresh.

Separate representations/stages:

- `tralens/balloons.py`: raw detector snapshots, closed light-interior hypotheses,
  background samples and protected boundary/core. Source masks and the bounded
  geometry cache are bit-packed; candidate discovery does not retain full-page
  masks for every possible region. Geometry alone establishes neither utterance
  membership nor annotation roles, speakers or erase permission.
- `ocr/balloon_dialogue.py`: at most four original crops/page, <=300,000 pixels
  each. Existing Qwen proposes a surface category and partition/order of whole
  original units. No source correction, annotation reassignment or English.
  Separate replies/interjections stay independent; malformed proposals reject.
- `translation/balloon_dialogue.py`: changed groups get new immutable utterances
  and requests bound to parent revisions, exact raw roles/order, annotation
  parents and the proposal. Japanese main readings are joined with explicit
  newlines; English is never joined. Original readings/alternatives/units remain
  in parent evidence. Known unreadable sources and ambiguous roles cannot be
  waived. Only changed groups go through existing TranslateGemma, <=4 calls/page.
- `rendering/balloon_clean.py`: independently supported original printed-cell
  cleanup, with no native-mask prerequisite or OCR/translation/model invocation.
- `rendering/balloon_flow.py`: English frames separate from detector columns and
  cleanup ownership. Full supported space first, then bounded anchored/shallow
  lobe rectangles if the common center is occupied. <=6 frames; exact target and
  requested font floor stay mandatory. Frame selection changes no cleanup pixel.
- `rendering/balloon_pass.py`: atomic activation. A changed group only suppresses
  its old parent renderings after fresh translation, cleanup and layout all
  succeed. Historical units/results remain. Failure keeps old edits active, and
  duplicate-member checks prevent old/new membership rendering twice. Portable
  tests exercise this full path with synthetic providers.

### Distinct cleanup evidence and limits

Interior/background are estimated **before glyph ownership**, using four original
light fields (220/245 thresholds, 4/8 connectivity). Their bounded closed-envelope
intersection is used, never a union across uncertain connections. Broad disputed
lobes reject. Thin boundary disagreements/spurs remain protected by intersection
and the 4px inset, not erased. Spatial tiles must support one measured neutral
background within 2 levels. This is currently a **near-white uniform-interior**
experiment, not general off-white/gray/gradient reconstruction.

Raw character count, orientation, regular pitch, blank seams and original blank
collars must support complete printed cells. Cleanup covers only these cells,
not an enclosure. Independent original whole-contrast checks veto clipped/shared
strokes and rim-connected ink. Solid/colored cells, smooth enclosed pictograms,
unassigned dark ink, substantial unexplained light marks, gradients, illustrations
and unsupported pitch reject. Small unassigned light marks stay protected.
Qwen plain-surface evidence is an additional **uncertain proposal**, not human
verification, and cannot override these pixel rejections.

This is not a relaxed gray-interior/single-ink cutoff, percentage-fringe clipping,
native-mask enlargement or the old successful-mask-dependent enclosure retry.
Those methods/default guards are unchanged. The cell route can still mistake
sufficiently glyph-like art for print; it is experimental machine ownership, not
a semantic safety guarantee.

If Qwen keeps units independent, their geometry can still coordinate English
frames without joining their language. Every source cell in the common interior
must first pass original-cell checks. Only the selected unit's complete cells are
erased; other cells and native English remain protected. Connectivity alone does
not merge sources or authorize cleanup.

Separate budgets: four proposal and four new grouped-translation calls, eight
cleanup attempts, 512,000,000 aggregate cell-work units, 256,000,000 printed-cell
raster units/proposal, six frames with existing 64,000,000 raster limit each,
and <=96 artifacts/128 MiB publication. Old native budgets/reservations precede
this pass. Artifacts retain original/preview/font evidence, exact cleanup,
boundary/core/flow masks, text, changed footprints and historical bindings.
Outside declared changes must equal the baseline; all unsuperseded native erase
and English pixels must remain exact.

The original clean-replacement source-confidence choice is also retained: when
the baseline is confidence-gated, every parent must independently meet that same
gate. New geometry or grouped translation cannot inherit/raise a failed raw
score. This gate still does not block the separate opaque visibility safety net.

The committed `visible_translations.display` is reused **unchanged, afterward**.
Clean, patch, caption, unchanged, available-undisplayed and missing counts remain
distinct. Opaque safety-net backing is not called clean balloon reconstruction.

## Results on reused saved evidence

Pre-tuning freeze: `test-results/balloon-first-freeze-v1/` (830 files, including
five reused originals/previews/native layers and pre-existing work).
Final-source replay: `test-results/balloon-first-evaluation-final/`, with source-code
snapshot and per-page `comparison-full.png`: original, native, clean experiment,
then the existing safety net. These are not held-out accuracy measurements.
Evaluation strings, IDs, names and coordinates are not production special cases.

### Motivating page

- Network: exact saved English is cleanly lettered at **19px**. Its 17 printed
  source cells use **5,904 cleanup pixels**. Balloon/panel boundaries, adjacent
  screen art and the tiny original boundary mark stay protected. No opaque patch
  or guessed artwork repair is used in this balloon.
- Address: exact saved `Chairman` is cleanly lettered at **16px** in the lower
  lobe, using **1,200 cleanup pixels** for two source cells. The main sentence is
  unchanged. A shallow supported frame fits where the narrow source footprint
  and the common mask's occupied center did not.
- All **16 prior clean replacements** survive. Final display: **18 clean,
  11 patches, 2 captions, 4 unchanged, 4 available-undisplayed, 1 missing**.
  Total displayed stays 31: one patch and one caption became clean, not merely
  higher visible coverage.

**Incomplete result: the address is not linguistically regrouped.** The first
Qwen pilot produced malformed multi-label `independent` groups; those were
rejected, not repaired into the desired answer. A general protocol clarification
(independent means singleton; detached vocatives differ from replies) yielded
valid partitions, but Qwen still kept this address separate because it considered
vocative-versus-label ambiguous. This remains unresolved. There was no manual
override, forced English join, source-ID rule or promotion to verified evidence.

| Reused case | Native clean preserved | Added clean | Main limitation retained |
| --- | ---: | ---: | --- |
| supplied saved page | 16 | 2 | uncertain address grouping; screen labels unsupported |
| `ubunchu04_0002` | 7 | 0 | no unresolved eligible supported interior |
| `ubunchu04_0005` | 10 | 0 | effect rejected as non-dialogue/nonuniform |
| `ubunchu06_0002` | 8 | 0 | separate reaction retained; unsupported pitch |
| `ubunchu08_0007` | 6 | 0 | competing text and unsupported pitch |
| **Total** | **47** | **2** | no general linguistic-accuracy improvement established |

The other four clean composites remain identical to native baselines. All 47 old
clean operations survive. No real old parent was superseded, since no real merge
was accepted. Qwen kept the sentence/reaction on the sixth-chapter page separate.
It rejected the supplied page's screen-label candidate; words invented in its
reason did not change the actual source. Furigana roles were not reclassified.

All five full pages were inspected at reading size, and both motivating balloons
at native resolution. No Japanese remnants or boundary/adjacent-art damage were
visible in the two new cleaned areas. The main sentence/title remain separate
English blocks: this is a modest clean-placement gain, not proof of coherent
dialogue translation. Existing source remnants, slash menus, awkward wording,
effect omissions and art obscured by the safety net remain visible separately.

## Inference and verification

Real work used **10 Qwen crop calls total**: three in the malformed pilot, three
in the clarified pilot, four on additional pages. **Zero page OCR calls, zero
translation reruns for unchanged membership, zero real grouped TranslateGemma
calls** (no real merge accepted). Final replays make no inference calls. Shared
Ollama work was serial after checking viewer idle. The viewer was not interrupted,
restarted or reconfigured. Existing configured tags were used, with tag identity
recorded as unauthenticated weights. Timings are diagnostics, not benchmarks.

Portable tests cover original-only cleanup and locality; gradients, in-box solid/
colored art and pictograms; out-of-box illustration; open/adjacent geometry;
clipped/shared strokes; stale masks/parents/results; retained annotations;
interjection partitions; unreadable/ambiguous roles; new grouped-source
translation; atomic suppression, overflow/budget rollback; exact targets/font
floor; no-inference replay; corrupt old artifacts; coordinated independent frames;
and visual-art rejection. Real linguistic regrouping is **not** validated by those
synthetic provider tests.

Final verification: **965 tests pass in both existing environments**; source-only
also passes with 39 optional-local-evidence skips. The 20 new tests are portable.
All 15 final clean/clean-composite/fallback-composite PNGs match the inspected
five-page images exactly; the four nonmotivating clean composites are identical
to their native baselines. Final source hashes match the replay's frozen code.
Of the 830 pre-task frozen files, only the owned coordinator file changed;
original saved artifacts, fallback code, dependency/ignore edits and network work
are byte-identical. Original staged documentation removals remain staged. No
files were force-added, no commit/push was made, and the viewer was not restarted.

### Reproduce without inference

```sh
.venv/bin/python tools/balloon_experiment.py \
  --preview test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/run/rendering/evidence-preview.json \
  --image test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/input.png \
  --fonts test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/run/rendering/evidence-fonts.json \
  --baseline test-results/dev-frontend/5ef2f588a3ef42d6ba24743c2f21819e/run/rendering \
  --replay test-results/balloon-first-pilot-v2/experiment.json \
  --output test-results/my-fresh-balloon-experiment
```

For fresh bounded proposals, omit `--replay` and explicitly add
`--run-local-model`; only changed groups require translation. Visibility defaults
to `always-show`; `--visibility-policy replacement-only` shows clean results
alone. No manual mask/correction/per-region confirmation is needed.

**Do not promote to a default yet.** Two cleanup/layout gains are concrete, but
real grouping improvement is unproved and four other pages gained no new clean
edits. Independent grouping/illustration negatives and successful real grouped
translation evidence are needed before normal viewer integration. Existing
models/fonts/dependencies, raw OCR, original units/translations, runs, network
binding and fallback implementation stay intact. No upstream implementation was
fetched/copied; no clean-room guarantee, push, publication, acquisition or
licensing/visibility change is claimed.
