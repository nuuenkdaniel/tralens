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
   A later mixed-line fallback, described below, keeps the original strings/scores
   on unchanged raw lines and requires paired corroboration only on changed lines.
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

## Follow-up: the LiveCD / desktop panel

The next user screenshot correctly identified a remaining failure: both speech
balloons were still Japanese. Broad erasure was gone, but preservation alone did
not finish the translation. The outlined keyboard sound effect also remains
unresolved.

### Bounded mixed-line recognition fallback

`src/tralens/ocr/quality_lines.py` adds a fallback inside the **already opt-in**
quality reread stage, not a new default recognizer or provider:

- Retain exact raw lines scoring at least the provisional `0.70` routing floor.
  These are not newly certified readings. Their original individual scores must
  still meet the requested rendering threshold; a paired reading of another line
  cannot launder a weaker raw line into a higher-confidence result.
- For weak lines in otherwise eligible multi-line blocks, derive separate
  recognition presentations from whole original ink components. Competing source
  geometry, ambiguous ownership, empty images and excessive extents reject the
  presentation. Removed context is white **only in the OCR presentation**; these
  masks do not grant any erase permission and never modify the original image.
- Try a single target image, then target plus original block context, under a
  shared **four-call fallback vision budget**. All eight primary calls remain
  separately counted. Uncertain or uncorroborated outputs do not select a block.
- Compare changed lines with Manga OCR on a fixed set of bounded presentations:
  rotations `0, -4, 4, -8, 8` degrees, expanded without clipping, with white
  presentation padding. All calls, responses, rejected readings, transforms and
  original-pixel bindings remain inspectable. This is presentation diversity,
  not five independent recognizers or a confidence probability.
- Keep exact per-line text separately, then concatenate in the already bound
  geometric reading order, like the raw main-text input. Do not introduce image
  row boundaries as new sentence breaks. An early live run showed that inserting
  artificial newlines made the native translation unnecessarily fragmentary.

No expected Japanese words or English targets were added to either model prompt.
The probes also retained failures: Paddle and the visual reader remained poor on
some isolated handwriting, and Manga OCR misread the outlined sound-effect band.
Neither improved fluency nor trying more transforms is independent source truth.

### Native cleanup and placement remain separate

The right balloon passed native stroke/background/collar checks, but requiring
an **extra layout-only collar** to fit inside the already inset placement space
vetoed the entire operation. In confidence-gated mode only, the native fallback
can now seed placement with the complete foreground instead. Every original
foreground and erase pixel stays intact, obstacles/insets are unchanged, and
exact-glyph containment and requested minimum type size still apply. Genuine
border/collar contact remains a rejection. This is not smaller erasure, a relaxed
background model or an art-overlay fallback.

### Actual outcomes and limitations

The `quality-panel-live-v1` run newly rendered both balloons, but its artificially
line-separated translation input made the English choppy. `quality-panel-live-v2`
retained line boundaries in evidence rather than inserting them into the composed
input. A fresh real-browser retry then produced a byte-identical composite:

<http://127.0.0.1:7860/?job=8d5c4cdaf1af4569a12871265db76aa1>

The viewer used real local readers and the configured native translator, not test
doubles. It shows 43 retained detections, 28 grouped translations and 14 actual
image replacements. Eight primary and four fallback vision calls plus 14 Manga
OCR calls are recorded. The previous instruction improvement remains present.
All 60 PNGs checked for the twelve previously rendered operations stay
byte-identical, as do the default-route 444 coverage and 129 strict-only PNGs.

Visual inspection covered the full served page, both balloon crops and the panel.
The original speech lettering is removed and replaced without broad smearing;
the balloon outlines, face and surrounding artwork are retained. The keyboard
effect band is pixel-identical to the original after conversion to the same RGB
representation. It is **not** counted as translated or fixed.

The output is still only a partial candidate. The retained raw desktop line still
contains a questionable extra `°`, and the resulting English “Whoa, it suddenly
appeared on my desktop!” is semantically awkward. The left target is “I heard
it’s called a LiveCD.” Neither is claimed to be an independently approved
translation. The keyboard effect, gloss and other previously noted gaps remain;
no successful replacement count is an artistic/linguistic quality score.

Local panel/browser evidence: `test-results/viewer-quality-panel-live-v1/`.
Fresh job: `test-results/dev-frontend/8d5c4cdaf1af4569a12871265db76aa1/`.
Offline translation review checked this job's source/request/result consistency;
all 62 reread artifacts and 223 rendering artifact hashes also verified.

| Retained viewer evidence | SHA256 |
| --- | --- |
| Preview | `96bad7e4553f49a8c6754e593e34a7a1ada00e4fa96fc4d9b52c473672cbf59e` |
| Rendering manifest | `b06719df981d1f15b7d1b8ffce600bca891cedb647cb1d146e2ffa9165f89536` |
| Composite (CLI v2 and viewer) | `9448e202fdf91e462925b0e2dcd9aad51f70b0497e347ac020cf27aac58a0686` |

Verification: **848 tests passed in each of `.venv` and `.venv-paddle6`**;
source-only verification also passed with 39 optional-local-evidence skips.
Portable regressions cover source/score laundering, disagreement, budgets,
rotation/payload/pixel drift, empty/shared components, full pipeline/offline
bindings, unchanged native erasure and genuine border-contact rejection. No
defaults, licenses, user dependency edits or earlier evidence were changed, and
nothing was downloaded or pushed. This is development evidence, not held-out
general OCR accuracy validation.

## Five-page visual development expansion

The user requested more images, direct model visual inspection, trial-and-error
fixes and fresh end-to-end reruns. Three existing pages were uploaded serially
through real Firefox and the viewer, followed by two newly acquired originals.
The same existing Paddle/Manga readers, private Ollama endpoint, fonts and opt-in
quality settings were used. No models, fonts or dependencies were acquired.

### Sources and initial visual findings

New originals and English license records are retained locally in ignored
`images/quality-more-v1/`, with acquisition URLs, byte counts, hashes and a runner.
They are pinned to the original-artwork repository revision
`f3c6473206a8a8b6f1a946e8c60b4cdaff5ea71b`, not a moving branch. Only Japanese
original PNGs and license text were fetched, not translated references or
implementations. Attribution: Hiroshi Seo / ASCII MEDIA WORKS Inc.; declared
CC BY-NC 3.0, retained for local noncommercial evaluation, not training or
redistribution. This does not select or change the project license.

| New original | SHA256 |
| --- | --- |
| `ubunchu03_0002.png` | `66813ea39c2acc026262860e6c5f46e4370c333f42a17eb3651b1e55c2184489` |
| `ubunchu05_0002.png` | `4625a3ecf3a5ba44855893561cbb398946292b187b9945c54b49c89fed62a054` |

The full five initial composites were visually inspected against their originals,
not assessed solely by successful execution or replacement count:

| Page | Translations / replacements | Visible strengths and unresolved failures |
| --- | --- | --- |
| `04_0002` | 17 / 7 | Ordinary edits preserve outlines/art; important annotation-bearing dialogue remains untranslated; detached `なーんて` and `べ` remain beside English. |
| `04_0005` | 16 / 11 | Most ordinary dialogue reads legibly; detached `そ` becomes a separate “That.”; sweat marks/effects and some text remain. |
| `06_0002` | 22 / 9 | Several large dialogue edits work; one top-left balloon has two independent English layouts; bottom-right `あ` is detached; other dialogue/effects remain. |
| `03_0002` | 17 / 9 | Many major balloons render cleanly; connected paragraphs remain distinct; small footnote/effects stay Japanese, with some oversimplified or awkward English. |
| `05_0002` | 15 / 8 | Ordinary speech is legible without broad smearing; two annotation-bearing units lack translations and other balloons remain mixed Japanese/English. |

All initial runs report **partial**, with zero browser errors, pipeline execution
errors or exhausted translation budgets. Missing translations are distinguished
from translations whose image replacement was deferred. Local baseline evidence:
`test-results/multipage-viewer-check-v1/` and `test-results/more-viewer-check-v1/`.
`visual-baseline.json` freezes the first three pages' observations and exact
source/preview/manifest/composite hashes before the grouping change.

### Conservative vertical continuation trial

`src/tralens/ocr/quality_groups.py` operates only inside the already opt-in quality
reread stage. It tests adjacent vertical columns, including a short leading
singleton, against original enclosure, comparable column scale, leading alignment
and a bounded, almost entirely white **original facing seam**. Sharing one white
component is insufficient: narrow connecting necks with a border through the
actual seam, separate paragraphs, annotations, new intervening source geometry,
ambiguous neighbors and greedy chaining all reject the hypothesis. At most one
unambiguous pair joins; source strings and model outputs do not choose a merge.

The exact raw lines concatenate in geometric right-to-left order; originals and
the previous partition remain saved. Existing corner crop conflicts are retained
only when already present in a previous crop and absent from the joining seam;
new union conflicts reject. The old conflict is not cleared, hidden or converted
into cleanup permission. Structural uncertainty remains explicit, and original
native-cleanup, source-score, obstacle, exact-glyph and minimum-size gates still
decide whether an image edit is allowed. No cleanup/layout predicate is relaxed.

Annotation ownership failures were investigated separately. They include tiny
single-component readings with inconclusive glyph support, non-kana labels such
as “AKA”, and small readings intersecting oversized main geometry. The hard
`tentative_annotation_ownership` guard is **not waived** to inflate translation
coverage. Those missing translations remain an identified limitation.

The first live grouping prototype exposed an integration mistake: it added a
grouping-policy field to proposal-ID hashing, which the renderer's existing
source-identity contract correctly rejected on all three changed pages. Those
failed jobs are retained in `test-results/grouped-viewer-check-v1/`; they are not
quality successes. The implementation now uses the existing ID contract and
records the grouping policy separately in evidence. A portable full-pipeline and
offline-review regression covers this exact interface. No renderer identity
validation was weakened to accept the prototype.

Portable verification now includes short/full vertical continuation, immutable
raw evidence, geometric order, distinct enclosures, original boundary/neck
rejection, separate paragraphs, annotations, new intervening detections, retained
corner conflicts, ambiguity/no chaining, and full pipeline/offline bindings.
**857 tests passed in each of `.venv` and `.venv-paddle6`**; source-only
verification passed with 39 optional-local-evidence skips. The 444 coverage and
129 strict-only default-route PNGs are byte-identical to their previous outputs.
All 219 PNGs of the previous confidence-gated LiveCD/desktop panel also remain
byte-identical on a fresh rerender. These are software/binding regressions, not
independent evidence of Japanese or translation accuracy. Defaults, protected
user dependency edits, licenses and repository visibility remain unchanged;
nothing was pushed.

To reproduce with the existing local assets, use the viewer's **Native cleanup +
OCR checks** and **Paired OCR quality rereads**, coverage-first translation,
best-effort rendering and a 64-attempt translation budget. Upload the five
originals individually, not the saved English composites. Keep live runs serial:
the shared private Ollama switches between the visual reader and translator.
Grouping can run without decoder calls in portable tests, but live weak-line
recovery still requires both explicitly configured readers. Downloaded artwork
and all browser/model artifacts remain ignored local evidence, not test fixtures
required by the portable suite.

### Corrected fresh viewer outcomes

All five originals were then uploaded again through real Firefox with the
corrected implementation. All five produced partial candidates without browser
or pipeline execution errors or translation-budget exhaustion:

| Page | Fresh viewer job | Grouped units / translations / replacements |
| --- | --- | --- |
| `04_0002` | `4a1138c539364c13ad7a6527ad8877af` | 19 / 16 / 7 |
| `04_0005` | `b34b72f876cc477eab894a6501667081` | 16 / 15 / 10 |
| `06_0002` | `c5a2300387444efdbd8813ec8f0367d7` | 20 / 20 / 8 |
| `03_0002` | `c98e5d07307f4976a6f8415b944543a3` | 17 / 17 / 9 |
| `05_0002` | `cc7c419c097646f7a3157b73ff58a0b1` | 17 / 15 / 8 |

Open an exact job with `http://127.0.0.1:7860/?job=ID`. Evidence, screenshots,
full composites, clean layers and four sets of original/clean/composite crops are in
`test-results/grouped-viewer-check-v2/`. `audit-v2.json` records all five composite
hashes and comparisons; the saved implementation hashes match the current code.

Model visual inspection of all five full final pages and all four clean/composite
join crops found the intended improvements:

- `04_0002`: `なーんて` no longer remains Japanese beside English. It joins the
  Stallman dialogue, with a clean background and intact frame/balloon/art edges.
- `04_0005`: the separate “That.” disappears. The stammer's exact raw syllables
  remain in the combined source; the translator emits one “It’s not exactly like
  that!” block. The small sweat mark is deliberately retained, not erased as text.
- `06_0002`: the top-left balloon now reads “I can’t give up until I see the new
  boot-up screen.” in one layout instead of two independent blocks. Bottom-right
  `あ` joins the “Oh, but…” dialogue instead of remaining as residual Japanese.
- The two additional pages keep their original paragraph/balloon partitions and
  produce byte-identical composites to their initial runs. They do **not** show
  new quality gains from this particular change.

Every raw detection (including strings, scores, polygons and metadata) and all
assigned member IDs are unchanged across these paired runs. All **190 checked
PNGs of unaffected rendered operations** are byte-identical. Composite pixels
outside the four joined source rectangles plus a fixed 55-pixel audit margin are
unchanged. Offline source/request/result consistency checks and all recorded
reread/render artifact hashes verified for every fresh job; this is not semantic
approval. No previously edited source ID is lost, and source IDs `34` on
`04_0002` and `38` on `06_0002` newly participate in successful edits.

Translation/replacement operation counts can decrease when two operations become
one; the above source-ID audit prevents presenting that bookkeeping change as
either lost text or an accuracy improvement. The remaining issues are still real:
untranslated annotation-bearing speech, detached `べ`, screen/blackboard text,
effects, inconsistent type sizes and awkward/mistaken English (including
`KarmicKoa-a` and “Don’t argue with OS.”). No whole page is claimed finished.
All visually inspected/tuned pages in this section are now development evidence,
not fresh held-out references or justification for promoting defaults.
