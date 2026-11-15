# Native source-component completion

## Disconnected weak strokes

`source_bound_semantic_strokes_v3_weak_components` extends the approximate
source-support route used by aggressive cleanup. A disconnected original-pixel
component can now be completed when at least 80% of it lies in the source polygon
and at least 16 of its pixels have saved CTD probability of 32 or higher. These
are routing thresholds, not probabilities of correctness or ownership proof.

Existing component bounds (256 pixels per side and 32,768 pixels of ink), the
12-pixel physical expansion limit, polarity handling and exact accepted-output
protection still apply. Native contours include hollow interiors. A low-probability
island without original ink is not promoted. Partial protection is subtracted
after measuring the whole component, so an earlier edit cannot veto cleanup of
the rest of a stroke. Narrow-line completion remains separately recorded.

The change consumes saved native probability and source geometry. It does not
read or edit words. Rendering records each weak addition's crop-local bounds,
original area and supporting probability-pixel count.

## Inspected six-page replay

Ignored evidence is retained under `test-results/production-best-effort-v8/`:

- `weak-components-render-01/` contains six fresh renders with the exact preceding
  translations and localization evidence. Four earlier-page composites and
  cleanplates are pixel-identical to `boundary-render-01/`.
- On `ubunchu01_07`, two disconnected parts of the large top sound effect are
  removed. Their native component areas are 439 and 69 pixels, at page bounds
  `[518,129,558,151]` and `[531,148,537,166]`. All 508 positions become RGB 254.
  The previous cleanplate retained 489 exact original pixels and 496 pixels below
  gray 230 at those positions; both counts become zero. The full cleanplate
  changes at 3,253 pixels and the composite at 3,110.
- On `ubunchu08_0011`, the added 215-pixel component at
  `[1130,407,1134,470]` is visibly a speed line, **not a text-removal benefit**.
  It has 27 weak-probability pixels. Its removal changes the reconstructed
  neighborhood: both cleanplate and composite differ at 21,683 pixels, within
  declared cleanup support. This is accepted local artwork loss under the
  best-effort source-removal priority, not faithful artwork restoration.
- `weak-components-render-01-review/` has four inspected changed-source cards;
  every changed pixel lies within their review crops. Both changed full composites
  were inspected. `weak-components-component-review-01/` contains three inspected
  native-component overlays and pixel measurements, bound to artifact hashes.
- The replay audit verifies 1,808 artifacts, exact source/preview/target bindings,
  recomposition, font floors, unique membership and zero changes outside declared
  support. Across the six pages it displays 133 replacements, with 15 identical
  targets, one existing missing translation and none available but undisplayed.

The weak-component policy is retained because it removes inspected source
fragments. It is deliberately approximate: detector support can include art.
The full working-tree suite passes 1,115 tests; the focused cleanup suite passes
40. Synthetic regressions cover weak disconnected hollow strokes in both
polarities, partial protection, probability-only islands, unpredicted artwork
and a weak neighboring component outside source ownership.

## Remaining failures

The V3 checkpoint is not complete source removal or semantic accuracy. The top sound effect
still has large portions connected to panel/art strokes; whole-component bounds
reject those. Repeated sounds outside detected crops, missed UI lines behind new
English and a white exclamation mark outside detected columns also remain.
The selected “Seriously.” and “Page 2.” strings and all earlier reading/translation
defects are unchanged. Mask completion cannot establish a faithful translation.

## Thick source pieces joined to thin artwork

The follow-up `source_bound_semantic_strokes_v4_thin_contacts` keeps V3 completion
and adds a source-bound contact split. A rejected whole native component can
supply thick pieces only when its ink inside the original source polygon has at
least 16 strong CTD pixels and at least 35% weak CTD coverage. A three-pixel native
erosion separates thin contacts; dilation restores each surviving piece against
the original ink. Each completed piece must remain inside the physical expansion,
have at least 80% source containment, and satisfy the same 256-side/32,768-area
bounds. A one-pixel contour collar is included. Accepted output is subtracted last.

The parent evidence permits a thick, unpredicted portion of a glyph to be cleaned
when it is connected to text-supported ink through a panel line. This is not
semantic ownership: attached artwork can still qualify. Thin branches do not
provide eroded cores that can pull the cleanup along an entire panel line. Native
pixels are never resized or altered for inference, and layout remains separate.
Completion records include crop-local piece/parent bounds and supporting counts.

### Compared controls

- `art-contact-masks-01/` is retained as an interrupted diagnostic: comparison of
  support metadata encountered the separately appended `cleanup_extent` record.
  It dispatched no inference. The corrected comparison retains that record and
  verifies the native-support fields independently.
- `art-contact-masks-02/` compares four masks at 82 saved support-call inputs.
  These include repeated recovery/final-pass regions, not 82 independent sources.
  All 31 changed-source cards were inspected. Cutting every native component at
  the physical boundary clears the main sound effect but removes substantial
  panel, balloon and book edges. Plain openings of radii two, three and four
  generally retain the thick top fragments because those pieces have no direct
  detector support.
- `art-contact-masks-03/` compares the parent-supported rule at the same three
  radii. All 11 cards were inspected. Radius three clears the large unsupported
  pieces and avoids a visible balloon-line loss introduced by radius two on the
  saved-example page. Radius four also helps but recovers less of the native edge.
- The bounded, crop-local implementation reproduces all 82 radius-three masks
  exactly (`contact-candidate-audit-02.json`), adding 2,162 mask pixels across those
  diagnostic calls. Those are geometric comparisons, not source-removal scores.

### Fresh rendering and remaining traces

`art-contact-render-01/` runs six fresh renders with the diagnostic adapter, using
the unchanged selected source, translation and localization records from V3.
The four other pages remain pixel-identical. On `ubunchu01_07`, the cleanplate
changes at 3,663 pixels and the composite at 3,625. On `saved-example`, those counts
are 1,766 and 1,646. The audit checks 1,808 artifacts and preserves exact wording,
font floors, unique membership, recomposition and zero outside-support changes.
Seven changed-source cards, both changed full cleanplates/composites, and nine
native-component measurement cards were inspected.

`art-contact-render-02/` repeats all six renders through the production
implementation. All **1,677 PNG artifacts are byte-identical** to the inspected
diagnostic run, including original/clean/composite images and native masks. Its
separate audit again checks 1,808 artifacts; both runs record idle runtimes before
and after local rendering. The 133 displayed replacements, 15 identical targets,
one existing missing translation and zero available-but-undisplayed targets are
unchanged.

The large top fragments are substantially reduced. Three opened pieces add
cleanup support at 1,549 previously uncovered original-ink positions. The two
unpredicted thick heads have areas 614 and 686; both include their panel contact.
Reconstruction restores a horizontal border, while small source-like tips and
marks remain below it. Consequently dark pixels and exact original-pixel matches
remain at some measured positions; neither mask coverage nor those counts can
certify complete removal. The saved-example additions mainly change tiny contact
edges and neighboring reconstructed texture, with no large semantic improvement.

The synthetic regressions exercise both polarities, a wholly unpredicted thick
piece sharing a supported parent, partial protection, immutable inputs, a long
thin art branch, a foreign thick piece, disconnected artwork, and sparse-parent
rejection. Source strings, translations and the earlier semantic defects remain
unchanged. Small contact remnants and crop-external/undetected lettering still
require work.

The focused cleanup suite passes 42 tests and the full working-tree suite passes
1,117. Those tests and the exact replay comparisons establish implementation and
preservation behavior, not bilingual accuracy or universal source removal.

## Native-resolution detector diagnostics

`native-ctd-trial-01/` compares the saved whole-page CTD probabilities with fresh
native crop and native context-tile predictions on 29 previously inspected
remnant-screen cases. These are reused development cases, including locations
where the residual screen marked artwork rather than text. The crop pixels are
unchanged; context tiles are at most 1024 pixels per side. CTD receives only its
required bottom/right padding, with no downscaling in these 58 forwards.

The forwards take about 65.7 seconds in total. All 87 page/crop/tile support
derivations succeed. The offline audit binds 432 artifacts and verifies original
crop/tile pixels, native probability mappings, checkpoint identity, support replay
and runtime idleness. All 29 comparison cards across ten sheets were inspected;
`native-ctd-trial-01-review/inspection.json` binds the observations to their hashes.

Native inference sharpens strokes and covers more of the difficult top sound
effect. For `24-ubunchu01_07-00`, crop support adds 3,021 pixels but omits 462 from
the page support; tile support adds 3,389 and omits 415. Other regions lose useful
support too: the tile view retains part of the weekly-publication character, and
the feature-heading tile retains a Latin fragment. Extra support can cut nearby
book, panel, balloon and UI edges.

All three resolutions still leave the white exclamation mark outside its selected
columns, repeated sounds beyond the detected footprint, and unowned UI lines.
The APT comparison is before the separately saved extent completion and is not a
final-render regression. Sharper probability alone fixes neither source ownership
nor wrong selected readings such as “Seriously.”, “Top—” and “Page 2.” The trial
therefore motivates an additive support experiment rather than replacement of the
page mask. None of its readings or translations is newly selected.

### Additive saved-native replay

`native-ctd-render-01/` unions independently derived crop/tile support with V4
page support, recomputing each candidate against current protection. Reuse requires
the exact original crop shape, RGB-pixel digest and source-mask digest. This is a
29-case diagnostic adapter, not a general production policy. Twenty-seven cases
reach that adapter; two white-text cases finish through earlier routes. The 91
support requests replay exactly, including 27 native completions adding 17,883
support pixels across calls, with no shrunken baseline mask or protected commits.

The fresh render audit checks 1,811 artifacts and preserves exact target wording,
font floors, unique membership, recomposition and zero outside-support changes.
It still displays 133 replacements, with 15 identical targets, one existing
missing translation and none available but undisplayed. One page is unchanged;
five changed full cleanplates/composites and all 47 changed-context cards were
inspected and hash-bound in `native-ctd-render-01-review-02/inspection.json`.

| Page | Changed clean pixels | Changed composite pixels |
| --- | ---: | ---: |
| `ubunchu01_07` | 6,623 | 6,587 |
| `ubunchu08_0011` | 33,169 | 32,057 |
| `saved-example` | 72,253 | 73,572 |
| `ubunchu07_0002` | 0 | 0 |
| `ubunchu08_0007` | 71,327 | 66,316 |
| `ubunchu03_0006` | 5,290 | 4,182 |

Small remnants under the top sound-effect border disappear, but a dark rectangular
block remains at the lower balloon contact. Subsequent native-pixel attribution
shows mixed retained original ink and reconstruction, not wholly synthesized ink.
The feature heading clears
a retained Latin edge while switching to a broader opaque-backed fallback that
removes more book/panel artwork. Other changes clip icons and panel/UI rules.
Most importantly, fresh LaMa output introduces distinct loop/letter-like marks
along the lower APT title boundary. **An additive erasure mask does not guarantee
monotonic visual removal after reconstruction.** This replay is retained as a
diagnostic rather than integrated into production.

Two initial audit failures are preserved with their scripts/logs. The old review
envelope missed 1,029 changed pixels when a heading changed placement route;
review now includes both old/new cleanup and backing bounds. A non-learned final
cleanup had no saved backend-protection image; its exact protection is rebuilt
from the accepted primary/recovery layers and ordered final commits, checked
against saved protection wherever present, and required to reproduce the complete
final erase/alpha masks. The corrected audits retain the original integrity
checks and require every changed pixel to appear in a review card.

### Native inference-only context replay

`native-ctd-render-02/` adds the same native probability fields to reconstruction's
inference-only mask, retaining the preceding additive support derivation. The
support/inference replay and 1,811-artifact render audit pass. All 82 saved
inference requests keep byte-identical commit masks; 25 gain inference-only
pixels. Twenty-six native-context calls add 60,751 pixels in total (one adds zero).
Final erase, alpha, text and fallback masks, target wording, layout and placement
routes are identical to replay 01 on all six pages. Changes are reconstruction
pixels, not extra erasure permission or different lettering.

All 39 changed-context cards and ten changed full-page cleanplates/composites were
inspected; `native-ctd-render-02-review/inspection.json` binds those observations,
13 compact sheets and the mask comparisons. Relative to replay 01:

| Page | Changed clean pixels | Changed composite pixels |
| --- | ---: | ---: |
| `ubunchu01_07` | 3,436 | 3,436 |
| `ubunchu08_0011` | 40,032 | 38,494 |
| `saved-example` | 39,110 | 35,018 |
| `ubunchu07_0002` | 0 | 0 |
| `ubunchu08_0007` | 72,278 | 66,646 |
| `ubunchu03_0006` | 5,548 | 4,446 |

The APT loop/letter-like marks are greatly reduced, and the copy/paste caption
loses most of its thin reconstructed curves. Small dark islands and rough tonal
boundaries remain. The top dark block persists while more balloon arc disappears;
book interior detail is lost; a new thin vertical reconstruction appears near the
repeated sound; longer horizontal marks appear behind Update and Codec. Unowned
UI lines, the white exclamation mark and repeated sounds remain unresolved.

This is useful evidence that reconstruction context matters independently of
commit support, but neither saved-case replay is a general production policy.
Native evidence must next be tested across all applicable source-support calls,
with inspected controls for collateral loss and reconstructed lettering. The
unchanged 133 displayed replacements still do not establish accurate wording or
complete source removal.

### Boundary attribution

Two inspected cards in `native-context-boundary-review/inspection.json` bind the
original pixels, source geometry, final erase mask and cleanplate. The top source
polygon ends within the original stroke: part of the dark root lies outside final
erase support, while reconstructed dark pixels also occur inside support. The APT
boundary similarly combines retained exterior artwork and reconstructed islands.
The diagnostic rectangles include artwork; their dark-pixel counts measure
neither residual lettering nor removal accuracy.

## All-source native crop/tile replay

`general-native-render-01/` removes the saved-case allowlist. Every successful V4
support request receives additive independently derived native crop and tiled-page
support, and the same fields enlarge inference-only context. Original source
pixels are preserved: 1024-side native tiles use at most 768-pixel stride and
overlapping probability maxima; unique crops up to 1024-side receive their own
forward. Base failures retain their original fallback behavior.

The six reused development pages dispatch 114 forwards (24 tiles, 90 crops).
Ninety support calls succeed; 69 add support, totaling 39,807 pixels across calls.
Eighty-one context calls add 124,330 inference-only pixels. There are no base
support failures. The 1,232-artifact native audit verifies exact original inputs,
tile mapping, additive support, inference replay and ordered primary/recovery/final
protection. The 1,803-artifact render audit verifies exact targets, font floors,
unique membership, recomposition and zero changes outside declared support.

All 69 changed-context cards across 23 compact sheets and all twelve full
cleanplates/composites were inspected. Hash-bound observations and exact routing
comparisons are in `general-native-render-01-review/inspection.json`, with a
subsequent card-attribution correction in `inspection-addendum.json`. Compared
with production V4:

| Page | Changed clean pixels | Changed composite pixels |
| --- | ---: | ---: |
| `ubunchu01_07` | 21,452 | 19,980 |
| `ubunchu08_0011` | 41,183 | 39,588 |
| `saved-example` | 96,922 | 99,746 |
| `ubunchu07_0002` | 7,306 | 6,275 |
| `ubunchu08_0007` | 74,798 | 68,847 |
| `ubunchu03_0006` | 8,014 | 6,737 |

The top root shrinks but remains while the panel border is cut. Further cuts affect
a hand near “Huh?”, a hair arc, book detail, the Synaptic balloon edge, chibi art,
clothing, a mouth/balloon contact and report rules. Update gains faint marks, the
bull caption gains a dark stroke, and APT/tagline reconstruction stays rough.
Unowned UI lines, repeated sounds and the white exclamation mark remain.

Routing changes explain some of the wider effects. The top sound switches from
LaMa to dominant-surface fill. On `saved-example`, the feature heading and “Eerin,
Eerin.” move to final opaque backing; the latter also wraps onto two lines.
“Established” shifts six pixels right. “Wisdom.” moves from final backing to
recovery placement and grows from 22px to 25px; “Success.” switches the same way
but shrinks from 22px to 18px, exposing crossing UI detail. Another sound switches
from Telea to LaMa, and chibi-adjacent cleanup switches from local occlusion to
source-supported LaMa. Exact selected strings and font floors are preserved;
their reading/translation defects are not repaired.

This all-source adapter remains diagnostic. Broader native completion increases
collateral loss without solving missing source ownership; passing integrity audits
and 133 displayed replacements do not support production promotion. Production
remains V4 thin-contact completion.

## Coverage outside raw OCR geometry

`native-coverage-screen-02/` screens saved page/native CTD evidence outside **all**
raw OCR polygons, not merely selected ones. Weak-probability components (32 or
higher) with at least 16 unowned strong-probability pixels (77 or higher) produce
original/geometry/probability/clean/composite review crops. The initial run's font
lookup failure remains in `native-coverage-screen-01.log`; the successful run uses
the existing Noto CJK face.

All 220 cards across 74 sheets were inspected and hash-bound in
`native-coverage-screen-02/inspection.json`. These are review components, not
unique passages or a recall/accuracy benchmark:

| Observed candidate type | Components |
| --- | ---: |
| Missed lettering | 30 |
| Clipped continuation / omitted column / punctuation | 36 |
| Mostly owned geometry fringe or mixed component | 13 |
| Artwork, structural detail or icon | 133 |
| Uncertain at inspected scale | 8 |

Real omissions include whole dependency/status UI rows, two title lines above a
credit, repeated outlined sounds, small handwritten annotations, utterance endings
and the white exclamation mark. A Linux annotation column lies outside most of its
neighbor's raw box: cleanup removes much of it, but “tone-deaf” still omits the
Linux-specific meaning. The report page also has a visible lower line clipped by
the panel, and an outlined keyboard sound lacking its own raw footprint. Several
components instead describe the already-English `VS` graphic. Detecting that
graphic is not by itself a requirement to alter it.

The largest false positives include faces, hair, clothing, hands and interface
icons; components can connect actual lettering to those shapes. Conversely, one
word or repeated sound can yield several components. Some lettering outside raw
geometry has already been removed by saved extent evidence. Neither raw-box
coverage nor a detector-component union can establish semantic ownership or
authorize erasure.

These observations motivate fresh OCR on original-pixel context windows around
all candidates, with independent prediction geometry, retained alternatives and
explicit crop-boundary evidence. Inspection labels and expected readings must not
enter that request planner. Cross-window agreement can help locate stable lines,
but inspection is still needed for complete readings and correct ownership.

### Fresh original-pixel OCR coverage trial

`coverage-ocr-01/` makes 102 local Paddle v6-medium calls across all six pages,
including every artwork and uncertain control. Its 512/768-side grids use
three-quarter strides and anchored final tiles, selected only by intersection
with coverage-component bounds. Crops contain unresampled original pixels;
Paddle's own detector/line preprocessing remains enabled. Inspection labels and
expected strings do not enter planning. All 1,026 detections remain alternatives.

The 370-artifact audit replays the complete plan, original RGB/BGR input pixels,
Paddle association, page-coordinate offsets and internal crop-edge flags. All
102 calls finish, and before/after remote-runtime records are idle. The first audit
failed because importing the nested script snapshot changed repository-relative
imports; the byte-identical original script is used by the corrected audit.
The failed script and log remain preserved.

Fresh detector polygons intersect 68 of the 220 screened components; 66 have
nonempty readings and 152 have no hit. These are component intersections, not
recovered passages. All 68 hit cards across 23 sheets and all six full originals
were inspected; `coverage-ocr-01-review-02/inspection.json` binds observations,
zero-hit records and 466 artifacts. Useful findings and failures include:

- Both grids recover the visible `依存関係を解決` dependency-row phrase and
  `完了`. The larger grid also reads `状態情報を読み`; the smaller one drops
  the visible `み`. Intersecting components can span adjacent UI rows.
- The larger grid recovers the leading `い` in `いやーっ`, but both grids omit
  its visible exclamation. Repeated outlined sounds become Latin letters or
  unrelated kanji; the title above the credit remains unread.
- Both grids agree on the wrong/incomplete `Pプデー` for the update label and
  `いいのかし` without final `ら`. Missing characters can occur well inside a
  tile, so crop-edge flags do not certify complete source geometry.
- Edge-flagged 512 crops read `APT-Man`, while larger internal 768 crops truncate
  it to `APT-Ma`/`APT-M`. Bigger context is not monotonically better.
- A crown icon is repeatedly read as `2`. The initial review attributed the
  0.959-confidence `乐` to an eyebrow; the native-line reinspection below corrects
  that attribution. Adjacent text boxes also touch fingers, icons and rules.
  Neither confidence nor cross-grid agreement establishes textual ownership.
- The Linux annotation, outlined keyboard sound and panel-clipped report row
  remain substantially unread. The quiet sound and white exclamation have no
  overlapping fresh detections.

Full-page inspection also corrects four earlier component attributions: IDs 26,
27, 29 and 30 are the four quarters of an **anger symbol**, not sound-effect
characters. The original observations are retained, with explicit corrections in
the new inspection record. Corrected screen counts are 27 missed-lettering,
35 clipped-continuation, 13 geometry-fringe, 137 artwork/symbol and eight uncertain
components. These counts still do not measure OCR recall or translation accuracy.

The next diagnostic reads original component/OCR envelopes with the independent
local Hayai reader. Existing source selections and production V4 remain the
baseline; these fresh detector hits alone do not justify automatic promotion.

### Independent component-envelope reader

`coverage-reader-01/` completed 405 deduplicated local Hayai calls over all 220
components, including artwork and zero-Paddle-hit controls. Each component's
bounds are united with its geometrically intersecting fresh OCR polygons, then
read with 24/96 and 72/256 padding/minimum-side contexts. Oversized envelopes use
overlapping native 512 crops. Planning uses geometry only; the reader receives
original RGB pixels without masks, expected text or inspection labels.

The pinned Hayai v2.7-supernova snapshot and CPU runtime are recorded. All calls
finished in 59.5 seconds. The corrected audit binds 838 artifacts and replays every
request, exact crop pixels, response and both-context coverage. All 220 cards
across 37 sheets were inspected; `coverage-reader-01-review-02/inspection.json`
binds explicit observations and 1,559 artifacts, including the earlier review's
anger-symbol corrections. Initial review images remain available; corrected
images wrap predictions by measured pixel width so long Japanese outputs remain
visible. The first two inspected sheets are byte-identical between layouts.

- Both contexts recover `世界遺産的な意味で`, including the heritage passage's
  final `で`; IDs 37/39 recover `ああああ`, ID199 recovers `はっ`, and ID205
  recovers `はやくっはやくっ`.
- Local crops recover `いやーっ!`, `アップデート`, `おめでとー`, and the
  previously omitted `リナックス音痴` annotation. Wider crops respectively add
  another sound, merge UI content, lose `とー`, or borrow neighboring dialogue
  and repeat Linux. ID201 changes from local `グッ` to wider `バッ`.
- UI envelopes merge separate rows, corrupt their beginnings and sometimes
  generate long repetitions. The crown-icon context and one APT/title context
  also produce long repeated strings. Neither title reading is complete.
- The quiet sound, outlined smiling repetitions, Latin VS, outlined keyboard
  sound and panel-clipped report continuation remain incomplete or wrong.
- Both contexts read the adjacent `ああああ` at ID59 even though its component
  is a background rule. Artwork also produces stable `バ` (ID52), `ド` (ID105)
  and `あ` (IDs113/146/154). Correct crop text and stable hallucinations both
  defeat promotion based on reader agreement alone.

Hayai supplies no glyph geometry. These useful recoveries remain unselected
alternatives: they establish neither ownership by a probability component nor
complete source support. The next trial reads each distinct fresh detector polygon
with tight, axial and narrow contextual crops, testing whether individual lines
reduce neighboring-row contamination while retaining all geometric false-positive
controls. These reused pages remain development evidence, not held-out accuracy
validation. Production V4 remains the baseline.

### Individual native-line readings

`coverage-line-reader-01/` reads all 115 exact-distinct fresh polygons associated
with the 68 hit components. Tight, axial and contextual views give 345 local Hayai
calls, all completed in 48.0 seconds. Planning uses only saved geometry; artwork,
empty-recognition and clipped polygons remain included. The 152 zero-hit components
retain their earlier component-reader evidence rather than receiving invented
line geometry. The pinned model, settings and source images are unchanged.

The native audit binds 725 artifacts, replays the complete geometry plan, and
verifies exact original crop pixels and all three response bindings. All 115 cards
across 29 sheets were inspected. Explicit observations in
`coverage-line-reader-observations.tsv` and the completed review in
`coverage-line-reader-01-review/inspection.json` bind 880 artifacts.

- Individual crops recover useful `アップデート`, `インストール`, `いやーっ!`,
  copy/paste headings and some `Masato` alternatives, reducing broad-envelope
  repetitions. They still omit visible `ー` in `アスキー`, misread `不撓` as
  `不挽`, truncate names/titles and merge `Wgetであうあう`.
- A detector polygon is not necessarily one passage. The Linux seed crosses two
  vertical columns horizontally; other seeds span title/credit fragments or UI
  plus sound effects. Their long axis does not establish writing direction.
- Even appropriate tilted-line polygons have axis-aligned rectangles containing
  adjacent rows. Dependency/status and roaming labels repeatedly acquire those
  neighboring readings. Conversely, tighter crops omit indented endings and
  repeated sounds; consistent partial predictions are not complete-source reads.
- Anger and crown symbols still produce stable kana. Geometric intersection does
  not confer text ownership on an adjacent finger, checkbox or background rule.

**Attribution correction:** coverage component ID10 remains eyebrow artwork, but
the fresh 0.959-confidence `乐` polygon spans the neighboring large stylized sound.
Native line ID3 and the rechecked component-reader ID10 card establish that the
earlier “eyebrow read as `乐`” wording overstated attribution. Broad Hayai
`ガチ`/`ガチン` also read the adjacent sound, not an eyebrow-only crop. The original
observations remain preserved and are explicitly superseded in the native
inspection record. Rechecked eye-detail controls ID52 (`バ`) and ID105 (`ド`)
remain valid artwork hallucinations. The corrected component-category counts do
not change.

### Native-pixel scanline presentations

`coverage-strip-reader-01/` tests the same 115 polygons and three views with
geometry-only integer scanline translations. X-major strips shift columns and
y-major strips shift rows around a fitted centerline; ambiguous shapes keep
rectangular views. Native RGB values are neither interpolated nor recolored.
Every view saves its page-coordinate sampling map. These maps describe reader
presentations, **not source ownership, layout or erasure geometry**.

All 345 local calls finish in 48.3 seconds. The independent 1,871-artifact audit
reconstructs every scanline, checks exact one-to-one original RGB sampling and
finds zero omitted raster pixels from the seed polygons. There are 219 changed
inputs and 91 changed readings relative to the native trial. All 1,628 clamped
scanlines occur in the three `SUPER!` presentations; clamping retains polygon
coverage but those readings still lose or misread the exclamation. These are
operational counts, not accuracy measures.

All 115 comparison cards across 29 sheets were inspected. The explicit
`coverage-strip-reader-observations.tsv` and
`coverage-strip-reader-01-review/inspection.json` bind 2,076 artifacts, including
the native inspection and its attribution correction. Findings are mixed:

- Lines12/13 isolate `依存関係を解決` in every view; lines14/21 retain
  `状態情報を読み` without the native context's tree-row additions. Line19
  isolates `依存関係ツリーを`, while line22's wider context still adds unrelated
  text. Lines72–74 isolate `ローミング・モードが` across all views.
- Line101 isolates `いいのかしら`, but line102's wider view still borrows and
  corrupts the preceding column. Lines80/81 isolate one `えーりん` rather than
  assigning both repeated rows to one seed.
- Full seed-polygon coverage does not mean full lettering coverage. Line4's
  narrow strips lose heritage-ending `で`; line83 regresses from `おめでとー`
  to `おめで`. Both endings lie partly outside the tilted seed's useful extent.
  Some complete predicted endings also come from visibly clipped input glyphs.
- Latin transcription regresses too: line46 changes `Ubuntu` to `Ubumtu`,
  lines76/78 corrupt `install`, line88 loses the `APT-Man` hyphen, and line93's
  context changes `Masato` to `Mosto`. Other name/heading views improve, so neither
  strip presentation nor wider context dominates the original rectangles.
- Wider lines110/113 change the second `はやくっ` to `まやくっ` while sampling
  a clipped neighboring column. Stable artwork hallucinations, wrong `不撓`,
  missing magazine punctuation, mixed `Wget` text and the unread Linux annotation
  remain. Integer shifting cannot repair incorrect seed membership.

These experiments support retaining mapped views alongside original rectangles,
with exact source maps and explicit disagreement/extent evidence. They do not
justify replacing existing selections on strip agreement. All predictions remain
unselected; the earlier production translations and V4 cleanup remain the
baseline. Complete readings, semantic ownership, untranslated omissions and
untouched end-to-end validation still require work.

### Independent recognition over native and strip inputs

`coverage-cross-reader-01/` runs the existing local `PP-OCRv6_medium_rec`
recognizer on all 690 native/strip presentations. Each is read as presented; a
lossless counterclockwise quarter-turn is also read when height is at least
1.5 times width. Exact RGB-content deduplication preserves all 798 orientation
associations in 669 calls, completed without execution failures in 30.6 seconds.
The isolated Paddle runtime uses CPU/four threads, batch size one, disabled
MKLDNN/HPI/TensorRT/CINN, blocked downloads and the shared inference lease.
This is recognition only: no new detector, expected text or masks are supplied.

The audit binds 3,345 artifacts and independently reconstructs every input,
quarter-turn, deduplicated call and source association. All 115 comparison cards
across 29 sheets were visually inspected. Explicit observations in
`coverage-cross-reader-observations.tsv` and the completed
`coverage-cross-reader-01-review/inspection.json` bind 3,582 artifacts, including
the preceding inspections and attribution corrections. The 103 exact Hayai
matches, 110 normalized matches and 103 empty orientation associations are
operational comparisons, not accuracy measurements.

- Line13's three strips corroborate `依存関係を解決`; lines73/74 corroborate
  `ローミング・モードが`. Line12's contextual Paddle reading still substitutes
  `法` for `決`. Strip row isolation is useful but not uniformly correct.
- All six Paddle views of line84 recover `不撓の牛王`, rejecting Hayai's
  persistent `不挽`/`不撹`. Line85 recovers `APT-Man`, and lines93/96/98 recover
  `Masato` in all views. These independently weighted readings help distinguish
  some name/title errors; adjacent polygons still truncate them or borrow text.
- Different readers share omissions. Rotated narrow strips of line4 agree on
  `な意味` while losing the indented `で`; lines82/83 agree on `おめで` without
  `とー`. Line45 agrees on `アスキ` without visible `ー`, and line28 agrees on
  only two or three of four repeated `あ`. Full detector-polygon coverage and
  cross-model agreement establish neither full lettering extent nor full text.
- Agreement can preserve wrong grouping too: line10's `路さ` spans title/credit
  fragments, and both readers mix `Wget` with neighboring `あう` in line56.
  The Linux seed still crosses two vertical columns horizontally, leaving its
  meaning unrecovered by these line crops.
- High confidence accompanies obvious failures: outlined `ブチッ` becomes `7`
  at 0.974–0.987; line76 returns `installz` at 0.995; line100's strip context
  gives `Masito` at 0.981. Crown icons repeatedly become `2`. Artwork, mixed
  passages and clipped text remain in the record rather than being filtered by
  score or inspection outcome.
- Paddle often drops final `み`, `ら` or small `っ`, and corrupts the update
  label even where Hayai reads it. Line91's axial strip recovers `SUPER!`, but
  its tight strip agrees on incomplete `SUPER`. No reader/presentation dominates.
- Bounding-box aspect misses compact tilted vertical writing. Magazine `週刊`
  and left-hand `はやくっ` crops receive no rotated presentation under this
  heuristic; their failures do not show that a correctly oriented crop is
  unreadable. Writing direction needs evidence beyond a rectangular long axis.

The independent recognizer supplies useful alternatives, but shares all source
geometry with Hayai. Its agreement cannot certify selection, semantic ownership
or erase support. All new readings remain unselected. Further recovery must
address source extent, orientation and passage membership; these reused pages
continue to be development evidence rather than untouched validation.
