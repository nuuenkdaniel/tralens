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
