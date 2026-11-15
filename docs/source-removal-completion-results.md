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
