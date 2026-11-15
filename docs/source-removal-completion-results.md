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

This is not complete source removal or semantic accuracy. The top sound effect
still has large portions connected to panel/art strokes; whole-component bounds
reject those. Repeated sounds outside detected crops, missed UI lines behind new
English and a white exclamation mark outside detected columns also remain.
The selected “Seriously.” and “Page 2.” strings and all earlier reading/translation
defects are unchanged. Mask completion cannot establish a faithful translation.
