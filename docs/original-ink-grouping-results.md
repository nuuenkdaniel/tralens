# Original-ink column continuation checkpoint

Use `--ink-column-groups` together with `--crop-reader-model` on the rendered,
coverage-first pipeline. The option is off by default. The coordinator runs
grouping before crop reading; translation and rendering receive saved evidence.

## Geometry and preservation

`src/tralens/ocr/ink_groups.py` measures the complete original dark components
uniquely supported by each detection. Oversized detector margins can overlap
while the actual lettering has a clear seam. Stacked pieces within one column
are ordered top-to-bottom, then columns right-to-left.

Only two unambiguously adjacent vertical proposals can join. They must share an
enclosed original white surface, align at the top, have a sufficiently narrow
and clear original seam, and introduce no new nonmember crop conflicts. An
intervening source, ambiguous neighbor, interleaved membership, annotation,
unknown orientation or unsupported ink prevents that join. Source strings and
OCR scores do not choose the grouping.

Raw polygons are never tightened or overwritten. The merged source crop remains
their exact native-coordinate union. All previous proposals and recognition
results remain in `ocr.ink_column_groups`, alongside original component/view
hashes, seam pixels, enclosure evidence, ambiguity and failed-line records.
These hypotheses confer neither complete-utterance status nor erase permission.

Translation and rendering recompute the grouping from original pixels. Saved
partition revisions, duplicate membership, geometry, seam evidence and source
pixels are checked. Rendering dispatches no OCR. Existing crop-reader precedence,
exact target strings, cleanup policy and font floors remain applicable.

## Local observations

Evidence remains in ignored `test-results/production-best-effort-v8/`:

- Five joins on the four working pages match the inspected original ink:
  open-source dialogue plus its final verb; a trailing explanatory column;
  Super Cow plus punctuation; a question plus punctuation; and the opening of
  the encouragement question plus its continuation.
- `ink-translation-01/` retains an interrupted replay caused by tuple/list
  serialization of a pre-existing corner conflict. It failed before TG27
  dispatch. The normalized snapshot fix has a JSON-roundtrip regression test.
- `ink-translation-02/` contains 67 fresh bound TG27 results on the three changed
  pages. All 62 ungrouped target strings match their preceding outputs exactly.
  Prior requests, results, source readings and previews remain available.
- `ink-render-01/` contains four fresh native-resolution renders. All five joined
  regions were inspected as original/prior/clean/final cards: their Japanese
  source text is removed, including the formerly detached punctuation. The
  audit checks 1,288 artifact hashes, exact recomposition/strings, font floors
  and zero changes outside declared support. It displays 96 replacement regions
  after five merges; every available target is displayed or intentionally
  identical to its source. One pre-existing source still lacks a translation.
  The saved-example composite is pixel-identical to the preceding crop checkpoint.
- The open-source passage improves from a heading plus “I was doing it” to
  “I had a misunderstanding about open source, and now I feel embarrassed.”
  Detached punctuation is now part of its sentence.
- Language errors persist. The dawn-related continuation becomes “We’ve already
  come this far!” without its wider context. The encouragement question gets
  “them” instead of the contextual “me.” Clapping, reassurance, ACPI/off and other
  earlier errors also remain. Better grouping does not certify translation.
- `ink-other-pages-01/` applies the geometry to 19 additional saved pages. It
  proposes three joins, all inspected against their original image crops. This
  is additional geometry evidence, not untouched end-to-end accuracy evaluation.

Seven portable tests cover overlapping detector margins, stacked pieces,
separators, empty ink, ambiguous/interleaved/duplicate membership, text-independent
grouping, retained prior results, serialized conflicts, tampering and the actual
coordinator/translation/renderer handoff with injected synthetic readers.
