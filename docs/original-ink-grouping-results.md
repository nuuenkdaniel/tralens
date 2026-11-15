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

## Opt-in boundary-alignment policy

`--ink-boundary-groups` with `--crop-reader-model` enables the separately versioned
`original_ink_boundary_continuations_v1` policy. It is mutually exclusive with
`--ink-column-groups`. Both options default off, and the original all-column
alignment policy remains the default for direct planner/refiner calls.

The new policy compares the tops of the two columns at the joining seam. Internal
indentation within either retained parent proposal no longer vetoes that seam.
The existing enclosure, clear/narrow seam, intervening-source, ambiguity,
membership, crop-conflict and original-pixel guards still apply. The selected
policy is saved in `ocr.ink_column_groups`; translation and rendering replay that
policy. Removing or downgrading its proof fails validation.

`seam-alignment-trial-01/` freezes the diagnostic before implementation.
`boundary-geometry-audit-01.json` confirms exact old-policy reproduction and new
policy equivalence on 23 saved paths (**22 unique pixel sources**; `saved-example`
duplicates `ubunchu01_06`). The boundary policy retains all eight prior joins and
adds three inspected continuations: the monopolist sentence on `ubunchu07_0002`,
the world-heritage aside on `ubunchu01_07`, and the pre-class rumination on
`ubunchu08_0011`. This validates the observed geometric change, not complete OCR.

`boundary-translation-01/` contains 66 fresh TG27 results on the three changed
working pages. All 65 common target strings match `ink-translation-02/` exactly;
the fourth page retains its 42 prior records. The additional joined source is:

> そのすばらしさにはどんな独占主義者だって心を入れ替えるわ

It now produces “Even the most ardent monopolist would change their ways after
experiencing its magnificence.” This is a coherent rendition of the visible
sentence, replacing the disconnected fragment outputs. It is somewhat more
elaborate than the Japanese; it is not an independently certified translation.

`boundary-render-01/` contains fresh localization and four native-resolution
renders. The six joined regions were inspected as original/prior/clean/final
cards, along with the changed full page. The new sentence is visibly lettered and
its original Japanese is removed. The six groups have zero remaining dark pixels
at 18,141 original supported-ink pixels, including 6,199 in the new join. This
support-specific measurement does not certify every source mark on the pages.

The audit checks 1,273 artifact hashes, exact selected targets and recomposition,
font floors, membership and zero changes outside declared support. There are 95
displayed replacements after six merges, zero available-but-undisplayed targets,
12 intentionally identical targets and one pre-existing missing translation.
Three composites are pixel-identical to `ink-render-01`; the changed page differs
at 12,120 pixels. The new join is the only new lettering region. Intrusive text,
local art loss and the earlier clapping/reassurance/pronoun errors remain visible.

The portable regressions exercise staggered internal columns, misaligned seams,
both policies' geometry guards, policy downgrade/removal, conflicting options,
pre-dispatch proof rejection and the actual coordinator/render handoff.
The full working-tree suite passes 1,113 tests with `PYTHONPATH=src`. An initial
invocation without that environment failed one subprocess import; both logs are
retained.
