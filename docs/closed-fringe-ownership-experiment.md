# Experiment: disputed detector fringes at a closed original boundary

## Why this experiment

The user rejected the enclosure-layout comparison as immaterial. That assessment
is correct: the major missing dialogue remained missing and the recovered short
line printed three alternatives. Layout counts are not the acceptance criterion.

The Synaptic balloon on `ubunchu01_06` has saved readable OCR/translation but
fails cleanup because detection 45, a large sound-effect rectangle below it,
overlaps 18 dark dialogue pixels. Its 66,306-pixel polygon has only a small fringe
inside the balloon. Raw polygons are evidence, not exact pixel ownership.
The network balloon's separate component/gray-interior failures are NOT addressed
by this experiment. No gray-art guard may be relaxed.

## Independent design and limits

Use only original pixels, original polygons, and existing whole-component flat
cleanup. Find a bounded closed original light region containing every member's
dominant light support; its inset usable core must agree for 4/8 connectivity.
For a conflicting nonmember, require its polygon center outside the enclosure,
at most 1% of its polygon inside, and substantial original dark support outside.
For EACH disputed original dark component, require complete containment in the
enclosure core, at least 90% outside every conflicting polygon, and at least 90%
inside the member union. Clear only the disputed component's local protection
collar, not the nonmember polygon as a whole. Keep every other neighbor protected.
Then rerun the unchanged native flat-mask/antialias/clearance/layout checks.

This is a bounded heuristic ownership hypothesis, not semantic certification.
The 1%/90% ratios are explicit experimental scope limits, not calibrated truth.
Do not rewrite raw geometry, OCR, source membership or target strings. Bind the
hypothesis to original pixels, typed geometry and exact members. No new models,
fonts, dependencies, remote providers or upstream implementation are involved.
Earlier upstream exposure still precludes a certified clean-room claim.

## Acceptance before enabling anything

- Recover the large Synaptic dialogue at the requested >=16px, with the exact
  saved target, no Japanese remnants in its source strokes and no changed border,
  screen or sound-effect pixels. Inspect a normal-size full page and enlarged crop.
- Keep all previously successful cleanup and lettering PNGs byte-identical.
- Controlled negatives: enclosed annotation/coincident box, adjacent balloons,
  open boundary, large shared-ink fraction, detached/foreign source, and original
  geometry/pixel drift. Never clip a whole native component to force ownership.
- Test six previously selected development pages, then the reused 15-page
  evaluation set with frozen source. Report all additional edits and rejection
  reasons, including subjective false ownership/art damage. This is not new
  held-out evidence or an accuracy benchmark.
- Keep opt-in. Default renders must stay byte-identical. Reject deployment if
  the visible gain is immaterial or damage/remnants appear.

Translation alternatives remain a separate unresolved defect; do not manually
choose or shorten them in a renderer experiment.

## Initial diagnostic result: hypothesis rejected, no application change

The original 4/8-connected usable balloon cores agree, but the proposed ownership
criteria fail on the very development example motivating the experiment:

- The conflicting sound-effect polygon has **3.02%**, not <=1%, of its area inside
  the original enclosure (the source-polygon intersection alone understated this).
- The disputed whole dark component has **17.65%**, not <=10%, inside the foreign
  polygon. It has complete member support, but that does not resolve the conflict.
- A diagnostic subtraction of neighbor protection near the member dark strokes
  still fails the unchanged native layout guard (`layout_seed_obstructed`). This
  was an in-memory diagnostic, not a rendered output or production modification.

Do not increase the percentage allowances or widen the protection subtraction to
fit this page. The design does not supply sufficient independent ownership
evidence. No new erase permission, source geometry or default setting has been
introduced; no recovered large dialogue is claimed. Further work needs better
source-localization/ownership evidence, not a more permissive version of this
particular heuristic. This diagnostic is neither an evaluated capability nor an
implementation improvement.
