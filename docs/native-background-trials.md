# Native-component association and phase-locked repair

## Kept improvements

The first CTD/LaMa feasibility pass showed that learned masks alone are too loose
and generic inpainting blurs periodic dots. The follow-up trials kept two smaller
methods instead of treating every model output as erase permission:

1. **Native component association:** use learned text probability to associate
   whole original residual strokes with the selected OCR membership. Components
   owned mostly outside that membership remain whole protected obstacles, not
   cropped pieces. Require complete native containment, neighbor exclusion,
   measured background clearance and a bounded monochrome ink/antialias model.
2. **Phase-locked periodic surface:** find the smallest supported horizontal and
   vertical translation periods (1–32px). Fit a tile using only original context
   outside all text geometry, retain every known sample, and require <=2 channel
   levels maximum deviation and <=1 mean error. Every phase needs eight samples.
   Native residual strokes become the removal mask; repair copies the aligned
   tile only through that mask. This supports stable axis-aligned local repetition,
   not arbitrary rotated/scanned/nonstationary screentones.

Letter placement uses the complete original 4-connected safe component within
the **unchanged** source-local window. It retains every original source-clearance
pixel and excludes the same art/nonmember obstacles. This recovers space a narrow
rectangle wastes in tapered balloons, without permitting rectangle erasure.

## What failed and was rejected

- Plain union/intersection of CTD masks was not adopted.
- Upscaling synthetic dot crops to 1024px often hurt CTD. Native-resolution crops
  padded to its fixed input worked better in development; this variant was fixed
  before confirmation. Large pages still require the recorded fit transform.
- An early native association erased a gray art contour inside a text footprint.
  It was rejected. The kept rule requires one measured ink core, bounded color
  variation/transition, and only tiny varying antialias clusters. A high CTD score
  alone cannot override a different-color native interior.
- Weak periodic matching that averages away a line is not used: any known strong
  crossing stroke invalidates the tile, rather than being discarded as an outlier.
- No thresholds were changed after the confirmation run. Its rotated-text skip
  remains a real limitation, not a case tuned until it passed.

## Measurements

`test-results/native-background-frozen-v1/` contains 12 development and 12 new
confirmation overlays/surfaces, frozen before model inference. The selected
policy and exact source hashes were recorded in `selected-policy-v1.json` before
confirmation. This is one synthetic font and previously available manga pages,
not an unseen-title/page generalization benchmark.

With native-pad CTD and automatic flat-then-periodic decisions:

| Split | Positive cases restored | Exact known backgrounds | Low-noise periodic MAE | Art negatives skipped |
| --- | ---: | ---: | ---: | ---: |
| Development | 10/10 | 9 | 0.402 | 2/2 |
| Confirmation | 9/10 | 8 | 0.387 | 2/2 |

The confirmation rotated case was skipped. All completed repairs preserved pixels
outside their machine-selected masks exactly, with no changes to unoccluded
protected neighbor/art pixels in these fixtures. Exact periodic cases reconstruct
their known dot/stripe phase, unlike the earlier LaMa outputs. The low-noise
results are estimates, not recovery of the original random noise.

The saved real page improved from **6/8 to 8/8 eligible translations**, using the
genuine strict-binding, original font limits, complete wording, collision checks
and pixel-audit path. There are still 21 other saved units without supported strict
replacements; this is not a complete verified page translation. The two recovered
units render at 25px and 20px; no wording shortening or smaller minimum was used.
Their cleanplates were inspected before lettering: neighboring handwritten text
and balloon outlines remain visible. This is provisional visual evidence without
a clean real-page reference, not proof of artistic/linguistic correctness.

Artifacts:

- `test-results/native-background-development-v1/`: rejected early variant.
- `test-results/native-background-development-v2/` and `v3/`: calibration history.
- `test-results/native-background-confirmation-v1/`: frozen confirmation outputs.
- `test-results/native-mask-saved-page-v1/`: rejected over-strict ink-interior guard,
  6/8; `v2/`: 7/8, last unit blocked by layout; `v3/`: 8/8 with connected safe space.
- [Saved-page review](../test-results/native-mask-saved-page-v3/review.html).

## Boundaries

These trials use only the already approved CTD checkpoint and local stack. No new
models, dependencies, image uploads, provider calls or training were introduced.
CTD checkpoint rights remain unresolved for distribution/commercial deployment.
Native masks remain machine hypotheses: completely contained same-color artwork
can still be semantically ambiguous. Do not claim all fonts/backgrounds/orientations.

The experiment runner archives loaded source bytes at process start, so later
edits cannot silently change a run's recorded implementation. Original fixtures,
source bytes, failed outputs and historical reports are retained.
