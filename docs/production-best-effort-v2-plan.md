# Checkpoint audit and bounded continuation v2

Starting HEAD: `23d90d2`; implementation ancestor `8d9832d`. Pre-existing
coordinator, network/frontend, dependencies, balloon/language experiments and two
staged document removals remain owned by earlier work. Exact pre-task bytes/index
are frozen under `test-results/production-best-effort-v2/`.

## Offline interventions and criteria (before changes)

1. Revalidate v1 hashes and pixel support independently, including final backing
   support and prior successful edits. Audit routing, classification and transcript
   bindings against actual coordinator/viewer code. No model dispatch.
2. Worker integrity: validate serialized request revision, decoded RGB/mask hashes,
   transform and explicit model identity before native model loading. Corrupt data
   must fail hard, not become an aesthetic fallback. Preserve symmetric image/mask
   padding and native scale. Test malformed payloads without ONNX inference.
3. Operations: durable process ownership + startup handshake, identity-checked
   orphan recovery, cancellation/timeout preserving checkpoints; atomic durable
   publication, injected disk-full/quota/crash faults. Never signal an unrelated
   process or trust a reused PID. Exercise isolated and embedding modes separately.
4. Remote scheduling: persist pending HTTP generation before dispatch. An interrupted
   request blocks further application dispatch to that endpoint until its terminal
   response or explicit operator confirmation of server-side completion. Neither
   `/api/ps`, local process death nor a timeout proves cancellation. Synthetic HTTP
   providers test this; no live generation/control calls.
5. Layout candidate investigation: bounded original closed-light presentation space
   may broaden narrow aggressive frames independently of removal. Freeze unchanged
   seven-page classical baseline first. Adopt only if exact wording/font floor,
   locality and every prior edit are preserved, reading-size review shows useful
   gains, and failures/exclusions remain visible. At most two offline page variants;
   no tuning loop on the same examples. Defer if evidence is insufficient.

## Inference-context matrix (approval required)

Maximum proposed allocation: **32 actual LaMa forwards**, no retries, new models or
downloads. Eight category-selected cases × two native context extents × two masks.
Categories: copied glyphs; white balloon control; line/border; screentone; outlined
lettering; dense neighboring text; artwork contact; protected prior success.
Use existing source-bound masks and originals. Reused pages stay labelled reused.
Freeze IDs, source/binding/mask hashes, native crop bounds and padding before any
fresh result. Select extents fitting 512² without rescaling. Within each extent,
only inference mask changes; final page-coordinate commit support is identical
across all four variants. Hide relevant detected neighbors in B, preserve them in
the final composition. Record exclusions instead of silently substituting cases.

Measure pre-English remnants/halos, copied/invented strokes, structural continuity,
texture/art damage, reading-size final output, exact outside-commit and prior-edit
preservation, load/inference/wall/RSS. Agent visual assessment remains provisional.
No default promotion from one repaired crop; severe control regressions or mixed
evidence retain the current policy. Runtime cap 30 seconds/forward, 960 seconds
total, CPU-only existing pinned worker, fresh output directories.

**No new inference is authorized by this plan.** Finish useful offline work, freeze
the source-bound matrix, then request explicit allocation approval.

## Language redesign (separate approval required)

Inventory configured local readers and exact file/tag identities without loading.
Attribute saved failures separately to generation, extent, transcription, grouping,
translation, contradictory metadata and selection. No primary changes.
Freeze a separate comparison of independent English candidate generation versus
primary-conditioned review on identical Japanese/context, and transcription/extent
and relation-specific grouping controls. Separate candidate quality from selector
quality. Only a changed selected source/membership permits a new bound TG27 result.
Do not deploy the rejected protocols or use grammaticality/agreement as truth.

## Stopping conditions

No fresh inference without explicit approval; no acquisition. Independently checked
bilingual release judgments, untouched permitted titles and model production
provenance remain required for release claims. Commit validated scoped checkpoints;
report remaining layout/removal/language/operational limits separately.
