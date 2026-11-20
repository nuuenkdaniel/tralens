# Private single-user best-effort integration

## Frozen starting point and approval

HEAD `afe40c2`, with balloon/source-review and unrelated dependency/network work
uncommitted. `test-results/production-best-effort-v1/pre-task*`, `index.diff` and
`worktree.diff` preserve the starting state. Viewer stopped, Ollama idle at start.

Target: **private single-user**, not a public upload service. User approved
reacquisition of the pinned 208 MB Carve LaMa conversion and <=150 MB verified
runtime wheels, for private noncommercial evaluation only. No CTD/manga-tuned or
generative checkpoint acquisition is approved. No upstream implementation copying.
The missing temporary runtime is a real dependency, not a classical substitute.

## Acceptance targets (before tuning)

Supported initial scope: existing local Japanese→English still-image pipeline,
RGB/opaque grayscale/alpha PNG/JPEG <=4M pixels and 4096px side, supplied licensed
font manifest, local configured TG27. Reconstruction trials use native crops up
to 512² (fixed ONNX input), at most two learned calls/region, 8/page, 120 seconds
total/page, 30 seconds/call. Evaluation cap: 64 learned forwards across mask/context
ablations and real pages, 24 visual/source/grouping calls and 16 English-review
calls plus 8 changed-source TG27 requests. Checkpoint before exceeding caps.

- Integrity: 100% exact pixels outside declared commit/text support; zero foreign
  bindings, overwritten accepted text, dropped target characters or duplicate members.
- Availability: 100% bound available English retained in image or source-bound
  transcript; transcript-only is not image replacement. No >=16px requested text
  reported readable when rendered below that floor.
- Synthetic removal: >=99% known target alpha support removed in the final bounded
  mask; damage/reference error and protected-neighbor changes separately measured.
  Zero protected edits; severe collateral damage is a release blocker, not hidden
  by a smoothness score. Learned results should outperform cheap fill/classical on
  at least the structural challenge class without degrading ordinary white balloons.
- Real pages: all protected successful baseline cleanup/text exact; reduce visible
  residual lettering on reviewed difficult regions. Report art loss, residuals,
  backing/captions and layout defects individually; generated art is speculative.
- Language: provisional progress is diagnostic only. Release requires independent
  bilingual assessment (>=98% adequate, zero severe meaning errors on a frozen
  independently checked subset) and confirmation on untouched titles/styles.
  No such independent reference set is currently supplied: release claim blocked.
- Operations: bounded job time/storage/retries, cancellation and worker timeout/OOM
  preserve completed checkpoints. Timings/RSS measured, not inferred from call counts.

## Recommendation ledger

| Category | Starting disposition / next justified work |
| --- | --- |
| Independent policies | Implement removal/reconstruction separately from visibility; legacy histories retain behavior. |
| Aggressive mask ladder | Implement crop-local geometry/support candidates; native connected-art rejection cannot veto bounded cuts. |
| Generic LaMa | Reacquire approved evaluation artifact; integrate isolated direct-inpainting worker, digest/contract/locality checks. Production provenance unresolved. |
| Manga-tuned LaMa | Blocked pending source-checkpoint terms and acquisition approval; model card MIT alone insufficient. |
| Residual-driven recovery | Implement mask expansion/context changes; compare independent signals, not OCR silence. |
| Exact composition/layout | Reuse exact-character typography, protect committed output; plan before cleanup, retain transcript fallback. |
| Crop-local/fair budgets | Optional recovery separate from native/visibility; crop-local neighbor index and masks. Baseline full-page work remains to audit. |
| Source rereads | Diagnose frozen protocol contradictions; simpler line versus block protocol and alternative reader, no original-reader veto as sole selector. |
| Utterance grouping | Geometry-independent source relationships; requires controlled real merges and negatives before selection. |
| English review | Correct source/bad English needs separate bound target/primary/context review, not OCR rerun. |
| Periodic/texture transfer | Existing strict phase-locked route retained; broaden only on measured phase/line failures. |
| AOT challenger | Deferred until generic/manga LaMa mask/context comparison warrants extra model; no approved weights/runtime. |
| RORem/SDXL | Deferred: expensive, separate terms/dependencies and approval required; direct inpainting first. |
| Structural/tone separation | Deferred until failure corpus demonstrates direct-model limit; known-line synthetic baseline needed. |
| Additional generative models | Deferred; no useful direct-route evidence yet, no acquisition approval. |
| Durable jobs/scheduling | Audit current thread worker; process deadlines/cancellation, lock, persisted settings/evidence. |
| Public security | Outside selected private scope; authentication/authorization/TLS/rate limiting required before public service. |
| Untouched confirmation | Blocked for release until genuinely unseen permitted titles and bilingual evaluation are supplied. Reused pages labeled honestly. |

## Checkpoint dispositions

See [results and release checklist](production-best-effort-results.md) for evidence.

- Implemented/evaluated: independent policies, crop-local geometry mask ladder,
  isolated LaMa interface/worker, exact composition/transcript, bounded mask retries,
  fairer learned-call scheduling, viewer settings/history, private process supervisor,
  deadlines/cancellation/storage admission and application inference lease.
- Experimentally rejected for deployment: simple line/block source transcription,
  independent English-only review (including a targeted follow-up), and two-case
  text/image grouping trial. Raw responses preserved; own harnesses isolated in
  `tools/experimental/`, with no production selection.
- Promising but not broadly validated: inference-only masking of neighboring text
  reduced copied glyphs in the final context probe. New evaluation budget required.
- Still blocked: learned-mask CTD weights/rights, manga-tuned LaMa provenance and
  acquisition, generic conversion production provenance, independent bilingual
  release assessment and untouched-title confirmation.
- Deferred for concrete reasons: new AOT/generative weights until mask/context
  diagnosis is exhausted; structural/tone separation and patch transfer until a
  broader reference-art corpus can distinguish real improvements from smoother
  damage. Current four overlays establish no generalization claim.
- Remaining architecture work: automatic source selection, renderer-independent
  validated utterance graph, page-wide layout optimization, remote cancellation,
  restart process recovery and sustained end-to-end private-service soak.

Frozen budgets exhausted: 64 learned, 24 source/grouping, 16 English review calls.
No tests/call counts/replacement totals alone constitute release readiness.
