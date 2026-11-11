# Private best-effort checkpoint (not release-ready)

## Verified continuation v2

Started from actual HEAD `23d90d2`, not only implementation ancestor `8d9832d`.
Independent replay verified **3,853 manifest artifacts across 15 v1 page runs**,
exact recomposition, outside-support equality, prior cleanup/English and transcript
bindings. This count excludes frozen-input files included in v1's 3,925 total.

- Integrated hardening: `5fd9433`, worker-entrypoint correction `71cca44`, and
  terminal-state orphan recovery/crash-publication tests `fdf8df5`.
- **28/28 approved context forwards completed**, 62.0s inference / 129.0s worker
  wall time; no outside-commit or protected-prior-edit changes. Neighbor masking
  has useful local gains **and structural regressions**, so it remains experimental.
  [Per-case outcomes, costs and rejection](reconstruction-context-results-v2.md).
- The seven-page offline presentation trial produced no coverage gain and only
  one 18px -> 19px change with identical wrapping. Rejected from production.
- Language failure attribution and reader availability are documented; 36 new
  request-only comparisons are frozen with **zero dispatches**. They require
  separate approval and independent assessment. [Language plan](language-redesign-v2.md).
- Portable full suite: **1,010 tests passed**, plus **128 scoped tests** in an isolated
  committed-content candidate tree. Synthetic isolated/embedded viewer
  soak and fault-injection evidence are below. These establish bounded engineering
  behavior, not live sustained operation or near-perfect translation quality.

Evidence index: [v2 README](../test-results/production-best-effort-v2/README.md).
All approved learned budgets are exhausted. No additional inference is authorized.
Final offline replay also checked 3,432 layout-manifest hashes, 4,312 synthetic-job
manifest hashes, 220 frozen context files, 196 context-output/inspection files and
61 frozen language files. Preservation checks matched 192 untouched pre-task files,
replayed the earlier shared-server edits exactly and kept the staged diff byte-identical.

## Integrated implementation

Explicit `removal_policy=aggressive` runs crop-local recovery after supported
cleanup and before independent `always-show` backing/captions. `preserve` retains
uncertain artwork. Missing settings in APIs/history remain `legacy`. New viewer
controls choose aggressive removal and always-show; LaMa is selected only when
the operator supplies an explicit approved model/runtime configuration.

`reconstruction_policy=lama` reaches a source-bound, digest-checked CPU ONNX worker
through `src/main.py` and the viewer. There is no discovery/download. Configuration
permits **private noncommercial evaluation only**, not a production provenance
claim. Classical-only runs are labelled classical. Exact bound English is retained
in `transcript.json` even when image placement fails; transcript-only is not counted
as replacement. Crop evidence, inference/commit masks, cleanplates, text and page
layers are retained separately. Generated art is speculative.

The private viewer CLI uses a fresh process group per job, a 30-minute wall deadline,
12 GiB worker address-space limit, 2 GiB job/8 GiB root storage admission limits,
durable worker state/logs, cancellation and no automatic retries. Stage checkpoints
survive process timeout. A cross-process application lease serializes the main
pipeline and preview entry points. This does not control unrelated Ollama clients;
killing a local client is **not proof that remote generation stopped**. Embedding
`App(Config(...))` retains the old thread mode unless `isolated_workers=True`.

Worker launch now waits for durable ownership publication. Restart recovery checks
boot/process identity and inherited ownership tokens before signaling an orphaned
group; foreign or unverifiable live groups block admission. It also checks terminal
worker records: a supervisor can crash after publishing completion but before its
descendant-cleanup `finally` block. `worker_mode` exposes actual process/thread mode.
Decoded worker crop/inference/commit/protected masks and output digest are validated.
Single-file publication is atomic and directory-fsynced; completion manifests are
published last. Partial runs retain earlier complete artifacts.

Example operator invocation (only with already approved local assets):

```sh
.venv-paddle6/bin/python tools/dev_frontend.py --fonts /absolute/fonts.json \
  --reconstruction-config /absolute/reconstruction.json
```

Config keys: `backend: lama-onnx-512`, absolute `model` and `python`, pinned
`sha256: 1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`,
`use_scope: private-noncommercial-evaluation`. The approved temporary assets are
not dependencies installed into the application environment.

The existing `.venv-paddle6` supplies the configured Paddle v6 runtime
(PaddleOCR 3.7.0/PaddleX 3.7.2). `.venv` has 3.4.0/3.4.2 and is unsuitable for live
v6 operation; portable synthetic tests do not validate its model initialization.

### Lost remote request recovery

Real Ollama HTTP dispatch journals intent under `TRALENS_STATE_ROOT` (default
`~/.cache/tralens`) before sending. A terminal `done: true` response clears it.
Crash, timeout, connection loss or nonterminal response leaves durable uncertainty
and **blocks further application dispatch to the same configured endpoint**.
No timeout expiry, automatic retry or `/api/ps` idle inference clears it. Injected
transports are caller-owned; they must provide their own remote scheduling contract.

```sh
.venv/bin/python tools/inference_state.py --endpoint http://100.121.133.88:11434
# Only after externally confirming this request has ended on the server:
.venv/bin/python tools/inference_state.py --endpoint http://100.121.133.88:11434 \
  --acknowledge-stopped REQUEST_ID
```

Use the same state root and exact configured endpoint for inspection/acknowledgement.
The request ID must match; acknowledgement retains an incident record. Without
reliable server-side confirmation, leave the endpoint blocked. This does not cancel
remote work, serialize unrelated clients, or identify aliases of the same daemon.

### Operational evidence and limits

Fault tests cover changed decoded worker inputs, failed durable admission,
disk-full-before-replace, quota failure, cancellation/deadline, cross-process lease,
orphan descendants versus foreign groups, terminal-state orphan recovery, local
HTTP timeout with continued synthetic remote work, and embedding history-write
failure. Actual child-process kills on both sides of checkpoint replacement leave
complete old/new JSON respectively. Failed rendering manifest publication leaves
earlier valid images and the exact original, without a partial completion marker.
This is process-crash testing, not a filesystem/power-loss certification.

The first synthetic soak correctly hit the existing eight-job session admission
limit after eight successful jobs. The revised harness restores a new App at each
eight-job boundary. After the terminal-recovery fix, **20 isolated + 20 embedded**
jobs completed in **50.86s**:
exact original/artifact hashes, final recomposition, locality and two bound English
transcripts per job; restored histories; no surviving owned group; descriptors
**5 -> 5**. Retained artifacts: 5,603,742 bytes. Isolated child peak RSS <=253,164 KiB;
embedded parent RSS rose 252,448 -> 303,492 KiB during its 20 jobs. No memory plateau
or long-duration readiness claim follows. Provider construction is synthetic; real
coordinator, renderer, supervisor and worker resource bootstrap execute. No real
model/OCR/provider inference occurred. Logs and all failed/partial attempts remain.
The preceding 40-job run (49.33s) and initial eight jobs are retained separately:
88 successful synthetic jobs in total, not 88 live-model translations.

## Historical v1 evidence and costs

Ignored local evidence: `test-results/production-best-effort-v1/`; start with its
[inspection index](../test-results/production-best-effort-v1/README.md). Reproduction:
`tools/evaluate_reconstruction.py --freeze`, then separate `--run synthetic` and
`--run pages` with explicit root/config/cases. Existing artifacts refuse overwrite.
Seven saved pages reuse existing OCR and English; they are not untouched-title
confirmation. Original/model/input hashes were frozen before forwards.

- **64 learned forwards total:** eight synthetic mask ablations, 47 initial page
  trials, eight scheduling-repair trials, one inference-context probe. No failures
  in these real worker calls. Typical CPU worker RSS was about 1.2 GiB; raw load,
  inference, wall time and RSS are saved per call. Initial page learned work took
  roughly 9–41 seconds/page; the repaired page used about 41 seconds.
- On expanded synthetic masks, LaMa versus classical mask MAE (0–255) was
  **9.18 vs 63.93** for interrupted lines, **19.14 vs 30.32** for screentone,
  **2.86 vs 13.00** for colored outlined lettering. White measured fill was exact
  and better than LaMa; production avoids learned calls for measured uniform fill.
  These are four constructed cases, not general quality estimates.
- Every frozen page's pre-existing cleanup/text support remained pixel-exact.
  Baseline preservation evidence covers **68 prior replacements**. All audited
  output pixels outside declared support remained exact.
- Final offline verification checked **3,925 artifact hashes and 15 page runs**
  (both backends on seven pages plus the scheduling-v2 page). Baseline cleanup/text
  and pixels outside declared cleanup/text/backing support were exact in every run.
  `final-verification.json` and `verify.py` retain the checks; no new inference ran.
- Initial seven-page output added **35 aggressive replacements**, including
  **15 selected learned results**; these are *unverified edits*, not clean passes.
  Backing/captions and transcript-only cases are counted separately in manifests.
- First scheduler wasted narrow-mask calls before choosing expanded classical
  output. Cheap residual assessment now defers those learned calls to expanded
  masks. On the fixed diagnostic page, learned selections increased **0 → 7** at
  the same eight-call ceiling. Visual gains include header/UI line continuation
  and balloon boundaries; stylized lettering still produces invented strokes.
- Final context probe masked neighboring detected lettering **for inference only**,
  retaining the same commit mask. It removed conspicuous copied glyphs from one
  balloon cleanplate with zero outside-commit edits. This is promising but remains
  an isolated diagnostic, not a default policy validated across pages.

## Historical v1 language diagnosis: do not deploy these trials

The v1 `language-v2/` directory freezes 20 simple block/line transcriptions, four geometry-independent
grouping calls, 14 English-only reviews and two targeted English follow-ups.
`tools/experimental/` isolates the rejected protocols. Neither is imported by the
production pipeline. TranslateGemma 27B remains primary.

- Simpler transcription recovered some letters but lost punctuation, copied
  neighboring text, invented names, normalized visible colloquial Japanese, and
  generated a long repeated string while reporting `uncertain=false`. Art negatives
  also produced confident kana. The original mistaken reader was not a veto;
  removing that veto did not establish reliable source selection.
- Grouping disagreed between text/image presentations; one output partition was
  empty. Two motivating cases cannot validate an automatic relation selector.
  No grouped candidate was promoted, so no parent duplication was introduced.
- English-only review can run without OCR change, but it retained the codec error
  and lost stutter/concessive meaning. Three of 14 outputs contradicted their own
  `retain/source_uncertain` status and were rejected. Independent-first informal
  dialogue guidance used the remaining two calls: it identified the concessive
  omission but still retained the wrong primary. **No validated selected English
  improvements; zero changed-source TG27 calls.**

Do not count flags, candidate letters or model calls as page-language improvements.
Reference judgments remain reused provisional agent assessments, not bilingual gold.

## Recommendation / failure ledger

| Item | Status and evidence |
| --- | --- |
| Independent removal/visibility, exact transcripts and prior-edit reservations | **Implemented and validated within tests/local evidence**; actual coordinator/viewer paths audited. |
| Residual-aware scheduler v2 | **Implemented**; earlier v1 gained relevant learned candidates at fixed quota. Quality ranking remains heuristic. |
| Old narrow-mask-first scheduler | **Superseded** by v2; old artifacts retained. |
| Worker bindings, atomic publication, private job ownership/recovery, remote uncertainty block | **Implemented and fault-tested**, including real process kills; bounded synthetic soak. |
| Blanket inference-neighbor masking | **Experimentally rejected for production**: copied-glyph gains coexist with lost panel structure; candidate experiment only. |
| Closed-light aggressive presentation retry | **Experimentally rejected**: one 1px font gain, same wrap, no coverage gain on seven pages. |
| Earlier transcription/grouping/English protocols | **Experimentally rejected**; no validated selected gains. Optional failed review cannot discard valid primaries. |
| New language comparison | **Frozen, blocked on approval/assessment/adapter checks**; 36 requests, no dispatch, no selector. |
| Structure-aware context, extent and phase-aware restoration extensions | **Deferred** pending independent evidence and a new frozen allocation; more masking is not automatically better. |
| Public-service security | **Separate release scope**; existing Host/origin/token protections retained, no public authentication/artifact-authorization claim. |

## Remaining failure classes and next bounded work

1. **Incomplete source membership/extents:** large outlined labels and associated
   punctuation remain beyond detector polygons. Expansion alone cannot determine
   semantic ownership. Next: alternative-reader/source-run extent evidence and
   source-bound grouping with independently checked negatives.
2. **Copied/invented lettering:** v2 confirms contamination in some crops but rejects
    blanket neighbor masking. Next: structure-aware context exclusion, preserving
    useful line evidence, evaluated against these retained failures and new controls.
3. **Art loss and damaged boundaries:** broad masks can erase faces, frames and
   texture; LaMa often improves Telea smearing but is not consistently acceptable.
   Native crop side is limited to 512. Context/scale ablations precede new families.
4. **English placement:** exact wording and font floors are preserved, but dense
   balloons have awkward narrow wrapping; captions and source text remain together.
   No page-wide aesthetic/layout or language quality guarantee is established.
5. **Operations:** orphan recovery, crash publication, disk-full/quota and remote
    scheduling now have fault evidence. Reliable remote cancellation remains absent;
    durable blocking enforces that limitation. Sustained live-model end-to-end soak
    and embedded memory stability remain unvalidated.

All v1 and v2 approved inference allocations are exhausted. The old proposed context
test has now run; do not repeat it. Next steps are to obtain independent bilingual
labels and genuinely untouched permitted examples, check the frozen language
request/adapter contracts, then request a **separate** <=36-call / 3,000-call-second
comparison. At most four changed-source TG27 calls (<=240s) are conditional on an
actually validated changed selection; no such selection currently exists. No
language allocation has been granted. New reconstruction methods need their own
frozen controls/approval. New model families still require separate provenance and
resource decisions; the existing assets do not have production clearance.

## Release checklist

Engineering scope is the private single-user Linux viewer/coordinator, explicitly
configured existing local assets, native reconstruction crops <=512² and bound
English with font coverage/floors. Production reconstruction limits remain eight
calls / 120 learned seconds per page. No title/style/language subset has yet passed
an independently frozen near-perfect-quality acceptance test.

| Criterion | Evidence / status |
| --- | --- |
| Separate policies and learned coordinator/viewer path | Implemented; portable routing tests and real renderer comparisons. |
| Exact bindings, wording, locality, protected successes | Independent v1 replay; decoded-worker fault tests; all 32 v2 slots pass fixed-commit/prior-edit equality. |
| Residual-driven relevant retry | Implemented; synthetic masks and scheduling diagnosis above. |
| Full source removal/readable image availability | **Not met:** visible residuals, captions, transcript-only and missing source remain. |
| Reliable source selection/grouping/English review | **Not met:** bounded new trials rejected; primaries preserved. |
| Private evaluation engineering | Implemented hardening and bounded synthetic coordinator/viewer soak; explicitly configured assets/runtime required. |
| Sustained live-model operation / embedded memory stability | **Not established**; synthetic providers and short trials cannot certify these. |
| Portable checks | **1,010-test** working-tree full suite and **128-test** isolated candidate-tree suite passed; fault/entrypoint logs retained. |
| Learned model production provenance | **Blocked:** conversion's original checkpoint chain/equivalence unresolved; evaluation approval only. |
| Independent bilingual adequacy/severe-error threshold | **Blocked:** no independently checked release subset. |
| Untouched titles/styles | **Blocked:** no genuinely untouched permitted confirmation set supplied. |

This checkpoint is useful integrated engineering with demonstrated local gains,
not a production-ready or near-perfect manga translation claim.

Implementation ancestry: `8d9832d`, `23d90d2`, `5fd9433`, `71cca44`, `fdf8df5`.
Earlier shared-file edits, dependency/network work and both staged documentation
removals were preserved. Nothing was pushed or published.
