# Private best-effort checkpoint (not release-ready)

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

Example operator invocation (only with already approved local assets):

```sh
.venv/bin/python tools/dev_frontend.py --fonts /absolute/fonts.json \
  --reconstruction-config /absolute/reconstruction.json
```

Config keys: `backend: lama-onnx-512`, absolute `model` and `python`, pinned
`sha256: 1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`,
`use_scope: private-noncommercial-evaluation`. The approved temporary assets are
not dependencies installed into the application environment.

## Frozen evidence and costs

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

## Language diagnosis: do not deploy these trials

`language-v2/` freezes 20 simple block/line transcriptions, four geometry-independent
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

## Remaining failure classes and next bounded work

1. **Incomplete source membership/extents:** large outlined labels and associated
   punctuation remain beyond detector polygons. Expansion alone cannot determine
   semantic ownership. Next: alternative-reader/source-run extent evidence and
   source-bound grouping with independently checked negatives.
2. **Copied/invented lettering:** neighboring unmasked glyphs contaminate LaMa.
   Next: controlled inference-only context masks, same commit masks, on all seven
   pages plus untouched permitted pages; compare residuals and structural damage.
3. **Art loss and damaged boundaries:** broad masks can erase faces, frames and
   texture; LaMa often improves Telea smearing but is not consistently acceptable.
   Native crop side is limited to 512. Context/scale ablations precede new families.
4. **English placement:** exact wording and font floors are preserved, but dense
   balloons have awkward narrow wrapping; captions and source text remain together.
   No page-wide aesthetic/layout or language quality guarantee is established.
5. **Operations:** local cancellation/timeout works, but remote request cancellation,
   restart recovery of an orphaned inference process group, crash-during-publication,
   disk-full behavior and a sustained private end-to-end soak remain release gaps.

All frozen learned/source/grouping/English budgets are exhausted. Next justified
checkpoint requires a new bounded evaluation allocation, not implicit extra calls:
**up to 32 learned context/mask forwards**, plus an independently designed language
comparison with alternative-reader availability verified first. No additional
weights are needed for the context test. Manga-tuned LaMa/AOT/generative acquisition
still requires separate provenance review and approval.

## Release checklist

| Criterion | Evidence / status |
| --- | --- |
| Separate policies and learned coordinator/viewer path | Implemented; portable routing tests and real renderer comparisons. |
| Exact bindings, wording, locality, protected successes | Portable corruption/timeout/OOM tests; seven-page pixel comparison passes. |
| Residual-driven relevant retry | Implemented; synthetic masks and scheduling diagnosis above. |
| Full source removal/readable image availability | **Not met:** visible residuals, captions, transcript-only and missing source remain. |
| Reliable source selection/grouping/English review | **Not met:** bounded new trials rejected; primaries preserved. |
| Private job hardening | Implemented local supervisor/lease/cancellation; outstanding gaps above. |
| Portable checks | 997-test full suite, 88 final working-tree scoped tests, and 82 isolated implementation-tree tests passed. |
| Learned model production provenance | **Blocked:** conversion's original checkpoint chain/equivalence unresolved; evaluation approval only. |
| Independent bilingual adequacy/severe-error threshold | **Blocked:** no independently checked release subset. |
| Untouched titles/styles | **Blocked:** no genuinely untouched permitted confirmation set supplied. |

This checkpoint is useful integrated engineering with demonstrated local gains,
not a production-ready or near-perfect manga translation claim.

Implementation checkpoint: `8d9832d` (bounded reconstruction and private viewer
jobs). Earlier shared-file edits and staged documentation removals were preserved.
