# Exact cleanup protection v6

Continues `24a8281`. Cleanup can now remove lettering in unused spacing around
earlier edits. **Partial source lettering, conspicuous reconstructed patches and
serious translation errors remain.** Retained local output and comparisons:
`test-results/production-best-effort-v6/render-v1/` and `render-v1-review/`.

## Change

The aggressive and final-removal passes previously protected a two-pixel dilation
of earlier cleanup and English. That margin could contain original glyph edges
from the next source region, pinning them in place even when reconstruction had
correctly located them.

Both passes now protect **exact earlier cleanup and lettering pixels** during
removal. English and opaque-backing placement keep their separate two-pixel
clearance. Composition checks cleanup against exact prior support and new English
against placement clearance. Failed placement still commits source cleanup.
Neighboring source-geometry obstacles retain the existing source-priority rules.

The policies are `crop_local_aggressive_v9_exact_edit_protection`,
`remove_then_source_centered_patch_v6_exact_edit_protection`, and shared
`exact_commits_with_separate_layout_clearance_v1`.

Exact selected wording, font coverage/floors, unique source membership, native
image coordinates and inference/commit-mask separation remain enforced. TG27,
OCR, translation prompts and reconstruction budgets retain their existing behavior.

## Experiments and visual review

Four diagnostic overlays traced the small remnants to protected clearance. An
alternative that derived source support before protection did not address those
fragments. Four controlled native-crop LaMa trials reconstructed actual previous
edits from saved artifacts; dilating those edits reproduced all four v5 protected
masks exactly. The exact-protection trials visibly removed the diagnosed remnants.

The retained four-page replay has **98 exact replacements**, with the same English
layer pixels as v5 on every page. All four composites, all four standalone
cleanplates, all four diagnostic/trial cards and all 19 changed-region comparisons
were inspected. Overlapping comparison cards are not independent successes.

- `saved-example`: removed the dotted remnant beside “Okay.” and small dark marks
  at the book/ASCII and wired-connection/Top label boundaries.
- `ubunchu03_0006`: removed the thin remnants around “from” and “It’s already…”,
  and reduced boundary marks around the top caption and “No.” The existing
  four-line security-scan caption is retained.
- `ubunchu08_0007`: removed the dotted stroke between the two top-left captions
  and horizontal black interruptions at the title/subtitle cleanup boundaries.
  Larger reconstructed areas also changed within their declared commit supports.
- `ubunchu07_0002`: cleanplate and composite pixels are identical to v5.

| Page | Exact replacements | Final removals | LaMa calls | Changed clean pixels vs v5 |
| --- | ---: | ---: | ---: | ---: |
| saved-example | 35 | 4 | 11 | 244 |
| ubunchu08_0007 | 19 | 2 | 7 | 10,261 |
| ubunchu07_0002 | 20 | 4 | 3 | 0 |
| ubunchu03_0006 | 24 | 4 | 9 | 614 |

Fourteen selected targets equal their source, one proposal lacks a selected
translation, and one exact mixed-script target lacks font coverage. The latter,
`Try adding “Of” after “aこPー”.`, still has its source removed but remains undrawn.
Counts and pixel differences do not establish complete removal or accuracy.

## Remaining defects

The partial original title, surviving Japanese interface text and sound effects,
white backings, smooth gray reconstruction bands, cut panel/book detail and
intrusive captions remain. The title area still loses chest/background detail.
This change removes the diagnosed boundary fragments; it does not establish
complete source extents or recovery of original artwork.

Language failures including “Wgetz, make,” “P-code,” “Wisdom,” “Top—,”
“APT (Advanced Persistent Threat),” “the art of coffee,” and “1941P Chi 1P4” remain.
Remote request `b85a0d050d8a4cbf940b9a245cbcf7fc` is still unresolved and remote
inference stays paused. Its durable incident bytes match the baseline.

## Verification and cost

- Working-tree suite: **1,072 tests passed**, 120.789 seconds, with isolated
  scheduling state. Three new regressions cover clearance-ink removal in both
  passes, immutable previous edits, source cleanup after failed placement, and
  English/backing spacing. All three fail against the parent implementation.
- Independent audit: **1,315 artifact hashes**, four crop trials and exact replay
  of **55 cleanup/placement transactions**. It checks exact wording/recomposition,
  native request bindings, zero changes outside declared support, exact prior
  output and two-pixel placement clearance. Cleanup used **1,591 previously
  reserved clearance pixels**; that count includes background and is not a text
  removal score. Thirty raw page-model outputs and 63 protected requests passed.
- Earlier accepted **137,916 cleanup pixels and 141,539 lettering pixels** remain
  exact. Source image, OCR/translation preview and font evidence bytes match v5,
  as do all selected bindings. The 28-path pre-existing working/index baseline is
  checked before the task-only commit.
- Total: **34 LaMa calls**, zero failures, **162.021 worker seconds** and
  **77.864 model-inference seconds**; **four CTD calls**, **4.689 seconds**.
  Four LaMa calls were crop trials and thirty were the full-page replay. No new
  OCR, translation or remote calls occurred. No currency or energy rate was given.
- Exact previously approved CTD/LaMa checkpoints and ten pinned runtime wheels
  were restored into ignored `test-results/approved-local-runtime-v1/` after their
  temporary directories disappeared. Checkpoint bytes and evaluation scope match
  the retained acquisition records. The path change produces configuration
  revision `9ef314d316aef2453b46441a5760ddd9014884bafe7dc2a2bc809dcf4ab048d8`.

Reproduce with `.venv/bin/python test-results/production-best-effort-v6/replay.py
test-results/production-best-effort-v6/NEW-DIRECTORY`. Each replay retains its
implementation snapshot. `audit.py`, `compare.py`, `costs.py`, `visual-review.md`
and `README.md` describe the local checks. Isolated task-only test results and
committed-tree equality are retained in `verification.json` and `commit.json`.
Independent bilingual review, untouched-page validation, sustained live soak,
memory plateau and distribution provenance remain open.
