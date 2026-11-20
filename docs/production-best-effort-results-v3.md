# Removal-first continuation v3

**Result: useful local gains, still not near-perfect manga translation.** Started
from `0e20212`. Final evidence is under the ignored local directory
`test-results/production-best-effort-v3/`; originals and earlier trials are retained.

## Changes retained

- Cleanup is independent of English placement. A font/fit failure reaches bounded
  source removal, and exact unavailable wording remains in the transcript.
  Large neighboring detection polygons cannot veto the current source's interior;
  prior committed cleanup and lettering remain protected.
- Final fallback expands intersecting strokes within a bounded collar, then uses
  an explicit local surface occlusion instead of broad Telea smears. Readable
  placement searches all collision-free local rectangle positions, preferring
  source proximity and respecting the requested font floor and exact wording.
- White-on-black text uses the correct foreground polarity for grouping and solid
  contrasting English. The old grouping remains reproducible with
  `automatic_manga_structure_v2`; current production uses
  `automatic_manga_structure_v3_polarity`. Six light-text controls retained their
  equivalent partitions against recomputed baselines.
- LaMa context margins can shrink to fit native 512px crops. Source pixels are
  never resized or clipped to satisfy that model's shape.
- TG27 keeps its official initial prompt. Explicit slash-separated alternate
  renditions may receive one bounded retry; both responses survive, and failed or
  still-alternative retries retain the primary. Literal source slashes are exempt.
- An explicitly configured local Manga OCR snapshot can reread isolated,
  annotation-free, low-score Japanese text in two presentations. Character
  agreement ignores whitespace/commas, but preserves questions, repetitions,
  long vowels and small kana. Raw readings, alternatives, crop hashes and exact
  source bindings survive. Actual crop conflicts, bounded geometry and raw-score
  consistency are checked. Agreement is not independent reader verification.
- Fixed the retained recognition-envelope crash (`crop=None` mistaken for a flat
  candidate). The coordinator and viewer forward the configured local reader.
  New rendering jobs default to aggressive removal / always-show; historical job
  options retain their saved behavior. The CLI translation budget defaults to 64.

OCR, translation and rendering remain separate, coordinated by `src/main.py`.
No manual masks/corrections, page-specific answer substitutions or model replacement
were used. Private single-user operation is the evaluated scope.

## Actual page results and visual inspection

Four fresh complete-pipeline runs are retained in `live-v3/`; final renderer
replays with their unchanged translations are in `render-v5/`. All four full
cleanplates and composites were inspected visually, including the failed v4 trial.

| Page | Fresh pipeline seconds / TG27 calls | Final displayed | Unchanged | Missing / unavailable | Final learned calls |
| --- | --- | ---: | ---: | --- | ---: |
| saved-example | 86.5 / 45 | 35 | 6 | 1 missing | 8 |
| ubunchu08_0007 | 57.0 / 26 | 19 | 5 | 0 | 5 |
| ubunchu07_0002 | 51.4 / 26 | 20 | 1 | 1 exact target lacks font coverage | 2 |
| ubunchu03_0006 | 52.7 / 30 | 24 | 2 | 0 | 4 |

These are display counts, not faithful-translation counts. Final removal fallback
handled 23 pending sources; 22 received exact readable placement. The remaining
source was cleaned despite its unsupported target glyphs. Final replay took
55.0 / 34.4 / 18.6 / 30.2 seconds in table order, with 19 learned calls and no backend
failures. Opaque cleanup does not reconstruct the lost art.

Selected crop evidence corrected the false “Good morning” reading to
“I don't really understand,” `ネットワー7の` to `ネットワークの`, and
`イニストール` to `インストール`. The four runs selected five alternatives using
34 local reads; the other two were Linux character normalization and long-vowel
punctuation, not five demonstrated semantic corrections.

### Visible failure ledger

| Evidence | Observed failure / disposition |
| --- | --- |
| `render-v5/saved-example/{clean,composite}.png` | Large white occlusions erase middle-panel art and borders. Top-panel learned cleanup invents texture. UI text remains missed/garbled; “Wgetz, make,” “P-code,” “Wisdom,” and “Top—” are not reliable readings/translations. |
| `render-v5/ubunchu08_0007/{clean,composite}.png` | Black-background English and bottom captions improved. The APT title is now an explicit gray occlusion; “APT (Advanced Persistent Threat)” and “the art of coffee” remain serious language errors. A partial title survives outside the detected source extent. |
| `render-v5/ubunchu07_0002/{clean,composite}.png` | One source is removed with no drawable exact replacement; large lower-panel white losses and “1941P Chi 1P4” remain. |
| `render-v5/ubunchu03_0006/{clean,composite}.png` | Missed lettering, disconnected fragments, intrusive top captions, an explanatory parenthesis, and invented lower-panel texture remain. This extra page did not establish generalization. |

## Tried and rejected

- Universal dialogue-prompt edits: controls introduced modality and worsened
  hallucination on garbled OCR. Retained only the narrow alternate-rendition retry.
- Blanket Manga replacement: 48 diagnostic reads included worse text. Retained
  only the bounded two-presentation selector above.
- Automatic reuse of failed-placement learned cleanplates: v4 reused seven
  candidates but introduced conspicuous repeated fake glyphs in the APT title.
  Final fallback no longer adopts those candidates. Their artifacts remain.
- Two extra large-mask LaMa calls and four inference-collar calls: native crops,
  unchanged commit support, zero outside-commit changes. They still produced
  copied glyphs, invented texture or seam damage. The wider inference collar was
  rejected, rather than treating smoother pixels as proof of reconstruction.

The available OCR/source extents and reconstruction quality are the demonstrated
limits here. More of the same calls showed diminishing returns. No robust general
cleanplate-quality selector or reliable recovery of missing source text was found.

## Verification and cost

- Full working-tree suite: **1,049 tests passed**, isolated mock scheduling root,
  120.1 seconds. Log: `/tmp/opencode/tralens-v3-tests-final.log`.
- Isolated commit-candidate suite: **1,004 tests, OK (40 unavailable private-evidence
  skips)**, 36.0 seconds. Log: `/tmp/opencode/tralens-v3-candidate-tests.log`.
  The final focused viewer/removal/reader check also passed 46 tests.
- `audit.py` imports neither renderer nor translation implementation. `audit.json`
  verifies **9,055 artifact hashes across 33 rendered runs**, original bytes/pixels,
  current font/license hashes, exact recomposition, source/request/result/transcript
  bindings, disjoint displayed membership, exact layout wording/font floors,
  native crop/mask requests and zero outside-support changes.
- `preservation-audit.json` verifies all 23 unrelated baseline file hashes, exact
  recovery of all three shared working files by reapplying only their pre-task
  changes to the commit candidate, and both original staged documentation deletions.
- Fresh matching native controls independently confirm no changes to **136,827
  earlier cleanup pixels and 139,318 earlier lettering pixels** across the four final
  pages. `native-audit-controls-v2/` matches visibility settings; the earlier
  `native-audit-controls/` used replacement-only and is retained as a nonmatching
  control, not used to assert preservation.
- Real browser upload through the isolated-worker viewer completed in **63.9s**:
  `viewer-final/6094e9707cc5445c8375f7d8d5f27b4a/`, 26 TG27 calls, eight local Manga
  reads, five LaMa calls, 19 displayed translations. Screenshot:
  `viewer-final-final.png`. The same visual limitations remain visible there.
- Recorded v3 total: **266 TG27 calls**, 92.25 daemon-seconds, 24,476 prompt tokens,
  2,010 generated tokens; **98 local Manga reads**; **91 LaMa calls**, 435.67 worker
  wall-seconds / 207.03 inference-seconds. These totals include diagnostics,
  failed-render/replay iterations and viewer verification, not synthetic tests.
  `costs.json` contains the per-response hashes and per-run breakdown. No monetary
  or energy price was supplied for the existing private runtimes.

The first full-suite attempt shared a live inference state root and encountered
lease conflicts; subsequent full tests used an isolated root. A later run exposed
one viewer-test indentation mistake, which was corrected before the passing suite.
Live Paddle v6 uses `.venv-paddle6`; the ordinary `.venv` lacks its runtime support.

## Evidence index and remaining limits

- `baseline.json`, `baseline-working.diff`, `baseline-index.diff`: pre-task work.
- `grouping-v1/`: seven-page polarity/control comparisons.
- `language-{v1,controls,minimal}/`, `manga-v2/`: retained requests/readings/results.
- `render-v1/`, `render-v2/`, `live-v1/`, `live-v2/`, `live-v2-render-v3/`,
  `live-v3/`, `render-v4/`, `render-v5/`: iteration history; v2's envelope crash is
  preserved, followed by a successful bound replay.
- `large-cleanup-diagnostic/`, `inference-collar-diagnostic/`: rejected final trials.
- `audit.py`, `audit.json`, `preservation-audit.json`, `costs.json`, `viewer-final/`:
  validation and costs. `evidence-index.json` inventories retained local evidence.

Model identities are unchanged: TG27 digest
`21b597a243ec4fd2f05bc0ce8ef9ef174a89b09255c330d670852dfaaaa74f72`;
LaMa SHA256 `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`.
The LaMa conversion's upstream checkpoint provenance remains unresolved for
production distribution. No independently assessed bilingual release subset,
genuinely untouched permitted title/style set, sustained live soak, or embedded
memory plateau has been established. These results do not establish release quality.
