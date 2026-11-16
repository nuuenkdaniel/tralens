# Source-review pilot: reject deployment of this route

The independent queue/source-candidate foundation is implemented and tested, but
**this frozen recovery route did not demonstrate a usable linguistic improvement**.
Do not enable it in the viewer or treat additional calls as success. Defaults,
TG27, selected sources/English, cleanup and rendering remain unchanged.

## Evidence

`test-results/language-review-foundation-v1/` contains frozen originals/previews,
native baselines, requests/crops, all raw responses, provisional references,
per-case judgments, costs, verification and `inspection.html` at native page size.
`PLAN.md` and `pre-inference-hashes.json` were written before fresh outputs. No
prompt/routing changes or inference retry followed the failed pilot.

Seven-page routing-only comparison: **170 units considered, 28 requested,
138 budget-excluded, 4 unsupported**. Five pages received inference: three reused
motivating/protected pages and two reserved-from-queue-tuning pages with reused
OCR/cleanup history. These are not globally untouched pages or independent gold.

| Intervention | Actual result |
| --- | --- |
| Review routing | Independent of renderer; high-confidence restart and APT errors requested (scores 0.946 and 0.944). Rendered codec eligible but budget-excluded. |
| Visual proposals | 20 calls: 12 uncertain retained, 2 unchanged clear corroborations, 6 changed clear proposals rejected by the separate reader. |
| Independent reading | 7 crop calls; no complete changed-source agreement. Agreement would still not certify truth. |
| New TG27 candidate translation | **0 calls, 0 candidates**. Not a translation-quality win. |
| Selected source / English / page edits | **0 changes**; all primary records and baseline page bytes preserved. |
| Grouping / contextual English review | Not implemented/evaluated in this step; 0 calls. |

Both clear unchanged corroborations (`おー`, `もういいのよ`) retain the valid
primary English without retranslating. Four additional matching rereads are
uncertain and remain inspectable, not falsely counted as clear corroboration.

## Material failures

- **APT-Man:** exact visible proposed text, but the typed state says `unreadable`
  while its prose calls it clear. Retain the contradiction; do not override it
  to force a motivating success. No independent-reader/TG27 request followed.
- **Computer restart:** Qwen corrects several characters but omits visible `の`:
  `コンピューター再起動`. Paddle repeats the old wrong reading. No complete
  source correction or new English was produced.
- **Technical/credit text:** bad Latin-to-kana substitutions, copy/copy-paste
  confusion, command omissions, and empty credit/update proposals. Exact reader
  agreement fails rather than granting plausible-software-word substitutions.
- **Art negative:** a false raw digit becomes Qwen `≒`; the independent reader
  returns empty text. Neither is accepted or used to erase artwork.
- **Protected ending:** Qwen regresses `めでたしめでたし` to
  `めずたいめずたい。`, labeled unreadable. Original selected text/English survives.
- **Incomplete visible codec UI:** the proposal misreads the Japanese and drops
  Latin characters; its reason names unseen `ffmpeg` completion. Clipping/uncertainty
  prevents acceptance. This is separate from the rendered hesitant-codec dialogue,
  which was not requested under the frozen budget.
- Adjacent replies/partial lines stay separate in the observed proposals; no
  grouping safety claim follows because this foundation cannot propose groups.
  Furigana/neighbor-center and stale-binding guards have synthetic portable tests,
  not a successful real grouped-dialogue evaluation.

Existing English alternative menus, omissions, hallucinations and tone problems
remain unresolved. No candidate English exists to score those dimensions; no
mechanical slash splitting or fluency-based selection was introduced.

## Cost and preservation

Actual ceilings: 20/20 visual, 7/8 crop-reader, 0/8 TG27 calls; 5 Qwen tag-scoped
release control calls, no preloads. Qwen request wall time totals **54.9 seconds**,
including cold loads; reported 15,487 prompt and 2,770 output tokens. This excludes
Paddle startup/crop time and is not end-to-end throughput or monetary cost.
No transport/parse failures occurred; **12 uncertain responses are not successes**.
Source/reader protocol failures and optional translation failures are covered in
portable tests. No original file, old preview, raw geometry or successful page
layer was modified. Viewer jobs/model server were idle before and after the run;
no service was restarted. Occupancy checks are not an atomic shared-server lock.

The final full portable suite passed **987 tests**, including the source-binding,
global-budget and occupancy tests. This is safety/contract
evidence, not language-quality evidence. Run `verify.py` offline to revalidate
all bindings, frozen code/requests/references and unrelated/staged work.

## Decision

Keep only the explicit, disabled-by-default audit/candidate harness as reproducible
foundation; **reject this prompt/reader/routing combination as a deployable quality
fix**. Do not accumulate more ranking rules, relax uncertainty to force APT-Man,
or add viewer/grouping/context review on this evidence. The next source experiment
needs a separately frozen protocol and more reliable independent text evidence;
do not retune against these reused/reserved pilot pages and claim held-out gains.
