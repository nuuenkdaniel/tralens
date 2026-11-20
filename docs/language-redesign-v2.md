# Language continuation: diagnosed, not deployed

No new language inference is authorized or used in this checkpoint. TG27 remains
primary. The earlier source, grouping and primary-conditioned English experiments
remain rejected. References below are provisional agent judgments, not bilingual
gold; no accuracy or selected-language improvement is claimed.

## Availability verified without loading

`test-results/production-best-effort-v2/reader-inventory.json` records full hashes,
installed runtime versions and read-only endpoint responses.

- Configured endpoint reports `qwen3.8:latest`, digest
  `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643`,
  **27.3B Q4_K_M qwen35**, vision capable. The alias is not an 8B reader.
- Primary `translategemma:27b`, digest
  `21b597a243ec4fd2f05bc0ce8ef9ef174a89b09255c330d670852dfaaaa74f72`.
  These are daemon-reported identities, not independent weight authentication.
- Existing Paddle v6 medium/small/tiny static exports have all three required
  files and matching export identities; each file hash is retained. Current primary
  medium already made mistakes, so repeating it cannot be the sole selector/veto.
- Existing Manga OCR snapshot `aa6573bd10b0d446cbf622e29c3e084914df9741` is present.
  Its `pytorch_model.bin` hash is
  `c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb`.
  File existence/hash checks do not establish successful execution or accuracy.
- **Runtime mismatch:** `.venv` has PaddleOCR 3.4.0/PaddleX 3.4.2, whereas configured
  v6 requires 3.7.0/3.7.2. `.venv-paddle6` has the required versions plus Manga OCR
  0.1.16, transformers 4.57.6, torch 2.14.1+cpu. The viewer must be launched explicitly
  with that existing runtime for v6. No dependency installation/fallback is implied.
- Other daemon tags exist but are not configured readers or newly approved model
  families. Do not put them into a comparison simply because they are installed.

## Saved failure attribution

Evidence: v1 `language-v2/*-request.json`, raw `*-response.txt`, `results.json`,
follow-ups, and the OCR/translation separation report. Cases are reused diagnostic
IDs, never production routing rules.

| Cases | Failure and causal layer | Consequence |
| --- | --- | --- |
| C01 | Correct source control; reviewer manufactured contextual uncertainty | No candidate or selection gain. |
| C04 | Block drops visible separator; line reads only the easier column. English retains “Co-Co Deck” | Transcription/presentation coverage plus translation generation/selection failures; line success does not validate the whole turn. |
| C06 | Block reaches the 1,000-character cap with repeated kana; line misreads Latin | Candidate generation/transcription failure despite confident metadata. |
| C08 | Block imports adjacent credit text; line invents a different name | Extent contamination plus transcription/generation failure. |
| C09 | Core Latin recovered, punctuation omitted | Partial transcription, not exact source selection. |
| C10 | Colloquial stutter normalized in Japanese; English drops stutter | Transcription and target adequacy failures; fluency is not a selector. |
| C13 | Block invents time/status text and misses `の`; line changes vowel length/adds suffix | Extent/generation/transcription failure; old Paddle agreement cannot certify correctness. |
| C14 | Block imports title/speech; line invents trailing text and corrupts the label | Source extent and recognition failure, before translation. |
| C15–16 | Unresolved source extents/art produce confident text | Unsupported candidate generation; no whole-source gold, retain exclusions. |
| C03 grouping | Text keeps address separate; image emits empty groups | Relation generation/contract failure; no legitimate merged candidate to select. |
| C11 grouping | Text/image disagree about joining fragments | Relation uncertainty. Old diagnostic nodes also use historical `original_units` rather than the later selected Koala reading: not a clean current-source grouping crossover. |
| C02/C07 English | Changed wording is labelled `source_uncertain` | Contradictory metadata; candidate quality must be assessed separately from parser rejection. |
| C11 English | “Apparently” candidate loses concessive force; follow-up notices omission but retains primary | Translation generation plus selection failure, not OCR. |
| C05/C08/C09/C14 English | Control primaries retained | Preservation, not improvement. |
| C12/C13 English | Conversational/UI-scope uncertainty without a selected correction | Scope/target-review limitation; do not infer improvements from flags. |

## Frozen independent comparison (not an allocation)

Maximum **36 generation/reader calls**, plus **at most four conditional TG27 calls**
only if a separately validated changed source/membership is selected. No retries.
Existing assets only; explicit approval required before dispatch. The offline-only
`tools/freeze_language_comparison.py` now freezes 36 **provider-neutral requests**,
current grouping source revisions, originals/previews, crop transforms, exact
English conditions and separate assessment criteria under
`test-results/production-best-effort-v2/language-matrix/`. No production selector or
dispatch adapter is installed. Conditional TG27 requests cannot be frozen until
there is an actually validated changed source/membership; none is selected.

1. **Transcription/extent:** four category cases (proper name, confident wrong
   UI label, truncated outlined label, art/unreadable negative) × two available
   readers (Qwen and Manga OCR) × line/block = 16 calls. Retain raw boxes, derived
   extents and neighbor membership separately. Reuse the old Paddle evidence as a
   baseline, not a veto. Score exact visible glyphs, omitted/introduced characters,
   scope leaks and abstentions independently. Exclude unresolved whole-source CER
   instead of choosing the most grammatical answer. Crops remain original pixels;
   no evaluation reference text enters reader requests.
2. **Relations:** four frozen positive/false-merge contrasts × text/image = eight
   calls. Nodes use the actual current selected reading revision. Distinguish
   text-run, utterance, annotation, balloon, panel and speaker relations; unknown is
   allowed for each. Do not collapse all relations into a partition or use cleanable
   background as eligibility. Check complete membership and no duplicated parents.
3. **English:** six frozen categories (ordinary control, technical hesitation,
   stutter/negation, concessive turn, scope/address, alternative-menu negative) ×
   independent generation/primary-conditioned review = 12 calls. Same exact Japanese,
   context and generation controls. Candidate payloads ask for a single rendition,
   not a contradictory self-rating. Keep raw response and primary separately. Blind
   candidate-level bilingual scoring precedes any selector design; explicitly score
   tone, terminology, additions, omissions and scope. Never split slash menus in the
   renderer. A protocol improvement is not a language improvement.

### Exact scope and limits

- Transcription: reused C08/C13/C14/C16 × Qwen/Manga OCR × line/block. Line uses the
  first frozen member's raw polygon bounds +3px; block uses all frozen members +24px.
  Both retain original RGB pixels at 1:1 externally. They have **different target
  coverage**: line accuracy is not whole-block accuracy. C16 is an unresolved/art
  negative; a confident generated string is not ground truth. Raw polygons and
  derived extents are separate records. No reference glyphs are passed to readers.
- Relations: G01 address/current turn versus adjacent reply; G02 selected Koala
  continuation versus reply; G03 separate similar interjections; G04 name/epithet
  versus dialogue. Every pair is assessed independently for all six relations,
  with `unknown` permitted. Text/image variants have identical current nodes;
  image adds one native union crop +12px. Historical `original_units` are not used
  to override the current selected Koala reading.
- English: C01 ordinary, C04 hesitation/technical term, C10 stutter/negation, C11
  concession, C03 address/scope, and E06 an actual saved three-alternative primary
  from saved-example row 32. The first five use the prior exact-source diagnostic
  inputs, explicitly provisional. E06 uses the current selected source/result.
  The independent arm never receives the primary; the conditioned arm does.
  Neither arm returns a retain/replace verdict. A nullable rendition plus specific
  ambiguities avoids the previous contradictory status/changed-text contract.
- Proposed resource ceiling: **28 Qwen calls <=90s + 8 Manga reader calls <=60s =
  3,000 call-seconds**, sequential with no retries; conditional <=4 TG27 calls
  <=60s each would add at most 240s. These are proposed bounds, **not approved**.
  Manga preprocessing remains the existing snapshot processor. The current wrapper
  returns the package's text result only; record normalization and any unavailable
  raw-decoder evidence explicitly instead of treating that result as exact pixels.

Offline verification checked 61 frozen hashes, all 20 image-bearing request slots,
current grouping revision bindings and each matched English condition. There were
**zero language calls**. Before execution: independently label the evaluation
subset, review the request-only contract, implement/test the bounded transport
adapter, and obtain explicit allocation approval. Independent bilingual assessment
and untouched-title requirements still block linguistic release claims.
