# Independent reading and source-context diagnostics

These local experiments extend the image-translation investigation. They retain
every request and failed attempt, use no expected answers in model inputs, and
have not supplied selected production translations. Improved examples coexist
with substantial source, meaning and region-binding errors.

## Independent Gemma4 reading

`gemma4-reading-trial-01/` and `gemma4-reading-trial-02/` under ignored
`test-results/production-best-effort-v8/` cover 24 difficult source regions with
native, aspect-preserving square and native-plus-scene presentations, plus three
synthetic controls. The installed `gemma4:26b` digest is
`001e5dafc3c77684c2307ebc6ab8e336e10c9b18eca52acf547d72fc83c3ca8c`, Q4_K_M.
Greedy requests use structured text/uncertainty/clipping fields, without prior OCR.

Of 75 requests, 73 return parseable readings. Two paired-image requests repeat
escaped newlines until the 512-token limit. Their `done:true, done_reason:length`
responses are retained as failures. The first run releases the model; continuation
records that release and skips every previous attempt, including the failure.
Both runs retain acknowledged unloads and empty runtime snapshots.

All 24 source/readings cards in `gemma4-reading-review-01/` were inspected:

- Useful readings include `週刊`, `まなざし`, the retrospective question and
  much of the encouragement question. Native/square name readings give “Masato.”
- The reader changes `ソース` to `リース`, duplicates the strokes of `いいの`
  into extra characters, misreads clapping, drops characters and reverses columns.
- Scene-assisted reading imports neighboring captions, expands the clipped title
  and name regions, and sometimes invents trailing characters.
- False uncertainty/clipping flags occur beside wrong readings; agreement across
  presentations is not source truth. Native and square agreement on an incorrect
  word is a concrete counterexample to a self-consistency-only selection policy.

## TG27 marked page-text controls

`tg27-page-text-trial-01/` uses the unchanged official text prompt and exact saved
selected readings. Each source has an XML unit ID. There are 109 isolated tagged
controls and eight full-page requests: each of the four pages in original proposal
array order and its reversal. Neither order is claimed to be comic reading order.
No OCR changes, image inputs, supplied translations or speaker labels are used.

All 117 requests complete. The eight page outputs preserve all IDs syntactically;
26 isolated controls lose or damage the XML wrapper. Both raw responses and parse
failures are retained, rather than silently extracting an apparently useful string.

Inspected gains include clapping, reassurance, unexpanded APT and contextual “me”
in the encouragement question. Important counterexamples remain:

- The forward page output attaches the open-source embarrassment dialogue to the
  preceding, different speaker's incomplete sentence, changing its first-person
  meaning to “all the embarrassing misunderstandings you had.” Correct XML IDs
  have not preserved the semantic boundary.
- The dawn-related continuation still becomes “We've come this far” / “We're
  almost there,” instead of referring back to the dawn.
- Order changes shift meanings and phrasing throughout the page. Only 18 of 109
  extracted strings are exactly identical between the two orderings; this is a
  sensitivity count, not an error count.
- Markup alone changes some interpretations: the isolated tagged encouragement
  control already says “encourage me,” and the reassurance control says “That's
  fine.” Those gains cannot be attributed solely to page context.

## Exact-target generalist controls

`source-bound-context-trial-01/` compares Gemma4 isolated text, page-text context,
and page text plus target-crop/original-page images on the same 24 selected regions
(72 completed, parseable requests). The prompt explicitly forbids OCR correction,
missing-text completion and importing neighbors; source-conflict and uncertainty
fields are retained with each exact target ID.

Reassurance, clapping and the encouragement question can improve. However:

- Page text turns the weekly-publication region into the neighboring “ASCII.”
- The arrow detection becomes the neighboring “Tone-deaf.”
- Page images silently change the selected ACPI/off source, “Masito” and the
  clipped APT title while reporting `source_conflict:false`.
- The gaze caption becomes “ambitious” or “Managashi”; the dawn continuation
  retains the wrong implied subject.

These are concrete failures of source/target ownership despite valid JSON and
explicit instructions. Model self-reports cannot certify unchanged source meaning.
The trials therefore support further separation of source recovery, context and
target binding, not an automatic model replacement or page-context rollout.

## Strong-reading corroboration pilot

`strong-crop-corroboration-01/` derives its five eligible regions from the retained
crop-reading evidence: known orientation, unchanged raw selection, raw scores at
least .92, and a changed three-view Hayai reading that would otherwise qualify.
The exact original native/isolated/padded pixels are revalidated. Gemma4 reads each
view independently, without seeing either source string (15 completed requests).

Three proposed corrections agree across both readers and all presentations:
`っぼい` → `っぽい`, `いたたきます` → `いただきます`, and `まなさし` →
`まなざし`. The five original crops were inspected. The first two still have
clipping flags on some Gemma4 views; only the gaze caption has no flags across all
three. The wrong Hayai readings of the cow title and clipped form heading disagree
with Gemma4 and remain rejected. This is a promising, small source-recovery pilot,
not evidence to remove strong-reading protection generally. No source was selected
or translation regenerated from these corroboration results.

## Expanded corroboration and framing controls

`strong-crop-expansion-01/` applies the same routing rules to 19 additional saved
page paths: 197 regions and 591 completed Hayai readings. Twenty-one changed,
consistent candidates receive 63 blind Gemma4 requests. Sixty-one parse; two
newline-repetition truncations remain failed attempts with their raw responses.

`square-corroboration-trial-01/` then presents all 26 pilot/expansion cases using
the same aspect-preserving 896-square framing, independently for each of the
three retained views. All 78 requests parse. Six cases agree with Hayai in every
view without uncertainty or clipping flags:

- The posting greeting and gaze-caption dakuten corrections from the pilot.
- `コニヒューターの再起重力` → `コンピューターの再起動` and
  `残時間約20分` → `残り時間約20分` on `ubunchu06_0002`.
- `Linu×` → `Linux` and `うぶんちゆ` → `うぶんちゅ` within longer sentences
  on `ubunchu08_0003`.

All 26 original/readings cards in `square-corroboration-review-01/` were inspected.
Framing removes some clipping flags, but also changes plausible readings into
invented text. The clipped form heading produces three different confident
expansions. The cow title still disagrees with Hayai, and the magic sentence
acquires extra characters. No framing policy establishes complete text ownership.

The 23 development paths represent **22 unique source images**: `saved-example`
and `ubunchu01_06` are pixel-identical. The 26 corroboration cases similarly
contain 25 distinct three-view pixel sets. These repeats are not independent
validation. `corroboration-audit-01.json` checks 1,040 artifacts, exact original
views, request/image/result bindings, retained truncations and acknowledged model
releases. These are integrity checks, not an accuracy score. Strong-source and
Latin-only selection protections remain in place; none of these diagnostic
corrections has been selected or rendered.

This remains development evidence on reused pages, without independent bilingual
approval or untouched end-to-end validation. Full fidelity is still unmet.
