# Experiment: cropped localization and contextual single-rendition translation

## Scope frozen before implementation

The previous layout retry did not materially improve the user's page. The next
acceptance case is `ubunchu01_06`: recover the large Synaptic and network dialogue
without damage or source remnants, at >=16px, and replace printed translation
menus with one coherent rendition. A partial result must be reported as partial,
not as meeting this milestone. Neither more edits nor more passing tests suffice.

Keep rendering/translation experiments independent. The rejected percentage-based
ownership hypothesis in `closed-fringe-ownership-experiment.md` stays rejected.
Do not loosen those ratios, gray-interior protection or coarse-footprint erasure.

## Localization pilot

Run the existing local Paddle v6 medium detector/recognizer on bounded original
crops derived from source polygons, with fixed 24px context, requesting its public
word-box output. Include the two missing dialogue blocks and both conflicting
large sound-effect detections. Preserve original detections, crop transforms,
model identities, raw model outputs and hashes. Re-detection/recognition agreement
is evidence of repeatability, not independent linguistic accuracy. Word/character
boxes are localization hypotheses, never rectangular erase masks.

If crop predictions do not resolve disputed ownership, record that failure rather
than substitute a hand-designed mask or silently drop a conflicting detection.
Any subsequent cleanup route must retain whole original strokes and explain flat
collars; original art, other sources and previously committed edits remain protected.

## Translation experiment

Use the existing `qwen3.8:latest` with the existing schema-capable generalist route.
Native TranslateGemma remains isolated and must reject contextual conditioning;
there is no implicit model switch. Explicit nearby-text context is a separate set
of source-bound units, with unknown inter-unit reading order and speaker identity.
Never concatenate context into the target's OCR or translate context as a target.

Design a bounded spatial-neighbor selector using original proposal bounds: up to
four nearby eligible units, ranked geometrically, with no page/order/name special
cases and a fixed text budget. Blocked contexts stay blocked unless the existing
coverage policy independently permits their exact text. Context is uncertain and
may cross panels; do not call proximity narrative evidence. One model-generated
rendition is requested without shortening to fit the balloon; preserve ambiguity
without printing a menu. Old translations remain in their untouched output runs.

Pilot selection before inspecting model output: the two large dialogue units,
the short lower-right reaction, and up to three other translated dialogue units
on the same page. Compare exact source/context requests and outputs. Then evaluate
all units on that page and a compact set of the existing development/evaluation
pages, explicitly labelling the latter as reused evidence. No independent accuracy
claim is possible without checked references.

## Gates

- No source corrections, manual masks, manually selected/shortened targets, new
  dependencies/fonts/models/providers, upstream implementation copying or push.
- Portable tests cover stale/cross-page bindings, exact target/context partition,
  blocked/blank contexts, bounded selection, untrusted source text, capability
  rejection, default behavior, resource/call budgets and rendering bindings.
- Inspect every changed full page and crop. Record awkward meaning, false joins,
  source remnants, border/art damage and new omissions, not just positive crops.
- Keep defaults unchanged and successful old cleanup/lettering intact in the
  cleanup-only comparison. Translation-only comparisons necessarily change text
  tiles, but must not obtain broader erase permission simply from new wording.
- Broader testing starts only after a visible example improvement. Failed
  localization/translation pilots are not deployment-ready capabilities.

## Provenance

Design against Tralens's own source and saved originals. Context7 was queried only
for the public PaddleOCR `predict(..., return_word_box=True)` API; returned internal
examples are not used to implement word-box decoding. Existing model-weight/font
rights restrictions remain. Prior research exposure is not a clean-room assurance.

## Localization pilot observation and next bounded hypothesis

`test-results/crop-localization-pilot-v2/` repeats all six original dialogue lines
exactly with the existing Paddle model. Word boxes are coarse (some kana remain
word-sized), not precise character masks. Both cropped sound-effect predictions
miss the actual stylized effect: their absence must NOT authorize its erasure.
Simply replacing the two dialogue polygon sets with the cropped detections still
fails native cleanup, so re-detection alone is not an accepted fix.

Test an explicit alternative **whole-component localization** hypothesis:
original and cropped line polygons bound extent; word regions seed complete
original RGB-contrast components; closed original 4/8-consistent usable cores
separate them from borders. Require a unique original member vote for every
selected whole component, beating every nonmember separately. Reject a competing
source centered inside the same enclosure, ties, missing source lines, unexpected
cropped text inside the enclosure, and source/geometry/prediction drift. Preserve
all unselected original ink as obstacles. If a coarse foreign rectangle conflicts,
only this independently corroborated component ownership plus the closed boundary
can replace its interior rectangle protection; its actual unselected ink remains
protected. This is a new model-assisted hypothesis, NOT a relaxation of the
rejected percentage-only design, and is not semantic certification.

Retain native light-neutral flat fill, whole strokes, 1px erase collar, 2px known
background clearance, 2-level flatness tolerance, and the requested font floor.
Do not alter `segmentation.py`'s gray-interior guard. In the new localized route,
add a separate single-ink audit: stable different-color interiors reject. A tiny
varying terminal fringe may pass only with no own interior and strictly increasing
original contrast paths to the measured ink core. Test this radiometric hypothesis
against known rendered thin strokes and gray patches, including attached gray art;
report that a small deliberately graded mark could still resemble antialiasing.
No learned semantic-mask inference is claimed.

First try these masks without changing saved English targets. Only after visual
inspection and negative controls should this become an opt-in integrated route;
the original native and semantic routes/defaults must remain unchanged.

The graded-terminal extension was rejected during development: it still rejects
the network stroke because quantized gray values form a plateau, and it also
rejects one of 18 known rendered thin-text controls. Do not relax its path rule to
fit this page. The retired implementation and controls are retained under the
localization pilot artifacts. The integrated localized route instead applies
the **unchanged existing single-fill/2px-transition/<=4px exception policy**.
Consequently the network balloon remains blocked; recovering it is still an unmet
part of the milestone. This is a deliberate negative, not a completed fix.

## Translation development result and narrower comparison

The full-page generalist replacement is rejected as a quality default. Of 39
previously translated inputs it produced 26 previews, 12 abstentions and one
failed output. It removes menus but also misrenders terminology (for example,
"Plugins or code?") and loses a previously successful dialogue result. All
previous requests/results remain retained; do not conceal these regressions by
showing only the short positive reaction crop.

A narrower **format-routed candidate comparison**, not a linguistic-selection
guarantee, retains all non-menu primary records unchanged. Eligibility is an old
English target containing literal ` / ` where the exact selected source has no
`/`. A new candidate must be a valid, nonempty source-bound result without that
menu separator. Do not split, trim or hand-select a phrase from old alternatives.
The first page selects indices 5, 28, 32, 38 by this rule; one is not renderable.
The other three improve visual clutter without changing their cleanup masks.

Before further inference, freeze additional menu-candidate inputs from
`ubunchu04_0005`, `ubunchu06_0002`, and `ubunchu08_0007`, up to eight calls total.
These are known/reused pages, not fresh held-out evidence. Keep failed, awkward
and unresolved outcomes in the report. No automatic production promotion is
implied by passing the format check, and the main pipeline still defaults to its
previous isolated specialist route.

## Implemented interfaces and results

- `ocr/source_localization.py` uses the already-open, explicitly configured local
  Paddle engine, with at most 12 crop calls / four million crop pixels. It retains
  raw prediction association fields, original crop PNGs, hashes and per-unit skips.
  Failed crop inference does not remove primary translation availability.
- `rendering/localized_strokes.py` independently checks those source bindings and
  hypotheses, using whole original residual components and original flat collars.
  A successful crop prediction is not a rectangle-erasure permission. No existing
  native/semantic threshold changed. All unselected original contrast remains an
  obstacle. The route runs only after successful old operations have been reserved.
- Main CLI `--source-localization` records the evidence; with `--render`, it also
  enables the experimental cleanup retry. It requires confidence-gated best-effort
  rendering. Offline rendering consumes existing evidence with
  `--cleanup-policy localized`; it never starts an OCR model.
- `translation/context.py` makes exact target/context partitions with unknown
  reading order and speaker identity, at most four neighbors / 1,024 context
  characters. A provisional raw-score routing floor is not an accuracy probability.
  `--translation-context nearby --model qwen3.8:latest --profile generalist` is an
  explicit experiment, not the default or a recommendation to replace all targets.
  Native TranslateGemma refuses it; visual-alternate review is a separate experiment.
- Render-time validators and the normal translation-review tool rederive contextual
  requests rather than trusting saved context/style strings. Derived comparison
  artifacts may explicitly retain isolated primaries beside contextual candidates;
  every primary/candidate request keeps its own revision. The **format-based menu
  selection is still an evaluation harness**, not a new automatic main-pipeline
  second pass. These mixed artifacts are renderer comparisons, not fresh workflow
  execution journals; their explicit experiment provenance distinguishes cached
  candidates and retained prior-run evidence.

| Frozen cleanup-only comparison | Before | After |
| --- | ---: | ---: |
| Six development pages | 58 | 59 |
| Fifteen reused evaluation pages | 123 | 123 |

`test-results/localized-dialogue-corpus-v1/` contains all 21 outcomes and the actual
frozen source snapshot. Its source-set digest is
`bb10877a26a73b571df83de416e10bbf38756b70d8eda91e24616c4ee691eeba`.
The final source hashes match it exactly. There were **237 crop inference calls**;
this is not a throughput improvement or evidence of generalization. The sole new
cleanup is the Synaptic balloon on the motivating development page, at **22px**.
The other 20 composites remain byte-identical. The network unit reaches the new
route and rejects at `localized_gray_interior`; the old guard was not weakened.

The source mask contains 41 whole original RGB-contrast components, 3,860 foreground
pixels and a 7,051-pixel one-pixel-dilated erase mask. Independent component/locality
checks and visual original/before/after inspection found no changed unselected
contrast, balloon border, screen lettering or sound effect below it. These checks
do not certify that arbitrary native components elsewhere are semantically text.

`test-results/menu-candidate-comparison-v1/` retains all non-menu primary records
on the example. Three previously printed menus become `Ohh`, `Whoa.`, and
`Right, right?`; the fourth candidate remains text-only behind the existing gate.
Only those three prior text PNGs change: **every prior foreground/erase PNG stays
byte-identical**, and all previous edited source IDs remain. The new Synaptic
target itself remains the exact old target, not a hand-shortened replacement.
This is a presentation gain, not independently established linguistic accuracy.

The additional frozen menu pilot found no eligible menus on `ubunchu04_0005` or
`ubunchu06_0002`; they stay in the report. It made three calls on the reused
`ubunchu08_0007`. The newly printed `Happily ever after.` is less cluttered than
the old three alternatives, at the same cleanup permission. Two other outputs
remain unrendered. They include the uncertain fragment `It's a matter of...`,
not a verified complete translation. No old edits are lost and no old masks change.

The full generalist comparison is retained as a **negative** under
`test-results/context-single-rendition-page-v1/`: six identical pilot requests
were reused and 33 additional calls made, rather than pretending all 39 were fresh.
It loses prior edit 35 and changes other meanings. Do not infer a linguistic gain
from fewer slash separators or a similar replacement total.

## Verification and decision

- **914 tests pass in each existing environment**; source-only passes with 39
  optional local-evidence skips. Thirty portable tests were added.
- All 444 default coverage and 129 default strict regression PNGs remain identical.
- `test-results/localized-dialogue-independent-audit-v1/` independently checks 23
  cleanup/menu comparison outputs: source hashes, actual alpha composition,
  outside-mask locality, retained old masks/text, full native components and the
  exact one-pixel erase collar. No linguistic score or semantic certification.
- Original/before/after full pages and changed balloon crops were visually inspected
  for the example and the additional menu page. Other cleanup outputs are unchanged.
- User-owned dependency/ignore changes and staged documentation deletions remain
  untouched. No new models, fonts, dependencies, remote providers, license changes,
  publication or push. CTD remains unused.

**Partial acceptance only:** one substantial missing dialogue balloon and less
cluttered candidate lettering are visible improvements on the example. Recovering
both large balloons, reliable terminology, broader OCR recovery, precise masks on
art/textures and independent quality validation are still unmet. Keep all new
capabilities opt-in; do not promote the full-model switch or the menu heuristic as
a general production quality solution.

Private read-only gallery: `http://100.121.133.88:7863/`. The default view shows the
focused comparison; cleanup-only, all 21 outcomes, full-size originals, close-ups,
and the rejected full-model switch are also available. Only allowlisted images
are served, not repository files or execution logs.

Example fresh CLI experiment with the **existing native translation model**:

```sh
.venv-paddle6/bin/python src/main.py \
  --image images/quality50-v1/ubunchu01_06.png \
  --output test-results/my-fresh-localization-run \
  --device cpu --model-root ~/.cache/tralens/paddle-v6-models \
  --run-local-ocr --run-local-model --endpoint http://100.121.133.88:11434 \
  --max-attempts 64 --render \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --replacement-policy confidence-gated --layout-policy enclosed-retry \
  --source-localization
```

This command runs new inference; the gallery's frozen-OCR comparisons are not a
claim that this exact fresh end-to-end command was benchmarked. Output must be
fresh. The development web viewer does not yet expose these new options.
