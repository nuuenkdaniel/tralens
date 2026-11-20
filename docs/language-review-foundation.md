# Independent language-review foundation (experimental)

## Scope frozen before fresh inference

Keep TG27 primary and all existing cleanup, replacement and patch/caption behavior.
The first intervention is **routing plus unselected source-recovery candidates**.
It is not grouping, contextual English review, automatic source selection, or a
viewer feature. Useful candidate English is not a displayed-page improvement.

`src/tralens/language_review.py` routes on original OCR, source-bound primary
translations and pixels, without consulting rendering. Signals are generic
investigation reasons, not correction rules. One deterministic per-page audit
slot supplements three ranked risk slots. All unsupported/budget-excluded units
remain visible in the plan. Four requests per page is intentionally small and
does not guarantee coverage of every motivating failure.

`src/tralens/ocr/source_review.py` gives the existing Qwen visual route an unmarked
original view and a distinct colored locator view. It supplies neither old OCR
wording nor English. The response binds each original main line to exact visible
text, uncertainty and a bounded proposed extent. It cannot change annotations,
speaker identity, geometry or ownership. Neighbor-center conflicts, clipping and
uncertainty block corroborated-candidate translation. These guards are conservative
and not a proof of isolation: unknown undetected neighboring text remains a risk.

Unchanged clear rereads are retained as **uncertain corroboration, not truth** and
reuse valid English. Changed clear proposals need exact normalized agreement from
the existing local Paddle crop reader, seeing only original pixels. Agreement is
not ground truth. New source means a new TG27 request preserving the original
members/roles/readings and containing candidate provenance; only its exact selected
Japanese enters the native translator prompt. Every English candidate is unselected.
Original results remain untouched on optional failure, disagreement or budget exhaustion.

## Frozen pilot and stopping rule

Evidence root: `test-results/language-review-foundation-v1/` (ignored research).
Seven frozen original/preview/native pages; five receive inference. Three are
reused motivating/protected pages; two additional pages (`ubunchu07_0002`,
`ubunchu08_0011`) were reserved from queue tuning, but have reused OCR/cleanup
history. Neither is globally untouched gold. References are provisional agent
pixel inspections, never human gold or production inputs.

Ceilings across the **whole pilot**, not each page: **20 Qwen source calls,
8 independent crop-reader calls, 8 changed-source TG27 calls**. No retry on failed
experiment runs. The serial evaluator subtracts actual persisted counts before
each next page and keeps failures/uncertain candidates. No page OCR, grouped or
contextual English-review calls. Initial request crops and provisional notes are
frozen before inspecting outputs. No prompt/routing tuning against this pilot.

Evaluate routing, proposed source correctness/extent, corroboration and candidate
English separately. Record technical terms, omissions, additions, alternatives
and tone separately where applicable; do not infer quality from fluent English.
Art/no-text, partial lines, adjacent dialogue and protected successful dialogue
are retained as negatives/controls. Inspect originals and baseline full pages at
reading size; because nothing is selected, candidate page output is explicitly
absent, not a newly rendered improvement. Preserve successful prior clean edits.

Stop rather than integrate a marginal/ineffective route. Grouping and contextual
review require separate frozen trials. Viewer integration remains conditional on
demonstrated utility and a safe selection design.

## Reproduction

Plan-only single-page CLI (no model/reader creation):

```sh
.venv/bin/python tools/language_review.py --preview PREVIEW --image ORIGINAL --output FRESH_DIRECTORY
```

Freeze requests, then execute once in fresh directories:

```sh
.venv/bin/python tools/evaluate_language_review.py --cases test-results/language-review-foundation-v1/cases.json --output FRESH_REQUEST_DIRECTORY --model-root /home/Danuu/.cache/tralens/paddle-v6-models
.venv-paddle6/bin/python tools/evaluate_language_review.py --cases test-results/language-review-foundation-v1/cases.json --output FRESH_RUN_DIRECTORY --model-root /home/Danuu/.cache/tralens/paddle-v6-models --run-local-model
```

Only existing explicitly configured `qwen3.8:latest`, `translategemma:27b`, the
Paddle v6 medium assets and installed environments are used. No acquisition.
Startup requires no running viewer jobs or resident Ollama model. Subsequent
checks refuse viewer work/unrelated model residency, including before release.
This is **not a shared-server lock**: Ollama cannot identify another client using
the same tag; check other work before starting and do not claim race-free exclusion.
Release is limited to explicitly used experiment tags, never wildcard eviction.

Tests: `.venv/bin/python -m unittest tests.test_language_review -q`.
