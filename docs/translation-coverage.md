# Coverage-first translation

The default `translation_policy='coverage-first'` attempts exact, source-bound
OCR text even when balloon ownership, crop isolation or reading orientation is
uncertain. `--translation-policy conservative` retains the original eligibility
rules. Translation and rendering are separate policies: uncertain replacements
require `--render-policy best-effort`; conservative rendering leaves them intact.

## What changes

The demonstrated development-viewer page had 29 proposed text units, but only
eight were eligible. The other 21 all had unknown ownership; eight also had crop
conflicts and two had unknown orientation. Increasing the call budget alone could
not help. All 21 now qualify for an explicitly uncertain text-region request,
including the LiveCD dialogue, double-click instructions and separate `ねえ`.
Eligibility is not a readability or translation-quality assessment.

Original `Utterance` snapshots, raw strings, reading order, selected revisions,
blocking reasons, alternative readings and upload bytes remain unchanged. The
schema-4 preview stores a separately derived `translation_input`,
`input_kind='coverage_text_region'`, and provenance binding its original unit and
selected raw reading. It does not silently clear the original OCR blocks, split
units, promote review-only fragments or invent missing wording.

Only these original blocks may be waived for the derived **text-only** request:
unknown/mixed region ownership, unknown orientation, unresolved crop isolation
and unsafe crop. No crop or image is sent to the translator. Rejected/unknown raw
statuses, unresolved detector/recognizer associations, ambiguous annotation
roles, unknown structural faults, blank/oversized text and malformed evidence
remain blocked. Existing nonblocking evidence that an annotation relationship
was rejected is informational, not a new annotation role or permission to merge.

The native TranslateGemma prompt is unchanged and still contains the exact raw
selected characters. Unsupported style/context warnings remain recorded in
execution metadata. Generalist prompts explicitly identify uncertain OCR regions.
Every successful coverage result requires review; models may still abstain on
unrecoverable source. Neither request type certifies complete sentences, full
balloons, speakers, reading order or linguistic correctness.

## Scheduling, budgets and replacement

Strict inputs run first, then coverage inputs, then optional review-only
fragments. Automatic coverage scheduling weakly favors grouped text, Kana/Han
presence and longer text; these are not language/readability filters. Explicit
selected-ID order is preserved within each kind. A coverage input is not also
scheduled as a fragment or visual alternate. All translation calls, including
failures and interrupted dispatches, share one 0–64 budget. CLI default remains
two calls; the development viewer defaults to 64 for new uploads. Older saved
runs retain their settings: choose **Coverage first**, increase the budget and
use **Retry pipeline** to create a fresh run. No startup/refresh inference occurs.

Automatic best-effort rendering re-derives coverage permission and verifies the
entire original/derived/request/provenance/result binding. Manual preparation,
manual composition and strict binding never accept these inputs. Strict outputs,
including their fallback layouts, reserve support before uncertain regions.
Coverage then tries source-local flat cleanup, native hints and footprint/overlay
fallbacks. Glyph support, exact target text, neighbor protection, pixel audits,
source bindings and resource limits are not waived. All uncertain replacements
are marked best-effort even if their local background is flat.

Counts separate strict, coverage and total translations and replacements; a
strict coverage fraction never divides total replacements by a strict-only
denominator. `full_page_translation_verified` and export approval remain false.
More attempts do not guarantee clean artwork, readable typography, complete OCR
or complete replacement.

## Dense-page storage

Retained per-unit foreground/erase/layout/obstacle masks use exact one-bit
packing; content masks use Pillow mode 1. Computation unpacks one unit at a time,
and published content PNGs are expanded back to L, preserving prior artifacts.
The automatic route permits 32 slots and 128M aggregate unit-page mask pixels
(four packed masks use at most 64MB, plus content masks and other bounded state).
It reports measured retained mask bytes. Manual-route limits are unchanged.
Page limits remain 4M pixels/4096 per side, raster-work remains 512M, and artifact
bytes remain 128MiB; the artifact-count ceiling is 512. Resource exhaustion still
produces explicit skips rather than truncation or weakened integrity checks.

Offline regressions cover native prompts, original/source immutability, hard
blocks, review and binding drift, conservative/fragment separation, scheduling,
strict placement priority, exact mask roundtrips and a 32-unit dense page. The
optional saved-page regression freezes the demonstrated 8+21 eligibility result.
The offline translation-review tool also validates schema 4, including original
and derived bindings, coverage call counts and scheduling, without opening model
sidecars or promoting OCR truth.

## Local end-to-end observation

`test-results/translation-coverage-live-v1/` ran fresh local Paddle v6 OCR and
native TranslateGemma 27B on the unchanged saved upload (SHA256
`bc5a90b6952e45f11f9d6de80195cef48d2c0dee211e6071521001dd0848abe3`).
No model/font download or manual masks were used. It produced **29 translations
and 29 replacements** from 29 calls: eight strict and 21 coverage inputs. There
were no budget skips. All outside-cleanup/outside-alpha/protected change counters
were zero. **23 replacements were best-effort**, including two of the original
strict results; integrity locality is not proof of artistic preservation.

Visual inspection found severe reconstruction damage in several broad/uncertain
OCR regions, and bad source wording still causes bad translations: the supposed
double-click instructions have garbled OCR and remain semantically wrong. The
new route fixes eligibility, not OCR accuracy or arbitrary artwork restoration.
LiveCD dialogue and the separately detected `ねえ` now receive replacements.
The output is deliberately not described as translation-quality success.

Retained live preview SHA256:
`ced1ce13cbc56c5d14c99966199b1977f138dde72c7349064694e7d7e0360a05`.
Composite SHA256:
`f2518edf6926beb9d58df451af5cdac0caba5a76ce90d6f86d6e4b11da0b1770`.
The evidence and images remain local/ignored. A separate rerender of the original
strict-only preview retained all **129 previous PNGs byte-for-byte**, while its
four retained masks fell from 65,190,720 unpacked bytes to 8,148,864 packed bytes.
The final strict-first raster-budget scheduler rerendered that live preview with
all **444 PNGs unchanged**, without additional model calls.

Verification: **797 tests pass** in both existing environments. The isolated
source-only suite passes with 38 explicit optional-local-evidence skips.
Headless Firefox checks cover upload, all five stages, uncertainty badges,
comparison/zoom/highlighting, refresh and retry with retained coverage settings;
they use injected OCR/provider fixtures, not simulated stage artifacts.
