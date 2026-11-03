# Upstream approaches and licensing review

Reviewed 2026-11-03. This is an engineering/licensing inventory, not legal advice
or commercial/distribution clearance. No upstream implementation was copied into
Tralens, no new dependencies or weights were acquired, and no Tralens license was
chosen. The existing development viewer and pipeline are unchanged by this review.

## Implementation findings

### Manga/Image Translator

Repository: <https://github.com/zyddnys/manga-image-translator>
Inspected revision: `d6de6ab62a7be6afc0d07a3184149cdefbcd32f2`.
Repository license: GPLv3; check the precise grant and any per-file exceptions
before copying or porting individual components.

- [Text-line merging](https://github.com/zyddnys/manga-image-translator/blob/d6de6ab62a7be6afc0d07a3184149cdefbcd32f2/manga_translator/textline_merge/__init__.py):
  constructs a graph of geometrically compatible lines, splits connected groups,
  derives a direction and orders lines within each resulting text block. It does
  not simply concatenate every nearby OCR box.
- [Mask refinement](https://github.com/zyddnys/manga-image-translator/blob/d6de6ab62a7be6afc0d07a3184149cdefbcd32f2/manga_translator/mask_refinement/__init__.py):
  takes the detector's mask plus recognized text-line geometry, refines the mask
  through `complete_mask`, and maps it back to page coordinates. A separate fill
  method and optional bubble filtering exist; these should not all be described
  as precise stroke segmentation or artwork-safe defaults.
- [English rendering](https://github.com/zyddnys/manga-image-translator/blob/d6de6ab62a7be6afc0d07a3184149cdefbcd32f2/manga_translator/rendering/text_render_eng.py):
  extracts balloon regions, measures word widths and lays out lines against a
  balloon mask rather than treating the OCR rectangle as the final lettering box.

### BallonsTranslator

Repository: <https://github.com/dmMaze/BallonsTranslator>
Inspected revision: `7fe9979f1233eb8edb5f431c060d2493efbc995b`.
Repository license: GPLv3, with the same per-component review requirement.

- [Text-block masks](https://github.com/dmMaze/BallonsTranslator/blob/7fe9979f1233eb8edb5f431c060d2493efbc995b/ballontranslator/utils/textblock_mask.py):
  provides edge/flood-fill and connected-component processing to distinguish
  lettering, balloon background and surrounding regions. Stroke-width checks and
  foreground/background estimates are separate pieces of evidence.
- [Inpainting base](https://github.com/dmMaze/BallonsTranslator/blob/7fe9979f1233eb8edb5f431c060d2493efbc995b/ballontranslator/modules/inpaint/base.py):
  evaluates background variation to decide whether flat fill suffices; otherwise
  processes enlarged local block crops with an inpainter. Some flat-fill branches
  fill extracted balloon masks, so this is not a universal promise of stroke-only
  edits. Its mask/editing UI also permits human repair; a showcase is not proof
  of fully unattended quality.

### Koharu

Repository: <https://github.com/koharu-rs/koharu>
Inspected revision: `cd31ad2c7016b64d0ff0400a09e86cd13fb5df13`.
Repository code is offered under MIT **or** Apache-2.0, at the recipient's option.
Check individual files, dependencies and attribution provenance before reuse.

- [Inpainting stage](https://github.com/koharu-rs/koharu/blob/cd31ad2c7016b64d0ff0400a09e86cd13fb5df13/crates/koharu-pipeline/src/stages/inpainting.rs):
  retains separate cleanup/text masks and bubble polygons. Uniform-region fill
  targets the intersection of the pending mask, text mask and bubble polygon.
  Remaining connected mask components are grouped into bounded contextual tiles
  and generated output is composited back using mask/component support.
  A comment cites BallonsTranslator's uniform-background approach. That citation
  is not, by itself, proof of copied expression or proof of legal clearance.
- [Bubble-aware layout](https://github.com/koharu-rs/koharu/blob/cd31ad2c7016b64d0ff0400a09e86cd13fb5df13/crates/koharu-renderer/src/bubble.rs):
  uses explicit geometry/relations, preserves authored rotation and partitions
  joined balloons into separate text-flow regions. Its fallback partitioning is
  not equivalent to image-proven ownership; Tralens must retain that distinction.

### Manga OCR

Repository: <https://github.com/kha-white/manga-ocr>
Inspected revision: `c333b5d36e88d539d6b040b4c4cf90ad5ecd4f69`.
Code license: Apache-2.0.

The recognizer supports multiline Japanese blocks, including vertical text and
furigana. Its README recommends smaller crops when long passages fail, but
explicitly warns that its generative decoder can invent plausible text even
when no text exists. A reread must remain a separately bound candidate, not
overwrite the raw Paddle reading or become verified truth through fluency.

## Application to Tralens

The current 29-unit trial increased replacement coverage but produced damaged
artwork and incorrect translations of incorrect OCR. The highest-value changes
to test are:

1. Compare source-bound block and line rereads, including neighbor/isolation and
   annotation-role evidence. Preserve original readings and every alternative.
2. Keep text-stroke masks, balloon geometry and lettering layout as distinct
   contracts. Never infer permission to erase a whole OCR polygon merely because
   that unit received a translation.
3. Route measured flat backgrounds to fill, genuine texture/artwork to a bounded
   masked reconstruction attempt, and keep failed attempts visible. No required
   manual masks or approvals, and no silent loss of translation coverage.
4. Fit exact target text within balloon-aware flow regions; do not hide poor OCR
   or layout behind emergency typography or a higher replacement count.
5. Benchmark source remnants and artwork damage as well as coverage. Existing
   rejected LaMa and morphological-stroke experiments remain rejected unless
   new, controlled evidence justifies a different approach.

Tralens already has separate OCR/translation/rendering stages, strict native-mask
checks and an opt-in CTD path. Its uncertain-coverage fallback currently tries
source-local flat cleanup, native grayscale hints and broad footprint repair;
the explicit semantic-mask route used for strict inputs is not currently in
that coverage fallback's route list. That routing gap is a concrete experiment
target, not a reason to enable an uncleared checkpoint or promise it will help.

## What license would Tralens need?

### Studying approaches, without copying protected implementation

Generally, learning techniques and independently implementing functionality does
not require adopting another project's copyright license. Patents, contractual
restrictions and jurisdiction-specific issues can still matter. Renaming code,
translating it from Python to Rust, or asking AI to rewrite it is not a reliable
way to turn an adapted implementation into independent work. Keep a provenance
record; use a genuinely separate specification/implementation process if a
strong clean-room claim is needed.

### Reusing MIT/Apache-2.0 components

These licenses generally permit commercial and closed-source products. They do
not force Tralens itself to become open source. Preserve applicable copyright and
license notices. For Apache-2.0, also mark modified files, retain relevant notices
and propagate a supplied NOTICE's applicable attribution content. Neither grants
unrestricted trademark rights. Keep the third-party terms with the component;
do not relabel upstream code as exclusively your own license.

MIT is a simple option for independently owned Tralens code. Apache-2.0 adds an
express contributor patent grant and patent-termination terms. A proprietary
license is also possible for owned code when the complete dependency/asset audit
allows it. The choice requires the copyright holders' authorization.

### Reusing GPLv3 implementation in a combined application

For a distributed combined/derivative program, plan to license the covered
combination under GPLv3-compatible terms, preserve notices, identify changes,
include the GPL and provide the exact version's Corresponding Source, including
required build/install materials. You may charge for GPL software; recipients
retain GPL rights. Adding an attribution link alone is not sufficient.

Private/internal use generally does not require publishing source. Ordinary
GPLv3 does not impose AGPL-style network source availability merely because a
program runs as a hosted backend. Distributing an app, container, customer-hosted
server or GPL-covered browser code is a separate distribution question. AGPL
dependencies require a separate network-use review. A subprocess, wrapper or
separate container is not automatically a copyleft exemption; whether programs
are independent or one combined work depends on the integration.

For proprietary distribution, use independently implemented approaches,
compatible permissive alternatives, or obtain a separate license from all
relevant rightsholders. Do not assume a project's maintainer can relicense every
third-party contribution or dependency.

## Model and asset terms are separate

Observed model-card metadata is evidence of declared terms, not a legal warranty:

| Artifact | Observed declaration | Deployment implication |
| --- | --- | --- |
| [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base/tree/aa6573bd10b0d446cbf622e29c3e084914df9741) | Apache-2.0; acknowledges Manga109-s | Keep the model's terms/provenance; do not claim a full data-rights audit from metadata alone. |
| [Koharu layout RF-DETR Seg 2XL](https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/blob/aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2/README.md) | Custom/`other`; explicitly references Manga109 academic-use terms | Not treated as cleared for commercial deployment. Obtain clarification/permission or use another detector. Koharu's application-code MIT license does not clear it. |
| [mayocream/lama-manga](https://huggingface.co/mayocream/lama-manga/blob/f91c85b26913b3e83f9877867b4c336da3675238/README.md) | MIT declared; asks users to review original checkpoint/implementation terms | Follow the original checkpoint's provenance too; this is not the same artifact as our existing pinned LaMa trial. |
| Existing Tralens CTD checkpoint | Acquired bytes are pinned; artifact-specific rights remain unresolved | Keep opt-in/local evaluation status; do not silently bundle or make it a production default. |

Font licenses, example-image licenses, translation-model terms and user-content
rights need their own entries. A software license does not grant rights to
translate, publish or sell copyrighted manga; it also does not automatically
make ordinary rendered output GPL-covered. Existing noncommercial sample-image
terms are not a license for a commercial product's example assets.

## Before integrating or shipping

- Decide hosted-only versus distributed/customer-hosted delivery and whether
  keeping application source private is a requirement.
- Inventory exact upstream revisions, reused files, dependencies, model hashes,
  original checkpoint sources, fonts and bundled examples.
- For each item, record declared terms, copyright notices, modifications,
  redistribution requirements and unresolved permissions.
- Add the owner-approved Tralens LICENSE and appropriate third-party notices;
  package required licenses/source/build materials with releases.
- Obtain clarification or replace unresolved/custom/noncommercial artifacts
  before commercial use. Get legal review for copyleft boundaries and unclear
  model rights, rather than treating this inventory as a legal opinion.

No root LICENSE or project license field was found in this checkout. Publicly
visible source is not automatically open-source licensed. Do not add a blanket
license until ownership and upstream compatibility have been checked.

## Primary licensing references

- [GNU FAQ: private use and source publication](https://www.gnu.org/licenses/gpl-faq.html#GPLRequireSourcePostedPublic)
- [GNU FAQ: hosting ordinary GPL software](https://www.gnu.org/licenses/gpl-faq.html#UnreleasedMods)
- [GNU FAQ: translating code is a modification](https://www.gnu.org/licenses/gpl-faq.html#TranslateCode)
- [GNU FAQ: containers and combined works](https://www.gnu.org/licenses/gpl-faq.html#AggregateContainers)
- [Why AGPL adds a network requirement](https://www.gnu.org/licenses/why-affero-gpl.html)
- [Apache-2.0 text, particularly section 4](https://www.apache.org/licenses/LICENSE-2.0)
- [Koharu MIT terms at the inspected revision](https://github.com/koharu-rs/koharu/blob/cd31ad2c7016b64d0ff0400a09e86cd13fb5df13/LICENSE-MIT)
