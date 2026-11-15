# Image translation diagnostic checkpoint

The installed TG27 image path does **not** support replacing the retained text
translation pipeline. Successful image transport is not a fidelity measurement.
No image-trial output was selected, rendered or added to a production policy.

## Reproducible setup

- Local Ollama 0.33.2, `translategemma:27b`, Q4_K_M, model digest
  `21b597a243ec4fd2f05bc0ce8ef9ef174a89b09255c330d670852dfaaaa74f72`.
- Google's public model-tree metadata identifies `chat_template.jinja` as Git blob
  `d571968f9770221512d29627697012202451983c`. A public MLX mirror's 16,982-byte
  template matches that blob exactly. The official file endpoint required access;
  the exact source URLs, bytes and hashes are retained locally.
- The rendered official image prompt requests Japanese-to-English translation
  only, without explanations, alternatives or surrounding-scene descriptions.
  Ollama's `[img]` replaces the image marker; the installed turn template supplies
  control tokens. Ollama 0.33.2's saved `server/prompt.go` confirms that placement.
- Server evidence confirms CUDA vision encoding and 256 image tokens. It loads
  the monolithic model as the projector through Gemma3 compatibility handling.
  That demonstrates execution, not independently validated image embeddings.

Evidence is under ignored `test-results/production-best-effort-v8/`:
`tg27-vision-research-01/`, `tg27-vision-trial-01/`,
`tg27-vision-trial-01-review/`, and `tg27-vision-controls-01/`.
Exact requests, image hashes, original previews, raw responses, model metadata,
runner snapshots and acknowledged release/empty-runtime records are retained.
The prior OCR and translations are review metadata, never model inputs.

## Inspected manga results

The initial diagnostic made 72 requests: 24 selected difficult regions across
four working pages, each as native, padded and wider-scene crops. All 24 bilingual
source/scene/output cards were inspected. Widespread failures include:

| Visible source / issue | Image output example | Assessment |
| --- | --- | --- |
| Embarrassment about misunderstanding open source | “I'm so glad I talked about it. It lifted my spirits.” | Unrelated meaning |
| `パチパチパチ` (clapping in this scene) | “Pikachu, Pikachu.” | Incorrect reading and meaning |
| ACPI/off handwritten instruction | “Turn off the AC, it's too cold.” | Invented expansion and dialogue |
| `……それ励ましてるつもり？` | “…Is it okay if I do it?” | Loses the encouragement question |
| `なんの話だっけ…？` | “What should I say?” | Loses the retrospective question |
| `まずはセキュリティスキャン` | “I'm doing my best.” | Unrelated meaning |

Some isolated results improve: a padded `もういいのよ` becomes “It's okay,”
and a native Latin name becomes “Masato.” Other framing produces unrelated
dialogue or invented expansions. These successes are not stable selection rules.
Wider-scene results also import neighbors and have no verified per-unit binding.

## Presentation and prompt controls

The follow-up completed 51 image requests plus one text-only control. It compared
the official prompt, image-first placement of that prompt, and a simple translation
instruction. Seven retained manga regions were both aspect-preserving 896-square
framed and enlarged 4×. The synthetic controls were an unrelated clean sentence
in horizontal and vertical layouts, plus a blank white image.

- All six typeset-sentence image requests correctly convey
  `今日の会議は午後三時からです。` as a meeting starting at 3 PM today.
  The text-only control is also correct. Both orientation controls were inspected.
- The blank image yields “[Image of a cat]” under the official prompt and invented
  company placeholders under image-first placement. Only the simple prompt
  correctly describes it as blank.
- Square framing, enlargement and placement changes do not repair the open-source
  passage, ACPI/off instruction or encouragement question. The large source crops
  visibly contain the required text; the outputs still invent different dialogue.
- The simple prompt can recognize the clapping effect, but imports a neighboring
  interjection and returns explanations. It also misreads clear typeset dialogue.
- The retrospective question becomes “What shall we talk about?” in multiple
  controls. Repeated fluent agreement is still semantically wrong.

The clean controls argue against a completely disconnected or universally broken
image path. They do not prove the runtime's numerical correctness or establish
general model capability. No inspected framing/prompt variant provides a reliable
manga-reading/translation policy, and no change to TG27's production text prompt,
greedy defaults, source selection or rendering is justified by this experiment.

The trials reuse difficult development pages and include no independent bilingual
review or held-out accuracy measurement. The existing text-pipeline language and
cleanup defects remain unresolved; 0% errors / 100% fidelity has not been achieved.
