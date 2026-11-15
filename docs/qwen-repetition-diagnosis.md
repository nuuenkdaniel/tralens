# Qwen repetition diagnosis and local OCR comparison

Local evidence: `test-results/production-best-effort-v8/` (ignored; retained on
this machine). This checkpoint improves runtime diagnostics and permits explicit
seeded experiments. It does not establish translation accuracy.

## Reproduced failure

The historical November 11 request `b85a0d050d8a4cbf940b9a245cbcf7fc`
had stopped: the saved service journal records task 322 cancellation/release,
idle slots, shutdown and a later restart. Its original incident bytes are retained
with SHA256 `c564b0fc3a7ed94adc603257e63d97d89e40c93cc7c92eefd6c3949ae9dfb1f8`.

Replaying the exact frozen V4 `021-request.json` against local Ollama 0.33.2
reproduced HTTP 200 with `done:false` and a JSON string ending in repeated zeros.
The installed `qwen3.8:latest` is a 27.3B Q4_K_M `qwen35` model, digest
`22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643`.

Ollama's pinned `v0.33.2/llm/llama_server.go` aborts when `tokenRepeat > 30`
and returns `ctx.Err()`, which can still be nil. This explains a successful HTTP
status without completed generation. The captured journals show cancellation,
release and idle slots, not a runner crash. Each incident was archived and
acknowledged only after that evidence was checked.

| Exact-input control | Seconds | Result |
| --- | ---: | --- |
| Original greedy payload | 23.443 | Incomplete repeated zeros |
| Disable speculative drafting (`draft_num_predict=0`) | 3.823 | Same failure |
| Temperature .7, seed 42 | 5.310 | Completed; incorrect `うらら。`, uncertain |
| Explicit empty-text abstention + maxLength 128 | 3.648 | Completed; empty, uncertain |

These single-input controls do not isolate prompt from schema effects. The
subsequent 65-crop comparison used explicit abstention, temperature .7, seed 42,
and the original maxLength 1000: **65 completed responses, zero incomplete
generations**. It correctly abstained on crop 021. Completion was not accuracy:
it still misread ACPI/off, kana sound effects and the copy-paste caption, often
with `uncertain:false`. Raw requests, responses, crops and logs are retained.

## Runtime contract

- Callers may explicitly supply temperature 0..2 and an integer seed 0..2³¹−1.
- `GenerationOptions`, its saved metadata, default temperature 0 and TG27's
  official prompt are unchanged.
- Incomplete outputs are rejected and their raw replies retained. A repeated
  non-whitespace character tail is diagnosed as `incomplete_repetition`, described
  as a **possible** server repetition abort, never a proven cause from text alone.
- No automatic retry, incident clearance or promotion of incomplete text occurs.

## Alternative reader experiment

Hayai OCR v2.7 Supernova was acquired from
`JustANormalTinkerer/hayai-ocr-v2.7-supernova`, pinned revision
`517bdeeb64b68c08939e12f7211e99943c1e1640`, Apache-2.0 per its model card.
Weights: 633,564,960 bytes, SHA256
`c7dd04f7138e74abb48ea0dcf6bad7f2bdedd704fe5f8ec6328962e799c65837`.
Inference was local CPU float32 with four threads, 256 patches, max128 tokens.
No source images were uploaded.

The first loader attempt failed before inference because its tokenizer metadata
targets Transformers 5. The retained second trial loads the unchanged tokenizer
JSON through Transformers 4's fast tokenizer. The missing vision config is
supplied explicitly from pinned SigLIP2 configuration; no base vision weights
were downloaded. All checkpoint weights loaded with zero missing/unexpected keys.

**65 frozen crops completed in 8.624 seconds**, plus 1.031 seconds model load,
with zero inference failures. These are local diagnostic timings, not a matched
benchmark against Qwen or a production accuracy measurement.

All 11 comparison sheets were inspected. Useful readings include `Wgetでmake`,
`週刊`, `ネットワークの`, `コピペを極めし者`, `パチパチパチ`, and a substantially better
ACPI/off line. Errors remain: `Ubumtu`, clipped Latin names, case, partial kana,
joined interface/dialogue text, extra neighboring text and a non-text arrow
read as `く`. Both models are limited by the existing crop/grouping mistakes.
Blanket replacement or using model self-confidence as source verification is
not supported by this experiment.

Research starting points:
- <https://github.com/muscgab/JMangaBench_Read>
- <https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova>
- <https://github.com/ollama/ollama/blob/v0.33.2/llm/llama_server.go>

Next priorities are source-bound reading selection, complete group extents,
faithful TG27 translations, and displaying exact mixed-script targets. Background
reconstruction quality remains secondary to removing and replacing source text.
