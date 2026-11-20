# Tralens · User-facing frontend brief

Original design handoff. The implemented reader, library endpoints, and SDXL configuration are documented in the [web reader guide](../frontend/README.md).

Handoff based on the `dev` branch at `4846074`, November 19, 2026.

## What the reader comes to do

A reader has a Japanese manga page and wants to read it naturally in English. Tralens finds lettering in the artwork, reads it, translates it, removes the original strokes, reconstructs the background, and letters the result back onto the page. It handles monochrome manga and full-color comics. Dialogue, narration, signs, handwritten asides, and stylized sound effects all matter.

**Product priority:** when an English translation exists, the default experience should put it on the page, even if the background repair is imperfect. Preserve the original so the reader can compare. Offer a deliberate artwork-protection choice for someone who would prefer to leave difficult areas intact. Keep the distinction between *found*, *translated*, and *drawn* visible. Each can succeed while the next fails.

**Suggested first-use experience:** drop one page, see which model will process it, choose Translate, and read the finished page at a comfortable size. Give the page the most space. Make comparison and retry easy to find. Put detector geometry, model budgets, masks and logs within reach for the reader who encounters a bad result. The existing development viewer is a working behavioral reference. Its layout, terminology and visual style are open to redesign.

## User choices the backend supports

| Choice | Behavior to convey |
|---|---|
| Output | **Translated page** runs the whole pipeline. **Text only** returns region-linked translations alongside the original. **Detect text** runs local OCR and grouping. Rendering requires a configured font manifest. The viewer starts in text-only mode when rendering is unavailable. |
| Replacement | **Replace all text** is the default. It attempts every available translation, with a patch or nearby caption when needed. **Protect artwork** uses cleanup and OCR checks and leaves unsafe placements in the translation list. “All” expresses the replacement policy, not a guarantee that every mark was detected or translated. |
| Translation model | **Local** uses the configured Ollama endpoint and model, normally TranslateGemma 27B. **GPT-6 Luna**, **GPT-6.1 Terra**, and **GPT-6.2 Sol** share the frontier pipeline and use the server operator’s ChatGPT OAuth connection by default. Local is the default for a new setup. All four retain local detection and rendering. |
| Model input | Frontier default `reviewed` uses the whole page plus numbered crops, then rechecks skipped regions while budget remains. Other supported modes are whole page (`page`), page plus numbered crops (`atlas`), page plus a magnified target per region (`panel`), and individual OCR readings (`isolated`). |
| Context | In visual frontier modes the reader can select up to six saved runs from the same manga as context. Only the current page receives lettering. Context selection is explicit. There is no automatic series library or chapter import in this API. |
| Reconstruction | Classical reconstruction is available. LaMa is an optional local inpainting backend used with aggressive removal when configured. A configured text-mask model helps separate letters from art. Without it, broad removal can erase a whole OCR box. |
| Advanced controls | Translation call budget is 0–64, default 64. Failed calls consume budget. Paired OCR rereads are available for compatible rendering runs when configured. Local crop rereading with Hayai can also be configured by the operator. Read capability flags at startup. |

**Make the data destination clear before Run.** Local translation uses the configured Ollama service, which may be on another machine. Visual frontier modes send the target image and relevant crops or context images to OpenAI. Frontier OCR-per-region mode sends text and coordinates. Choosing “Text only” changes the *output*, so a user who selects that output and a visual frontier mode can still send an image. The reader starts sign-in and the server manages the connection. The frontend does not collect a ChatGPT password or subscription token. Tralens currently has no application-level NSFW screen, blur, or age gate.

## What happens to a page

1. **Original.** Upload one opaque PNG or JPEG, up to 32 MiB, 4 million pixels, and 4096 pixels per side. Keep the original accessible in every run. Loading the app, opening history and refreshing are read-only actions.
2. **Detected text.** Local PaddleOCR detects individual polygons. Tralens groups detections into translation units, including vertical lettering and multi-line dialogue. One region in the inspector can own several boxes. Configured crop readers may revise the Japanese reading.
3. **Translation.** The model returns English per eligible unit or a reason for skipping it. In normal full-page visual frontier runs with at least two calls available, the selected model also scans the original and detector outlines for missed lettering. Accepted additions enter grouping, translation and rendering. The snapshot reports discovery in notices. The model can abstain on duplicate or already-English text, unreadable fragments, or non-text. Display the OCR reading, the model’s reading when different, the English, and any reason.
4. **Cleaned page.** The renderer identifies strokes, removes them and reconstructs the background. A cleanplate is an intermediate artifact worth inspecting when a drawing looks damaged.
5. **Final.** The renderer lays out English with configured fonts and publishes a composite. Show it as soon as the artifact exists. A partial run can have a useful final image. An OCR-only job, failed job or job with zero usable translations can have no final at all.

A useful example comes from *Ubunchu!*: the frontier scan found a handwritten interjection next to previously detected dialogue. Joining it to the utterance helped recover the meaning of an interrupted line about a network install. In a color *Pepper & Carrot* page, the models found repeated applause, but several proposed boxes overlapped. These are reasons to let the reader click an area, inspect the Japanese and English, and compare the pixels. The earlier four-page language trial also found that a model could read the kana correctly and still choose the wrong English sound effect. Counts and confidence are useful diagnostics, not translation accuracy scores. [Matched translation trial](luna-translation-trial.md)

## Reading, inspection and recovery

- Preserve the existing fit-page, fit-width, actual-size and expanded views. Offer original/final swipe or side-by-side comparison with aligned scale. Keep comparison usable with a keyboard. On a narrow screen, a region list can move below the page or into a drawer.
- Let users switch among the five stage views and inspect published output while processing continues. Selecting an earlier stage currently pauses automatic following. Preserve the reader’s place when another checkpoint arrives.
- Selecting a region should connect the transcript row with its page polygons. Offer grouped regions by default and raw detections for detailed inspection. Show the available English even when the renderer could not place it. Put masks, lettering layers, raw OCR and artifact files in secondary details.
- Preserve completed, partial, failed and interrupted runs. Show the actual stage, available counts, and error or skip reasons. The backend publishes checkpoints; the current client polls roughly once a second. A spinner and stage label are more truthful than an invented percentage.
- History supports opening a saved result by `?job=<id>`. Retry reuses the original upload, current settings and a fresh call budget, creates a new job, and preserves the older result. This is how users compare models on exactly the same image. The existing viewer can also retry with broad replacement when a previous run left translations off-page.
- One job runs at a time. The current server admits eight new jobs per session and restores up to 24 recent runs. Cancellation depends on isolated-worker support. A cancelled or interrupted run may already have published useful artifacts. Surface busy, unavailable, and session-limit responses without silently launching another run.

## Frontend integration contract

Build the reader app in `frontend/` and connect through the existing same-origin HTTP API. See the [client integration notes](../frontend/README.md) for local development. The Python coordinator owns inference, file handling and reconstruction. The reference client’s `runOptions` and `currentSettings` functions encode working preset combinations. Send the **complete current options** on retry because the server merges supplied fields with the old job’s settings. Keep the wire keys `luna_mode` and `luna_reading` for all three frontier models.

| Request | Purpose and payload |
|---|---|
| `GET /api/config` | Session token, configured local model and endpoint, size limits, and flags for rendering, cancellation, reconstruction, text masks, paired rereads and worker recovery. |
| `GET /api/jobs` | Recent jobs with ID, status and output mode. |
| `POST /api/jobs` | Raw image bytes in the body. Set `X-Tralens-Options` to JSON and `X-Tralens-Token` to the session token. Success returns HTTP 202 and a job ID. |
| `GET /api/jobs/{id}` | Poll for job/stage status, options, original size/name, polygon detections, grouped units, translations, coverage, display fallbacks, notices, errors and published artifact paths. Older runs can lack newer fields. |
| `POST /api/jobs/{id}/retry` | JSON options body and token header. Returns the new ID; switch polling to that job. |
| `POST /api/jobs/{id}/cancel` | Send the token and a JSON body such as `{}`. Offer when cancellation is available and the job is running. |
| `GET /api/jobs/{id}/artifacts/{relative-path}` | Use paths listed in the snapshot and encode each path segment. The final image is normally `run/rendering/composite.png`; the cleanplate is `run/rendering/clean.png`. The original is identified by `input`. |

The current server is a local development service with origin checks, a session token on writes, and a same-origin content security policy. Its static handler explicitly serves `/`, `/app.js`, and `/style.css`. If a new build emits hashed JS, fonts or additional assets, wire up asset delivery and verify it under that policy. A public multi-user service would also need a separate deployment and account/privacy design. The current API exposes local job history and filesystem details.

## Scope for a first release

The shipping unit is **one source page and one saved run**. Batch chapters, CBZ/PDF import, automatic series management, user accounts, billing, and in-browser text or mask editing would require additional backend work. The repository has deeper CLI and research tooling for OCR candidates, manual review, localization, alternative grouping, restoration and translation evaluation. The missed-lettering discovery pass is integrated today. SDXL inpainting trials and the specially tuned outline cleanup from the applause experiment are research results. The viewer’s supported reconstruction choices are classical and configured LaMa.

**Suggested acceptance walkthrough:** upload one monochrome page and one color page; run a local translation; inspect a region and compare the final to the original; retry the same upload with a frontier model and optional context; open the old result again after refresh. Also exercise missing fonts, an exhausted budget, a partial output, a failed provider, and cancellation where available. Existing test fixtures can exercise failures and retries without paid inference.

## Where to start in the repository

- [Backend service entry point](../backend/serve.py), [launch flags](../backend/tools/dev_frontend.py) and [README setup](../README.md)
- [HTTP contract, validation and snapshots](../backend/src/tralens/devui/server.py) and [worker lifecycle](../backend/src/tralens/devui/jobs.py)
- [Reference browser interactions and preset serialization](../backend/src/tralens/devui/static/app.js) and [current UI structure](../backend/src/tralens/devui/static/index.html)
- [Pipeline coordinator](../backend/src/main.py), [translation workflow](../backend/src/tralens/translation/workflow.py), [frontier provider](../backend/src/tralens/translation/luna.py), and [missed-lettering discovery](../backend/src/tralens/translation/discovery.py)
- [Viewer HTTP tests](../tests/test_dev_frontend.py), [visible frontend tests](../tests/test_visible_frontend.py), and [job isolation tests](../tests/test_job_isolation.py)
