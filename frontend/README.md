# Tralens web reader

The browser application in `web/` serves a manga library and reader. The browser only needs HTML, CSS, JavaScript and the same-origin HTTP API. OCR, translation, GPU inpainting, and library storage run on the backend.

## Run the web app

From this checkout, with the installed evaluation runtimes:

```sh
.venv-paddle6/bin/python backend/serve.py --port 7871 \
  --fonts test-results/production-best-effort-v8/font-render-v3/fonts.json --font-id dialogue-bold \
  --crop-reader-model test-results/production-best-effort-v8/hayai-supernova \
  --text-mask-model test-results/approved-local-runtime-v1/models/comictextdetector.pt.onnx \
  --reconstruction-config test-results/web-reader-config/lama.json \
  --sdxl-config test-results/web-reader-config/sdxl.json
```

Open [the reader](http://localhost:7871/) or [the diagnostic viewer](http://localhost:7871/developer). On a fresh checkout, supply your licensed fonts and available model paths. The font manifest is required for translated output. Both learned-reconstruction flags are optional. Local translation uses the configured Ollama service. Selecting a GPT model in Settings starts ChatGPT sign-in the first time. In the Android app, approve access in the phone’s browser and tap **Open Tralens** to finish. Local translation needs no account. Codex CLI is an explicit server override. For HTTPS, owner sign-in, and direct account setup, follow the [single-owner deployment guide](../docs/self-hosted-reader.md).

**Settings → Translation** selects Local, GPT-6 Luna, GPT-6.1 Terra, or GPT-6.2 Sol. **Inpainting model** selects Classical, LaMa or SDXL. The backend advertises the configured options. Model selections apply to new imports and explicit retries. Keep the artwork uses classical cleanup. Replace it anyway uses the selected model for eligible textured regions. Uniform fills, mask checks, time budgets and classical fallbacks still apply.

For SDXL, pass an operator-owned JSON file to `--sdxl-config`:

```json
{
  "backend": "sdxl-inpaint",
  "model": "/path/to/local/sdxl-inpainting-diffusers-directory",
  "python": "/path/to/gpu-environment/bin/python"
}
```

The worker expects the fp16 safetensors layout of `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, a CUDA-capable PyTorch runtime, Diffusers, Transformers, NumPy and Pillow. Model and runtime paths can also be relative to the config file. Inference uses local files, 25 steps, strength 0.78, and a fixed seed. Only the committed mask is composited back onto the page. LaMa uses the existing pinned ONNX configuration described in the backend guide.

## Using the reader

- Import PNG, JPEG, still WebP, multipage PDFs, or CBZ/ZIP chapter archives, arrange their reading order, and create a manga or append a chapter. Imports are atomic. The current image limits are 32 MiB and 4 megapixels per opaque page, 4096 pixels per side, and 128 MiB per import (up to 256 expanded pages). Archive images (PNG, JPEG, still WebP) are read in natural folder and filename order, so page 2 precedes page 10. Metadata and hidden files are skipped. PDFs use server-side Poppler (pdfinfo and pdftoppm) and util-linux prlimit. Each PDF page becomes a PNG with a 2000-pixel long edge, in document order. Original downloads for PDFs are these rendered pages. WebP originals retain their uploaded bytes; transparency is flattened onto white for translation.
- Open a chapter while translation is running. The reader shows the original until a translated image becomes available. The queue processes one page at a time on the server and continues when the browser closes.
- Choose single page, two-page spread, or continuous scroll. Use a swipe, the arrow keys, the edge controls, or the page slider to navigate. Tap the page to toggle reader controls. Compare offers a draggable divider and side-by-side originals.
- Use the manga actions to rename, translate again, download a ZIP, or delete a title. The page menu offers individual downloads and retries. A failed retry retains the last available translation.
- The library, page images, chapters and reading positions are stored in `reader.sqlite3` under `--output-root` (default `test-results/web-reader/`). Appearance and model preferences are per browser. Deleting a title removes its library entries. The diagnostic job directories are separate and retain their pipeline artifacts. Back up the output directory while the service is stopped.

## Client contract

Serve `web/` as static assets under the backend origin. A local development proxy can forward `/api/` with Host and Origin checks preserved. In owner mode, sign in through the reader over HTTPS. Both the web and Capacitor clients use the server’s secure session cookie.

| Method and path | Payload / result |
|---|---|
| GET `/api/config` | Capabilities and the token required in `X-Tralens-Token` on writes |
| GET `/api/library` | Books, chapters, ordered pages, availability, statuses, read positions, image storage |
| POST `/api/library/add` | Multipart with a JSON `metadata` field containing `book` (optional existing ID), `title`, `chapter`, and job `options`, plus repeated `page` files in reading order |
| POST `/api/library/position` | JSON containing `book` and `page` |
| POST `/api/library/retry` | JSON containing `book`, optional `page`, and new job `options` |
| POST `/api/library/rename`, `/api/library/delete` | JSON containing `book`, plus `title` for rename |
| GET `/api/library/pages/{id}/original` or `translated` | Original upload or last published translated PNG |
| GET `/api/library/books/{id}/{variant}.zip` | Chapter-ordered archive; variant is `original`, `translated`, or `both` |

The standalone [job endpoints](../docs/user-facing-frontend-brief.md#frontend-integration-contract) also remain available for clients that manage their own libraries.

The empty-library illustration uses reduced-size demo page thumbnails from *Ubunchu!* by Hiroshi Seo (CC BY-NC-SA) and *Pepper & Carrot* by David Revoy (CC BY 4.0). Model weights and runtime environments are operator-supplied.

## Disposable Android build tools

From the repository root, run `frontend/android/provision` once. For a stand-alone build, set `JAVA_HOME` to a JDK 21 installation, `GRADLE_HOME` to Gradle 8.11.1, and `ANDROID_HOME` to an SDK containing platform 35, build-tools 35.0.0, platform-tools, cmdline-tools/latest and accepted licenses. Node and npm must be on PATH. Provisioning copies those tools into this checkout. On this Linux workstation it copies the productivity app’s JDK 21, Gradle 8.11.1 distribution, Android SDK platform 35, build-tools 35.0.0, command-line tools and platform-tools into Tralens’s ignored `.android-build/`. It also copies the local Node and npm programs. Pass a seed directory as the first argument if the productivity checkout is elsewhere. The seed is only required during provisioning.

Use `frontend/android/android-env` to invoke the tools. For example, run `frontend/android/android-env gradle --version` or `frontend/android/android-env npm config get cache`. Commands start in `.android-build/work/`; keep generated Capacitor projects, node_modules, intermediates and APKs there. Gradle runs offline by default; use `gradle --online` when fetching dependencies for the first time. The launcher scopes Java and Android user files, npm and Gradle caches, temporary files and process home to this workspace. Host Linux libraries and basic shell utilities are still required.

To clean up, stop active builds and any Android tools you launched, keep any APKs or future signing keys you intend to reuse, then delete **only the Tralens checkout’s `.android-build/` directory**. The small scripts stay in `frontend/android/` for reprovisioning. Reader library data and backend model files have separate lifecycles.

## Build and run the Capacitor Android app

The Android debug app bundles the reader in the APK. The backend on your computer owns the library and runs the models. On first launch, enter the server IPv4 address and port. The app saves your choice. Use **Settings → Server → Change server** to switch backends, or **Change server** on the connection screen.

From the repository root, provision the toolchain once, then run:

```sh
frontend/android/build-apk --online
```

After dependencies are cached, `frontend/android/build-apk` rebuilds offline. The installable file is `.android-build/outputs/tralens-debug.apk`. The source config, lockfile and native overrides live in `frontend/android/`; the generated Capacitor/Gradle project and debug keystore live under `.android-build/`.

For LAN or Tailscale access, add `--host 0.0.0.0 --port 7871` to the backend command above. Use the computer’s reachable IPv4 address in the app. The port field also accepts a different backend port. This trusted-network mode has no account authentication: anyone who can reach that listener can access the library and run jobs. Limit network reachability to trusted devices.

For HTTP on Android 7 and newer, the build permits the workstation's specific IPv4 addresses and loopback. To choose the permitted addresses explicitly, e.g. when building on another machine, run `TRALENS_HTTP_HOSTS="192.168.1.50 100.64.1.2" frontend/android/build-apk`. If DHCP assigns a new HTTP address, rebuild with that address. For a TLS-enabled endpoint select **Use HTTPS**; Android must trust its certificate for the entered IP. The built-in Python listener serves plain HTTP.

Install the APK directly, or use the isolated ADB launcher:

```sh
frontend/android/android-env adb devices
frontend/android/android-env adb -s SERIAL install -r "$PWD/.android-build/outputs/tralens-debug.apk"
```

USB forwarding is optional. For a USB-only session, run `frontend/android/android-env adb -s SERIAL reverse tcp:7871 tcp:7871` and select `localhost` and port `7871` in the app. The dedicated ADB daemon uses port 5059. Stop it with `frontend/android/android-env adb kill-server` before cleaning up. Keep the debug keystore for future in-place updates.

## Android release build

Run `frontend/android/build-apk --online --release` the first time, then `frontend/android/build-apk --release` for cached builds. The artifact is `.android-build/outputs/tralens-release-unsigned.apk`. Release builds permit HTTPS and loopback HTTP. The private-network HTTP exceptions used by debug builds are excluded.

To distribute a release, sign the APK with your Android release key and verify the signature. The command prompts for the keystore password. Keep the same signing identity for future updates and keep its backup outside the disposable build workspace.

```sh
frontend/android/android-env apksigner sign --ks /absolute/path/to/release.jks \
  --out "$PWD/.android-build/outputs/tralens-release.apk" \
  "$PWD/.android-build/outputs/tralens-release-unsigned.apk"
frontend/android/android-env apksigner verify --verbose "$PWD/.android-build/outputs/tralens-release.apk"
```
