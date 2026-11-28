# Tralens

Tralens is a self-hosted manga reader that translates Japanese, Chinese, and Korean pages into English. Import your own PNG, JPEG, WebP, PDF, CBZ, or ZIP files, read while pages translate, and switch between the original and translated pages.

## Quick start

Docker Compose is the recommended install on **Linux x86-64** with Docker Engine
and Docker Compose v2. The image includes the web reader, CPU OCR models, English lettering fonts, PDF tools, and classical inpainting. Ollama can use CPU or a supported GPU.

### Get the reader

For a versioned release, download its Compose bundle from [GitHub Releases](https://github.com/nuuenkdaniel/tralens/releases). The bundle pins the tested image by digest. To try the current development build, get its matching Compose file and image:

```sh
git clone --branch dev https://github.com/nuuenkdaniel/tralens.git
cd tralens
export TRALENS_IMAGE="ghcr.io/nuuenkdaniel/tralens-dev:dev-$(git rev-parse HEAD)"
docker compose pull reader
```

Save that exact `TRALENS_IMAGE` value in `.env` when you create it below so future Compose commands use the same image. For a local source build, follow the [build instructions](#build-from-source).

### Start the reader

Run these commands in the checkout or extracted release bundle. Use the non-root
account that will own the library.

```sh
if [ ! -e .env ]; then
  cp .env.example .env
  printf 'TRALENS_UID=%s\nTRALENS_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
fi
install -d -m 700 tralens-data/library tralens-data/config \
  tralens-data/state tralens-data/models tralens-data/assets tralens-data/ollama
```

If `TRALENS_IMAGE` points to GHCR, run `docker compose pull reader` first.
For local translation on a fresh machine, start the optional Ollama container and download the model once.

```sh
docker compose --profile ollama up -d --wait ollama
docker compose exec ollama ollama pull translategemma:12b
```

If you already run Ollama, pull the same model on that host and set `TRALENS_OLLAMA` if its address differs from the default. ChatGPT-only owners can skip Ollama and select a GPT model before importing. Start the reader with the OCR models already in the image.

```sh
docker compose up -d --wait --pull never
```

Open [http://127.0.0.1:7871](http://127.0.0.1:7871) on the host. From another
computer, use `ssh -N -L 7871:127.0.0.1:7871 user@server` and open that same URL.
On a fresh server, verify the address and choose a username and password for the first owner. The owner can connect ChatGPT. Each additional reader gets a private library. Existing owner-key installations keep their current sign-in.

Choose a model under **Settings → Translation**, then **Add manga**. Try one page
to check speed and quality on your hardware.

### Create the owner account from your phone

Configure [HTTPS access](#remote-access-over-https) first. In the Android app, enter the HTTPS hostname configured in `TRALENS_ORIGIN` and tap **Verify server**. If you use Tailscale Serve, connect your phone to Tailscale and enter the full `https://…ts.net` hostname shown by `tailscale serve status`.

On the server, run this from the Compose project directory to get your private, one-time setup link:

```sh
docker compose exec -T reader cat /library/owner-setup.url
```

Paste the **entire link**, including `#setup=…`, into the app’s **Private setup link** field and tap **Continue setup**. Choose your owner username and password.

You can also scan the link with your phone’s camera. If the server has `qrencode`, display a QR code in its terminal:

```sh
docker compose exec -T reader cat /library/owner-setup.url | tr -d '\r\n' | qrencode -t ANSIUTF8
```

If the camera opens a browser, finish setup there, then reopen the app and sign in with the same credentials. Keep the link and QR code private until setup finishes. If the reader restarts before you finish, run the command again for the new link.
### Android app

Download the signed `tralens-android-<version>.apk` and its `.sha256` file from
the same [GitHub release](https://github.com/nuuenkdaniel/tralens/releases) as the
Compose bundle. Verify the download with `sha256sum -c <apk-filename>.sha256`,
then open the APK on Android to install it. The Android app connects to your
self-hosted reader. Set up remote HTTPS access,
then enter the full HTTPS address in the app and tap **Verify server**.

#### Android release signing

Keep the PKCS#12 keystore and its password in a secure backup outside the
checkout. Use the same signing key for every public APK so Android can install
updates. If this is the first release and you need a key, run this from a
private directory and choose a strong password at the prompt.

```sh
keytool -genkeypair -storetype PKCS12 -keystore tralens-release.p12 \
  -alias tralens-release -keyalg RSA -keysize 3072 -validity 10000
```

Add the base64-encoded keystore and its password as Actions secrets in the
`public-beta` environment. From an authenticated GitHub CLI, run the following
and enter the keystore password when prompted by the second command.

```sh
base64 < tralens-release.p12 | gh secret set ANDROID_RELEASE_KEYSTORE_BASE64 \
  --env public-beta --repo nuuenkdaniel/tralens
gh secret set ANDROID_RELEASE_KEYSTORE_PASSWORD \
  --env public-beta --repo nuuenkdaniel/tralens
```

For **Stage public beta**, use Android version code `1` for the first public
APK and a higher code for each later release. Verify the draft's APK and checksum,
and test installation and reader sign-in before publishing.

## Translation models

| Option | Setup | Data sent for translation |
| --- | --- | --- |
| **Local Ollama** | Use the Compose Ollama service or your own Ollama server. Pull the default `translategemma:12b` or choose `translategemma:27b` on a larger machine. Set `TRALENS_MODEL` in `.env` and recreate the reader after changing it. | Text and any inputs required by optional local vision features go to your configured Ollama service. |
| **OpenAI via ChatGPT OAuth** | Open **Settings → Translation**, select an available GPT model, and complete **Continue with ChatGPT** using your own eligible account. No API key is needed. | Tralens sends page images, detected text, and translation prompts to OpenAI. Usage is subject to your plan and limits. |

Each reader account has a private library. Only the server owner can connect and use ChatGPT. Other accounts translate with the server’s local model. For a headless HTTPS server, the [OAuth guide](#connect-chatgpt-with-oauth) shows how to complete sign-in from your computer. Use **Disconnect ChatGPT** in Settings to revoke access. See the [privacy notice](PRIVACY.txt).

## Using the reader

New chapters default to **Source language → Auto**. Tralens compares text from up to three pages locally and saves one language for the chapter. If the sample is inconclusive, the chapter shows **Japanese, uncertain**. Pick a language during import to skip detection, or select **Language** beside a chapter to change it and re-translate. Existing chapters keep their Japanese setting. If you supply your own OCR model directory, include the Korean recognizer listed in the model notices or choose Japanese or Chinese manually.

- Import PNG, JPEG, still WebP, multipage PDFs, or CBZ/ZIP chapter archives, arrange their reading order, and create a manga or append a chapter. Imports are atomic. The current image limits are 32 MiB and 4 megapixels per opaque page, 4096 pixels per side, and 4 GiB per import (up to 256 expanded pages). Archive images (PNG, JPEG, still WebP) are read in natural folder and filename order, so page 2 precedes page 10. Metadata and hidden files are skipped. PDFs use server-side Poppler (pdfinfo and pdftoppm) and util-linux prlimit. Each PDF page becomes a PNG with a 2000-pixel long edge, in document order. Original downloads for PDFs are these rendered pages. WebP originals retain their uploaded bytes; transparency is flattened onto white for translation.
- Open a chapter while translation is running. The reader shows the original until a translated image becomes available. The queue processes one page at a time on the server and continues when the browser closes.
- Choose single page, two-page spread, or continuous scroll. Use a swipe, the arrow keys, the edge controls, or the page slider to navigate. Tap the page to toggle reader controls. Compare offers a draggable divider and side-by-side originals.
- Use the manga actions to rename, translate again, download a ZIP, or delete a title. The page menu offers individual downloads and retries. A failed retry retains the last available translation.
- The library, page images, chapters and reading positions are stored in `reader.sqlite3` under `--output-root` (default `test-results/web-reader/`). Appearance and model preferences are per browser. Deleting a title removes its library entries. The diagnostic job directories are separate and retain their pipeline artifacts. Back up the output directory while the service is stopped.

On Android, open a book and tap **Save offline** on a chapter while connected. The app saves each translated page that is ready and uses the original for remaining pages. Open **Downloads on this device** to read, resume a partial download, or remove a copy. Downloads and offline reading position belong to the signed-in account on that device. To refresh a saved chapter after more pages finish translating, remove it and save it again.

## Local models

The optional `ollama` Compose profile runs CPU inference and stores models in `tralens-data/ollama`. Use `TRALENS_OLLAMA_PORT` to change its loopback port. For GPU inference, point `TRALENS_OLLAMA` at your GPU-enabled Ollama service.

Keep Ollama on the same host at `http://127.0.0.1:11434` or set `TRALENS_OLLAMA` to a private IP address you control. Use [Ollama's local-only mode](https://docs.ollama.com/faq) if you want Ollama to avoid cloud features. Verify the service and model before adding pages.

```sh
curl -fsS http://127.0.0.1:11434/api/tags
ollama list
```

Compose defaults to `translategemma:12b`. For 27B, pull `translategemma:27b` and set `TRALENS_MODEL` to match. If you configure a different local model family, use `TRALENS_MODEL_PROFILE=generalist`. Restart with `docker compose up -d --wait`. Choose **Local** in reader Settings. Set `TRALENS_FRONTIER_PROVIDER=disabled` to hide GPT models and disable the ChatGPT integration.

## Remote access over HTTPS

For personal access, an SSH tunnel from your computer can reach the loopback reader as shown in the README. Keep the remote end of the tunnel private. On a fresh loopback reader, verify the server address, then choose your owner username and password. Only the owner can connect and use ChatGPT. Each account has its own library. By default other readers can register after the owner finishes setup. Set `TRALENS_REGISTRATION=closed` to allow only the initial owner setup and sign-in for existing accounts.

For HTTPS access from your own devices, set `TRALENS_ORIGIN=https://manga.example.com` in `.env` using the exact browser origin without a trailing slash. Configure the proxy below and start the reader. On a fresh HTTPS install, retrieve the private first-time setup link on the server:

```sh
docker compose up -d --wait
docker compose exec reader cat /library/owner-setup.url
```

Open that link on your phone or computer to choose the owner's credentials. The link expires when setup succeeds or the reader restarts. After a restart during setup, read the file again. Future sign-ins use the chosen username and password.

Existing owner-key installations continue to use the username `owner` and the same key. Keep the library and key together when restoring backups. The reader remembers each signed-in browser or app across restarts. Active devices renew their sessions daily, and a session expires after about 90 days of inactivity. On a shared device, use **Settings → Sign out** to revoke its session. Other devices stay signed in.

Set up a local reverse proxy that preserves the original Host header. For example, configure DNS and use Caddy.

```caddyfile
manga.example.com {
    reverse_proxy 127.0.0.1:7871
}
```

Keep the HTTP listener and Ollama private. If your proxy enforces upload limits, choose an appropriate cap and timeout. The reader caps a batch at 256 pages and a multipart request at 4 GiB. Individual images are limited to 32 MiB, 4 megapixels, and a 4096-pixel side. PDF rasterization and uploads need temporary disk space under `state/tmp`.

## Connect ChatGPT with OAuth

On Android, open the Tralens app, sign in as the server owner, and select a GPT model under **Settings → Translation**. Tap **Continue with ChatGPT**, finish in the phone browser, then tap **Open Tralens**. If you started in Android Chrome, **Open Tralens** takes you to the installed app. [Privacy notice](PRIVACY.txt)

For desktop browser sign-in, choose a GPT model and continue. If your browser runs on the server or reaches the reader over SSH, the OAuth callback still needs a loopback connection to the machine running Tralens. For a browser on the host, follow the link the reader shows. For a remote host, use a fixed callback port through SSH.

From your computer, open a dedicated SSH session and run the login command on the server.

```sh
ssh -L 1455:127.0.0.1:1455 user@server
cd /path/to/tralens
docker compose exec reader python backend/chatgpt.py login --no-browser --port 1455
```

Read the [privacy notice](PRIVACY.txt) before opening the printed private link in your local browser. Keep SSH open until consent finishes. Return to the reader and choose a model. If the port is in use, use the same alternate port for the SSH forward and login. A just-started sign-in in Settings may hold the OAuth lock until that attempt ends.

OpenAI determines which models your ChatGPT plan can use. These requests use your plan's eligible allowance and limits. In the Android reader, if you built and installed it, complete consent in the phone browser and tap **Open Tralens**. Sign in only as the owner, from devices you control. [OpenAI's open-source integration guide](https://developers.openai.com/siwc/token-sharing-open-source)

Disconnect in Tralens Settings or run `docker compose exec reader python backend/chatgpt.py logout`. To connect a different account, disconnect, stop the reader, remove the protected `tralens-data/config/tralens/chatgpt/account.json` file, and start again. Manage remaining access through your ChatGPT account's connections.

## Data, upgrades, and recovery

| Default path | Contents |
| --- | --- |
| `tralens-data/library` | Account names and password verifiers, local owner key, SQLite library, originals, translated pages, and per-job evidence |
| `tralens-data/config` | Owner key for HTTPS and protected ChatGPT configuration |
| `tralens-data/state` | Runtime and inference state, caches, and upload scratch space |
| `tralens-data/models` | Optional external OCR exports |
| `tralens-data/assets` | Optional operator-supplied assets |

The default upload and job storage budget is 8 GiB, with up to 2 GiB reserved per job. Keep free disk space beyond that budget for temporary files, models, and backups. Deleting a book in the reader removes its library entries. To reclaim older job artifacts, stop the service and back up the tree before archiving selected job directories. Removing a job directory also removes its job history and can affect saved transcript or diagnostic access.

Stop the reader before backing up. Protect backups as they may contain owner keys and OAuth credentials. Restore into an empty directory while the reader is stopped, preserving ownership and permissions. On a backup containing the reader_sessions table, run `DELETE FROM reader_sessions;` in the restored `reader.sqlite3` before starting the reader to discard saved logins, including any revoked since the backup. Use the same image version for the first start after restoring. Store the data on a local filesystem with file locks and atomic renames.

Before upgrading, save the current digest-pinned `TRALENS_IMAGE` and take a stopped backup. Change `TRALENS_IMAGE` to the new release digest, check the release notes, and run:

```sh
docker compose pull reader
docker compose up -d --wait --pull never
```

To roll back, stop the reader, restore the pre-upgrade backup, and recreate the service with its original image digest. A newer image can migrate SQLite or job records. Use the same state directory when multiple separate libraries share an Ollama endpoint so Tralens can coordinate inference.

### Troubleshooting

| Symptom | Check |
| --- | --- |
| Reader fails to start | Run `docker compose logs --tail=100 reader`. Confirm UID/GID, bind paths, model setup, and private owner-key permissions. |
| Browser cannot connect | Confirm `docker compose ps`, the port, the tunnel, or the proxy Host header and exact HTTPS origin. |
| Local translations fail | Confirm the Ollama endpoint and installed tag. Make sure Ollama has enough memory and test with one page. |
| Queue waits for a previous local inference | Inspect with `docker compose exec reader python backend/tools/inference_state.py --endpoint http://127.0.0.1:11434`. Use the configured endpoint if different. Acknowledge an incident only after confirming on the Ollama host that the request has stopped. |
| GPT model is missing | Confirm account eligibility and model availability. Reconnect if the grant was revoked or expired. |
| Upload fails | Check the file types, archive contents, page and size limits, writable state, available space, and proxy upload policy. |

## Build from source

Clone the release tag or branch you want. Follow the README for `.env` and directories. Compose uses `tralens:local` by default. If your `.env` pins a release image, set `TRALENS_IMAGE=tralens:local` for the source build.

```sh
docker compose -f compose.yml -f compose.build.yml build reader
docker compose up -d --wait --pull never
```

The image includes DejaVu Sans and Comic Neue. It uses the pinned PaddleOCR detector and recognizer under `/opt/tralens/ocr`. To use an external copy, set `TRALENS_OCR_ROOT=/models` and place the exports in `TRALENS_MODELS`. The optional `docker compose run --rm setup` command copies the bundled exports into that directory. To use a licensed custom static font, place it and its license under `tralens-data/assets` and point `TRALENS_FONTS` at its manifest under `/assets`. Use container paths and hashes in the manifest. Set `default_font_id` in a manifest with multiple faces. Classical inpainting runs on CPU. Auto uses the inpainting backends configured on the server. The base image supplies classical inpainting. GPU SDXL requires a compatible GPU runtime and local weights. The pinned LaMa checkpoint is supported for private noncommercial evaluation only. See the [model notices](THIRD_PARTY_NOTICES.txt) before adding checkpoints.

## Security policy

The Tralens beta supports one owner per Linux instance and optional reader accounts with private libraries. Set `TRALENS_REGISTRATION=closed` before exposing a personal server to other people. Each registered reader can consume local inference and storage capacity. The reader and Ollama should bind to loopback. For remote access, use an SSH tunnel or configure the reader's HTTPS owner mode behind a reverse proxy. See the [self-hosting guide](#remote-access-over-https).

Report suspected vulnerabilities privately using the repository's [private vulnerability reporting](https://github.com/nuuenkdaniel/tralens/security/advisories/new). If reporting is unavailable, use the repository owner's private contact information. Include the image tag or Git commit, affected component, impact, and steps that help confirm the issue. Remove any account tokens, owner keys, or private library content from the report.

## Release process

Open a PR against `dev` and wait for **Reader checks** to pass. Its merge builds
`ghcr.io/nuuenkdaniel/tralens-dev:dev`. Promote it through a PR from `dev` into
`main`. The main push build publishes a candidate to
`ghcr.io/nuuenkdaniel/tralens:main`. Each run records the pushed image digest
in its job summary. Use the digest from the successful main run for acceptance
and release. Keep both candidate packages private.

## Configure GitHub

Require PRs and the Reader checks on `dev` and `main`. Use the actual check
names from a successful PR, `clean-install` and `container / image`. Restrict
creation of `v*` tags to release maintainers and block tag updates and deletions.
Enable private vulnerability reporting in the repository settings.
[GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)

Create an Actions environment named `public-beta`. Restrict its deployments to
`main`, add a required reviewer, and disable administrator bypass of protection
rules. For a solo maintainer, allow the reviewer to approve their own run.
The staging workflow verifies that required reviewers are configured before it
writes to the release registry. [GitHub deployment environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)

## Approve the exact candidate

On a fresh Linux x86-64 host, check out the commit from the candidate's
`org.opencontainers.image.revision` label. Set `TRALENS_IMAGE` to
`ghcr.io/nuuenkdaniel/tralens@sha256:<candidate digest>` in `.env`. Authenticate
to GHCR with read access to the private candidate, then follow the
[Compose setup](#start-the-reader), including `docker compose pull reader`.
Record the digest and results.

- Import a page you have permission to use and complete a translation with
  local Ollama. Check the rendered result and original.
- Sign in through the actual HTTPS endpoint. Connect your eligible ChatGPT
  account, translate a page, and verify that disconnect works.
- Stop the reader, back up its data and test restoration. If there is a
  previous release, exercise upgrade and rollback with a copy of its data.
- Review the [third-party notices](THIRD_PARTY_NOTICES.txt) against the
  exact image. Check the Debian source inventory and the installed wheels'
  notices, including corresponding source for covered bundled binaries.
  Keep a copy of the review with the candidate digest.

CI exercises fresh Compose startup, the owner key, real CPU OCR, and recovery
of a completed job from a stopped backup.

## Stage and publish the beta

1. Create the beta tag, such as `v0.2.0-beta.1`, on the approved tip of
   `main`. Run **Stage public beta** from that commit on `main` with the tag and exact
   candidate digest. Confirm that the acceptance checklist is complete and
   approve the environment deployment.
2. The workflow pulls and tests that digest, then copies it into the separate
   `ghcr.io/nuuenkdaniel/tralens-release` package. It refuses to replace a
   version that points to different content. It creates a draft prerelease
   with a digest-pinned Compose bundle, signed Android APK and checksums.
3. After redistribution is cleared, make only the release package public.
   GitHub treats this change as permanent. Verify that the repository has
   Actions access to the package. [GitHub package visibility](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility)
4. Run **Verify public beta image** from `main` with the tag and digest. It
   tests anonymous pulls of the tag and digest on a fresh runner. Download
   the draft's Compose bundle and check that its `.env.example` contains
   the same digest. Publish the draft prerelease after these checks pass.

Use a new version tag for each release and retain the exact source and digest
for every distributed image. Keep the previous bundle and a stopped backup for
[rollback](self-hosted-reader.md#data-upgrades-and-recovery).
