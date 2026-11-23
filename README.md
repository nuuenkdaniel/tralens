# Tralens

Tralens is a self-hosted manga reader that translates Japanese pages into English. Import your own PNG, JPEG, WebP, PDF, CBZ, or ZIP files, read while pages translate, and switch between the original and translated pages.

## Quick start

Docker Compose is the recommended install on **Linux x86-64** with Docker Engine
and Docker Compose v2. The reader uses CPU OCR. Ollama can use CPU or a supported GPU.

Tralens is in preview. Use the source build below today. When a public beta is
available, download the Compose bundle from [GitHub Releases](https://github.com/nuuenkdaniel/tralens/releases).
Its `.env.example` pins the image digest used for that release.

### Build the preview

```sh
git clone https://github.com/nuuenkdaniel/tralens.git
cd tralens
docker compose -f compose.yml -f compose.build.yml build reader
```

### Start the reader

Run these commands in the checkout or extracted release bundle. Use the non-root
account that will own the library. To translate locally, [install Ollama](https://docs.ollama.com/linux)
on the host and run `ollama pull translategemma:4b` (about 3.3 GB). ChatGPT-only
installs can select a GPT model before importing pages.

```sh
if [ ! -e .env ]; then
  cp .env.example .env
  printf 'TRALENS_UID=%s\nTRALENS_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
fi
install -d -m 700 tralens-data/library tralens-data/config \
  tralens-data/state tralens-data/models tralens-data/assets
```

If `TRALENS_IMAGE` points to GHCR, run `docker compose pull reader` first.
Then download the pinned PaddleOCR models and start the reader.

```sh
docker compose run --rm --pull never setup
docker compose up -d --wait --pull never
```

Open [http://127.0.0.1:7871](http://127.0.0.1:7871) on the host. From another
computer, use `ssh -N -L 7871:127.0.0.1:7871 user@server` and open that same URL.
For HTTPS access, follow the [single-owner deployment guide](docs/self-hosted-reader.md#remote-access-over-https).

Choose a model under **Settings → Translation**, then **Add manga**. Try one page
to check speed and quality on your hardware.

### Android app

Download the signed `tralens-android-<version>.apk` and its `.sha256` file from
the same [GitHub release](https://github.com/nuuenkdaniel/tralens/releases) as the
Compose bundle. Verify the download with `sha256sum -c <apk-filename>.sha256`,
then open the APK on Android to install it. The Android app connects to your
self-hosted reader. Set up remote HTTPS access,
then enter the reader's hostname in the app, enable **Use HTTPS**, and use port
`443` unless your reverse proxy uses a different port.

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
| **Local Ollama** | Start Ollama on the host. Pull `translategemma:4b` for a smaller model or use `translategemma:12b` or `translategemma:27b` with sufficient memory. Set `TRALENS_MODEL` in `.env` and recreate the reader after changing it. | Text and any inputs required by optional local vision features go to your configured Ollama service. |
| **OpenAI via ChatGPT OAuth** | Open **Settings → Translation**, select an available GPT model, and complete **Continue with ChatGPT** using your own eligible account. No API key is needed. | Tralens sends page images, detected text, and translation prompts to OpenAI. Usage is subject to your plan and limits. |

The server and its connected ChatGPT account are for one owner. For a headless HTTPS server, the [OAuth guide](docs/self-hosted-reader.md#connect-chatgpt-with-oauth) shows how to complete sign-in from your computer. Use **Disconnect ChatGPT** in Settings to revoke access. See the [privacy notice](PRIVACY.md).

## Operating your server

The [self-hosting guide](docs/self-hosted-reader.md) covers HTTPS, storage, backups, upgrades, rollback, model settings, and troubleshooting. See the [source build instructions](docs/self-hosted-reader.md#build-from-source) to run a checkout without a registry image.

OCR can miss handwritten text and sound effects. Translation and artwork cleanup vary by page. **Keep the artwork** favors preserving the background when cleanup is uncertain. Use **Compare** to check the result against the original.

Use content you have permission to process and share. Tralens is GPL-3.0-or-later. Font, dependency, and model terms are listed separately. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

Report reproducible issues on [GitHub](https://github.com/nuuenkdaniel/tralens/issues), using a sample you have permission to share. See the [security policy](SECURITY.md) for private reports and the [development guide](DEVELOPMENT.md) for CLI and pipeline work.
