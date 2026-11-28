# Tralens

Tralens is a self-hosted manga reader that translates Japanese pages into English. Import your own PNG, JPEG, WebP, PDF, CBZ, or ZIP files, read while pages translate, and switch between the original and translated pages.

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

Save that exact `TRALENS_IMAGE` value in `.env` when you create it below so future Compose commands use the same image. For a local source build, follow the [build instructions](docs/self-hosted-reader.md#build-from-source).

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

Configure [HTTPS access](docs/self-hosted-reader.md#remote-access-over-https) first. In the Android app, enter the HTTPS hostname configured in `TRALENS_ORIGIN` and tap **Verify server**. If you use Tailscale Serve, connect your phone to Tailscale and enter the full `https://…ts.net` hostname shown by `tailscale serve status`.

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
| **Local Ollama** | Use the Compose Ollama service or your own Ollama server. Pull the default `translategemma:12b` or choose `translategemma:27b` on a larger machine. Set `TRALENS_MODEL` in `.env` and recreate the reader after changing it. | Text and any inputs required by optional local vision features go to your configured Ollama service. |
| **OpenAI via ChatGPT OAuth** | Open **Settings → Translation**, select an available GPT model, and complete **Continue with ChatGPT** using your own eligible account. No API key is needed. | Tralens sends page images, detected text, and translation prompts to OpenAI. Usage is subject to your plan and limits. |

Each reader account has a private library. Only the server owner can connect and use ChatGPT. Other accounts translate with the server’s local model. For a headless HTTPS server, the [OAuth guide](docs/self-hosted-reader.md#connect-chatgpt-with-oauth) shows how to complete sign-in from your computer. Use **Disconnect ChatGPT** in Settings to revoke access. See the [privacy notice](PRIVACY.md).

## Operating your server

The [self-hosting guide](docs/self-hosted-reader.md) covers HTTPS, storage, backups, upgrades, rollback, model settings, and troubleshooting. See the [source build instructions](docs/self-hosted-reader.md#build-from-source) to run a checkout without a registry image.

OCR can miss handwritten text and sound effects. Translation and artwork cleanup vary by page. **Keep the artwork** favors preserving the background when cleanup is uncertain. Use **Compare** to check the result against the original.

Use content you have permission to process and share. Tralens is GPL-3.0-or-later. Font, dependency, and model terms are listed separately. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

Report reproducible issues on [GitHub](https://github.com/nuuenkdaniel/tralens/issues), using a sample you have permission to share. See the [security policy](SECURITY.md) for private reports and the [development guide](DEVELOPMENT.md) for CLI and pipeline work.
