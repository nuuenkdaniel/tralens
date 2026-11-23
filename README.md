# Tralens

Tralens is a self-hosted manga reader that translates Japanese pages into English. Import your own PNG, JPEG, WebP, PDF, CBZ, or ZIP files, read while pages translate, and switch between the original and translated pages.

## Quick start

Docker Compose is the recommended install on **Linux x86-64** with Docker Engine
and Docker Compose v2. The reader uses CPU OCR. Ollama can use CPU or a supported GPU.

Tralens is in preview. Use the source build below today. When a public beta is
available, download the Compose bundle from [GitHub Releases](https://github.com/nuuenkdaniel/tralens/releases).
Verify the archive with the attached checksum before extracting it. Its
`.env.example` pins the image digest for that release.

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
computer, use `ssh -N -L 7871:127.0.0.1:7871 user@server` and open the same URL.

Choose a model under **Settings → Translation**, then **Add manga**. Try one page
to check speed and quality on your hardware.

## Translation models

| Option | Setup | Data sent for translation |
| --- | --- | --- |
| **Local Ollama** | Select **Local** with Ollama running on the host. Set `TRALENS_MODEL` in `.env` to the installed tag and recreate the reader after changing it. Use `TRALENS_MODEL_PROFILE=generalist` for other model families. | Text and any inputs required by optional local vision features go to your configured Ollama service. |
| **OpenAI via ChatGPT OAuth** | Open **Settings → Translation**, select an available GPT model, and complete **Continue with ChatGPT** using your own eligible account. No API key is needed. | Tralens sends page images, detected text, and translation prompts to OpenAI. Usage is subject to your plan and limits. |

The reader supports one owner per instance. Set `TRALENS_FRONTIER_PROVIDER=disabled`
to disable ChatGPT. Use **Disconnect ChatGPT** in Settings to revoke access. See
the [privacy notice](PRIVACY.txt).

For a headless server, forward the OAuth callback from your computer, then run
the login command from the directory containing `compose.yml` on the server.

```sh
ssh -L 1455:127.0.0.1:1455 user@server
cd /path/to/tralens
docker compose exec reader python backend/chatgpt.py login --no-browser --port 1455
```

Open the printed link in your computer’s browser. Keep SSH open until consent
finishes, then return to **Settings → Translation** to choose a GPT model.

## Remote access over HTTPS

Keep the reader and Ollama on loopback. Anyone who can connect to the reader’s
loopback port can reach it without a key until owner mode is enabled. To enable
owner mode behind a reverse proxy, set `TRALENS_ORIGIN=https://manga.example.com`
in `.env` to the exact browser origin without a trailing slash. Create the key
once in the configured `TRALENS_CONFIG` directory, which defaults to the path below.

```sh
(umask 077; set -C; openssl rand -hex 32 > tralens-data/config/owner.key)
docker compose up -d --wait --pull never
```

Proxy HTTPS to `127.0.0.1:7871` and preserve the original `Host` header. Enter the
key at the HTTPS reader. Store it securely. Restart after replacing the key to
end existing sessions.

## Operating your server

Keep `tralens-data` on a local filesystem. It holds your library, job diagnostics,
credentials, runtime state, models, and optional assets. Stop the reader before
backing up the directory and `.env`. Protect backups and preserve ownership and
permissions. If you customized the data paths, back up those paths too.

For upgrades, save the current image digest and take a stopped backup. Set
`TRALENS_IMAGE` to the new release digest and repeat the pull, setup, and start
commands. To roll back, stop the reader and restore the pre-upgrade data into
empty directories before starting the original image. For startup problems, run
`docker compose ps` and `docker compose logs --tail=100 reader`.

OCR can miss handwritten text and sound effects. Translation and artwork cleanup vary by page. **Keep the artwork** favors preserving the background when cleanup is uncertain. Use **Compare** to check the result against the original.

Use content you have permission to process and share. Tralens is GPL-3.0-or-later.
[License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.txt)

Report reproducible issues in [GitHub Issues](https://github.com/nuuenkdaniel/tralens/issues).
Send vulnerability reports [privately](https://github.com/nuuenkdaniel/tralens/security/advisories/new).

## Release process

Merge tested PRs into `dev`, then promote `dev` to `main` by PR. Wait for
`clean-install` and `container / image`. Merges build separate private candidate
images tagged `tralens-dev:dev` and `tralens:main` on `ghcr.io/nuuenkdaniel`.

1. Pull the exact digest from the successful `main` run with GHCR read access.
   On a fresh Linux x86-64 host, test Compose setup, local translation, ChatGPT
   sign-in and translation over HTTPS, disconnect, a stopped backup and restore,
   and upgrade and rollback when there is a previous release. Review the exact
   image’s dependency notices and corresponding source before distribution.
2. Protect `dev`, `main`, and `v*` tags. Restrict the `public-beta` Actions
   environment to `main`, require a reviewer, and disable administrator bypass.
   Tag the approved `main` commit and run **Stage public beta** from that commit
   with the tag and candidate digest. Review the environment deployment.
3. After redistribution is cleared, make only the `tralens-release` package public.
   Run **Verify public beta image** with the same tag and digest. Confirm the
   draft’s Compose bundle pins that digest before publishing the prerelease.
