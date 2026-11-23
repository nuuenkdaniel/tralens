# Tralens

Tralens is a self-hosted manga reader that translates Japanese pages into English. Import your own PNG, JPEG, WebP, PDF, CBZ, or ZIP files, read while pages translate, and switch between the original and translated pages.

## Quick start

Docker Compose is the recommended install. The beta image targets **Linux x86-64** with Docker Engine and Docker Compose v2. The reader uses CPU OCR. Ollama can use CPU or a supported GPU on the host. Start with one page to check speed and quality on your hardware.

1. On the Linux host, [install Ollama](https://docs.ollama.com/linux) and pull the default local translation model. The model download is about 3.3 GB.

   ```sh
   ollama pull translategemma:4b
   ```

   If you plan to use only ChatGPT, you can skip Ollama and choose a GPT model before importing pages.

2. Download the Compose file and make the persistent directories. Run the commands as the non-root user who will own your library.

   ```sh
   mkdir -p tralens && cd tralens
   curl -fL https://raw.githubusercontent.com/nuuenkdaniel/tralens/v0.1.0-beta.1/compose.yml -o compose.yml
   curl -fL https://raw.githubusercontent.com/nuuenkdaniel/tralens/v0.1.0-beta.1/.env.example -o .env.example
   cp .env.example .env
   printf 'TRALENS_UID=%s\nTRALENS_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
   install -d -m 700 tralens-data/library tralens-data/config \
     tralens-data/state tralens-data/models tralens-data/assets
   ```

3. Pull the image, download the pinned PaddleOCR models, and start the reader.

   ```sh
   docker compose pull reader
   docker compose run --rm setup
   docker compose up -d --wait
   ```

   Open [http://127.0.0.1:7871](http://127.0.0.1:7871) on the host. From another computer, use an SSH tunnel with `ssh -N -L 7871:127.0.0.1:7871 user@server`, then open that same URL. For HTTPS access, follow the [single-owner deployment guide](docs/self-hosted-reader.md#remote-access-over-https).

4. Choose a model under **Settings → Translation**, then **Add manga**. Try one page. The reader shows the original while the translation runs.

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
