# Self-host the Tralens reader

Start with the [Docker Compose quick start](../README.md#quick-start) on Linux x86-64. This beta uses Linux host networking to reach the host's Ollama service and bind the reader and OAuth callback to loopback. Keep one reader per library. macOS, Windows, and ARM deployments have not been validated.

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

On Android, open the Tralens app, sign in as the server owner, and select a GPT model under **Settings → Translation**. Tap **Continue with ChatGPT**, finish in the phone browser, then tap **Open Tralens**. If you started in Android Chrome, **Open Tralens** takes you to the installed app. [Privacy notice](../PRIVACY.md)

For desktop browser sign-in, choose a GPT model and continue. If your browser runs on the server or reaches the reader over SSH, the OAuth callback still needs a loopback connection to the machine running Tralens. For a browser on the host, follow the link the reader shows. For a remote host, use a fixed callback port through SSH.

From your computer, open a dedicated SSH session and run the login command on the server.

```sh
ssh -L 1455:127.0.0.1:1455 user@server
cd /path/to/tralens
docker compose exec reader python backend/chatgpt.py login --no-browser --port 1455
```

Read the [privacy notice](../PRIVACY.md) before opening the printed private link in your local browser. Keep SSH open until consent finishes. Return to the reader and choose a model. If the port is in use, use the same alternate port for the SSH forward and login. A just-started sign-in in Settings may hold the OAuth lock until that attempt ends.

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

The image includes DejaVu Sans and Comic Neue. It uses the pinned PaddleOCR detector and recognizer under `/opt/tralens/ocr`. To use an external copy, set `TRALENS_OCR_ROOT=/models` and place the exports in `TRALENS_MODELS`. The optional `docker compose run --rm setup` command copies the bundled exports into that directory. To use a licensed custom static font, place it and its license under `tralens-data/assets` and point `TRALENS_FONTS` at its manifest under `/assets`. Use the [manifest format](../DEVELOPMENT.md#explicit-font-manifest), with container paths and hashes. Set `default_font_id` in a manifest with multiple faces. Classical inpainting runs on CPU. Auto uses the inpainting backends configured on the server. The base image supplies classical inpainting. GPU SDXL requires a compatible GPU runtime and local weights. The pinned LaMa checkpoint is supported for private noncommercial evaluation only. See the [model notices](../THIRD_PARTY_NOTICES.md#model-and-evaluation-assets) before adding checkpoints. More advanced startup arguments and a native developer install are in the [development guide](../DEVELOPMENT.md).
