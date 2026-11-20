# One-owner Tralens server

The owner signs into the reader with a server key. Web and Android clients share the library and the server's translation account. The key grants full control of that library and its jobs. The owner can authorize ChatGPT from the desktop reader or Android app.

## Prepare the reader

From a fresh Linux x86-64 checkout with uv installed, create the pinned Python 3.10 environment. The default runtime uses CPU OCR. Existing GPU deployments can keep their separate GPU environment. LaMa and SDXL use their configured weights and runtimes independently.

```sh
uv sync --locked --python 3.10
```

Download the medium Japanese-capable OCR exports once into the model directory. These are the revisions used in the release check.

```sh
models="$HOME/.cache/tralens/paddle-v6-models"
.venv/bin/hf download PaddlePaddle/PP-OCRv6_medium_det --revision 8e0f56fb2ef86b461d99cfc7ac5c137738985f61 --local-dir "$models/PP-OCRv6_medium_det"
.venv/bin/hf download PaddlePaddle/PP-OCRv6_medium_rec --revision e5a92bcbc5cc1b494628e458d267778f0704fd7c --local-dir "$models/PP-OCRv6_medium_rec"
```

Prepare a [licensed font manifest](../README.md#explicit-font-manifest) for lettering. Local translation uses the configured Ollama service and installed model (default `translategemma:27b`). PDF import needs Poppler and util-linux. For a local test, start the reader and visit `http://127.0.0.1:7871`:

```sh
.venv/bin/python backend/serve.py --port 7871 --fonts /path/to/fonts.json
```

Use one reader process per output directory. Back up the entire output directory while the service is stopped.

Generate the owner key once under the Linux user that will run the service. Copy its value into your password manager. Generating a new key and restarting the service invalidates all existing sessions.

```sh
install -d -m 700 ~/.config/tralens
(umask 077; set -C; openssl rand -hex 32 > ~/.config/tralens/owner.key)
```

## Connect ChatGPT directly

Start the reader and open **Settings → Translation model**. Local translation works immediately. Choose a GPT model and select **Continue with ChatGPT**. After consent, the reader checks model availability and applies your selection. Closing the dialog keeps your previous model selected.

In the Android app, select a GPT model, approve access in the phone’s browser, and tap **Open Tralens**. The server exchanges the one-time code and stores the credentials. Use an owner-protected HTTPS connection for network access. For a remote web browser outside the Android app, finish consent on the server desktop. For a headless server, the reader shows the loopback link and the port to forward over SSH. All readers of this single-owner server share the resulting connection.

The standard backend requirements include OAuth support. For an existing environment, or to connect from the command line:

```sh
uv pip install --python .venv/bin/python -r backend/requirements-chatgpt.txt
.venv/bin/python backend/chatgpt.py login
.venv/bin/python backend/chatgpt.py models
```

Your system browser opens OpenAI's consent screen and returns to a temporary loopback listener. The server stores renewable credentials in its private configuration directory under `~/.config/tralens/chatgpt` (or `$XDG_CONFIG_HOME/tralens/chatgpt`). Use `backend/chatgpt.py status` to check the connection and `backend/chatgpt.py logout` to revoke it. For a headless server, connect from your desktop with `ssh -L 1455:127.0.0.1:1455 user@server`. In that SSH session run `backend/chatgpt.py login --no-browser --port 1455` with the backend Python interpreter. Open the printed private loopback link on the desktop and leave the SSH session connected through sign-in. Usage and limits are available in [ChatGPT settings](https://chatgpt.com/settings/usage).

## Serve over HTTPS

Point your DNS name at a TLS reverse proxy on the server. Configure its certificate for that name and forward the original Host header. For example, this Caddyfile serves the reader on standard HTTPS.

```caddyfile
manga.example.com {
    reverse_proxy 127.0.0.1:7871
}
```

Stop any existing reader using the same library before switching deployments. Start the backend with the same model, font and output paths as your existing service. Substitute your domain and actual paths below.

```sh
.venv/bin/python backend/serve.py \
  --host 127.0.0.1 --port 7871 \
  --public-origin https://manga.example.com \
  --owner-key-file ~/.config/tralens/owner.key \
  --output-root /path/to/library-and-jobs \
  --model-root /path/to/ocr-models --fonts /path/to/fonts.json \
  --frontier-provider chatgpt
```

Open the HTTPS site and enter the owner key. For Android, enter the same hostname, select HTTPS and port 443. Sessions expire after seven days or a reader restart. Settings offers sign-out for the current device. The reverse proxy should expose only the owner-mode reader. Keep any separate diagnostic servers and Ollama listeners private.

The ChatGPT provider checks account model availability when a GPT model is selected and before inference. Both the missed-text scan and the translation/review stages use the selected provider. The existing Codex transport can be selected explicitly with `--frontier-provider codex`. Local jobs continue to use the configured local models.

See OpenAI’s [registration flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in) and [model access](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).

## Docker Compose on Linux

Use Docker Compose on an x86-64 Linux host. Build the reader with the repository’s `Dockerfile` and run it with `compose.yml`. The image includes CPU OCR and the runtime for optional ONNX LaMa inpainting.

Set these values in an untracked `.env` alongside the Compose file. Use absolute paths on the host and make the writable directories accessible to the configured UID and GID.

```dotenv
TRALENS_UID=1000
TRALENS_GID=1000
TRALENS_PORT=7871
TRALENS_ORIGIN=https://manga.example.com
TRALENS_LIBRARY=/srv/tralens/library
TRALENS_CONFIG=/srv/tralens/config
TRALENS_STATE=/srv/tralens/state
TRALENS_MODELS=/srv/tralens/models
TRALENS_ASSETS=/srv/tralens/assets
TRALENS_OWNER_KEY=/srv/tralens/owner.key
TRALENS_OLLAMA=http://127.0.0.1:11434
```

Put the two OCR exports inside the models directory. Put `fonts.json`, a licensed font, and its license inside the assets directory. In the manifest, use container paths such as `/assets/dialogue.ttf` and `/assets/font-license.txt`. The default Compose command expects a single font face in the manifest. Keep the owner key at mode 0600. Stop any previous reader using the same port or library before starting Compose.

```sh
docker compose up -d --build
docker compose logs --tail=50 reader
```

The container uses Linux host networking so the reader can bind to host loopback. Point your HTTPS proxy at `127.0.0.1` and the chosen port. For Tailscale Serve, use the HTTPS hostname for `TRALENS_ORIGIN` and the Android app’s server address. Keep the library, configuration, and runtime state across container replacements. Share the runtime state directory with other Tralens instances using the same Ollama service so they share the inference lock.

To enable LaMa, mount its checkpoint read-only and supply a reconstruction configuration that names `/app/.venv/bin/python` as the runtime. Pass `--reconstruction-config` to the reader using a Compose command override. SDXL needs its own CUDA-enabled runtime.
