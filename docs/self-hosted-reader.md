# One-owner Tralens server

The owner signs into the reader with a server key. Web and Android clients share the library and the server's translation account. The key grants full control of that library and its jobs. ChatGPT authorization is a separate, one-time operation on the server desktop.

## Prepare the reader

Use the existing Linux OCR and rendering environment. Supply the OCR weights and a licensed font manifest. Local translation requires Ollama and its installed model. LaMa and SDXL require their configured weights and runtime. PDF import needs Poppler and util-linux. Back up the entire output directory while the service is stopped. Use one reader process per output directory.

Generate the owner key once under the Linux user that will run the service. Copy its value into your password manager. Generating a new key and restarting the service invalidates all existing sessions.

```sh
install -d -m 700 ~/.config/tralens
(umask 077; set -C; openssl rand -hex 32 > ~/.config/tralens/owner.key)
```

## Connect ChatGPT directly

Start the reader and open **Settings → Translation model**. Local translation works immediately. Choose a GPT model and select **Continue with ChatGPT**. After consent, the reader checks model availability and applies your selection. Closing the dialog keeps your previous model selected.

For a phone or a remote browser, finish consent in the system browser on the server desktop. For a headless server, the reader shows the loopback link and the port to forward over SSH. All readers of this single-owner server share the resulting connection.

The standard backend requirements include OAuth support. For an existing environment, or to connect from the command line:

```sh
uv pip install --python .venv-paddle6/bin/python -r backend/requirements-chatgpt.txt
.venv-paddle6/bin/python backend/chatgpt.py login
.venv-paddle6/bin/python backend/chatgpt.py models
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
.venv-paddle6/bin/python backend/serve.py \
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
