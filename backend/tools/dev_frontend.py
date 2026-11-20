"""Run the loopback-by-default pipeline development viewer. No new dependencies."""
import argparse
import ipaddress
import os
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND/'src'))


def configured_endpoint(explicit=None):
    """Respect the operator's Ollama client address without probing/fallbacks."""
    from tralens.translation.runtime import local_endpoint
    from urllib.parse import urlsplit
    if explicit is not None:
        return local_endpoint(explicit)
    host = os.environ.get('OLLAMA_HOST', '').strip()
    if not host:
        return 'http://127.0.0.1:11434'
    if '://' in host:
        return local_endpoint(host)
    try:
        address = ipaddress.ip_address(host)
        if address.version == 6:
            host = '['+host+']'
    except ValueError:
        pass
    endpoint = local_endpoint('http://'+host)
    return endpoint if urlsplit(endpoint).port is not None else endpoint+':11434'


def main(argv=None, *, reader=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IPv4 bind address; use 0.0.0.0 only on trusted networks (no user authentication)')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--output-root', type=Path, default=BACKEND.parent/'test-results'/('web-reader' if reader else 'dev-frontend'))
    parser.add_argument('--model-root', type=Path, default=Path.home()/'.cache/tralens/paddle-v6-models',
                        help='existing v6 medium exports (default: ~/.cache/tralens/paddle-v6-models); never downloaded')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fonts', type=Path, help='explicit licensed font manifest; enables rendering')
    parser.add_argument('--font-id')
    parser.add_argument('--model', default='translategemma:27b')
    parser.add_argument('--profile', choices=('generalist', 'translategemma'), default='translategemma')
    parser.add_argument('--endpoint', help='explicit local Ollama origin; default: OLLAMA_HOST, otherwise http://127.0.0.1:11434')
    parser.add_argument('--manga-reread-model', type=Path,
                        help='existing local Manga OCR snapshot; enables the optional quality-reread control, never downloads')
    parser.add_argument('--reread-model', default='qwen3.8:latest', help='installed local vision tag for opt-in quality rereads')
    parser.add_argument('--reconstruction-config',type=Path,help='explicit approved local LaMa/runtime configuration; no downloads')
    parser.add_argument('--sdxl-config', type=Path, help='local SDXL inpainting weights and GPU runtime configuration')
    parser.add_argument('--crop-reader-model',type=Path,
                        help='existing Hayai snapshot for source-bound crop readings in coverage-first renders')
    parser.add_argument('--text-mask-model',type=Path,help='existing pinned CTD checkpoint for source-bound cleanup; no downloads')
    if reader:
        parser.add_argument('--owner-key-file', type=Path)
        parser.add_argument('--public-origin', help='Public HTTPS origin served by a local reverse proxy')
        parser.add_argument('--frontier-provider', choices=('codex', 'chatgpt'), default='chatgpt')
    args = parser.parse_args(argv)
    reader_options = {}
    provider = 'chatgpt' if args.host == '127.0.0.1' else 'disabled'
    if reader:
        if bool(args.public_origin) != bool(args.owner_key_file):
            parser.error('--owner-key-file and --public-origin must be supplied together')
        if args.public_origin:
            from tralens.devui.owner import Owner
            if args.host != '127.0.0.1':
                parser.error('Use loopback binding behind the HTTPS reverse proxy.')
            try:
                reader_options['owner'] = Owner(args.public_origin, args.owner_key_file)
            except (OSError, ValueError) as error:
                parser.error(str(error))
        provider = args.frontier_provider
        if provider == 'chatgpt' and not args.public_origin and args.host != '127.0.0.1':
            provider = 'disabled'
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1–65535')
    try:
        host = str(ipaddress.IPv4Address(args.host))
        endpoint = configured_endpoint(args.endpoint)
        if args.crop_reader_model is not None:
            from tralens.ocr.hayai_adapter import verify_snapshot
            verify_snapshot(args.crop_reader_model)
        if args.manga_reread_model is not None:
            from main import quality_refiner
            # Check local files without importing readers or running inference.
            quality_refiner(endpoint=endpoint, model=args.reread_model, manga_model=args.manga_reread_model)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    from tralens.devui.server import App, Config, create_server
    if reader:
        from tralens.devui.reader_server import create_reader_server as create_server
    app = App(Config(root=args.output_root, model_root=args.model_root.expanduser().absolute(),
        device=args.device, fonts=args.fonts.expanduser().absolute() if args.fonts else None,
        font_id=args.font_id, model=args.model, profile=args.profile, endpoint=endpoint,
        manga_reread_model=args.manga_reread_model.expanduser().absolute() if args.manga_reread_model else None,
        reread_model=args.reread_model,reconstruction_config=args.reconstruction_config,
        crop_reader_model=args.crop_reader_model.expanduser().absolute() if args.crop_reader_model else None,
        text_mask_model=args.text_mask_model.expanduser().absolute() if args.text_mask_model else None,isolated_workers=True,
        sdxl_config=args.sdxl_config,reader_mode=reader,frontier_provider=provider))
    with create_server(app, args.port, host=host, **reader_options) as server:
        print(f'{"Tralens reader" if reader else "Pipeline viewer"}: http://{host}:{args.port}', flush=True)
        if host != '127.0.0.1':
            print('Network access enabled: reachable clients can inspect artifacts and start runs. '
                  'Use only trusted networks; this development viewer has no user authentication.', flush=True)
        print(f'Ollama endpoint: {endpoint}', flush=True)
        if args.fonts and args.text_mask_model is None:
            print('Warning: no --text-mask-model. Replace all text will erase whole OCR boxes, '
                  'not just the lettering, and can leave large blank areas.', flush=True)
        instructions = 'Add manga in the reader to queue pages. Saved queued pages resume on startup.' if reader else 'No inference until you upload and click Run.'
        print(f'Job artifacts: {app.root}\n{instructions} Ctrl+C stops the server.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
