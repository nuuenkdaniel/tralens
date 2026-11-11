"""Run the minimal, local-only pipeline development viewer. No new dependencies."""
import argparse
import ipaddress
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--output-root', type=Path, default=ROOT/'test-results/dev-frontend')
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
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1–65535')
    try:
        endpoint = configured_endpoint(args.endpoint)
        if args.manga_reread_model is not None:
            from main import quality_refiner
            # Check local files without importing readers or running inference.
            quality_refiner(endpoint=endpoint, model=args.reread_model, manga_model=args.manga_reread_model)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    from tralens.devui.server import App, Config, create_server
    app = App(Config(root=args.output_root, model_root=args.model_root.expanduser().absolute(),
        device=args.device, fonts=args.fonts.expanduser().absolute() if args.fonts else None,
        font_id=args.font_id, model=args.model, profile=args.profile, endpoint=endpoint,
        manga_reread_model=args.manga_reread_model.expanduser().absolute() if args.manga_reread_model else None,
        reread_model=args.reread_model,reconstruction_config=args.reconstruction_config,isolated_workers=True))
    with create_server(app, args.port) as server:
        print(f'Pipeline viewer: http://127.0.0.1:{args.port}', flush=True)
        print(f'Ollama endpoint: {endpoint}', flush=True)
        print(f'Job artifacts: {app.root}\nNo inference until you upload and click Run. Ctrl+C stops the server.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
