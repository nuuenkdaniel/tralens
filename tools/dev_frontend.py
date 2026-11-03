"""Run the minimal, local-only pipeline development viewer. No new dependencies."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


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
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1–65535')
    from tralens.devui.server import App, Config, create_server
    app = App(Config(root=args.output_root, model_root=args.model_root.expanduser().absolute(),
        device=args.device, fonts=args.fonts.expanduser().absolute() if args.fonts else None,
        font_id=args.font_id, model=args.model, profile=args.profile, endpoint=args.endpoint))
    with create_server(app, args.port) as server:
        print(f'Pipeline viewer: http://127.0.0.1:{args.port}', flush=True)
        print(f'Job artifacts: {app.root}\nNo inference until you upload and click Run. Ctrl+C stops the server.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
