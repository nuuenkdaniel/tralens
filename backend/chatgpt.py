"""Manage the server owner's ChatGPT connection on the machine running Tralens."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from tralens.translation import chatgpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('login', 'status', 'models', 'logout'))
    parser.add_argument('--no-browser', action='store_true', help='Print a private loopback sign-in link')
    parser.add_argument('--port', type=int, default=0, help='Loopback callback port for SSH forwarding')
    args = parser.parse_args()
    action = args.action
    if args.port < 0 or args.port > 65535 or (action != 'login' and (args.port or args.no_browser)):
        parser.error('Use --port 1–65535 and --no-browser only with login.')
    if action == 'status':
        with chatgpt.record() as account:
            print('Connected as ' + (account.get('email') or account['subject'])
                  if account.get('refresh_token') else 'Disconnected')
    elif action == 'models':
        print(*chatgpt.models(), sep='\n')
    else:
        try:
            if action == 'login':
                chatgpt.login(browser=not args.no_browser, port=args.port)
            else:
                chatgpt.logout()
        except (RuntimeError, OSError, ValueError) as exc:
            parser.exit(1, str(exc) + '\n')
        print('Connected.' if action == 'login' else 'Disconnected.')
