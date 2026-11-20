"""Coordinate one owner-initiated sign-in across this reader's browser tabs."""
import json
import threading
import webbrowser

from ..translation import chatgpt
from ..translation.luna import MODELS
from .server import RequestError


class ReaderConnection:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = False
        self.url = None
        self.error = None

    def connected(self):
        # Writers atomically replace this file. Reading a snapshot lets local
        # jobs and other tabs keep working while the consent window is open.
        try:
            account = json.loads((chatgpt.directory() / 'account.json').read_text())
        except FileNotFoundError:
            return False
        return bool(account.get('refresh_token'))

    def status(self):
        with self.lock:
            if self.pending:
                return dict(state='pending', url=self.url)
            if self.error:
                return dict(state='failed', error=self.error)
        return dict(state='connected' if self.connected() else 'disconnected')

    def select(self, *, remote):
        with self.lock:
            if self.pending:
                return dict(state='pending', url=self.url)
        if self.connected():
            try:
                available = chatgpt.models()
                return dict(state='connected', models=[model for model in MODELS if model in available])
            except chatgpt.SignInRequired:
                pass
            except (ValueError, OSError, RuntimeError) as error:
                raise RequestError('Could not check ChatGPT model availability. Try again shortly.', 503) from error
        with self.lock:
            if not self.pending:
                self.pending, self.error, self.url = True, None, None
                threading.Thread(target=self._login, args=(remote,), daemon=True).start()
            return dict(state='pending', url=self.url)

    def _login(self, remote):
        def ready(url):
            with self.lock:
                self.url = url
            if remote:
                webbrowser.open(url)
        error = None
        try:
            chatgpt.login(on_ready=ready)
        except Exception:
            error = 'Sign-in did not finish. Select the model again to retry.'
        finally:
            with self.lock:
                self.pending, self.error, self.url = False, error, None
