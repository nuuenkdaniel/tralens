"""Coordinate one owner-initiated sign-in across this reader's browser tabs."""
import json
import secrets
import time
import threading
import webbrowser

from ..translation import chatgpt
from ..translation.luna import MODELS
from .server import RequestError


class ReaderConnection:
    def __init__(self):
        self.lock = threading.Lock()
        self.mobile = None
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

    def available(self):
        if self.connected():
            try:
                available = chatgpt.models()
                return dict(state='connected', models=[model for model in MODELS if model in available])
            except chatgpt.SignInRequired:
                pass
            except (ValueError, OSError, RuntimeError) as error:
                raise RequestError('Could not check ChatGPT model availability. Try again shortly.', 503) from error

    def select_phone(self, port):
        if type(port) is not int or not 1024 <= port <= 65535:
            raise RequestError('Invalid phone sign-in port.')
        with self.lock:
            if self.pending:
                raise RequestError('Another sign-in is already underway.', 409)
        available = self.available()
        if available:
            return available
        with self.lock:
            if self.pending:
                raise RequestError('Another sign-in is already underway.', 409)
            with chatgpt.record() as old:
                state = secrets.token_urlsafe(48)
                url, redirect, nonce, verifier = chatgpt.authorization(old, port, state, include_hint=False)
                self.mobile = (state, time.monotonic() + 300, old.get('client_id'), redirect, nonce, verifier)
        return dict(state='authorize', url=url)

    def finish_phone(self, fields):
        if not isinstance(fields, dict) or not isinstance(fields.get('state'), str) or not fields['state'].isascii():
            raise RequestError('Invalid phone sign-in response.')
        with self.lock:
            if not self.mobile or not secrets.compare_digest(self.mobile[0], fields['state']):
                raise RequestError('This phone sign-in is no longer active.', 409)
            _, deadline, client, redirect, nonce, verifier = self.mobile
            self.mobile = None
            if time.monotonic() >= deadline:
                raise RequestError('Phone sign-in expired. Select the model to try again.', 409)
            try:
                with chatgpt.record() as old:
                    if old.get('client_id') != client:
                        raise RequestError('The connected account changed. Try selecting the model again.', 409)
                    chatgpt.finish_login(fields, old, redirect, verifier, nonce)
            except RequestError:
                raise
            except Exception as exc:
                raise RequestError('ChatGPT sign-in did not complete. Select the model to try again.') from exc
        return self.available() or dict(state='disconnected')

    def select(self, *, remote):
        with self.lock:
            if self.pending:
                return dict(state='pending', url=self.url)
        available = self.available()
        if available:
            return available
        with self.lock:
            if not self.pending:
                self.mobile = None
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
