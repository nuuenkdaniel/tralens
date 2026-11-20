"""Single-owner reader behind an HTTPS reverse proxy on the same host."""
import hashlib
from http.cookies import SimpleCookie, CookieError
import secrets
import threading
import time
from urllib.parse import urlsplit

from .server import RequestError


class Owner:
    def __init__(self, origin, key_path):
        url = urlsplit(origin)
        if url.scheme != 'https' or not url.hostname or url.username is not None or origin != 'https://' + url.netloc:
            raise ValueError('Set the exact public HTTPS origin without a trailing slash.')
        if key_path.stat().st_mode & 0o077:
            raise ValueError('The owner key file must be private (mode 0600).')
        key = key_path.read_bytes().strip()
        if not 32 <= len(key) <= 256 or not key.isascii():
            raise ValueError('Use a random owner key of 32–256 ASCII characters.')
        self.digest = hashlib.sha256(key).digest()
        self.origin = origin
        self.host = url.netloc
        self.sessions = {}
        self.lock = threading.Lock()

    def authorize(self, handler):
        cookie = SimpleCookie()
        try:
            cookie.load(handler.headers.get('Cookie', ''))
        except CookieError:
            pass
        item = cookie.get('__Host-tralens')
        with self.lock:
            if not item or self.sessions.get(item.value, 0) <= time.monotonic():
                raise RequestError('Sign in to your Tralens server.', 401)
        return item.value

    def logout(self, handler):
        session = self.authorize(handler)
        with self.lock:
            self.sessions.pop(session, None)
        self.cookie(handler, '', 0)

    def login(self, handler):
        if handler.headers.get('Origin') != self.origin:
            raise RequestError('Sign in from the reader.', 403)
        lengths = handler.headers.get_all('Content-Length') or []
        if (handler.headers.get('Transfer-Encoding') or len(lengths) != 1 or
                not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0]) > 3
                or not 1 <= int(lengths[0]) <= 256):
            raise RequestError('Invalid owner key.', 400)
        supplied = handler.rfile.read(int(lengths[0]))
        if not secrets.compare_digest(hashlib.sha256(supplied).digest(), self.digest):
            raise RequestError('Incorrect owner key.', 401)
        session = secrets.token_urlsafe(48)
        lifetime = 7 * 86400
        with self.lock:
            if len(self.sessions) >= 64:
                self.sessions.pop(next(iter(self.sessions)))
            self.sessions[session] = time.monotonic() + lifetime
        self.cookie(handler, session, lifetime)

    def cookie(self, handler, session, lifetime):
        handler.send_response(204)
        handler.send_header('Set-Cookie', '__Host-tralens=' + session +
                            f'; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age={lifetime}')
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('Content-Length', '0')
        handler.end_headers()
