"""Reader credentials and bounded, revocable server-side sessions."""
import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import json
import os
import re
import secrets
import sqlite3
import threading
import time

from .server import RequestError


def local_owner_digest(root):
    path = root / "owner.key"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w") as out:
            out.write(secrets.token_urlsafe(32) + "\n")
    if path.stat().st_mode & 0o077:
        raise ValueError("Set permissions on the reader owner.key to 0600.")
    key = path.read_bytes().strip()
    if not 32 <= len(key) <= 256:
        raise ValueError("Invalid reader owner.key.")
    return hashlib.sha256(key).digest()


def validate_new_password(password, username):
    from zxcvbn import zxcvbn
    from zxcvbn.matching import RANKED_DICTIONARIES

    if len(password) < 15:
        raise RequestError("Use at least 15 characters for your password.")
    candidate = password.strip().casefold()
    # A repeated phrase only needs to be guessed once. The first offset of an
    # exact match in the doubled string gives the length of its repeating unit.
    period = (candidate + candidate).find(candidate, 1)
    unit = candidate[:period]
    # Check the whole blocklist even for long values. Limit only the costlier
    # pattern estimation, so password managers can generate long passphrases.
    if (not unit or candidate in RANKED_DICTIONARIES["passwords"] or unit in {username, "tralens"} or
            len(unit) <= 32 and zxcvbn(unit)["guesses"] < 100_000):
        raise RequestError("This password is easy to guess. Try several unrelated words.")


class Accounts:
    lifetime = 7 * 86400

    def __init__(self, library, owner, registration=True):
        self.db = library.db
        self.owner_digest = owner.digest if owner else local_owner_digest(library.app.root)
        self.origin = owner.origin if owner else None
        self.cookie_name = "__Host-tralens" if owner else "tralens-local"
        self.registration = registration
        self.lock = threading.Lock()
        self.hash_slots = threading.BoundedSemaphore(2)
        self.sessions = {}
        self.download_key = secrets.token_bytes(32)
        self.attempts = {}
        self.dummy_salt = secrets.token_bytes(16)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS accounts(username TEXT PRIMARY KEY, salt BLOB NOT NULL, verifier BLOB NOT NULL)")

    @staticmethod
    def derive(password, salt):
        return hashlib.scrypt(password, salt=salt, n=2**15, r=8, p=3, maxmem=64*1024**2)

    def session_id(self, handler):
        cookie = SimpleCookie()
        try:
            cookie.load(handler.headers.get("Cookie", ""))
        except CookieError:
            return None
        value = cookie.get(self.cookie_name)
        return hashlib.sha256(value.value.encode()).digest() if value else None

    def identify(self, handler):
        with self.lock:
            entry = self.sessions.get(self.session_id(handler))
            if entry and entry[1] > time.monotonic():
                return entry[0]
        raise RequestError("Sign in to your server.", 401)

    def download_ticket(self, handler, path):
        # Bind the capability to one file and the issuing session. Logout and
        # server restart invalidate outstanding downloads.
        session = self.session_id(handler).hex()
        expiry = int(time.monotonic()) + 600
        payload = f"{session}:{expiry}"
        signature = hmac.digest(self.download_key, f"{payload}:{path}".encode(), "sha256").hex()
        return f"{payload}:{signature}"

    def download_account(self, ticket, path):
        match = re.fullmatch(r"([0-9a-f]{64}):([0-9]{1,16}):([0-9a-f]{64})", ticket)
        if match:
            session, expiry, signature = match.groups()
            expected = hmac.digest(self.download_key, f"{session}:{expiry}:{path}".encode(), "sha256").hex()
            if secrets.compare_digest(signature, expected):
                with self.lock:
                    entry = self.sessions.get(bytes.fromhex(session))
                    if entry and time.monotonic() < min(int(expiry), entry[1]):
                        return entry[0]
        raise RequestError("Download expired. Start it again from the reader.", 401)

    def limit(self, username, action):
        now = time.monotonic()
        with self.lock:
            self.attempts = {k:v for k,v in self.attempts.items() if v[1] > now}
            # Legacy and current sign-in share a budget, separate from enrollment.
            group = "register" if action == "register" else "login"
            pending = {}
            for key, maximum in (((group, username), 12), ((group, None), 120)):
                count, until = self.attempts.get(key, (0, now + 60))
                if count >= maximum:
                    raise RequestError("Too many attempts. Try again in a minute.", 429)
                pending[key] = (count + 1, until)
            self.attempts.update(pending)

    def respond(self, handler, username=None):
        old = self.session_id(handler)
        value = secrets.token_urlsafe(32) if username else ""
        with self.lock:
            now = time.monotonic()
            self.sessions = {k:v for k,v in self.sessions.items() if v[1] > now and k != old}
            if username:
                if len(self.sessions) >= 4096:
                    self.sessions.pop(next(iter(self.sessions)))
                self.sessions[hashlib.sha256(value.encode()).digest()] = (username, now + self.lifetime)
        handler.send_response(204)
        secure = "; Secure" if self.origin else ""
        handler.send_header("Set-Cookie", f"{self.cookie_name}={value}; HttpOnly; SameSite=Strict; Path=/"
                            f"; Max-Age={self.lifetime if username else 0}{secure}")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def handle(self, handler, action):
        origin = self.origin or "http://" + handler.headers["Host"]
        if handler.headers.get("Origin") != origin:
            raise RequestError("Open the reader on this server to continue.", 403)
        if action == "logout":
            self.identify(handler)
            return self.respond(handler)
        sizes = handler.headers.get_all("Content-Length") or []
        if handler.headers.get("Transfer-Encoding") or len(sizes) != 1 or not re.fullmatch(r"[0-9]{1,4}", sizes[0]) or not 0 < int(sizes[0]) <= 4096:
            raise RequestError("Invalid sign-in request.")
        raw = handler.rfile.read(int(sizes[0]))
        if len(raw) != int(sizes[0]):
            raise RequestError("Incomplete sign-in request.")
        if action == "owner":
            username, password = "owner", raw
        else:
            if handler.headers.get_content_type() != "application/json":
                raise RequestError("Use a JSON sign-in request.", 415)
            try:
                data = json.loads(raw)
                username = data["username"].strip().lower()
                password = data["password"].encode("utf-8")
            except (ValueError, KeyError, AttributeError, TypeError, RecursionError):
                raise RequestError("Invalid account details.") from None
        valid_name = re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,31}", username)
        if not valid_name or not 1 <= len(password) <= 1024:
            raise RequestError("Use a 3–32 character username and a password of at most 1024 bytes.")
        if action == "register" and not self.registration:
            raise RequestError("Account creation is closed on this server.", 403)
        self.limit(username, action)
        if not self.hash_slots.acquire(blocking=False):
            raise RequestError("Sign-in is busy. Try again shortly.", 429)
        try:
            if action == "register":
                validate_new_password(password.decode(), username)
                salt = secrets.token_bytes(16)
                verifier = self.derive(password, salt)
                try:
                    with self.db() as db:
                        if username == "owner":
                            raise sqlite3.IntegrityError
                        db.execute("INSERT INTO accounts VALUES(?,?,?)", (username, salt, verifier))
                except sqlite3.IntegrityError:
                    raise RequestError("Choose another username.", 409) from None
            elif username == "owner":
                if not secrets.compare_digest(hashlib.sha256(password).digest(), self.owner_digest):
                    raise RequestError("Incorrect username or password.", 401)
            else:
                with self.db() as db:
                    row = db.execute("SELECT salt,verifier FROM accounts WHERE username=?", (username,)).fetchone()
                actual = self.derive(password, row["salt"] if row else self.dummy_salt)
                if not secrets.compare_digest(actual, row["verifier"] if row else bytes(len(actual))) or not row:
                    raise RequestError("Incorrect username or password.", 401)
        finally:
            self.hash_slots.release()
        self.respond(handler, username)
