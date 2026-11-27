"""Reader credentials and bounded, revocable server-side sessions."""
import hashlib
import hmac
from http.cookies import CookieError, SimpleCookie
import json
import os
import re
import secrets
import sqlite3
import tempfile
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
    lifetime = 90 * 86400

    def __init__(self, library, owner, registration=True, *, fresh=False):
        self.db = library.db
        self.origin = owner.origin if owner else None
        self.cookie_name = "__Host-tralens" if owner else "tralens-local"
        self.registration = registration
        self.lock = threading.Lock()
        self.hash_slots = threading.BoundedSemaphore(2)
        self.download_key = secrets.token_bytes(32)
        self.attempts = {}
        self.dummy_salt = secrets.token_bytes(16)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS accounts(username TEXT PRIMARY KEY, salt BLOB NOT NULL, verifier BLOB NOT NULL)")

            db.execute("""CREATE TABLE IF NOT EXISTS reader_owner(
                slot INTEGER PRIMARY KEY CHECK(slot=1), username TEXT REFERENCES accounts(username))""")
            db.execute("""CREATE TABLE IF NOT EXISTS reader_sessions(
                digest BLOB PRIMARY KEY, account TEXT NOT NULL, expires INTEGER NOT NULL)""")
            if fresh and (owner is None or owner.digest is None):
                db.execute("INSERT INTO reader_owner(slot) VALUES(1)")
            # A missing singleton is an existing key-owned installation. A NULL
            # username reserves first-time setup until the owner chooses credentials.
            named_setup = db.execute("SELECT 1 FROM reader_owner").fetchone() is not None
        self.owner_digest = None
        if not named_setup:
            if owner and owner.digest is None:
                raise ValueError("Restore the configured owner key for this existing HTTPS reader.")
            self.owner_digest = owner.digest if owner else local_owner_digest(library.app.root)

        self.setup_link = library.app.root / "owner-setup.url"
        self.setup_token = None
        if self.origin and self.setup_pending():
            self.setup_token = secrets.token_urlsafe(32)
            with tempfile.NamedTemporaryFile(mode="w", dir=library.app.root, delete=False) as link:
                link.write(f"{self.origin}/#setup={self.setup_token}\n")
            os.replace(link.name, self.setup_link)
        else:
            self.setup_link.unlink(missing_ok=True)

    def setup_pending(self):
        with self.db() as db:
            return db.execute("SELECT 1 FROM reader_owner WHERE username IS NULL").fetchone() is not None

    def owner_username(self):
        with self.db() as db:
            row = db.execute("SELECT username FROM reader_owner").fetchone()
        return row[0] if row else "owner"

    @staticmethod
    def derive(password, salt):
        return hashlib.scrypt(password, salt=salt, n=2**15, r=8, p=3, maxmem=64*1024**2)

    def cookie_value(self, handler):
        cookie = SimpleCookie()
        try:
            cookie.load(handler.headers.get("Cookie", ""))
        except CookieError:
            return None
        value = cookie.get(self.cookie_name)
        return value.value if value and re.fullmatch(r"[A-Za-z0-9_-]{43}", value.value) else None

    def digest(self, value):
        # Keep grants scoped to this origin and invalidate legacy grants on key rotation.
        payload = ((self.origin or "local") + "\0" + value).encode()
        return hmac.digest(self.owner_digest or b"", payload, "sha256")

    def session_id(self, handler):
        value = self.cookie_value(handler)
        return self.digest(value) if value else None

    def session_account(self, digest, handler=None):
        if digest is None:
            return None
        now = int(time.time())
        with self.db() as db:
            row = db.execute("SELECT account, expires FROM reader_sessions WHERE digest=? AND expires>?",
                             (digest, now)).fetchone()
            if row and handler and row["expires"] <= now + self.lifetime - 86400:
                # Extend active sessions at most once a day. Never recreate a revoked grant.
                db.execute("UPDATE reader_sessions SET expires=? WHERE digest=? AND expires=?",
                           (now + self.lifetime, digest, row["expires"]))
                handler.renewed_cookie = self.cookie_header(self.cookie_value(handler))
        return row["account"] if row else None

    def identify(self, handler):
        account = self.session_account(self.session_id(handler), handler)
        if account:
            return account
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
            if secrets.compare_digest(signature, expected) and time.monotonic() < int(expiry):
                account = self.session_account(bytes.fromhex(session))
                if account:
                    return account
        raise RequestError("Download expired. Start it again from the reader.", 401)

    def limit(self, username, action):
        now = time.monotonic()
        with self.lock:
            self.attempts = {k:v for k,v in self.attempts.items() if v[1] > now}
            # Legacy and current sign-in share a budget, separate from enrollment.
            group = "register" if action in ("register", "setup") else "login"
            pending = {}
            for key, maximum in (((group, username), 12), ((group, None), 120)):
                count, until = self.attempts.get(key, (0, now + 60))
                if count >= maximum:
                    raise RequestError("Too many attempts. Try again in a minute.", 429)
                pending[key] = (count + 1, until)
            self.attempts.update(pending)

    def cookie_header(self, value):
        secure = "; Secure" if self.origin else ""
        return (f"{self.cookie_name}={value}; HttpOnly; SameSite=Strict; Path=/"
                f"; Max-Age={self.lifetime if value else 0}{secure}")

    def respond(self, handler, username=None):
        value = secrets.token_urlsafe(32) if username else ""
        now = int(time.time())
        with self.db() as db:
            db.execute("DELETE FROM reader_sessions WHERE digest=? OR expires<=?", (self.session_id(handler), now))
            if username:
                db.execute("INSERT INTO reader_sessions VALUES(?,?,?)", (self.digest(value), username, now + self.lifetime))
                db.execute("""DELETE FROM reader_sessions WHERE rowid IN
                    (SELECT rowid FROM reader_sessions ORDER BY expires DESC, rowid DESC LIMIT -1 OFFSET 4096)""")
        handler.send_response(204)
        handler.send_header("Set-Cookie", self.cookie_header(value))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def handle(self, handler, action):
        origin = self.origin or "http://" + handler.headers["Host"]
        if handler.headers.get("Origin") != origin:
            raise RequestError("Open the reader on this server to continue.", 403)
        if action == "logout":
            if not self.session_account(self.session_id(handler)):
                raise RequestError("Sign in to your server.", 401)
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
            if action in ("register", "setup"):
                if action == "setup" and self.origin:
                    token = data.get("setup_token")
                    if not self.setup_token or not isinstance(token, str) or not secrets.compare_digest(token.encode(), self.setup_token.encode()):
                        raise RequestError("Open the private setup link from your server.", 403)
                validate_new_password(password.decode(), username)
                salt = secrets.token_bytes(16)
                verifier = self.derive(password, salt)
                try:
                    with self.db() as db:
                        db.execute("BEGIN IMMEDIATE")
                        pending = db.execute("SELECT 1 FROM reader_owner WHERE username IS NULL").fetchone()
                        if action == "setup" and not pending:
                            raise RequestError("The owner is already configured. Sign in to continue.", 409)
                        if action == "register" and pending:
                            raise RequestError("Finish setting up the server owner first.", 409)
                        if action == "register" and username == "owner":
                            raise sqlite3.IntegrityError
                        db.execute("INSERT INTO accounts VALUES(?,?,?)", (username, salt, verifier))
                        if action == "setup":
                            db.execute("UPDATE reader_owner SET username=? WHERE slot=1", (username,))
                    if action == "setup":
                        self.setup_link.unlink(missing_ok=True)
                        username = "owner"
                except sqlite3.IntegrityError:
                    raise RequestError("Choose another username.", 409) from None
            elif action == "owner" or (username == "owner" and self.owner_digest is not None):
                if self.owner_digest is None or not secrets.compare_digest(hashlib.sha256(password).digest(), self.owner_digest):
                    raise RequestError("Incorrect username or password.", 401)
            else:
                with self.db() as db:
                    row = db.execute("""SELECT salt, verifier, reader_owner.slot AS owner_slot FROM accounts
                        LEFT JOIN reader_owner USING(username) WHERE accounts.username=?""", (username,)).fetchone()
                actual = self.derive(password, row["salt"] if row else self.dummy_salt)
                if not secrets.compare_digest(actual, row["verifier"] if row else bytes(len(actual))) or not row:
                    raise RequestError("Incorrect username or password.", 401)
                if row["owner_slot"] is not None:
                    username = "owner"
        finally:
            self.hash_slots.release()
        self.respond(handler, username)
