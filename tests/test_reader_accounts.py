"""Account lifecycle and server-enforced isolation through the reader HTTP API."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from PIL import Image

from tralens.devui.owner import Owner
from tralens.devui.reader_server import create_reader_server
from tralens.devui.server import App, Config


class AccountTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.app = App(Config(self.root, self.root, fonts=self.root / "fonts", reader_mode=True), runner=Mock(return_value={}))
        # Exercise existing owner-key installations in this suite.
        (self.root / "owner.key").write_text("k" * 48)
        (self.root / "owner.key").chmod(0o600)
        self.addCleanup(self.stop)
        self.start()

    def start(self, owner=None, registration=True):
        self.http = create_reader_server(self.app, 0, host="127.0.0.1", owner=owner, registration=registration)
        self.origin = owner.origin if owner else f"http://127.0.0.1:{self.http.server_port}"
        thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        thread.start()


    def call(self, path, data=None, cookie="", token="", origin=None, mime="application/json", download=""):
        conn = HTTPConnection("127.0.0.1", self.http.server_port, timeout=10)
        headers = {"Host": self.origin.split("://")[1], "Cookie": cookie, "Origin": self.origin if origin is None else origin,
                   "Content-Type": mime, "X-Tralens-Token": token, "X-Tralens-Download": download}
        try:
            conn.request("POST" if data is not None else "GET", path,
                         data if isinstance(data, bytes) else json.dumps(data) if data is not None else None, headers)
            response = conn.getresponse()
            return response.status, response.getheader("Set-Cookie", ""), response.read()
        finally:
            conn.close()

    def register(self, name):
        status, cookie, _ = self.call("/api/auth/register", {"username": name, "password": "a long test passphrase"})
        self.assertEqual(status, 204)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        cookie = cookie.split(";")[0]
        config = json.loads(self.call("/api/config", cookie=cookie)[2])
        return cookie, config["token"]

    def test_account_lifecycle_and_restart(self):
        cookie, token = self.register("Reader.One")
        data = {"username": "READER.ONE", "password": "a long test passphrase"}
        with self.http.library.db() as db:
            row = db.execute("SELECT * FROM accounts").fetchone()
            self.assertEqual(row["username"], "reader.one")
            self.assertEqual(len(row["salt"]), 16)
            self.assertEqual(len(row["verifier"]), 64)
        self.assertEqual(self.call("/api/auth/register", data)[0], 409)
        self.assertEqual(self.call("/api/auth/register", dict(data, username="owner"))[0], 409)
        self.assertEqual(self.call("/api/auth/login", dict(data, password="wrong"))[0], 401)
        self.assertEqual(self.call("/api/auth/logout", {}, cookie, origin="https://other.example")[0], 403)
        self.assertEqual(self.call("/api/auth/logout", {}, cookie)[0], 204)
        self.assertEqual(self.call("/api/config", cookie=cookie)[0], 401)
        self.stop()
        self.start()
        self.http.accounts.registration = False
        self.assertEqual(self.call("/api/auth/register", dict(data, username="newuser"))[0], 403)
        status, fresh, _ = self.call("/api/auth/login", data)
        self.assertEqual(status, 204)
        self.assertNotEqual(cookie, fresh.split(";")[0])

    def test_remembered_devices_renew_and_revoke_independently(self):
        first, _ = self.register("reader")
        status, header, _ = self.call("/api/auth/login", {"username": "reader", "password": "a long test passphrase"})
        self.assertEqual(status, 204)
        second = header.split(";")[0]
        self.assertIn("Max-Age=7776000", header)
        self.stop()
        self.start()
        self.assertEqual(self.call("/api/library", cookie=first)[:2], (200, ""))
        with self.http.library.db() as db:
            db.execute("UPDATE reader_sessions SET expires=?", (int(time.time()) + 100,))
        status, renewal, _ = self.call("/api/config", cookie=first)
        self.assertEqual(status, 200)
        self.assertTrue(renewal.startswith(first + ";"))
        self.assertIn("Max-Age=7776000", renewal)
        with self.http.library.db() as db:
            expiry = db.execute("SELECT MAX(expires) FROM reader_sessions").fetchone()[0]
            self.assertGreater(expiry, time.time() + 89 * 86400)
        status, deletion, _ = self.call("/api/auth/logout", {}, first)
        self.assertEqual(status, 204)
        self.assertIn("Max-Age=0", deletion)
        self.stop()
        self.start()
        self.assertEqual(self.call("/api/config", cookie=first)[0], 401)
        self.assertEqual(json.loads(self.call("/api/config", cookie=second)[2])["username"], "reader")

    def test_rotating_legacy_owner_key_invalidates_remembered_session(self):
        _, header, _ = self.call("/api/owner/login", (self.root / "owner.key").read_bytes())
        cookie = header.split(";")[0]
        self.assertEqual(self.call("/api/config", cookie=cookie)[0], 200)
        self.stop()
        (self.root / "owner.key").write_text("r" * 48)
        self.start()
        self.assertEqual(self.call("/api/config", cookie=cookie)[0], 401)

    def stop(self):
        # Keep the temporary library for the restart.
        self.http.shutdown()
        self.http.server_close()


    def test_private_library_reads_writes_and_operator_routes(self):
        alice, alice_token = self.register("alice")
        bob, bob_token = self.register("bob")
        self.assertNotEqual(alice_token, bob_token)
        image = BytesIO()
        Image.new("RGB", (18, 24), "white").save(image, "PNG")
        info = {"title": "Private", "chapter": "One", "options": {"mode": "render"}}
        def upload(metadata, cookie, token):
            body = (b'--qa\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n' +
                    json.dumps(metadata).encode() +
                    b'\r\n--qa\r\nContent-Disposition: form-data; name="page"; filename="page.png"\r\n'
                    b'Content-Type: image/png\r\n\r\n' + image.getvalue() + b'\r\n--qa--\r\n')
            return self.call("/api/library/add", body, cookie, token, mime="multipart/form-data; boundary=qa")
        status, _, created = upload(info, alice, alice_token)
        self.assertEqual(status, 201, created)
        book = json.loads(created)["book"]
        self.assertEqual(upload(dict(info, book=book), bob, bob_token)[0], 404)

        catalog = json.loads(self.call("/api/library", cookie=alice)[2])
        page = catalog["books"][0]["chapters"][0]["pages"][0]["id"]
        self.assertGreater(catalog["bytes"], 0)
        self.assertEqual(json.loads(self.call("/api/library", cookie=bob)[2]), {"books": [], "bytes": 0})
        for variant in ["original", "translated", "transcript"]:
            self.assertEqual(self.call(f"/api/library/pages/{page}/{variant}", cookie=bob)[0], 404)
        for variant in ["original", "translated", "both"]:
            self.assertEqual(self.call(f"/api/library/books/{book}/{variant}.zip", cookie=bob)[0], 404)
        self.assertEqual(self.call(f"/api/library/pages/{page}/original", cookie=alice)[0], 200)
        original = f"/api/library/pages/{page}/original"
        thumbnail = original + "/thumbnail"
        status, _, data = self.call(thumbnail, cookie=alice)
        self.assertEqual(status, 200)
        with Image.open(BytesIO(data)) as preview:
            self.assertEqual((preview.format, preview.size), ("WEBP", (18, 24)))
        self.assertEqual(self.call(thumbnail, cookie=bob)[0], 404)
        self.assertEqual(self.call(thumbnail)[0], 401)
        archive = f"/api/library/books/{book}/original.zip"
        issue = "/api/library/download-ticket"
        self.assertEqual(self.call(issue, {"path": original}, alice, bob_token)[0], 403)
        self.assertEqual(self.call(issue, {"path": "/api/config"}, alice, alice_token)[0], 400)
        def ticket(path, cookie, token):
            status, _, body = self.call(issue, {"path": path}, cookie, token)
            self.assertEqual(status, 200, body)
            return json.loads(body)["ticket"]
        grant = ticket(original, alice, alice_token)
        self.assertEqual(self.call(original, download=grant)[2], image.getvalue())
        self.assertEqual(self.call("/api/config", download=grant)[0], 401)
        self.assertEqual(self.call(archive, download=grant)[0], 401)
        self.assertEqual(self.call(original, download=ticket(original, bob, bob_token))[0], 404)
        with patch("tralens.devui.accounts.time.monotonic", return_value=int(grant.split(":")[1])):
            self.assertEqual(self.call(original, download=grant)[0], 401)
        self.http.archive_slot.acquire()
        try:
            self.assertEqual(self.call(archive, cookie=alice)[0], 429)
        finally:
            self.http.archive_slot.release()
        with patch("tralens.devui.import_pages.os.statvfs", return_value=Mock(f_bavail=0, f_frsize=4096)):
            self.assertEqual(self.call(archive, cookie=alice)[0], 507)
        self.assertEqual(self.call(archive, download=ticket(archive, alice, alice_token))[0], 200)
        self.assertEqual(self.call("/api/library/rename", {"book": []}, alice, alice_token)[0], 400)
        self.assertEqual(self.call("/api/library/position", {"book": book, "page": {}}, alice, alice_token)[0], 400)
        for action in ["rename", "position", "retry", "skip", "delete"]:
            self.assertEqual(self.call("/api/library/"+action, {"book": book, "page": page, "title": "Other"},
                                       bob, bob_token)[0], 404)
        self.assertEqual(self.call("/api/library/delete", {"book": book}, alice, bob_token)[0], 403)
        for path in ["/developer", "/api/jobs", "/api/chatgpt/models", "/api/chatgpt/status"]:
            self.assertEqual(self.call(path, cookie=bob)[0], 404)
        self.assertEqual(self.call("/api/chatgpt/connect", {}, bob, bob_token)[0], 404)
        self.assertEqual(self.call("/api/library/retry", {"book": book, "options":{"translation_model":"gpt-6-luna"}}, alice, alice_token)[0], 403)
        self.assertEqual(self.call("/api/library", cookie=bob)[0], 200)

        self.assertEqual(self.call("/api/auth/logout", {}, alice)[0], 204)
        self.assertEqual(self.call(original, download=grant)[0], 401)

    def test_password_screening_at_registration(self):
        username = "long.reader.name"
        for password in ("123456789987654321", "PaSsWoRdPaSsWoRd", "password123456789", username, " " * 20):
            status, cookie, body = self.call("/api/auth/register", {"username": username, "password": password})
            self.assertEqual(status, 400)
            self.assertFalse(cookie)
            self.assertIn("easy to guess", json.loads(body)["error"])
        data = {"username": username, "password": "Unrelated lanterns alongside 123456789987654321"}
        self.assertEqual(self.call("/api/auth/register", data)[0], 204)
        self.assertEqual(self.call("/api/auth/login", data)[0], 204)
        # Existing accounts retain access when registration policy changes.
        legacy = {"username": "legacy.reader", "password": "123456789987654321"}
        salt = bytes(range(16))
        with self.http.library.db() as db:
            db.execute("INSERT INTO accounts VALUES(?,?,?)",
                       (legacy["username"], salt, self.http.accounts.derive(legacy["password"].encode(), salt)))
        self.assertEqual(self.call("/api/auth/login", legacy)[0], 204)

    def fresh_server(self, **options):
        self.stop()
        self.root = self.root / "fresh"
        self.app = App(Config(self.root, self.root, reader_mode=True), runner=Mock(return_value={}))
        self.start(**options)

    def test_first_owner_is_atomic_and_libraries_stay_separate(self):
        self.fresh_server(registration=False)
        self.assertTrue(json.loads(self.call("/api/auth/settings")[2])["setup_required"])
        barrier = threading.Barrier(2)
        def claim(name):
            barrier.wait()
            return name, self.call("/api/auth/setup", {"username": name, "password": "a long test passphrase"})
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(claim, ("reader.one", "reader.two")))
        self.assertEqual(sorted(response[0] for _, response in attempts), [204, 409])
        name, (_, cookie, _) = next(attempt for attempt in attempts if attempt[1][0] == 204)
        owner = cookie.split(";")[0]
        self.assertEqual(json.loads(self.call("/api/config", cookie=owner)[2])["username"], name)
        with patch.object(self.http.chatgpt, "select", return_value={"state": "test"}) as connect:
            token = json.loads(self.call("/api/config", cookie=owner)[2])["token"]
            self.assertEqual(self.call("/api/chatgpt/connect", {}, owner, token)[0], 200)
            connect.assert_called_once()
        self.stop()
        self.start()
        self.assertFalse(json.loads(self.call("/api/auth/settings")[2])["setup_required"])
        self.assertEqual(json.loads(self.call("/api/config", cookie=owner)[2])["username"], name)
        status, cookie, _ = self.call("/api/auth/login", {"username": name, "password": "a long test passphrase"})
        self.assertEqual(status, 204)
        owner = cookie.split(";")[0]
        member, token = self.register("member")
        self.assertEqual(self.call("/api/chatgpt/connect", {}, member, token)[0], 404)
        with self.http.library.db() as db:
            db.executemany("INSERT INTO books(id,title,account) VALUES(?,?,?)",
                           [("a","Owner book","owner"), ("b","Member book","member")])
        for who, title in ((owner, "Owner book"), (member, "Member book")):
            books = json.loads(self.call("/api/library", cookie=who)[2])["books"]
            self.assertEqual([book["title"] for book in books], [title])

    def test_remote_setup_uses_single_use_private_link(self):
        owner = Owner("https://reader.example", None)
        self.fresh_server(owner=owner)
        path = self.root / "owner-setup.url"
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        link = path.read_text().strip()
        self.assertTrue(link.startswith(owner.origin + "/#setup="))
        data = {"username": "owner", "password": "a long test passphrase"}
        self.assertEqual(self.call("/api/auth/setup", data)[0], 403)
        self.assertEqual(self.call("/api/auth/register", dict(data, username="member"))[0], 409)
        self.stop()
        self.start(owner=owner)
        current = path.read_text().strip()
        self.assertNotEqual(current, link)
        data["setup_token"] = current.split("#setup=")[1]
        status, cookie, _ = self.call("/api/auth/setup", data)
        self.assertEqual(status, 204)
        self.assertIn("; Secure", cookie)
        self.assertFalse(path.exists())
        self.assertEqual(self.call("/api/auth/setup", data)[0], 409)
        self.assertEqual(self.call("/api/auth/login", data)[0], 204)

    def test_key_owner_is_already_configured(self):
        self.assertFalse(json.loads(self.call("/api/auth/settings")[2])["setup_required"])
        self.assertEqual(self.call("/api/auth/setup", {"username":"first", "password":"a long test passphrase"})[0], 409)

    def test_owner_token_rotates_between_sessions(self):
        password = (self.root / "owner.key").read_text().strip()
        def login():
            status, cookie, _ = self.call("/api/auth/login", {"username": "owner", "password": password})
            self.assertEqual(status, 204)
            cookie = cookie.split(";")[0]
            return cookie, json.loads(self.call("/api/config", cookie=cookie)[2])["token"]
        _, first = login()
        cookie, second = login()
        self.assertNotEqual(first, second)
        self.assertEqual(self.call("/api/library/rename", {}, cookie, first)[0], 403)

    def test_existing_books_stay_with_operator_after_schema_upgrade(self):
        self.stop()
        with sqlite3.connect(self.root / "reader.sqlite3") as db:
            db.execute("DROP INDEX book_account")
            db.execute("ALTER TABLE books DROP COLUMN account")
            db.execute("INSERT INTO books VALUES('legacy','Existing manga',0,0,NULL)")
        self.start()
        cookie, _ = self.register("new-reader")
        self.assertEqual(json.loads(self.call("/api/library", cookie=cookie)[2])["books"], [])
        password = (self.root / "owner.key").read_text().strip()
        status, cookie, _ = self.call("/api/auth/login", {"username":"owner", "password":password})
        self.assertEqual(status, 204)
        catalog = json.loads(self.call("/api/library", cookie=cookie.split(";")[0])[2])
        self.assertEqual([b["id"] for b in catalog["books"]], ["legacy"])

    def test_limits_and_expiry(self):
        cookie, _ = self.register("one")
        data = {"username": "one", "password": "wrong"}
        self.assertEqual(self.call("/api/auth/login", data, origin="https://other.example")[0], 403)
        for _ in range(12):
            self.assertEqual(self.call("/api/auth/login", data)[0], 401)
        self.assertEqual(self.call("/api/auth/login", data)[0], 429)
        self.assertEqual(self.http.accounts.attempts[("login", None)][0], 12)
        with self.http.library.db() as db:
            db.execute("UPDATE reader_sessions SET expires=0")
        self.assertEqual(self.call("/api/library", cookie=cookie)[0], 401)
