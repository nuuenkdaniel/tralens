"""Serve the independent web client and its library under the job API's origin."""

import hmac
import json
import mimetypes
from pathlib import Path
import re
import secrets
import shutil
import threading
from urllib.parse import quote, urlsplit

from .accounts import Accounts
from .library import Library
from .multipart_upload import MAX_IMPORT_BYTES, read_import
from .server import Handler, RequestError, STATIC, ThreadingHTTPServer

PROJECT = Path(__file__).resolve().parents[4]
WEB = PROJECT / "frontend/web"
ID = "[0-9a-f]{32}"


DOWNLOAD_PATH = re.compile(rf"/api/library/(?:books/{ID}/(?:original|translated|both)\.zip|pages/{ID}/(?:original|translated))")


class ReaderHandler(Handler):
    def end_headers(self):
        if getattr(self, "renewed_cookie", None):
            self.send_header("Set-Cookie", self.renewed_cookie)
            self.renewed_cookie = None
        super().end_headers()

    def request_token(self):
        return hmac.digest(self.server.app.token.encode(), self.server.accounts.session_id(self), "sha256").hex()

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/auth/discovery":
                self.local_request(discovery=True)
                return self.respond(200, b'{"service":"tralens-reader","version":1}', "application/json", discovery=True)
            self.local_request()
            if path == "/verify-server.js":
                return self.respond(200, (WEB / "verify-server.js").read_bytes(), "text/javascript", probe_worker=True)
            if path == "/healthz":
                return self.json(200, {"status": "ok"})
            if path == "/privacy":
                return self.respond(200, (PROJECT / "PRIVACY.txt").read_bytes(), "text/plain; charset=utf-8")
            if path in (
                "/",
                "/reader.js",
                "/offline.js",
                "/reader.css",
            ):
                file = WEB / ("index.html" if path == "/" else path[1:])
                self.respond(200, file.read_bytes(), mimetypes.guess_type(file.name)[0])
            elif path == "/api/auth/settings":
                accounts = self.server.accounts
                self.json(200, {"registration": accounts.registration, "setup_required": accounts.setup_pending(),
                                "setup_link_required": bool(accounts.origin)})
            else:
                ticket = self.headers.get("X-Tralens-Download")
                if ticket and DOWNLOAD_PATH.fullmatch(path):
                    self.account = self.server.accounts.download_account(ticket, path)
                else:
                    self.account = self.server.accounts.identify(self)
                return self._private_get(path)
        except RequestError as exc:
            self.json(exc.status, dict(error=str(exc)))

    def _private_get(self, path):
        try:
            if path == "/api/config":
                config = self.server.app.public_config()
                config['owner_mode'] = bool(self.server.owner)
                config['username'] = self.server.accounts.owner_username() if self.account == 'owner' else self.account
                config['token'] = self.request_token()
                if self.account != "owner":
                    config = {k:config[k] for k in ("token", "username", "model", "render_enabled", "max_upload_bytes",
                              "reconstruction_available", "sdxl_available")}
                    return self.json(200, dict(config, frontier_provider="disabled", chatgpt_models=[]))
                config['frontier_provider'] = self.server.frontier_provider
                if self.server.chatgpt:
                    config['chatgpt_connected'] = self.server.chatgpt.connected()
                elif self.server.frontier_provider == 'disabled':
                    config['chatgpt_models'] = []
                self.json(200, config)
            elif self.account != "owner" and not path.startswith("/api/library/") and path != "/api/library":
                raise RequestError("Not found.", 404)
            elif path == "/api/chatgpt/models" and self.server.chatgpt:
                self.json(200, self.server.chatgpt.available() or {"state": "disconnected"})
            elif path == "/api/chatgpt/status" and self.server.chatgpt:
                self.json(200, self.server.chatgpt.status())
            elif path == "/developer":
                self.respond(
                    200,
                    (STATIC / "index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif match := re.fullmatch(rf"/api/library/imports/({ID})", path):
                self.json(200, self.server.library.import_result(self.account, match[1]))
            elif path == "/api/library":
                self.json(200, self.server.library.catalog(self.account))
            elif match := re.fullmatch(f"/api/library/pages/({ID})/transcript", path):
                self.json(200, self.server.library.transcript(match[1], self.account))
            elif match := re.fullmatch(
                f"/api/library/pages/({ID})/(original|translated)(/thumbnail)?", path
            ):
                page, variant, thumb = match.groups()
                load = self.server.library.thumbnail if thumb else self.server.library.image
                mime, data = load(page, variant, self.account)
                self.respond(200, data, mime)
            elif match := re.fullmatch(
                rf"/api/library/books/({ID})/(original|translated|both)\.zip", path
            ):
                if not self.server.archive_slot.acquire(blocking=False):
                    raise RequestError("Another download is being prepared. Try again shortly.", 429)
                try:
                    with self.server.library.archive(*match.groups(), self.account) as archive:
                        name = self.server.library.archive_name(*match.groups(), self.account)
                        fallback = re.sub(r"[^a-zA-Z0-9 ._-]", "_", name)
                        self.send_response(200)
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Type", "application/zip")
                        self.send_header(
                            "Content-Disposition",
                            f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{quote(name, safe="")}',
                        )
                        self.send_header("Content-Length", str(archive.tell()))
                        self.end_headers()
                        archive.seek(0)
                        shutil.copyfileobj(archive, self.wfile)
                finally:
                    self.server.archive_slot.release()
            else:
                return super().do_GET()
        except RequestError as exc:
            self.json(exc.status, dict(error=str(exc)))

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            self.local_request()
            paths = {"/api/auth/setup": "setup", "/api/auth/login": "login", "/api/auth/register": "register", "/api/auth/logout": "logout",
                     "/api/owner/login": "owner", "/api/owner/logout": "logout"}
            if path in paths:
                return self.server.accounts.handle(self, paths[path])
            self.account = self.server.accounts.identify(self)
            if self.account != "owner" and not path.startswith("/api/library/"):
                raise RequestError("Not found.", 404)
        except RequestError as exc:
            self.close_connection = True
            self.json(exc.status, dict(error=str(exc)))
            return
        # Multipart parsing and document rasterization can each use substantially
        # more memory than the upload. Admit one upload before reading its body.
        uploading = path in ("/api/library/add", "/api/jobs")
        if uploading and not self.server.upload_slot.acquire(blocking=False):
            self.close_connection = True
            self.json(429, {"error": "Another upload is being processed. Try again shortly."})
            return
        try:
            self._post(path)
        finally:
            if uploading:
                self.server.upload_slot.release()

    def _post(self, path):
        connecting = path in (
            "/api/chatgpt/connect", "/api/chatgpt/disconnect",
            "/api/chatgpt/phone/start", "/api/chatgpt/phone/finish",
        ) and self.server.chatgpt
        if not connecting and not path.startswith("/api/library/"):
            return super().do_POST()
        try:
            self.local_request()
            if not secrets.compare_digest(
                self.headers.get("X-Tralens-Token", "").encode(), self.request_token().encode()
            ):
                raise RequestError("Reload the page to reconnect.", 403)
            lengths = self.headers.get_all("Content-Length") or []
            if (
                self.headers.get("Transfer-Encoding")
                or len(lengths) != 1
                or not re.fullmatch("[0-9]{1,10}", lengths[0])
            ):
                raise RequestError("Content length required.", 411)
            length = int(lengths[0])
            adding = path == "/api/library/add"
            if adding:
                if not 0 < length <= MAX_IMPORT_BYTES:
                    raise RequestError("Import up to 4 GiB at a time.", 413)
                with read_import(self.rfile, length, self.headers.get("Content-Type", "")) as (meta, files):
                    return self.json(201, self.server.library.add(meta, files, self.account))
            if not 0 < length <= 8192:
                raise RequestError("Request body is too large.", 413)
            data = self.rfile.read(length)
            if len(data) != length:
                raise RequestError("Incomplete upload.")
            if connecting:
                body = json.loads(data)
                if path == "/api/chatgpt/phone/start" and isinstance(body, dict):
                    return self.json(200, self.server.chatgpt.select_phone(body.get('port')))
                if path == "/api/chatgpt/phone/finish":
                    return self.json(200, self.server.chatgpt.finish_phone(body))
                if body != {}:
                    raise RequestError("Unexpected sign-in parameters.")
                if path == "/api/chatgpt/disconnect":
                    return self.json(200, self.server.chatgpt.disconnect())
                if path != "/api/chatgpt/connect":
                    raise RequestError("Unexpected sign-in parameters.")
                self.json(200, self.server.chatgpt.select())
            else:
                body = json.loads(data)
                if not isinstance(body, dict):
                    raise RequestError("Expected library settings.")
                if path == "/api/library/download-ticket":
                    target = body.get("path")
                    if not isinstance(target, str) or not DOWNLOAD_PATH.fullmatch(target):
                        raise RequestError("Choose a reader download.")
                    # The GET performs the same ownership check as a browser download.
                    self.json(200, {"ticket": self.server.accounts.download_ticket(self, target)})
                    return
                self.json(
                    200,
                    self.server.library.change(
                        path.removeprefix("/api/library/"), body, self.account
                    ),
                )
        except RequestError as exc:
            self.json(exc.status, dict(error=str(exc)))
        except RecursionError:
            self.json(400, dict(error="Request is nested too deeply."))
        except (ValueError, TypeError, OSError) as exc:
            self.json(400, dict(error=str(exc)))


class ReaderServer(ThreadingHTTPServer):
    def process_request(self, request, client_address):
        if not self.connection_slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()

    def server_close(self):
        if hasattr(self, "library"):
            self.library.close()
        super().server_close()


def create_reader_server(app, port, *, host, owner=None, registration=True):
    frontier_provider = app.config.frontier_provider
    if host != "127.0.0.1":
        raise ValueError("The HTTPS proxy must connect over loopback.")
    server = ReaderServer((host, port), ReaderHandler)
    server.connection_slots = threading.BoundedSemaphore(32)
    from .chatgpt_connection import ReaderConnection
    if frontier_provider == 'chatgpt' and not owner and host != '127.0.0.1':
        server.server_close()
        raise ValueError('Remote ChatGPT translation requires owner mode.')
    server.frontier_provider = frontier_provider
    server.chatgpt = ReaderConnection() if frontier_provider == 'chatgpt' else None
    server.owner = owner
    server.archive_slot = threading.BoundedSemaphore(1)
    server.upload_slot = threading.BoundedSemaphore(1)
    server.app = app
    try:
        fresh = not any(app.root.iterdir())
        server.library = Library(app)
        server.accounts = Accounts(server.library, owner, registration, fresh=fresh)
    except BaseException:
        server.server_close()
        raise
    return server
