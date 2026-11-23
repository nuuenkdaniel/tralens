"""Serve the independent web client and its library under the job API's origin."""

import json
import mimetypes
from pathlib import Path
import re
import secrets
import shutil
import threading
from urllib.parse import urlsplit

from .library import Library
from .multipart_upload import MAX_IMPORT_BYTES, read_import
from .server import Handler, RequestError, STATIC, ThreadingHTTPServer

PROJECT = Path(__file__).resolve().parents[4]
WEB = PROJECT / "frontend/web"
ID = "[0-9a-f]{32}"


class ReaderHandler(Handler):
    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            self.local_request()
            if path == "/healthz":
                return self.json(200, {"status": "ok"})
            if path == "/privacy":
                return self.respond(200, (PROJECT / "PRIVACY.md").read_bytes(), "text/plain; charset=utf-8")
            if path in (
                "/",
                "/reader.js",
                "/reader.css",
            ):
                file = WEB / ("index.html" if path == "/" else path[1:])
                self.respond(200, file.read_bytes(), mimetypes.guess_type(file.name)[0])
            else:
                if self.server.owner:
                    self.server.owner.authorize(self)
                return self._private_get(path)
        except RequestError as exc:
            self.json(exc.status, dict(error=str(exc)))

    def _private_get(self, path):
        try:
            if path == "/api/config":
                config = self.server.app.public_config()
                config['owner_mode'] = bool(self.server.owner)
                config['frontier_provider'] = self.server.frontier_provider
                if self.server.chatgpt:
                    config['chatgpt_connected'] = self.server.chatgpt.connected()
                elif self.server.frontier_provider == 'disabled':
                    config['chatgpt_models'] = []
                self.json(200, config)
            elif path == "/api/chatgpt/status" and self.server.chatgpt:
                self.json(200, self.server.chatgpt.status())
            elif path == "/developer":
                self.respond(
                    200,
                    (STATIC / "index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif path == "/api/library":
                self.json(200, self.server.library.catalog())
            elif match := re.fullmatch(f"/api/library/pages/({ID})/transcript", path):
                self.json(200, self.server.library.transcript(match[1]))
            elif match := re.fullmatch(
                f"/api/library/pages/({ID})/(original|translated)", path
            ):
                mime, data = self.server.library.image(*match.groups())
                self.respond(200, data, mime)
            elif match := re.fullmatch(
                rf"/api/library/books/({ID})/(original|translated|both)\.zip", path
            ):
                with self.server.library.archive(*match.groups()) as archive:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header(
                        "Content-Disposition",
                        'attachment; filename="manga-' + match[2] + '.zip"',
                    )
                    self.send_header("Content-Length", str(archive.tell()))
                    self.end_headers()
                    archive.seek(0)
                    shutil.copyfileobj(archive, self.wfile)
            else:
                return super().do_GET()
        except RequestError as exc:
            self.json(exc.status, dict(error=str(exc)))

    def do_POST(self):
        path = urlsplit(self.path).path
        if self.server.owner:
            try:
                self.local_request()
                if path == "/api/owner/login":
                    return self.server.owner.login(self)
                if path == "/api/owner/logout":
                    if self.headers.get("Origin") != self.server.owner.origin:
                        raise RequestError("Unexpected origin.", 403)
                    return self.server.owner.logout(self)
                self.server.owner.authorize(self)
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
                self.headers.get("X-Tralens-Token", ""), self.server.app.token
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
                    return self.json(201, self.server.library.add(meta, files))
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
                self.json(
                    200,
                    self.server.library.change(
                        path.removeprefix("/api/library/"), body
                    ),
                )
        except RequestError as exc:
            self.json(exc.status, dict(error=str(exc)))
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


def create_reader_server(app, port, *, host, owner=None):
    frontier_provider = app.config.frontier_provider
    if owner and host != "127.0.0.1":
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
    server.upload_slot = threading.BoundedSemaphore(1)
    server.app = app
    try:
        server.library = Library(app)
    except BaseException:
        server.server_close()
        raise
    return server
