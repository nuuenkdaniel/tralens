"""Server-owned reader library. One queued page at a time enters the existing job API."""

from contextlib import contextmanager
import fcntl
import json
import math
import re
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import uuid
import zipfile

from PIL import Image
from .server import RequestError, checkpoint, options
from .import_pages import MAX_PAGES, expand, webp_png


def label(value):
    if not isinstance(value, str) or not 0 < len(value.strip()) <= 200:
        raise RequestError("Use a name between 1 and 200 characters.")
    return value.strip()


def reading_text(manifest, detections):
    polygons = {d['source_id']: d['polygon'] for d in detections}
    lines = []
    minimum = None
    unsafe = False
    for row in sorted(manifest['units'], key=lambda item: item.get('reading_order', len(manifest['units']))):
        binding = row.get('binding')
        if not binding or row.get('reason_code') in ('uncorroborated_source', 'replacement_unchanged_text'):
            continue
        points = [pt for member in binding['member_ids'] for pt in polygons[member]]
        xs, ys = zip(*points)
        lines.append(dict(text=binding['text'], box=[min(xs), min(ys), max(xs), max(ys)]))
        if row.get('display_status') == 'unchanged':
            continue
        size = (row.get('layout') or {}).get('size')
        if size and size > 0:
            minimum = min(size, minimum or size)
        if (not size or row['status'] != 'automatic_composed' or row.get('removal_fallback')
                or row.get('display_status') in ('patch', 'caption', 'aggressive', 'learned')):
            unsafe = True
    # CSS pixels at fit-to-page scale. A negative width requests original art at any scale.
    width = manifest['source']['size'][0]
    return lines, -1 if unsafe else math.ceil(12 * width / minimum) if minimum else 0


def saved_reading(directory):
    notices = []
    manifest = checkpoint(directory / 'run/rendering/manifest.json', notices)
    if manifest is None:
        if notices:
            raise ValueError(notices[0])
        return None, None
    preview = checkpoint(directory / 'run/preview/preview.json', notices)
    if not preview:
        raise ValueError("The saved OCR page could not be read.")
    lines, width = reading_text(manifest, preview['ocr']['raw_detections'][0])
    return json.dumps(lines, ensure_ascii=False), width


class Library:
    def __init__(self, app):
        self.app = app
        self.path = app.root / "reader.sqlite3"
        self.owner = (app.root / "library.lock").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise RuntimeError("Another reader already owns this library.") from None
        self.wake = threading.Event()
        self.stopped = threading.Event()
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS books(id TEXT PRIMARY KEY,title TEXT,added REAL,last_read REAL,position TEXT);
                CREATE TABLE IF NOT EXISTS chapters(id TEXT PRIMARY KEY,book TEXT REFERENCES books ON DELETE CASCADE,title TEXT);
                CREATE TABLE IF NOT EXISTS pages(id TEXT PRIMARY KEY,chapter TEXT REFERENCES chapters ON DELETE CASCADE,
                    name TEXT,mime TEXT,source BLOB,final BLOB,status TEXT,error TEXT,options TEXT,job TEXT,updated INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS chapter_book ON chapters(book);
                CREATE INDEX IF NOT EXISTS page_chapter ON pages(chapter);
                CREATE INDEX IF NOT EXISTS page_queue ON pages(status);
            """)
            columns = {r["name"] for r in db.execute("PRAGMA table_info(pages)")}
            for dimension in ("width", "height"):
                if dimension not in columns:
                    db.execute(f"ALTER TABLE pages ADD COLUMN {dimension} INTEGER")
            if 'reading' not in columns:
                db.execute("ALTER TABLE pages ADD COLUMN reading TEXT")
            if 'reading_width' not in columns:
                db.execute("ALTER TABLE pages ADD COLUMN reading_width INTEGER")
            if "position" not in {
                r["name"] for r in db.execute("PRAGMA table_info(chapters)")
            }:
                db.execute("ALTER TABLE chapters ADD COLUMN position TEXT")
            db.execute(
                "UPDATE pages SET status='failed',error='Translation was interrupted. Try again.' WHERE status='running'"
            )
        if "reading" not in columns:
            self.restore_reading()
        self.thread = threading.Thread(
            target=self.work, name="reader-library", daemon=True
        )
        self.thread.start()

    def restore_reading(self):
        """Attach legacy evidence only to the exact composite it produced."""
        with self.db() as db:
            rows = db.execute("SELECT id,job FROM pages WHERE final IS NOT NULL AND job IS NOT NULL").fetchall()
            for row in rows:
                try:
                    if not re.fullmatch('[0-9a-f]{32}', row['job']):
                        continue
                    directory = self.app.root / row['job']
                    composite = directory / 'run/rendering/composite.png'
                    if (any(p.is_symlink() for p in (composite, *composite.parents))
                            or not composite.is_file() or composite.stat().st_size > 64 * 1024**2):
                        continue
                    reading, width = saved_reading(directory)
                    if reading is not None:
                        db.execute("UPDATE pages SET reading=?,reading_width=? WHERE id=? AND final=?",
                                   (reading, width, row['id'], composite.read_bytes()))
                except (RequestError, OSError, ValueError, KeyError, TypeError):
                    continue

    def close(self):
        self.stopped.set()
        self.wake.set()
        self.thread.join()
        self.owner.close()

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def catalog(self):
        with self.db() as db:
            db.execute("BEGIN")
            books = {
                r["id"]: dict(r, chapters=[])
                for r in db.execute("SELECT * FROM books ORDER BY rowid")
            }
            chapters = {}
            for r in db.execute("SELECT * FROM chapters ORDER BY rowid"):
                chap = chapters[r["id"]] = dict(r, pages=[])
                books[r["book"]]["chapters"].append(chap)
            for r in db.execute(
                "SELECT id,chapter,name,width,height,status,error,reading_width,reading IS NOT NULL AND reading != '[]' AS has_text,updated,final IS NOT NULL AS ready FROM pages ORDER BY rowid"
            ):
                chapters[r["chapter"]]["pages"].append(dict(r))
            used = db.execute(
                "SELECT coalesce(sum(length(source)+coalesce(length(final),0)),0) FROM pages"
            ).fetchone()[0]
        return dict(books=list(books.values()), bytes=used)

    def settings(self, supplied):
        settings = options(supplied, render_enabled=self.app.config.fonts is not None)
        if settings["mode"] != "render":
            raise RequestError("Reader pages require rendering.")
        config = self.app.config
        model = settings["reconstruction_policy"]
        if (model == "lama" and not config.reconstruction_config) or (
            model == "sdxl" and not config.sdxl_config
        ):
            raise RequestError(
                "The selected inpainting model is unavailable on this server."
            )
        return json.dumps(settings)

    def add(self, meta, files):
        settings = self.settings(meta.get("options"))
        if not 1 <= len(files) <= 256:
            raise RequestError("Choose between 1 and 256 pages per import.")
        with tempfile.TemporaryDirectory(prefix="tralens-import-") as temp:
            directory = Path(temp)
            pages = []
            total = 0
            for name, data in files:
                name = label(name)
                try:
                    for filename, mime, content, size in expand(name, data, directory, MAX_PAGES - len(pages)):
                        if len(pages) >= MAX_PAGES:
                            raise ValueError(f"An import can contain at most {MAX_PAGES} pages.")
                        total += len(content)
                        if total > self.app.config.root_storage_bytes:
                            raise ValueError("Expanded pages exceed library storage capacity.")
                        source = directory / str(len(pages))
                        source.write_bytes(content)
                        pages.append((uuid.uuid4().hex, filename, mime, source, *size))
                except (ValueError, OSError, Image.DecompressionBombError) as exc:
                    raise RequestError(f"{name}: {exc}") from exc
            return self.store_pages(meta, settings, pages, total)

    def store_pages(self, meta, settings, pages, total):
        book = meta.get("book")
        chapter = uuid.uuid4().hex
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            free = (
                db.execute("PRAGMA freelist_count").fetchone()[0]
                * db.execute("PRAGMA page_size").fetchone()[0]
            )
            if (
                self.path.stat().st_size - free + total
                > self.app.config.root_storage_bytes
            ):
                raise RequestError("Library storage is full.", 409)
            if not book:
                book = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO books VALUES(?,?,?,0,NULL)",
                    (book, label(meta.get("title")), time.time()),
                )
            elif not db.execute("SELECT id FROM books WHERE id=?", (book,)).fetchone():
                raise RequestError("Manga no longer exists.", 404)
            db.execute(
                "INSERT INTO chapters(id,book,title) VALUES(?,?,?)",
                (chapter, book, label(meta.get("chapter"))),
            )
            db.executemany(
                "INSERT INTO pages(id,chapter,name,mime,source,width,height,status,options) VALUES(?,?,?,?,?,?,?,'queued',?)",
                ((i, chapter, name, mime, path.read_bytes(), w, h, settings)
                 for i, name, mime, path, w, h in pages),
            )
        self.wake.set()
        return dict(book=book, chapter=chapter)

    def change(self, action, body):
        book = body.get("book")
        with self.db() as db:
            if action == "rename":
                updated = db.execute(
                    "UPDATE books SET title=? WHERE id=?",
                    (label(body.get("title")), book),
                )
            elif action == "delete":
                updated = db.execute("DELETE FROM books WHERE id=?", (book,))
            elif action == "position":
                page = body.get("page")
                updated = db.execute(
                    "UPDATE books SET position=?,last_read=? WHERE id=? AND EXISTS "
                    "(SELECT 1 FROM pages JOIN chapters ON chapters.id=pages.chapter WHERE pages.id=? AND chapters.book=books.id)",
                    (page, time.time(), book, page),
                )
                if updated.rowcount:
                    db.execute(
                        "UPDATE chapters SET position=? WHERE id=(SELECT chapter FROM pages WHERE id=?)",
                        (page, page),
                    )
            elif action == "skip":
                db.execute("BEGIN IMMEDIATE")
                target = db.execute("SELECT status,job FROM pages WHERE id=? AND chapter IN "
                    "(SELECT id FROM chapters WHERE book=?)", (body.get("page"), book)).fetchone()
                if not target or target["status"] not in ("queued", "waiting", "running"):
                    raise RequestError("This page has already finished.", 409)
                if target["status"] == "running":
                    if not target["job"]:
                        raise RequestError("Page is starting. Try skipping again in a moment.", 409)
                    self.app.cancel(target["job"])
                updated = db.execute("UPDATE pages SET status='skipped',error=NULL WHERE id=?", (body["page"],))
            elif action == "retry":
                settings = self.settings(body.get("options"))
                updated = db.execute(
                    "UPDATE pages SET options=?,status='queued',error=NULL WHERE status NOT IN ('queued','running') "
                    "AND chapter IN (SELECT id FROM chapters WHERE book=?) AND (? IS NULL OR id=?)",
                    (settings, book, body.get("page"), body.get("page")),
                )
            else:
                raise RequestError("Unknown library action.", 404)
            if not updated.rowcount:
                raise RequestError("Item is missing or already processing.", 409)
        self.wake.set()
        return {}

    def transcript(self, page):
        with self.db() as db:
            row = db.execute("SELECT reading FROM pages WHERE id=?", (page,)).fetchone()
        if not row or row["reading"] is None:
            raise RequestError("Page text unavailable.", 404)
        return json.loads(row["reading"])

    def image(self, page, variant):
        with self.db() as db:
            field = "source" if variant == "original" else "final"
            r = db.execute(
                f"SELECT mime,{field} AS data FROM pages WHERE id=?", (page,)
            ).fetchone()
        if not r or r["data"] is None:
            raise RequestError("Page unavailable.", 404)
        return (r["mime"] if field == "source" else "image/png"), r["data"]

    def archive(self, book, variant):
        archive = tempfile.TemporaryFile()
        try:
            with self.db() as db, zipfile.ZipFile(archive, "w") as z:
                rows = db.execute(
                    "SELECT chapters.id,mime,source,final FROM pages JOIN chapters ON chapters.id=pages.chapter "
                    "WHERE chapters.book=? ORDER BY chapters.rowid,pages.rowid",
                    (book,),
                )
                previous = None
                chapter = 0
                for cid, mime, source, final in rows:
                    if previous != cid:
                        chapter += 1
                        number = 0
                        previous = cid
                    number += 1
                    stem = f"chapter-{chapter:03d}/page-{number:04d}"
                    if variant in ("both", "original"):
                        z.writestr(
                            "original/"
                            + stem
                            + {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[mime],
                            source,
                        )
                    if final and variant in ("both", "translated"):
                        z.writestr("translated/" + stem + ".png", final)
                if not z.namelist():
                    raise RequestError("No pages available for this download.", 404)
            return archive
        except BaseException:
            archive.close()
            raise

    def work(self):
        from ..scheduling import remote_state_path
        from ..translation.runtime import local_endpoint
        fence = remote_state_path(local_endpoint(self.app.config.endpoint))
        while not self.stopped.is_set():
            with self.db() as db:
                db.execute("BEGIN IMMEDIATE")
                if fence.exists():
                    db.execute("UPDATE pages SET status='waiting',error=? WHERE status='queued' AND "
                               "(coalesce(json_extract(options,'$.translation_model'),'local')='local' "
                               "OR json_extract(options,'$.quality_reread')=1)",
                               ("Local translator is waiting for the server operator to confirm the previous request finished.",))
                else:
                    db.execute("UPDATE pages SET status='queued',error=NULL WHERE status='waiting'")
                row = db.execute(
                    "SELECT id,source,mime,options FROM pages WHERE status='queued' ORDER BY rowid LIMIT 1"
                ).fetchone()
                if row:
                    db.execute(
                        "UPDATE pages SET status='running',job=NULL WHERE id=?", (row["id"],)
                    )
            if not row:
                self.wake.wait(2)
                self.wake.clear()
                continue
            try:
                source = webp_png(row["source"])[0] if row["mime"] == "image/webp" else row["source"]
                job = self.app.start(source, json.loads(row["options"]))
            except RequestError as exc:
                if exc.status == 409 and "already running" in str(exc):
                    with self.db() as db:
                        db.execute(
                            "UPDATE pages SET status='queued' WHERE id=?", (row["id"],)
                        )
                    self.stopped.wait(1)
                    continue
                self.finish(row["id"], None, str(exc))
                continue
            except Exception as exc:
                self.finish(row["id"], None, str(exc))
                continue
            with self.db() as db:
                db.execute(
                    "UPDATE pages SET status='running',job=? WHERE id=?",
                    (job, row["id"]),
                )
            try:
                done = self.app.job(job)["done"]
                while not done.wait(0.25):
                    if self.stopped.is_set():
                        return
                record = self.app.job(job)
                if record["status"] not in (
                    "complete",
                    "partial",
                    "complete_with_failures",
                ):
                    problem = record.get("error") or record["status"]
                    raise ValueError(
                        problem.get("message", problem)
                        if isinstance(problem, dict)
                        else problem
                    )
                try:
                    final = self.app.artifact(
                        job, "run/rendering/composite.png"
                    ).read_bytes()
                except RequestError as exc:
                    raise ValueError(
                        "No translated image was produced. Check the translator or try again."
                    ) from exc
                reading, width = saved_reading(record["directory"])
                self.finish(row["id"], final, None, reading, width)
            except Exception as exc:
                message = str(exc)
                settings = json.loads(row['options'])
                if fence.exists() and (settings.get('translation_model', 'local') == 'local' or settings['quality_reread']):
                    message = "The local translator stopped responding. Check the previous request on the model server before retrying."
                self.finish(row["id"], None, message)

    def finish(self, page, final, error, reading=None, width=None):
        with self.db() as db:
            db.execute(
                "UPDATE pages SET status=?,error=?,final=coalesce(?,final),reading=CASE WHEN ? THEN ? ELSE reading END, "
                "reading_width=CASE WHEN ? THEN ? ELSE reading_width END,updated=updated+1 WHERE id=? AND status='running'",
                ("ready" if final else "failed", error, final, final is not None, reading, final is not None, width, page),
            )
