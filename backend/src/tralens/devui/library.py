"""Server-owned reader library. One queued page at a time enters the existing job API."""

from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
import fcntl
import json
import math
import os
import re
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

from PIL import Image
from .server import RequestError, checkpoint, options
from .import_pages import MAX_PAGES, expand, require_space, working_page


def label(value):
    if not isinstance(value, str) or not 0 < len(value.strip()) <= 200:
        raise RequestError("Use a name between 1 and 200 characters.")
    return value.strip()


def chapter_language(value):
    if value not in ("auto", "ja", "zh", "ko"):
        raise RequestError("Choose Auto, Japanese, Chinese, or Korean.")
    return value


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


def private_library_file(path, *, create):
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags | (os.O_CREAT if create else 0), 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise ValueError(f"Use a private, service-owned regular file for {path.name}.")
        if stat.S_IMODE(info.st_mode) != 0o600:
            os.fchmod(fd, 0o600)
        return os.fdopen(fd, "r+b")
    except BaseException:
        os.close(fd)
        raise


class Library:
    def __init__(self, app):
        self.app = app
        self.path = app.root / "reader.sqlite3"
        self.owner = private_library_file(app.root / "library.lock", create=True)
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            private_library_file(self.path, create=True).close()
            # SQLite creates new journals with the database's mode. Tighten
            # journals left by a prior process before SQLite opens them.
            for suffix in ("-journal", "-wal", "-shm"):
                try:
                    private_library_file(self.path.with_name(self.path.name + suffix), create=False).close()
                except FileNotFoundError:
                    pass
        except BlockingIOError:
            self.owner.close()
            raise RuntimeError("Another reader already owns this library.") from None
        except BaseException:
            self.owner.close()
            raise
        self.thumbnail_slots = threading.BoundedSemaphore(2)
        self.cached_thumbnail = lru_cache(maxsize=160)(self._make_thumbnail)
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
            if "account" not in {r["name"] for r in db.execute("PRAGMA table_info(books)")}:
                db.execute("ALTER TABLE books ADD COLUMN account TEXT NOT NULL DEFAULT 'owner'")
            db.execute("CREATE INDEX IF NOT EXISTS book_account ON books(account)")
            columns = {r["name"] for r in db.execute("PRAGMA table_info(pages)")}
            for dimension in ("width", "height"):
                if dimension not in columns:
                    db.execute(f"ALTER TABLE pages ADD COLUMN {dimension} INTEGER")
            if 'reading' not in columns:
                db.execute("ALTER TABLE pages ADD COLUMN reading TEXT")
            if 'reading_width' not in columns:
                db.execute("ALTER TABLE pages ADD COLUMN reading_width INTEGER")
            chapter_columns = {r["name"] for r in db.execute("PRAGMA table_info(chapters)")}
            # Imported chapters predating language selection were processed as Japanese.
            for name, declaration in (("source_language", "TEXT NOT NULL DEFAULT 'ja'"),
                                      ("detected_language", "TEXT"), ("detection_seconds", "REAL"),
                                      ("language_uncertain", "INTEGER"), ("position", "TEXT")):
                if name not in chapter_columns:
                    db.execute(f"ALTER TABLE chapters ADD COLUMN {name} {declaration}")
            if "source_language" not in {r["name"] for r in db.execute("PRAGMA table_info(books)")}:
                db.execute("BEGIN IMMEDIATE")
                db.execute("ALTER TABLE books ADD COLUMN source_language TEXT NOT NULL DEFAULT 'auto'")
                db.execute("ALTER TABLE books ADD COLUMN detected_language TEXT")
                db.execute("ALTER TABLE books ADD COLUMN detection_seconds REAL")
                db.execute("ALTER TABLE books ADD COLUMN language_uncertain INTEGER")
                # Seed the shared result from an existing detection, keeping divergent translations as overrides.
                db.execute("""UPDATE books SET (detected_language,detection_seconds,language_uncertain) =
                    (SELECT detected_language,detection_seconds,language_uncertain FROM chapters
                     WHERE book=books.id AND source_language='auto' AND detected_language IS NOT NULL
                     ORDER BY language_uncertain, rowid LIMIT 1)""")
                db.execute("""UPDATE books SET source_language=(
                    SELECT MIN(source_language) FROM chapters WHERE book=books.id)
                    WHERE id IN (SELECT book FROM chapters GROUP BY book
                                 HAVING COUNT(DISTINCT source_language)=1 AND MIN(source_language)!='auto')""")
                db.execute("""UPDATE chapters SET source_language=detected_language
                    WHERE source_language='auto' AND detected_language IS NOT NULL AND
                    detected_language != (SELECT detected_language FROM books WHERE id=chapters.book)""")
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

    def catalog(self, account="owner"):
        with self.db() as db:
            db.execute("BEGIN")
            books = {
                r["id"]: dict(r, chapters=[])
                for r in db.execute("SELECT id,title,added,last_read,position,source_language,detected_language,detection_seconds,language_uncertain FROM books WHERE account=? ORDER BY rowid", (account,))
            }
            chapters = {}
            for r in db.execute("SELECT * FROM chapters WHERE book IN (SELECT id FROM books WHERE account=?) ORDER BY rowid", (account,)):
                chap = chapters[r["id"]] = dict(r, pages=[])
                book = books[r["book"]]
                if chap["source_language"] == "auto":
                    inherited = book["source_language"] == "auto"
                    chap.update(detected_language=book["detected_language"] if inherited else book["source_language"],
                                language_uncertain=book["language_uncertain"] if inherited else False,
                                detection_seconds=book["detection_seconds"] if inherited else None)
                book["chapters"].append(chap)
            for r in db.execute(
                "SELECT id,chapter,name,width,height,status,error,reading_width,reading IS NOT NULL AND reading != '[]' AS has_text,updated,final IS NOT NULL AS ready FROM pages WHERE chapter IN (SELECT chapters.id FROM chapters JOIN books ON chapters.book=books.id WHERE account=?) ORDER BY rowid", (account,)
            ):
                chapters[r["chapter"]]["pages"].append(dict(r))
            used = db.execute(
                "SELECT coalesce(sum(length(source)+coalesce(length(final),0)),0) FROM pages JOIN chapters ON pages.chapter=chapters.id JOIN books ON chapters.book=books.id WHERE books.account=?", (account,)
            ).fetchone()[0]
        return dict(books=list(books.values()), bytes=used)

    def settings(self, supplied, account="owner"):
        settings = options(supplied, render_enabled=self.app.config.fonts is not None)
        if account != "owner" and settings.get("translation_model", "local") != "local":
            raise RequestError("Use the local translator for this account.", 403)
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

    @staticmethod
    def import_chapter(account, key):
        if not isinstance(key, str) or not re.fullmatch("[0-9a-f]{32}", key):
            raise RequestError("Invalid import identifier.")
        return uuid.uuid5(uuid.NAMESPACE_URL, f"tralens:import:{account}:{key}").hex

    @staticmethod
    def saved_chapter(db, account, chapter):
        row = db.execute("SELECT book FROM chapters JOIN books ON books.id=chapters.book "
                         "WHERE chapters.id=? AND books.account=?", (chapter, account)).fetchone()
        return dict(book=row["book"], chapter=chapter) if row else None

    def import_result(self, account, key):
        chapter = self.import_chapter(account, key)
        with self.db() as db:
            return self.saved_chapter(db, account, chapter)

    def add(self, meta, files, account="owner"):
        if "import_id" in meta:
            existing = self.import_result(account, meta["import_id"])
            if existing:
                return existing
        if meta.get("book") is not None and not isinstance(meta["book"], str):
            raise RequestError("Invalid manga identifier.")
        settings = self.settings(meta.get("options"), account)
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
                        require_space(directory, len(content))
                        source.write_bytes(content)
                        pages.append((uuid.uuid4().hex, filename, mime, source, *size))
                except RequestError:
                    raise
                except (ValueError, OSError, Image.DecompressionBombError) as exc:
                    raise RequestError(f"{name}: {exc}") from exc
            return self.store_pages(meta, settings, pages, total, account)

    def check_capacity(self, db, growth):
        pages = db.execute("PRAGMA page_count").fetchone()[0]
        unused = db.execute("PRAGMA freelist_count").fetchone()[0]
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        if (pages - unused) * page_size + growth > self.app.config.root_storage_bytes:
            raise RequestError("Library storage is full.", 409)

    def store_pages(self, meta, settings, pages, total, account):
        book = meta.get("book")
        chapter = self.import_chapter(account, meta["import_id"]) if "import_id" in meta else uuid.uuid4().hex
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if "import_id" in meta:
                prior = self.saved_chapter(db, account, chapter)
                if prior:
                    return prior
            require_space(self.path, total)
            self.check_capacity(db, total)
            language = chapter_language(meta.get("source_language", "auto"))
            if not book:
                book = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO books(id,title,added,last_read,position,account,source_language) VALUES(?,?,?,0,NULL,?,?)",
                    (book, label(meta.get("title")), time.time(), account, language),
                )
                language = "auto"  # The first chapter inherits the new manga’s setting.
            elif not db.execute("SELECT id FROM books WHERE id=? AND account=?", (book, account)).fetchone():
                raise RequestError("Manga no longer exists.", 404)
            db.execute(
                "INSERT INTO chapters(id,book,title,source_language) VALUES(?,?,?,?)",
                (chapter, book, label(meta.get("chapter")), language),
            )
            db.executemany(
                "INSERT INTO pages(id,chapter,name,mime,source,width,height,status,options) VALUES(?,?,?,?,?,?,?,'queued',?)",
                ((i, chapter, name, mime, path.read_bytes(), w, h, settings)
                 for i, name, mime, path, w, h in pages),
            )
            self.check_capacity(db, 0)
        self.wake.set()
        return dict(book=book, chapter=chapter)

    def change(self, action, body, account="owner"):
        book = body.get("book")
        if any(value is not None and not isinstance(value, str) for value in (book, body.get("page"))):
            raise RequestError("Invalid manga or page identifier.")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT source_language,detected_language FROM books WHERE id=? AND account=?",
                                 (book, account)).fetchone()
            if not current:
                raise RequestError("Manga unavailable.", 404)
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
                target = db.execute("SELECT status,job FROM pages WHERE id=? AND chapter IN "
                    "(SELECT id FROM chapters WHERE book=?)", (body.get("page"), book)).fetchone()
                if not target or target["status"] not in ("queued", "waiting", "running"):
                    raise RequestError("This page has already finished.", 409)
                if target["status"] == "running":
                    if not target["job"]:
                        raise RequestError("Page is starting. Try skipping again in a moment.", 409)
                    self.app.cancel(target["job"])
                updated = db.execute("UPDATE pages SET status='skipped',error=NULL WHERE id=?", (body["page"],))
            elif action == "book-language":
                language = chapter_language(body.get("source_language"))
                updated = db.execute("UPDATE books SET source_language=? WHERE id=?", (language, book))
                previous = current["source_language"]
                old_language = current["detected_language"] if previous == "auto" else previous
                new_language = current["detected_language"] if language == "auto" else language
                if old_language == new_language:
                    return {}
                db.execute("""UPDATE pages SET status='queued',job=NULL,error=NULL WHERE chapter IN
                              (SELECT id FROM chapters WHERE book=? AND source_language='auto')""", (book,))
            elif action == "chapter-language":
                language = chapter_language(body.get("source_language"))
                chapter = body.get("chapter")
                if not isinstance(chapter, str):
                    raise RequestError("Choose a chapter.")
                updated = db.execute("UPDATE chapters SET source_language=? WHERE id=? AND book=?",
                                     (language, chapter, book))
                if updated.rowcount:
                    db.execute("UPDATE pages SET status='queued',job=NULL,error=NULL WHERE chapter=?",
                               (chapter,))
            elif action == "retry":
                settings = self.settings(body.get("options"), account)
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

    def transcript(self, page, account="owner"):
        with self.db() as db:
            row = db.execute("SELECT reading FROM pages JOIN chapters ON pages.chapter=chapters.id JOIN books ON chapters.book=books.id WHERE pages.id=? AND books.account=?", (page, account)).fetchone()
        if not row or row["reading"] is None:
            raise RequestError("Page text unavailable.", 404)
        return json.loads(row["reading"])

    def image(self, page, variant, account="owner"):
        with self.db() as db:
            field = "source" if variant == "original" else "final"
            r = db.execute(
                f"SELECT mime,{field} AS data FROM pages JOIN chapters ON pages.chapter=chapters.id JOIN books ON chapters.book=books.id WHERE pages.id=? AND books.account=?", (page, account)
            ).fetchone()
        if not r or r["data"] is None:
            raise RequestError("Page unavailable.", 404)
        return (r["mime"] if field == "source" else "image/png"), r["data"]

    def thumbnail(self, page, variant, account="owner"):
        # Check ownership and the current revision even on a cache hit.
        with self.db() as db:
            row = db.execute(
                "SELECT updated, length(final) AS translated FROM pages "
                "JOIN chapters ON pages.chapter=chapters.id JOIN books ON chapters.book=books.id "
                "WHERE pages.id=? AND books.account=?", (page, account)
            ).fetchone()
        if not row or (variant == "translated" and row["translated"] is None):
            raise RequestError("Page unavailable.", 404)
        revision = row["updated"] if variant == "translated" else 0
        with self.thumbnail_slots:
            return self.cached_thumbnail(page, variant, account, revision)

    def _make_thumbnail(self, page, variant, account, revision):
        _, data = self.image(page, variant, account)
        with Image.open(BytesIO(data)) as image:
            image.thumbnail((320, 448), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, "WEBP", quality=82, method=2)
            return "image/webp", output.getvalue()

    def archive_name(self, book, variant, account="owner"):
        with self.db() as db:
            row = db.execute("SELECT title FROM books WHERE id=? AND account=?", (book, account)).fetchone()
        title = row["title"] if row else "Manga"
        title = "".join(c if c.isalnum() or c in " -_." else "_" for c in title)
        # Leave space for the variant and extension on byte-limited filesystems.
        title = title.encode("utf-8")[:190].decode("utf-8", "ignore").strip(" .") or "Manga"
        return f"{title}-{variant}.zip"

    def archive(self, book, variant, account="owner"):
        archive = tempfile.TemporaryFile()
        try:
            with self.db() as db, zipfile.ZipFile(archive, "w") as z:
                rows = db.execute(
                    "SELECT chapters.id,mime,source,final FROM pages JOIN chapters ON chapters.id=pages.chapter "
                    "JOIN books ON books.id=chapters.book WHERE chapters.book=? AND books.account=? ORDER BY chapters.rowid,pages.rowid",
                    (book, account),
                )
                previous = None
                chapter = 0
                for cid, mime, source, final in rows:
                    size = (len(source) if variant != "translated" else 0) + (len(final or b"") if variant != "original" else 0)
                    require_space(archive.fileno(), size)
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

    def resolve_language(self, chapter):
        with self.db() as db:
            choice = db.execute("""SELECT chapters.book, chapters.source_language AS chapter_language,
                                          books.source_language, books.detected_language
                                   FROM chapters JOIN books ON books.id=chapters.book WHERE chapters.id=?""",
                                (chapter,)).fetchone()
            if not choice:
                raise RequestError("Chapter was removed.")
            for setting in ("chapter_language", "source_language"):
                if choice[setting] != "auto":
                    return choice[setting]
            if choice["detected_language"]:
                return choice["detected_language"]
            started = time.monotonic()
            book = choice["book"]
            pages = db.execute("""SELECT pages.id FROM pages JOIN chapters ON chapters.id=pages.chapter
                                  WHERE chapters.book=? AND chapters.source_language='auto' ORDER BY pages.rowid""",
                               (book,)).fetchall()
            # One interior page and two ends across the manga available at first import.
            indexes = dict.fromkeys((len(pages)//2, 0, len(pages)-1))
            with tempfile.TemporaryDirectory(prefix="tralens-language-") as temp:
                paths = []
                for i in indexes:
                    path = Path(temp) / str(i)
                    path.write_bytes(db.execute("SELECT source FROM pages WHERE id=?", (pages[i]["id"],)).fetchone()[0])
                    paths.append(str(path))
                env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
                result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / "tools/detect_chapter_language.py"),
                                         str(self.app.config.model_root), self.app.config.device, *paths],
                                        capture_output=True, timeout=180, env=env, text=True)
                if result.returncode:
                    raise RequestError("Language detection failed. Check the installed OCR models or choose a chapter language.")
                report = json.loads(result.stdout.strip().splitlines()[-1])
        language = chapter_language(report["language"])
        with self.db() as db:
            db.execute("UPDATE books SET detected_language=?,detection_seconds=?,language_uncertain=? WHERE id=?",
                       (language, time.monotonic()-started, int(report["fallback"]), book))
        return language

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
                    "SELECT id,chapter,source,mime,options FROM pages WHERE status='queued' ORDER BY rowid LIMIT 1"
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
                source = working_page(row["source"], row["mime"])
                settings = json.loads(row["options"])
                settings["source_language"] = self.resolve_language(row["chapter"])
                job = self.app.start(source, settings)
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
                    "UPDATE pages SET job=? WHERE id=? AND status='running'",
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
            db.execute("BEGIN IMMEDIATE")
            if final is not None:
                previous = db.execute(
                    "SELECT coalesce(length(final),0) + coalesce(length(CAST(reading AS BLOB)),0) FROM pages "
                    "WHERE id=? AND status='running'", (page,)
                ).fetchone()
                if previous is None:
                    return
                size = len(final) + len((reading or "").encode())
                require_space(self.path, size)
                self.check_capacity(db, max(0, size - previous[0]))
            db.execute(
                "UPDATE pages SET status=?,error=?,final=coalesce(?,final),reading=CASE WHEN ? THEN ? ELSE reading END, "
                "reading_width=CASE WHEN ? THEN ? ELSE reading_width END,updated=updated+1 WHERE id=? AND status='running'",
                ("ready" if final else "failed", error, final, final is not None, reading, final is not None, width, page),
            )
            if final is not None:
                self.check_capacity(db, 0)
