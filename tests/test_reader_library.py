from dataclasses import replace
from io import BytesIO
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import time
import json
import unittest
import zipfile

from PIL import Image
from tralens.devui.library import Library
from tralens.devui.server import App, Config, RequestError


class ReaderLibraryTests(unittest.TestCase):
    def test_private_database_and_rollback_journal_on_existing_mount(self):
        path = self.library.path
        with self.library.db() as db:
            db.execute("PRAGMA user_version=17")
        self.library.close()
        path.parent.chmod(0o755)
        path.chmod(0o644)
        self.library = Library(self.app)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.library.db() as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 17)
            db.execute("BEGIN IMMEDIATE")
            db.execute("PRAGMA user_version=18")
            journal = path.with_name(path.name + "-journal")
            self.assertEqual(journal.stat().st_mode & 0o777, 0o600)

    def test_rejects_writable_mount_and_linked_library_files(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "mount"
            root.mkdir()
            root.chmod(0o777)
            with self.assertRaisesRegex(ValueError, "library directory"):
                App(Config(root, root, reader_mode=True))
            root.chmod(0o700)
            app = App(Config(root, root, reader_mode=True))
            target = base / "other"
            target.write_bytes(b"private external data")
            target.chmod(0o644)
            for name in ("library.lock", "reader.sqlite3", "reader.sqlite3-journal", "reader.sqlite3-wal", "reader.sqlite3-shm"):
                path = root / name
                for link in (os.symlink, os.link):
                    with self.subTest(name=name, link=link.__name__):
                        path.unlink(missing_ok=True)
                        link(target, path)
                        with self.assertRaises((ValueError, OSError)):
                            Library(app)
                        path.unlink()
                        self.assertEqual(target.read_bytes(), b"private external data")
                        self.assertEqual(target.stat().st_mode & 0o777, 0o644)

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.calls = []

        def runner(config, job):
            self.calls.append(job["options"])
            with Image.open(job["image"]) as page:
                output = job["directory"] / "run/rendering"
                output.mkdir(parents=True)
                page.save(output / "composite.png")
            return {"status": "complete"}

        self.app = App(
            Config(root, root, fonts=root / "fonts.json", reader_mode=True),
            runner=runner,
        )
        self.library = Library(self.app)
        self.addCleanup(lambda: self.library.close())
        page = BytesIO()
        Image.new("RGB", (60, 90), "white").save(page, "PNG")
        self.file = ("page.png", page.getvalue())
        self.meta = dict(
            title="Manga",
            chapter="One",
            options={"mode": "render", "translation_model": "gpt-6.2-sol"},
        )

    def finished(self):
        end = time.monotonic() + 8
        while time.monotonic() < end:
            book = self.library.catalog()["books"][0]
            if all(
                p["status"] == "ready" for c in book["chapters"] for p in c["pages"]
            ):
                return book
            time.sleep(0.02)
        self.fail(self.library.catalog())

    def test_translation_storage_limit_preserves_previous_page_and_releases_transaction(self):
        self.library.add(self.meta, [self.file])
        page = self.finished()["chapters"][0]["pages"][0]["id"]
        previous = self.library.image(page, "translated")
        with self.library.db() as db:
            db.execute("UPDATE pages SET status='running' WHERE id=?", (page,))
        self.app.config = replace(self.app.config, root_storage_bytes=self.library.path.stat().st_size + 1024)
        with self.assertRaisesRegex(RequestError, "storage is full"):
            self.library.finish(page, b"x" * 65536, None)
        self.library.finish(page, None, "Library storage is full.")
        self.assertEqual(self.library.catalog()["books"][0]["chapters"][0]["pages"][0]["status"], "failed")
        self.assertEqual(self.library.image(page, "translated"), previous)

    def test_thumbnail_refreshes_on_new_translation_and_preserves_original(self):
        self.library.add(self.meta, [self.file])
        page = self.finished()["chapters"][0]["pages"][0]["id"]
        original = self.library.thumbnail(page, "original")
        first = self.library.thumbnail(page, "translated")
        with self.library.db() as db:
            db.execute("UPDATE pages SET status='running' WHERE id=?", (page,))
        updated = BytesIO()
        Image.new("RGB", (1000, 1500), "blue").save(updated, "PNG")
        self.library.finish(page, updated.getvalue(), None)
        self.assertNotEqual(first, self.library.thumbnail(page, "translated"))
        with Image.open(BytesIO(self.library.thumbnail(page, "translated")[1])) as thumb:
            self.assertEqual(thumb.size, (299, 448))
        self.assertEqual(self.library.thumbnail(page, "original"), original)
        self.assertEqual(self.library.image(page, "translated")[1], updated.getvalue())

    def test_chapters_pages_position_download_and_restart(self):
        first = self.library.add(self.meta, [self.file, self.file])
        self.library.add(
            {**self.meta, **{"book": first["book"], "chapter": "Two"}}, [self.file]
        )
        book = self.finished()
        self.assertEqual([len(c["pages"]) for c in book["chapters"]], [2, 1])
        self.assertTrue(
            all(c["translation_model"] == "gpt-6.2-sol" for c in self.calls)
        )
        p = book["chapters"][1]["pages"][0]
        self.library.change("position", dict(book=book["id"], page=p["id"]))
        self.library.close()
        self.library = Library(self.app)
        self.assertEqual(self.library.catalog()["books"][0]["position"], p["id"])
        self.assertEqual(self.library.image(p["id"], "original")[1], self.file[1])
        with (
            self.library.archive(book["id"], "both") as file,
            zipfile.ZipFile(file) as z,
        ):
            self.assertEqual(len(z.namelist()), 6)
            self.assertIn("translated/chapter-002/page-0001.png", z.namelist())
        self.library.change(
            "retry",
            dict(
                book=book["id"],
                page=p["id"],
                options={"mode": "render", "translation_model": "local"},
            ),
        )
        self.finished()
        self.assertEqual(self.calls[-1]["translation_model"], "local")
        self.library.change("delete", dict(book=book["id"]))
        self.assertEqual(self.library.catalog()["books"], [])

    def test_chapter_language_is_cached_and_can_be_overridden(self):
        added = self.library.add(self.meta, [self.file, self.file])
        chapter = self.finished()["chapters"][0]
        self.assertEqual((chapter["source_language"], chapter["detected_language"]), ("auto", "ja"))
        self.assertTrue(chapter["language_uncertain"])  # The test pages are blank.
        duration = chapter["detection_seconds"]
        self.assertGreater(duration, 0)
        change = {**added, "source_language": "ko"}
        with self.assertRaises(RequestError):
            self.library.change("chapter-language", change, "another-account")
        self.library.change("chapter-language", change)
        self.finished()
        self.assertEqual([call["source_language"] for call in self.calls], ["ja", "ja", "ko", "ko"])
        self.library.change("chapter-language", {**change, "source_language": "auto"})
        chapter = self.finished()["chapters"][0]
        self.assertEqual(chapter["detection_seconds"], duration)
        self.assertEqual(self.calls[-1]["source_language"], "ja")
        with self.assertRaises(RequestError):
            self.library.change("chapter-language", {**change, "source_language": "xx"})

    def test_explicit_chinese_skips_detection(self):
        self.library.add({**self.meta, "source_language": "zh"}, [self.file])
        chapter = self.finished()["chapters"][0]
        self.assertIsNone(chapter["detection_seconds"])
        self.assertEqual(self.calls[-1]["source_language"], "zh")

    def test_archive_name_uses_current_title_and_fits_android_downloads(self):
        added = self.library.add(self.meta, [self.file])
        book = added["book"]
        self.library.change("rename", {"book": book, "title": "夜空 / Manga"})
        self.assertEqual(self.library.archive_name(book, "both"), "夜空 _ Manga-both.zip")
        self.library.change("rename", {"book": book, "title": "夜" * 180})
        self.assertLessEqual(len(self.library.archive_name(book, "translated").encode()), 255)

    def test_lost_import_response_can_be_retried_once_per_account(self):
        key = "23" * 16
        meta = dict(self.meta, import_id=key, options={"mode": "render", "translation_model": "local"})
        self.assertIsNone(self.library.import_result("owner", key))
        with self.assertRaises(RequestError):
            self.library.add(meta, [("bad.pdf", b"%PDF-broken")])
        self.assertIsNone(self.library.import_result("owner", key))
        receipt = self.library.add(meta, [self.file])
        self.assertEqual(self.library.add(meta, [self.file]), receipt)
        self.assertEqual(self.library.import_result("owner", key), receipt)
        self.assertIsNone(self.library.import_result("second", key))
        second = self.library.add(meta, [self.file], "second")
        self.assertNotEqual(second, receipt)
        self.assertEqual(len(self.library.catalog()["books"]), 1)
        self.assertEqual(len(self.library.catalog()["books"][0]["chapters"]), 1)

    def test_pdf_document_order_and_webp_original(self):
        pdf, webp = BytesIO(), BytesIO()
        red = Image.new("RGB", (60, 90), "red")
        blue = Image.new("RGB", (100, 50), "blue")
        red.save(pdf, format="PDF", save_all=True, append_images=[blue])
        Image.new("RGBA", (12, 16), (0, 0, 0, 0)).save(webp, format="WEBP")
        self.library.add(self.meta, [("chapter.pdf", pdf.getvalue()), ("extra.webp", webp.getvalue())])
        book = self.finished()
        pages = book["chapters"][0]["pages"]
        self.assertEqual([p["name"] for p in pages], ["chapter-001.png", "chapter-002.png", "extra.webp"])
        for page, channel in zip(pages, (0, 2)):
            mime, content = self.library.image(page["id"], "original")
            self.assertEqual(mime, "image/png")
            with Image.open(BytesIO(content)) as im:
                self.assertGreater(im.getpixel((10, 10))[channel], 240)
                self.assertEqual(max(im.size), 2000)
        self.assertEqual(self.library.image(pages[2]["id"], "original"), ("image/webp", webp.getvalue()))
        with Image.open(BytesIO(self.library.image(pages[2]["id"], "translated")[1])) as converted:
            self.assertEqual(converted.getpixel((0, 0)), (255, 255, 255))
        with self.library.archive(book["id"], "original") as file, zipfile.ZipFile(file) as z:
            self.assertEqual(z.read("original/chapter-001/page-0003.webp"), webp.getvalue())

    def test_archives_use_natural_path_order_and_ignore_metadata(self):
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in ("Ch10/Page1.PNG", "__MACOSX/._page.png", "Ch2/Page10.png",
                         "ComicInfo.xml", "Ch2/Page2.png", ".extras/cover.png"):
                archive.writestr(path, self.file[1])
        for extension in ("cbz", "zip"):
            self.library.add(self.meta, [("chapter." + extension, stream.getvalue())])
            book = self.finished()
            self.assertEqual([p["name"] for p in book["chapters"][0]["pages"]],
                             ["Page2.png", "Page10.png", "Page1.PNG"])
            self.library.change("delete", {"book":book["id"]})

    def test_partial_job_with_composite_is_readable_without_warning(self):
        runner = self.app.runner
        def partial(config, job):
            runner(config, job)
            return {"status": "partial"}
        self.app.runner = partial
        self.library.add(self.meta, [self.file])
        page = self.finished()["chapters"][0]["pages"][0]
        self.assertTrue(page["ready"])
        self.assertIsNone(page["error"])

    def test_malformed_pdf_rolls_back_mixed_import(self):
        with self.assertRaises(RequestError):
            self.library.add(self.meta, [self.file, ("bad.pdf", b"%PDF-broken")])
        self.assertEqual(self.library.catalog()["books"], [])

    def test_invalid_import_rolls_back_and_unconfigured_model_rejected(self):
        with self.assertRaises(RequestError):
            self.library.add(self.meta, [self.file, ("bad.jpg", b"bad")])
        with self.assertRaises(RequestError):
            self.library.add(
                {
                    **self.meta,
                    "options": {"mode": "render", "reconstruction_policy": "sdxl"},
                },
                [self.file],
            )
        self.assertFalse(self.library.catalog()["books"])

    def test_phone_transcript_survives_failed_retry(self):
        runner = self.app.runner
        def with_reading(config, job):
            result = runner(config, job)
            run = job['directory'] / 'run'
            (run / 'preview').mkdir()
            raw = dict(source_id=1, polygon=[[1, 2], [8, 2], [8, 9]])
            (run / 'preview/preview.json').write_text(json.dumps({'ocr': {'raw_detections': [[raw]]}}))
            unit = dict(binding=dict(text="Hello!", member_ids=[1]), status='automatic_composed',
                        layout={'size': 20}, display_status='clean')
            (run / 'rendering/manifest.json').write_text(json.dumps({'units':[unit], 'source':{'size':[1000,1500]}}))
            return result
        self.app.runner = with_reading
        self.library.add(self.meta, [self.file])
        b = self.finished()
        p = b['chapters'][0]['pages'][0]
        self.assertEqual(p['reading_width'], 600)
        self.assertEqual(self.library.transcript(p['id']), [dict(text='Hello!', box=[1, 2, 8, 9])])
        self.app.runner = lambda *_: {'status': 'failed', 'error': {'message': 'provider unavailable'}}
        self.library.change('retry', dict(book=b['id'], page=p['id'], options=self.meta['options']))
        for _ in range(100):
            page = self.library.catalog()['books'][0]['chapters'][0]['pages'][0]
            if page['status'] == 'failed':
                break
            time.sleep(.02)
        self.assertEqual(page['status'], 'failed')
        self.assertTrue(page['ready'])
        self.assertEqual(self.library.transcript(p['id'])[0]['text'], 'Hello!')

    def test_fenced_local_page_waits_while_frontier_page_completes(self):
        from tralens.scheduling import remote_state_path
        from unittest.mock import patch
        with patch.dict('os.environ', TRALENS_STATE_ROOT=str(self.app.root / 'state')):
            # Start a new worker so its provider state is isolated for this test.
            self.library.close()
            fence = remote_state_path(self.app.config.endpoint)
            fence.write_text('{}')
            self.library = Library(self.app)
            local = self.library.add({**self.meta, 'options':{'mode':'render'}}, [self.file])
            self.library.add(self.meta, [self.file])
            for _ in range(150):
                pages = [b['chapters'][0]['pages'][0] for b in self.library.catalog()['books']]
                if [p['status'] for p in pages] == ['waiting','ready']:
                    break
                time.sleep(.02)
            self.assertEqual([p['status'] for p in pages], ['waiting','ready'])
            self.assertEqual(len(self.calls), 1)
            self.library.change('skip', dict(book=local['book'], page=pages[0]['id']))
            self.assertTrue(fence.exists())
            self.assertEqual(self.library.catalog()['books'][0]['chapters'][0]['pages'][0]['status'], 'skipped')

    def test_library_has_one_worker_owner(self):
        with self.assertRaisesRegex(RuntimeError, "already owns"):
            Library(self.app)
