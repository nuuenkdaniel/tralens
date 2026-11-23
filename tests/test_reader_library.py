from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import json
import unittest
import zipfile

from PIL import Image
from tralens.devui.library import Library
from tralens.devui.server import App, Config, RequestError


class ReaderLibraryTests(unittest.TestCase):
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
