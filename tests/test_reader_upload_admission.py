import threading
import unittest
from unittest.mock import Mock, patch

from tralens.devui.reader_server import ReaderHandler


class UploadAdmissionTests(unittest.TestCase):
    def test_busy_upload_returns_before_reading_and_control_routes_stay_available(self):
        request = Mock(path="/api/library/add")
        request.server.accounts.identify.return_value = "owner"
        slot = request.server.upload_slot = threading.BoundedSemaphore(1)
        slot.acquire()
        ReaderHandler.do_POST(request)
        request._post.assert_not_called()
        self.assertEqual(request.json.call_args.args[0], 429)
        self.assertTrue(request.close_connection)
        request.path = "/api/jobs/" + "1" * 32 + "/cancel"
        ReaderHandler.do_POST(request)
        request._post.assert_called_once_with(request.path)

    def test_invalid_upload_releases_both_entry_points(self):
        for path in ("/api/jobs", "/api/library/add"):
            with self.subTest(path=path):
                request = Mock(path=path)
                request.server.accounts.identify.return_value = "owner"
                slot = request.server.upload_slot = threading.BoundedSemaphore(1)
                request._post.side_effect = ValueError("bad upload")
                with self.assertRaises(ValueError):
                    ReaderHandler.do_POST(request)
                self.assertTrue(slot.acquire(blocking=False))


class MultipartUploadTests(unittest.TestCase):
    def test_low_space_rejected_before_upload_body_read(self):
        from tralens.devui.multipart_upload import read_import
        from tralens.devui.server import RequestError

        stream = Mock()
        with patch("tralens.devui.import_pages.os.statvfs", return_value=Mock(f_bavail=1, f_frsize=4096)):
            with self.assertRaises(RequestError) as failure:
                with read_import(stream, 1024, "multipart/form-data; boundary=test"):
                    self.fail("Upload accepted on a full volume")
        self.assertEqual(failure.exception.status, 507)
        stream.read.assert_not_called()

    def test_spools_raw_pages_and_rejects_truncated_form(self):
        from io import BytesIO
        from tralens.devui.multipart_upload import read_import
        from tralens.devui.server import RequestError

        class LimitedReads(BytesIO):
            def read(self, size=-1):
                if not 0 < size <= 256 * 1024:
                    raise AssertionError('Upload read exceeded the streaming buffer.')
                return super().read(size)

        content = b'contents' * 100000
        prefix = (b'--test\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{}\r\n'
                  b'--test\r\nContent-Disposition: form-data; name="page"; filename="chapter.cbz"\r\n\r\n')
        body = prefix + content + b'\r\n--test--\r\n'
        mime = 'multipart/form-data; boundary=test'
        with read_import(LimitedReads(body), len(body), mime) as (meta, files):
            self.assertEqual(meta, {})
            [(name, staged)] = files
            self.assertEqual(name, 'chapter.cbz')
            self.assertEqual(staged.read_bytes(), content)
        self.assertFalse(staged.exists())
        with self.assertRaises(RequestError):
            with read_import(LimitedReads(body[:-12]), len(body) - 12, mime):
                self.fail('An incomplete form reached the library.')
