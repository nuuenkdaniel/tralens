import threading
import unittest
from unittest.mock import Mock

from tralens.devui.reader_server import ReaderHandler


class UploadAdmissionTests(unittest.TestCase):
    def test_busy_upload_returns_before_reading_and_control_routes_stay_available(self):
        request = Mock(path="/api/library/add")
        request.server.owner = None
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
                request.server.owner = None
                slot = request.server.upload_slot = threading.BoundedSemaphore(1)
                request._post.side_effect = ValueError("bad upload")
                with self.assertRaises(ValueError):
                    ReaderHandler.do_POST(request)
                self.assertTrue(slot.acquire(blocking=False))
