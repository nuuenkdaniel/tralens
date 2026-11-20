import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tralens.devui.chatgpt_connection import ReaderConnection
from tralens.translation import chatgpt, frontier, luna


class ReaderChatGPTTests(unittest.TestCase):
    def test_first_selection_waits_for_consent_then_reads_account_catalog(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, XDG_CONFIG_HOME=home):
            connection = ReaderConnection()
            self.assertEqual(frontier.provider(), 'ChatGPT')
            self.assertEqual(connection.status()['state'], 'disconnected')
            self.assertFalse(chatgpt.directory().exists())
            ready, consent = threading.Event(), threading.Event()

            def login(*, on_ready):
                on_ready('http://127.0.0.1:12345/connect/test')
                ready.set()
                if not consent.wait(3):
                    raise RuntimeError('consent expired')
                with chatgpt.record():
                    chatgpt.save({'refresh_token': 'synthetic'})

            with patch.object(chatgpt, 'login', side_effect=login) as signin, \
                    patch.object(chatgpt, 'models', return_value=[luna.MODEL]) as catalog:
                self.assertEqual(connection.select(remote=False)['state'], 'pending')
                self.assertTrue(ready.wait(2))
                self.assertEqual(connection.select(remote=False)['url'], 'http://127.0.0.1:12345/connect/test')
                signin.assert_called_once()
                catalog.assert_not_called()
                consent.set()
                deadline = time.monotonic() + 3
                while connection.status()['state'] == 'pending' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(connection.select(remote=False), dict(state='connected', models=[luna.MODEL]))
                signin.assert_called_once()

    def test_legacy_transport_is_explicit(self):
        with patch.dict(os.environ, TRALENS_FRONTIER_PROVIDER='codex'):
            self.assertEqual(frontier.provider(), 'Codex CLI')
