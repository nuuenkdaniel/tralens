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
                self.assertEqual(connection.select()['state'], 'pending')
                self.assertTrue(ready.wait(2))
                self.assertEqual(connection.select()['url'], 'http://127.0.0.1:12345/connect/test')
                signin.assert_called_once()
                catalog.assert_not_called()
                consent.set()
                deadline = time.monotonic() + 3
                while connection.status()['state'] == 'pending' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(connection.select(), dict(state='connected', models=[luna.MODEL]))
                signin.assert_called_once()

    def test_disconnect_revokes_the_saved_connection(self):
        connection = ReaderConnection()
        with patch.object(chatgpt, 'logout') as logout:
            self.assertEqual(connection.disconnect(), dict(state='disconnected'))
        logout.assert_called_once_with()

    def test_legacy_transport_is_explicit(self):
        with patch.dict(os.environ, TRALENS_FRONTIER_PROVIDER='codex'):
            self.assertEqual(frontier.provider(), 'Codex CLI')

    def test_phone_callback_is_bound_to_its_port_and_consumed_once(self):
        from urllib.parse import urlsplit, parse_qs
        from tralens.devui.server import RequestError
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, XDG_CONFIG_HOME=home):
            connection = ReaderConnection()
            with self.assertRaises(RequestError):
                connection.select_phone('49200')
            start = connection.select_phone(49200)
            auth = parse_qs(urlsplit(start['url']).query)
            self.assertEqual(auth['redirect_uri'], ['http://127.0.0.1:49200/auth/callback'])
            self.assertNotIn('code_verifier', auth)
            with self.assertRaises(RequestError):
                connection.finish_phone(dict(state='different'))
            callback = dict(state=auth['state'][0], client_id='oaiapp_test', code='synthetic')
            def finish(fields, old, redirect, verifier, nonce):
                import hashlib, base64
                self.assertEqual(fields, callback)
                self.assertEqual(auth['nonce'], [nonce])
                self.assertEqual(auth['redirect_uri'], [redirect])
                self.assertEqual(auth['code_challenge'], [base64.urlsafe_b64encode(hashlib.sha256(
                    verifier.encode()).digest()).decode().rstrip('=')])
                chatgpt.save(dict(old, refresh_token='synthetic'))
            with patch.object(chatgpt, 'finish_login', side_effect=finish), patch.object(chatgpt, 'models', return_value=[luna.MODEL]):
                self.assertEqual(connection.finish_phone(callback), dict(state='connected', models=[luna.MODEL]))
                with self.assertRaises(RequestError):
                    connection.finish_phone(callback)
                self.assertEqual(connection.select_phone(49201)['state'], 'connected')
