from http.client import HTTPConnection
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock

from tralens.devui.owner import Owner
from tralens.devui.reader_server import create_reader_server
from tralens.devui.server import App, Config


class OwnerReaderTests(unittest.TestCase):
    def test_reader_session_and_api_boundaries(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = root / 'owner-key'
            key.write_text('s' * 48)
            key.chmod(0o600)
            owner = Owner('https://reader.example', key)
            app = App(Config(root/'jobs', root, reader_mode=True), runner=Mock())
            with create_reader_server(app, 0, host='127.0.0.1', owner=owner) as http:
                thread = threading.Thread(target=http.serve_forever, daemon=True)
                thread.start()
                def call(method, path, body=None, **headers):
                    connection = HTTPConnection('127.0.0.1', http.server_port, timeout=5)
                    try:
                        connection.request(method, path, body, {'Host': owner.host, **headers})
                        response = connection.getresponse()
                        return response.status, response.getheader('Set-Cookie'), response.read()
                    finally:
                        connection.close()
                try:
                    self.assertEqual(call('GET', '/')[0], 200)
                    for path in ['/developer', '/api/config', '/api/jobs', '/api/library',
                                 '/api/library/pages/' + '0'*32 + '/original',
                                 '/api/library/books/' + '0'*32 + '/both.zip']:
                        self.assertEqual(call('GET', path)[0], 401, path)
                    self.assertEqual(call('POST', '/api/jobs', 'x')[0], 401)
                    self.assertEqual(call('POST', '/api/owner/login', key.read_text(), Origin='https://other.example')[0], 403)
                    self.assertEqual(call('POST', '/api/owner/login', 'bad', Origin=owner.origin)[0], 401)
                    status, cookie, _ = call('POST', '/api/owner/login', key.read_text(), Origin=owner.origin)
                    self.assertEqual(status, 204)
                    self.assertIn('HttpOnly', cookie)
                    self.assertIn('Secure', cookie)
                    cookie = cookie.split(';')[0]
                    self.assertTrue(json.loads(call('GET', '/api/config', Cookie=cookie)[2])['owner_mode'])
                    self.assertEqual(call('GET', '/api/library', Cookie=cookie)[0], 200)
                    self.assertEqual(call('POST', '/api/owner/logout', Origin=owner.origin, Cookie=cookie)[0], 204)
                    self.assertEqual(call('GET', '/api/library', Cookie=cookie)[0], 401)
                finally:
                    http.shutdown()
                    thread.join()


if __name__ == '__main__':
    unittest.main()
