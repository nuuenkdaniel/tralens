import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tralens.translation import chatgpt, discovery, luna
from PIL import Image


class ChatGPTTests(TestCase):
    def test_stream_requires_completion_and_structured_text(self):
        delta = b'data: {"type":"response.output_text.delta","delta":"partial"}'
        for end in (b'data: [DONE]', b'data: {"type":"response.failed"}', b'data: {"type":"response.incomplete"}'):
            with self.assertRaises(RuntimeError):
                chatgpt.completed_output([delta, end])
        content = [dict(type='output_text', text='{"regions":[]}')]
        event = dict(type='response.completed', response=dict(output=[dict(type='message', content=content)]))
        self.assertEqual(chatgpt.completed_output([delta, b'data: ' + json.dumps(event).encode()]), {'regions': []})

    def test_both_pipeline_stages_use_selected_provider(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, TRALENS_FRONTIER_PROVIDER='chatgpt'):
            path = Path(tmp)/'test.png'
            Image.new('RGB', (40, 40), 'white').save(path)
            with patch.object(chatgpt, 'infer', return_value={'regions': []}) as request:
                self.assertEqual(discovery.propose(path, [], luna.MODEL, 10), [])
                self.assertEqual(len(request.call_args.args[2]), 2)
                self.assertEqual(luna.decide([], image=path, related=[path]), {})
                self.assertEqual(request.call_args.args[2], [path, path])

    def test_refresh_is_persisted_privately_and_used_once(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, XDG_CONFIG_HOME=tmp):
            with chatgpt.record():
                chatgpt.save(dict(host='host', subject='owner', client_id='client', expires_at=1,
                                  scope=chatgpt.PLAN, refresh_token='old'))
            reply = dict(token_type='Bearer', access_token='access', refresh_token='new', expires_in=600)
            with patch.object(chatgpt, 'exchange', return_value=reply) as exchange:
                self.assertEqual(chatgpt.access_token(), 'access')
                self.assertEqual(chatgpt.access_token(), 'access')
                exchange.assert_called_once()
            self.assertEqual((chatgpt.directory()/'account.json').stat().st_mode & 0o777, 0o600)
            with chatgpt.record() as saved:
                self.assertEqual(saved['refresh_token'], 'new')
            with self.assertRaises(RuntimeError):
                chatgpt.credentials({}, dict(reply, scope='openid'))

    def test_browser_callback_pkce_and_verified_owner(self):
        import base64
        import hashlib
        import http.client
        import threading
        import time
        from urllib.parse import parse_qs, urlencode, urlsplit
        from types import SimpleNamespace
        try:
            import jwt
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            self.skipTest('Optional ChatGPT sign-in dependencies are unavailable')
        signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        auth = {}
        client_id = 'oaiapp_test'

        def browser(url):
            start = urlsplit(url)
            self.assertEqual(start.hostname, '127.0.0.1')
            def visit():
                client = http.client.HTTPConnection(start.hostname, start.port)
                try:
                    client.request('GET', start.path)
                    response = client.getresponse()
                    redirect_url = response.getheader('Location')
                    response.read()
                    auth.update({k: v[0] for k, v in parse_qs(urlsplit(redirect_url).query).items()})
                    redirect = urlsplit(auth['redirect_uri'])
                    client.request('GET', redirect.path + '?' + urlencode(dict(state=auth['state'], code='code', client_id=client_id)))
                    client.getresponse().read()
                finally:
                    client.close()
            threading.Thread(target=visit, daemon=True).start()
            return True

        def token(**form):
            digest = hashlib.sha256(form['code_verifier'].encode()).digest()
            self.assertEqual(base64.urlsafe_b64encode(digest).decode().rstrip('='), auth['code_challenge'])
            self.assertEqual(form['client_id'], client_id)
            claims = dict(iss=chatgpt.ISSUER, aud=client_id, exp=int(time.time())+600, sub='owner', nonce=auth['nonce'])
            return dict(token_type='Bearer', access_token='access', refresh_token='refresh', expires_in=600,
                        scope=chatgpt.PLAN, id_token=jwt.encode(claims, signing_key, algorithm='RS256'))

        with TemporaryDirectory() as tmp, patch.dict(os.environ, XDG_CONFIG_HOME=tmp), \
                patch.object(chatgpt.webbrowser, 'open', side_effect=browser), \
                patch.object(chatgpt, 'exchange', side_effect=token), \
                patch.object(jwt.PyJWKClient, 'get_signing_key_from_jwt',
                             return_value=SimpleNamespace(key=signing_key.public_key())):
            chatgpt.login()
            self.assertEqual(auth['client_id'], 'dynamic_agent_client')
            self.assertEqual(auth['agent_name_hint'], 'Tralens')
            host = auth['ext_agent_host_id']
            auth.clear()
            chatgpt.login()
            self.assertNotIn('agent_name_hint', auth)
            self.assertEqual(auth['client_id'], client_id)
            self.assertEqual(auth['ext_agent_host_id'], host)
            self.assertTrue(auth['id_token_hint'])
            with chatgpt.record() as account:
                self.assertEqual(account['subject'], 'owner')
