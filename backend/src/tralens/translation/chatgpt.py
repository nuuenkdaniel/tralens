"""Owner-managed ChatGPT plan connection for the self-hosted Linux backend."""
import base64
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from contextlib import contextmanager
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid
import webbrowser

import requests

ISSUER = 'https://auth.openai.com'
RESOURCE = 'https://api.openai.com/v1'
TOKEN = ISSUER + '/api/accounts/oauth/token'
PLAN = 'chatgpt.tokens.use.direct'


def directory():
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'tralens' / 'chatgpt'


@contextmanager
def record():
    root = directory()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077 or root.stat().st_uid != os.getuid():
        raise RuntimeError('ChatGPT configuration directory must be owned by this user and mode 0700.')
    with (root / 'account.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'account.json'
        yield json.loads(path.read_text()) if path.exists() else {}


def save(data):
    with tempfile.NamedTemporaryFile(mode='w', dir=directory(), delete=False) as file:
        try:
            json.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            os.unlink(file.name)
            raise
    os.replace(file.name, directory() / 'account.json')


class SignInRequired(RuntimeError):
    pass


def exchange(**form):
    reply = requests.post(TOKEN, data=dict(resource=RESOURCE, **form), timeout=30, allow_redirects=False)
    if (form.get('grant_type') == 'refresh_token' and reply.status_code in (400, 401)
            and reply.json().get('error') == 'invalid_grant'):
        raise SignInRequired('Sign in to ChatGPT again.')
    if reply.status_code != 200:
        raise RuntimeError(f'ChatGPT token exchange failed (HTTP {reply.status_code}). Run chatgpt.py login.')
    return reply.json()


def credentials(old, reply):
    scopes = reply.get('scope', old.get('scope', ''))
    if PLAN not in scopes.split() or reply.get('token_type', '').lower() != 'bearer':
        raise RuntimeError('ChatGPT plan permission was not granted.')
    refresh = reply.get('refresh_token') or old.get('refresh_token')
    access = reply.get('access_token')
    if not access or not refresh or int(reply.get('expires_in', 0)) <= 0:
        raise RuntimeError('ChatGPT returned incomplete credentials.')
    return dict(old, access_token=access, refresh_token=refresh, scope=scopes,
                expires_at=time.time() + int(reply['expires_in']))


def access_token():
    with record() as account:
        if not account.get('refresh_token'):
            raise RuntimeError('Connect the server owner account with backend/chatgpt.py login.')
        if account['expires_at'] <= time.time() + 60:
            reply = exchange(grant_type='refresh_token', refresh_token=account['refresh_token'],
                             client_id=account['client_id'])
            account = credentials(account, reply)
            save(account)
        return account['access_token']


def finish_login(result, old, redirect, verifier, nonce):
    import jwt
    if result.get('error') or not result.get('code'):
        raise RuntimeError('OpenAI did not grant this connection. Check the consent screen and try again.')
    client = result.get('client_id', old.get('client_id'))
    if not client or client == 'dynamic_agent_client' or old.get('client_id', client) != client:
        raise RuntimeError('ChatGPT registration did not match the pending sign-in.')
    reply = exchange(grant_type='authorization_code', client_id=client, code=result['code'],
                     redirect_uri=redirect, code_verifier=verifier)
    token = reply['id_token']
    key = jwt.PyJWKClient(ISSUER + '/.well-known/jwks.json').get_signing_key_from_jwt(token)
    claims = jwt.decode(token, key.key, algorithms=['RS256'], audience=client, issuer=ISSUER,
                        options={'require': ['exp', 'sub', 'nonce']})
    if claims['nonce'] != nonce or old.get('subject', claims['sub']) != claims['sub']:
        raise RuntimeError('ChatGPT identity did not match the pending sign-in.')
    save(dict(credentials({}, reply), host=old['host'], client_id=client, subject=claims['sub'],
              email=claims.get('email', ''), id_token=token))


def result_page(connected, nonce):
    content = Path(__file__).with_name('signin-result.html').read_text()
    fields = dict(nonce=nonce, state='success' if connected else 'error',
                  title='ChatGPT connected' if connected else 'Sign-in incomplete',
                  symbol='✓' if connected else '!',
                  message='Your account is connected to this Tralens server. You can close this tab.'
                          if connected else 'Return to Tralens and start sign-in again to connect your account.')
    for name, value in fields.items():
        content = content.replace('{{' + name + '}}', value)
    return content.encode()


def callback_handler(state, result):
    class Callback(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            pass

        def do_GET(self):
            url = urlsplit(self.path)
            if (self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'
                    and url.path == getattr(self.server, 'start_path', None)):
                self.send_response(302)
                self.send_header('Location', self.server.authorize_url)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.end_headers()
                return
            fields = parse_qs(url.query, max_num_fields=20)
            valid = (self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'
                     and url.path == '/auth/callback' and fields.get('state') == [state]
                     and all(len(values) == 1 for values in fields.values()))
            if valid:
                result.update({key: values[0] for key, values in fields.items()})
                try:
                    self.server.finish_login(result)
                except Exception as exc:
                    result['failure'] = str(exc) or 'Could not complete ChatGPT sign-in.'
            connected = valid and 'failure' not in result
            nonce = secrets.token_urlsafe(18)
            body = result_page(connected, nonce)
            self.send_response(200 if connected else 400)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(body)
    return Callback


def authorization(old, port, state, *, include_hint):
    nonce, verifier = (secrets.token_urlsafe(48) for _ in range(2))
    if 'host' not in old:
        old['host'] = 'urn:uuid:' + str(uuid.uuid4())
        save(old)
    redirect = f'http://127.0.0.1:{port}/auth/callback'
    params = dict(client_id=old.get('client_id', 'dynamic_agent_client'),
                  ext_agent_host_id=old['host'], response_type='code', redirect_uri=redirect,
                  scope='openid profile email offline_access resource.invoke ' + PLAN,
                  resource=RESOURCE, state=state, nonce=nonce, code_challenge_method='S256',
                  code_challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('='))
    if 'client_id' not in old:
        params['agent_name_hint'] = 'Tralens'
    elif include_hint and old.get('id_token'):
        params['id_token_hint'] = old['id_token']
    return ISSUER + '/api/accounts/authorize?' + urlencode(params), redirect, nonce, verifier


def login(*, browser=True, port=0, on_ready=None):
    import jwt
    state = secrets.token_urlsafe(48)
    result = {}
    with record() as old, HTTPServer(('127.0.0.1', port), callback_handler(state, result)) as server:
        url, redirect, nonce, verifier = authorization(old, server.server_port, state, include_hint=True)
        server.finish_login = lambda fields: finish_login(fields, old, redirect, verifier, nonce)
        server.authorize_url = url
        server.start_path = '/connect/' + secrets.token_urlsafe(32)
        start_url = f'http://127.0.0.1:{server.server_port}{server.start_path}'
        if on_ready is not None:
            on_ready(start_url)
        elif browser and webbrowser.open(start_url):
            print('Continue with ChatGPT in your browser.', flush=True)
        else:
            print('Continue with ChatGPT at ' + start_url, flush=True)
        deadline = time.monotonic() + 300
        server.timeout = 1
        while not result and time.monotonic() < deadline:
            server.handle_request()
        if not result:
            raise RuntimeError('Sign-in expired before a callback arrived. Run login again.')
        if result.get('failure'):
            raise RuntimeError(result['failure'])


def logout():
    with record() as account:
        if account.get('refresh_token'):
            config = requests.get(ISSUER + '/.well-known/openid-configuration', timeout=20)
            config.raise_for_status()
            endpoint = config.json()['revocation_endpoint']
            if urlsplit(endpoint).scheme != 'https' or urlsplit(endpoint).netloc != 'auth.openai.com':
                raise RuntimeError('Unexpected OpenAI revocation endpoint.')
            response = requests.post(endpoint, data=dict(token=account['refresh_token'], token_type_hint='refresh_token',
                                     client_id=account['client_id']), timeout=20, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError('Revocation was not confirmed. Retry logout or disconnect Tralens in ChatGPT Settings.')
            save({k: account[k] for k in ('host', 'client_id', 'subject', 'email')})


def models():
    with requests.get(RESOURCE + '/models', headers={'Authorization': 'Bearer ' + access_token()},
                      timeout=30, allow_redirects=False) as reply:
        if reply.status_code == 401:
            raise SignInRequired('Sign in to ChatGPT again.')
        if reply.status_code != 200:
            raise RuntimeError(f'ChatGPT model list unavailable (HTTP {reply.status_code}).')
        return [m['slug'] for m in reply.json()['models'] if m.get('visibility') == 'list']


def completed_output(lines):
    for line in lines:
        if not line.startswith(b'data:'):
            continue
        value = line[5:].strip()
        if value == b'[DONE]':
            break
        event = json.loads(value)
        kind = event.get('type')
        if kind in ('response.failed', 'response.incomplete', 'error'):
            error = event.get('response', {}).get('error') or event.get('error') or {}
            raise RuntimeError(f"ChatGPT request ended with {error.get('code') or kind}.")
        if kind == 'response.completed':
            output = event['response']['output']
            message = ''.join(c['text'] for item in output if item['type'] == 'message'
                              for c in item['content'] if c['type'] == 'output_text')
            if not message:
                raise RuntimeError('ChatGPT returned no translated text.')
            return json.loads(message)
    raise RuntimeError('ChatGPT stream ended before completion.')


def infer(model, prompt, images, schema, timeout):
    from PIL import Image
    if model not in models():
        raise RuntimeError(f'{model} is unavailable for this ChatGPT account.')
    content = [dict(type='input_text', text=prompt)]
    for path in images:
        with Image.open(path) as image:
            mime = Image.MIME[image.format]
        data = base64.b64encode(Path(path).read_bytes()).decode()
        content.append(dict(type='input_image', image_url=f'data:{mime};base64,{data}'))
    body = dict(model=model, input=[dict(role='user', content=content)], store=False, stream=True,
                text={'format': dict(type='json_schema', name='lettering', strict=True, schema=schema)})
    with requests.post(RESOURCE + '/responses', headers={'Authorization': 'Bearer ' + access_token()},
                       json=body, timeout=timeout, stream=True, allow_redirects=False) as reply:
        if reply.status_code != 200:
            raise RuntimeError(f'ChatGPT inference failed (HTTP {reply.status_code}).')
        return completed_output(reply.iter_lines())
