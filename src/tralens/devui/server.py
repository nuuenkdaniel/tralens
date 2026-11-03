"""Minimal loopback-only pipeline viewer, not a production upload service.

Polls the real pipeline's atomic checkpoints; no duplicate stage algorithms and
no simulated progress. Models are initialized only by an explicit uploaded run.
"""
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid


STATIC = Path(__file__).with_name('static')
MAX_UPLOAD = 32*1024*1024
MAX_JOBS = 8
ARTIFACT_SUFFIXES = {'.png', '.jpg', '.jpeg', '.json', '.jsonl', '.txt', '.html', '.bin'}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


@dataclass(frozen=True)
class Config:
    root: Path
    model_root: Path
    device: str = 'cpu'
    fonts: Path | None = None
    font_id: str | None = None
    model: str = 'translategemma:27b'
    profile: str = 'translategemma'
    endpoint: str = 'http://127.0.0.1:11434'


def options(value, *, render_enabled):
    if type(value) is not dict or set(value) - {'mode', 'max_attempts', 'allow_fragments', 'render_policy'}:
        raise RequestError('Unknown run options.')
    result = dict(mode='render' if render_enabled else 'ocr', max_attempts=2,
        allow_fragments=False, render_policy='best-effort')
    result.update(value)
    if type(result['mode']) is not str or result['mode'] not in ('ocr', 'translate', 'render'):
        raise RequestError('Choose OCR, translation, or rendering.')
    if result['mode'] == 'render' and not render_enabled:
        raise RequestError('Start the server with --fonts to enable rendering.')
    if type(result['max_attempts']) is not int or not 0 <= result['max_attempts'] <= 20:
        raise RequestError('Translation call budget must be 0–20.')
    if type(result['allow_fragments']) is not bool:
        raise RequestError('Fragment preview option must be a boolean.')
    if type(result['render_policy']) is not str or result['render_policy'] not in ('best-effort', 'conservative'):
        raise RequestError('Unknown rendering policy.')
    return result


def checkpoint(path, notices):
    """Read published checkpoints only; never interpret missing data as success."""
    try:
        if path.is_symlink():
            raise ValueError('symlink checkpoint')
        with path.open('rb') as stream:
            content = stream.read(16*1024*1024+1)
        if len(content) > 16*1024*1024:
            raise ValueError('checkpoint exceeds viewer detail limit')
        value = json.loads(content)
        if type(value) is not dict:
            raise ValueError('expected a checkpoint object')
        return value
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        notices.append(f'{path.name}: {error}')
        return None


def selected_text(unit):
    from ..translation.contracts import revision
    return next((r['text'] for r in unit.get('readings', [])
        if revision(r) == unit.get('selected_revision')), '')


def execute_job(config, job):
    """Only this worker invokes the existing coordinator/workflow."""
    settings = job['options']
    kwargs = dict(device=config.device, model_root=config.model_root, run_local_ocr=True,
        run_local_model=settings['mode'] != 'ocr', model=config.model, profile=config.profile,
        endpoint=config.endpoint, max_attempts=settings['max_attempts'],
        allow_fragment_previews=settings['allow_fragments'])
    if settings['mode'] == 'render':
        from main import run_pipeline
        return run_pipeline(job['image'], job['directory']/'run', fonts=config.fonts,
            font_id=config.font_id, render_policy=settings['render_policy'], **kwargs)
    from ..translation.workflow import run_preview
    return run_preview(job['image'], job['directory']/'run'/'preview', **kwargs)


class App:
    def __init__(self, config, *, runner=execute_job):
        self.config, self.runner = config, runner
        self.root = Path(config.root).expanduser().absolute()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError('Development job root must not use symlinks.')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.jobs, self.active = {}, None

    def public_config(self):
        return dict(token=self.token, device=self.config.device, model=self.config.model,
            model_root=str(self.config.model_root), fonts=str(self.config.fonts) if self.config.fonts else None,
            render_enabled=self.config.fonts is not None, max_upload_bytes=MAX_UPLOAD,
            page_pixel_limit=4_000_000, max_jobs=MAX_JOBS)

    def list_jobs(self):
        with self.lock:
            return [dict(id=job['id'], status=job['status'], mode=job['options']['mode']) for job in self.jobs.values()]

    def start(self, data, supplied):
        settings = options(supplied, render_enabled=self.config.fonts is not None)
        if not 0 < len(data) <= MAX_UPLOAD:
            raise RequestError('Upload must be 1 byte–32 MiB.')
        with self.lock:
            if self.active is not None:
                raise RequestError('A job is already running. Wait for it to finish.', 409)
            if len(self.jobs) >= MAX_JOBS:
                raise RequestError('Session job limit reached. Restart the server to run more jobs.', 429)
            from PIL import Image
            from ..rendering.preparation import source_rgb
            try:
                with Image.open(BytesIO(data)) as opened:
                    if opened.format not in ('PNG', 'JPEG'):
                        raise ValueError('Use a PNG or JPEG image.')
                    extension = 'png' if opened.format == 'PNG' else 'jpg'
                    with source_rgb(opened):
                        size = list(opened.size)
            except (ValueError, OSError, Image.DecompressionBombError) as error:
                raise RequestError(str(error)) from error
            identity = uuid.uuid4().hex
            directory = self.root/identity
            directory.mkdir(mode=0o700)
            image = directory/('input.'+extension)
            with image.open('xb') as stream:
                stream.write(data)
            # Non-rendering workflow expects an existing output parent too.
            if settings['mode'] != 'render':
                (directory/'run').mkdir(mode=0o700)
            job = dict(id=identity, directory=directory, image=image, size=size,
                options=settings, status='running', started=time.monotonic(), error=None, done=threading.Event())
            self.jobs[identity] = job
            self.active = identity
            thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            try:
                thread.start()
            except RuntimeError:
                self.active = None
                job.update(status='failed', error='Could not start worker.', ended=time.monotonic())
                job['done'].set()
                raise
        return identity

    def _run(self, job):
        try:
            report = self.runner(self.config, job)
            status = report['status']
            error = report.get('error')
        except (Exception, KeyboardInterrupt) as failure:
            status, error = 'failed', dict(type=type(failure).__name__, message=str(failure)[:2000])
        with self.lock:
            job.update(status=status, error=error, ended=time.monotonic())
            self.active = None
            job['done'].set()

    def job(self, identity):
        with self.lock:
            if not re.fullmatch('[0-9a-f]{32}', identity) or identity not in self.jobs:
                raise RequestError('Job not found.', 404)
            return dict(self.jobs[identity])

    def artifact(self, identity, relative):
        job = self.job(identity)
        parts = relative.split('/')
        if not re.fullmatch('[A-Za-z0-9._/-]+', relative) or any(p in ('', '.', '..') for p in parts):
            raise RequestError('Artifact not found.', 404)
        path = job['directory'].joinpath(*parts)
        if path.suffix.lower() not in ARTIFACT_SUFFIXES or any(p.startswith('.') for p in parts):
            raise RequestError('Artifact not found.', 404)
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise RequestError('Artifact not found.', 404)
        if not path.resolve().is_relative_to(job['directory']):
            raise RequestError('Artifact not found.', 404)
        return path

    def snapshot(self, identity):
        job = self.job(identity)
        directory = job['directory']
        notices = []
        pipeline = checkpoint(directory/'run/pipeline.json', notices)
        preview = checkpoint(directory/'run/preview/preview.json', notices)
        rendering = checkpoint(directory/'run/rendering/manifest.json', notices)
        preview = preview or {}
        rendering = rendering or {}
        raw = (preview.get('ocr') or {}).get('raw_detections') or [[]]
        detections = [dict(source_id=d['source_id'], text=d.get('text', ''),
            confidence=d.get('confidence'), polygon=d['polygon']) for d in raw[0]][:512]
        rendered = {r['unit_id']: r for r in rendering.get('units', [])}
        units = []
        for record in preview.get('units', [])[:512]:
            unit = record['unit']
            result = record.get('result') or {}
            row = rendered.get(unit['unit_id'], {})
            fragments = [dict(status=f['status'], translations=(f.get('result') or {}).get('fragments', []))
                for f in record.get('fragment_previews', [])]
            units.append(dict(id=unit['unit_id'], member_ids=unit['member_ids'],
                source=selected_text(unit), status=record['status'],
                target=dict(result.get('translations', [])).get(unit['unit_id']),
                reasons=record.get('blocking_reasons', []) +
                    [v for v in (result.get('failure'), result.get('detail')) if v] +
                    [reason for outcome in result.get('outcomes', []) for reason in outcome.get('reasons', [])],
                fragments=fragments, render_status=row.get('status'), render_reason=row.get('reason'),
                quality=row.get('render_quality', 'conservative') if row.get('status') == 'automatic_composed' else None,
                warnings=(row.get('best_effort') or {}).get('warnings', [])))
        if job['status'] == 'running':
            stage = 'rendering' if pipeline and pipeline.get('stage') == 'rendering' else preview.get('stage', 'starting')
            if stage == 'finished':
                stage = 'finalizing'
        else:
            stage = 'finished'
        artifacts = []
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(directory)
            if (path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES
                    and not any(p.startswith('.') for p in relative.parts)
                    and not any(p.is_symlink() for p in (path, *path.parents))):
                artifacts.append(path.relative_to(directory).as_posix())
            if len(artifacts) >= 512:
                notices.append('Artifact list truncated at 512 files.')
                break
        return dict(id=identity, status=job['status'], stage=stage, options=job['options'],
            elapsed_seconds=round(job.get('ended', time.monotonic())-job['started'], 1),
            preview_stage=preview.get('stage'),
            rendering_skipped=(pipeline or {}).get('rendering', {}).get('reason'),
            size=job['size'], input=job['image'].name, detections=detections, units=units,
            model_calls=preview.get('attempts', 0), coverage=(pipeline or {}).get('coverage'),
            rendering_coverage=rendering.get('coverage'), quality=rendering.get('quality_summary'),
            error=job['error'] or (pipeline or {}).get('error') or preview.get('error'),
            artifacts=artifacts, notices=notices, output_directory=str(directory))


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        pass  # Polling is intentionally quiet; the browser shows job errors.

    def respond(self, status, body, mime='application/json', *, attachment=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; connect-src 'self'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="'+attachment+'"')
        self.end_headers()
        self.wfile.write(body)

    def json(self, status, value):
        self.respond(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

    def local_request(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in hosts:
            raise RequestError('This viewer only accepts its loopback Host.', 403)
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://'+host for host in hosts}:
            raise RequestError('Cross-origin requests are not allowed.', 403)
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise RequestError('Cross-site requests are not allowed.', 403)

    def do_GET(self):
        try:
            self.local_request()
            path = unquote(urlsplit(self.path).path)
            if path in ('/', '/app.js', '/style.css'):
                name = 'index.html' if path == '/' else path[1:]
                self.respond(200, (STATIC/name).read_bytes(), mimetypes.guess_type(name)[0]+'; charset=utf-8')
            elif path == '/api/config':
                self.json(200, self.server.app.public_config())
            elif path == '/api/jobs':
                self.json(200, self.server.app.list_jobs())
            elif re.fullmatch('/api/jobs/[0-9a-f]{32}', path):
                self.json(200, self.server.app.snapshot(path.rsplit('/', 1)[1]))
            elif match := re.fullmatch('/api/jobs/([0-9a-f]{32})/artifacts/(.+)', path):
                artifact = self.server.app.artifact(*match.groups())
                if artifact.stat().st_size > 64*1024*1024:
                    raise RequestError('Artifact exceeds download limit.', 413)
                image = artifact.suffix.lower() in ('.png', '.jpg', '.jpeg')
                self.respond(200, artifact.read_bytes(), mimetypes.guess_type(artifact.name)[0] if image else 'application/octet-stream',
                    attachment=None if image else artifact.name)
            else:
                raise RequestError('Not found.', 404)
        except RequestError as error:
            self.json(error.status, dict(error=str(error)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.json(500, dict(error='Viewer error: '+str(error)[:1000]))

    def do_POST(self):
        try:
            self.local_request()
            if self.path != '/api/jobs':
                raise RequestError('Not found.', 404)
            if not secrets.compare_digest(self.headers.get('X-Tralens-Token', ''), self.server.app.token):
                raise RequestError('Invalid session token; reload the page.', 403)
            lengths = self.headers.get_all('Content-Length') or []
            if self.headers.get('Transfer-Encoding') or len(lengths) != 1 or not re.fullmatch('[0-9]{1,10}', lengths[0]):
                raise RequestError('A single Content-Length is required.', 411)
            length = int(lengths[0])
            if not 0 < length <= MAX_UPLOAD:
                raise RequestError('Upload limit is 32 MiB.', 413)
            supplied = self.headers.get('X-Tralens-Options', '{}')
            if len(supplied) > 2048:
                raise RequestError('Run options too large.')
            try:
                supplied = json.loads(supplied)
            except ValueError as error:
                raise RequestError('Invalid run options JSON.') from error
            data = self.rfile.read(length)
            if len(data) != length:
                raise RequestError('Incomplete upload.')
            self.json(202, dict(id=self.server.app.start(data, supplied)))
        except RequestError as error:
            self.json(error.status, dict(error=str(error)))
        except (OSError, ValueError, RuntimeError) as error:
            self.json(500, dict(error='Could not start run: '+str(error)[:1000]))


def create_server(app, port=7860):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.app = app
    return server
