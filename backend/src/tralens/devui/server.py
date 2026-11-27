"""Minimal loopback-by-default pipeline viewer, not a production upload service.

Polls the real pipeline's atomic checkpoints; no duplicate stage algorithms and
no simulated progress. Models are initialized only by an explicit uploaded run.
"""
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import ipaddress
import json
import mimetypes
import math
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
MAX_HISTORY = 24
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
    manga_reread_model: Path | None = None
    reread_model: str = 'qwen3.8:latest'
    reconstruction_config: Path | None = None
    text_mask_model: Path | None = None
    isolated_workers: bool = False  # Embedding/test compatibility; CLI enables this.
    job_seconds: int = 1800
    job_storage_bytes: int = 2*1024**3
    root_storage_bytes: int = 8*1024**3
    crop_reader_model: Path | None = None
    sdxl_config: Path | None = None
    reader_mode: bool = False
    frontier_provider: str = "chatgpt"


def options(value, *, render_enabled):
    if type(value) is not dict or set(value) - {'mode', 'max_attempts', 'allow_fragments', 'render_policy', 'translation_policy',
                                             'replacement_policy', 'quality_reread', 'layout_policy', 'source_localization', 'visibility_policy', 'removal_policy', 'reconstruction_policy', 'translation_model', 'luna_mode', 'related_jobs'}:
        raise RequestError('Unknown run options.')
    result = dict(mode='render' if render_enabled else 'ocr', max_attempts=64,
        allow_fragments=False, render_policy='best-effort', translation_policy='coverage-first',
        replacement_policy='all', quality_reread=False, layout_policy='source-local', source_localization=False,
        visibility_policy='replacement-only', removal_policy='legacy', reconstruction_policy='classical')
    if value.get('mode',result['mode'])=='render':
        result.update(visibility_policy='always-show',removal_policy='aggressive')
    result.update(value)
    if type(result['mode']) is not str or result['mode'] not in ('ocr', 'translate', 'render'):
        raise RequestError('Choose OCR, translation, or rendering.')
    if result['mode'] == 'render' and not render_enabled:
        raise RequestError('Start the server with --fonts to enable rendering.')
    if type(result['max_attempts']) is not int or not 0 <= result['max_attempts'] <= 64:
        raise RequestError('Translation call budget must be 0–64.')
    if type(result['allow_fragments']) is not bool:
        raise RequestError('Fragment preview option must be a boolean.')
    if type(result['render_policy']) is not str or result['render_policy'] not in ('best-effort', 'conservative'):
        raise RequestError('Unknown rendering policy.')
    if type(result['translation_policy']) is not str or result['translation_policy'] not in ('coverage-first', 'conservative'):
        raise RequestError('Unknown translation policy.')
    if type(result['replacement_policy']) is not str or result['replacement_policy'] not in ('all', 'confidence-gated'):
        raise RequestError('Unknown image replacement policy.')
    if type(result['quality_reread']) is not bool:
        raise RequestError('Quality reread option must be a boolean.')
    if result['quality_reread'] and (result['mode'] != 'render' or result['translation_policy'] != 'coverage-first'):
        raise RequestError('Quality rereads require Final rendering and Coverage first translation.')
    if result['replacement_policy'] != 'all' and result['mode'] != 'render':
        raise RequestError('Image replacement checks require Final rendering.')
    if type(result['layout_policy']) is not str or result['layout_policy'] not in ('source-local', 'enclosed-retry'):
        raise RequestError('Unknown enclosure layout policy.')
    if type(result['source_localization']) is not bool:
        raise RequestError('Source localization option must be a boolean.')
    if (result['layout_policy'] != 'source-local' or result['source_localization']) and (
            result['mode'] != 'render' or result['render_policy'] != 'best-effort'
            or result['replacement_policy'] != 'confidence-gated'):
        raise RequestError('Enclosure retries and source localization require Final rendering, Best effort, and Native cleanup + OCR checks.')
    if type(result['visibility_policy']) is not str or result['visibility_policy'] not in ('replacement-only','always-show'):
        raise RequestError('Unknown translation visibility policy.')
    if result['visibility_policy'] != 'replacement-only' and result['mode'] != 'render':
        raise RequestError('Always show available translations requires Final rendering.')
    if result['removal_policy'] not in ('legacy','preserve','aggressive') or result['reconstruction_policy'] not in ('auto','classical','lama','sdxl'):
        raise RequestError('Unknown removal or reconstruction policy.')
    if result['reconstruction_policy']!='classical' and (result['mode']!='render' or result['removal_policy']!='aggressive'):
        raise RequestError('Learned inpainting requires Final rendering and Aggressive removal.')
    from ..translation import luna
    model = result.get('translation_model', 'local')
    mode = result.get('luna_mode', 'reviewed')
    related = result.get('related_jobs', [])
    if model not in ('local', *luna.MODELS) or mode not in luna.MODES:
        raise RequestError('Unknown translation model or input mode.')
    if (not isinstance(related, list) or len(related) > 6 or
            any(type(i) is not str or not re.fullmatch('[0-9a-f]{32}', i) for i in related)
            or len(set(related)) != len(related)):
        raise RequestError('Select up to six distinct saved context pages.')
    if related and (model not in luna.MODELS or mode == 'isolated' or result['mode'] == 'ocr'):
        raise RequestError('Context pages require a visual frontier model.')
    return result


def checkpoint(path, notices):
    """Read published checkpoints only; never interpret missing data as success."""
    try:
        if any(p.is_symlink() for p in (path, *path.parents)):
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
        allow_fragment_previews=settings['allow_fragments'], translation_policy=settings['translation_policy'])
    from ..translation.luna import MODELS
    model = settings.get('translation_model', 'local')
    if model in MODELS and settings['mode'] != 'ocr':
        related = []
        for identity in settings.get('related_jobs', []):
            directory = config.root / identity
            paths = [directory/name for name in ('input.jpg', 'input.png') if (directory/name).is_file()]
            if directory.is_symlink() or len(paths) != 1 or paths[0].is_symlink():
                raise ValueError('Saved context image is unavailable')
            related.append(paths[0])
        kwargs.update(model=model, luna_mode=settings.get('luna_mode', 'reviewed'), related_pages=related)
    if settings['mode'] == 'render':
        from main import run_pipeline
        if (config.crop_reader_model is not None and settings['translation_policy']=='coverage-first'
                and not settings.get('quality_reread')):
            kwargs.update(crop_reader_model=config.crop_reader_model, reader_presentations=not config.reader_mode)
        if config.reader_mode:
            if model == 'local' and config.model in ('translategemma:27b','translategemma:12b'):
                # The Ollama request can include paging a cold model into GPU memory.
                kwargs['timeout'] = min(300, config.job_seconds)
            from PIL import Image
            with Image.open(job['image']) as page:
                phone_size = min(144, max(16, math.ceil(page.width * 12 / 390)))
            # Dense balloons need the option to fit smaller type. The reader already
            # measures the actual lettering and offers a zoomable English transcript.
            kwargs.update(max_title_size=144, min_size=max(12, phone_size // 2), max_size=min(144, max(28, math.ceil(phone_size * 1.3))))
        model_configs = {'lama': config.reconstruction_config, 'sdxl': config.sdxl_config}
        reconstruction = settings.get('reconstruction_policy', 'classical')
        return run_pipeline(job['image'], job['directory']/'run', fonts=config.fonts,
            font_id=config.font_id, render_policy=settings['render_policy'],
            replacement_policy=settings.get('replacement_policy', 'all'),
            layout_policy=settings.get('layout_policy', 'source-local'),
            source_localization=settings.get('source_localization', False),
            visibility_policy=settings.get('visibility_policy', 'replacement-only'),
            removal_policy=settings.get('removal_policy','legacy'),
            reconstruction_policy=reconstruction,
            reconstruction_config=model_configs if reconstruction == 'auto' else model_configs.get(reconstruction),
            text_mask_model=config.text_mask_model,
            quality_reread=settings.get('quality_reread', False), reread_model=config.reread_model,
            manga_reread_model=config.manga_reread_model if settings.get('translation_policy','coverage-first')=='coverage-first' and 'crop_reader_model' not in kwargs else None, **kwargs)
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
        self.started_jobs = 0
        self.recovery_blocked = []
        if config.isolated_workers:
            from .jobs import recover_jobs
            self.recovery_blocked = recover_jobs(self.root)
        self._restore_jobs()

    def _restore_jobs(self):
        """Restore inspection/retry targets, never resume work or rewrite evidence."""
        directories = [p for p in self.root.iterdir() if re.fullmatch('[0-9a-f]{32}', p.name)
                       and not p.is_symlink() and p.is_dir()]
        for directory in sorted(directories, key=lambda p: p.stat().st_mtime)[-MAX_HISTORY:]:
            notices = []
            meta = checkpoint(directory/'viewer.json', notices)
            pipeline = checkpoint(directory/'run/pipeline.json', notices) or {}
            preview = checkpoint(directory/'run/preview/preview.json', notices) or {}
            if not meta and not pipeline and not preview:
                continue
            try:
                if meta and (meta.get('purpose') != 'development_viewer_job' or meta.get('id') != directory.name):
                    continue
                source = pipeline.get('source') or preview.get('input') or {}
                settings = (meta or {}).get('options') or dict(
                    mode='render' if pipeline else 'translate' if preview.get('settings', {}).get('run_local_model') else 'ocr',
                    max_attempts=preview.get('settings', {}).get('max_attempts', 2),
                    allow_fragments=preview.get('settings', {}).get('allow_fragment_previews', False),
                    render_policy=pipeline.get('render_policy', 'best-effort'),
                    translation_policy=preview.get('settings', {}).get('translation_policy', pipeline.get('translation_policy', 'conservative')))
                if 'translation_policy' not in settings:
                    settings = {**settings, 'translation_policy': preview.get('settings', {}).get('translation_policy', pipeline.get('translation_policy', 'conservative'))}
                replacement = pipeline.get('replacement_policy', {})
                settings = {'replacement_policy': replacement.get('name', 'all') if isinstance(replacement, dict) else replacement,
                    'quality_reread': bool(pipeline.get('quality_reread', {}).get('enabled')),
                    'layout_policy': pipeline.get('layout_policy', 'source-local'),
                    'source_localization': bool(pipeline.get('source_localization', {}).get('enabled')), **settings}
                settings.setdefault('visibility_policy',pipeline.get('visibility_policy','replacement-only'))
                settings.setdefault('removal_policy',pipeline.get('removal_policy','legacy'))
                settings.setdefault('reconstruction_policy',pipeline.get('reconstruction_policy','classical'))
                settings = options(settings, render_enabled=True)  # View old jobs even without a current font manifest.
                images = [directory/name for name in ('input.png', 'input.jpg') if (directory/name).is_file()]
                if len(images) != 1 or images[0].is_symlink():
                    continue
                with images[0].open('rb') as stream:
                    data = stream.read(MAX_UPLOAD+1)
                expected = (meta or {}).get('source_sha256') or source.get('sha256')
                if len(data) > MAX_UPLOAD or not expected or hashlib.sha256(data).hexdigest() != expected:
                    continue
                size = (meta or {}).get('size') or source.get('size')
                if (type(size) is not list or len(size) != 2 or
                        any(type(v) is not int or not 0 < v <= 4096 for v in size) or size[0]*size[1] > 4_000_000):
                    continue
                report = pipeline or preview
                status = (meta or {}).get('status', report.get('status', 'interrupted'))
                error = (meta or {}).get('error') or report.get('error')
                if status == 'running':
                    status = report.get('status', 'running')
                    if status == 'running':
                        status, error = 'interrupted', dict(type='ServerRestart',
                            message='The server stopped before this run finished. Retry starts a fresh run.')
                if status not in ('complete', 'partial', 'complete_with_failures', 'failed', 'interrupted'):
                    continue
                elapsed = (meta or {}).get('elapsed_seconds', 0)
                if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                    continue
                done = threading.Event()
                done.set()
                now = time.monotonic()
                self.jobs[directory.name] = dict(id=directory.name, directory=directory, image=images[0],
                    size=size, source_sha256=expected, options=settings, status=status, error=error,
                    started=now-elapsed, ended=now, done=done, restored=True,
                    retry_of=(meta or {}).get('retry_of'))
            except (ValueError, OSError, KeyError, TypeError):
                continue

    def _save_job(self, job, *, replace_owned=False):
        from ..translation.workflow import _atomic_json
        value = {key: job.get(key) for key in
                 ('id', 'size', 'source_sha256', 'options', 'status', 'error', 'retry_of', 'cancel_requested')}
        value.update(purpose='development_viewer_job', schema_version=1,
            elapsed_seconds=round(job.get('ended', time.monotonic())-job['started'], 1))
        _atomic_json(job['directory']/'viewer.json', value, replace_owned=replace_owned)

    def public_config(self):
        return dict(token=self.token, device=self.config.device, model=self.config.model,
            endpoint=self.config.endpoint,
            model_root=str(self.config.model_root), fonts=str(self.config.fonts) if self.config.fonts else None,
            render_enabled=self.config.fonts is not None, max_upload_bytes=MAX_UPLOAD,
            page_pixel_limit=4_000_000, max_jobs=None if self.config.reader_mode else MAX_JOBS,
            quality_reread_available=self.config.manga_reread_model is not None, reread_model=self.config.reread_model,
            sdxl_available=self.config.sdxl_config is not None,
            reconstruction_available=self.config.reconstruction_config is not None,
            text_mask_available=self.config.text_mask_model is not None,
            cancellation_available=self.config.isolated_workers,
            worker_mode='isolated_process' if self.config.isolated_workers else 'embedded_thread',
            recovery_blocked=self.recovery_blocked)

    def list_jobs(self):
        with self.lock:
            return [dict(id=job['id'], status=job['status'], mode=job['options']['mode']) for job in self.jobs.values()]

    def start(self, data, supplied, *, retry_of=None):
        settings = options(supplied, render_enabled=self.config.fonts is not None)
        if self.config.frontier_provider == 'disabled' and settings.get('translation_model', 'local') != 'local':
            raise RequestError('Remote GPT translation requires an owner-protected reader.', 403)
        if self.config.isolated_workers:
            from .jobs import tree_bytes, recover_jobs
            self.recovery_blocked = recover_jobs(self.root)
            if self.recovery_blocked:
                raise RequestError('Unresolved worker ownership prevents another isolated job; inspect worker recovery records.',409)
            if tree_bytes(self.root)>=self.config.root_storage_bytes-self.config.job_storage_bytes:
                raise RequestError('Retained job storage is full. Archive old evidence before starting another run.',409)
        if settings['reconstruction_policy']=='sdxl':
            from ..rendering.reconstruction import configuration
            if self.config.sdxl_config is None:
                raise RequestError('Configure --sdxl-config to enable SDXL inpainting.')
            configuration(self.config.sdxl_config, 'sdxl')
        if settings['reconstruction_policy']=='lama':
            if self.config.reconstruction_config is None:
                raise RequestError('Start the viewer with an explicit --reconstruction-config to enable local LaMa evaluation.')
            from ..rendering.reconstruction import configuration
            configuration(self.config.reconstruction_config)
        if settings['quality_reread'] and self.config.manga_reread_model is None:
            raise RequestError('Start the viewer with --manga-reread-model pointing to the existing local snapshot to enable quality rereads.')
        if not 0 < len(data) <= MAX_UPLOAD:
            raise RequestError('Upload must be 1 byte–32 MiB.')
        with self.lock:
            if self.active is not None:
                raise RequestError('A job is already running. Wait for it to finish.', 409)
            if not self.config.reader_mode and self.started_jobs >= MAX_JOBS:
                raise RequestError('Session job limit reached. Restart the server to run more jobs.', 429)
            from PIL import Image
            from ..rendering.preparation import source_rgb
            try:
                with Image.open(BytesIO(data), formats=('PNG', 'JPEG')) as opened:
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
                options=settings, status='running', started=time.monotonic(), error=None, done=threading.Event(),
                source_sha256=hashlib.sha256(data).hexdigest(), retry_of=retry_of, restored=False)
            self._save_job(job)
            self.jobs[identity] = job
            self.started_jobs += 1
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

    def retry(self, identity, supplied):
        previous = self.job(identity)
        if type(supplied) is not dict:
            raise RequestError('Run options must be an object.')
        if previous['status'] == 'running':
            raise RequestError('This run is still active. Wait before retrying.', 409)
        image = self.artifact(identity, previous['image'].name)
        with image.open('rb') as stream:
            data = stream.read(MAX_UPLOAD+1)
        if len(data) > MAX_UPLOAD or hashlib.sha256(data).hexdigest() != previous['source_sha256']:
            raise RequestError('The retained upload changed. Choose the original image again.', 409)
        # No previous preview/results are promoted or overwritten. The coordinator
        # receives a fresh source path and uses the current server configuration.
        return self.start(data, {**previous['options'], **supplied}, retry_of=identity)

    def _run(self, job):
        try:
            if self.config.isolated_workers:
                from .jobs import run
                with self.lock:job.setdefault('cancel',threading.Event())
                report=run(self.config,job,job['cancel'])
            else:report = self.runner(self.config, job)
            status = report['status']
            error = report.get('error')
        except (Exception, KeyboardInterrupt) as failure:
            status, error = 'failed', dict(type=type(failure).__name__, message=str(failure)[:2000])
        with self.lock:
            job.update(status=status, error=error, ended=time.monotonic())
            try:
                self._save_job(job, replace_owned=True)
            except (OSError, ValueError, TypeError) as failure:
                job['history_warning'] = 'Could not save viewer history: '+str(failure)[:500]
            self.active = None
            job['done'].set()

    def cancel(self,identity):
        with self.lock:
            job=self.jobs.get(identity)
            if job is None:raise RequestError('Job not found.',404)
            if job['status']!='running':raise RequestError('Job is not running.',409)
            if not self.config.isolated_workers:raise RequestError('Cancellation requires the isolated CLI worker.',409)
            job.setdefault('cancel',threading.Event()).set()
            job['cancel_requested']=True
            self._save_job(job,replace_owned=True)
        return identity

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
        if job.get('history_warning'):
            notices.append(job['history_warning'])
        pipeline = checkpoint(directory/'run/pipeline.json', notices)
        preview = checkpoint(directory/'run/preview/preview.json', notices)
        rendering = checkpoint(directory/'run/rendering/manifest.json', notices)
        native = checkpoint(directory/'run/rendering-native/manifest.json', notices) if job['options'].get('source_localization') else None
        preview = preview or {}
        discovery = preview.get('frontier_discovery')
        if discovery:
            if discovery.get('error'):
                notices.append('Page discovery failed. Translation continued with the local detections.')
            else:
                notices.append(f"Page discovery added {len(discovery['added_ids'])} of {discovery['proposed']} proposed regions.")
        rendering = rendering or {}
        replacement = rendering.get('replacement_policy', (pipeline or {}).get('replacement_policy', 'all'))
        replacement = replacement.get('name', 'all') if isinstance(replacement, dict) else replacement
        raw = (preview.get('ocr') or {}).get('raw_detections') or [[]]
        detections = [dict(source_id=d['source_id'], text=d.get('text', ''),
            confidence=d.get('confidence'), polygon=d['polygon']) for d in raw[0]][:512]
        rendered = {r['unit_id']: r for r in rendering.get('units', [])}
        plan = rendering.get('localization_request_plan') or (native or {}).get('localization_request_plan') or {}
        crop_requests = {r['unit_id']: r for r in plan.get('records', [])}
        units = []
        for record in preview.get('units', [])[:512]:
            unit = record['unit']
            result = record.get('result') or {}
            row = rendered.get(unit['unit_id'], {})
            fragments = [dict(status=f['status'], translations=(f.get('result') or {}).get('fragments', []))
                for f in record.get('fragment_previews', [])]
            units.append(dict(id=unit['unit_id'], member_ids=unit['member_ids'],
                input_kind=record.get('input_kind', 'strict_utterance'),
                source=selected_text(record.get('translation_input') or unit), raw_source=selected_text(unit),
                luna_reading=(record.get('luna_decision') or {}).get('source'),
                reread_selected=bool(record.get('coverage', {}).get('reread_selected_revision')), status=record['status'],
                target=dict(result.get('translations', [])).get(unit['unit_id']),
                reasons=record.get('blocking_reasons', []) +
                    [v for v in (result.get('failure'), result.get('detail')) if v] +
                    [reason for outcome in result.get('outcomes', []) for reason in outcome.get('reasons', [])],
                fragments=fragments, render_status=row.get('status'), render_reason=row.get('reason'),
                quality=row.get('render_quality', 'conservative') if row.get('status') == 'automatic_composed' else None,
                warnings=(row.get('best_effort') or {}).get('warnings', []),
                crop_request_reason=crop_requests.get(unit['unit_id'], {}).get('reason'),
                crop_request_status=crop_requests.get(unit['unit_id'], {}).get('status'),
                localized_cleanup_attempt=row.get('localized_cleanup_attempt'),
                enclosure_layout_attempt=row.get('enclosure_layout_attempt'),
                display_status=row.get('display_status'),source_visibility=row.get('source_visibility'),
                display_fallback=row.get('display_fallback')))
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
            if len(artifacts) >= 768:
                notices.append('Artifact list truncated at 768 files.')
                break
        return dict(id=identity, status=job['status'], stage=stage, options=job['options'],
            retry_of=job.get('retry_of'), restored=job.get('restored', False),
            elapsed_seconds=round(job.get('ended', time.monotonic())-job['started'], 1),
            preview_stage=preview.get('stage'),
            rendering_skipped=(pipeline or {}).get('rendering', {}).get('reason'),
            size=job['size'], input=job['image'].name, detections=detections, units=units,
            model_calls=preview.get('attempts', 0), coverage=(pipeline or {}).get('coverage'),
            rendering_coverage=rendering.get('coverage'), quality=rendering.get('quality_summary'),
            replacement_policy=replacement,
            replacement_confidence=rendering.get('replacement_confidence'),
            quality_reread=(pipeline or {}).get('quality_reread'),
            layout_policy=rendering.get('layout_policy', (pipeline or {}).get('layout_policy', 'source-local')),
            source_localization=(pipeline or {}).get('source_localization'),
            visibility_policy=rendering.get('visibility_policy',(pipeline or {}).get('visibility_policy','replacement-only')),
            display_summary=rendering.get('display_summary',(pipeline or {}).get('display_summary')),
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
        owner = getattr(self.server, 'owner', None)
        if owner:
            if (self.headers.get('Host') != owner.host or
                    self.headers.get('Origin', owner.origin) != owner.origin or
                    self.headers.get('Sec-Fetch-Site') == 'cross-site'):
                raise RequestError('Unexpected reader host or origin.', 403)
            return
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.server.server_address[0] != '127.0.0.1':
            # Accept the actual destination interface, never arbitrary Host names
            # or forwarded headers. Wildcard binding does not disable DNS-rebinding
            # or browser origin checks.
            hosts.add(f'{self.connection.getsockname()[0]}:{port}')
        if port == 80:
            hosts.update(host.rsplit(':', 1)[0] for host in tuple(hosts))
        if self.headers.get('Host') not in hosts:
            raise RequestError('This viewer only accepts a configured local-interface Host.', 403)
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
            retry = re.fullmatch('/api/jobs/([0-9a-f]{32})/retry', self.path)
            cancel = re.fullmatch('/api/jobs/([0-9a-f]{32})/cancel', self.path)
            if self.path != '/api/jobs' and retry is None and cancel is None:
                raise RequestError('Not found.', 404)
            if not secrets.compare_digest(self.headers.get('X-Tralens-Token', ''), self.server.app.token):
                raise RequestError('Invalid session token; reload the page.', 403)
            lengths = self.headers.get_all('Content-Length') or []
            if self.headers.get('Transfer-Encoding') or len(lengths) != 1 or not re.fullmatch('[0-9]{1,10}', lengths[0]):
                raise RequestError('A single Content-Length is required.', 411)
            length = int(lengths[0])
            if not 0 < length <= (4096 if retry else MAX_UPLOAD):
                raise RequestError('Retry options must be 1–4096 bytes.' if retry else 'Upload limit is 32 MiB.', 413)
            data = self.rfile.read(length)
            if len(data) != length:
                raise RequestError('Incomplete request.')
            supplied = data if retry or cancel else self.headers.get('X-Tralens-Options', '{}')
            if len(supplied) > 2048:
                raise RequestError('Run options too large.')
            try:
                supplied = json.loads(supplied)
            except ValueError as error:
                raise RequestError('Invalid run options JSON.') from error
            identity = self.server.app.cancel(cancel.group(1)) if cancel else self.server.app.retry(retry.group(1), supplied) if retry else self.server.app.start(data, supplied)
            self.json(202, dict(id=identity))
        except RequestError as error:
            self.json(error.status, dict(error=str(error)))
        except (OSError, ValueError, RuntimeError) as error:
            self.json(500, dict(error='Could not start run: '+str(error)[:1000]))


def create_server(app, port=7860, *, host='127.0.0.1'):
    host = str(ipaddress.IPv4Address(host))
    server = ThreadingHTTPServer((host, port), Handler)
    server.app = app
    return server
