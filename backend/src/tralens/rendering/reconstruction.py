"""Explicit crop-bound direct reconstruction, with isolated local model execution.

No model search, acquisition, source-code imports, or deployment-rights inference.
Predicted pixels are candidates, never a recovery of unknowable original artwork.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from ..translation.contracts import canonical, revision


MODEL_SHA256 = '1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6'
MODEL_BYTES = 208044816
POLICY = 'bound_crop_reconstruction_v1'


class BindingError(ValueError):
    """Integrity failure, never aesthetic uncertainty or retry authorization."""


def require(value,message):
    if not value:raise BindingError(message)


def pixels_hash(array):
    return hashlib.sha256(str((array.dtype.str,array.shape)).encode()+array.tobytes()).hexdigest()


def model_identity(config):
    identity = {key:config.get(key) for key in ('backend','sha256','config_revision')}
    if config.get('prefill'):
        identity['prefill'] = config['prefill']
    return identity


@lru_cache(maxsize=128)
def _file_digest(path, size, modified):
    """Avoid rereading unchanged multi-gigabyte weights on each page preflight."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def configuration(path, policy=None):
    """Validate explicit operator config and bytes before any upstream inference."""
    if policy == 'auto':
        # Reader jobs supply configured paths directly. CLI manifests resolve
        # model config paths relative to the manifest, just like SDXL assets.
        entries = {} if path is None else path
        if not isinstance(entries, dict):
            manifest = Path(entries).expanduser().absolute()
            require(manifest.is_file() and manifest.stat().st_size <= 16384, 'invalid Auto manifest')
            entries = json.loads(manifest.read_bytes())
            require(isinstance(entries, dict), 'Auto manifest must contain model config paths')
            require(all(isinstance(value, str) and value for value in entries.values()),
                    'Auto manifest values must be local config paths')
            entries = {key: manifest.parent/value for key, value in entries.items()}
        require(entries.keys() <= {'lama', 'sdxl'}, 'unknown Auto model')
        return dict(backend='auto', models={key: configuration(value, key)
                    for key, value in entries.items() if value})
    path=Path(path).expanduser().absolute()
    require(path.is_file() and path.stat().st_size<=16384,'explicit reconstruction configuration required')
    config=json.loads(path.read_bytes())
    if config.get('backend') == 'sdxl-inpaint':
        require(policy in (None, 'sdxl'), 'inpainting configuration does not match selection')
        require(set(config) == {'backend', 'model', 'python'}, 'invalid SDXL configuration')
        model = (path.parent/config['model']).absolute()
        python = (path.parent/config['python']).absolute()
        require(python.is_file() and os.access(python, os.X_OK), 'SDXL Python runtime unavailable')
        index = model/'model_index.json'
        require(index.is_file() and json.loads(index.read_bytes()).get('_class_name') == 'StableDiffusionXLInpaintPipeline',
                'local SDXL inpainting pipeline required')
        files = [model/component/filename for component, filename in (
            ('unet','diffusion_pytorch_model.fp16.safetensors'), ('vae','diffusion_pytorch_model.fp16.safetensors'),
            ('text_encoder','model.fp16.safetensors'), ('text_encoder_2','model.fp16.safetensors'))]
        require(all(f.is_file() for f in files), 'incomplete SDXL fp16 weights')
        assets = sorted(set(files) | set(model.rglob('*.json')) | set(model.rglob('*.txt')))
        identity = revision({str(f.relative_to(model)): _file_digest(f, f.stat().st_size, f.stat().st_mtime_ns)
                             for f in assets})
        return dict(backend='sdxl-inpaint', model=str(model), python=str(python), sha256=identity,
                    config_revision=revision(config))
    require(policy != 'sdxl', 'inpainting configuration does not match selection')
    require(set(config)=={'backend','model','sha256','python','use_scope'}
        and config['backend']=='lama-onnx-512' and config['sha256']==MODEL_SHA256
        and config['use_scope']=='private-noncommercial-evaluation', 'unsupported reconstruction configuration/scope')
    model=Path(config['model']);python=Path(config['python'])
    require(model.is_absolute() and model.is_file() and not model.is_symlink()
        and model.stat().st_size==MODEL_BYTES,'explicit pinned LaMa model unavailable')
    require(hashlib.sha256(model.read_bytes()).hexdigest()==MODEL_SHA256,'reconstruction model digest mismatch')
    require(python.is_absolute() and python.is_file() and os.access(python,os.X_OK),'explicit reconstruction runtime unavailable')
    return {**config,'config_revision':revision(config),'model_bytes':MODEL_BYTES,
        'provenance':'Carve/LaMa-ONNX@c3c0c9e468934d62e79c329e35d82dd09ff8c444; upstream checkpoint chain unresolved',
        'commercial_or_distribution_clearance':False}


def request(source,binding,bounds,crop,inference,commit,protected,*,attempt,model=None):
    require(crop.dtype==np.uint8 and crop.ndim==3 and crop.shape[2]==3,'RGB original crop required')
    h,w=crop.shape[:2];x,y,r,b=bounds
    require(0<w<=512 and 0<h<=512 and [r-x,b-y]==[w,h]
        and 0<=x<r<=source['size'][0] and 0<=y<b<=source['size'][1],'crop/page transform invalid')
    require(all(m.dtype==bool and m.shape==(h,w) for m in (inference,commit,protected)), 'typed local masks required')
    require(commit.any() and not (commit & ~inference).any() and not (commit & protected).any(),
        'commit support escapes inference mask or overwrites protected output')
    require(binding.get('member_ids') and binding.get('request_revision') and binding.get('text'), 'source-bound translation required')
    evidence={'policy':POLICY,'source':source,'binding':binding,'bounds':list(bounds),'origin':[x,y],
        'original_crop':pixels_hash(crop),'inference_mask':pixels_hash(inference),'commit_mask':pixels_hash(commit),
        'protected_mask':pixels_hash(protected),'attempt':attempt,'seed':0,'model':model,
        'parameters':{'padding':'symmetric image AND mask to 512; no resize','mask_one':'remove'},
        'limits':{'seconds':30,'address_space_bytes':6*1024**3,'cpu_seconds':30,'threads':2},
        'generated_art_is_speculative':True}
    if model and model.get('backend') == 'sdxl-inpaint':
        evidence.update(seed=42, parameters={'padding':'symmetric to 1024', 'steps':25, 'strength':0.78, 'guidance':6},
                        limits={'seconds':120, 'device':'cuda'})
    return {**evidence,'revision':revision(evidence)}


def validate(req,crop,inference,commit,protected):
    expected=request(req['source'],req['binding'],req['bounds'],crop,inference,commit,protected,attempt=req['attempt'],model=req.get('model'))
    require(req==expected,'stale reconstruction request/pixel/mask binding')


def compose(req,crop,inference,commit,protected,result):
    validate(req,crop,inference,commit,protected)
    require(type(result) is dict and result.get('request_revision')==req['revision'],'foreign reconstruction result')
    candidate=result.get('pixels')
    require(type(candidate) is np.ndarray and candidate.dtype==np.uint8 and candidate.shape==crop.shape,
        'invalid reconstruction output shape/type')
    output=crop.copy();output[commit]=candidate[commit]
    require(np.array_equal(output[~commit],crop[~commit]) and np.array_equal(output[protected],crop[protected]),
        'reconstruction composition outside declared support')
    return output


@dataclass
class Budget:
    maximum_calls:int=8
    maximum_seconds:float=120
    calls:int=0
    seconds:float=0

    def available(self):return self.calls<self.maximum_calls and self.seconds<self.maximum_seconds


class LocalReconstruction:
    def __init__(self,config):
        self.config=config

    def reconstruct(self,req,crop,inference,commit,protected,*,timeout=30):
        validate(req,crop,inference,commit,protected)
        require(req.get('model')==model_identity(self.config),'request/model configuration mismatch')
        # Each native session is in a killable process; no parser/model lifetime
        # survives timeout/OOM. Inputs/outputs use PNG and JSON, never pickle.
        with tempfile.TemporaryDirectory(prefix='tralens-reconstruct-') as directory:
            directory=Path(directory)
            Image.fromarray(crop).save(directory/'input.png')
            Image.fromarray(inference.astype(np.uint8)*255).save(directory/'mask.png')
            Image.fromarray(commit.astype(np.uint8)*255).save(directory/'commit.png')
            Image.fromarray(protected.astype(np.uint8)*255).save(directory/'protected.png')
            (directory/'request.json').write_text(canonical({'request':req,'config':self.config}))
            worker=Path(__file__).with_name('reconstruction_worker.py')
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''}
            if self.config['backend'] == 'sdxl-inpaint':
                worker = worker.with_name('sdxl_worker.py')
                env.pop('CUDA_VISIBLE_DEVICES')
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
            start=time.monotonic()
            entry='import runpy,sys,pathlib; runpy.run_path(sys.argv[1])["main"](pathlib.Path(sys.argv[2]))'
            result=subprocess.run([self.config['python'],'-c',entry,str(worker),str(directory)],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(req['limits']['seconds'],timeout),check=False)
            if result.returncode:
                if (directory/'binding-error.json').exists():
                    raise BindingError('reconstruction worker rejected input binding')
                raise RuntimeError(self.config['backend']+' worker failed (exit '+str(result.returncode)+'): '+result.stderr.decode(errors='replace')[-1500:])
            with Image.open(directory/'output.png') as image:
                require(image.mode=='RGB' and image.size==(crop.shape[1],crop.shape[0]),'invalid worker output image')
                output=np.asarray(image).copy()
            meta=json.loads((directory/'result.json').read_bytes())
            require(type(meta) is dict and meta.get('request_revision')==req['revision'] and meta.get('model_sha256')==self.config['sha256']
                and meta.get('output_pixels_sha256')==pixels_hash(output),
                'worker output binding changed')
            return {**meta,'pixels':output,'wall_seconds':time.monotonic()-start}


def outline_strength(crop, repaired, commit, *, relative=False):
    """Measure stroke-aligned contrast, optionally subtracting adjacent paint contrast."""
    import cv2

    def gradient(image):
        rgb = image.astype(np.float32)
        return np.stack((cv2.Sobel(rgb, -1, 1, 0, scale=.125),
                         cv2.Sobel(rgb, -1, 0, 1, scale=.125)), axis=-1)

    ink = gradient(crop)
    channel_strength = np.linalg.norm(ink, axis=-1)
    channel = channel_strength.argmax(axis=2)
    y, x = np.indices(commit.shape)
    normal = ink[y, x, channel]
    magnitude = channel_strength.max(axis=2)
    binary = commit.astype(np.uint8)
    band = cv2.dilate(binary, np.ones((3, 3), np.uint8)) != 0
    edges = band & (magnitude >= 15)
    if edges.sum() < 32:
        return None if relative else 0.
    # A highlight or panel line passing through a stroke can dominate its
    # residual. Prefer stroke edges next to observable, low-texture artwork.
    known = (cv2.dilate(binary, np.ones((5, 5), np.uint8)) == 0).astype(np.float32)
    coverage = cv2.blur(known, (21, 21))
    surrounding = cv2.blur(magnitude * known, (21, 21))
    quiet = edges & (coverage > .12) & (surrounding < 3 * coverage)
    if quiet.sum() >= 32:
        edges = quiet
    grad = gradient(repaired)
    if not relative:
        projected = (grad[edges] * normal[edges, None, :]).sum(axis=-1)
        response = np.abs(projected).max(axis=1) / magnitude[edges]
        return float(np.minimum(response, 25).mean())
    y, x = np.nonzero(edges)
    direction = normal[edges] / magnitude[edges, None]

    def contrast(y, x):
        projected = (grad[y, x] * direction[:, None, :]).sum(axis=-1)
        return np.minimum(np.abs(projected).max(axis=1), 25)

    center = contrast(y, x)
    # Raw contrast can increase when LaMa restores paint texture. A ghost
    # concentrates that contrast on the old glyph rather than on its flanks.
    dx, dy = np.rint(3 * direction).astype(int).T
    h, w = commit.shape
    left, right = (x-dx).clip(0, w-1), (x+dx).clip(0, w-1)
    top, bottom = (y-dy).clip(0, h-1), (y+dy).clip(0, h-1)
    valid = (commit[y, x] & commit[top, left] & commit[bottom, right]
             & (x >= abs(dx)) & (x < w-abs(dx)) & (y >= abs(dy)) & (y < h-abs(dy)))
    if valid.sum() < 32:
        return None
    flanks = .5 * (contrast(top, left) + contrast(bottom, right))
    return float((center - flanks)[valid].mean())


def review_outline(router, req, result, crop, inference, commit, protected, budget, deadline):
    """Compare a suspect SDXL repair with one budgeted LaMa alternative."""
    if (not router or router.config.get('backend') != 'auto'
            or (req.get('model') or {}).get('backend') != 'sdxl-inpaint'
            or 'lama' not in router.config['models']):
        return req, inference, result
    initial = compose(req, crop, inference, commit, protected, result)
    strength = outline_strength(crop, initial, commit)
    audit = dict(sdxl_request_revision=req['revision'], sdxl_outline=strength,
                 selected='sdxl-inpaint', status='below_threshold')
    result = {**result, 'outline_review': audit}
    from . import line_continuity
    bridges = line_continuity.bridges(crop, inference | protected, commit)
    initial_line_error = line_continuity.error(initial, bridges) if bridges else 0.
    if bridges:
        audit.update(line_crossings=len(bridges), sdxl_line_error=initial_line_error)
    if strength < 1.5 and initial_line_error < 4:
        return req, inference, result
    if not budget.available() or deadline <= time.monotonic():
        audit['status'] = 'budget_exhausted'
        return req, inference, result

    import cv2
    # Hide the antialiased fringe from LaMa. Only original commit pixels
    # can reach the output, including when this region was a tile.
    context = inference
    binary = commit.astype(np.uint8)
    neighborhood = np.ones((3, 3), np.uint8)
    boundary = commit & (cv2.erode(binary, neighborhood) == 0)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, neighborhood)
    # Tight, high-contrast cutouts already end at the ink edge. A wider
    # inference hole would also hide the art that should continue through it.
    if boundary.any() and np.mean(contrast[boundary] >= 24) < .75:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expanded = cv2.dilate(binary, kernel) != 0
        context = inference | (expanded & ~protected)
    lama = LocalReconstruction(router.config['models']['lama'])
    retry = request(req['source'], req['binding'], req['bounds'], crop, context, commit, protected,
                    attempt=req['attempt'], model=model_identity(lama.config))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        audit['status'] = 'budget_exhausted'
        return req, inference, result
    budget.calls += 1
    audit['lama_request_revision'] = retry['revision']
    try:
        answer = lama.reconstruct(retry, crop, context, commit, protected, timeout=remaining)
        alternate = compose(retry, crop, context, commit, protected, answer)
    except BindingError:
        raise
    except (RuntimeError, OSError, MemoryError, subprocess.SubprocessError) as error:
        audit.update(status='retry_failed', error=str(error)[:500])
        return req, inference, result
    alternative = outline_strength(crop, alternate, commit)
    audit.update(status='compared', lama_outline=alternative)
    reason = None
    if strength - alternative >= max(.35, .2 * strength):
        reason = 'weaker_outline'
    if bridges:
        other_line_error = line_continuity.error(alternate, bridges)
        audit['lama_line_error'] = other_line_error
        if reason is None and initial_line_error - other_line_error >= max(2., .2 * initial_line_error):
            if alternative <= max(1.5, strength + .35):
                reason = 'line_continuity'
            else:
                excess = outline_strength(crop, alternate, commit, relative=True)
                audit['lama_outline_above_surroundings'] = excess
                if excess is not None and excess <= 0:
                    reason = 'line_continuity'
    if reason:
        audit['reason'] = reason
        audit['selected'] = 'lama-onnx-512'
        return retry, context, {**answer, 'outline_review': audit}
    if bridges and initial_line_error >= 4:
        from . import stroke_repair
        refined = stroke_repair.refine(crop, commit, inference | protected, initial)
        if refined is not None:
            refined_error = line_continuity.error(refined, bridges)
            if initial_line_error - refined_error >= max(2., .2 * initial_line_error):
                audit.update(reason='preserved_paint', refined_line_error=refined_error,
                             sdxl_output_pixels_sha256=pixels_hash(result['pixels']))
                result = {**result, 'pixels': refined, 'output_pixels_sha256': pixels_hash(refined)}
    return req, inference, result


def region_backend(backend, crop, mask):
    """Choose using only surviving pixels immediately around the lettering."""
    if backend is None or backend.config.get('backend') != 'auto':
        return backend
    import cv2

    models = backend.config['models']
    if not models or not mask.any():
        return None
    binary = mask.astype(np.uint8)
    outer = cv2.dilate(binary, np.ones((25, 25), np.uint8))
    inner = cv2.dilate(binary, np.ones((7, 7), np.uint8))
    ring = outer > inner
    choice = 'lama'
    if ring.sum() >= 32:
        pixels = crop[ring].astype(np.float32)
        spread = (np.percentile(pixels, 90, axis=0) - np.percentile(pixels, 10, axis=0)).max()
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edge_fraction = np.mean(cv2.Canny(gray, 45, 110)[ring] != 0)
        if spread <= 12 and edge_fraction < 0.012:
            return None
        # Prefer structure completion for ink and screen tones. Diffusion is
        # reserved for broad holes amid varied paint without strong contours.
        width = cv2.distanceTransform(binary, cv2.DIST_L2, 3).max()
        if (width >= 5 and np.median(np.ptp(pixels, axis=1)) >= 20
                and spread >= 32 and edge_fraction < 0.10 and 'sdxl' in models):
            choice = 'sdxl'
    if choice == 'sdxl':
        return LocalReconstruction({**models[choice], 'prefill': 'classical_context'})
    return LocalReconstruction(models[choice]) if choice in models else None


# Existing callers may inject the LaMa worker by its original name.
LocalLama = LocalReconstruction
