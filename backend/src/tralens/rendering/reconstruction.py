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
    return {key:config.get(key) for key in ('backend','sha256','config_revision')}


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


# Existing callers may inject the LaMa worker by its original name.
LocalLama = LocalReconstruction
