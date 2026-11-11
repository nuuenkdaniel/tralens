"""Standalone installed-runtime entry point. No downloads and no upstream code."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main(directory):
    # Limits precede native imports/model parsing; CPU execution only.
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    import numpy as np
    from PIL import Image
    import onnxruntime as ort
    doc=json.loads((directory/'request.json').read_bytes());config=doc['config'];req=doc['request']
    model=Path(config['model']).read_bytes()
    if len(model)!=208044816 or hashlib.sha256(model).hexdigest()!=config['sha256']:
        raise ValueError('model bytes changed before worker inference')
    with Image.open(directory/'input.png') as image:rgb=np.asarray(image.convert('RGB'))
    with Image.open(directory/'mask.png') as image:mask=np.asarray(image.convert('L'))>0
    h,w=mask.shape
    if rgb.shape!=(h,w,3) or not 0<max(h,w)<=512:raise ValueError('invalid native crop')
    # Reflect masked lettering as masked too. Zero mask padding would expose
    # mirrored source glyphs to the model as known context.
    image=np.pad(rgb,((0,512-h),(0,512-w),(0,0)),mode='symmetric')
    removal=np.pad(mask,((0,512-h),(0,512-w)),mode='symmetric')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    started=time.monotonic();session=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])
    session.disable_fallback()
    inputs=session.get_inputs()
    feed={}
    for value in inputs:
        if value.type!='tensor(float)' or len(value.shape)!=4 or value.shape[0] not in (1,'batch'):
            raise ValueError('unexpected LaMa input type/batch')
        if value.shape[1:]==[3,512,512]:feed[value.name]=image.transpose(2,0,1)[None].astype(np.float32)/255
        elif value.shape[1:]==[1,512,512]:feed[value.name]=removal[None,None].astype(np.float32)
        else:raise ValueError('unexpected LaMa input contract '+str(value.shape))
    if len(feed)!=2:raise ValueError('two model inputs required')
    load=time.monotonic()-started;started=time.monotonic()
    outputs=session.run(None,feed)
    if len(outputs)!=1 or outputs[0].shape!=(1,3,512,512) or not np.isfinite(outputs[0]).all():
        raise ValueError('invalid LaMa output contract')
    result=np.clip(outputs[0][0].transpose(1,2,0)[:h,:w],0,255).astype(np.uint8)
    elapsed=time.monotonic()-started
    Image.fromarray(result).save(directory/'output.png')
    (directory/'result.json').write_text(json.dumps({'request_revision':req['revision'],
        'model_sha256':config['sha256'],'backend':'lama-onnx-512','runtime':ort.__version__,
        'providers':session.get_providers(),'load_seconds':load,'inference_seconds':elapsed,
        'output_minimum':float(outputs[0].min()),'output_maximum':float(outputs[0].max()),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pid':os.getpid()}))
