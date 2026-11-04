"""Explicit bounded crop predictions using the already-open local OCR engine.

No raw boxes, readings, proposal IDs, translations or source pixels are rewritten.
Prediction rectangles are evidence only; rendering independently derives strokes.
"""
from dataclasses import asdict
import hashlib
from io import BytesIO
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..translation.contracts import automatic_units, canonical, revision


POLICY = 'original_crop_localization_evidence_v1'
MAX_CALLS = 12
MAX_CROP_PIXELS = 262144
FIELDS = ('dt_polys','rec_texts','rec_scores','rec_polys','text_word_region')


def geometry_revision(detections):
    return revision([(type(d['source_id']).__name__,d['source_id'],d['polygon'],d['text'],d['confidence'])
                     for d in detections])


def crop_bounds(detections, member_ids, size):
    selected={(type(i).__name__,i) for i in member_ids}
    points=[pt for d in detections if (type(d['source_id']).__name__,d['source_id']) in selected for pt in d['polygon']]
    if not points:
        raise ValueError('localization crop has no original source geometry')
    xs,ys=zip(*points)
    return [max(0,math.floor(min(xs))-24),max(0,math.floor(min(ys))-24),
        min(size[0],math.ceil(max(xs))+25),min(size[1],math.ceil(max(ys))+25)]


def refine(engine, *, output, max_calls=MAX_CALLS):
    if type(max_calls) is not int or not 0 <= max_calls <= MAX_CALLS:
        raise ValueError('bounded crop-localization call budget required')
    if hasattr(engine,'source_localization'):
        raise ValueError('source localization already attempted; use a fresh OCR run')
    batches=engine.automatic_unit_proposals
    if len(batches)!=1 or batches[0] is None:
        return
    batch=batches[0]
    def read_original():
        with Path(engine.image_path).open('rb') as stream:return stream.read(32*1024*1024+1)
    content=read_original()
    if len(content)>32*1024*1024 or hashlib.sha256(content).hexdigest()!=batch.page.image_sha256:
        raise ValueError('localization original image binding mismatch')
    with Image.open(BytesIO(content)) as opened:
        image=opened.convert('RGB')
    from .recognition_units import image_fingerprint
    if image.width*image.height>4_000_000 or image_fingerprint(image)!=batch.page.pixels_sha256:
        raise ValueError('localization original pixel binding/budget mismatch')
    candidates=getattr(engine,'automatic_recognition_results',())
    units=automatic_units(batch,candidates[0] if candidates else ())
    raw=[asdict(d) for d in batch.detections]
    by_id={(type(d['source_id']).__name__,d['source_id']):d for d in raw}
    stage={'policy':POLICY,'source_sha256':hashlib.sha256(content).hexdigest(),
        'source_pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
        'geometry_revision':geometry_revision(raw),'model_identity':engine.paddle_provenance,
        'calls':0,'maximum_calls':max_calls,'crop_pixels':0,'artifact_bytes':0,'artifacts':{},'records':[]}
    engine.source_localization=stage
    output=Path(output)
    def retain(name, value):
        if stage['artifact_bytes']+len(value)>32*1024*1024 or len(stage['artifacts'])>=2*MAX_CALLS:
            raise ValueError('localization artifact byte/count budget')
        with (output/name).open('xb') as stream:stream.write(value)
        stage['artifact_bytes']+=len(value)
        stage['artifacts'][name]={'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)}
    jobs=sorted(zip(batch.proposals,units),key=lambda pair:(len(pair[0].main_ids)<2,-len(pair[1].selected.text),pair[0].proposal_id))
    for index,(proposal,unit) in enumerate(jobs):
        item={'unit_id':unit.unit_id,'unit_revision':unit.revision,'member_ids':list(unit.member_ids),
              'status':'not_attempted','reason':'source_not_eligible_for_localization'}
        stage['records'].append(item)
        main=[by_id[(type(i).__name__,i)] for i in proposal.main_ids]
        if (proposal.annotation_ids or len(proposal.main_ids)>8 or proposal.main_ids!=proposal.member_ids
                or any(d['confidence']<.7 for d in main)
                or any(not d['text'].strip() for d in main)):
            continue
        if stage['calls']>=max_calls:
            item['reason']='localization_call_budget';continue
        bounds=crop_bounds(raw,proposal.member_ids,image.size)
        native=image.crop(bounds)
        if native.width*native.height>MAX_CROP_PIXELS or max(native.size)>1024:
            item['reason']='localization_crop_budget';continue
        if stage['crop_pixels']+native.width*native.height>4_000_000:
            item['reason']='localization_aggregate_crop_budget';continue
        prefix=f'source-localization-{index:03d}'
        item.update(bounds=bounds,crop_pixels_sha256=hashlib.sha256(native.tobytes()).hexdigest())
        stream=BytesIO();native.save(stream,format='PNG');retain(prefix+'.png',stream.getvalue())
        item['native_artifact']=prefix+'.png'
        try:
            stage['calls']+=1;stage['crop_pixels']+=native.width*native.height
            result=engine.ocr.predict(cv2.cvtColor(np.asarray(native),cv2.COLOR_RGB2BGR),return_word_box=True)
            if len(result)!=1:
                raise ValueError('one cropped prediction page required')
            prediction=getattr(result[0],'res',result[0])
            def arrays(value):
                if isinstance(value,np.ndarray):return value.tolist()
                if isinstance(value,np.generic):return value.item()
                if isinstance(value,(list,tuple)):return [arrays(v) for v in value]
                if isinstance(value,dict):return {k:arrays(v) for k,v in value.items()}
                return value
            # Rendering consumes only these association/localization fields.
            # UI-only font objects are not source evidence and aren't serialized.
            retained=arrays({key:prediction[key] for key in FIELDS})
            retain(prefix+'.json',canonical(retained).encode())
            item.update(status='predicted',reason=None,prediction=retained,prediction_artifact=prefix+'.json',
                        prediction_revision=revision(retained))
        except Exception as error:
            item.update(status='failed',reason=type(error).__name__+': '+str(error)[:512])
    if hashlib.sha256(read_original()).hexdigest()!=stage['source_sha256']:
        raise ValueError('localization original image changed during inference')
