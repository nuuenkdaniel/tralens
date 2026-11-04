"""Explicit bounded crop predictions using the already-open local OCR engine.

No raw boxes, readings, proposal IDs, translations or source pixels are rewritten.
Prediction rectangles are evidence only; rendering independently derives strokes.
"""
from dataclasses import asdict
import copy
import hashlib
from io import BytesIO
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..translation.contracts import automatic_units, canonical, revision


POLICY = 'original_crop_localization_evidence_v1'
MAX_CALLS = 12
MAX_CROP_PIXELS = 262144
FIELDS = ('dt_polys','rec_texts','rec_scores','rec_polys','text_word_region')


def execute_requests(preview, plan, *, image, output, model_identity, reader=None, replay=None):
    """Crop-only pass on coordinator-supplied demand, never full OCR/translation.

    `replay` is saved evidence, not a provider fallback. Missing saved predictions
    remain explicit; they never cause inference. The caller owns reader lifetime.
    """
    started = time.monotonic()
    if reader is not None and replay is not None:
        raise ValueError('choose explicit saved replay or a supplied crop reader')
    raw = preview['ocr']['raw_detections'][0]
    units = automatic_units(preview['ocr']['batches'][0],
        (preview['ocr'].get('automatic_recognition_results') or [()])[0])
    bound_preview = revision({**preview, 'ocr': {k:v for k,v in preview['ocr'].items() if k != 'source_localization'}})
    if (plan.get('policy') != 'unresolved_cleanup_crop_demand_v1'
            or plan['source'] != preview['input'] or plan['preview_revision'] != bound_preview
            or plan['geometry_revision'] != geometry_revision(raw)
            or model_identity.get('profile') != 'v6_medium' or model_identity.get('local_weights') is not True):
        raise ValueError('localization request source/model/preview drift')
    requests = plan['requests']
    rows = plan['records']
    by_id = {u.unit_id:u for u in units}
    if (type(requests) is not list or len(requests) != len(set(requests)) or len(requests) > 4
            or not set(requests) <= set(by_id) or len(rows) != len(units)
            or len({r['unit_id'] for r in rows}) != len(rows)
            or {r['unit_id'] for r in rows if r['status'] == 'requested'} != set(requests)):
        raise ValueError('localization request coverage/budget drift')
    with Path(image).open('rb') as stream:content = stream.read(32*1024*1024+1)
    if len(content) > 32*1024*1024 or hashlib.sha256(content).hexdigest() != preview['input']['sha256']:
        raise ValueError('localization request original image drift')
    with Image.open(BytesIO(content)) as opened:
        if opened.width*opened.height > 4_000_000:
            raise ValueError('localization original pixel budget')
        rgb = opened.convert('RGB')
    from .recognition_units import image_fingerprint
    if image_fingerprint(rgb) != preview['input']['pixels_sha256']:
        raise ValueError('localization request original pixels drift')
    row_map = {}
    for row in rows:
        unit = by_id.get(row['unit_id'])
        if (unit is None or row['unit_revision'] != unit.revision
                or canonical(row['member_ids']) != canonical(unit.member_ids)):
            raise ValueError('localization request unit drift')
        row_map[unit.unit_id] = row
    crop_pixels = retry_work = 0
    for uid in requests:
        row = row_map[uid]
        bounds = crop_bounds(raw, row['member_ids'], rgb.size)
        crop = rgb.crop(bounds)
        crop_pixels += crop.width*crop.height
        retry_work += rgb.width*rgb.height*(len(raw)+128)
        if (bounds != row['bounds'] or hashlib.sha256(crop.tobytes()).hexdigest() != row['crop_pixels_sha256']
                or crop.width*crop.height > MAX_CROP_PIXELS or max(crop.size) > 1024):
            raise ValueError('localization request crop drift/budget')
    if crop_pixels > 4_000_000 or retry_work > 1_024_000_000:
        raise ValueError('localization request aggregate budget')
    saved = {}
    if replay is not None:
        if (replay.get('policy') != POLICY or replay['source_sha256'] != preview['input']['sha256']
                or replay['source_pixels_sha256'] != hashlib.sha256(rgb.tobytes()).hexdigest()
                or replay['geometry_revision'] != geometry_revision(raw)
                or canonical(replay['model_identity']) != canonical(model_identity)):
            raise ValueError('saved localization evidence binding drift')
        saved = {r['unit_id']:r for r in replay['records']}
        if len(saved) != len(replay['records']):
            raise ValueError('duplicate saved localization records')
    output = Path(output)
    if output.exists() or not output.parent.is_dir() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('crop-only output must be fresh without symlinks')
    output.mkdir(mode=0o700)
    stage = {'policy':POLICY, 'request_plan':copy.deepcopy(plan), 'source_sha256':preview['input']['sha256'],
        'source_pixels_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(), 'geometry_revision':geometry_revision(raw),
        'model_identity':model_identity, 'calls':0, 'reused_predictions':0, 'maximum_calls':len(requests),
        'crop_pixels':0, 'artifact_bytes':0, 'artifacts':{}, 'records':[]}
    for unit in units:
        row = row_map[unit.unit_id]
        stage['records'].append({'unit_id':unit.unit_id,'unit_revision':unit.revision,
            'member_ids':list(unit.member_ids),'status':'not_attempted','reason':row['reason']})
    records = {r['unit_id']:r for r in stage['records']}
    def retain(name, data):
        if stage['artifact_bytes']+len(data) > 32*1024*1024:
            raise ValueError('localization artifact byte budget')
        with (output/name).open('xb') as stream:stream.write(data)
        stage['artifact_bytes'] += len(data)
        stage['artifacts'][name] = {'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)}
    for index, uid in enumerate(requests):
        row, item = row_map[uid], records[uid]
        item.update(bounds=row['bounds'],crop_pixels_sha256=row['crop_pixels_sha256'])
        crop = rgb.crop(row['bounds'])
        prefix = f'source-localization-{index:03d}'
        stream = BytesIO(); crop.save(stream,format='PNG'); retain(prefix+'.png',stream.getvalue())
        item['native_artifact'] = prefix+'.png'
        try:
            if replay is not None:
                old = saved.get(uid)
                if old is None or old['status'] != 'predicted':
                    item.update(reason='saved_prediction_unavailable')
                    continue
                if any(canonical(old.get(k)) != canonical(item[k]) for k in
                       ('unit_revision','member_ids','bounds','crop_pixels_sha256')):
                    raise ValueError('saved crop source drift')
                prediction = old['prediction']
                if old['prediction_revision'] != revision(prediction):
                    raise ValueError('saved crop prediction drift')
                stage['reused_predictions'] += 1
                item['replay_origin_revision'] = revision(old)
            else:
                if reader is None:
                    raise ValueError('explicit existing crop reader required')
                stage['calls'] += 1
                stage['crop_pixels'] += crop.width*crop.height
                result = reader.predict(cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR),return_word_box=True)
                if len(result) != 1:
                    raise ValueError('one cropped prediction page required')
                prediction = getattr(result[0],'res',result[0])
            def arrays(value):
                if isinstance(value,np.ndarray):return value.tolist()
                if isinstance(value,np.generic):return value.item()
                if isinstance(value,(list,tuple)):return [arrays(v) for v in value]
                return value
            retained = {k:arrays(prediction[k]) for k in FIELDS}
            retain(prefix+'.json',canonical(retained).encode())
            item.update(status='predicted',reason=None,prediction=retained,prediction_artifact=prefix+'.json',
                        prediction_revision=revision(retained))
        except Exception as error:
            item.update(status='failed',reason=type(error).__name__+': '+str(error)[:512])
    with Path(image).open('rb') as stream:current = stream.read(32*1024*1024+1)
    if hashlib.sha256(current).hexdigest() != preview['input']['sha256']:
        raise ValueError('localization original changed during crop pass')
    stage['execution_seconds'] = time.monotonic()-started
    derived = copy.deepcopy(preview)
    derived['ocr']['source_localization'] = stage
    # New evidence, not a new execution journal or a translation rerun.
    with (output/'preview.json').open('x') as stream:stream.write(canonical(derived))
    return derived


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
