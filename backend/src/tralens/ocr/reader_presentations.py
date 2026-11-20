"""Optional reader-only evidence over every saved detector polygon.

Plan from geometry, deduplicate exact reader pixels, and retain every association
and failed response. No candidates, selections, detector memberships or utterance
revisions are modified. The coordinator supplies and owns the reader lifetime.
"""
from dataclasses import asdict, is_dataclass
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import reader_geometry
from .recognition_units import image_fingerprint
from ..translation.contracts import automatic_units, canonical, revision, strict_json


POLICY = 'all_detector_quad_reader_evidence_v1'
DIRECTORY = 'reader-presentations'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(value):
    return strict_json(canonical(asdict(value) if is_dataclass(value) else value))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (canonical(value)+'\n').encode()


def artifact(name, data, output=None):
    path=DIRECTORY+'/'+name
    if output is not None:
        with (Path(output)/path).open('xb') as stream:
            stream.write(data)
    return {'path':path, 'sha256':digest(data)}


def plan(batch, image, *, model_identity, quarter_turns=(0,), output=None):
    """Replayable geometry-only planning; output optionally saves unique inputs.

    All detector records participate, including empty/rejected readings and
    invalid quadrilaterals. Geometric failures remain explicit unavailable views.
    No inspection label, expected text, confidence or reading chooses a crop.
    """
    batch=snapshot(batch)
    automatic_units(batch)
    require(image.mode=='RGB' and image_fingerprint(image)==batch['page']['pixels_sha256'],
        'reader evidence source pixels changed')
    require(type(model_identity) is str and bool(model_identity.strip()), 'reader evidence requires model identity')
    require(type(quarter_turns) in (tuple,list) and 1<=len(quarter_turns)<=4
        and all(type(t) is int and 0<=t<=3 for t in quarter_turns)
        and len(set(quarter_turns))==len(quarter_turns), 'invalid reader evidence orientations')
    value={'policy':POLICY, 'page':batch['page'], 'batch_revision':revision(batch),
        'model_identity':model_identity, 'quarter_turns':list(quarter_turns),
        'views':list(reader_geometry.VIEWS), 'records':[], 'requests':[],
        'selected':False, 'erase_permission':False}
    unique={}
    for detection in batch['detections']:
        for view in reader_geometry.VIEWS:
            for turn in quarter_turns:
                record={'index':len(value['records']), 'source_id':detection['source_id'],
                    'source_polygon':detection['polygon'], 'view':view, 'quarter_turns':turn}
                value['records'].append(record)
                try:
                    crop,evidence=reader_geometry.present(image,detection['polygon'],view=view,quarter_turns=turn)
                except ValueError as error:
                    record.update(status='unavailable', error=type(error).__name__+': '+str(error))
                    continue
                key=(crop.size,evidence['view_pixels_sha256'])
                if key not in unique:
                    index=len(value['requests']);stream=BytesIO();crop.save(stream,format='PNG')
                    request={'index':index, 'size':list(crop.size), 'pixels_sha256':key[1],
                        'input':artifact(f'input-{index:04d}.png',stream.getvalue(),output), 'records':[]}
                    unique[key]=request;value['requests'].append(request)
                request=unique[key];request['records'].append(record['index'])
                record.update(status='planned', presentation=evidence, request_index=request['index'])
    return value


def collect(engine, image, original, *, reader, model_identity, output, quarter_turns=(0,)):
    """Read all planned inputs once; keep predictions separate from selections."""
    require(not hasattr(engine,'reader_presentations'), 'reader presentations already attempted')
    require(len(engine.automatic_unit_proposals)==1, 'one reader evidence page required')
    batch=engine.automatic_unit_proposals[0]
    if batch is None:
        return
    require(digest(original)==snapshot(batch)['page']['image_sha256'], 'reader evidence source bytes changed')
    folder=Path(output)/DIRECTORY;folder.mkdir()
    stage={'policy':POLICY, 'status':'planning', 'plan':None, 'results':[],
        'selected':False, 'erase_permission':False}
    engine.reader_presentations=stage
    try:
        stage['plan']=plan(batch,image,model_identity=model_identity,quarter_turns=quarter_turns,output=output)
        stage['plan_artifact']=artifact('plan.json',encoded(stage['plan']),output)
        stage['status']='reading'
        for request in stage['plan']['requests']:
            response={'index':request['index'], 'status':'failed', 'selected':False, 'erase_permission':False}
            path=Path(output)/request['input']['path']
            # Storage/binding faults are not recognition failures. Do not call a
            # model on pixels that differ from the immutable saved plan.
            require(digest(path.read_bytes())==request['input']['sha256'], 'reader input artifact changed')
            with Image.open(path) as source:
                crop=source.convert('RGB')
            require(image_fingerprint(crop)==request['pixels_sha256'], 'reader input pixels changed')
            try:
                text=reader(crop)
                # Preserve even invalid returned text before applying the bound.
                if type(text) is str:
                    response['text']=text
                else:
                    response['returned_type']=type(text).__name__
                    response['returned_repr']=repr(text)
                require(type(text) is str and len(text)<=512, 'reader response type/length')
                require(image_fingerprint(crop)==request['pixels_sha256'], 'reader mutated its input')
                response['status']='read'
            except Exception as error:
                response['error']=type(error).__name__+': '+str(error)
            saved=artifact(f'response-{request["index"]:04d}.json',encoded(response),output)
            stage['results'].append({**response, 'response':saved})
        unavailable=any(r['status']=='unavailable' for r in stage['plan']['records'])
        failed=any(r['status']!='read' for r in stage['results'])
        stage['status']='partial' if unavailable or failed else 'complete'
    except BaseException:
        stage['status']='interrupted'
        raise
    finally:
        artifact('evidence.json',encoded(stage),output)


def validate_saved(ocr, image, *, output=None):
    """Replay source/geometry/pixel bindings without calling any OCR reader.

    With output, also check every saved input/response and the full evidence file.
    The readings remain diagnostic even when this mechanical validation passes.
    """
    stage=ocr.get('reader_presentations')
    if stage is None:
        return
    require(stage['policy']==POLICY and stage['selected'] is False and stage['erase_permission'] is False,
        'reader evidence policy/selection drift')
    require(len(ocr['batches'])==1 and stage['status'] in ('complete','partial'), 'incomplete reader evidence')
    current=stage['plan']
    expected=plan(ocr['batches'][0],image,model_identity=current['model_identity'],quarter_turns=current['quarter_turns'])
    require(current==expected, 'reader evidence geometry/association drift')
    require(stage['plan_artifact']==artifact('plan.json',encoded(current)), 'reader evidence plan artifact drift')
    results=stage['results'];requests=current['requests']
    require(len(results)==len(requests), 'reader evidence request coverage drift')
    for index,(result,request) in enumerate(zip(results,requests)):
        require(result['index']==request['index']==index and result['selected'] is False
            and result['erase_permission'] is False and result['status'] in ('read','failed'),
            'reader evidence result association drift')
        response={k:v for k,v in result.items() if k!='response'}
        if result['status']=='read':
            require(type(result.get('text')) is str and len(result['text'])<=512 and 'error' not in result,
                'reader evidence prediction drift')
        else:
            require(type(result.get('error')) is str and bool(result['error']), 'missing reader failure evidence')
        require(result['response']==artifact(f'response-{index:04d}.json',encoded(response)),
            'reader evidence response artifact drift')
    partial=any(r['status']=='unavailable' for r in current['records']) or any(r['status']=='failed' for r in results)
    require(stage['status']==('partial' if partial else 'complete'), 'reader evidence completion drift')
    if output is not None:
        artifacts=[stage['plan_artifact']]+[r['input'] for r in requests]+[r['response'] for r in results]
        for item in artifacts:
            require(digest((Path(output)/item['path']).read_bytes())==item['sha256'], 'reader evidence saved artifact changed')
        require(strict_json((Path(output)/DIRECTORY/'evidence.json').read_text())==stage, 'reader evidence snapshot drift')
