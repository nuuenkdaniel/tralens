"""Opt-in atomic clean-balloon experiment over an immutable native render.

Consumes source-bound dialogue requests, never invokes OCR/translation. Existing
opaque display is reused only AFTER native and new clean edits have committed.
"""
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..balloons import LightInteriors, mask_revision, require
from ..translation import coverage as tc
from ..translation import balloon_dialogue as td
from ..translation.context import saved_requests
from . import natural as n, preparation as p, visible_translations
from .automatic import _font, apply_cleanup
from .balloon_clean import derive


POLICY = 'atomic_plain_balloon_experiment_v1'
MAX_ATTEMPTS = 8
MAX_CELL_WORK = 512_000_000


def load(*, preview, image, fonts, baseline):
    paths, contents, doc, fonts_doc, rgb, source = p._sources(preview,image,fonts)
    contract = {'schema_version':1, 'purpose':'automatic_candidate_only', 'source':source,
        'preview_sha256':n.digest(contents[0]), 'fonts_sha256':n.digest(contents[2]), 'units':[]}
    n.bind(doc,paths[1],rgb,contents[1],contents[0],contract,contents[2],selection_kind='automatic_candidate_only')
    baseline = Path(baseline)
    manifest_bytes = n.read_explicit(baseline/'manifest.json')
    native = n.strict_json(manifest_bytes.decode())
    require(native['source'] == source and native['preview_sha256'] == n.digest(contents[0])
        and native['fonts_sha256'] == n.digest(contents[2]), 'balloon native source/preview/font binding drift')
    require(native.get('visibility_policy','replacement-only') == 'replacement-only',
        'balloon experiment needs native layers before opaque display')
    blobs,consumed = {},0
    for name,sha in native['artifacts'].items():
        require(Path(name).name == name, 'unsafe baseline artifact path')
        content = n.read_explicit(baseline/name)
        consumed += len(content)
        require(consumed <= 128*1024*1024, 'balloon baseline byte budget')
        require(n.digest(content) == sha, 'balloon baseline artifact hash mismatch')
        blobs[name] = content
    require(sum(map(len,blobs.values())) <= 128*1024*1024, 'balloon baseline byte budget')
    def raster(name,mode):
        with Image.open(BytesIO(blobs[name])) as opened:
            require(opened.size == rgb.size, 'balloon baseline raster size drift')
            return opened.convert(mode)
    clean = raster('clean.png','RGB')
    old_text = raster('text.png','RGBA')
    old_composite = raster('composite.png','RGB')
    require(Image.alpha_composite(clean.convert('RGBA'),old_text).convert('RGB').tobytes() == old_composite.tobytes(),
        'native layer composition mismatch')
    recognition = doc['ocr'].get('automatic_recognition_results',[])
    units = n.automatic_units(doc['ocr']['batches'][0],recognition[0] if recognition else ())
    require(len(units) == len(doc['units']) == len(native['units']), 'balloon native unit count drift')
    contextual = saved_requests(doc,units)
    for record,unit,row in zip(doc['units'],units,native['units']):
        require(row['unit_id'] == unit.unit_id and row['source_record_sha256'] == n.revision(record), 'native unit binding drift')
        if row.get('binding'):
            binding = (tc.bound_result(record,unit,request=contextual.get(unit.unit_id))
                if record.get('input_kind') == tc.KIND else n.strict_result_binding(record,unit,request=contextual.get(unit.unit_id)))
            require(binding == row['binding'], 'native translation binding drift')
    native_masks, native_letters = {}, {}
    for i,row in enumerate(native['units']):
        if row['status'] == 'automatic_composed':
            native_masks[i] = np.asarray(raster(f'unit-{i:03d}-applied-erase.png','L')) != 0
            native_letters[i] = raster(f'unit-{i:03d}-text.png','RGBA')
    return {'paths':paths, 'contents':contents, 'doc':doc, 'fonts_doc':fonts_doc, 'rgb':rgb,
        'source':source, 'units':units, 'native':native, 'clean':clean, 'text':old_text,
        'composite':old_composite, 'masks':native_masks, 'letters':native_letters,
        'baseline_manifest_sha256':n.digest(manifest_bytes),
        'context':LightInteriors(rgb,doc['ocr']['raw_detections'][0])}


def run(loaded, candidates, proposals, translations, *, output, visibility_policy='always-show'):
    visible_translations.settings(visibility_policy)
    output = Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and
        not any(p.is_symlink() for p in (output,*output.parents)), 'balloon output must be fresh without symlinks')
    context,rgb,units = loaded['context'],loaded['rgb'],loaded['units']
    font_id,(font_data,coverage) = _font(loaded['fonts_doc'],loaded['native']['font_id'])
    style = loaded['native']['style']
    historical = deepcopy(loaded['native']['units'])
    rows, accepted, suppressed, failures = deepcopy(historical), [], set(), []
    candidate_by_id = {c['candidate_id']:c for c in candidates}
    require(len(candidate_by_id) == len(candidates) <= 4 and len(proposals) <= 4, 'balloon candidate/proposal budget')
    require(len({p['candidate_id'] for p in proposals}) == len(proposals), 'duplicate balloon proposals')
    groups = []
    translated_by_id = {r['request_revision']:r for r in translations}
    require(len(translated_by_id) == len(translations) <= 4, 'balloon translation budget')
    for proposal in proposals:
        candidate = candidate_by_id[proposal['candidate_id']]
        require(proposal['evidence'] == candidate['evidence'], 'stale balloon proposal evidence')
        if proposal['status'] != 'proposed' or proposal['proposal']['surface'] != 'plain_dialogue':
            failures.append({'candidate_id':candidate['candidate_id'],'reason':'unsupported/uncertain visual surface proposal'})
            continue
        # Reparse even a saved proposal; model output cannot bypass the exact
        # partition/order/relation checks through a modified Python dictionary.
        from ..ocr.balloon_dialogue import parse
        parse(candidate,json.dumps(proposal['proposal']))
        label_records = {r['label']:r for r in candidate['evidence']['units']}
        indices = {u.unit_id:i for i,u in enumerate(units)}
        for group in proposal['proposal']['groups']:
            parents = [indices[label_records[label]['unit_id']] for label in group['order']]
            require(all(units[i].revision == label_records[label]['unit_revision']
                for i,label in zip(parents,group['order'])), 'stale balloon singleton unit revision')
            if len(parents) == 1:
                i = parents[0]
                if historical[i]['status'] == 'automatic_composed' or not historical[i].get('binding'):
                    continue
                if historical[i]['binding']['text'] == units[i].selected.text:
                    continue
                if historical[i]['binding'].get('reread_selected_revision'):
                    failures.append({'parents':parents,'reason':'printed-cell route does not certify changed raw/alternate glyph counts'})
                    continue
                unit = units[i]
                target = historical[i]['binding']['text']
                record = None
            else:
                try:
                    request,evidence = td.derive(candidate,proposal,group,units,loaded['doc']['ocr']['batches'][0])
                except ValueError as error:
                    failures.append({'parents':parents,'reason':str(error)});continue
                record = translated_by_id.get(request.revision)
                if record is None:
                    failures.append({'parents':parents,'reason':'derived grouping has no fresh translation'});continue
                require(n.canonical(record['request']) == n.canonical(asdict(request))
                    and record['request_revision'] == request.revision, 'derived translation request drift')
                from ..translation.contracts import TranslationResult, UnitOutcome
                data = dict(record['result'])
                data['translations'] = tuple(tuple(pair) for pair in data['translations'])
                data['outcomes'] = tuple(UnitOutcome(**{**o,'reasons':tuple(o['reasons'])}) for o in data['outcomes'])
                result = TranslationResult(**data)
                result.assert_current(request)
                if result.status != 'preview':
                    failures.append({'parents':parents,'reason':'derived translation failed/abstained'});continue
                unit = request.units[0]
                target = dict(result.translations)[unit.unit_id]
            groups.append((len(parents) == 1,candidate,parents,unit,target,record))
    groups.sort(key=lambda item:(item[0],min(item[2])))
    attempted,work = 0,0

    def native_reservation(excluding):
        mask = np.zeros(context.pixels.shape[:2],bool)
        for i,erase in loaded['masks'].items():
            if i in excluding or i in suppressed:
                continue
            mask |= erase | (np.asarray(loaded['letters'][i].getchannel('A')) != 0)
        for plan,layer,*_ in accepted:
            mask |= plan.erase | (np.asarray(layer.getchannel('A')) != 0)
        return mask

    for _,candidate,parents,unit,target,record in groups:
        failure = {'candidate_id':candidate['candidate_id'],'parents':parents,'unit_id':unit.unit_id}
        if set(parents) & suppressed:
            failures.append({**failure,'reason':'overlapping proposed parent already superseded'});continue
        cells = sum(len(context.raw[(type(i).__name__,i)]['text']) for i in unit.member_ids)
        cost = context.size[0]*context.size[1]*(cells+len(unit.member_ids)+8)
        if attempted >= MAX_ATTEMPTS or work+cost > MAX_CELL_WORK:
            failures.append({**failure,'reason':'independent balloon cleanup work budget'});continue
        attempted += 1;work += cost
        try:
            confidence = loaded['native'].get('replacement_policy')
            source_assessments = []
            if confidence is not None:
                from .replacement_confidence import assess_source
                source_assessments = [assess_source(units[i],target,confidence['minimum_main_line_score']) for i in parents]
                require(all(a['allowed'] for a in source_assessments), 'balloon retains configured source-confidence gate')
            from ..balloons import dilate
            reserved = dilate(native_reservation(set(parents)),2)
            interior = context.locate(unit.member_ids)
            require(mask_revision(interior.envelope) == candidate['evidence']['boundary_mask_revision'],
                'balloon cleanup boundary binding drift')
            coordinated = [member for i in candidate['indices'] for member in units[i].member_ids]
            plan = derive(context,interior,list(unit.member_ids),reserved=reserved,coordinated_ids=coordinated)
            from .balloon_flow import fit
            plan,layer,proof,content = fit(target,plan,font_data,coverage,style)
            alpha = np.asarray(layer.getchannel('A')) != 0
            require(proof['original_text'] == target and proof['size'] >= style['min_size'], 'balloon exact target/font floor drift')
            require(not (alpha & (~plan.layout | plan.obstacles)).any(), 'balloon alpha outside flow frame')
            binding = deepcopy(historical[parents[0]]['binding']) if record is None else {
                'member_ids':list(unit.member_ids),'text':target,'unit_revision':unit.revision,
                'selected_reading_revision':unit.selected_revision,'request_revision':record['request_revision']}
            row = {'unit_id':unit.unit_id,'status':'automatic_composed','binding':binding, 'reason_code':'balloon_clean',
                'cleanup_method':plan.evidence['policy'], 'layout':proof,
                'parent_indices':parents, 'translation_request_revision':record['request_revision'] if record else None,
                'pixel_ownership':plan.evidence, 'display_status':'clean', 'speaker':'unknown'}
            row['source_assessments'] = source_assessments
            accepted.append((plan,layer,row,unit,parents,content))
            if len(parents) > 1:
                suppressed.update(parents)
            else:
                rows[parents[0]] = row
        except (ValueError,OSError) as error:
            failures.append({**failure,'reason':str(error)})
    plate = np.asarray(rgb).copy()
    text = Image.new('RGBA',rgb.size)
    union = np.zeros(context.pixels.shape[:2],bool)
    for i,erase in loaded['masks'].items():
        if i in suppressed:
            continue
        plate[erase] = np.asarray(loaded['clean'])[erase]
        text.alpha_composite(loaded['letters'][i]);union |= erase
    native_only = not accepted
    for plan,layer,row,unit,parents,content in accepted:
        apply_cleanup(plate,plan);text.alpha_composite(layer);union |= plan.erase
    clean = Image.fromarray(plate,'RGB')
    composite = Image.alpha_composite(clean.convert('RGBA'),text).convert('RGB')
    allowed_change = np.zeros_like(union)
    for plan,layer,row,unit,parents,content in accepted:
        allowed_change |= plan.erase | (np.asarray(layer.getchannel('A')) != 0)
        for i in parents:
            if i in suppressed and i in loaded['masks']:
                allowed_change |= loaded['masks'][i] | (np.asarray(loaded['letters'][i].getchannel('A')) != 0)
    before,after = np.asarray(loaded['composite']),np.asarray(composite)
    require(np.array_equal(before[~allowed_change],after[~allowed_change]), 'balloon outside-footprint pixel change')
    for i,erase in loaded['masks'].items():
        if i not in suppressed:
            occupied = erase | (np.asarray(loaded['letters'][i].getchannel('A')) != 0)
            require(np.array_equal(before[occupied],after[occupied]), 'balloon displaced native success')
    if native_only:
        require(composite.tobytes() == loaded['composite'].tobytes(), 'balloon rollback changed native output')
    active_units = [u for i,u in enumerate(units) if i not in suppressed]
    active_rows = [r for i,r in enumerate(rows) if i not in suppressed]
    for plan,layer,row,unit,parents,content in accepted:
        if len(parents) > 1:
            active_units.append(unit);active_rows.append(row)
    members = [i for row in active_rows if row['status']=='automatic_composed' for i in row['binding']['member_ids']]
    require(len(set(members)) == len(members), 'duplicate rendered source membership')
    artifacts = {'source.png':rgb, 'native-composite.png':loaded['composite'], 'clean.png':clean,
        'clean-text.png':text, 'clean-composite.png':composite,
        'erase-mask.png':Image.fromarray(union.astype(np.uint8)*255),
        'changed-footprint.png':Image.fromarray(allowed_change.astype(np.uint8)*255)}
    summary = None
    if visibility_policy == 'always-show':
        composite,text,patch_layer,fallback_text,footprint,tiles,summary = visible_translations.display(
            rgb,loaded['doc']['ocr']['raw_detections'][0],active_units,active_rows,
            clean,text,composite,font_data,coverage,style,native_erase=union)
        artifacts.update({'patch.png':patch_layer,'fallback-text.png':fallback_text,
            'fallback-mask.png':Image.fromarray(footprint.astype(np.uint8)*255)})
    artifacts.update({'text.png':text,'composite.png':composite})
    for j,(plan,layer,row,unit,parents,content) in enumerate(accepted):
        for name,mask in (('cleanup',plan.erase),('interior',plan.layout),('obstacles',plan.obstacles)):
            artifacts[f'balloon-{j:03d}-{name}.png'] = Image.fromarray(mask.astype(np.uint8)*255)
        artifacts[f'balloon-{j:03d}-text.png'] = layer
        original_interior = context.locate(unit.member_ids)
        artifacts[f'balloon-{j:03d}-boundary.png'] = Image.fromarray(original_interior.envelope.astype(np.uint8)*255)
        artifacts[f'balloon-{j:03d}-core.png'] = Image.fromarray(original_interior.core.astype(np.uint8)*255)
    report = {'policy':POLICY,'experimental':True,'source':loaded['source'],
        'preview_sha256':n.digest(loaded['contents'][0]),'fonts_sha256':n.digest(loaded['contents'][2]),
        'baseline_manifest_sha256':loaded['baseline_manifest_sha256'],
        'accepted':[row for _,_,row,*_ in accepted], 'failures':failures,
        'suppressed_historical_indices':sorted(suppressed), 'historical_units':historical,
        'active_units':active_rows, 'display_summary':summary, 'attempted':attempted,
        'cleanup_work':work, 'maximum_cleanup_work':MAX_CELL_WORK,
        'maximum_flow_frames_per_dialogue':6, 'maximum_layout_raster_work_per_frame':64_000_000,
        'native_preserved':len(loaded['masks'])-len(set(loaded['masks']) & suppressed),
        'native_atomically_superseded':len(set(loaded['masks']) & suppressed),
        'new_clean_dialogues':len(accepted), 'outside_declared_changed_pixels':0,
        'duplicate_rendered_members':0,'semantic_accuracy_verified':False,
        'visibility_policy':visibility_policy}
    blobs = dict(zip(('evidence-preview.json','original-file.bin','evidence-fonts.json'),loaded['contents']))
    for name,raster in artifacts.items():
        stream = BytesIO();raster.save(stream,format='PNG');blobs[name] = stream.getvalue()
    report['artifacts'] = {name:n.digest(data) for name,data in blobs.items()}
    blobs['manifest.json'] = (json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
    require(len(blobs) <= 96 and sum(map(len,blobs.values())) <= 128*1024*1024, 'balloon artifact publication budget')
    require(all(n.read_explicit(path) == data for path,data in zip(loaded['paths'],loaded['contents'])),
        'balloon source inputs changed before publication')
    n.load_fonts(loaded['fonts_doc'])
    output.mkdir(mode=0o700)
    for name,data in blobs.items():
        with (output/name).open('xb') as stream:
            stream.write(data)
    return report
