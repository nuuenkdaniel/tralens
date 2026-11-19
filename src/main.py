"""Image pipeline coordinator. Stage algorithms live in tralens/, not this file.

Import and --help are engine-free. Without --render, the existing preview-only
CLI and its output format remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time
from functools import wraps


def serialized(function):
    @wraps(function)
    def run(*args,**kwargs):
        from tralens.scheduling import inference_lease
        with inference_lease():return function(*args,**kwargs)
    return run


def run_language_review(*,preview,image,output,run_local_model=False,
                        endpoint='http://100.121.133.88:11434',review_model='qwen3.8:latest',
                        model='translategemma:27b',max_reviews=4,max_reader_calls=8,max_translation_calls=8,
                        proposal_runtime=None,translation_runtime=None,reader_factory=None,idle_check=lambda:None):
    """First language-quality step: independent queue and source candidates.

    No grouping/English-review selection or renderer changes. Old primary input,
    translation and successful page layers remain authoritative and untouched.
    Only changed corroborated sources receive new candidate translation requests.
    """
    import json
    from dataclasses import asdict
    from io import BytesIO
    from PIL import Image
    from tralens import language_review as lr
    from tralens.ocr import source_review as sr
    from tralens.translation import source_recovery as recover
    from tralens.translation.contracts import automatic_units,canonical,revision
    from tralens.translation.workflow import _atomic_json
    from tralens.translation.runtime import OllamaRuntime,PreviewTranslator,TRANSLATEGEMMA,GenerationOptions,local_endpoint
    lr.require(type(run_local_model) is bool and model == 'translategemma:27b',
        'source review retains TG27 primary; explicit boolean inference opt-in required')
    lr.require(review_model == 'qwen3.8:latest','use the explicitly configured existing visual reviewer')
    lr.require(all(type(v) is int and 0 <= v <= 8 for v in (max_reader_calls,max_translation_calls)),
        'independent reader/translation budgets must be 0..8')
    endpoint = local_endpoint(endpoint)
    output = Path(output).absolute();paths = (Path(preview).absolute(),Path(image).absolute())
    lr.require(not output.exists() and output.parent.is_dir() and not any(p.is_symlink() for p in (output,*output.parents)),
        'language review output must be fresh without symlinks')
    lr.require(all(p.stat().st_size <= 32*1024*1024 for p in paths),'source review input byte budget')
    contents = [p.read_bytes() for p in paths];doc = json.loads(contents[0])
    lr.require(hashlib.sha256(contents[1]).hexdigest() == doc['input']['sha256'],'source review original file binding')
    with Image.open(BytesIO(contents[1])) as original:
        lr.require(getattr(original,'n_frames',1) == 1,'one original frame required');rgb = original.convert('RGB')
    plan = lr.plan(doc,rgb,max_requests=max_reviews)
    units = automatic_units(doc['ocr']['batches'][0],(doc['ocr'].get('automatic_recognition_results') or [()])[0])
    by_id = {u.unit_id:u for u in units};rows = {r['unit_id']:r for r in plan['records']}
    state = {'policy':'language_review_foundation_v1','experimental':True,'source':doc['input'],
        'preview_revision':revision(doc),'plan':plan,'source_reviews':[], 'candidate_translations':[],
        'counts':{'requested':len(plan['requests']),'attempted':0,'failed':0,'corroborating_reader_calls':0,
            'reader_failures':0,'translation_failures':0,'corroborated_candidates':0,
            'unchanged_corroborations':0,'uncertain':0,'translation_calls':0},
        'primary_records':doc['units'],'primary_preview_revision':revision(doc),
        'selection_policy':'retain primary; candidates only, never fluency-based automatic promotion',
        'grouping_stage':'not implemented in this foundation','english_review_stage':'not implemented in this foundation',
        'limits':{'source_calls':max_reviews,'reader_calls':max_reader_calls,'candidate_translation_calls':max_translation_calls},
        'status':'running','artifacts':{}}
    output.mkdir(mode=0o700)
    def persist():_atomic_json(output/'review.json',state,replace_owned=True)
    def retain(name,data):
        lr.require(Path(name).name == name and len(data) <= 32*1024*1024
            and sum(v['bytes'] for v in state['artifacts'].values())+len(data) <= 96*1024*1024,
            'source review artifact byte/path budget')
        with (output/name).open('xb') as f:f.write(data)
        state['artifacts'][name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    def assert_input():
        lr.require(all(p.read_bytes() == data for p,data in zip(paths,contents)),'language-review inputs changed')
    retain('primary-preview.json',contents[0]);retain('original-file.bin',contents[1]);persist()
    if not run_local_model:
        state['status'] = 'planned_no_inference';persist();return state
    provider = proposal_runtime if proposal_runtime is not None else OllamaRuntime(endpoint)
    pending = [];reader = close_reader = None
    try:
        for index,uid in enumerate(plan['requests']):
            idle_check();assert_input()
            request,native,view,overlay = sr.prepare(doc,plan,uid,rgb)
            prefix = f'review-{index:03d}'
            for suffix,raster in (('original',native),('presentation',view),('localization-overlay',overlay)):
                retain(prefix+'-'+suffix+'.png',sr.png(raster))
            retain(prefix+'-request.json',canonical(request).encode())
            item = {'unit_id':uid,'request_revision':request['revision'],'binding':request['binding'], 'phase':'visual_source',
                'status':'attempted','model':review_model,'source_verified':False,'primary_retained':True}
            state['source_reviews'].append(item);state['counts']['attempted'] += 1;persist()
            try:
                started = time.monotonic()
                response = provider.generate(model=review_model,payload=request['payload'],timeout=120)
                item['seconds'] = time.monotonic()-started
                text = response.text if hasattr(response,'text') else response
                raw = response.response_json if hasattr(response,'response_json') else text
                retain(prefix+'-response.txt',raw.encode())
                lr.require(type(text) is str and len(text) <= 32768,'source proposal output byte budget')
                candidate = sr.parse(request,text,rgb);item['candidate'] = candidate
                if candidate['clear'] and sr.normalized(candidate['text']) == sr.normalized(rows[uid]['source_text']):
                    item['status'] = 'unchanged_visual_corroboration'
                    state['counts']['unchanged_corroborations'] += 1
                    item['translation_reused'] = rows[uid]['primary'] is not None
                elif not candidate['clear']:
                    item['status'] = 'uncertain_candidate_retained';state['counts']['uncertain'] += 1
                elif reader_factory is None:
                    item['status'] = 'changed_candidate_without_independent_reader'
                elif state['counts']['corroborating_reader_calls'] >= max_reader_calls:
                    item['status'] = 'independent_reader_budget_excluded'
                else:
                    item['phase'] = 'independent_reader'
                    if reader is None:reader,close_reader = reader_factory()
                    def counted(crop):
                        idle_check();assert_input()
                        state['counts']['corroborating_reader_calls'] += 1;persist()
                        return reader(crop)
                    supported,evidence = sr.corroborate(candidate,rgb,counted,
                        maximum_calls=max_reader_calls-state['counts']['corroborating_reader_calls'])
                    item['corroboration'] = evidence
                    state['counts']['reader_failures'] += sum('error' in e for e in evidence)
                    if supported:
                        item['status'] = 'corroborated_changed_candidate_not_selected'
                        state['counts']['corroborated_candidates'] += 1
                        request = recover.request_for(by_id[uid],rows[uid],candidate,evidence,preview=doc,plan=plan,rgb=rgb)
                        pending.append((item,request))
                    else:item['status'] = 'reader_disagreement_or_budget_candidate_retained'
            except Exception as error:
                item.update(status='failed_primary_retained',error=type(error).__name__+': '+str(error)[:1024])
                raw = getattr(error,'response_json',None)
                if isinstance(raw,str):retain(prefix+'-failed-response.txt',raw.encode())
                state['counts']['failed'] += 1
            persist()
    finally:
        if close_reader is not None:
            try:close_reader()
            except Exception as error:state['reader_release_error'] = str(error)[:512]
        if proposal_runtime is None and state['counts']['attempted']:
            try:
                idle_check();state['reviewer_release'] = provider.unload(model=review_model,timeout=30)
            except Exception as error:state['reviewer_release_error'] = str(error)[:512]
            persist()
    provider = translation_runtime if translation_runtime is not None else OllamaRuntime(endpoint)
    translator = PreviewTranslator(provider,model=model,profile=TRANSLATEGEMMA,max_attempts=1,timeout=120,
        generation_options=GenerationOptions(num_ctx=4096,num_predict=1024,keep_alive=60))
    try:
        for i,(item,request) in enumerate(pending):
            if state['counts']['translation_calls'] >= max_translation_calls:
                item['translation_status'] = 'candidate_translation_budget_excluded';continue
            idle_check();assert_input();state['counts']['translation_calls'] += 1;persist()
            events = []
            result = translator.translate(request,observe=lambda event:events.append(event));result.assert_current(request)
            record = {'parent_unit_id':item['unit_id'],'request':asdict(request),'request_revision':request.revision,
                'result':asdict(result),'events':events,'selected':False,'primary_retained':True}
            state['candidate_translations'].append(record);item['translation_status'] = result.status;persist()
            state['counts']['translation_failures'] += result.status == 'failed'
    finally:
        if translation_runtime is None and state['counts']['translation_calls']:
            try:
                idle_check();state['translator_release'] = provider.unload(model=model,timeout=30)
            except Exception as error:state['translator_release_error'] = str(error)[:512]
    assert_input()
    lr.require(state['primary_records'] == doc['units'],'optional source recovery mutated primary results')
    state['status'] = 'complete_candidates_only';persist();return state


def run_balloon_experiment(*, preview, image, fonts, baseline, output, run_local_model=False,
                          endpoint='http://100.121.133.88:11434', proposal_model='qwen3.8:latest',
                          model='translategemma:27b', replay=None, proposal_runtime=None,
                          translation_runtime=None, visibility_policy='always-show'):
    """Coordinate saved-evidence balloon experiment; never rerun page OCR.

    Explicitly separate from default pipeline behavior while its support boundary
    is evaluated. Replay consumes immutable bound proposals/derived translations,
    not arbitrary masks or manual region confirmations.
    """
    import json
    from tralens.ocr import balloon_dialogue as bd
    from tralens.translation import balloon_dialogue as td
    from tralens.translation.runtime import OllamaRuntime, PreviewTranslator, TRANSLATEGEMMA, GenerationOptions, local_endpoint
    from tralens.translation.workflow import _atomic_json
    from tralens.rendering import balloon_pass
    if type(run_local_model) is not bool or (replay is not None and run_local_model):
        raise ValueError('choose bounded model inference or saved replay, never both')
    output = Path(output).absolute()
    if output.exists() or not output.parent.is_dir() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('balloon experiment needs fresh output without symlinks')
    endpoint = local_endpoint(endpoint)
    loaded = balloon_pass.load(preview=preview,image=image,fonts=fonts,baseline=baseline)
    candidates,rejections = bd.candidates(loaded['context'],loaded['units'],loaded['native']['units'])
    state = {'experimental':True,'source':loaded['source'],'preview_sha256':hashlib.sha256(loaded['contents'][0]).hexdigest(),
        'baseline_manifest_sha256':loaded['baseline_manifest_sha256'],
        'candidates':[c['evidence'] for c in candidates], 'geometry_rejections':rejections,
        'proposal_calls':0,'translation_calls':0,'proposals':[],'translations':[],
        'maximum_proposal_calls':4,'maximum_translation_calls':4,'page_ocr_calls':0,
        'grouping_verified':False,'speaker':'unknown','status':'running'}
    output.mkdir(mode=0o700)
    def persist():
        _atomic_json(output/'experiment.json',state,replace_owned=True)
    persist()
    started = time.monotonic()
    try:
        if replay is not None:
            prior = json.loads(Path(replay).read_bytes())
            if any(prior[k] != state[k] for k in ('source','preview_sha256','candidates')):
                raise ValueError('balloon replay source/unit/candidate drift')
            state.update(proposals=prior['proposals'],translations=prior['translations'],
                replay_sha256=hashlib.sha256(Path(replay).read_bytes()).hexdigest())
        elif run_local_model:
            provider = proposal_runtime if proposal_runtime is not None else OllamaRuntime(endpoint)
            try:
                for candidate in candidates:
                    state['proposal_calls'] += 1;persist()
                    state['proposals'].append(bd.propose(candidate,loaded['rgb'],provider,proposal_model));persist()
            finally:
                if proposal_runtime is None and state['proposal_calls']:
                    state['proposal_release'] = provider.unload(model=proposal_model,timeout=30)
            requests = []
            for candidate,proposal in zip(candidates,state['proposals']):
                if proposal['status'] != 'proposed' or proposal['proposal']['surface'] != 'plain_dialogue':
                    continue
                for group in proposal['proposal']['groups']:
                    if len(group['order']) < 2:
                        continue
                    try:
                        request,evidence = td.derive(candidate,proposal,group,loaded['units'],loaded['doc']['ocr']['batches'][0])
                        requests.append(request)
                    except ValueError as error:
                        state.setdefault('derivation_rejections',[]).append(str(error))
            provider = translation_runtime if translation_runtime is not None else OllamaRuntime(endpoint)
            from dataclasses import asdict
            state['derived_translation_requests'] = [asdict(request) for request in requests]
            state['translation_budget_excluded'] = max(0,len(requests)-4)
            persist()
            translator = PreviewTranslator(provider,model=model,profile=TRANSLATEGEMMA,max_attempts=1,timeout=120,
                generation_options=GenerationOptions(num_ctx=4096,num_predict=1024,keep_alive=60))
            try:
                for request in requests[:4]:
                    state['translation_calls'] += 1;persist()
                    state['translations'].append(td.execute(request,translator));persist()
            finally:
                if translation_runtime is None and state['translation_calls']:
                    state['translation_release'] = provider.unload(model=model,timeout=30)
        rendered = balloon_pass.run(loaded,candidates,state['proposals'],state['translations'],
            output=output/'rendering',visibility_policy=visibility_policy)
        state.update(status='complete_experiment_not_verified', rendering='rendering/manifest.json',
            new_clean_dialogues=rendered['new_clean_dialogues'],
            native_preserved=rendered['native_preserved'],native_atomically_superseded=rendered['native_atomically_superseded'],
            display_summary=rendered['display_summary'])
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error)[:2000])
        raise
    finally:
        state['seconds'] = time.monotonic()-started
        persist()
    return state


def crop_localization_pass(preview, plan, *, image, output, device, model_root, ocr_factory=None):
    """Coordinate a fresh crop-only reader; never repeat page OCR or translation."""
    from tralens.ocr.source_localization import execute_requests
    engine = None
    started = time.monotonic()
    try:
        if plan['requests']:
            if ocr_factory is None:
                from tralens.ocr import OCR
                ocr_factory = OCR
            engine = ocr_factory(str(image), device=device, model_root=str(Path(model_root).expanduser()),
                                 model_profile='v6_medium', enable_manga_ocr=False)
            identity = engine.paddle_provenance
        else:
            # No reader construction for an empty request plan.
            from tralens.ocr.paddle_profiles import resolve_profile
            identity = resolve_profile(device=device, model_root=Path(model_root).expanduser()).identity()
        derived = execute_requests(preview, plan, image=image, output=output, model_identity=identity,
                                   reader=engine.ocr if engine is not None else None)
    finally:
        if engine is not None:
            engine.ocr.close()
    return derived, time.monotonic()-started


def quality_refiner(*, endpoint, model, manga_model):
    """Coordinate explicit OCR readers; algorithms/evidence stay in the OCR stage."""
    path = Path(manga_model).expanduser().resolve(strict=True)
    required = ('config.json', 'preprocessor_config.json', 'pytorch_model.bin', 'vocab.txt',
                'special_tokens_map.json', 'tokenizer_config.json')
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise ValueError('quality reread needs the complete existing local Manga OCR snapshot; no downloads')
    if any((path / name).stat().st_size > 512*1024*1024 for name in required):
        raise ValueError('quality reread local model artifact exceeds the explicit byte budget')
    from tralens.translation.runtime import local_endpoint
    endpoint = local_endpoint(endpoint)
    def refine(engine, *, output):
        from io import BytesIO
        from importlib.metadata import version
        import json
        from PIL import Image
        from tralens.ocr import quality_reread
        from tralens.ocr.manga_adapter import MangaOCRAdapter
        from tralens.translation.runtime import OllamaRuntime
        if not engine.automatic_unit_proposals or engine.automatic_unit_proposals[0] is None:
            return
        original = Path(engine.image_path).read_bytes()
        identity = json.dumps({'vision': model, 'vision_identity': 'configured installed tag, not authenticated weights',
            'manga_snapshot': str(path), 'manga_artifact_sha256': {
                name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in required},
            'manga_runtime': {name: version(name) for name in ('manga-ocr', 'transformers', 'torch')}}, sort_keys=True)
        manga = MangaOCRAdapter(path)
        provider = OllamaRuntime(endpoint)
        try:
            with Image.open(BytesIO(original)) as source:
                quality_reread.refine(engine, source.convert('RGB'), original,
                    vision=lambda payload: provider.generate(model=model, payload=payload, timeout=120),
                    manga=manga, model_identity=identity, output=output)
        finally:
            stage = getattr(engine, 'quality_rereads', {})
            if stage.get('vision_calls', 0):
                try:
                    stage['reviewer_release'] = provider.unload(model=model, timeout=30)
                except Exception as error:
                    stage['reviewer_release'] = {'status': 'failed', 'error': type(error).__name__+': '+str(error)[:512]}
                    raise
            elif stage:
                stage['reviewer_release'] = {'status': 'not_dispatched_no_vision_calls'}
    return refine


def dialogue_refiner(model_path):
    """Construct an explicitly configured local reader; no automatic downloads."""
    import json
    path=Path(model_path).expanduser().resolve(strict=True)
    required=('config.json','preprocessor_config.json','pytorch_model.bin','vocab.txt',
        'special_tokens_map.json','tokenizer_config.json')
    if not path.is_dir() or any(not (path/name).is_file() for name in required):
        raise ValueError('dialogue reread requires a complete existing Manga OCR snapshot')
    identity=json.dumps({'path':str(path),'sha256':{name:hashlib.sha256((path/name).read_bytes()).hexdigest()
        for name in required}},sort_keys=True)
    def refine(engine,*,output):
        from io import BytesIO
        from PIL import Image
        from tralens.ocr import manga_dialogue
        from tralens.ocr.manga_adapter import MangaOCRAdapter
        import torch
        original=Path(engine.image_path).read_bytes()
        old_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            reader=MangaOCRAdapter(path)
            with Image.open(BytesIO(original)) as source:
                manga_dialogue.refine(engine,source.convert('RGB'),original,
                    reader=lambda crop:reader(crop,source_box_id='bound-dialogue',crop_bounds=(0,0,*crop.size)),
                    model_identity=identity,output=output)
        finally:torch.set_num_threads(old_threads)
    return refine


def crop_reader_refiner(model_path, *, ink_column_groups=False, ink_boundary_groups=False, reader_presentations=False):
    """Optional pinned local reader; OCR owns candidates, never translation/rendering."""
    if (type(ink_column_groups) is not bool or type(ink_boundary_groups) is not bool
            or (ink_column_groups and ink_boundary_groups)):
        raise ValueError('choose one explicit ink grouping policy')
    if type(reader_presentations) is not bool:
        raise ValueError('reader presentations requires explicit boolean enablement')
    from tralens.ocr.hayai_adapter import verify_snapshot
    path=verify_snapshot(model_path)
    def refine(engine,*,output):
        from io import BytesIO
        from PIL import Image
        from tralens.ocr import crop_readings
        from tralens.ocr.hayai_adapter import HayaiReader
        from tralens.translation.contracts import canonical
        if not engine.automatic_unit_proposals or engine.automatic_unit_proposals[0] is None:return
        original=Path(engine.image_path).read_bytes()
        reader=HayaiReader(path)
        try:
            with Image.open(BytesIO(original)) as source:
                image=source.convert('RGB')
                if reader_presentations and hasattr(engine,'raw_detections') and hasattr(engine,'ocr'):
                    from tralens.ocr import outlined_sounds
                    outlined_sounds.discover(engine,image,reader)
                if ink_column_groups or ink_boundary_groups:
                    from tralens.ocr import ink_groups
                    ink_groups.refine(engine,image,original,
                        policy=ink_groups.BOUNDARY_POLICY if ink_boundary_groups else ink_groups.POLICY)
                crop_readings.refine(engine,image,original,reader=reader,
                    model_identity=canonical(reader.identity),output=output)
                if reader_presentations:
                    from tralens.ocr.reader_presentations import collect
                    collect(engine,image,original,reader=reader,model_identity=canonical(reader.identity),output=output)
        finally:reader.close()
    return refine


@serialized
def run_pipeline(image, output, *, fonts, font_id=None, min_size=16, max_size=28,
                 max_units=None, background_method='auto', text_mask_model=None, render_policy='best-effort',
                 replacement_policy='all', min_ocr_confidence=None, layout_policy='source-local', quality_reread=False,
                 reread_model='qwen3.8:latest', manga_reread_model=None, source_localization=False,
                 visibility_policy='always-show', removal_policy='aggressive', reconstruction_policy='classical',
                  reconstruction_config=None, crop_reader_model=None, ink_column_groups=False, ink_boundary_groups=False,
                  reader_presentations=False, **preview_options):
    """Connect the existing OCR/translation preview to automatic rendering.

    Source/translation bindings stay strict; rendering quality follows the chosen
    coverage-first or conservative policy. A fresh job owns preview/ and rendering/ plus a
    pipeline.json checkpoint; failures retain evidence, never resume/overwrite.
    OCR factories/providers may be injected through the existing preview API.
    """
    if preview_options.get('run_local_ocr') is not True or preview_options.get('run_local_model') is not True:
        raise ValueError('the rendered pipeline requires explicit OCR and translation opt-ins')
    output = Path(output).expanduser().absolute()
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or any(path.is_symlink() for path in output.parents)):
        raise ValueError('pipeline output must be fresh with an existing parent and no symlinks')
    from tralens.rendering.automatic import preflight_automatic, run_automatic
    from tralens.translation.workflow import run_preview, _atomic_json
    from tralens.translation.coverage import validate_policy

    translation_policy = validate_policy(preview_options.get('translation_policy', 'coverage-first'))
    preview_options.setdefault('max_attempts',64)
    if type(source_localization) is not bool:
        raise ValueError('source localization requires explicit boolean enablement')
    if (crop_reader_model is not None and preview_options.get('model') != 'gpt-6-luna' and preview_options.get('profile')=='translategemma'
            and not preview_options.get('review_ocr')):
        preview_options.setdefault('translation_context','sound-effects')
    if preview_options.get('translation_context', 'isolated') != 'isolated':
        from tralens.translation.context import validate_policy as validate_context
        from tralens.translation.runtime import GENERALIST, TRANSLATEGEMMA
        validate_context(preview_options['translation_context'],
            GENERALIST if preview_options.get('profile') == 'generalist' else TRANSLATEGEMMA)
        if preview_options.get('review_ocr'):
            raise ValueError('nearby translation context and visual alternate review must be separate experiments')
    if type(quality_reread) is not bool:
        raise ValueError('quality reread requires explicit enablement and existing local model assets')
    if (type(ink_column_groups) is not bool or type(ink_boundary_groups) is not bool
            or ((ink_column_groups or ink_boundary_groups) and crop_reader_model is None)
            or (ink_column_groups and ink_boundary_groups)):
        raise ValueError('choose one ink grouping policy with the explicit local crop reader')
    if type(reader_presentations) is not bool or (reader_presentations and crop_reader_model is None):
        raise ValueError('reader presentations requires the explicit local crop reader')
    if crop_reader_model is not None:
        if quality_reread or manga_reread_model is not None or 'ocr_refine' in preview_options or translation_policy!='coverage-first':
            raise ValueError('crop reader requires coverage-first and one explicit refinement callback')
        preview_options['ocr_refine']=crop_reader_refiner(crop_reader_model,ink_column_groups=ink_column_groups,
            ink_boundary_groups=ink_boundary_groups,reader_presentations=reader_presentations)
    if quality_reread:
        if manga_reread_model is None or translation_policy != 'coverage-first':
            raise ValueError('quality reread requires explicit local Manga OCR snapshot and coverage-first translation')
        from tralens.translation.ocr_review import validate_options
        validate_options(True, reread_model, preview_options.get('model', 'translategemma:27b'), 8, 120, True)
        if 'ocr_refine' in preview_options:
            raise ValueError('choose the built-in quality refiner or an injected refinement callback')
        preview_options['ocr_refine'] = quality_refiner(endpoint=preview_options.get('endpoint', 'http://127.0.0.1:11434'),
            model=reread_model, manga_model=manga_reread_model)
    elif manga_reread_model is not None:
        if translation_policy!='coverage-first' or 'ocr_refine' in preview_options:
            raise ValueError('local dialogue reread requires coverage-first and one refinement callback')
        preview_options['ocr_refine']=dialogue_refiner(manga_reread_model)

    rendering_options = dict(image=Path(image).expanduser().absolute(), fonts=Path(fonts).expanduser().absolute(),
        font_id=font_id, min_size=min_size, max_size=max_size, max_units=max_units, background_method=background_method,
        text_mask_model=text_mask_model, render_policy=render_policy)
    if replacement_policy != 'all' or min_ocr_confidence is not None:
        rendering_options.update(replacement_policy=replacement_policy, min_ocr_confidence=min_ocr_confidence)
    if layout_policy != 'source-local':
        rendering_options['layout_policy'] = layout_policy
    if source_localization:
        rendering_options['cleanup_policy'] = 'localized'
    if visibility_policy != 'replacement-only':
        rendering_options['visibility_policy'] = visibility_policy
    if removal_policy!='legacy' or reconstruction_policy!='classical' or reconstruction_config is not None:
        rendering_options.update(removal_policy=removal_policy,reconstruction_policy=reconstruction_policy,
            reconstruction_config=reconstruction_config)
    checked = preflight_automatic(**rendering_options)
    report = {'schema_version': 1, 'purpose': 'automatic_pipeline_candidate', 'status': 'running',
        'stage': 'preview', 'source': checked['source'], 'fonts_sha256': checked['fonts_sha256'],
        'font_id': checked['font_id'], 'background_method': background_method,
        'render_policy': render_policy,
        'translation_policy': translation_policy,
        'approved_export': False, 'linguistic_approval': 'not_assessed',
        'full_page_translation_verified': False, 'preview': None,
        'rendering': {'status': 'not_started'}, 'coverage': None, 'error': None}
    if 'text_mask_model' in checked:
        report['text_mask_model'] = checked['text_mask_model']
    if 'replacement_policy' in checked:
        report['replacement_policy'] = checked['replacement_policy']
    if 'layout_policy' in checked:
        report['layout_policy'] = checked['layout_policy']
    if quality_reread:
        report['quality_reread'] = {'enabled': True, 'model': reread_model, 'truth_approval': False}
    if crop_reader_model is not None:
        report['crop_reader'] = {'enabled': True, 'source_accuracy_verified': False}
    if reader_presentations:
        report['reader_presentations'] = {'enabled': True, 'selected': False, 'erase_permission': False}
    if ink_column_groups:
        report['ink_column_groups'] = {'enabled': True, 'full_utterance_verified': False}
    if ink_boundary_groups:
        report['ink_boundary_groups'] = {'enabled': True, 'full_utterance_verified': False}
    if source_localization:
        report['source_localization'] = {'enabled': True, 'semantic_ownership_verified': False}
    if visibility_policy != 'replacement-only':
        report['visibility_policy'] = visibility_policy
    for key in ('removal_policy','reconstruction_policy','reconstruction_config'):
        if key in checked:report[key]=checked[key]
    output.mkdir(mode=0o700)
    checkpoint = output/'pipeline.json'
    _atomic_json(checkpoint, report)

    def persist():
        _atomic_json(checkpoint, report, replace_owned=True)

    def fingerprint(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        preview = run_preview(rendering_options['image'], output/'preview', **preview_options)
        records = preview['units']
        statuses = Counter(row['status'] for row in records)
        strict_count = sum(row['status'] == 'preview' and row.get('input_kind', 'strict_utterance') == 'strict_utterance'
                           for row in records)
        coverage_count = sum(row['status'] == 'preview' and row.get('input_kind') == 'coverage_text_region' for row in records)
        report['preview'] = {'status': preview['status'], 'path': 'preview/preview.json',
            'sha256': fingerprint(output/'preview/preview.json'), 'attempts': preview['attempts'],
            'unit_status_counts': dict(statuses), 'error': preview['error'],
            'ocr_cleanup': preview['cleanup'], 'fragment_previews': preview.get('fragment_preview_count', 0),
            'visual_alternates': preview.get('alternate_preview_count', 0)}
        report['coverage'] = {'saved_units': len(records), 'strict_translations': strict_count,
            'coverage_translations': coverage_count, 'total_translations': strict_count+coverage_count,
            'source_uncertain_units': sum(row.get('input_kind') == 'coverage_text_region' for row in records),
            'ocr_blocked_units': statuses['blocked'],
            'ocr_blocking_reasons': dict(Counter(reason for row in records for reason in row['blocking_reasons'])),
            'translation_budget_exhausted': statuses['budget_exhausted'], 'rendered': 0,
            'render_skip_reasons': {}, 'partial_output': True}
        if preview['status'] in ('failed', 'interrupted'):
            report.update(status=preview['status'], error=preview['error'])
            report['rendering'] = {'status': 'skipped', 'reason': 'upstream_failed_or_interrupted'}
        else:
            # Check the original preflight binding again, not just the current
            # preview binding. A source/font change cannot become a new job input.
            if preflight_automatic(**rendering_options) != checked or preview['input'] != checked['source']:
                raise ValueError('pipeline image/font inputs changed after preflight')
            if not strict_count and not (coverage_count and (render_policy == 'best-effort' or visibility_policy == 'always-show')):
                report['status'] = 'partial'
                report['rendering'] = {'status': 'skipped', 'reason': 'coverage_requires_best_effort' if coverage_count else 'no_strict_translations',
                    'note': 'No composite was produced; original image and preview evidence remain unchanged.'}
            else:
                report['stage'] = 'rendering'
                report['rendering'] = {'status': 'running'}
                persist()
                render_preview = output/'preview/preview.json'
                render_options = dict(rendering_options)
                if source_localization:
                    native_options = {**rendering_options, 'cleanup_policy': 'native', 'visibility_policy':'replacement-only',
                        'removal_policy':'preserve','reconstruction_policy':'classical','reconstruction_config':None}
                    started = time.monotonic()
                    native = run_automatic(preview=render_preview, output=output/'rendering-native',
                                           plan_localization=True, **native_options)
                    plan = native['localization_request_plan']
                    report['source_localization'].update(native_seconds=time.monotonic()-started,
                        planning_seconds=native['localization_planning_seconds'],
                        native_manifest='rendering-native/manifest.json', requested=len(plan['requests']),
                        budget_missed=sum('budget' in r['reason'] for r in plan['records']))
                    persist()
                    if preflight_automatic(**rendering_options) != checked:
                        raise ValueError('pipeline image/font inputs changed before crop inference')
                    try:
                        derived, seconds = crop_localization_pass(preview, plan, image=rendering_options['image'],
                            output=output/'localization', device=preview_options['device'],
                            model_root=preview_options['model_root'], ocr_factory=preview_options.get('ocr_factory'))
                        stage = derived['ocr']['source_localization']
                        report['source_localization'].update(calls=stage['calls'], execution_seconds=seconds,
                            predicted=sum(r['status'] == 'predicted' for r in stage['records']),
                            failed=sum(r['status'] == 'failed' for r in stage['records']))
                        render_preview = output/'localization/preview.json'
                    except Exception as error:
                        # Keep native successes and originals, but do not present
                        # failed optional inference as a completed quality pass.
                        report['source_localization']['error'] = type(error).__name__+': '+str(error)[:2000]
                        render_options['cleanup_policy'] = 'native'
                    if preflight_automatic(**rendering_options) != checked:
                        raise ValueError('pipeline image/font inputs changed during crop inference')
                render_preview_sha256 = fingerprint(render_preview)
                started = time.monotonic()
                rendered = run_automatic(preview=render_preview, output=output/'rendering', **render_options)
                if source_localization:
                    report['source_localization'].update(retry_render_seconds=time.monotonic()-started,
                        predictions_consumed=rendered.get('localized_cleanup_summary', {}).get('retries', 0),
                        rendered_preview=str(render_preview.relative_to(output)),
                        rendered_preview_sha256=render_preview_sha256)
                if (rendered['source'] != checked['source'] or rendered['fonts_sha256'] != checked['fonts_sha256']
                        or rendered['preview_sha256'] != render_preview_sha256
                        or fingerprint(render_preview) != render_preview_sha256
                        or fingerprint(output/'preview/preview.json') != report['preview']['sha256']):
                    raise ValueError('rendered output does not bind the original pipeline inputs')
                report['rendering'] = {'status': 'complete', 'manifest': 'rendering/manifest.json',
                    'manifest_sha256': fingerprint(output/'rendering/manifest.json'),
                    'composite': 'rendering/composite.png', 'review': 'rendering/review.html',
                    'coverage': rendered['coverage'], 'background_summary': rendered['background_summary'],
                     'quality_summary': rendered['quality_summary']}
                if 'replacement_confidence' in rendered:
                    report['rendering']['replacement_confidence'] = rendered['replacement_confidence']
                if 'display_summary' in rendered:
                    report['display_summary'] = rendered['display_summary']
                    report['rendering']['display_summary'] = rendered['display_summary']
                    report['coverage']['displayed_translations'] = rendered['coverage']['displayed_translations']
                report['coverage'].update(rendered=rendered['coverage']['rendered'],
                    strict_rendered=rendered['coverage']['strict_rendered'],
                    coverage_rendered=rendered['coverage']['coverage_rendered'],
                    render_skip_reasons=dict(Counter(row['reason_code'] for row in rendered['units']
                        if row['machine_selected'] and row['status'] == 'skipped')),
                    partial_output=rendered['coverage']['partial_output'])
                report['status'] = ('complete' if preview['status'] == 'complete'
                    and not rendered['coverage']['partial_output'] else 'partial')
                if report.get('source_localization', {}).get('error') or report.get('source_localization', {}).get('failed'):
                    report['status'] = 'partial'
        report['stage'] = 'finished'
    except (Exception, KeyboardInterrupt) as error:
        failed_stage = report['stage']
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            error={'stage': failed_stage, 'type': type(error).__name__, 'message': str(error)[:2000]})
        if failed_stage == 'rendering':
            report['rendering'] = {'status': report['status']}
        else:
            report['rendering'] = {'status': 'skipped', 'reason': 'upstream_failed_or_invalid'}
    persist()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create an OCR/translation preview; add --render for an automatic candidate page.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="fresh output directory; parent must exist")
    parser.add_argument("--device", required=True, help="cpu or one gpu:N")
    parser.add_argument("--model-root", required=True, type=Path, help="existing local medium Paddle exports")
    parser.add_argument("--run-local-ocr", action="store_true")
    parser.add_argument("--run-local-model", action="store_true")
    parser.add_argument("--allow-fragment-previews", action="store_true",
                        help="opt in to review-required raw singleton text fragments; original structural blocks remain")
    parser.add_argument("--review-ocr", action="store_true",help="optional local visual reread proposals before translation; requires model opt-in")
    parser.add_argument("--review-model",default="qwen3.8:latest")
    parser.add_argument("--max-review-attempts",type=int,default=4,help="independent review-generation budget, 0..20")
    parser.add_argument("--review-timeout",type=float,default=90,help="review timeout, at most 120 seconds")
    parser.add_argument("--model", default="translategemma:27b", help="installed model (default: translategemma:27b); no download/fallback")
    parser.add_argument("--luna-mode", choices=("page","isolated","panel","atlas","reviewed"), default=None)
    parser.add_argument("--related-page", type=Path, action="append", default=[], help="explicit same-manga context image for Luna, repeatable")
    parser.add_argument("--profile", choices=("generalist", "translategemma"), default="translategemma",
                        help="capability profile (default: translategemma); Qwen overrides require generalist")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, default=64, help="global model call budget, 0..64, including single-rendition retries")
    parser.add_argument('--translation-policy', choices=('coverage-first', 'conservative'), default='coverage-first',
                         help='default coverage-first: attempt source-bound OCR text despite uncertain ownership/crop/orientation')
    parser.add_argument('--translation-context', choices=('isolated', 'nearby','sound-effects'), default=None,
                         help='use page dialogue for specialist sound effects (automatic with the crop reader), isolated text, or nearby context for a generalist')
    parser.add_argument('--source-localization', action='store_true',
                          help='opt-in bounded crop predictions using existing OCR; with render, retry failed cleanup as unverified whole strokes')
    parser.add_argument('--visibility-policy', choices=('replacement-only','always-show'),default=None,
                        help='render: always-show prioritizes visible translations; opaque patches may cover artwork, then nearby captions')
    parser.add_argument('--removal-policy',choices=('legacy','preserve','aggressive'),default=None)
    parser.add_argument('--reconstruction-policy',choices=('classical','lama'),default='classical')
    parser.add_argument('--reconstruction-config',type=Path,help='explicit approved local model/runtime JSON; never downloaded')
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--keep-alive", type=int, default=60)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--unit-id", action="append", default=[], help="optional selected proposal ID; repeat to set call order")
    parser.add_argument('--render', action='store_true', help='connect OCR/translation to automatic cleanup and rendering')
    parser.add_argument('--fonts', type=Path, help='explicit local licensed font manifest; required with --render')
    parser.add_argument('--font-id', help='global font ID; required if the manifest has multiple faces')
    parser.add_argument('--min-size', type=int, help='preferred rendering minimum, default 16; conservative mode enforces it strictly')
    parser.add_argument('--max-size', type=int, help='rendering maximum size, default 28')
    parser.add_argument('--max-units', type=int, help='optional rendering evaluation cap, 1..32; not a translation budget')
    parser.add_argument('--background-method', choices=('auto', 'flat'),
                        help='render: flat fill then classical restoration (default auto); flat disables restoration')
    parser.add_argument('--text-mask-model', type=Path,
                        help='optional acquired CTD ONNX file: automatic native-mask/periodic fallback; no download')
    parser.add_argument('--render-policy', choices=('best-effort', 'conservative'),
                         help='default best-effort: attempt risk-labelled fallback replacements; conservative leaves uncertain regions unchanged')
    parser.add_argument('--replacement-policy', choices=('all', 'confidence-gated'),
                        help='render: optional experimental OCR/native-cleanup confidence gate; translation coverage is unchanged')
    parser.add_argument('--min-ocr-confidence', type=float,
                         help='confidence-gated render: weakest raw main-line score 0..1; provisional default 0.70, not an accuracy probability')
    parser.add_argument('--layout-policy', choices=('source-local', 'enclosed-retry'),
                         help='render: opt-in closed original-space layout retry; unchanged native cleanup and minimum font size')
    parser.add_argument('--quality-reread', action='store_true',
                        help='render: opt-in enclosed terminal grouping and paired visual/line OCR rereads; original readings retained')
    parser.add_argument('--reread-model', default='qwen3.8:latest', help='existing local vision model for quality rereads; never downloaded')
    parser.add_argument('--manga-reread-model', type=Path, help='existing local Manga OCR snapshot; enables isolated low-score dialogue rereading, or paired visual rereads with --quality-reread')
    parser.add_argument('--crop-reader-model', type=Path, help='existing pinned local Hayai snapshot; opt-in three-view source hypotheses before translation')
    parser.add_argument('--reader-presentations', action='store_true', help='with --crop-reader-model, retain unselected source-bound projective readings of every detector polygon')
    parser.add_argument('--ink-column-groups', action='store_true', help='with --crop-reader-model, join unambiguous original-ink column continuations')
    parser.add_argument('--ink-boundary-groups', action='store_true', help='with --crop-reader-model, use join-boundary alignment for staggered column continuations; alternative to --ink-column-groups')
    args = parser.parse_args(argv)
    if not args.run_local_ocr:
        parser.error("image processing requires explicit --run-local-ocr; no engine was initialized")
    if args.review_ocr and not args.run_local_model:
        parser.error("--review-ocr requires --run-local-model; OCR-only performs no image review")
    if args.render and (not args.run_local_model or args.fonts is None):
        parser.error('--render requires --run-local-model and --fonts; no engine was initialized')
    if not args.render and (args.quality_reread or args.manga_reread_model is not None or args.crop_reader_model is not None or args.ink_column_groups or args.ink_boundary_groups or args.reader_presentations):
        parser.error('quality reread options require --render')
    if not args.render and args.visibility_policy not in (None,'replacement-only'):
        parser.error('--visibility-policy requires --render')
    if not args.render and (args.removal_policy not in (None,'legacy') or args.reconstruction_policy!='classical' or args.reconstruction_config):
        parser.error('removal/reconstruction options require --render')
    if not args.render and any(getattr(args, name) is not None for name in ('fonts', 'font_id', 'min_size', 'max_size', 'max_units', 'background_method', 'text_mask_model', 'render_policy', 'replacement_policy', 'min_ocr_confidence', 'layout_policy')):
        parser.error('font/rendering options require --render')
    # Keep help/import and missing opt-in paths free of OCR/provider imports.
    from tralens.translation.workflow import run_preview
    try:
        preview_options = dict(device=args.device, model_root=args.model_root,
            run_local_ocr=args.run_local_ocr, run_local_model=args.run_local_model, model=args.model,
            profile=args.profile, endpoint=args.endpoint, max_attempts=args.max_attempts, timeout=args.timeout,
            keep_alive=args.keep_alive, num_predict=args.num_predict, selected_ids=tuple(args.unit_id),
            allow_fragment_previews=args.allow_fragment_previews,review_ocr=args.review_ocr,review_model=args.review_model,
            max_review_attempts=args.max_review_attempts,review_timeout=args.review_timeout)
        if args.model == "gpt-6-luna" or args.related_page or args.luna_mode:
            preview_options.update(luna_mode=args.luna_mode or "reviewed", related_pages=args.related_page)
        preview_options['translation_policy'] = args.translation_policy
        if args.translation_context is not None:
            preview_options['translation_context'] = args.translation_context
        if args.source_localization:
            preview_options['source_localization'] = True
        if args.render:
            report = run_pipeline(args.image, args.output, fonts=args.fonts, font_id=args.font_id,
                min_size=16 if args.min_size is None else args.min_size,
                max_size=28 if args.max_size is None else args.max_size, max_units=args.max_units,
                background_method=args.background_method or 'auto', text_mask_model=args.text_mask_model,
                render_policy=args.render_policy or 'best-effort', replacement_policy=args.replacement_policy or 'all',
                min_ocr_confidence=args.min_ocr_confidence, layout_policy=args.layout_policy or 'source-local', quality_reread=args.quality_reread,
                reread_model=args.reread_model, manga_reread_model=args.manga_reread_model, crop_reader_model=args.crop_reader_model,
                reader_presentations=args.reader_presentations,
                ink_column_groups=args.ink_column_groups,ink_boundary_groups=args.ink_boundary_groups,
                visibility_policy=args.visibility_policy or 'always-show',removal_policy=args.removal_policy or 'aggressive',
                reconstruction_policy=args.reconstruction_policy,reconstruction_config=args.reconstruction_config, **preview_options)
            coverage = report['coverage'] or {}
            print(f"{report['status']}: {args.output / 'pipeline.json'}; "
                  f"strict translations={coverage.get('strict_translations', 0)}; rendered={coverage.get('rendered', 0)}; "
                  f"uncertain translations={coverage.get('coverage_translations', 0)}; "
                  'candidate only, not verified full-page translation')
            if report['rendering'].get('composite'):
                print(f"Composite: {args.output / report['rendering']['composite']}; review: {args.output / report['rendering']['review']}")
            if report['error']:
                print(f"Pipeline error: {report['error']['message']}")
            return 130 if report['status'] == 'interrupted' else 0 if report['status'] == 'complete' else 1
        preview = run_preview(args.image, args.output, **preview_options)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, ("pipeline" if args.render else "preview") + " rejected: " + str(error) + "\n")
    print(f"{preview['status']}: {args.output / 'preview.json'}; model calls={preview['attempts']}; "
          f"strict previews={preview.get('strict_preview_count',0)}; uncertain regions={preview.get('coverage_preview_count',0)}; "
          f"unverified fragments={preview.get('fragment_preview_count',0)}")
    if args.review_ocr:
        stage = preview.get("ocr_review",{})
        print(f"visual review calls={stage.get('attempts',0)}; release/control calls={stage.get('control_calls',0)}; "
              f"unverified alternates={preview.get('alternate_preview_count',0)}")
    return 0 if preview["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
