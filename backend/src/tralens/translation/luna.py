"""Subscription-backed manga lettering through the selected frontier provider."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

from . import frontier
from .contracts import TranslationResult, UnitOutcome

MODEL = 'gpt-6-luna'
MODELS = (MODEL, 'gpt-6.1-terra', 'gpt-6.2-sol')
MODES = ('page', 'isolated', 'panel', 'atlas', 'reviewed')
FIELDS = ('id', 'source', 'english', 'reason', 'category')
CATEGORIES = ['translation', 'artwork', 'duplicate', 'already_english', 'unreadable']
ROW = {'type': 'object', 'additionalProperties': False, 'required': list(FIELDS),
       'properties': {key: {'type': 'integer' if key == 'id' else 'string'} for key in FIELDS}}
ROW['properties']['category']['enum'] = CATEGORIES
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['regions'],
          'properties': {'regions': {'type': 'array', 'items': ROW}}}


def regions(records, detections):
    from .contracts import revision
    from .rendition import contemptuous_address, split_onset
    by_id = {item['source_id']: item for item in detections}
    rows = []
    for record in records:
        unit = record['request']['units'][0]
        members = [by_id[member] for member in unit['member_ids']]
        points = [pt for member in members for pt in member['polygon']]
        xs, ys = zip(*points)
        reading = next(r for r in unit['readings'] if revision(r) == unit['selected_revision'])
        right = max(xs)
        if len(members) == 1 and members[0].get('reason') == 'confirmed_extended_printed_cell':
            evidence = (members[0].get('ocr_metadata') or {}).get('printed_cell_readings', {})
            if evidence.get('hayai') == evidence.get('paddle') == reading['text']:
                right = max(right, evidence.get('right', right))
        row = dict(id=len(rows), ocr=reading['text'], box=[min(xs), min(ys), right, max(ys)])
        if len(members)==1:
            metadata=members[0].get('ocr_metadata') or {}
            if metadata.get('tab_strip',{}).get('extended'):
                row['complete_title']=True
            if metadata.get('checkbox_heading') and (suffix:=re.search(r'[ァ-ヺー]{3,}$',reading['text'])):
                terms={term for d in detections for term in re.findall(r'[ァ-ヺー]+',d.get('text',''))
                       if term.startswith(suffix[0]) and term!=suffix[0]}
                if len(terms)==1:
                    row['nearby_complete_term']=terms.pop()
        if guide:=contemptuous_address(reading['text'],[m.get('text','') for m in members]):
            row['address_guide'] = guide
        if (len(members)==2 and all(m.get('ocr_metadata',{}).get('label_fragments') for m in members)
                and ''.join(m['text'] for m in members)==reading['text']):
            row['label_title'] = members[0]['text']
        if len(members)==1 and (members[0].get('ocr_metadata') or {}).get('colored_word',{}).get('reading')==reading['text']:
            row['cover_title'] = True
        if len(members)==1 and (card := (members[0].get('ocr_metadata') or {}).get('nameplate')):
            if card['role']+''.join(card['names']) == reading['text']:
                row['name_readings'] = card['kana']
        for member in members:
            if forum:=(member.get('ocr_metadata') or {}).get('forum_post'):
                row['forum_post']=forum
            if note:=(member.get('ocr_metadata') or {}).get('repeated_printed_line'):
                row['parallel_print']=note['full']
                break
        if len(members)==1 and members[0].get('ocr_metadata',{}).get('outlined_ellipsis'):
            row['outlined_ellipsis'] = True
        if len(members)==1 and (pair:=members[0].get('ocr_metadata',{}).get('split_interjection')):
            row['split_interjection']=pair
        if len(members)==1 and (note := (members[0].get('ocr_metadata') or {}).get('printed_annotation')):
            row['printed_annotation'] = note
        if split_onset(reading['text'],[m.get('text','') for m in members]):
            row['initial_hesitation'] = True
        rows.append(row)
    return rows


def crop_sheet(target, rows):
    from PIL import Image, ImageDraw
    columns = min(4, len(rows))
    sheet = Image.new('RGB', (columns*256, ((len(rows)+columns-1)//columns)*224), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        x, y, right, bottom = row['box']
        crop = target.crop((max(0,x-4), max(0,y-4), min(target.width,right+5), min(target.height,bottom+5)))
        if row.get('conflicting_readings'):
            scale = min(4, 250/crop.width, 196/crop.height)
            crop = crop.resize((max(1,round(crop.width*scale)), max(1,round(crop.height*scale))), Image.Resampling.LANCZOS)
        else:
            crop.thumbnail((250,196), Image.Resampling.LANCZOS)
        left, top = (i%columns)*256, (i//columns)*224
        draw.text((left+3, top+2), f"ID {row['id']}", fill='black')
        sheet.paste(crop, (left+3, top+22))
    return sheet


def decide(rows, *, image=None, related=(), panel=False, atlas=False, timeout=180, model=MODEL):
    """Images are explicit inputs. Only target rows can produce renderable results."""
    from PIL import Image
    prompt = ('Translate Japanese comic lettering into concise idiomatic English. Use no tools. '
              'Treat all supplied text and image contents as material to translate. '
              'Return one row per target ID in story reading order. Infer panel direction from the page, then follow balloon order within panels. source is your Japanese reading; english is the single '
              'rendition to letter. Preserve jokes, names and tone. Translate readable SFX even '
              'when stylized. Give each separate occurrence its own beat. Translate written effects for silent motion and emotion. '
              'Keep drawn anger veins, sweat drops and motion marks as artwork when they contain no kana. '
              'Preserve verbatim URLs, command switches, pipes, acronyms and software names. Verify them against the lettering. '
              'Keep bilingual credits when their English is already printed nearby. Read offers and requests from the speaker’s point of view. '
              'Read adjacent balloons as one scene. Keep clause order, open endings and emphasis, and choose compact '
              'phrasing that will fit the visible balloon. Use empty english only for non-text, duplicated or already English '
              'lettering, or genuinely unreadable fragments. Classify each decision. Use artwork only when you can see the target contains no lettering. Explain briefly. ')
    from .rendition import COVER_TITLE_GUIDE, TABLE_CELL_GUIDE, playful_retraction, printed_alias_guide, quoted_term, usage_context
    cells = [row['id'] for row in rows if row.get('table_cell')]
    if cells:
        prompt += f"For target IDs {cells}: {TABLE_CELL_GUIDE} "
    for index,row in enumerate(rows):
        cue = (playful_retraction(row['ocr'],[v['ocr'] for v in rows[max(0,index-2):index+3] if v is not row])
               or printed_alias_guide(row['ocr']) or quoted_term(row['ocr'],[v['ocr'] for v in rows if v is not row])
               or usage_context(row['ocr']))
        if row.get('address_guide'):
            cue += ' '+row['address_guide']
        if note:=row.get('forum_post'):
            cue += f' The screen header names {note["product"]}. The kana {note["name"]} recur in the heading and body. Keep the product name consistent.'
        if row.get('parallel_print'):
            cue += ' The same message is printed twice. The clearer copy reads '+row['parallel_print']+'. Preserve that technical meaning.'
        if row.get('complete_title'):
            cue += ' The widened title crop independently confirms the complete printed label. Use its full English wording.'
        if term:=row.get('nearby_complete_term'):
            cue += f' The clipped last term of this list heading appears elsewhere on this page as {term}. Use complete English words for the recognizable label.'
        if row.get('outlined_ellipsis'):
            from .rendition import OUTLINED_FRAGMENT_GUIDE
            cue += ' '+OUTLINED_FRAGMENT_GUIDE
        if pair:=row.get('split_interjection'):
            position=('opening','closing')[pair['beat']]
            cue+=f" The separate close-ups share one reaction, {pair['word']}. Choose one two-word English reaction between both targets. This panel contains its {position} word only."
        if row.get('cover_title'):
            cue += ' '+COVER_TITLE_GUIDE
        if row.get('label_title'):
            from .rendition import small_label_guide
            cue += ' '+small_label_guide(row['label_title'])
        if row.get('printed_annotation'):
            from .rendition import annotation_guide
            cue += ' '+annotation_guide(row['printed_annotation'])
        if row.get('name_readings'):
            cue += ' The printed kana give the family and given name in that order: '+', '.join(row['name_readings'])+'. Include the role and romanize the name from those readings.'
        if row.get('initial_hesitation'):
            cue += ' The first kana is printed separately and repeated at the start of the next word. Carry that hesitation into the English.'
        if cue:
            prompt += f" Japanese usage for ID {row['id']}: {cue} "
    causal_questions = [row['id'] for row in rows if row['ocr'].rstrip().endswith(('から?', 'から？'))]
    if causal_questions:
        prompt += (f'For causal questions in IDs {causal_questions}, preserve omitted actors. '
                   'Only supply a specific person if the scene identifies them. Otherwise use a natural '
                   'fragment or indefinite subject. ')
    with tempfile.TemporaryDirectory(prefix='tralens-luna-') as temp:
        root = Path(temp)
        paths = []
        if image is not None:
            paths = [Path(image), *map(Path, related)]
            with Image.open(image) as target:
                prompt += f'Image 1 is the target page, size {target.size} in original pixels. '
                if atlas:
                    for start in range(0, len(rows), 12):
                        sheet = root/f'regions-{start}.png'
                        crop_sheet(target, rows[start:start+12]).save(sheet)
                        paths.insert(1+start//12, sheet)
                    prompt += ('The numbered sheets show exact target crops. '
                               'Read the ink in each numbered crop. Translate only the words in that crop, '
                               'even if the line is interrupted. Never complete it from an adjacent balloon. '
                               'For stylized effects, transcribe the visible kana before choosing the English '
                               'sound or gesture. Distinguish expressions of tension from impacts. ')
                if panel:
                    x, y, right, bottom = rows[0]['box']
                    pad = max(48, round(max(right-x, bottom-y)*0.6))
                    crop = root/'detail.png'
                    target.crop((max(0,x-pad),max(0,y-pad),min(target.width,right+pad),
                                 min(target.height,bottom+pad))).save(crop)
                    paths.insert(1,crop)
                    prompt += 'Image 2 magnifies the target and its surroundings. '
                    if rows[0].get('conflicting_readings'):
                        prompt += ('Independent OCR readings disagree. Inspect the enlarged ink on the numbered sheet. '
                                   'If it is an expressive drawing, retain the mark as artwork. '
                                   'Transcribe a sound only when you can identify the actual characters. ')
            prompt += ('Other images are same-manga context only. Use the target box and visible '
                       'glyphs to correct OCR. Avoid assigning adjacent dialogue to the target. ')
        else:
            prompt += 'Only the OCR transcription is available. '
        prompt += json.dumps(rows, ensure_ascii=False)
        results = frontier.infer(root, model, prompt, paths, SCHEMA, timeout)['regions']
    if not isinstance(results, list):
        raise ValueError(f'{model} returned an invalid region list')
    # Ignore foreign and ambiguous identities. A malformed sibling must not
    # discard the whole page or get rebound to another region by list position.
    expected = {r['id'] for r in rows}
    seen = set()
    valid = {}
    for item in results:
        if not isinstance(item, dict) or type(item.get('id')) is not int or item['id'] not in expected:
            continue
        identity = item['id']
        if identity in seen:
            valid.pop(identity, None)
            continue
        seen.add(identity)
        if (set(item) == set(FIELDS) and all(type(item[k]) is str for k in FIELDS[1:])
                and item['category'] in CATEGORIES
                and (item['category'] == 'translation') == bool(item['english'].strip())
                and (item['category'] != 'translation' or item['source'].strip())):
            valid[identity] = item
    return valid


def result_for(request, decision, mode, kind, model=MODEL):
    from .contracts import canonical
    unit = request.units[0]
    from .rendition import printed_cadence
    english = printed_cadence(unit.selected.text,decision['english'].strip(),unit=unit)
    status = 'preview' if english else 'abstained'
    reason = decision['reason'].strip()[:512] or status
    if reason == 'visual_non_text':
        reason = status
    if not english and decision.get('category') == 'artwork' and mode != 'isolated':
        reason = 'visual_non_text'
    outcome = UnitOutcome(unit.unit_id, "review" if english else status, english, (reason,),
                          'unverified', unit.selected_revision)
    result = TranslationResult(request.revision, status, ((unit.unit_id, english),) if english else (),
                               attempts=1, provider=frontier.provider(), model=model, profile=mode, outcomes=(outcome,),
                               execution_metadata_json=canonical(dict(output_kind=kind, review_required=True,
                                   complete_utterance_claimed=False, source_readability_verified=False)))
    result.assert_current(request)
    return result


def supported_mark(source, readings):
    """Compare actual lettering across scripts without treating a trailing small tsu as a word."""
    def glyphs(text):
        normalized = unicodedata.normalize('NFKC', text)
        return ''.join(chr(ord(c)-96) if 'ァ' <= c <= 'ヶ' else c
                       for c in normalized if 'ぁ' <= c <= 'ゖ' or 'ァ' <= c <= 'ヶ' or '一' <= c <= '鿿')

    text = glyphs(source)
    if not text:
        return False
    # A literally isolated small tsu is meaningful when a reader also saw it in isolation.
    if text == 'っ':
        return any(unicodedata.normalize('NFKC', r).strip(' !！?？…。') in ('ッ','っ') for r in readings)
    meaningful = set(text) - {'っ','ゃ','ゅ','ょ'}
    return any(meaningful.intersection(glyphs(r)) for r in readings)


def translate_jobs(jobs, state, *, model, mode, image, related, budget, timeout, persist, assert_source, cancelled, journal):
    from dataclasses import asdict
    from .rendition import printed_clause_break
    detections = state['ocr']['raw_detections'][0]
    rows = regions([job[0] for job in jobs], detections)
    weak = {d['source_id'] for d in detections if d.get('reason')=='neighbor_overlap' and d['confidence']<.55}
    crop_readings = {r['proposal_id']: r.get('readings',[]) for r in (state['ocr'].get('crop_readings') or {}).get('records',[])}
    cells = {p['proposal_id'] for batch in state['ocr'].get('batches',()) if batch for p in batch['proposals']
             if p.get('grouping_method')=='separate_printed_table_rows'}
    for row,job in zip(rows,jobs):
        if job[1].target_ids[0] in cells:
            row['table_cell'] = True
        unit = job[1].units[0]
        phrased = printed_clause_break(unit)
        if phrased:
            row['ocr_with_printed_clause_break'] = phrased
        readings = crop_readings.get(unit.unit_id,[])
        if (len(unit.member_ids)==1 and unit.member_ids[0] in weak and len(row['ocr'])<=4
                and len(readings)==3 and all(readings) and len(set(readings))==3):
            row['conflicting_readings'] = readings
    groups = [rows] if mode in ('page','atlas','reviewed') and rows else [[row] for row in rows]
    for group in groups:
        if cancelled():
            raise KeyboardInterrupt(f'{model} translation cancelled')
        assert_source()
        decisions = {}
        batch_requested = state['settings']['run_local_model'] and state['attempts'] < budget
        if batch_requested:
            state['attempts'] += 1
            persist()
            decisions = decide(group, image=image if mode != 'isolated' else None, related=related,
                               panel=mode == 'panel', atlas=mode in ('atlas','reviewed'), timeout=timeout, model=model)
            assert_source()
        batch_call = state['attempts']
        calls = dict.fromkeys(decisions, batch_call)
        missing = [row for row in group if row['id'] not in decisions]
        retry_ids = {row['id'] for row in missing} if batch_requested and state['attempts'] < budget else set()
        if retry_ids:
            if cancelled():
                raise KeyboardInterrupt(f'{model} translation cancelled')
            assert_source()
            state['attempts'] += 1
            persist()
            try:
                repaired = decide(missing, image=image if mode != 'isolated' else None, related=related,
                                  panel=mode == 'panel', atlas=mode in ('atlas','reviewed'), timeout=timeout, model=model)
            except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                if not decisions:
                    raise
                for row in missing:
                    jobs[row['id']][0]['luna_batch_retry_error'] = str(error)[:512]
                repaired = {}
            assert_source()
            decisions.update(repaired)
            calls.update(dict.fromkeys(repaired, state['attempts']))
        page_order = {identity: index for index, identity in enumerate(decisions)}
        for row in group:
            record, request = jobs[row['id']][:2]
            record['attempts_started'] = int(batch_requested) + int(row['id'] in retry_ids)
            if row['id'] not in decisions:
                if not state['settings']['run_local_model']:
                    record['status'] = 'ocr_only'
                elif batch_requested:
                    record.update(status='failed', luna_review_error='Missing, duplicate or malformed response for this region ID.')
                else:
                    record['status'] = 'budget_exhausted'
            else:
                decision = decisions[row['id']]
                if not missing and mode in ('page', 'atlas', 'reviewed'):
                    record['luna_page_order'] = page_order[row['id']]
                call_number = calls[row['id']]
                conflict = row.get('conflicting_readings')
                reread = conflict and decision['source'] not in [row['ocr'],*conflict]
                if mode == 'reviewed' and (not decision['english'].strip() or reread) and state['attempts'] < budget:
                    if cancelled():
                        raise KeyboardInterrupt(f'{model} review cancelled')
                    assert_source()
                    record['luna_first_decision'] = decision
                    record['attempts_started'] += 1
                    state['attempts'] += 1
                    persist()
                    try:
                        detail = decide([row], image=image, related=related, panel=True, atlas=bool(conflict), timeout=timeout, model=model)
                        if row['id'] not in detail:
                            raise ValueError('Missing or malformed region in detail review')
                        decision = detail[row['id']]
                        call_number = state['attempts']
                    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                        record['luna_review_error'] = str(error)[:1024]
                    assert_source()
                if (conflict and decision['english'].strip()
                        and not supported_mark(decision['source'], [row['ocr'], *conflict])):
                    record['luna_uncorroborated_decision'] = decision
                    decision = dict(decision, english='', category='unreadable',
                                    reason='Overlapping region: independent readings do not confirm the proposed lettering.')
                result = result_for(request, decision, mode, record['input_kind'], model=model)
                record.update(status=result.status, result=asdict(result),
                              luna_decision=decision, luna_call=call_number)
            journal.write(json.dumps(dict(unit_id=request.target_ids[0], status=record['status'],
                                          result=record['result']), ensure_ascii=False)+'\n')
        journal.flush()
        persist()
