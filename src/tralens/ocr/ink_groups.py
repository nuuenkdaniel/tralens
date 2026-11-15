"""Opt-in column continuations measured on original ink, not detector margins.

The saved partition, alternatives, component evidence and original seams remain
available. Grouping is an unverified hypothesis; it grants no erase permission.
"""
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np

from .automatic_units import crop_conflicts
from .quality_lines import OriginalLines
from .quality_reread import enclosure_map
from .recognition_units import image_fingerprint
from ..translation.contracts import automatic_units, canonical, revision, strict_json


POLICY = 'original_ink_column_continuations_v1'
BOUNDARY_POLICY = 'original_ink_boundary_continuations_v1'
POLICIES = (POLICY, BOUNDARY_POLICY)
REASON = 'automatic_vertical_continuation_hypothesis'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(value):
    return strict_json(canonical(asdict(value) if is_dataclass(value) else value))


def plan(batch, image, *, policy=POLICY):
    """One disjoint pair per continuation; strings/scores never choose a join."""
    require(policy in POLICIES, 'unknown ink grouping policy')
    batch = snapshot(batch)
    automatic_units(batch)  # Includes duplicate membership and raw evidence checks.
    require(image.mode == 'RGB' and image_fingerprint(image) == batch['page']['pixels_sha256'],
            'ink grouping original pixels changed')
    require(image.width*image.height <= 4_000_000 and len(batch['detections'])*image.width*image.height <= 512_000_000
            and len(batch['proposals']) <= 256, 'ink grouping geometry budget')
    require(not any(p['grouping_method'] in POLICIES for p in batch['proposals']),
            'ink grouping already attempted; use its original partition')
    raw = {d['source_id']: d for d in batch['detections']}
    detections = [SimpleNamespace(**d) for d in batch['detections']]
    owners = enclosure_map(image, detections)
    source = OriginalLines(image, batch['detections'])
    pixels = np.asarray(image)
    eligible = [p for p in batch['proposals'] if p['direction'] == 'vertical'
        and not p['reviewed'] and not p['annotation_ids'] and p['main_ids'] == p['member_ids']]
    needed = {i for p in eligible for i in p['main_ids']}
    tight, line_evidence, failures = {}, {}, []
    for identity in raw:
        if identity not in needed or identity not in owners:
            continue
        try:
            _, binding = source.line(identity)
            stats = source.stats[binding['components']]
            # Half-open original component extents, including page-edge ink.
            bounds = [int(stats[:, 0].min()), int(stats[:, 1].min()),
                      int((stats[:, 0]+stats[:, 2]).max()), int((stats[:, 1]+stats[:, 3]).max())]
            tight[identity] = bounds
            line_evidence[identity] = {**binding, 'ink_bounds': bounds}
        except ValueError as error:
            failures.append({'source_id': identity, 'reason': str(error)})
    options = []
    for index, first in enumerate(eligible):
        for second in eligible[index+1:]:
            ids = first['main_ids']+second['main_ids']
            if (not 2 <= len(ids) <= 8 or not all(i in tight for i in ids)
                    or len({owners[i][0] for i in ids}) != 1):
                continue
            columns = []
            for identity in sorted(ids, key=lambda i: (-(tight[i][0]+tight[i][2]), tight[i][1], type(i).__name__, str(i))):
                b = tight[identity]
                if columns:
                    a = columns[-1]['bounds']
                    overlap = min(a[2], b[2])-max(a[0], b[0])
                    if overlap > .5*min(a[2]-a[0], b[2]-b[0]):
                        columns[-1] = {'ids': columns[-1]['ids']+[identity],
                            'bounds': [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]}
                        continue
                columns.append({'ids': [identity], 'bounds': b})
            if len(columns) < 2 or any(any(i in first['main_ids'] for i in c['ids'])
                    and any(i in second['main_ids'] for i in c['ids']) for c in columns):
                continue
            belongs = [c['ids'][0] in first['main_ids'] for c in columns]
            if sum(a != b for a, b in zip(belongs, belongs[1:])) != 1:
                continue
            scale = float(np.median([c['bounds'][2]-c['bounds'][0] for c in columns]))
            if policy == POLICY and max(c['bounds'][1] for c in columns)-min(c['bounds'][1] for c in columns) > .9*scale:
                continue
            seams = []
            for right, left, a, b in zip(columns, columns[1:], belongs, belongs[1:]):
                if a == b:
                    continue
                r, l = right['bounds'], left['bounds']
                # Internal indentation belongs to the retained parent partition.
                # The new policy measures alignment only where those parents join.
                if policy == BOUNDARY_POLICY and abs(r[1]-l[1]) > .9*scale:
                    break
                gap = r[0]-l[2]; top = max(r[1], l[1]); bottom = min(r[3], l[3])
                if not 2 <= gap <= .9*scale or bottom-top < .5*scale:
                    break
                bounds = [l[2], top, r[0], bottom]
                corridor = pixels[top:bottom, l[2]:r[0]]
                white = corridor.min(axis=2) >= 245
                if (float(white.mean()) < .98 or float(white.all(axis=0).mean()) < .5
                        or crop_conflicts(bounds, ids, detections)):
                    break
                seams.append({'bounds': bounds, 'white_fraction': float(white.mean()),
                    'fully_white_column_fraction': float(white.all(axis=0).mean()),
                    'original_rgb_sha256': hashlib.sha256(corridor.tobytes()).hexdigest()})
            if len(seams) != 1:
                continue
            order = [i for c in columns for i in sorted(c['ids'], key=lambda i: (tight[i][1], str(i)))]
            points = [point for i in order for point in raw[i]['polygon']]
            bounds = [math.floor(min(x for x, y in points)), math.floor(min(y for x, y in points)),
                      math.ceil(max(x for x, y in points)), math.ceil(max(y for x, y in points))]
            conflicts = crop_conflicts(bounds, order, detections)
            # Existing corner conflicts remain explicit; no newly imported source.
            previous = {c.source_id for p in (first, second)
                        for c in crop_conflicts(p['crop_bounds'], p['member_ids'], detections)}
            if any(c.source_id not in previous for c in conflicts):
                continue
            options.append({'previous_proposal_ids': [first['proposal_id'], second['proposal_id']],
                'main_ids': order, 'bounds': bounds, 'columns': columns, 'seams': seams,
                'lines': [line_evidence[i] for i in order],
                'enclosure_evidence': [[i, *owners[i]] for i in order],
                'crop_conflicts': [asdict(c) for c in conflicts]})
    proposals, records, ambiguous = list(batch['proposals']), [], []
    by_id = {p['proposal_id']: p for p in proposals}
    for option in options:
        parents = option['previous_proposal_ids']
        if any(sum(identity in other['previous_proposal_ids'] for other in options) != 1 for identity in parents):
            ambiguous.append(option)
            continue
        first, second = [by_id[i] for i in parents]
        ids = option['main_ids']
        identity = {'policy': batch['policy'], 'page': batch['page'], 'main_ids': ids, 'members': ids,
            'member_polygons': [(i, raw[i]['polygon']) for i in ids], 'annotation_parents': (), 'direction': 'vertical'}
        if batch.get('ownership_policy', 'production8') != 'production8':
            identity['ownership_policy'] = batch['ownership_policy']
        if batch.get('companion_policy', 'disabled') != 'disabled':
            identity['companion_policy'] = batch['companion_policy']
        proposal_id = 'proposal-'+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        merged = {**first, 'proposal_id': proposal_id, 'main_ids': ids, 'member_ids': ids,
            'grouping_method': policy, 'crop_bounds': option['bounds'], 'crop_conflicts': option['crop_conflicts'],
            'crop_safe': not option['crop_conflicts'] and first['crop_safe'] and second['crop_safe'],
            'rejected_merge_evidence': first['rejected_merge_evidence']+second['rejected_merge_evidence'],
            'region_ids': sorted(set(first['region_ids']+second['region_ids']))}
        for key in ('structural_reasons', 'word_reasons', 'crop_reasons'):
            merged[key] = list(dict.fromkeys(first[key]+second[key]+([REASON] if key == 'structural_reasons' else [])))
        proposals[proposals.index(first)] = merged
        proposals.remove(second)
        records.append({**option, 'proposal_id': proposal_id, 'policy': policy,
            'erase_permission': False, 'full_utterance_verified': False})
    updated = {**batch, 'proposals': proposals}
    automatic_units(updated)
    return snapshot(updated), snapshot({'records': records, 'ambiguous': ambiguous, 'line_failures': failures})


def refine(engine, image, original, *, policy=POLICY):
    """Keep the previous partition/results and expose only disjoint hypotheses."""
    require(not hasattr(engine, 'ink_column_groups'), 'ink grouping already attempted')
    require(len(engine.automatic_unit_proposals) == 1, 'one ink grouping page required')
    original_batch = engine.automatic_unit_proposals[0]
    before = snapshot(original_batch)
    require(hashlib.sha256(original).hexdigest() == before['page']['image_sha256'], 'ink grouping source bytes changed')
    prior = tuple(getattr(engine, 'automatic_recognition_results', ()) or ())
    require(len(prior) <= 1, 'one recognition result page required')
    previous_results = [snapshot(r) for r in prior[0]] if prior else []
    automatic_units(before, previous_results)
    require(not any(c.get('reviewed') is True for r in previous_results for c in r['candidates']),
            'ink grouping must preserve reviewed source selections')
    updated, evidence = plan(before, image, policy=policy)
    # Keep typed batches for the OCR facade and source-localization handoff.
    typed = {p.proposal_id: p for p in original_batch.proposals}
    merged = {r['proposal_id']: r for r in evidence['records']}
    detections = original_batch.detections
    proposals = []
    for p in updated['proposals']:
        if p['proposal_id'] not in merged:
            proposals.append(typed[p['proposal_id']]); continue
        first = typed[merged[p['proposal_id']]['previous_proposal_ids'][0]]
        fields = {key: tuple(p[key]) for key in ('main_ids', 'member_ids', 'crop_bounds', 'structural_reasons',
            'word_reasons', 'crop_reasons', 'region_ids', 'rejected_merge_evidence')}
        proposals.append(replace(first, **fields, proposal_id=p['proposal_id'], grouping_method=policy,
            crop_safe=p['crop_safe'], crop_conflicts=crop_conflicts(p['crop_bounds'], p['member_ids'], detections)))
    batch = replace(original_batch, proposals=tuple(proposals))
    require(snapshot(batch) == updated, 'ink grouping typed partition drift')
    from .automatic_recognition import AutomaticRecognitionResult
    by_id = {r.proposal_id: r for r in prior[0]} if prior else {}
    results = tuple(by_id.get(p.proposal_id, AutomaticRecognitionResult(p.proposal_id)) for p in batch.proposals)
    engine.ink_column_groups = {'policy': policy, 'previous_batch': before, 'previous_results': previous_results,
        'previous_batch_revision': revision(before), 'previous_results_revision': revision(previous_results),
        'result_batch_revision': revision(updated), 'source_pixels_sha256': image_fingerprint(image), **evidence}
    engine.automatic_unit_proposals = (batch,)
    engine.automatic_recognition_results = (results,)


def validate_saved(ocr, image):
    """Recompute geometry from original pixels before translation or rendering."""
    batch = snapshot(ocr['batches'][0])
    stage = ocr.get('ink_column_groups')
    marked = any(p['grouping_method'] in POLICIES for p in batch['proposals'])
    require(not marked or stage is not None, 'missing saved ink grouping evidence')
    if stage is None:
        return
    before = stage['previous_batch']
    require(stage['policy'] in POLICIES and stage['previous_batch_revision'] == revision(before)
        and stage['previous_results_revision'] == revision(stage['previous_results'])
        and stage['result_batch_revision'] == revision(batch)
        and stage['source_pixels_sha256'] == batch['page']['pixels_sha256'] == image_fingerprint(image),
        'saved ink grouping revision drift')
    automatic_units(before, stage['previous_results'])
    expected, evidence = plan(before, image, policy=stage['policy'])
    require(expected == batch and all(snapshot(stage[key]) == snapshot(value) for key, value in evidence.items()),
            'saved ink grouping geometry/pixels drift')
