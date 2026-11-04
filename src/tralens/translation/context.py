"""Opt-in nearby OCR context, never an inferred dialogue order or source correction."""
from dataclasses import asdict, replace
import math

from .contracts import TranslationRequest, Utterance, canonical, revision, strict_json


POLICY = 'nearby_bound_text_single_rendition_v1'
MAX_NEIGHBORS = 4
MAX_CONTEXT_CHARS = 1024
MAX_UNIT_CHARS = 512
RAW_CONTEXT_FLOOR = .70  # Routing only; not an accuracy probability.
INSTRUCTIONS = (' Produce one coherent translation for each target, not a list of alternate renditions. '
    'Preserve ambiguity in that rendition; do not invent missing wording or shorten it to fit a layout. '
    'Nearby OCR context is uncertain, may belong to another panel, and establishes neither reading '
    'order nor speaker identity. Use it only when relevant; never import its wording into the target.')


def validate_policy(policy, profile):
    if type(policy) is not str or policy not in ('isolated', 'nearby'):
        raise ValueError('unknown translation context policy')
    if policy != 'isolated' and not (profile.context and profile.schema_json and not profile.specialist_template):
        raise ValueError('nearby translation context requires an explicitly selected schema-capable generalist model/profile')
    return policy


def _bounds(value):
    if (type(value) not in (list, tuple) or len(value) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise ValueError('bounded original proposal geometry required for translation context')
    return tuple(value)


def eligible(unit):
    """Keep weak/alternative text available as a target, but out of this context pilot."""
    if (unit.disposition == 'blocked' or not 0 < len(unit.selected.text) <= MAX_UNIT_CHARS
            or unit.selected.origin != 'raw_main_reading'):
        return False
    try:
        lines = strict_json(unit.selected.evidence_json)
        return (type(lines) is list and 0 < len(lines) <= 32
            and len(lines) == len(unit.order)
            and all(type(line['confidence']) in (int, float) and math.isfinite(line['confidence'])
                    and RAW_CONTEXT_FLOOR <= line['confidence'] <= 1 for line in lines))
    except (KeyError, TypeError, ValueError):
        return False


def plan(request, pool, bounds):
    """Deterministic selection from independently permitted exact text inputs.

    The caller reconstructs `pool` from the page's original units and existing
    coverage derivations, never from previous English output or model opinions.
    """
    if (type(request) is not TranslationRequest or len(request.units) != 1
            or len(request.target_ids) != 1 or request.context_ids or request.mode != 'A'
            or request.page_order or request.image_evidence_json
            or type(pool) is not tuple or not 1 <= len(pool) <= 512
            or any(type(unit) is not Utterance for unit in pool)
            or len({u.unit_id for u in pool}) != len(pool)):
        raise ValueError('context planning requires one isolated target and a bounded unique source pool')
    target = request.units[0]
    if (any(u.page_json != target.page_json for u in pool)
            or target.unit_id not in {u.unit_id for u in pool}
            or next(u for u in pool if u.unit_id == target.unit_id) != target):
        raise ValueError('foreign page or changed target in translation context pool')
    a = _bounds(bounds[target.unit_id])
    ranked = []
    for unit in pool:
        b = _bounds(bounds[unit.unit_id])
        if unit.unit_id == target.unit_id or not eligible(unit):
            continue
        dx = max(0, a[0]-b[2], b[0]-a[2])
        dy = max(0, a[1]-b[3], b[1]-a[3])
        center = ((a[0]+a[2]-b[0]-b[2])**2 + (a[1]+a[3]-b[1]-b[3])**2)/4
        ranked.append(((dx*dx+dy*dy, center, unit.unit_id), unit))
    selected, rows, characters = [], [], 0
    for rank, unit in sorted(ranked, key=lambda item: item[0]):
        if len(selected) >= MAX_NEIGHBORS:
            break
        if characters + len(unit.selected.text) > MAX_CONTEXT_CHARS:
            continue
        selected.append(unit)
        characters += len(unit.selected.text)
        rows.append({'unit_id': unit.unit_id, 'unit_revision': unit.revision,
            'selected_revision': unit.selected_revision, 'bounds': list(bounds[unit.unit_id]),
            'edge_distance_squared': rank[0], 'center_distance_squared': rank[1]})
    planned = replace(request, units=(target, *selected), context_ids=tuple(u.unit_id for u in selected),
        mode='B', style=replace(request.style, instructions=request.style.instructions+INSTRUCTIONS))
    evidence = {'policy': POLICY, 'target_revision': target.revision, 'target_bounds': list(a),
        'pool_revision': revision([{'unit_revision': u.revision, 'bounds': list(bounds[u.unit_id])} for u in pool]),
        'selected': rows, 'context_characters': characters, 'maximum_neighbors': MAX_NEIGHBORS,
        'raw_context_floor': RAW_CONTEXT_FLOOR, 'score_is_accuracy_probability': False,
        'inter_unit_order': 'unknown; spatial ranking is not reading order',
        'speaker_identity': 'unknown', 'semantic_relevance': 'unverified',
        'source_characters_changed': 0, 'request_revision': planned.revision}
    return planned, evidence


def page_plan(units, records, batch):
    """Shared reconstruction for execution AND render-time binding validation."""
    from . import coverage
    if len(units) != len(records) or len(units) > 512:
        raise ValueError('translation context page coverage mismatch')
    pool, bases = [], {}
    for unit, record in zip(units, records):
        if (record.get('unit_revision') != unit.revision
                or canonical(record.get('unit')) != canonical(asdict(unit))):
            raise ValueError('translation context original source drift')
        if record.get('input_kind', 'strict_utterance') == coverage.KIND:
            derived = coverage.derive(unit, record.get('coverage', {}).get('reread_selected_revision'))
            if (canonical(record.get('translation_input')) != canonical(asdict(derived))
                    or canonical(record.get('coverage')) != canonical(coverage.provenance(unit,
                        record.get('coverage', {}).get('reread_selected_revision')))):
                raise ValueError('translation context coverage derivation drift')
        else:
            derived = unit
        pool.append(derived)
        bases[unit.unit_id] = TranslationRequest((derived,), (derived.unit_id,))
    bounds = {p['proposal_id']: p['crop_bounds'] for p in batch['proposals']}
    if len(bounds) != len(units) or set(bounds) != set(bases):
        raise ValueError('translation context proposal identity mismatch')
    return {identity: plan(base, tuple(pool), bounds) for identity, base in bases.items()}


def saved_requests(preview, units):
    """Do not accept a saved context/style/request merely because hashes match."""
    policy = preview.get('settings', {}).get('translation_context', 'isolated')
    if policy == 'isolated':
        return {}
    from .runtime import GENERALIST, validate_model_profile
    validate_policy(policy, GENERALIST)
    if preview['settings'].get('profile') != 'generalist':
        raise ValueError('saved contextual request lacks explicit generalist selection')
    validate_model_profile(preview['settings']['model'], GENERALIST)
    scope = preview['settings'].get('context_scope', 'all_targets')
    if scope not in ('all_targets', 'explicit_candidates'):
        raise ValueError('unknown saved context selection scope')
    planned = page_plan(units, preview['units'], preview['ocr']['batches'][0])
    selected = {}
    for record in preview['units']:
        request, proof = planned[record['unit']['unit_id']]
        if scope == 'explicit_candidates' and 'context_selection' not in record:
            target = request.units[0]
            isolated = TranslationRequest((target,), (target.unit_id,))
            if (canonical(record.get('request')) != canonical(asdict(isolated))
                    or record.get('request_revision') != isolated.revision):
                raise ValueError('retained primary request is not its original isolated source request')
            continue
        if (canonical(record.get('context_selection')) != canonical(proof)
                or canonical(record.get('request')) != canonical(asdict(request))
                or record.get('request_revision') != request.revision):
            raise ValueError('saved context selection/request drift')
        selected[record['unit']['unit_id']] = request
    return selected
