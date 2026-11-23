"""Source-bound text-region translation despite uncertain spatial ownership.

Original OCR utterances stay blocked and unchanged. A separately bound request
may translate their exact raw main text; it does not certify a full sentence,
balloon ownership, reading order or hidden artwork. Review-only fragments are
never promoted through this route.
"""
from dataclasses import asdict, replace
import math
import unicodedata

from .contracts import Utterance, TranslationRequest, TranslationResult, UnitOutcome, canonical, revision, strict_json


POLICY = 'source_bound_text_region_v1'
KIND = 'coverage_text_region'
COMPLETENESS = 'unverified_ocr_region_not_certified_full_utterance'
MAX_CALLS = 64
WAIVABLE = frozenset({'unknown_region_ownership', 'mixed_region_ownership', 'unknown_reading_orientation',
                     'crop_isolation_unresolved', 'unsafe_crop', 'ambiguous_annotation_parents',
                     'ambiguous_annotation_chain_or_role', 'unresolved_annotation_relationship',
                     'annotation_role_unresolved_not_independent_text_proof'})
INFORMATIONAL_STRUCTURE = frozenset({'automatic_not_human_reviewed', 'complementary_component_ownership',
    'annotation_crosses_original_boundary', 'annotation_geometry_crosses_distinct_regions',
    'original_glyph_scale_not_smaller_than_main', 'original_glyph_flow_inconsistent_with_reading_annotation',
    'automatic_terminal_row_hypothesis', 'automatic_vertical_continuation_hypothesis',
    'original_light_lettering_on_dark_surface', 'different_original_lettering_polarity',
    'relative_side_extent_alignment_and_scale', 'missing_kana_and_main_kanji_corroboration',
    'original_glyph_support_inconclusive', 'horizontal_clause_with_case_particle', 'punctuated_horizontal_heading'})


def validate_policy(policy):
    if type(policy) is not str or policy not in ('coverage-first', 'conservative'):
        raise ValueError('unknown translation policy')
    return policy


def rejection_reasons(unit):
    if type(unit) is not Utterance:
        return ('requires_original_utterance',)
    if unit.disposition != 'blocked':
        return ('strict_preview_available',)
    reasons = ['non_waivable:'+r for r in unit.blocking_reasons if r not in WAIVABLE]
    reasons.extend('non_waivable_structure:'+r for r in unit.structural_reasons
                   if r not in WAIVABLE | INFORMATIONAL_STRUCTURE)
    if unit.selected.origin != 'raw_main_reading':
        reasons.append('requires_exact_raw_main_reading')
    if not 0 < len(unit.selected.text) <= 1000 or not unit.selected.text.strip():
        reasons.append('blank_or_over_budget_source_text')
    if len(unit.member_ids) > 32:
        reasons.append('over_budget_source_members')
    try:
        raw = strict_json(unit.selected.evidence_json)
        if type(raw) is not list or len(raw) != len(unit.order) or len(raw) > 32:
            raise ValueError('main source coverage')
        for item, identity in zip(raw, unit.order):
            required = {'source_id', 'page', 'text', 'confidence', 'polygon', 'status', 'reason', 'selected_text', 'metadata_json'}
            if type(item) is not dict or set(item) != required:
                raise ValueError('raw fields')
            if (type(item['source_id']) is not type(identity) or item['source_id'] != identity
                    or canonical(item['page']) != unit.page_json or type(item['text']) is not str
                    or not item['text'].strip() or type(item['selected_text']) is not str
                    or type(item['reason']) is not str):
                raise ValueError('raw identity/text/page')
            if item['status'] not in ('accepted', 'review'):
                reasons.append('raw_reading_rejected_or_unknown_status')
            if type(item['confidence']) not in (int, float) or not math.isfinite(item['confidence']):
                raise ValueError('raw confidence')
            page = item['page']
            if (type(page) is not dict or type(page.get('page_id')) is not str or not page['page_id']
                    or type(page.get('image_path')) is not str or not page['image_path']):
                raise ValueError('page identity/path')
            for key in ('image_sha256', 'pixels_sha256', 'comparison_sha256'):
                digest = page.get(key)
                if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('page fingerprint')
            polygon = item['polygon']
            if (type(polygon) is not list or not 3 <= len(polygon) <= 128 or
                    any(type(p) is not list or len(p) != 2 or
                        any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in p) for p in polygon) or
                    abs(sum(p[0]*q[1]-q[0]*p[1] for p, q in zip(polygon, polygon[1:]+polygon[:1]))) < 1e-7):
                raise ValueError('raw geometry')
            metadata = strict_json(item['metadata_json'])
            if type(metadata) is not dict:
                raise ValueError('raw metadata')
            association = metadata.get('paddle_association')
            if item['reason'] == 'unresolved_paddle_association' or (association is not None and
                    (type(association) is not dict or association.get('status') != 'matched'
                     or type(association.get('issues', [])) is not list or association.get('issues', []))):
                reasons.append('unresolved_raw_source_association')
        if ''.join(item['text'] for item in raw) != unit.selected.text:
            raise ValueError('raw character drift')
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        reasons.append('missing_or_inconsistent_raw_evidence')
    return tuple(dict.fromkeys(reasons))


def provenance(unit, reading_revision=None):
    result = dict(policy=POLICY, input_kind=KIND, original_unit_revision=unit.revision,
        source_reading_revision=unit.selected_revision, source_ids=list(unit.member_ids),
        waived_blocking_reasons=list(unit.blocking_reasons),
        complete_utterance_claimed=False, source_readability_verified=False,
        scope='exact raw main text; no crop/image input or source correction; best-effort local replacement only')
    if reading_revision is not None:
        from ..ocr.quality_reread import preferred_reading, POLICY as reread_policy
        candidate = preferred_reading(unit)
        if candidate is None:
            from ..ocr.manga_dialogue import preferred_reading, POLICY as reread_policy
            candidate=preferred_reading(unit)
        if candidate is None:
            from ..ocr.crop_readings import preferred_reading, POLICY as reread_policy
            candidate=preferred_reading(unit)
        if candidate is None or candidate.revision != reading_revision:
            raise ValueError('reread selection is not the exact corroborated source-bound candidate')
        result.update(reread_selected_revision=reading_revision, reread_policy=reread_policy,
            scope='exact unverified source-bound reread candidate; original raw reading retained; no source truth or erasure certification')
    return result


def derive(unit, reading_revision=None):
    rejected = rejection_reasons(unit)
    if reading_revision is not None and rejected==('strict_preview_available',):
        rejected=()
    if rejected:
        raise ValueError('coverage input rejected: '+', '.join(rejected))
    bound = provenance(unit, reading_revision)
    return replace(unit, blocked_reasons=(), completeness=COMPLETENESS,
        selected_revision=reading_revision or unit.selected_revision,
        lexical_reasons=tuple(dict.fromkeys(unit.lexical_reasons+('coverage_spatial_uncertainty',))),
        evidence_revision=revision(bound))


def priority(unit):
    """Weak scheduling hints, NOT a language/readability or dialogue certificate."""
    text = unicodedata.normalize('NFKC', unit.selected.text)
    kana_han = any(unicodedata.name(c, '').startswith(('HIRAGANA LETTER ', 'KATAKANA LETTER ',
        'CJK UNIFIED IDEOGRAPH-', 'CJK COMPATIBILITY IDEOGRAPH-')) for c in text)
    return (len(unit.order) == 1, not kana_han, -min(sum(c.isalnum() for c in text), 80))


def bound_result(record, original, *, request=None):
    """Re-derive every permission/binding from the unchanged original utterance."""
    reading_revision = record.get('coverage', {}).get('reread_selected_revision')
    derived = derive(original, reading_revision)
    request = TranslationRequest((derived,), (derived.unit_id,)) if request is None else request
    if (type(request) is not TranslationRequest or request.target_ids != (derived.unit_id,)
            or next((u for u in request.units if u.unit_id == derived.unit_id), None) != derived):
        raise ValueError('coverage request changed the target input')
    if (record.get('input_kind') != KIND or record.get('status') != 'preview'
            or record.get('binding_current') is not True or record.get('unit_revision') != original.revision
            or canonical(record.get('unit')) != canonical(asdict(original))
            or canonical(record.get('coverage')) != canonical(provenance(original, reading_revision))
            or canonical(record.get('translation_input')) != canonical(asdict(derived))
            or canonical(record.get('request')) != canonical(asdict(request))
            or record.get('request_revision') != request.revision):
        raise ValueError('stale or unsupported coverage translation binding')
    data = dict(record['result'])
    data['translations'] = tuple(tuple(p) for p in data['translations'])
    data['outcomes'] = tuple(UnitOutcome(**{**o, 'reasons': tuple(o['reasons'])}) for o in data['outcomes'])
    result = TranslationResult(**data)
    result.assert_current(request)
    metadata = strict_json(result.execution_metadata_json)
    if (result.status != 'preview' or len(result.translations) != 1
            or result.translations[0][0] != original.unit_id or result.review_needed is not True
            or any(o.disposition != 'review' for o in result.outcomes)
            or metadata.get('output_kind') != KIND or metadata.get('review_required') is not True
            or metadata.get('complete_utterance_claimed') is not False
            or metadata.get('source_readability_verified') is not False):
        raise ValueError('coverage replacement requires an explicitly uncertain bound translation')
    binding = dict(unit_id=original.unit_id, member_ids=list(original.member_ids), order=list(original.order),
        unit_revision=original.revision, request_revision=request.revision,
        result_sha256=revision(record['result']), text=result.translations[0][1], input_kind=KIND)
    if reading_revision is not None:
        binding['reread_selected_revision'] = reading_revision
    return binding
