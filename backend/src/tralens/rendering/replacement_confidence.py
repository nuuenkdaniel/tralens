"""Opt-in replacement triage, never a translation gate or accuracy probability.

The weakest bound raw main-line score limits OCR trust; annotations are not mixed
into that score. Alternative readings cannot inherit Paddle's raw score. Native
cleanup support and requested-size lettering are separate requirements. No text,
script, source ID, visual-review label or expected word is special-cased here.
"""
import math
import re
from dataclasses import replace

from ..translation.contracts import canonical, strict_json


POLICY = 'weakest_raw_line_native_cleanup_v1'
DEFAULT_MINIMUM = .70


def settings(policy='all', minimum=None):
    if type(policy) is not str or policy not in ('all', 'confidence-gated'):
        raise ValueError('unknown replacement policy')
    if policy == 'all':
        if minimum is not None:
            raise ValueError('minimum OCR confidence requires confidence-gated replacement policy')
        return None
    minimum = DEFAULT_MINIMUM if minimum is None else minimum
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError('minimum OCR confidence must be a finite number in 0..1')
    return {'name': policy, 'policy': POLICY, 'minimum_main_line_score': float(minimum),
        'score_is_accuracy_probability': False, 'coarse_footprint_cleanup_allowed': False,
        'below_requested_font_size_allowed': False, 'unchanged_target_preserves_source': True,
        'translation_policy_unchanged': True,
        'calibration_scope': 'provisional single-page model visual assessment; not held-out accuracy validation'}


def retains_printed_words(unit, target):
    """Keep confident Latin labels when the model only punctuates their rows."""
    reading = unit.selected
    if reading.origin != 'raw_main_reading' or not re.fullmatch(r'[A-Za-z]{2,128}', reading.text):
        return False
    if len(target) > 256 or not re.fullmatch(r'[A-Za-z]+(?:[ ,/]+[A-Za-z]+)*[.]?', target):
        return False
    lines = strict_json(reading.evidence_json)
    if type(lines) is not list or len(lines) != len(unit.order):
        return False
    words = re.findall(r'[A-Za-z]+', target)
    if len(words) != len(lines):
        return False
    exact_words = set()
    for line, source_id, word in zip(lines, unit.order, words):
        if (type(line) is not dict or line.get('source_id') != source_id
                or line.get('status') != 'accepted' or type(line.get('confidence')) not in (int, float)
                or not .97 <= line['confidence'] <= 1):
            return False
        source = line.get('text')
        if type(source) is not str or not re.fullmatch(r'[A-Za-z]{2,32}', source):
            return False
        if source == word:
            exact_words.add(source)
        elif not (source in exact_words and source.istitle() and word == source.lower()):
            return False
    return ''.join(line['text'] for line in lines) == reading.text


def assess_source(unit, target, minimum):
    """Caller first validates original/unit/request/result with existing bindings."""
    settings('confidence-gated', minimum)
    assessment = {'policy': POLICY, 'source_unit_revision': unit.revision,
        'source_reading_revision': unit.selected_revision, 'minimum_required': float(minimum),
        'score_kind': 'minimum uncalibrated raw OCR main-line score, not accuracy probability',
        'source_target_exact_match': target == unit.selected.text,
        'line_scores': [], 'minimum_main_line_score': None, 'allowed': False,
        'reason_code': 'replacement_confidence_unavailable',
        'reason': 'Selected reading lacks usable source-bound raw main-line scores.'}
    if unit.selected.origin == 'raw_main_reading':
        try:
            raw = strict_json(unit.selected.evidence_json)
            if type(raw) is not list or not 1 <= len(raw) <= 32 or len(raw) != len(unit.order):
                raise ValueError('main-line coverage')
            for item, source_id in zip(raw, unit.order):
                if (type(item) is not dict or type(item['source_id']) is not type(source_id)
                        or item['source_id'] != source_id or canonical(item['page']) != unit.page_json
                        or type(item['text']) is not str or item['status'] not in ('accepted', 'review')):
                    raise ValueError('main-line binding/status')
                score = item['confidence']
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('unusable confidence')
            if ''.join(item['text'] for item in raw) != unit.selected.text:
                raise ValueError('raw characters differ from selected reading')
            assessment['line_scores'] = [{'source_id': item['source_id'], 'score': float(item['confidence'])}
                                         for item in raw]
            score = min(item['score'] for item in assessment['line_scores'])
            assessment['minimum_main_line_score'] = score
            assessment.update(allowed=score >= minimum,
                reason_code=None if score >= minimum else 'replacement_low_ocr_confidence',
                reason=('Bound main-line scores meet the experimental threshold; correctness remains unverified.'
                        if score >= minimum else 'At least one bound main-line OCR score is below the replacement threshold.'))
        except (ValueError, TypeError, KeyError):
            pass
    elif unit.selected.origin == 'paired_visual_reread_unverified':
        from ..ocr.quality_reread import preferred_reading
        raw = [reading for reading in unit.readings if reading.origin == 'raw_main_reading']
        if len(raw) == 1:
            candidate = preferred_reading(replace(unit, selected_revision=raw[0].revision))
            if candidate is not None and candidate.revision == unit.selected_revision:
                assessment.update(allowed=True, reason_code=None,
                    score_kind='unscored paired visual/line reread; no raw Paddle score inherited',
                    reason='Explicitly enabled, source-bound paired-reader hypothesis; semantic correctness remains unverified.')
                from ..ocr import quality_lines
                proof = strict_json(candidate.evidence_json)['quality_reread']
                if proof['policy'] == quality_lines.POLICY:
                    scores = [{'source_id': line['source_id'], 'score': line['confidence']}
                        for line in proof['lines'] if line['kind'] == 'raw_unchanged']
                    assessment.update(line_scores=scores,
                        minimum_main_line_score=min((line['score'] for line in scores), default=None),
                        score_kind='retained raw lines keep their own scores; changed lines are unscored paired-reader hypotheses')
                    if any(line['score'] < minimum for line in scores):
                        assessment.update(allowed=False, reason_code='replacement_low_ocr_confidence',
                            reason='A retained raw line is below the requested replacement threshold; reread scores are not inherited.')
    # Exact string identity justifies preserving source pixels, not certifying OCR.
    if assessment['source_target_exact_match']:
        assessment.update(allowed=False, reason_code='replacement_unchanged_text',
            reason='Translation exactly matches the selected source string; preserve original lettering without erasure.')
    return assessment


def layout_rejection(layout, requested_minimum):
    size = layout.get('size')
    if type(size) is not int or size < requested_minimum:
        return 'Translated text would need lettering below the requested minimum size.'
    return None


def summary(rows):
    def count(code):
        return sum(row.get('reason_code') == code for row in rows)
    return {'bound_translation_results_retained': sum('binding' in row for row in rows),
        'actual_replacements': sum(row['status'] == 'automatic_composed' for row in rows),
        'unchanged_text_preserved': count('replacement_unchanged_text'),
        'low_ocr_confidence_deferred': count('replacement_low_ocr_confidence'),
        'missing_confidence_deferred': count('replacement_confidence_unavailable'),
        'cleanup_support_deferred': count('replacement_cleanup_confidence'),
        'readability_deferred': count('replacement_layout_confidence'),
        'bound_translations_without_replacement': sum('binding' in row and row['status'] == 'skipped' for row in rows),
        'semantic_quality_verified': False,
        'note': 'Unedited source text remains; translations are retained, not counted as image replacements.'}
