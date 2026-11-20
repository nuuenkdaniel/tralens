"""Explicit checked OCR outputs. No engine calls, metrics or box mutations."""
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum

from .recognition_units import PageProvenance


class ReviewKind(str, Enum):
    PROVISIONAL_VISUAL = "provisional_visual"
    INDEPENDENT_REFERENCE = "independent_reference"


def _ids(values):
    return (isinstance(values, (list, tuple)) and bool(values)
            and all(type(i) in (int, str) and i != "" for i in values)
            and len(set(values)) == len(values))


def _bounds(values):
    return (isinstance(values, (list, tuple)) and len(values) == 4
            and all(type(v) in (int, float) and (type(v) is int or math.isfinite(v))
                    and v == int(v) for v in values)
            and 0 <= values[0] < values[2] and 0 <= values[1] < values[3])


@dataclass(frozen=True)
class CandidateEvidence:
    unit_id: str
    member_ids: tuple
    page: PageProvenance
    candidate_index: int
    text: str
    engine: str
    crop_bounds: tuple
    crop_json: str
    candidate_id: str


def bind_candidate(unit_id, member_ids, page, candidate_index, text, engine, crop):
    """Snapshot the ENTIRE supplied crop provenance and exact saved text/identity."""
    if not isinstance(crop, dict):
        crop = asdict(crop)
    if (not isinstance(unit_id, str) or not unit_id or not _ids(member_ids)
            or type(candidate_index) is not int or candidate_index < 0
            or not isinstance(text, str) or not isinstance(engine, str) or not engine
            or not isinstance(page, PageProvenance)
            or not isinstance(page.page_id, str) or not isinstance(page.image_path, str)):
        raise ValueError("invalid_candidate_identity")
    if (crop.get("unit_id") != unit_id or not _ids(crop.get("member_ids"))
            or tuple(crop["member_ids"]) != tuple(member_ids)
            or crop.get("page") != asdict(page) or not _bounds(crop.get("bounds"))):
        raise ValueError("candidate_crop_binding_mismatch")
    crop_json = json.dumps(crop, sort_keys=True, ensure_ascii=False, allow_nan=False)
    identity = {"unit_id": unit_id, "member_ids": tuple(member_ids), "page": asdict(page),
                "candidate_index": candidate_index, "text": text, "engine": engine, "crop": crop}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode("utf-8")).hexdigest()
    return CandidateEvidence(unit_id, tuple(member_ids), page, candidate_index, text, engine,
                             tuple(crop["bounds"]), crop_json, digest)


@dataclass(frozen=True)
class TranscriptionReview:
    unit_id: str
    member_ids: tuple
    page: PageProvenance
    candidate_id: str
    candidate_text: str
    crop_bounds: tuple
    mode: str
    review_kind: str
    reviewer: str
    source: str
    reviewed: bool = False
    wording_checked: bool = False
    order_checked: bool = False
    full_utterance_checked: bool = False
    wording_uncertain: bool = True
    order_uncertain: bool = True
    coverage_uncertain: bool = True
    punctuation_uncertain: bool = True
    corrected_text: object = None
    independence_confirmed: bool = False
    reference_source: str = ""

    def __post_init__(self):
        # Freeze collection inputs, without coercing invalid flags or IDs.
        if isinstance(self.member_ids, (list, tuple)):
            object.__setattr__(self, "member_ids", tuple(self.member_ids))
        if isinstance(self.crop_bounds, (list, tuple)):
            object.__setattr__(self, "crop_bounds", tuple(self.crop_bounds))


@dataclass(frozen=True)
class SelectedTranscription:
    """Separate OCR output, never an accepted box or a render/translation permit."""
    text: str
    candidate: CandidateEvidence
    review: TranscriptionReview
    review_kind: ReviewKind
    full_utterance_checked: bool = True


@dataclass(frozen=True)
class SelectionResult:
    unit_id: str
    selected: object = None
    errors: tuple = ()


def select_transcription(candidates, reviews):
    """Atomic selection for ONE unit. Any conflicting/incomplete record blocks it.

    Candidates are immutable snapshots. Agreement/confidence/status never
    substitute for explicit checks; exact text preserves punctuation and kana.
    """
    candidates, reviews = tuple(candidates), tuple(reviews)
    unit_id = candidates[0].unit_id if candidates and isinstance(candidates[0], CandidateEvidence) else ""
    def unresolved(reason):
        return SelectionResult(unit_id, errors=(reason,))
    if not candidates:
        return unresolved("no_saved_candidates")
    if any(not isinstance(c, CandidateEvidence) or not isinstance(c.candidate_id, str)
           or not _ids(c.member_ids) or not isinstance(c.crop_json, str)
           or type(c.candidate_index) is not int for c in candidates):
        return unresolved("invalid_candidate_evidence")
    first = candidates[0]
    if (any(c.unit_id != unit_id or c.member_ids != first.member_ids or c.page != first.page
            for c in candidates) or len({c.candidate_id for c in candidates}) != len(candidates)
            or len({c.candidate_index for c in candidates}) != len(candidates)):
        return unresolved("ambiguous_candidate_unit")
    # Reject manually fabricated/drifted evidence as well as drifted review text.
    for candidate in candidates:
        try:
            rebound = bind_candidate(candidate.unit_id, candidate.member_ids, candidate.page,
                candidate.candidate_index, candidate.text, candidate.engine, json.loads(candidate.crop_json))
        except (TypeError, ValueError, KeyError):
            return unresolved("invalid_candidate_evidence")
        if rebound != candidate:
            return unresolved("candidate_evidence_drift")
    if len(reviews) != 1:
        return unresolved("missing_review" if not reviews else "ambiguous_review_records")
    review = reviews[0]
    if not isinstance(review, TranscriptionReview):
        return unresolved("invalid_review_type")
    if (review.unit_id != unit_id or not _ids(review.member_ids)
            or review.member_ids != first.member_ids or review.page != first.page):
        return unresolved("review_unit_page_or_members_mismatch")
    matches = [c for c in candidates if c.candidate_id == review.candidate_id]
    if len(matches) != 1:
        return unresolved("foreign_or_stale_candidate_identity")
    candidate = matches[0]
    if (not isinstance(review.candidate_text, str) or review.candidate_text != candidate.text
            or not _bounds(review.crop_bounds) or review.crop_bounds != candidate.crop_bounds):
        return unresolved("review_candidate_text_or_crop_mismatch")
    checked = (review.reviewed, review.wording_checked, review.order_checked, review.full_utterance_checked)
    uncertain = (review.wording_uncertain, review.order_uncertain, review.coverage_uncertain,
                 review.punctuation_uncertain)
    if any(type(flag) is not bool for flag in checked + uncertain + (review.independence_confirmed,)):
        return unresolved("invalid_review_flag_type")
    if (not all(checked) or any(uncertain) or not isinstance(review.reviewer, str)
            or not review.reviewer.strip() or not isinstance(review.source, str) or not review.source.strip()):
        return unresolved("incomplete_or_uncertain_review")
    if not isinstance(review.reference_source, str):
        return unresolved("invalid_reference_source_type")
    try:
        kind = ReviewKind(review.review_kind)
    except (TypeError, ValueError):
        return unresolved("invalid_review_kind")
    if review.mode == "existing_candidate":
        if review.corrected_text is not None:
            return unresolved("ambiguous_selected_text")
        text = candidate.text
    elif review.mode == "corrected_text":
        text = review.corrected_text
    else:
        return unresolved("invalid_selection_mode")
    if not isinstance(text, str) or not text.strip():
        return unresolved("invalid_selected_text")
    if kind == ReviewKind.INDEPENDENT_REFERENCE:
        if (review.mode != "corrected_text" or review.independence_confirmed is not True
                or not isinstance(review.reference_source, str) or not review.reference_source.strip()):
            return unresolved("unestablished_independent_reference")
    elif review.independence_confirmed or review.reference_source:
        return unresolved("ambiguous_reference_claim")
    return SelectionResult(unit_id, SelectedTranscription(text, candidate, review, kind))
