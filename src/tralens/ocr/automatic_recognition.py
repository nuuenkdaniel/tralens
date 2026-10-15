"""Bounded opt-in automatic candidates, never reviewed or accepted transcriptions."""
import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from .automatic_units import (POLICY, COMPLEMENTARY_OWNERSHIP_POLICY, COMPANION_POLICY, crop_conflicts,
                              detection_evidence_sha256, propose_automatic_units)
from .recognition_units import _valid_polygon, image_fingerprint


def script_evidence(text):
    """Routing hint only. Preserve the unnormalized text for all comparisons."""
    kana = han = latin = other = 0
    for c in unicodedata.normalize("NFKC",text):
        if "\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa":
            kana += 1
        elif "\u3400" <= c <= "\u9fff" or "\U00020000" <= c <= "\U0002fa1f":
            han += 1
        elif unicodedata.category(c).startswith("L") and "LATIN" in unicodedata.name(c,""):
            latin += 1
        elif unicodedata.category(c).startswith("L") and c != "ー":
            other += 1
    if (kana or han) and (latin or other):
        return "mixed_script", "mixed_script_not_routed_to_manga_ocr"
    if not kana and not han and (latin or other):
        return "clearly_non_japanese", "unsupported_script_for_manga_ocr"
    if kana:
        return "japanese_kana_evidence", None
    if han:
        return "han_only_language_ambiguous", None
    return "no_lexical_script_evidence", "script_inconclusive_no_lexical_evidence"


@dataclass(frozen=True)
class AutomaticCropEvidence:
    proposal_id: str
    page: object
    member_ids: tuple
    main_ids: tuple
    bounds: tuple
    checked_neighbor_ids: tuple
    batch_evidence_sha256: str
    proposal_sha256: str
    script: str
    structural_reasons: tuple
    word_reasons: tuple
    completeness: str


@dataclass(frozen=True)
class AutomaticTextCandidate:
    text: str
    reference_main_text: str
    differing_spans: tuple
    crop: AutomaticCropEvidence
    engine: str
    model_identity: str
    model_settings_json: str
    status: str = "automated_unverified"
    reviewed: bool = False


@dataclass
class AutomaticRecognitionResult:
    proposal_id: str
    status: str = "not_requested"
    reason: str = "automatic_recognition_not_requested"
    attempts: int = 0
    candidates: tuple = ()
    crop: object = None
    error: object = None
    model_identity: object = None
    model_settings_json: str = "{}"


def prepare_automatic_crop(proposal, batch, image, *, original_image_bytes):
    """Recompute proposal evidence and ALL-neighbor isolation before each attempt.

    Reliable automated region support is required, but never proves ownership or
    full-utterance coverage. No reviewed UnitPlan is manufactured or consumed.
    """
    if (not isinstance(original_image_bytes, bytes)
            or hashlib.sha256(original_image_bytes).hexdigest() != batch.page.image_sha256
            or proposal.page != batch.page or image_fingerprint(image) != batch.page.pixels_sha256):
        return None,None,"original_page_or_image_provenance_mismatch"
    ids = [d.source_id for d in batch.detections]
    if (any(type(i) not in (int,str) or i == "" for i in ids) or len(set(ids)) != len(ids)
            or any(d.page != batch.page for d in batch.detections)):
        return None,None,"foreign_or_duplicate_raw_ids_or_page"
    if batch.policy != POLICY or batch.reviewed is not False or proposal.reviewed is not False:
        return None,None,"invalid_automatic_policy_or_review_claim"
    if batch.ownership_policy not in ("production8",COMPLEMENTARY_OWNERSHIP_POLICY):
        return None,None,"invalid_automatic_ownership_policy"
    if (batch.companion_policy not in ("disabled",COMPANION_POLICY)
            or (batch.companion_policy == COMPANION_POLICY and batch.ownership_policy != COMPLEMENTARY_OWNERSHIP_POLICY)):
        return None,None,"invalid_automatic_companion_policy"
    if detection_evidence_sha256(batch.detections) != batch.evidence_sha256:
        return None,None,"raw_evidence_drift"
    proposal_ids = [p.proposal_id for p in batch.proposals]
    assigned = [i for p in batch.proposals for i in p.member_ids]
    if (len(set(proposal_ids)) != len(proposal_ids) or len(set(assigned)) != len(assigned)
            or any(type(i) not in (int,str) or i not in ids for i in assigned)
            or not any(p == proposal for p in batch.proposals)):
        return None,None,"foreign_duplicate_or_mutated_proposal_membership"
    if any(not _valid_polygon(d.polygon) for d in batch.detections):
        return None,None,"invalid_raw_geometry"
    if any(x < 0 or y < 0 or x > image.width or y > image.height for d in batch.detections for x,y in d.polygon):
        return None,None,"clipped_raw_geometry"
    fresh = propose_automatic_units(batch.detections,image,max_detections=max(512,len(batch.detections)),
        complementary_ownership=batch.ownership_policy == COMPLEMENTARY_OWNERSHIP_POLICY,
        regroup_companions=batch.companion_policy == COMPANION_POLICY)
    if batch != fresh:
        return None,None,"automatic_proposal_or_structural_evidence_drift"
    sources = {d.source_id:d for d in batch.detections}
    reference = "".join(sources[i].text for i in proposal.main_ids)
    script,script_error = script_evidence(reference)
    digest = hashlib.sha256(json.dumps(asdict(proposal),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    provenance = AutomaticCropEvidence(proposal.proposal_id,proposal.page,proposal.member_ids,proposal.main_ids,
        proposal.crop_bounds,tuple(i for i in ids if i not in proposal.member_ids),batch.evidence_sha256,digest,
        script,proposal.structural_reasons,proposal.word_reasons,proposal.completeness)
    # Recheck the actual supplied final rectangle independently of crop_safe.
    conflicts = crop_conflicts(proposal.crop_bounds,proposal.member_ids,batch.detections)
    if conflicts or not proposal.crop_safe:
        return None,provenance,"unsafe_automatic_crop"
    if proposal.direction not in ("vertical","horizontal"):
        return None,provenance,"unknown_automatic_reading_orientation"
    if script_error:
        return None,provenance,script_error
    owners = {source_id:region_id for source_id,region_id,_ in batch.regions}
    if any(i not in owners for i in proposal.member_ids):
        return None,provenance,"unknown_automated_region_ownership"
    if len({owners[i] for i in proposal.member_ids}) != 1:
        return None,provenance,"conflicting_automated_region_ownership"
    blocked = {"ambiguous_annotation_parents","ambiguous_annotation_chain_or_role",
               "unresolved_annotation_relationship","mixed_region_ownership"}
    if blocked & set(proposal.structural_reasons):
        return None,provenance,"conflicting_automatic_relationship_or_ownership"
    return image.crop(proposal.crop_bounds),provenance,None


def recognize_automatic_proposal(result, proposal, batch, image, recognizer, *, original_image_bytes,
                                 model_identity, model_settings=None):
    """One primary attempt, including failures. Caller controls global budget."""
    if result.proposal_id != proposal.proposal_id:
        raise ValueError("result must bind the exact proposal")
    if result.attempts:
        return result
    if not isinstance(model_identity,str) or not model_identity.strip():
        raise ValueError("explicit model identity required")
    if model_settings is not None and not isinstance(model_settings,dict):
        raise ValueError("model_settings must be an explicit dictionary")
    settings = json.dumps(model_settings or {},sort_keys=True,ensure_ascii=False,allow_nan=False)
    try:
        crop,provenance,error = prepare_automatic_crop(proposal,batch,image,original_image_bytes=original_image_bytes)
    except (TypeError,ValueError,KeyError,OverflowError) as failure:
        crop,provenance,error = None,None,"invalid_automatic_evidence: " + str(failure)
    result.crop = provenance
    if error:
        result.status,result.reason = "skipped",error
        return result
    result.attempts = 1
    result.model_identity,result.model_settings_json = model_identity,settings
    result.status,result.reason = "in_progress","automatic_recognition_in_progress"
    try:
        text = recognizer(crop,source_proposal_id=proposal.proposal_id,
                          member_ids=proposal.member_ids,crop_bounds=provenance.bounds)
        if not isinstance(text,str):
            raise TypeError("automatic recognizer must return text only")
    except Exception as failure:
        result.status,result.reason = "error","automatic_recognition_failed"
        result.error = f"{type(failure).__name__}: {failure}"
        return result
    sources = {d.source_id:d for d in batch.detections}
    reference = "".join(sources[i].text for i in proposal.main_ids)
    spans = tuple((op,(i,j),(k,l),reference[i:j],text[k:l])
                  for op,i,j,k,l in SequenceMatcher(None,reference,text,autojunk=False).get_opcodes() if op != "equal")
    result.candidates += (AutomaticTextCandidate(text,reference,spans,provenance,"manga_ocr",model_identity,settings),)
    result.status,result.reason = "automated_unverified","automatic_candidate_unverified_not_truth"
    return result


def recognize_automatic_batch(batch, image, recognizer, *, original_image_bytes, model_identity,
                              model_settings=None, max_attempts=20, results=None):
    """Global page budget includes errors; retained results prevent primary retries."""
    if type(max_attempts) is not int or not 0 <= max_attempts <= 20:
        raise ValueError("max_attempts must be 0..20")
    if results is None:
        results = tuple(AutomaticRecognitionResult(p.proposal_id) for p in batch.proposals)
    else:
        results = tuple(results)
    if ([r.proposal_id for r in results] != [p.proposal_id for p in batch.proposals]
            or any(type(r.attempts) is not int or r.attempts not in (0,1) for r in results)
            or len({r.proposal_id for r in results}) != len(results)):
        raise ValueError("result state must bind unique exact proposals/attempts")
    spent = sum(r.attempts for r in results)
    if spent > 20 or any(r.candidates and r.attempts != 1 for r in results):
        raise ValueError("invalid retained automatic attempt state")
    for result,proposal in zip(results,batch.proposals):
        if result.attempts:
            continue
        if spent >= max_attempts:
            result.status,result.reason = "skipped","automatic_request_limit"
            continue
        recognize_automatic_proposal(result,proposal,batch,image,recognizer,
            original_image_bytes=original_image_bytes,model_identity=model_identity,model_settings=model_settings)
        spent += result.attempts
    return results
