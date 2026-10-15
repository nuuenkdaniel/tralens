"""Immutable, provider-neutral utterance previews. No OCR or rendering side effects."""
import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def revision(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("nonfinite JSON number")
        return parsed
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


@dataclass(frozen=True)
class Reading:
    text: str
    origin: str
    evidence_json: str

    def __post_init__(self):
        if not isinstance(self.text, str) or not isinstance(self.origin, str) or not self.origin:
            raise ValueError("reading needs text and origin")
        if canonical(strict_json(self.evidence_json)) != self.evidence_json:
            raise ValueError("reading evidence must be canonical JSON")

    @property
    def revision(self):
        return revision(asdict(self))


@dataclass(frozen=True)
class Utterance:
    unit_id: str
    page_json: str
    member_ids: tuple
    order: tuple
    readings: tuple
    selected_revision: str
    evidence_revision: str
    structural_reasons: tuple = ()
    lexical_reasons: tuple = ()
    blocked_reasons: tuple = ()
    completeness: str = "unverified_full_utterance"

    def __post_init__(self):
        if (not isinstance(self.unit_id, str) or not self.unit_id
                or canonical(strict_json(self.page_json)) != self.page_json):
            raise ValueError("unit/page identity required")
        for field in (self.member_ids, self.order, self.readings, self.structural_reasons,
                      self.lexical_reasons, self.blocked_reasons):
            if type(field) is not tuple:
                raise ValueError("immutable tuples required")
        if any(not isinstance(s, str) for values in (self.structural_reasons,
                self.lexical_reasons, self.blocked_reasons) for s in values):
            raise ValueError("uncertainty reasons must be strings")
        if not isinstance(self.evidence_revision, str) or not isinstance(self.completeness, str):
            raise ValueError("immutable revision and completeness required")
        if (not self.member_ids or any(type(i) not in (int, str) or i == "" for i in self.member_ids)
                or len(set(self.member_ids)) != len(self.member_ids)
                or len(set(self.order)) != len(self.order) or not self.order
                or not set(self.order) <= set(self.member_ids)):
            raise ValueError("invalid membership/reading order")
        if not self.readings or any(not isinstance(r, Reading) for r in self.readings):
            raise ValueError("immutable source readings required")
        if len({r.revision for r in self.readings}) != len(self.readings):
            raise ValueError("duplicate reading revisions")
        if self.selected_revision not in {r.revision for r in self.readings}:
            raise ValueError("selected reading must bind an exact candidate revision")

    @property
    def selected(self):
        return next(r for r in self.readings if r.revision == self.selected_revision)

    @property
    def disposition(self):
        return "blocked" if self.blocked_reasons or not self.selected.text.strip() else "preview"

    @property
    def blocking_reasons(self):
        return tuple(dict.fromkeys(self.blocked_reasons +
            (("blank_selected_reading",) if not self.selected.text.strip() else ())))

    @property
    def source_uncertain(self):
        return (len({r.text for r in self.readings}) > 1 or bool(set(self.lexical_reasons) -
                {"wording_not_human_checked"}))

    @property
    def revision(self):
        return revision(asdict(self))


@dataclass(frozen=True)
class Style:
    source_language: str = "Japanese"
    target_language: str = "English"
    honorific_policy: str = "preserve"
    glossary: tuple = ()
    instructions: str = "Natural manga dialogue; preserve ambiguity and register."

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.source_language, self.target_language, self.honorific_policy, self.instructions)):
            raise ValueError("explicit languages and style required")
        if (type(self.glossary) is not tuple or any(type(p) is not tuple or len(p) != 2
                or any(not isinstance(s, str) or not s for s in p) for p in self.glossary)):
            raise ValueError("glossary must be immutable source/target pairs")


@dataclass(frozen=True)
class TranslationRequest:
    units: tuple
    target_ids: tuple
    context_ids: tuple = ()
    style: Style = Style()
    mode: str = "A"
    page_order: tuple = ()
    page_order_status: str = "unknown"
    image_evidence_json: str = ""
    contract_version: str = "utterance_preview_v2"

    def __post_init__(self):
        if self.contract_version != "utterance_preview_v2":
            raise ValueError("unsupported translation request contract version")
        if any(type(v) is not tuple for v in (self.units, self.target_ids, self.context_ids)):
            raise ValueError("request collections must be immutable")
        if not isinstance(self.style, Style) or self.mode not in ("A", "B", "C"):
            raise ValueError("invalid style or experiment mode")
        if any(not isinstance(u, Utterance) for u in self.units):
            raise ValueError("immutable utterances required")
        if any(not isinstance(i, str) or not i for i in self.target_ids + self.context_ids):
            raise ValueError("target/context IDs must be nonempty strings")
        ids = tuple(u.unit_id for u in self.units)
        requested = self.target_ids + self.context_ids
        if (not self.target_ids or len(set(ids)) != len(ids) or len(set(requested)) != len(requested)
                or set(requested) != set(ids)):
            raise ValueError("target/context IDs must partition exact unique supplied units")
        if len({u.page_json for u in self.units}) != 1:
            raise ValueError("one bound page per request")
        if self.mode == "A" and self.context_ids:
            raise ValueError("A is utterance text only")
        if (type(self.page_order) is not tuple or any(not isinstance(i, str) for i in self.page_order)
                or len(set(self.page_order)) != len(self.page_order)
                or (self.page_order and not set(ids) <= set(self.page_order))
                or not isinstance(self.page_order_status, str)):
            raise ValueError("invalid explicit page order")
        if self.image_evidence_json:
            evidence = strict_json(self.image_evidence_json)
            page = strict_json(self.units[0].page_json)
            if (type(evidence) is not dict or any(not isinstance(evidence.get(key), str)
                    or len(evidence[key]) != 64 or any(c not in "0123456789abcdef" for c in evidence[key])
                    for key in ("original_sha256", "original_pixels_sha256", "presentation_sha256"))):
                raise ValueError("image evidence requires original/pixel/presentation SHA256")
            if (self.mode != "C" or canonical(evidence) != self.image_evidence_json
                    or evidence["original_sha256"] != page["image_sha256"]
                    or evidence["original_pixels_sha256"] != page["pixels_sha256"]):
                raise ValueError("image evidence must bind the exact original page in C")

    @property
    def revision(self):
        return revision(asdict(self))


@dataclass(frozen=True)
class UnitOutcome:
    unit_id: str
    disposition: str
    translation: str
    reasons: tuple
    source_interpretation: str
    reading_revision: str = ""

    def __post_init__(self):
        if (not isinstance(self.unit_id, str) or not self.unit_id or not isinstance(self.translation, str)
                or type(self.reasons) is not tuple or len(self.reasons) > 16
                or any(not isinstance(s, str) or not s.strip() or len(s) > 512 for s in self.reasons)
                or self.disposition not in ("preview", "review", "abstained", "blocked")
                or self.source_interpretation not in ("unverified", "ambiguous", "unreadable")
                or not isinstance(self.reading_revision, str)):
            raise ValueError("invalid typed unit outcome")
        if self.disposition in ("abstained", "blocked"):
            if self.translation != "" or not self.reasons:
                raise ValueError("abstained/blocked units need reasons and no translation")
        elif not self.translation.strip():
            raise ValueError("preview/review units need nonempty translation")
        if self.disposition == "review" and not self.reasons:
            raise ValueError("review disposition requires reasons")
        if self.source_interpretation == "unreadable" and self.disposition not in ("abstained", "blocked"):
            raise ValueError("unreadable sources cannot carry translations")


@dataclass(frozen=True)
class TranslationResult:
    request_revision: str
    status: str
    translations: tuple = ()
    failure: str = ""
    detail: str = ""
    attempts: int = 0
    provider: str = ""
    model: str = ""
    profile: str = ""
    generation_seconds: float = 0.0
    preview_only: bool = True
    execution_metadata_json: str = "{}"
    outcomes: tuple = ()
    review_needed: bool = True

    def __post_init__(self):
        if (type(strict_json(self.execution_metadata_json)) is not dict
                or canonical(strict_json(self.execution_metadata_json)) != self.execution_metadata_json):
            raise ValueError("execution metadata must be canonical immutable JSON")
        if (type(self.translations) is not tuple or any(type(p) is not tuple or len(p) != 2
                or any(not isinstance(s, str) for s in p) for p in self.translations)):
            raise ValueError("immutable ID/translation pairs required")
        if (type(self.outcomes) is not tuple or any(not isinstance(o, UnitOutcome) for o in self.outcomes)
                or len({o.unit_id for o in self.outcomes}) != len(self.outcomes)
                or type(self.review_needed) is not bool):
            raise ValueError("immutable unique typed outcomes required")
        if self.status not in ("preview", "abstained", "blocked", "failed", "cancelled") or self.preview_only is not True:
            raise ValueError("only explicit translation preview dispositions are supported")
        if self.status != "preview" and self.translations:
            raise ValueError("failed/blocked/cancelled results cannot carry output")
        if self.outcomes and self.translations != tuple((o.unit_id, o.translation) for o in self.outcomes if o.translation):
            raise ValueError("translation pairs must match explicit outcomes")
        if self.status == "preview" and (not self.outcomes or not self.translations):
            raise ValueError("preview results require typed outcomes and nonempty output")
        if self.status == "abstained" and (not self.outcomes or any(o.disposition != "abstained" for o in self.outcomes)):
            raise ValueError("abstained results require explicit abstention for every unit")
        if not self.review_needed and any(o.disposition != "preview" for o in self.outcomes):
            raise ValueError("review/abstention outcomes require a visible review flag")

    def assert_current(self, request):
        if type(request) is not TranslationRequest:
            raise TypeError("strict results require a strict TranslationRequest")
        if self.request_revision != request.revision:
            raise ValueError("stale translation result: request/source/style/image revisions changed")
        if self.outcomes:
            sources = {u.unit_id: u for u in request.units}
            if (set(o.unit_id for o in self.outcomes) != set(request.target_ids)
                    or any(o.reading_revision != sources[o.unit_id].selected_revision for o in self.outcomes)):
                raise ValueError("stale or foreign outcome source binding")


# Generic uncertainty is deliberately absent: these are known structural faults.
STRUCTURAL_BLOCKS = frozenset({
    "unknown_region_ownership", "mixed_region_ownership", "unknown_reading_orientation",
    "crop_isolation_unresolved", "ambiguous_annotation_parents", "ambiguous_annotation_chain_or_role",
    "unresolved_annotation_relationship",
    "annotation_role_unresolved_not_independent_text_proof",
})


def automatic_units(batch, candidates=()):
    """Snapshot automatic batch/candidates without selecting OCR truth or mutating inputs.

    Accept dataclasses or their saved JSON representation. Raw main text is the
    deterministic default; candidate alternatives remain explicit, selectable revisions.
    This checks saved evidence consistency, not original pixels or semantic quality.
    """
    batch = strict_json(canonical(asdict(batch) if is_dataclass(batch) else batch))
    candidates = [strict_json(canonical(asdict(c) if is_dataclass(c) else c)) for c in candidates]
    batch_revision = revision(batch)
    detections = batch["detections"]
    ids = [d["source_id"] for d in detections]
    if any(type(i) not in (int, str) or i == "" for i in ids):
        raise ValueError("invalid raw source IDs")
    proposals = batch["proposals"]
    assigned = [i for p in proposals for i in p["member_ids"]]
    expected = hashlib.sha256(json.dumps(sorted(detections, key=lambda d:
        (type(d["source_id"]).__name__, str(d["source_id"]))), sort_keys=True,
        ensure_ascii=False).encode()).hexdigest()
    if (len(set(ids)) != len(ids) or len(set(assigned)) != len(assigned)
            or not set(assigned) <= set(ids) or batch["evidence_sha256"] != expected
            or any(d["page"] != batch["page"] for d in detections)
            or len({p["proposal_id"] for p in proposals}) != len(proposals)):
        raise ValueError("duplicate/foreign IDs or raw evidence drift")
    by_id = {d["source_id"]: d for d in detections}
    by_proposal = {p["proposal_id"]: p for p in proposals}
    for c in candidates:
        crop = c["crop"]
        p = by_proposal.get(crop["proposal_id"])
        if (p is None or crop["page"] != batch["page"] or crop["member_ids"] != p["member_ids"]
                or crop["main_ids"] != p["main_ids"] or crop["bounds"] != p["crop_bounds"]
                or crop["batch_evidence_sha256"] != expected
                or crop["proposal_sha256"] != hashlib.sha256(json.dumps(p, sort_keys=True,
                    ensure_ascii=False).encode()).hexdigest()
                or c["reference_main_text"] != "".join(by_id[i]["text"] for i in p["main_ids"])):
            raise ValueError("candidate provenance drift")
    result = []
    if len({row[0] for row in batch["regions"]}) != len(batch["regions"]):
        raise ValueError("duplicate/conflicting region ownership records")
    owners = {row[0]: row[1] for row in batch["regions"]}
    for p in proposals:
        if type(p["crop_safe"]) is not bool:
            raise ValueError("crop safety must be a boolean")
        if p["page"] != batch["page"]:
            raise ValueError("foreign proposal page")
        main = tuple(p["main_ids"])
        members = tuple(p["member_ids"])
        if (not main or len(set(main)) != len(main) or not set(main) <= set(members)
                or set(p["annotation_ids"]) & set(main)
                or set(main) | set(p["annotation_ids"]) != set(members)):
            raise ValueError("invalid main/annotation membership")
        raw = Reading("".join(by_id[i]["text"] for i in main), "raw_main_reading",
                      canonical([by_id[i] for i in main]))
        readings = [raw]
        for c in candidates:
            if c["crop"]["proposal_id"] == p["proposal_id"]:
                reading = Reading(c["text"], c["engine"], canonical(c))
                if reading not in readings:
                    readings.append(reading)
        structural = tuple(p["structural_reasons"])
        blocked = list(sorted(STRUCTURAL_BLOCKS.intersection(structural)))
        if any(i not in owners for i in members):
            blocked.append("unknown_region_ownership")
        if len({owners[i] for i in members if i in owners}) > 1:
            blocked.append("mixed_region_ownership")
        if not p["crop_safe"] or p.get("crop_conflicts"):
            blocked.append("unsafe_crop")
        if p["direction"] not in ("vertical", "horizontal"):
            blocked.append("unknown_reading_orientation")
        if any(not by_id[i]["text"].strip() for i in main):
            blocked.append("blank_main_reading")
        result.append(Utterance(p["proposal_id"], canonical(batch["page"]), tuple(p["member_ids"]),
            main, tuple(readings), raw.revision, revision({"batch_revision": batch_revision, "proposal": p}),
            structural, tuple(p["word_reasons"]), tuple(dict.fromkeys(blocked)), p["completeness"]))
    return tuple(result)


FRAGMENT_POLICY_REVISION = "unverified_raw_singleton_fragment_v1"
FRAGMENT_WAIVABLE_BLOCKS = frozenset({"unknown_region_ownership", "unsafe_crop", "crop_isolation_unresolved"})


def fragment_rejection_reasons(unit):
    """Permission for an unverified text input, never evidence of readability/ownership."""
    if type(unit) is not Utterance:
        return ("requires_original_utterance",)
    if unit.disposition != "blocked":
        return ("strict_preview_available",)
    reasons = ["non_waivable:" + r for r in unit.blocking_reasons if r not in FRAGMENT_WAIVABLE_BLOCKS]
    permitted_structure = FRAGMENT_WAIVABLE_BLOCKS | {"automatic_not_human_reviewed", "complementary_component_ownership"}
    reasons.extend("non_waivable_structure:" + r for r in unit.structural_reasons if r not in permitted_structure)
    # Existing combined-unit contracts do not certify independent member roles/order.
    # Do not split, infer or concatenate them in this first singleton-only policy.
    if len(unit.member_ids) != 1 or unit.order != unit.member_ids:
        reasons.append("requires_independent_singleton_raw_input")
        return tuple(dict.fromkeys(reasons))
    if unit.selected.origin != "raw_main_reading":
        reasons.append("selected_reading_is_not_raw")
    try:
        raw = strict_json(unit.selected.evidence_json)
        if type(raw) is not list or len(raw) != 1 or type(raw[0]) is not dict:
            raise ValueError("missing singleton evidence")
        raw = raw[0]
        required = {"source_id", "page", "text", "confidence", "polygon", "status", "reason", "selected_text", "metadata_json"}
        if set(raw) != required:
            raise ValueError("missing raw fields")
        page = raw["page"]
        if (type(raw["source_id"]) not in (int,str) or (raw["source_id"],) != unit.member_ids
                or canonical(page) != unit.page_json or raw["text"] != unit.selected.text
                or type(raw["text"]) is not str or type(raw["selected_text"]) is not str
                or type(raw["reason"]) is not str or type(page) is not dict
                or type(page.get("page_id")) is not str or not page["page_id"]
                or type(page.get("image_path")) is not str or not page["image_path"]):
            raise ValueError("foreign raw source")
        for key in ("image_sha256", "pixels_sha256", "comparison_sha256"):
            value = page.get(key)
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("missing page provenance")
        polygon = raw["polygon"]
        if (type(polygon) is not list or not 3 <= len(polygon) <= 64 or any(type(p) is not list or len(p) != 2
                or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in p) for p in polygon)
                or abs(sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(polygon,polygon[1:]+polygon[:1]))) < 1e-7):
            raise ValueError("invalid raw geometry")
        if type(raw["confidence"]) not in (int,float) or not math.isfinite(raw["confidence"]):
            raise ValueError("invalid raw score")
        if raw["status"] not in ("accepted", "review"):
            reasons.append("raw_reading_rejected_or_unknown_status")
        metadata = strict_json(raw["metadata_json"])
        if type(metadata) is not dict:
            raise ValueError("invalid raw metadata")
        association = metadata.get("paddle_association")
        if association is not None and (type(association) is not dict or association.get("status") != "matched"
                or type(association.get("issues",[])) is not list or association.get("issues")):
            reasons.append("unresolved_raw_source_association")
        if raw["reason"] == "unresolved_paddle_association":
            reasons.append("unresolved_raw_source_association")
    except (ValueError, TypeError, KeyError, IndexError):
        reasons.append("missing_or_inconsistent_raw_evidence")
    return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class FragmentInput:
    original: Utterance
    original_unit_revision: str
    source_reading_revision: str
    source_ids: tuple
    policy_revision: str = FRAGMENT_POLICY_REVISION

    def __post_init__(self):
        if self.policy_revision != FRAGMENT_POLICY_REVISION:
            raise ValueError("unsupported fragment policy")
        rejected = fragment_rejection_reasons(self.original)
        if rejected:
            raise ValueError("fragment derivation rejected: " + ", ".join(rejected))
        self.assert_current(self.original)

    def assert_current(self, original):
        if (type(original) is not Utterance or type(self.source_ids) is not tuple
                or any(type(i) not in (int,str) or i == "" for i in self.source_ids)
                or self.original_unit_revision != original.revision
                or self.source_reading_revision != original.selected_revision or self.source_ids != original.member_ids):
            raise ValueError("stale fragment original/reading/source binding")

    @property
    def provenance(self):
        return {"kind": "unverified_text_fragment", "policy_revision": self.policy_revision,
                "original_unit_id": self.original.unit_id, "original_unit_revision": self.original_unit_revision,
                "source_reading_revision": self.source_reading_revision, "source_ids": self.source_ids,
                "original_blocking_reasons": self.original.blocking_reasons,
                "permissions": "text only; no ownership, crop, image, recognition, rendering or erasure permission"}

    @property
    def revision(self):
        return revision(self.provenance)

    @property
    def unit_id(self):
        return "fragment-" + self.revision

    @property
    def selected(self):
        return Reading(self.original.selected.text, "unverified_raw_detection_fragment", canonical(self.provenance))

    @property
    def selected_revision(self):
        return self.selected.revision

    @property
    def readings(self):
        return (self.selected,)

    @property
    def source_uncertain(self):
        return True

    @property
    def blocking_reasons(self):
        return self.original.blocking_reasons

    @property
    def lexical_reasons(self):
        return self.original.lexical_reasons


def fragment_plan(unit, *, allow_fragment_previews=False):
    if type(allow_fragment_previews) is not bool:
        raise ValueError("allow_fragment_previews must be a boolean")
    if not allow_fragment_previews:
        return (), ("fragment_previews_disabled",)
    rejected = fragment_rejection_reasons(unit)
    if rejected:
        return (), rejected
    return (FragmentInput(unit, unit.revision, unit.selected_revision, unit.member_ids),), ()


@dataclass(frozen=True)
class FragmentRequest:
    fragment: FragmentInput
    style: Style = Style()
    contract_version: str = "unverified_fragment_request_v1"

    def __post_init__(self):
        if (type(self.fragment) is not FragmentInput or type(self.style) is not Style
                or self.contract_version != "unverified_fragment_request_v1"):
            raise ValueError("typed immutable fragment request required")
        self.fragment.assert_current(self.fragment.original)

    @property
    def revision(self):
        return revision(asdict(self))

    @property
    def units(self):
        return (self.fragment,)

    @property
    def target_ids(self):
        return (self.fragment.unit_id,)


@dataclass(frozen=True)
class FragmentResult:
    request_revision: str
    input_revision: str
    source_binding_json: str
    status: str
    fragments: tuple = ()
    failure: str = ""
    detail: str = ""
    attempts: int = 0
    provider: str = ""
    model: str = ""
    profile: str = ""
    generation_seconds: float = 0.0
    execution_metadata_json: str = "{}"
    outcomes: tuple = ()
    review_needed: bool = True
    preview_only: bool = True
    output_kind: str = "unverified_text_fragment"

    def __post_init__(self):
        for value in (self.source_binding_json,self.execution_metadata_json):
            if type(strict_json(value)) is not dict or canonical(strict_json(value)) != value:
                raise ValueError("canonical fragment provenance/metadata required")
        if (self.status not in ("fragment_preview","fragment_abstained","fragment_failed","fragment_cancelled","fragment_blocked")
                or self.review_needed is not True or self.preview_only is not True or self.output_kind != "unverified_text_fragment"
                or type(self.fragments) is not tuple or any(type(p) is not tuple or len(p) != 2 or
                    any(type(s) is not str for s in p) for p in self.fragments)
                or type(self.outcomes) is not tuple or any(type(o) is not UnitOutcome for o in self.outcomes)):
            raise ValueError("explicit review-only fragment result required")
        if self.fragments != tuple((o.unit_id,o.translation) for o in self.outcomes if o.translation):
            raise ValueError("fragment pairs must match outcomes")
        if self.status == "fragment_preview":
            if not self.fragments or any(o.disposition != "review" for o in self.outcomes):
                raise ValueError("fragment output must require review")
        elif self.fragments:
            raise ValueError("non-preview fragment result cannot carry output")
        if self.status == "fragment_abstained" and (not self.outcomes or any(o.disposition != "abstained" for o in self.outcomes)):
            raise ValueError("explicit fragment abstention required")

    def assert_current(self, request, original):
        if type(request) is not FragmentRequest:
            raise TypeError("fragment result needs FragmentRequest and current original")
        request.fragment.assert_current(original)
        if (self.request_revision != request.revision or self.input_revision != request.fragment.revision
                or self.source_binding_json != canonical(request.fragment.provenance)):
            raise ValueError("stale fragment request/input binding")
        if self.outcomes and (tuple(o.unit_id for o in self.outcomes) != request.target_ids
                or any(o.reading_revision != request.fragment.selected_revision for o in self.outcomes)):
            raise ValueError("stale fragment output reading binding")
