"""Optional reviewed recognition units; no OCR initialization or diagnostics imports."""
import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from enum import Enum

from PIL import Image


class UnitStatus(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


class UnitRelationship(str, Enum):
    FURIGANA = "furigana"
    COMPANION = "same_text_block"
    DUPLICATE = "duplicate"
    INDEPENDENT = "independent"
    UNCERTAIN = "uncertain"


class MemberRole(str, Enum):
    MAIN = "main"
    COMPANION = "companion"
    ANNOTATION = "reading_annotation"
    DUPLICATE = "duplicate"


class WritingDirection(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    UNKNOWN = "unknown"


class ReadingGranularity(str, Enum):
    COLUMN = "column"
    BLOCK = "block"
    UNKNOWN = "unknown"


def _id(value):
    if type(value) not in (int, str) or value == "":
        raise ValueError("source IDs must be nonempty strings or integers")
    return value


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def _reviewed(flag, reviewer):
    return flag is True and isinstance(reviewer, str) and bool(reviewer.strip())


def image_fingerprint(image):
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("recognition units require the original RGB PIL image")
    prefix = f"RGB:{image.width}:{image.height}:".encode()
    return hashlib.sha256(prefix + image.tobytes()).hexdigest()


@dataclass(frozen=True)
class PageProvenance:
    page_id: str
    image_path: str
    image_sha256: str
    comparison_sha256: str
    pixels_sha256: str

    def __post_init__(self):
        if not self.page_id or not self.image_path:
            raise ValueError("page ID and original image path are required")
        for digest in (self.image_sha256, self.comparison_sha256, self.pixels_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("page provenance requires SHA256 digests")


@dataclass(frozen=True)
class RawDetection:
    source_id: object
    page: PageProvenance
    text: str
    confidence: float
    polygon: tuple
    status: str
    reason: str = ""
    selected_text: str = ""
    metadata_json: str = "{}"

    def __post_init__(self):
        _id(self.source_id)
        object.__setattr__(self, "polygon", tuple(tuple(point) for point in self.polygon))
        object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.text, str) or not isinstance(self.selected_text, str):
            raise ValueError("raw text must be a string")


def snapshot_detections(boxes, page):
    """Copy original Paddle evidence and current status; never retain mutable arrays."""
    def get(box, *names, default=None):
        for name in names:
            if isinstance(box, dict) and name in box:
                return box[name]
            if not isinstance(box, dict) and hasattr(box, name):
                return getattr(box, name)
        return default
    snapshots = []
    for box in boxes:
        poly = get(box, "paddle_poly", "polygon", "poly")
        if hasattr(poly, "tolist"):
            poly = poly.tolist()
        status = get(box, "status", default="review")
        status = status.value if isinstance(status, Enum) else status
        snapshots.append(RawDetection(
            get(box, "paddle_box_id", "id", "box_id"), page,
            get(box, "paddle_text", "paddle", "text", default=""),
            get(box, "paddle_confidence", "score", "confidence", default=0.), poly, status,
            get(box, "reason", default=""),
            get(box, "selected_text", "selected", "text", default=""),
            json.dumps(get(box, "ocr_metadata", "metadata", default={}), ensure_ascii=False)))
    ids = [snapshot.source_id for snapshot in snapshots]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate raw source IDs")
    return tuple(snapshots)


@dataclass(frozen=True)
class ReviewedRelationship:
    source_id: object
    parent_id: object
    relationship: UnitRelationship
    page: PageProvenance
    reviewed: bool = False
    reviewer: str = ""

    def __post_init__(self):
        _id(self.source_id)
        _id(self.parent_id)
        object.__setattr__(self, "relationship", UnitRelationship(self.relationship))


@dataclass(frozen=True)
class ReviewedOwnership:
    source_id: object
    bubble_id: str
    page: PageProvenance
    reviewed: bool = False
    reviewer: str = ""


@dataclass(frozen=True)
class UnitRequest:
    main_ids: tuple
    direction: WritingDirection
    granularity: ReadingGranularity
    reviewed: bool = False
    order_reviewed: bool = False
    granularity_reviewed: bool = False
    reviewer: str = ""

    def __post_init__(self):
        object.__setattr__(self, "main_ids", tuple(self.main_ids))
        for source_id in self.main_ids:
            _id(source_id)
        object.__setattr__(self, "direction", WritingDirection(self.direction))
        object.__setattr__(self, "granularity", ReadingGranularity(self.granularity))


@dataclass(frozen=True)
class UnitPlan:
    unit_id: str
    page: PageProvenance
    member_ids: tuple
    main_ids: tuple
    roles: tuple
    annotation_parents: tuple
    duplicate_parents: tuple
    bubble_id: str
    direction: WritingDirection
    granularity: ReadingGranularity
    comparison_valid: bool
    reference_reading: object
    errors: tuple


@dataclass(frozen=True)
class ReviewedUtteranceExtent:
    """Human-approved recognition rectangle, NOT a detection or render region."""
    unit_id: str
    member_ids: tuple
    page: PageProvenance
    bounds: tuple
    source: str
    reviewed: bool = False
    reviewer: str = ""

    def __post_init__(self):
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        object.__setattr__(self, "bounds", tuple(self.bounds))


@dataclass(frozen=True)
class CropProvenance:
    page: PageProvenance
    unit_id: str
    member_ids: tuple
    bounds: tuple
    padding: int
    alternate: bool
    checked_neighbor_ids: tuple
    detection_union_bounds: tuple = ()
    reviewed_extent: object = None
    comparison_granularity: str = "joined_main_readings"


@dataclass(frozen=True)
class UnitCandidate:
    text: str
    engine: str
    reference_reading: object
    status: UnitStatus
    reason: str
    differing_spans: tuple
    crop: CropProvenance


@dataclass
class RecognitionUnit:
    plan: UnitPlan
    status: UnitStatus = UnitStatus.REVIEW
    reason: str = "unit_candidate_not_requested"
    candidates: tuple = ()
    attempts: int = 0
    alternate_attempts: int = 0
    crop: object = None

    @property
    def approved(self):
        return not self.plan.errors


@dataclass(frozen=True)
class UnitBatch:
    detections: tuple
    units: tuple
    fallback_source_ids: tuple
    ownerships: tuple
    utterance_extents: tuple = ()


def plan_recognition_units(detections, relationships, requests, ownerships, *, utterance_extents=()):
    """Explicit review only; no small-box, confidence or model-label inference."""
    detections, relationships, ownerships = tuple(detections), tuple(relationships), tuple(ownerships)
    if not detections:
        raise ValueError("a unit page requires raw detections")
    page = detections[0].page
    sources = {d.source_id: d for d in detections}
    if len(sources) != len(detections) or any(d.page != page for d in detections):
        raise ValueError("raw detections must have unique IDs and one page provenance")
    global_errors, parents, seen = [], {}, set()
    for relation in relationships:
        if (type(relation.source_id) not in (int, str) or type(relation.parent_id) not in (int, str)
                or relation.source_id not in sources or relation.parent_id not in sources):
            global_errors.append("foreign_relationship_id")
        if relation.page != page:
            global_errors.append("relationship_page_mismatch")
        if not _reviewed(relation.reviewed, relation.reviewer):
            global_errors.append("unreviewed_relationship")
        key = (relation.source_id, relation.parent_id, relation.relationship)
        if key in seen:
            global_errors.append("duplicate_relationship")
        seen.add(key)
        if relation.relationship == UnitRelationship.UNCERTAIN:
            global_errors.append("uncertain_relationship")
        elif relation.relationship != UnitRelationship.INDEPENDENT:
            if relation.source_id in parents:
                global_errors.append("multiple_or_conflicting_parents")
            parents[relation.source_id] = relation.parent_id
    for child in parents:
        trail, current = set(), child
        while current in parents:
            if current in trail:
                global_errors.append("relationship_cycle")
                break
            trail.add(current)
            current = parents[current]
    annotations_as_parents = {r.source_id for r in relationships
                             if r.relationship in (UnitRelationship.FURIGANA, UnitRelationship.DUPLICATE)}
    if any(r.parent_id in annotations_as_parents for r in relationships
           if r.relationship in (UnitRelationship.FURIGANA, UnitRelationship.DUPLICATE)):
        global_errors.append("annotation_parent_is_not_main")
    owners = {}
    for ownership in ownerships:
        if (type(ownership.source_id) not in (int, str) or ownership.source_id not in sources
                or ownership.page != page or not _reviewed(ownership.reviewed, ownership.reviewer)
                or not isinstance(ownership.bubble_id, str) or not ownership.bubble_id.strip()):
            global_errors.append("invalid_or_unreviewed_ownership")
        if ownership.source_id in owners:
            global_errors.append("duplicate_ownership")
        owners[ownership.source_id] = ownership.bubble_id
    units, used = [], set()
    for request in requests:
        errors = list(global_errors)
        main = request.main_ids
        if (not main or any(type(i) not in (int, str) or i not in sources for i in main)
                or len(set(main)) != len(main)):
            errors.append("invalid_main_ids")
        if not _reviewed(request.reviewed, request.reviewer):
            errors.append("unreviewed_unit")
        if request.order_reviewed is not True or request.direction == WritingDirection.UNKNOWN:
            errors.append("unreviewed_or_unknown_reading_order")
        annotations, duplicates, companions = [], [], []
        for relation in relationships:
            if (relation.source_id in main and relation.relationship
                    in (UnitRelationship.FURIGANA, UnitRelationship.DUPLICATE)):
                errors.append("main_also_annotation_or_duplicate")
            if (relation.relationship == UnitRelationship.COMPANION
                    and (relation.source_id in main or relation.parent_id in main)
                    and not (relation.source_id in main and relation.parent_id in main)):
                errors.append("companion_missing_from_ordered_main_ids")
            if relation.parent_id not in main:
                continue
            if relation.relationship == UnitRelationship.FURIGANA:
                annotations.append((relation.source_id, relation.parent_id))
            elif relation.relationship == UnitRelationship.DUPLICATE:
                duplicates.append((relation.source_id, relation.parent_id))
            elif relation.relationship == UnitRelationship.COMPANION:
                if relation.source_id not in main:
                    errors.append("companion_missing_from_ordered_main_ids")
                companions.append((relation.source_id, relation.parent_id))
            elif relation.relationship == UnitRelationship.INDEPENDENT and relation.source_id in main:
                errors.append("independent_unit_members")
        key = lambda pair: (type(pair[0]).__name__, str(pair[0]), type(pair[1]).__name__, str(pair[1]))
        annotations.sort(key=key)
        duplicates.sort(key=key)
        companions.sort(key=key)
        attached = tuple(sorted({child for child, _ in annotations + duplicates}, key=lambda x: (type(x).__name__, str(x))))
        if any(child in main for child in attached):
            errors.append("main_also_annotation_or_duplicate")
        members = tuple(dict.fromkeys(main + attached))
        if len(main) > 1:
            connected = {main[0]}
            for _ in main:
                for child, parent in companions:
                    if child in connected or parent in connected:
                        connected.update((child, parent))
            if set(main) - connected:
                errors.append("unreviewed_companion_connection")
        bubble_ids = {owners.get(i) for i in members}
        bubble = next(iter(bubble_ids)) if len(bubble_ids) == 1 else None
        if bubble is None:
            errors.append("mixed_or_unknown_bubble_ownership")
        if any(i in used for i in members):
            errors.append("member_in_multiple_units")
        comparable = (request.granularity_reviewed is True and request.order_reviewed is True
                      and request.direction != WritingDirection.UNKNOWN
                      and request.granularity != ReadingGranularity.UNKNOWN
                      and (request.granularity == ReadingGranularity.BLOCK or len(main) == 1)
                      and all(i in sources and normalized(sources[i].text) for i in main))
        reference = "".join(sources[i].text for i in main) if comparable else None
        roles = tuple((i, MemberRole.MAIN if index == 0 else MemberRole.COMPANION)
                      for index, i in enumerate(main))
        roles += tuple((i, MemberRole.ANNOTATION) for i, _ in annotations)
        roles += tuple((i, MemberRole.DUPLICATE) for i, _ in duplicates)
        identity = {"page": asdict(page), "main_ids": main, "members": members,
                    "annotations": annotations, "duplicates": duplicates, "companions": companions,
                    "direction": request.direction, "granularity": request.granularity, "bubble": bubble}
        unit_id = "unit-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        plan = UnitPlan(unit_id, page, members, main, roles, tuple(annotations), tuple(duplicates),
                        bubble, request.direction, request.granularity, comparable, reference,
                        tuple(dict.fromkeys(errors)))
        unit = RecognitionUnit(plan, reason=";".join(plan.errors) if plan.errors else "unit_candidate_not_requested")
        units.append(unit)
        if unit.approved:
            used.update(members)
    # A later overlapping declaration invalidates ALL affected units, not just
    # the last one. No partial promotion from conflicting review records.
    counts = {}
    for unit in units:
        for member in unit.plan.member_ids:
            counts[member] = counts.get(member, 0) + 1
    for index, unit in enumerate(units):
        if any(counts[i] > 1 for i in unit.plan.member_ids):
            errors = tuple(dict.fromkeys(unit.plan.errors + ("member_in_multiple_units",)))
            plan = replace(unit.plan, errors=errors)
            units[index] = RecognitionUnit(plan, reason=";".join(errors))
    # Unit translation/erasure has no contract yet. Even an approved candidate
    # must not silently suppress any original per-box processing path.
    extents = tuple(utterance_extents)
    # A foreign/stale binding cannot silently fall back to a legacy crop. An
    # ambiguous declaration blocks the page's units rather than being ignored.
    foreign_extent = any(not any(e.unit_id == u.plan.unit_id for u in units) for e in extents)
    for unit in units:
        matching = [e for e in extents if e.unit_id == unit.plan.unit_id]
        errors = list(unit.plan.errors)
        if foreign_extent:
            errors.append("foreign_utterance_extent_unit")
        if len(matching) > 1:
            errors.append("ambiguous_utterance_extent")
        for extent in matching:
            if (extent.page != page or extent.member_ids != unit.plan.member_ids
                    or any(type(i) not in (int, str) for i in extent.member_ids)):
                errors.append("utterance_extent_binding_mismatch")
            if (not _reviewed(extent.reviewed, extent.reviewer)
                    or not isinstance(extent.source, str) or not extent.source.strip()):
                errors.append("unreviewed_utterance_extent")
        if errors:
            unit.plan = replace(unit.plan, errors=tuple(dict.fromkeys(errors)))
            unit.reason = ";".join(unit.plan.errors)
    return UnitBatch(detections, tuple(units), tuple(d.source_id for d in detections), ownerships, extents)


def _area(poly):
    return sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(poly, poly[1:]+poly[:1])) / 2 if len(poly) >= 3 else 0.


def _cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _valid_polygon(poly):
    if (len(poly) < 3 or any(len(p) != 2 or any(type(v) not in (int, float)
                                             or not math.isfinite(v) for v in p) for p in poly)):
        return False
    area = _area(poly)
    if not math.isfinite(area) or abs(area) <= 1e-7:
        return False
    sign = 1 if area > 0 else -1
    for a, b in zip(poly, poly[1:]+poly[:1]):
        if a == b or any(sign * _cross((b[0]-a[0], b[1]-a[1]), (p[0]-a[0], p[1]-a[1])) < -1e-7 for p in poly):
            return False
    return True


def _intersection(subject, clip):
    output, sign = list(subject), 1 if _area(clip) > 0 else -1
    for a, b in zip(clip, clip[1:]+clip[:1]):
        if not output:
            break
        vertices, previous = [], output[-1]
        edge = (b[0]-a[0], b[1]-a[1])
        prior = sign * _cross(edge, (previous[0]-a[0], previous[1]-a[1]))
        for current in output:
            side = sign * _cross(edge, (current[0]-a[0], current[1]-a[1]))
            if (side >= 0) != (prior >= 0):
                ratio = prior / (prior-side)
                vertices.append((previous[0]+(current[0]-previous[0])*ratio,
                                 previous[1]+(current[1]-previous[1])*ratio))
            if side >= 0:
                vertices.append(current)
            previous, prior = current, side
        output = vertices
    return output


def _rectangle(bounds):
    a, b, c, d = bounds
    return [(a, b), (c, b), (c, d), (a, d)]


def prepare_unit_crop(unit, batch, image, *, padding=2, alternate=False):
    """Include ONLY approved members; the final rectangular crop stays conservative."""
    if type(padding) is not int or not 0 <= padding <= 4:
        raise ValueError("unit padding must be 0..4")
    if not unit.approved:
        return None, None, unit.reason
    if image_fingerprint(image) != unit.plan.page.pixels_sha256:
        return None, None, "original_image_provenance_mismatch"
    sources = {d.source_id: d for d in batch.detections}
    if (any(d.page != unit.plan.page for d in batch.detections)
            or any(i not in sources for i in unit.plan.member_ids)
            or not any(existing.plan == unit.plan for existing in batch.units)):
        return None, None, "raw_page_provenance_mismatch"
    included = [sources[i] for i in unit.plan.member_ids]
    if any(not _valid_polygon(d.polygon) for d in included):
        return None, None, "invalid_member_polygon"
    points = [p for detection in included for p in detection.polygon]
    core = (math.floor(min(p[0] for p in points)), math.floor(min(p[1] for p in points)),
             math.ceil(max(p[0] for p in points)), math.ceil(max(p[1] for p in points)))
    extents = [e for e in batch.utterance_extents if e.unit_id == unit.plan.unit_id]
    if (len(extents) > 1 or any(not any(e.unit_id == existing.plan.unit_id for existing in batch.units)
                               for e in batch.utterance_extents)):
        return None, None, "ambiguous_or_foreign_utterance_extent"
    extent = extents[0] if len(extents) == 1 else None
    if extent is not None:
        # Explicit pixel-edge rectangles: fractional bounds are rejected rather
        # than rounded silently by Pillow. Float-valued integers are permitted.
        requested = extent.bounds
        checked = tuple(d.source_id for d in batch.detections if d.source_id not in unit.plan.member_ids)
        provenance = CropProvenance(unit.plan.page, unit.plan.unit_id, unit.plan.member_ids,
                                    requested, 0, alternate, checked, core, extent,
                                    "full_utterance_without_complete_reference")
        if (extent.page != unit.plan.page or extent.member_ids != unit.plan.member_ids
                or any(type(i) not in (int, str) for i in extent.member_ids)):
            return None, provenance, "utterance_extent_binding_mismatch"
        if (not _reviewed(extent.reviewed, extent.reviewer)
                or not isinstance(extent.source, str) or not extent.source.strip()):
            return None, provenance, "unreviewed_utterance_extent"
        if (len(requested) != 4 or any(type(v) not in (int, float)
                                      or (type(v) is float and not math.isfinite(v))
                                      or v != int(v) for v in requested)):
            return None, provenance, "invalid_utterance_extent_bounds"
        a, b, c, d = requested
        if not (0 <= a < c <= image.width and 0 <= b < d <= image.height):
            return None, provenance, "invalid_utterance_extent_bounds"
        if any(not (a <= x <= c and b <= y <= d) for x, y in points):
            return None, provenance, "utterance_extent_excludes_member"
        bounds = tuple(int(v) for v in requested)
    else:
        bounds = (max(0, core[0]-padding), max(0, core[1]-padding),
                  min(image.width, core[2]+padding), min(image.height, core[3]+padding))
    checked = tuple(d.source_id for d in batch.detections if d.source_id not in unit.plan.member_ids)
    provenance = CropProvenance(unit.plan.page, unit.plan.unit_id, unit.plan.member_ids,
                                bounds, 0 if extent else padding, alternate, checked, core, extent,
                                "full_utterance_without_complete_reference" if extent else "joined_main_readings")
    if any(x < 0 or y < 0 or x > image.width or y > image.height for x, y in points):
        return None, provenance, "clipped_member_polygon"
    for child, parent in unit.plan.duplicate_parents:
        if abs(_area(_intersection(sources[child].polygon, list(sources[parent].polygon)))) <= 1e-7:
            return None, provenance, "duplicate_geometry_conflict"
    owners = {o.source_id: o.bubble_id for o in batch.ownerships
              if o.reviewed is True and o.page == unit.plan.page}
    neighbors = []
    for detection in batch.detections:
        if detection.source_id in unit.plan.member_ids:
            continue
        polygon = detection.polygon
        if not _valid_polygon(polygon):
            return None, provenance, "invalid_neighbor_polygon"
        neighbors.append(polygon)
        overlap = abs(_area(_intersection(polygon, _rectangle(bounds if extent else core))))
        if overlap > 1e-7:
            reason = ("unknown_bubble_ownership" if detection.source_id not in owners else
                      "mixed_bubble_ownership" if owners[detection.source_id] != unit.plan.bubble_id
                      else "unrelated_neighbor_glyphs")
            return None, provenance, reason
        if abs(_area(_intersection(polygon, _rectangle(bounds)))) <= 1e-7:
            continue
        a, b = math.floor(min(p[0] for p in polygon)), math.floor(min(p[1] for p in polygon))
        c, d = math.ceil(max(p[0] for p in polygon)), math.ceil(max(p[1] for p in polygon))
        left, top, right, bottom = {bounds[0], core[0]}, {bounds[1], core[1]}, {bounds[2], core[2]}, {bounds[3], core[3]}
        if bounds[0] <= c <= core[0]:
            left.add(c)
        if bounds[1] <= d <= core[1]:
            top.add(d)
        if core[2] <= a <= bounds[2]:
            right.add(a)
        if core[3] <= b <= bounds[3]:
            bottom.add(b)
        candidates = [(l, t, r, s) for l in left for t in top for r in right for s in bottom
                      if abs(_area(_intersection(polygon, _rectangle((l, t, r, s))))) <= 1e-7]
        bounds = min(candidates, key=lambda v: (-(v[2]-v[0])*(v[3]-v[1]), v))
    provenance = CropProvenance(unit.plan.page, unit.plan.unit_id, unit.plan.member_ids,
                                bounds, 0 if extent else padding, alternate, checked, core, extent,
                                "full_utterance_without_complete_reference" if extent else "joined_main_readings")
    if any(abs(_area(_intersection(polygon, _rectangle(bounds)))) > 1e-7 for polygon in neighbors):
        return None, provenance, "unit_crop_contamination"
    return image.crop(bounds), provenance, None


def recognize_unit(unit, batch, image, recognizer, *, alternate=False, padding=2):
    """One primary and at most one explicitly requested alternate, never per-member writes."""
    if alternate:
        if unit.attempts != 1 or unit.alternate_attempts or unit.status != UnitStatus.REVIEW:
            return unit
    elif unit.attempts:
        return unit
    crop, provenance, error = prepare_unit_crop(unit, batch, image, padding=0 if alternate else padding,
                                               alternate=alternate)
    if (alternate and unit.crop is not None and unit.crop.reviewed_extent is not None
            and (provenance is None or provenance.reviewed_extent != unit.crop.reviewed_extent
                 or provenance.bounds != unit.crop.bounds)):
        unit.status, unit.reason = UnitStatus.REVIEW, "alternate_utterance_extent_changed"
        return unit
    unit.crop = provenance
    if error:
        unit.status, unit.reason = UnitStatus.REVIEW, error
        return unit
    unit.attempts += 1
    unit.alternate_attempts += int(alternate)
    unit.status, unit.reason = UnitStatus.REVIEW, "unit_recognition_in_progress"
    try:
        text = recognizer(crop, source_unit_id=unit.plan.unit_id, member_ids=unit.plan.member_ids,
                          crop_bounds=provenance.bounds)
        if not isinstance(text, str):
            raise TypeError("unit recognizer must return text only")
    except Exception:
        unit.reason = "unit_recognition_failed"
        raise
    reference, spans = unit.plan.reference_reading, ()
    status = UnitStatus.REVIEW
    if not normalized(text):
        reason = "blank_unit_candidate"
    elif provenance.reviewed_extent is not None:
        reason = "full_utterance_reference_completeness_unverified"
        if reference is not None:
            a, b = normalized(reference), normalized(text)
            spans = tuple((op, (i, j), (k, l), a[i:j], b[k:l])
                          for op, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                          if op != "equal")
    elif not unit.plan.comparison_valid:
        reason = "incomparable_granularity_or_main_readings"
    elif normalized(text) == normalized(reference):
        status, reason = UnitStatus.ACCEPTED, "normalized_joined_main_agreement_not_truth"
    else:
        reason = "unit_engine_disagreement"
        a, b = normalized(reference), normalized(text)
        spans = tuple((op, (i, j), (k, l), a[i:j], b[k:l])
                      for op, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                      if op != "equal")
    if alternate and any(normalized(c.text) != normalized(text) for c in unit.candidates):
        status, reason = UnitStatus.REVIEW, "alternate_unit_candidates_disagree"
    candidate = UnitCandidate(text, "manga", reference, status, reason, spans, provenance)
    unit.candidates += (candidate,)
    unit.status, unit.reason = status, reason
    return unit
