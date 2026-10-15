"""Opt-in automatic manga structure proposals, never reviewed recognition units."""
import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from .grouping import GroupingMixin, analyze_manga_regions, group_manga_boxes
from .automatic_evidence import OriginalInkEvidence
from .types import Text_Group
from .recognition_units import (PageProvenance, _area, _intersection, _rectangle,
                                _valid_polygon, image_fingerprint, snapshot_detections)


POLICY = "automatic_manga_structure_v2"
COMPLEMENTARY_OWNERSHIP_POLICY = "enclosed_complementary_connectivity_v2"
COMPANION_POLICY = "bounded_vertical_companions_v2"


@dataclass(frozen=True)
class RelationshipProposal:
    source_id: object
    parent_id: object
    possible_parent_ids: tuple
    relationship: str
    reasons: tuple
    region_support: str
    image_evidence: tuple = ()


@dataclass(frozen=True)
class NonmemberCropConflict:
    source_id: object
    reason: str
    polygon: tuple
    intersection_polygon: tuple
    intersection_area: object
    nonmember_area: object
    member_intersection_areas: tuple


def crop_conflicts(bounds, member_ids, detections):
    """Exact geometry only; do not infer a conflict's role or balloon ownership."""
    # Evidence order must not depend on the caller's detection sequence. Keep
    # integer source indices numeric, with a deterministic separate string tier.
    members = sorted((d for d in detections if d.source_id in member_ids),
                     key=lambda d:(type(d.source_id).__name__,d.source_id))
    conflicts = []
    for raw in detections:
        if raw.source_id in member_ids:
            continue
        if not _valid_polygon(raw.polygon):
            conflicts.append(NonmemberCropConflict(raw.source_id,"invalid_nonmember_geometry",
                raw.polygon,(),None,None,()))
            continue
        intersection = _intersection(raw.polygon,_rectangle(bounds))
        area = abs(_area(intersection))
        if area > 1e-7:
            overlaps = tuple((d.source_id,abs(_area(_intersection(raw.polygon,list(d.polygon))))) for d in members)
            conflicts.append(NonmemberCropConflict(raw.source_id,"nonmember_polygon_intersects_crop",
                raw.polygon,tuple(tuple(p) for p in intersection),area,abs(_area(raw.polygon)),overlaps))
    return tuple(sorted(conflicts,key=lambda c: _key(c.source_id)))


def detection_evidence_sha256(detections):
    """Bind all raw readings/geometry/status/provenance, not just member IDs."""
    snapshots = sorted((asdict(d) for d in detections),key=lambda d: _key(d["source_id"]))
    return hashlib.sha256(json.dumps(snapshots,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class AutomaticUnitProposal:
    proposal_id: str
    page: PageProvenance
    main_ids: tuple
    annotation_ids: tuple
    member_ids: tuple
    annotation_parents: tuple
    direction: str
    region_ids: tuple
    grouping_method: str
    structural_reasons: tuple
    word_reasons: tuple
    crop_bounds: tuple
    crop_safe: bool
    crop_reasons: tuple
    completeness: str = "unverified_full_utterance"
    structural_status: str = "proposal_unverified"
    reviewed: bool = False
    crop_conflicts: tuple = ()
    rejected_merge_evidence: tuple = ()


@dataclass(frozen=True)
class AutomaticProposalBatch:
    page: PageProvenance
    detections: tuple
    proposals: tuple
    relationships: tuple
    unassigned: tuple
    regions: tuple
    region_status: str
    policy: str = POLICY
    reviewed: bool = False
    rejected_merges: tuple = ()
    evidence_sha256: str = ""
    ownership_policy: str = "production8"
    ownership_repairs: tuple = ()
    companion_policy: str = "disabled"
    companion_regroupings: tuple = ()
    companion_rejections: tuple = ()

    @property
    def coverage(self):
        return {"raw_detections": len(self.detections),
                "main_detections": sum(len(p.main_ids) for p in self.proposals),
                "tentative_annotations": sum(len(p.annotation_ids) for p in self.proposals),
                "unassigned_detections": len(self.unassigned), "proposed_units": len(self.proposals),
                "relationship_conflicts": sum(r.parent_id is None for r in self.relationships),
                "unsafe_crops": sum(not p.crop_safe for p in self.proposals),
                "rejected_merges": len(self.rejected_merges),
                "full_utterance_verified_units": 0}


def _key(source_id):
    return type(source_id).__name__, str(source_id)


def _kana(text):
    letters = [c for c in unicodedata.normalize("NFKC", text) if not c.isspace()
               and not unicodedata.category(c).startswith(("P", "S"))]
    return (any("\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa" for c in letters)
            and sum("\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa"
                    or c == "ー" for c in letters) / len(letters) >= .7)


def _kanji(text):
    return any("\u3400" <= c <= "\u9fff" or "\U00020000" <= c <= "\U0002fa1f" for c in text)


def _annotation_geometry(child, parent):
    direction = GroupingMixin._manga_direction(parent)
    # Short one/two-glyph reading runs need not have the parent's aspect ratio.
    # The parent defines the reading axis; image glyph flow corroborates it later.
    if direction == "unknown":
        return False
    a, b = child.bbox, parent.bbox
    if direction == "vertical":
        minor_child, minor_parent = a[2]-a[0], b[2]-b[0]
        extent_child, extent_parent = a[3]-a[1], b[3]-b[1]
        overlap = max(0, min(a[3], b[3])-max(a[1], b[1]))
        gap = a[0]-b[2]  # Reading annotation on the right of a vertical main.
        contained = a[1] >= b[1]-.25*minor_parent and a[3] <= b[3]+.25*minor_parent
        side = (a[0]+a[2])/2 >= b[0]+.6*minor_parent
    else:
        minor_child, minor_parent = a[3]-a[1], b[3]-b[1]
        extent_child, extent_parent = a[2]-a[0], b[2]-b[0]
        overlap = max(0, min(a[2], b[2])-max(a[0], b[0]))
        gap = b[1]-a[3]  # Reading annotation above a horizontal main.
        contained = a[0] >= b[0]-.25*minor_parent and a[2] <= b[2]+.25*minor_parent
        side = (a[1]+a[3])/2 <= b[1]+.4*minor_parent
    return (.15 <= minor_child/minor_parent <= .65 and -.65*minor_parent <= gap <= .8*minor_parent
            and extent_child <= 1.1*extent_parent and overlap/extent_child >= .8 and contained and side)


def _singleton_companion_evidence(first, second, regions, by_id, glyphs, ink, annotations,
                                  detections, relationships, context):
    """A terminal, reciprocal-conflict repair, not general singleton proximity."""
    if sorted((len(first.group),len(second.group)))[0] != 1 or max(len(first.group),len(second.group)) < 2:
        return None
    singleton,host = (first,second) if len(first.group) == 1 else (second,first)
    mains = first.group+second.group
    ids = tuple(box.box_id for box in mains)
    owners = [regions.get(by_id[i][2],(None,None))[0] for i in ids]
    if None in owners or len(set(owners)) != 1 or owners[0] <= 0:
        return None
    roles = {r.source_id:r for r in relationships if r.parent_id is not None}

    def members(group):
        main_ids = tuple(box.box_id for box in group.group)
        children = tuple(sorted((child for child,parent in annotations.items() if parent in main_ids),key=_key))
        return main_ids+children,children

    first_members,first_children = members(first)
    second_members,second_children = members(second)
    if not first_children or not second_children:
        return None
    children = first_children+second_children
    image_reasons = {"original_small_glyph_scale_and_side_placement","original_single_leading_glyph_position_and_containment"}
    if any(child not in roles or roles[child].parent_id != annotations[child]
           or roles[child].relationship != "tentative_furigana"
           or roles[child].possible_parent_ids != (annotations[child],)
           or not image_reasons.intersection(roles[child].reasons) for child in children):
        return None
    all_members = first_members+second_members
    reasons = []
    if host.method == "singleton_companion_geometry":
        reasons.append("singleton_extension_is_terminal_not_a_chain")
    if any(regions.get(by_id[i][2],(owners[0],None))[0] != owners[0] for i in children):
        reasons.append("conflicting_annotation_region_support")
    if any(GroupingMixin._manga_direction(box) != "vertical" or glyphs[box.box_id].flow != "vertical"
           or glyphs[box.box_id].component_count < 2 for box in mains):
        reasons.append("unsupported_singleton_main_glyph_run")
    for n,left in enumerate(mains):
        for right in mains[n+1:]:
            x,y,r,s = left.bbox
            u,v,w,t = right.bbox
            scale = min(r-x,w-u)
            overlap = max(0,min(s,t)-max(y,v))/min(s-y,t-v)
            sizes = (glyphs[left.box_id].scale,glyphs[right.box_id].scale)
            left_core,right_core = glyphs[left.box_id].bounds,glyphs[right.box_id].bounds
            if (not left_core or not right_core
                    or max(left_core[0],right_core[0]) < min(left_core[2],right_core[2])):
                reasons.append("singleton_mains_not_distinct_ink_columns")
            if (max(r-x,w-u) > 1.6*scale or overlap < .6 or abs(y-v) > 1.5*scale
                    or max(0,max(x,u)-min(r,w)) > 6*scale or min(sizes) <= 0 or max(sizes) > 1.6*min(sizes)):
                reasons.append("singleton_complete_span_scale_or_alignment_limit")

    def bounds(member_ids):
        points = [p for i in member_ids for p in by_id[i][0].polygon]
        return (math.floor(min(x for x,y in points)),math.floor(min(y for x,y in points)),
                math.ceil(max(x for x,y in points)),math.ceil(max(y for x,y in points)))

    first_conflicts = crop_conflicts(bounds(first_members),first_members,detections)
    second_conflicts = crop_conflicts(bounds(second_members),second_members,detections)
    if (not {c.source_id for c in first_conflicts}.intersection(box.box_id for box in second.group)
            or not {c.source_id for c in second_conflicts}.intersection(box.box_id for box in first.group)):
        reasons.append("no_reciprocal_main_footprint_conflict")
    main_left,main_right = sorted((first,second),key=lambda g:min(glyphs[b.box_id].bounds[0] if glyphs[b.box_id].bounds else b.bbox[0] for b in g.group))
    edge_left = max(main_left.group,key=lambda b:glyphs[b.box_id].bounds[2] if glyphs[b.box_id].bounds else b.bbox[2])
    edge_right = min(main_right.group,key=lambda b:glyphs[b.box_id].bounds[0] if glyphs[b.box_id].bounds else b.bbox[0])
    left_members,_ = members(main_left)
    right_members,_ = members(main_right)
    cores = {i:glyphs[i].bounds for i in all_members}
    corridor = {}
    if any(not core for core in cores.values()):
        reasons.append("missing_original_member_ink_core")
    else:
        # Include the visible ink of already assigned readings in each envelope.
        # Never remove their ink from the original corridor or boundary scan.
        x1,x2 = max(cores[i][2] for i in left_members),min(cores[i][0] for i in right_members)
        y1,y2 = max(cores[edge_left.box_id][1],cores[edge_right.box_id][1]),min(cores[edge_left.box_id][3],cores[edge_right.box_id][3])
        corridor["bounds"] = (x1,y1,x2,y2)
        if x2-x1 < 2 or y2 <= y1:
            reasons.append("no_contiguous_original_singleton_seam")
        else:
            white = ink.ink[y1:y2,x1:x2] == 0
            corridor.update(white_fraction=float(np.mean(white)),fully_white_column_fraction=float(np.mean(np.all(white,axis=0))))
            if (x2-x1 > 1.8*min(glyphs[edge_left.box_id].scale,glyphs[edge_right.box_id].scale)
                    or corridor["white_fraction"] < .98 or corridor["fully_white_column_fraction"] < .5):
                reasons.append("original_singleton_corridor_not_clear_or_local")
    boundary = ink.boundary_between(edge_left,edge_right,
        first_glyphs=glyphs[edge_left.box_id],second_glyphs=glyphs[edge_right.box_id])
    if boundary["blocked"] or boundary["reason"] == "no_resolvable_facing_corridor":
        reasons.append("original_boundary_blocks_singleton_seam")
    if context is None:
        context = ink.single_glyph_context()
    _,_,white_labels,white_stats = context
    backgrounds,white_support = [],[]
    height,width = ink.ink.shape
    for i in all_members:
        box = by_id[i][1]
        x,y,r,s = box.bbox
        if not (0 <= x < r < width and 0 <= y < s < height):
            reasons.append("singleton_original_enclosure_inconclusive")
            break
        mask = np.zeros((s-y+1,r-x+1),dtype=np.uint8)
        cv2.fillPoly(mask,[box.poly.astype(np.int32)-np.array([x,y])],255)
        values = white_labels[y:s+1,x:r+1][mask != 0]
        votes = np.bincount(values,minlength=len(white_stats))
        if not votes[1:].sum():
            reasons.append("singleton_original_enclosure_inconclusive")
            break
        label = int(np.argmax(votes[1:])+1)
        bx,by,bw,bh,_ = (int(v) for v in white_stats[label])
        support = float(votes[label]/votes[1:].sum())
        if support < .9 or not (bx > 0 and by > 0 and bx+bw < width and by+bh < height):
            reasons.append("singleton_original_enclosure_inconclusive")
        backgrounds.append(label)
        white_support.append((i,support))
    if len(set(backgrounds)) != 1:
        reasons.append("distinct_original_singleton_enclosures")
    ordered = main_right.group+main_left.group
    record = {"kind":"singleton_companion","singleton_id":singleton.group[0].box_id,
        "main_ids":tuple(b.box_id for b in ordered),"previous_main_groups":(tuple(b.box_id for b in main_right.group),tuple(b.box_id for b in main_left.group)),
        "region_id":owners[0],"corridor":corridor,"seam_ids":(edge_left.box_id,edge_right.box_id),"boundary":boundary,
        "reciprocal_conflicts":tuple({"main_ids":tuple(b.box_id for b in group.group),
            "conflicting_source_ids":tuple(c.source_id for c in conflicts)}
            for group,conflicts in ((first,first_conflicts),(second,second_conflicts))),
        "original_background_ids":tuple(backgrounds),"original_white_support":tuple(white_support),
        "structural_reasons":tuple(dict.fromkeys(reasons)),"crop_bounds":bounds(all_members),
        "completeness":"unverified_full_utterance"}
    conflicts = crop_conflicts(record["crop_bounds"],all_members,detections)
    if conflicts:
        reasons.append("singleton_union_has_nonmember_conflicts")
    record.update(crop_conflicts=tuple(asdict(c) for c in conflicts),reasons=tuple(dict.fromkeys(reasons)))
    confidence = min(regions[by_id[i][2]][1] for i in ids)
    record["annotation_region_support"] = tuple((i,owners[0],min(confidence,support)) for i,support in white_support
        if i in children and by_id[i][2] not in regions)
    return ordered,record,context


def _regroup_companions(groups, regions, by_id, glyphs, ink, annotations, detections,
                        relationships=(), original_context=None):
    """Join only contiguous, bounded vertical runs across a clear original seam.

    Existing groups retain their internal barrier validation. Checking the seam
    between neighboring groups avoids mistaking an intervening main column for
    a boundary in a corridor between nonadjacent columns. No ink is masked here.
    """
    groups = list(groups)
    accepted, rejected = [], {}
    while True:
        candidates = []
        singleton_options = {}
        for a,first in enumerate(groups):
            for b in range(a+1,len(groups)):
                second = groups[b]
                mains = first.group + second.group
                ids = tuple(box.box_id for box in mains)
                members = ids + tuple(sorted((child for child,parent in annotations.items() if parent in ids),key=_key))
                owners = [regions.get(by_id[i][2],(None,None))[0] for i in members]
                if (len(first.group) < 2 or len(second.group) < 2 or None in owners
                        or len(set(owners)) != 1 or owners[0] >= 0):
                    extension = _singleton_companion_evidence(first,second,regions,by_id,glyphs,ink,
                        annotations,detections,relationships,original_context)
                    if extension is not None:
                        ordered,record,original_context = extension
                        if not record["structural_reasons"]:
                            singleton_options.setdefault(record["singleton_id"],[]).append(record)
                        if record["reasons"]:
                            rejected[tuple(sorted(ids,key=_key))] = record
                        else:
                            candidates.append((record["corridor"]["bounds"][2]-record["corridor"]["bounds"][0],a,b,ordered,record))
                    continue
                reasons = []
                if any(GroupingMixin._manga_direction(box) != "vertical"
                       or glyphs[box.box_id].flow != "vertical"
                       or glyphs[box.box_id].component_count < 2 for box in mains):
                    reasons.append("unsupported_companion_glyph_flow_or_orientation")
                for n,left in enumerate(mains):
                    for right in mains[n+1:]:
                        x,y,r,s = left.bbox
                        u,v,w,t = right.bbox
                        scale = min(r-x,w-u)
                        overlap = max(0,min(s,t)-max(y,v))/min(s-y,t-v)
                        if (max(r-x,w-u) > 1.6*scale or overlap < .6
                                or abs(y-v) > 1.5*scale or max(0,max(x,u)-min(r,w)) > 6*scale):
                            reasons.append("companion_complete_span_or_alignment_limit")
                left,right = sorted((first,second),key=lambda g:min(box.bbox[0] for box in g.group))
                edge_left = max(left.group,key=lambda box:box.bbox[2])
                edge_right = min(right.group,key=lambda box:box.bbox[0])
                x1,x2 = edge_left.bbox[2],edge_right.bbox[0]
                y1,y2 = max(edge_left.bbox[1],edge_right.bbox[1]),min(edge_left.bbox[3],edge_right.bbox[3])
                corridor = {"bounds": (x1,y1,x2,y2)}
                if x2-x1 < 2 or y2 <= y1:
                    reasons.append("no_contiguous_resolvable_companion_seam")
                else:
                    scale = min(edge_left.bbox[2]-edge_left.bbox[0],edge_right.bbox[2]-edge_right.bbox[0])
                    white = ink.ink[y1:y2,x1:x2] == 0
                    corridor.update(white_fraction=float(np.mean(white)),
                        minimum_row_white_fraction=float(np.min(np.mean(white,axis=1))),
                        fully_white_column_fraction=float(np.mean(np.all(white,axis=0))))
                    # Require a wide, uninterrupted white core over the entire
                    # facing extent, allowing sparse glyph-edge protrusions.
                    # Narrow horizontal connectors cannot supply that core.
                    if (x2-x1 > 1.8*scale or corridor["white_fraction"] < .98
                            or corridor["fully_white_column_fraction"] < .5):
                        reasons.append("original_companion_corridor_not_clear_or_local")
                boundary = ink.boundary_between(edge_left,edge_right,
                    first_glyphs=glyphs[edge_left.box_id],second_glyphs=glyphs[edge_right.box_id])
                if boundary["blocked"] or boundary["reason"] == "no_resolvable_facing_corridor":
                    reasons.append("original_boundary_blocks_companion_seam")
                points = [point for i in members for point in by_id[i][0].polygon]
                bounds = (math.floor(min(x for x,y in points)),math.floor(min(y for x,y in points)),
                          math.ceil(max(x for x,y in points)),math.ceil(max(y for x,y in points)))
                conflicts = crop_conflicts(bounds,members,detections)
                if conflicts:
                    reasons.append("companion_union_has_nonmember_conflicts")
                ordered = right.group + left.group
                record = {"main_ids": tuple(box.box_id for box in ordered),
                    "previous_main_groups": (tuple(box.box_id for box in right.group),tuple(box.box_id for box in left.group)),
                    "region_id": owners[0], "corridor": corridor,
                    "seam_ids": (edge_left.box_id,edge_right.box_id), "boundary": boundary,
                    "crop_bounds": bounds, "crop_conflicts": tuple(asdict(c) for c in conflicts),
                    "reasons": tuple(dict.fromkeys(reasons)), "completeness": "unverified_full_utterance"}
                if reasons:
                    rejected[tuple(sorted(ids,key=_key))] = record
                else:
                    candidates.append((x2-x1,a,b,ordered,record))
        ambiguous = {i for i,options in singleton_options.items() if len(options) > 1}
        for source_id in ambiguous:
            for record in singleton_options[source_id]:
                record["reasons"] += ("ambiguous_singleton_companion_groups",)
                rejected[tuple(sorted(record["main_ids"],key=_key))] = record
        candidates = [row for row in candidates if row[4].get("singleton_id") not in ambiguous]
        if not candidates:
            break
        _,a,b,ordered,record = min(candidates,key=lambda row:row[:3])
        accepted.append(record)
        if record.get("kind") == "singleton_companion":
            for source_id,region_id,confidence in record["annotation_region_support"]:
                regions[by_id[source_id][2]] = (region_id,confidence)
        groups[a] = Text_Group(ordered,method="singleton_companion_geometry" if record.get("kind") == "singleton_companion"
                               else "companion_geometry",status="grouped",direction="vertical")
        groups.pop(b)
    groups.sort(key=lambda g:(min(box.bbox[1] for box in g.group),-max(box.bbox[2] for box in g.group)))
    return groups,tuple(accepted),tuple(rejected[key] for key in sorted(rejected,key=lambda ids:tuple(map(_key,ids))))


def propose_automatic_units(detections, image, *, max_detections=512, complementary_ownership=False,
                            regroup_companions=False):
    """Concrete engine-free proposals from immutable snapshots + original RGB image.

    Every valid in-page detection is exactly one main OR tentative annotation;
    malformed/clipped detections are explicitly unassigned, never discarded.
    """
    detections = tuple(detections)
    if type(complementary_ownership) is not bool:
        raise ValueError("complementary_ownership must be an explicit boolean")
    if type(regroup_companions) is not bool or (regroup_companions and not complementary_ownership):
        raise ValueError("regroup_companions requires explicit complementary_ownership=True")
    if type(max_detections) is not int or not 1 <= max_detections <= 2048:
        raise ValueError("max_detections must be 1..2048")
    if not detections or len(detections) > max_detections:
        raise ValueError("nonempty bounded detections required; none are truncated")
    page = detections[0].page
    if len({d.source_id for d in detections}) != len(detections) or any(d.page != page for d in detections):
        raise ValueError("unique source IDs and exact single-page provenance required")
    if image_fingerprint(image) != page.pixels_sha256:
        raise ValueError("original_image_provenance_mismatch")
    ordered, unassigned = [], []
    for raw in detections:
        if not _valid_polygon(raw.polygon):
            unassigned.append((raw.source_id, "invalid_polygon"))
            continue
        if any(x < 0 or y < 0 or x > image.width or y > image.height for x, y in raw.polygon):
            unassigned.append((raw.source_id, "clipped_polygon"))
            continue
        bounds = (math.floor(min(x for x, _ in raw.polygon)), math.floor(min(y for _, y in raw.polygon)),
                  math.ceil(max(x for x, _ in raw.polygon)), math.ceil(max(y for _, y in raw.polygon)))
        ordered.append((raw, SimpleNamespace(box_id=raw.source_id, bbox=bounds,
            poly=np.array(raw.polygon, dtype=np.float64), text=raw.text, confidence=raw.confidence)))
    ordered.sort(key=lambda pair: (pair[1].bbox[1], -pair[1].bbox[0], _key(pair[0].source_id)))
    boxes = [pair[1] for pair in ordered]
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    regions, region_status = analyze_manga_regions(boxes, bgr, include_uncertain=True)
    by_id = {raw.source_id: (raw, box, index) for index, (raw, box) in enumerate(ordered)}
    ink = OriginalInkEvidence(bgr)
    glyphs = {raw.source_id: ink.glyphs(box) for raw,box in ordered}
    possibilities, relationship_reasons, rejected_relationships = {}, {}, []
    relation_evidence = {}
    single_context, single_supports = None, {}
    for child, child_box in ordered:
        options, rejected_options, evidence_rows = [], [], []
        for parent, parent_box in ordered:
            if child.source_id == parent.source_id or not _annotation_geometry(child_box, parent_box):
                continue
            ci, pi = by_id[child.source_id][2], by_id[parent.source_id][2]
            child_glyphs, parent_glyphs = glyphs[child.source_id], glyphs[parent.source_id]
            boundary = ink.boundary_between(child_box,parent_box,
                first_glyphs=child_glyphs,second_glyphs=parent_glyphs)
            rejects = []
            if ci in regions and pi in regions and regions[ci][0] != regions[pi][0]:
                rejects.append("annotation_geometry_crosses_distinct_regions")
            if boundary["blocked"]:
                rejects.append("annotation_crosses_original_boundary")
            direction = GroupingMixin._manga_direction(parent_box)
            if child_glyphs.flow not in ("unknown",direction):
                rejects.append("original_glyph_flow_inconsistent_with_reading_annotation")
            if (child_glyphs.scale and parent_glyphs.scale
                    and child_glyphs.scale > .75*parent_glyphs.scale):
                rejects.append("original_glyph_scale_not_smaller_than_main")
            row = {"parent_id": parent.source_id, "child_glyphs": asdict(child_glyphs),
                "parent_glyphs": asdict(parent_glyphs), "boundary": boundary, "rejected_reasons": tuple(rejects)}
            child_reading = unicodedata.normalize("NFKC",child.text).strip()
            parent_reading = unicodedata.normalize("NFKC",parent.text).strip()
            if (complementary_ownership and not rejects and child_glyphs.component_count == 1
                    and len(child_reading) == 1 and unicodedata.category(child_reading).startswith(("L","N"))
                    and unicodedata.category(child_reading) != "Lm"
                    and not _kana(child_reading) and parent_reading and _kanji(parent_reading[0])
                    and direction == "vertical" and pi in regions):
                if single_context is None:
                    single_context = ink.single_glyph_context()
                proof = ink.single_glyph_position(child_box,parent_box,child_glyphs,parent_glyphs,single_context,boxes)
                row["single_glyph_position"] = proof
                if proof["supported"]:
                    single_supports[(child.source_id,parent.source_id)] = proof
            evidence_rows.append(row)
            if rejects:
                rejected_options.append(parent.source_id)
                continue
            options.append(parent.source_id)
        relation_evidence[child.source_id] = tuple(evidence_rows)
        if rejected_options:
            rejection_reasons = tuple(dict.fromkeys(reason for row in evidence_rows for reason in row["rejected_reasons"]))
            rejected_relationships.append(RelationshipProposal(child.source_id, None,
                tuple(sorted(rejected_options,key=_key)), "rejected_annotation_candidate", rejection_reasons,
                "original_image_rejection", tuple(row for row in evidence_rows if row["rejected_reasons"])))
        if options:
            possibilities[child.source_id] = tuple(sorted(options, key=_key))
        elif rejected_options:
            relationship_reasons[child.source_id] = rejection_reasons
    # Nested/dual roles cannot make an annotation parent disappear from mains.
    parent_roles = {i for options in possibilities.values() for i in options}
    annotations, relationships = {}, list(rejected_relationships)
    for child_id, options in possibilities.items():
        raw, _, ci = by_id[child_id]
        reasons = ["relative_side_extent_alignment_and_scale"]
        parent_id = None
        if len(options) != 1:
            reasons.append("ambiguous_annotation_parents")
        elif child_id in parent_roles or options[0] in possibilities:
            reasons.append("ambiguous_annotation_chain_or_role")
        else:
            cg, pg = glyphs[child_id], glyphs[options[0]]
            wording_support = _kana(raw.text) and _kanji(by_id[options[0]][0].text)
            image_support = (cg.component_count >= 2 and pg.component_count >= 2
                             and .15 <= cg.scale/pg.scale <= .7)
            single_support = single_supports.get((child_id,options[0]))
            reasons.append("kana_and_main_kanji_corroboration" if wording_support
                           else "missing_kana_and_main_kanji_corroboration")
            if image_support:
                reasons.append("original_small_glyph_scale_and_side_placement")
            elif single_support:
                reasons.append("multi_component_glyph_support_unavailable")
            else:
                reasons.append("original_glyph_support_inconclusive")
            if single_support:
                reasons.append("original_single_leading_glyph_position_and_containment")
            if wording_support or image_support or single_support:
                parent_id = options[0]
                annotations[child_id] = parent_id
                reasons.append("tentative_not_reviewed_furigana")
                if single_support:
                    parent_region = regions[by_id[parent_id][2]]
                    regions[ci] = (parent_region[0],min(parent_region[1],single_support["original_white_support"][0]))
                    single_support["parent_region_support"] = regions[ci]
                    reasons.append("tentative_single_glyph_reading_not_transcription")
            else:
                reasons.append("annotation_role_unresolved_not_independent_text_proof")
        pi = by_id[options[0]][2] if len(options) == 1 else None
        support = "same_region" if ci in regions and pi in regions else "geometry_only"
        if parent_id is not None and (child_id,parent_id) in single_supports:
            support = "original_single_glyph_parent_region"
        relationships.append(RelationshipProposal(child_id, parent_id, options,
            "tentative_furigana" if parent_id is not None else "unresolved", tuple(reasons), support,
            relation_evidence[child_id]))
        if parent_id is None:
            relationship_reasons[child_id] = tuple(reasons)
            for possible_parent in options:
                relationship_reasons[possible_parent] = tuple(dict.fromkeys(
                    relationship_reasons.get(possible_parent, ()) + ("unresolved_annotation_relationship",)))
    mains = [pair for pair in ordered if pair[0].source_id not in annotations]
    main_regions = {i: regions[by_id[raw.source_id][2]] for i, (raw, _) in enumerate(mains)
                    if by_id[raw.source_id][2] in regions}
    rejected_merges = {}
    boundary_cache = {}
    def merge_guard(first,second):
        members = tuple(sorted((box.box_id for box in first+second),key=_key))
        reasons, barriers = [], []
        unsupported = [i for i in members if glyphs[i].component_count < 2
                       and not (_kana(by_id[i][0].text) or _kanji(by_id[i][0].text))]
        if len(members) > 2 and unsupported:
            reasons.append("unsupported_blank_or_noisy_main_cannot_bridge_group")
        for left in first:
            for right in second:
                pair = tuple(sorted((left.box_id,right.box_id),key=_key))
                # A pair's exclusion proof belongs only to this merge attempt.
                # Reusing it with a different member set could create a chain bridge.
                key = (pair,members)
                if key not in boundary_cache:
                    included = tuple((box,glyphs[box.box_id]) for box in first+second
                                     if box.box_id not in pair) if not unassigned else ()
                    nonmembers = tuple(box for box in boxes if box.box_id not in members)
                    boundary_cache[key] = ink.boundary_between(left,right,
                        first_glyphs=glyphs[left.box_id],second_glyphs=glyphs[right.box_id],
                        included_members=included,nonmember_boxes=nonmembers)
                evidence = boundary_cache[key]
                if evidence["blocked"]:
                    reasons.append("original_image_boundary_blocks_column_merge")
                    barriers.append({"source_ids": pair, "evidence": evidence})
        if reasons:
            prior = rejected_merges.get(members, {"reasons": (), "boundaries": ()})
            combined_barriers = {row["source_ids"]: row for row in prior["boundaries"] + tuple(barriers)}
            rejected_merges[members] = {"member_ids": members,
                "reasons": tuple(dict.fromkeys(prior["reasons"] + tuple(reasons))),
                "unsupported_source_ids": tuple(unsupported),
                "boundaries": tuple(combined_barriers[pair] for pair in sorted(combined_barriers,
                    key=lambda ids: tuple(map(_key,ids))))}
            return False
        return True
    groups = group_manga_boxes([box for _, box in mains], regions=main_regions, region_status=region_status,
                              merge_guard=merge_guard)
    ownership_repairs = ()
    if complementary_ownership:
        member_groups = []
        for group in groups:
            group_ids = {box.box_id for box in group.group}
            member_groups.append((tuple(by_id[box.box_id][2] for box in group.group),
                tuple(by_id[child][2] for child in sorted(
                    (child for child,parent in annotations.items() if parent in group_ids),key=_key))))
        regions,ownership_repairs = ink.complementary_ownership(boxes,member_groups,regions)
    repaired_ids = {i for repair in ownership_repairs for i in repair["member_ids"]}
    companion_regroupings,companion_rejections = (),()
    if regroup_companions:
        groups,companion_regroupings,companion_rejections = _regroup_companions(
            groups,regions,by_id,glyphs,ink,annotations,detections,relationships,single_context)
    proposals = []
    for group in groups:
        main_ids = tuple(box.box_id for box in group.group)
        attached = tuple(sorted((child for child, parent in annotations.items() if parent in main_ids), key=_key))
        members = main_ids + attached
        parents = tuple((child, annotations[child]) for child in attached)
        points = [p for i in members for p in by_id[i][0].polygon]
        bounds = (math.floor(min(x for x, _ in points)), math.floor(min(y for _, y in points)),
                  math.ceil(max(x for x, _ in points)), math.ceil(max(y for _, y in points)))
        crop_reasons = ["detection_union_only", "punctuation_outside_detections_not_ruled_out"]
        safe = True
        conflicts = crop_conflicts(bounds,members,detections)
        if conflicts:
            safe = False
            crop_reasons.extend(conflict.reason for conflict in conflicts)
        structural = ["automatic_not_human_reviewed"]
        if repaired_ids.intersection(members):
            structural.append("complementary_component_ownership")
        if group.method in ("companion_geometry","singleton_companion_geometry"):
            structural.append("companion_regrouping_not_full_utterance_proof")
        if group.method == "singleton_companion_geometry":
            structural.append("guarded_singleton_companion_extension")
        region_ids = tuple(sorted({regions[by_id[i][2]][0] for i in members if by_id[i][2] in regions}))
        if any(by_id[i][2] not in regions for i in members):
            structural.append("unknown_region_ownership")
        if len(region_ids) > 1:
            structural.append("mixed_region_ownership")
            safe = False
        if group.direction == "unknown":
            structural.append("unknown_reading_orientation")
        for i in members:
            structural.extend(relationship_reasons.get(i, ()))
        if attached:
            structural.append("tentative_annotation_ownership")
            if any((child,annotations[child]) in single_supports for child in attached):
                structural.append("tentative_single_glyph_annotation_ownership")
        if not safe:
            structural.append("crop_isolation_unresolved")
        words = ["wording_not_human_checked"]
        if any(not by_id[i][0].text.strip() for i in main_ids):
            words.append("blank_main_reading")
        if any(by_id[i][0].status != "accepted" for i in main_ids):
            words.append("paddle_wording_unresolved")
        if any(not _kana(by_id[i][0].text) or not _kanji(by_id[annotations[i]][0].text) for i in attached):
            words.append("annotation_wording_corroboration_missing")
        if any(not by_id[i][0].text.strip() for i in attached):
            words.append("blank_annotation_reading")
        identity = {"policy": POLICY, "page": asdict(page), "main_ids": main_ids,
                    "members": members, "member_polygons": [(i, by_id[i][0].polygon) for i in members],
                    "annotation_parents": parents, "direction": group.direction}
        if complementary_ownership:
            identity["ownership_policy"] = COMPLEMENTARY_OWNERSHIP_POLICY
        if regroup_companions:
            identity["companion_policy"] = COMPANION_POLICY
        proposal_id = "proposal-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        proposals.append(AutomaticUnitProposal(proposal_id, page, main_ids, attached, members, parents,
            group.direction, region_ids, group.method, tuple(dict.fromkeys(structural)),
            tuple(words), bounds, safe, tuple(dict.fromkeys(crop_reasons)),crop_conflicts=conflicts,
            rejected_merge_evidence=tuple(value for key,value in rejected_merges.items() if set(key)&set(main_ids))))
    return AutomaticProposalBatch(page, detections, tuple(proposals),
        tuple(sorted(relationships, key=lambda r: _key(r.source_id))), tuple(sorted(unassigned, key=lambda r: _key(r[0]))),
        tuple((raw.source_id, *regions[index]) for index, (raw, _) in enumerate(ordered) if index in regions), region_status,
        rejected_merges=tuple(rejected_merges[key] for key in sorted(rejected_merges,key=lambda ids: tuple(map(_key,ids)))),
        evidence_sha256=detection_evidence_sha256(detections),
        ownership_policy=COMPLEMENTARY_OWNERSHIP_POLICY if complementary_ownership else "production8",
        ownership_repairs=ownership_repairs, companion_policy=COMPANION_POLICY if regroup_companions else "disabled",
        companion_regroupings=companion_regroupings,companion_rejections=companion_rejections)


def propose_page_from_boxes(boxes, image_path, *, complementary_ownership=False, regroup_companions=False):
    """Explicit production coordination: original file + unchanged Paddle snapshots."""
    content = Path(image_path).read_bytes()
    with Image.open(BytesIO(content)) as source:
        image = source.convert("RGB")
    # This is a detection-snapshot hash, NOT a saved comparison file hash or review.
    evidence = [{"id": b.paddle_box_id, "text": b.paddle_text, "confidence": float(b.paddle_confidence),
                 "polygon": b.paddle_poly.tolist()} for b in boxes]
    evidence.sort(key=lambda row: _key(row["id"]))
    page = PageProvenance(str(Path(image_path).resolve()), str(Path(image_path).resolve()),
        hashlib.sha256(content).hexdigest(), hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
        image_fingerprint(image))
    return propose_automatic_units(snapshot_detections(boxes, page), image,
        complementary_ownership=complementary_ownership,regroup_companions=regroup_companions)
