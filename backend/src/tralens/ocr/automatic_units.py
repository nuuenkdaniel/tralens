"""Opt-in automatic manga structure proposals, never reviewed recognition units."""
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from .grouping import GroupingMixin, analyze_manga_regions, group_manga_boxes
from .automatic_evidence import OriginalInkEvidence
from . import table_rows
from .types import Text_Group, is_credit_list, is_printed_literal
from .recognition_units import (PageProvenance, _area, _intersection, _rectangle,
                                _valid_polygon, image_fingerprint, snapshot_detections)


POLICY = "automatic_manga_structure_v3_polarity"
LEGACY_POLICY = "automatic_manga_structure_v2"
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


def join_reread_columns(groups, by_id):
    """Two locally reread vertical columns whose original OCR split off a digit."""
    result = list(groups)
    single = {g.group[0].box_id:g for g in groups if len(g.group)==1}
    for source,group in single.items():
        meta = json.loads(by_id[source][0].metadata_json)
        anchor = meta.get('short_column_anchor')
        if (anchor not in single or group not in result or single[anchor] not in result or anchor==source
                or not meta.get('corroborated_short_reading')
                or not json.loads(by_id[anchor][0].metadata_json).get('corroborated_short_reading')):
            continue
        result[result.index(single[anchor])] = Text_Group(single[anchor].group+group.group,
            method='paired_short_columns',direction='vertical')
        result.remove(group)
    return result


def join_stuttered_onsets(groups, pixels, excluded, marked=None):
    """Bind a separately printed initial kana to the word that repeats it."""
    result = list(groups)
    for onset in groups:
        first = onset.group[0]
        if (len(onset.group)!=1 or first.box_id in excluded or (marked is not None and first.box_id not in marked) or first.confidence<.67
                or onset.direction not in ('unknown','vertical') or not re.fullmatch('[ぁ-ゖァ-ヺ]',first.text)):
            continue
        x,y,r,b = first.bbox
        candidates = []
        for body in result:
            word = body.group[0]
            if (body is onset or body.direction!='vertical' or any(v.box_id in excluded for v in body.group) or word.confidence<.9
                    or len(word.text)<2 or not word.text.startswith(first.text)):
                continue
            l,t,right,bottom = word.bbox
            scale = min(r-x,right-l)
            if (scale<6 or max(r-x,right-l)>1.55*scale or not 1<x-right<.55*scale
                    or abs(y-t)>.4*scale or min(b,bottom)-max(y,t)<.45*scale):
                continue
            seam = pixels[max(y,t):min(b,bottom)+1,right+1:x]
            if seam.size and (seam.min(2)>240).mean()>=.99:
                candidates.append(body)
        if len(candidates)==1 and onset in result:
            body = candidates[0]
            result[result.index(body)] = Text_Group(onset.group+body.group,method='stuttered_onset',direction='vertical')
            result.remove(onset)
    return result


def join_shared_balloon(groups, pixels, detections, excluded):
    """Read adjacent vertical passages sharing one enclosed sheet of white paper."""
    def suffix(group):
        return len(group.group)==1 and re.fullmatch(r'し(?:て(?:た|いた|いる|る)|ました|ません|なかった|ないで(?:ください)?)[。！？…]*',group.group[0].text)

    def genitive(group):
        return len(group.group)==1 and re.fullmatch(r'[一-鿿々]{2,8}の',group.group[0].text)

    def lead_in(group):
        return len(group.group)==1 and re.fullmatch(r'(?:[ァ-ヺー一-鿿A-Za-z]{2,15}は|とっ?ても|非常に|かなり|すごく|とてつもなく)',group.group[0].text)

    def dependent(group):
        # These endings leave the subject, object, time limit or referent waiting
        # for a predicate. A reporting と can instead point to a preceding quote.
        return re.search(r'(?:[ぁ-ゖァ-ヺ一-鿿々](?:は|が|を|に|で|まで)|(?:^|[・.…]+)(?:これ|それ|あれ))$',
                         group.group[-1].text)

    candidates = [g for g in groups if g.direction=='vertical' and (2<=len(g.group)<=6 or
                  suffix(g) or genitive(g) or lead_in(g) or (len(g.group)==1 and dependent(g)))
                  and all(b.box_id not in excluded for b in g.group)]
    if len(candidates) < 2 or pixels.shape[0]*pixels.shape[1]>8_000_000:
        return groups
    # Find a staggered pair before doing any pixel work. Inspect only its local
    # balloon so a large white panel or the page margin cannot join passages.
    result = list(groups)
    ordered = sorted(candidates,key=lambda g:-max(b.bbox[0] for b in g.group))
    for i,right in enumerate(ordered):
        for left in ordered[i+1:]:
            if right not in result or left not in result:
                continue
            boxes = right.group+left.group
            widths = [b.bbox[2]-b.bbox[0] for b in boxes]
            scale = min(widths)
            lhs = max(b.bbox[2] for b in left.group)
            rhs = min(b.bbox[0] for b in right.group)
            stagger = abs(right.group[0].bbox[1]-left.group[0].bbox[1])
            attached = (suffix(left) and len(right.group)>=2 and -scale*.45<rhs-lhs<scale*.7
                        and stagger<scale*.5 and re.search(r'[一-鿿]{2,}$',min(right.group,key=lambda v:v.bbox[0]).text))
            modifier = (genitive(right) and len(left.group)>=2 and re.match(r'[一-鿿々]{2,}',left.group[0].text)
                        and 0<rhs-lhs<.65*scale and stagger<.45*scale
                        and abs(right.group[0].bbox[3]-max(v.bbox[3] for v in left.group))<scale)
            prefix = (lead_in(right) and 2<=len(left.group)<=5 and 0<rhs-lhs<.7*scale
                       and stagger<.4*scale and max(widths)<1.5*scale
                       and all(abs(v.bbox[1]-right.group[0].bbox[1])<.45*scale for v in left.group))
            # A nominalized explanation split immediately before its copula
            # changes from an assertion to a question when translated alone.
            explanation = (len(right.group)>=2 and re.search(r'(?:ている|てる|た|ない)の$',right.group[-1].text)
                           and re.fullmatch(r'(?:だから|ですから|だもの|ですもの)[ー―—！!…。]*',
                                            ''.join(v.text for v in left.group))
                           and -.15*scale<=rhs-lhs<=.65*scale and stagger<=.4*scale)
            clause = (dependent(right) and 2<=len(left.group)<=5 and len(boxes)<=8
                      and min(v.confidence for v in boxes)>=.85 and max(widths)<1.50*scale
                      and -.15*scale<rhs-lhs<.90*scale and stagger<1.1*scale
                      and re.search(r'(?:ます|ました|ません|だった|ですが|たろ|つもり|したり|だろう)[！？!?。…ー]*$',
                                    left.group[-1].text))
            # A long first column can start in a narrow upper lobe and finish
            # beside a relative clause below ("suited to ...", "based on ...").
            aligned_modifier = (len(right.group)==1 and 2<=len(left.group)<=5
                    and right.group[0].text.endswith('に') and re.match(r'(?:合わせ|応じ|適し|沿っ|従っ|基づい|見合っ)た',left.group[0].text)
                    and min(v.confidence for v in boxes)>.90 and -.3*scale<rhs-lhs<.8*scale
                    and scale<left.group[0].bbox[1]-right.group[0].bbox[1]<5*scale
                    and abs(left.group[0].bbox[3]-right.group[0].bbox[3])<1.2*scale)
            clause = clause or aligned_modifier
            # A negative request can occupy its own column after a particle.
            request = (len(right.group)==1 and dependent(right) and suffix(left)
                       and left.group[0].text.startswith('しないで') and -.15*scale<rhs-lhs<.6*scale
                       and stagger<.6*scale and max(widths)<2*scale)
            # A noun and the following time clause can be split at a balloon lobe.
            last=right.group[-1]
            time_clause=(len(right.group)>=2 and len(left.group)>=2 and len(boxes)<=8
                         and re.fullmatch(r'[一-鿿々]{2,6}',last.text)
                         and re.fullmatch(r'(?:始まる|終わる|来る|戻る|帰る|届く|着く)まで(?:に)?',left.group[0].text)
                         and -.15*scale<rhs-lhs<.55*scale
                         and abs(last.bbox[1]-left.group[0].bbox[1])<.5*scale)
            if (lead_in(right) or lead_in(left)) and not (prefix or clause or request):
                continue
            attached = attached or modifier or prefix or explanation or clause or request or time_clause
            if (genitive(right) or genitive(left)) and not modifier:
                continue
            if not (clause or request) and any(len(g.group)==1 and dependent(g) and not (lead_in(g) or suffix(g) or genitive(g))
                                  for g in (right,left)):
                continue
            if suffix(right) or (not attached and (suffix(left) or not
                    (0<rhs-lhs<2*scale and max(widths)<1.6*scale and .5*scale<stagger<1.4*scale))):
                continue
            x,y,r,b = (min(v.bbox[0] for v in boxes),min(v.bbox[1] for v in boxes),
                       max(v.bbox[2] for v in boxes),max(v.bbox[3] for v in boxes))
            conflicts = crop_conflicts((x,y,r,b),tuple(v.box_id for v in boxes),detections)
            if conflicts:
                if not (explanation or time_clause) or any(c.reason=='invalid_nonmember_geometry' for c in conflicts):
                    continue
                # Aligned columns of different lengths can have applause in an
                # empty corner of their rectangle. Keep it outside their hull.
                hull = cv2.convexHull(np.float32([xy for box in boxes for xy in _rectangle(box.bbox)]))
                if any(cv2.intersectConvexConvex(hull,np.float32(c.polygon))[0]>0 for c in conflicts):
                    continue
            pad = min(128,3*scale)
            l,t = max(0,x-pad),max(0,y-pad)
            paper = pixels[t:b+pad+1,l:r+pad+1].min(axis=2)>240
            if paper.size>512*512:
                continue
            _,labels = cv2.connectedComponents(paper.astype(np.uint8),connectivity=4)
            owners = set()
            for box in boxes:
                x1,y1,x2,y2 = box.bbox
                tile = labels[y1-t:y2-t+1,x1-l:x2-l+1]
                votes = np.bincount(tile.ravel())
                votes[0] = 0
                owners.add(int(votes.argmax()) if votes.max()>.5*tile.size else 0)
            if len(owners)!=1 or 0 in owners:
                continue
            surface = labels==owners.pop()
            open_edges = sum(bool(edge.any()) for edge in (surface[0],surface[-1],surface[:,0],surface[:,-1]))
            if open_edges:
                # A genitive and its noun can share a balloon clipped on one side.
                # Require uninterrupted paper directly between their actual columns.
                seam = surface[max(right.group[0].bbox[1],y)-t:min(right.group[0].bbox[3],b)-t,
                               lhs-l+1:rhs-l]
                if not (modifier or prefix or explanation) or open_edges!=1 or not seam.size or seam.mean()<.95:
                    continue
            # Both the ceiling and floor bend inward between balloon lobes.
            top,bottom = surface.argmax(0),len(surface)-surface[::-1].argmax(0)
            ca = int(np.median([v.bbox[0]+v.bbox[2] for v in left.group])/2)-l
            cb = int(np.median([v.bbox[0]+v.bbox[2] for v in right.group])/2)-l
            if not attached and (top[lhs-l:rhs-l+1].max()<max(top[ca],top[cb])+scale*.12
                    or bottom[lhs-l:rhs-l+1].min()>min(bottom[ca],bottom[cb])-scale*.12):
                continue
            result[result.index(right)] = Text_Group(boxes,method='shared_balloon_columns',direction='vertical')
            result.remove(left)
    return result


def join_terminal_marks(groups, pixels, detections, regions, by_id, excluded):
    """Attach a punctuation column immediately following aligned vertical dialogue."""
    pairs = []
    for mark_group in groups:
        if len(mark_group.group) != 1:
            continue
        mark = mark_group.group[0]
        text = unicodedata.normalize('NFKC',mark.text.strip())
        if (mark.box_id in excluded or not 1 <= len(text) <= 12 or not any(c in '!?' for c in text)
                or any(c not in '.!?' for c in text)):
            continue
        x,y,r,b = mark.bbox
        for group in groups:
            if group.direction != 'vertical' or group is mark_group:
                continue
            last = min(group.group,key=lambda box:box.bbox[0])
            lx,ly,lr,lb = last.bbox
            ids = tuple(box.box_id for box in group.group)+(mark.box_id,)
            if (not excluded.isdisjoint(ids) or not any(_kana(box.text) or _kanji(box.text) for box in group.group)
                    or b-y < 1.6*(r-x) or not .65 <= (r-x)/max(1,lr-lx) <= 1.4
                    or not 0 <= lx-r <= .45*(r-x) or abs(y-ly) > .35*(r-x)
                    or min(b,lb)-max(y,ly) < .85*min(b-y,lb-ly)):
                continue
            owners = {regions[by_id[key][2]][0] for key in ids if by_id[key][2] in regions}
            bounds = (x,min(y,*(box.bbox[1] for box in group.group)),
                max(box.bbox[2] for box in group.group),max(b,*(box.bbox[3] for box in group.group)))
            seam = pixels[max(y,ly):min(b,lb),r:lx]
            if (len(owners) <= 1 and not crop_conflicts(bounds,ids,detections)
                    and seam.size and np.mean(seam.min(axis=2)>=235) >= .96):
                pairs.append((group,mark_group))
    result = list(groups)
    for group,mark in pairs:
        if sum(g is group or m is mark for g,m in pairs) == 1:
            result[result.index(group)] = Text_Group(group.group+mark.group,method='terminal_punctuation',direction='vertical')
            result.remove(mark)
    return result


def stacked_vertical_clauses(groups, pixels, detections, regions, by_id, excluded):
    """Join aligned vertical tiers when grammar and paper support one passage."""
    blocks = []
    for group in groups:
        stammer = len(group.group)==1 and re.fullmatch(r'[ぁ-ゖァ-ヺ][っッ]',group.group[0].text)
        if group.direction != 'vertical' or not (stammer or 2 <= len(group.group) <= 5):
            continue
        ids = tuple(box.box_id for box in group.group)
        if any(i in excluded for i in ids):
            continue
        boxes = [box.bbox for box in group.group]
        widths = [r-x for x,y,r,b in boxes]
        if min(widths) < 6 or max(widths) > 1.45*min(widths):
            continue
        bounds = (min(v[0] for v in boxes),min(v[1] for v in boxes),
            max(v[2] for v in boxes),max(v[3] for v in boxes))
        blocks.append((group,ids,bounds,min(widths)))
    pairs = []
    for upper,ids,(x,y,r,b),scale in blocks:
        ending = upper.group[-1].text.rstrip()
        subject = bool(re.search(r'(?:のは|[一-鿿]が)$', ending))
        stammer = len(ids)==1 and re.fullmatch(r'[ぁ-ゖァ-ヺ][っッ]',ending)
        if not stammer and not subject and not ending.endswith(('だろうと','であろうと','であっても')):
            continue
        for lower,other,(lx,ly,lr,lb),other_scale in blocks:
            if len(other)<2 or (stammer and not lower.group[0].text.startswith(ending[0])):
                continue
            gap = ly-b
            left,right = max(x,lx),min(r,lr)
            if not (2 <= gap <= (1.6 if subject else 1)*scale and .75 <= other_scale/scale <= 1.33
                    and right-left >= (.65 if subject else .8)*min(r-x,lr-lx)
                    and abs((x+r)-(lx+lr)) <= (3.6 if subject else 1.5)*scale):
                continue
            members=ids+other
            known={regions[by_id[i][2]][0] for i in members if by_id[i][2] in regions}
            if subject and (len(known) != 1 or not all(any(by_id[i][2] in regions for i in tier) for tier in (ids, other))):
                continue
            if len(known)>1 or crop_conflicts((min(x,lx),y,max(r,lr),lb),members,detections):
                continue
            seam = pixels[b+1:ly,left:right]
            if seam.size and (seam.min(axis=2)>=245).mean() >= .98:
                method = 'stacked_stammer' if stammer else 'stacked_vertical_subject' if subject else 'stacked_vertical_concessive'
                pairs.append((upper,lower,method))
    result=list(groups)
    for upper,lower,method in pairs:
        if any(sum(g in pair[:2] for pair in pairs)!=1 for g in (upper,lower)):
            continue
        result[result.index(upper)]=Text_Group(upper.group+lower.group,method=method,direction='vertical')
        result.remove(lower)
    return result


def join_horizontal_continuations(groups, pixels, detections, regions, by_id, excluded, clause_parents):
    """Bind a grammatical continuation across aligned rows on the same paper."""
    pairs = []
    for upper in groups:
        last = upper.group[-1]
        obligation = last.text.rstrip().endswith(('ないと','なくては','なければ'))
        object_phrase = last.text.rstrip().endswith('を')
        short_topic = len(upper.group) == 1 and re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿]{1,10}は',last.text.strip())
        quoted_cause = (upper.group[0].text.startswith(('「','『')) and last.text.endswith(('ので','ため','から'))
                        and not any('」' in row.text or '』' in row.text for row in upper.group))
        presented = re.search(r'([一-鿿々])があれば[!！]$',last.text)
        modifier = len(upper.group)>=2 and re.search(r'[一-鿿々]{2,}な$',last.text)
        if upper.direction != 'horizontal' or last.confidence < .90 or not (obligation or object_phrase or short_topic or presented or modifier or quoted_cause or last.box_id in clause_parents):
            continue
        x,y,r,b = last.bbox
        topic_owner = regions.get(by_id[last.box_id][2], (None,))[0] if short_topic else None
        for lower in groups:
            first = lower.group[0]
            emphatic = first.box_id in clause_parents.get(last.box_id, ())
            # A short topic above a wider multi-line predicate is often split by
            # the detector. Require every row to share a measured balloon interior.
            same_topic = (topic_owner and len(lower.group) >= 2 and
                          all(regions.get(by_id[box.box_id][2],(None,))[0] == topic_owner for box in lower.group))
            # An attributive adjective cannot end the sentence before its noun.
            # Require left-aligned body type of equal size across both blocks.
            bound_noun = (modifier and len(lower.group)>=2 and re.match(r'[ァ-ヺー一-鿿々]{2,}',first.text)
                          and all(abs(v.bbox[0]-x)<.35*(b-y) and .8*(b-y)<=v.bbox[3]-v.bbox[1]<=1.2*(b-y)
                                  for v in upper.group+lower.group))
            quoted_tail = (quoted_cause and lower.group[-1].text.endswith('」' if upper.group[0].text.startswith('「') else '』')
                           and all(not any(mark in v.text for mark in '「『') for v in lower.group)
                           and all(abs(v.bbox[0]-x)<.62*(b-y) and .78*(b-y)<=v.bbox[3]-v.bbox[1]<=1.32*(b-y)
                                   for v in upper.group+lower.group))
            introduction = (presented and re.search(presented[1]+r'です[!！。]*$',first.text)
                            and len(upper.group)==len(lower.group)==1)
            if (lower is upper or lower.direction != 'horizontal' or first.confidence < .90
                    or not (emphatic or same_topic or bound_noun or quoted_tail or introduction or (object_phrase and first.text.rstrip('。！').endswith(('ます','ましょう','ください'))) or (obligation and first.text.lstrip().startswith(('いけな','ならな'))))):
                continue
            lx,ly,lr,lb = first.bbox
            height = b-y
            left,right = max(lx,x),min(lr,r)
            if (not 1 <= ly-b <= height*.6 or not .75 <= (lb-ly)/height <= (1.85 if emphatic or introduction else 1.33)
                    or right-left < .8*min(r-x,lr-lx) or (not (bound_noun or quoted_tail) and abs((lx+lr)-(x+r)) > height and not (introduction and abs(lr-r)<=height*.4))):
                continue
            boxes = upper.group+lower.group
            ids = tuple(box.box_id for box in boxes)
            known = {regions[by_id[i][2]][0] for i in ids if by_id[i][2] in regions}
            bounds = (min(box.bbox[0] for box in boxes),min(box.bbox[1] for box in boxes),
                      max(box.bbox[2] for box in boxes),max(box.bbox[3] for box in boxes))
            seam = pixels[b:ly,left:right]
            if (excluded.isdisjoint(set(ids) - ({last.box_id} if emphatic else set())) and len(known)<=1 and not crop_conflicts(bounds,ids,detections)
                    and (seam.min(2)>240).mean()>=.97):
                pairs.append((upper,lower))
    result = list(groups)
    for upper,lower in pairs:
        if all(sum(g in pair for pair in pairs)==1 for g in (upper,lower)):
            result[result.index(upper)] = Text_Group(upper.group+lower.group,
                method='horizontal_clause_continuation',direction='horizontal')
            result.remove(lower)
    return result


def horizontal_rule_between(first, second, pixels):
    """Find a thin, near-horizontal printed rule crossing overlapping row boxes."""
    a,b=sorted((first.bbox,second.bbox),key=lambda box:box[1]+box[3])
    h=min(a[3]-a[1],b[3]-b[1])
    left,right=max(a[0],b[0]),min(a[2],b[2])
    if h < 6 or right-left < max(48,3*h) or b[1]-a[3] > 2*h or b[1] < a[1]+.45*h:
        return False
    # Sample across the shared row width. The strip may tilt slightly in scans.
    top=max(0,int((a[1]+a[3])/2)+1)
    bottom=min(pixels.shape[0],int((b[1]+b[3])/2))
    left,right=max(left,0),min(right,pixels.shape[1])
    if right-left < 48 or bottom-top < 3:
        return False
    ink=pixels[top:bottom,left:right].max(axis=2)<110
    height,width=ink.shape
    slopes=min(6,max(1,round(.02*width)))
    xs=np.unique(np.linspace(0,width-1,min(256,width)).astype(int))
    for rise in range(-slopes,slopes+1):
        offsets=np.rint(xs*rise/(width-1)).astype(int)
        ys=np.arange(2-min(0,rise),height-2-max(0,rise))[:,None]+offsets
        dark=ink[ys,xs].mean(axis=1)
        light=np.maximum((~ink[ys-2,xs]).mean(axis=1),(~ink[ys+2,xs]).mean(axis=1))
        if ((dark >= .90) & (light >= .7)).any():
            return True
    return False


def upright_clause_beside_slanted_type(child, parent):
    # Small upright dialogue can abut much larger hand lettering on a slant.
    # A kana-only parent and two different baselines give no support for ruby.
    if (child.confidence<.90 or len(child.text)<3 or not re.search(r'[ぁ-ゖァ-ヺ]',child.text)
            or not re.fullmatch(r'[ぁ-ゖァ-ヺー]+',parent.text)):
        return False
    def tilt(box):
        edges = np.roll(box.poly,-1,axis=0)-box.poly
        dx,dy = edges[np.linalg.norm(edges,axis=1).argmax()]
        return math.degrees(math.atan2(abs(dx),abs(dy)))
    return tilt(child)<7 and 12<=tilt(parent)<=44


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


def heading_above_slanted_rows(child, parent, boxes):
    """Two intervening baselines rule out a phonetic gloss on the proposed parent."""
    if len(parent.poly)!=4 or len(child.poly)!=4 or GroupingMixin._manga_direction(parent)!='horizontal':
        return False
    edges=np.roll(parent.poly,-1,axis=0)-parent.poly
    dx,dy=max(edges,key=lambda edge:abs(edge[0]))
    if not .12*abs(dx)<abs(dy)<.5*abs(dx):
        return False
    normal=np.array([-dy/dx,1.])
    main,head=parent.poly@normal,child.poly@normal
    height=np.ptp(main)
    if main.min()-head.max()<1.3*height or not .65*height<np.ptp(head)<1.5*height:
        return False
    between=[]
    for row in boxes:
        if row is child or row is parent or len(row.poly)!=4 or abs(row.bbox[0]-child.bbox[0])>height:
            continue
        projected=row.poly@normal
        if (row.bbox[2]-row.bbox[0]>2.8*height and .55*height<np.ptp(projected)<1.6*height
                and head.mean()+.5*height<projected.mean()<main.mean()-.5*height):
            between.append(projected.mean())
    return len(between)>=2 and max(between)-min(between)>.65*height


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
                            regroup_companions=False, policy=POLICY):
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
    if policy not in (POLICY,LEGACY_POLICY):raise ValueError('unknown automatic structure policy')
    inverse,dark_runs=ink.reverse_lettering(boxes) if policy==POLICY else (ink,{})
    for source_id,(label,support,glyph) in dark_runs.items():
        glyphs[source_id]=glyph
        regions[by_id[source_id][2]]=(-1_000_000-label,support)

    def foreground_for(*ids):
        return inverse if all(i in dark_runs for i in ids) else ink
    possibilities, relationship_reasons, rejected_relationships = {}, {}, []
    relation_evidence = {}
    single_context, single_supports = None, {}
    credit_rows = {raw.source_id for raw, _ in ordered if policy == POLICY and is_credit_list(raw.text)}
    literal_rows = {raw.source_id for raw, _ in ordered if policy == POLICY and is_printed_literal(raw.text)}
    # A separately discovered beat in a repeated trail can be small enough
    # to resemble a reading gloss above the next sound.
    beats = {raw.source_id:raw.text for raw, _ in ordered if policy == POLICY
             and raw.reason == 'frontier_discovery' and re.fullmatch(r'[ァ-ヺぁ-ゖ]', raw.text)}
    frequencies = Counter(beats.values())
    separate_rows = (literal_rows | {raw.source_id for raw,_ in ordered if policy==POLICY and raw.reason in ('confirmed_tab_label','confirmed_checkbox_item')}
                     | {i for i, text in beats.items() if frequencies[text] >= 4}
                     | {raw.source_id for raw,_ in ordered if raw.reason in ('confirmed_trailing_kana','confirmed_handwritten_line','corroborated_repeated_effect','corroborated_printed_twin','corroborated_mask_line','corroborated_native_line')})
    independent_rows = credit_rows | separate_rows
    for child, child_box in ordered:
        options, rejected_options, evidence_rows = [], [], []
        for parent, parent_box in ordered:
            if child.source_id in independent_rows or parent.source_id in independent_rows:
                continue
            if child.source_id == parent.source_id or not _annotation_geometry(child_box, parent_box):
                continue
            if policy==POLICY and upright_clause_beside_slanted_type(child_box,parent_box):
                continue
            ci, pi = by_id[child.source_id][2], by_id[parent.source_id][2]
            child_glyphs, parent_glyphs = glyphs[child.source_id], glyphs[parent.source_id]
            boundary = foreground_for(child.source_id,parent.source_id).boundary_between(child_box,parent_box,
                first_glyphs=child_glyphs,second_glyphs=parent_glyphs)
            rejects = []
            if (child.source_id in dark_runs)!=(parent.source_id in dark_runs):
                rejects.append("different_original_lettering_polarity")
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
            # A complete inflected horizontal clause can sit above a larger
            # emphatic line, in the same position as a phonetic gloss.
            wording = unicodedata.normalize("NFKC", child.text).strip()
            if (direction == "horizontal" and child.confidence >= .9 and len(wording) >= 4
                    and _kanji(wording) and wording.endswith(("は", "が", "を", "には", "では"))):
                rejects.append("horizontal_clause_with_case_particle")
            # A punctuated greeting above a kana title is independent lettering.
            # Its small size and placement alone can resemble phonetic ruby.
            if (policy == POLICY and not rejects and direction == 'horizontal' and wording.endswith('!')
                    and len(wording) >= 4 and _kana(wording) and _kana(parent.text)
                    and not _kanji(parent.text) and child.confidence >= .70 and parent.confidence >= .70):
                rejects.append('punctuated_horizontal_heading')
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
    headings={}
    if policy==POLICY:
        for raw,_ in ordered:
            if 'forum_post' in raw.metadata_json:
                note=json.loads(raw.metadata_json)['forum_post']
                if note.get('heading') and note.get('body_id') in by_id:
                    headings[raw.source_id]=note['body_id']
        for child,parent in list(annotations.items()):
            if heading_above_slanted_rows(by_id[child][1],by_id[parent][1],boxes):
                headings[child]=annotations.pop(child)
                relationships=[RelationshipProposal(child,None,(parent,),'rejected_annotation_candidate',
                    ('intervening_printed_rows',),'separate_baselines',r.image_evidence)
                    if r.source_id==child and r.parent_id==parent else r for r in relationships]
    mains = [pair for pair in ordered if pair[0].source_id not in annotations and pair[0].source_id not in headings]
    main_regions = {i: regions[by_id[raw.source_id][2]] for i, (raw, _) in enumerate(mains)
                    if by_id[raw.source_id][2] in regions}
    rejected_merges = {}
    boundary_cache = {}
    rule_cache = {}
    def merge_guard(first,second):
        members = tuple(sorted((box.box_id for box in first+second),key=_key))
        reasons, barriers = [], []
        if separate_rows.intersection(members) or (credit_rows.intersection(members) and not credit_rows.issuperset(members)):
            return False
        if any(i in dark_runs for i in members) and not all(i in dark_runs for i in members):
            return False
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
                    boundary_cache[key] = foreground_for(*pair).boundary_between(left,right,
                        first_glyphs=glyphs[left.box_id],second_glyphs=glyphs[right.box_id],
                        included_members=included,nonmember_boxes=nonmembers)
                evidence = boundary_cache[key]
                if pair not in rule_cache:
                    rule_cache[pair] = (GroupingMixin._manga_direction(left) == GroupingMixin._manga_direction(right) == 'horizontal'
                        and horizontal_rule_between(left,right,bgr))
                if rule_cache[pair]:
                    reasons.append("printed_rule_between_horizontal_rows")
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
                              merge_guard=merge_guard, original_pixels=bgr)
    ownership_repairs = ()
    if complementary_ownership:
        member_groups = []
        for group in groups:
            group_ids = {box.box_id for box in group.group}
            member_groups.append((tuple(by_id[box.box_id][2] for box in group.group),
                tuple(by_id[child][2] for child in sorted(
                    (child for child,parent in annotations.items() if parent in group_ids),key=_key))))
        regions,ownership_repairs = ink.complementary_ownership(boxes,member_groups,regions)
    if complementary_ownership and policy==POLICY:
        for mains, children in member_groups:
            known = {regions[i][0] for i in mains if i in regions}
            if (len(children)!=1 or children[0] in regions or len(mains)<3
                    or len(known)>1 or (known and any(i not in regions for i in mains))
                    or any(boxes[i].confidence < .90 for i in mains)
                    or len(possibilities[boxes[children[0]].box_id])!=1):
                continue
            child = boxes[children[0]]
            parent = by_id[annotations[child.box_id]][1]
            x,y,r,b = child.bbox
            px,py,pr,pb = parent.bbox
            if (len(child.text.strip())!=1 or parent.confidence<.9
                    or not (px<x<pr and py<=y<b<=pb and r-x <= .5*(pr-px))):
                continue
            if single_context is None:
                single_context = ink.single_glyph_context()
            support = ink.enclosed_annotation_support(boxes,mains,children,single_context)
            if support:
                # A discovery polygon can break the mask used to assign regions
                # even though the unmodified page still encloses every column.
                owner = next(iter(known)) if known else -2_000_000-support[0][1]
                confidence = min(row[2] for row in support)
                for i in mains + children:
                    regions.setdefault(i,(owner,confidence))
                ownership_repairs += ({"method":"shared_original_annotation_enclosure",
                    "main_ids":tuple(boxes[i].box_id for i in mains),
                    "member_ids":tuple(boxes[i].box_id for i in mains+children),
                    "region_id":owner,"source_evidence":support},)
    repaired_ids = {i for repair in ownership_repairs for i in repair["member_ids"]}
    companion_regroupings,companion_rejections = (),()
    if regroup_companions:
        groups,companion_regroupings,companion_rejections = _regroup_companions(
            groups,regions,by_id,glyphs,ink,annotations,detections,relationships,single_context)
    if not unassigned:
        handwriting = {}
        for group in groups:
            if len(group.group)==1:
                box = group.group[0]
                raw = by_id[box.box_id][0]
                if raw.reason=='confirmed_handwritten_line':
                    evidence = json.loads(raw.metadata_json)
                    handwriting.setdefault(evidence['handwritten_anchor'],[]).append((evidence['line_index'],group))
        for sequence in handwriting.values():
            ordered_lines = [g for index,g in sorted(sequence,key=lambda pair:pair[0])]
            if len(ordered_lines)>1:
                groups = [g for g in groups if g not in ordered_lines]
                groups.append(Text_Group([g.group[0] for g in ordered_lines],method='corroborated_handwritten_lines',direction='horizontal'))
        excluded=set(annotations) | set(annotations.values()) | set(relationship_reasons) | set(dark_runs)
        groups=table_rows.split(groups,bgr,excluded,confirmed={d.source_id for d in detections if d.reason in ('confirmed_printed_cell','confirmed_tab_label')})
        groups=join_reread_columns(groups,by_id)
        marked = {d.source_id for d in detections if 'printed_onset' in d.metadata_json}
        if marked:
            groups=join_stuttered_onsets(groups,bgr,excluded,marked)
        groups=join_shared_balloon(groups,bgr,detections,excluded)
        groups=join_terminal_marks(groups,bgr,detections,regions,by_id,excluded)
        groups=stacked_vertical_clauses(groups,bgr,detections,regions,by_id,excluded)
        clause_parents = {child: tuple(row["parent_id"] for row in rows
                if row["rejected_reasons"] == ("horizontal_clause_with_case_particle",))
            for child,rows in relation_evidence.items() if relationship_reasons.get(child)==("horizontal_clause_with_case_particle",)}
        groups=join_horizontal_continuations(groups,bgr,detections,regions,by_id,excluded,clause_parents)
        for raw,_ in ordered:
            if not any(key in raw.metadata_json for key in ('corroborated_columns','curved_repeat','masked_passage')):
                continue
            metadata = json.loads(raw.metadata_json)
            curved = metadata.get('curved_repeat')
            masked = metadata.get('masked_passage')
            evidence = curved or masked or metadata.get('corroborated_columns')
            ids = evidence.get('ids',[]) if isinstance(evidence,dict) else evidence
            if not ids or len(set(ids))!=len(ids) or ids[0]!=raw.source_id or not set(ids).isdisjoint(annotations):
                continue
            matched = [g for g in groups if any(b.box_id in ids for b in g.group)]
            if {b.box_id for g in matched for b in g.group}==set(ids):
                position = groups.index(matched[0])
                groups = [g for g in groups if g not in matched]
                groups.insert(position,Text_Group([by_id[i][1] for i in ids],
                                         method='curved_repeat' if curved else 'corroborated_mask_lines' if masked else 'corroborated_handwritten_columns',
                                         direction='horizontal' if curved or (masked and masked.get('direction')=='horizontal') else 'vertical'))
    # A newly discovered effect beside an already lettered logo needs its own erase region.
    independent = []
    for group in groups:
        if len(group.group) == 2:
            first, second = sorted(group.group, key=lambda box: box.bbox[0])
            originals = [by_id[box.box_id][0] for box in (first, second)]
            if (first.bbox[2] <= second.bbox[0]
                    and any(raw.reason == 'frontier_discovery' for raw in originals)
                    and any(raw.status == 'accepted' and raw.text.isascii() and raw.text.isalpha()
                            and len(raw.text) >= 3 for raw in originals)):
                independent.extend(Text_Group([box], method='singleton', direction=group.direction)
                                   for box in group.group)
                continue
        independent.append(group)
    for child,parent in headings.items():
        position=next((i for i,g in enumerate(independent) if any(b.box_id==parent for b in g.group)),len(independent))
        independent.insert(position,Text_Group([by_id[child][1]],method='singleton',direction='horizontal'))
    proposals = []
    for group in independent:
        main_ids = tuple(box.box_id for box in group.group)
        # A two-line heading that overlaps its title can be sorted by x by
        # the mixed-layout grouper. Read the independently identified heading first.
        if group.direction == 'horizontal' and len(main_ids) == 2:
            title, heading = main_ids
            if any(row['parent_id'] == title and 'punctuated_horizontal_heading' in row['rejected_reasons']
                   for row in relation_evidence[heading]):
                main_ids = (heading, title)
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
        if all(i in dark_runs for i in main_ids):
            structural.append("original_light_lettering_on_dark_surface")
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
        identity = {"policy": policy, "page": asdict(page), "main_ids": main_ids,
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
        policy=policy,ownership_policy=COMPLEMENTARY_OWNERSHIP_POLICY if complementary_ownership else "production8",
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
