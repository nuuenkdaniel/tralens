"""Original-pixel structural evidence for opt-in automatic proposals only."""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .grouping import GroupingMixin


@dataclass(frozen=True)
class GlyphEvidence:
    component_count: int
    ink_pixels: int
    scale: float
    flow: str
    bounds: tuple


class OriginalInkEvidence:
    """Analysis masks are separate arrays; never edit recognition/original pixels."""
    def __init__(self, bgr):
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, self.ink = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def single_glyph_context(self):
        """Original labels scoped to one fresh proposal calculation, never a cache."""
        _,dark,stats,_ = cv2.connectedComponentsWithStats(self.ink,connectivity=8)
        _,white,white_stats,_ = cv2.connectedComponentsWithStats(cv2.bitwise_not(self.ink),connectivity=8)
        return dark,stats,white,white_stats

    def single_glyph_position(self, child, parent, child_glyphs, parent_glyphs, context, other_boxes=()):
        """Conservative leading vertical-glyph attachment evidence, not transcription.

        A compact connected mark alone is insufficient. Require complete original
        child ink, a nontrivial two-axis stroke pattern, a contained multi-glyph
        parent run, leading-glyph alignment, overlapping detection margins but
        separate actual ink, and the same enclosed original white background.
        """
        dark,stats,white,white_stats = context
        height,width = self.ink.shape

        def mask(box, bounds):
            x,y,r,s = bounds
            local = np.zeros((s-y,r-x),dtype=np.uint8)
            cv2.fillPoly(local,[box.poly.astype(np.int32)-np.array([x,y])],255)
            return local != 0

        def contained(label, box):
            x,y,w,h,area = (int(v) for v in stats[label])
            return int(np.count_nonzero((dark[y:y+h,x:x+w] == label) & mask(box,(x,y,x+w,y+h)))) == area

        result = {"method":"single_leading_glyph_position_v1","supported":False}
        if (child_glyphs.component_count != 1 or parent_glyphs.component_count < 2
                or parent_glyphs.flow != "vertical" or not parent_glyphs.bounds
                or parent_glyphs.bounds[3]-parent_glyphs.bounds[1] < 3*parent_glyphs.scale):
            result["reason"] = "single_glyph_or_parent_run_support_inconclusive"
            return result
        x,y,r,s = child.bbox
        labels = np.unique(dark[y:s,x:r][mask(child,child.bbox)])
        labels = labels[labels != 0]
        if len(labels) != 1 or not contained(int(labels[0]),child):
            result["reason"] = "child_original_ink_not_one_wholly_contained_component"
            return result
        child_label = int(labels[0])
        cx,cy,cw,ch,area = (int(v) for v in stats[child_label])
        if any(box is not child and box is not parent and contained(child_label,box) for box in other_boxes):
            result["reason"] = "child_original_ink_also_contained_by_another_detection"
            return result
        pixels = dark[cy:cy+ch,cx:cx+cw] == child_label
        row_runs = int(np.max(np.sum(pixels & ~np.pad(pixels[:,:-1],((0,0),(1,0))),axis=1)))
        column_runs = int(np.max(np.sum(pixels & ~np.pad(pixels[:-1,:],((1,0),(0,0))),axis=0)))
        result.update(child_component=child_label,child_ink_bounds=(cx,cy,cx+cw,cy+ch),child_ink_pixels=area,
                      maximum_row_runs=row_runs,maximum_column_runs=column_runs)
        if (min(cw,ch) < .25*parent_glyphs.scale or max(cw,ch) > .7*parent_glyphs.scale
                or min(row_runs,column_runs) < 2):
            result["reason"] = "single_mark_shape_is_simple_punctuation_noise_or_comparable_scale"
            return result
        # A ring's white hole must not manufacture apparent stroke complexity.
        # Fill holes in this tiny analysis silhouette, never in source pixels.
        contours,_ = cv2.findContours(pixels.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        silhouette = np.zeros_like(pixels,dtype=np.uint8)
        cv2.drawContours(silhouette,contours,-1,1,cv2.FILLED)
        solid = silhouette != 0
        outer_rows = int(np.max(np.sum(solid & ~np.pad(solid[:,:-1],((0,0),(1,0))),axis=1)))
        outer_columns = int(np.max(np.sum(solid & ~np.pad(solid[:-1,:],((1,0),(0,0))),axis=0)))
        result.update(silhouette_row_runs=outer_rows,silhouette_column_runs=outer_columns)
        if max(outer_rows,outer_columns) < 2:
            result["reason"] = "single_mark_silhouette_is_simple_dot_ring_or_stroke"
            return result
        # Reading annotations tucked into an oversized main detection margin
        # differ from a separate small-dialogue box. Actual ink must not overlap.
        margin_support = float(np.count_nonzero(pixels & mask(parent,(cx,cy,cx+cw,cy+ch)))/area)
        result["parent_detection_margin_ink_support"] = margin_support
        if margin_support < .5:
            result["reason"] = "child_not_in_parent_detection_margin"
            return result
        gx,gy,gr,gs = parent_glyphs.bounds
        # Use the existing supported main-glyph core, not every dark component
        # touched by an oversized OCR rectangle (which can clip adjacent text).
        # Its whole run must have an interior margin in the source polygon.
        contour = parent.poly.astype(np.float32)
        if any(cv2.pointPolygonTest(contour,(float(x),float(y)),False) <= 0
               for x,y in ((gx,gy),(gr,gy),(gr,gs),(gx,gs))):
            result["reason"] = "parent_main_glyph_core_not_interior_to_polygon"
            return result
        start = gy
        end = min(gs,int(np.ceil(start+parent_glyphs.scale)))
        leading = self.ink[start:end,gx:gr] != 0
        ys,xs = np.nonzero(leading)
        if not len(xs) or xs.max()-xs.min()+1 < .5*parent_glyphs.scale or ys.max()-ys.min()+1 < .5*parent_glyphs.scale:
            result["reason"] = "insufficient_original_leading_glyph_ink"
            return result
        bounds = (gx+int(xs.min()),start+int(ys.min()),gx+int(xs.max())+1,start+int(ys.max())+1)
        gap = cx-bounds[2]
        overlap = max(0,min(cy+ch,bounds[3])-max(cy,bounds[1]))/ch
        result.update(parent_leading_glyph_bounds=bounds,parent_glyph_scale=parent_glyphs.scale,
                      leading_axis_overlap=overlap,original_ink_gap=gap)
        if (not 1 <= gap <= .8*parent_glyphs.scale or overlap < .8
                or not start <= cy+ch/2 <= start+parent_glyphs.scale):
            result["reason"] = "child_not_beside_leading_parent_glyph"
            return result
        corridor = self.ink[max(cy,bounds[1]):min(cy+ch,bounds[3]),bounds[2]:cx]
        if not corridor.size or np.any(corridor):
            result["reason"] = "original_ink_crosses_single_glyph_gap"
            return result
        backgrounds = []
        supports = []
        for box in (child,parent):
            x,y,r,s = box.bbox
            values = white[y:s,x:r][mask(box,box.bbox)]
            votes = np.bincount(values,minlength=len(white_stats))
            if not votes[1:].sum():
                result["reason"] = "original_white_enclosure_inconclusive"
                return result
            label = int(np.argmax(votes[1:])+1)
            bx,by,bw,bh,_ = (int(v) for v in white_stats[label])
            support = float(votes[label]/votes[1:].sum())
            if support < .9 or not (bx > 0 and by > 0 and bx+bw < width and by+bh < height):
                result["reason"] = "original_white_enclosure_inconclusive"
                return result
            backgrounds.append(label)
            supports.append(support)
        if len(set(backgrounds)) != 1:
            result["reason"] = "distinct_original_white_enclosures"
            return result
        result.update(supported=True,reason="contained_single_glyph_beside_leading_main_glyph",
                      original_background=backgrounds[0],original_white_support=tuple(supports))
        return result

    def complementary_ownership(self, boxes, member_groups, known_regions):
        """Narrow opt-in ownership evidence using complementary white connectivity.

        An 8-connected dark outline need not separate 8-connected white pixels at
        diagonal contacts. Four-connected white components respect that outline
        without closing gaps, pruning tails or adding/removing any mask pixels.
        Only already-grouped, glyph-supported multi-main dialogue can use this
        fallback. Original unmasked enclosure and every original area/overlap
        predicate remain necessary; rectangular fill is tested on the entire
        four-connected component, never on a hull or a detection-local crop.
        """
        regions = dict(known_regions)
        repairs = []
        member_groups = tuple(member_groups)
        if not any(len(mains) >= 2 and not any(i in regions for i in mains + annotations)
                   for mains, annotations in member_groups):
            return regions, ()
        text_mask = np.zeros_like(self.ink)
        for box in boxes:
            cv2.fillPoly(text_mask, [box.poly.astype(np.int32)], 255)
        free = cv2.bitwise_not(cv2.bitwise_and(self.ink, cv2.bitwise_not(text_mask)))
        components = []
        for mask, connectivity in ((free, 8), (free, 4), (cv2.bitwise_not(self.ink), 8)):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
            components.append((count, labels, stats))
        height, width = self.ink.shape

        def evidence(index, component, *, white_only=False):
            box = boxes[index]
            x1,y1,x2,y2 = box.bbox
            if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
                return None
            count, labels, stats = component
            mask = np.zeros((y2-y1+1,x2-x1+1), dtype=np.uint8)
            cv2.fillPoly(mask, [box.poly.astype(np.int32)-np.array([x1,y1])], 255)
            values = labels[y1:y2+1,x1:x2+1][mask != 0]
            votes = np.bincount(values, minlength=count)
            if count <= 1 or not votes[1:].sum():
                return None
            label = int(np.argmax(votes[1:])+1)
            denominator = votes[1:].sum() if white_only else len(values)
            overlap = float(votes[label]/denominator)
            x,y,w,h,area = (int(v) for v in stats[label])
            box_area = (x2-x1)*(y2-y1)
            if (overlap < .9 or not (x > 0 and y > 0 and x+w < width and y+h < height)
                    or (white_only and votes[1:].sum() < .5*len(values))
                    or (not white_only and not box_area <= area <= 30*box_area)):
                return None
            return {"label": label, "bbox": (x,y,w,h), "area": area,
                    "fill": area/(w*h), "overlap": overlap}

        for mains, annotations in member_groups:
            members = mains + annotations
            if len(mains) < 2 or any(i in regions for i in members):
                continue
            direction = GroupingMixin._manga_direction(boxes[mains[0]])
            glyphs = {i: self.glyphs(boxes[i]) for i in mains}
            if (direction == "unknown" or any(GroupingMixin._manga_direction(boxes[i]) != direction
                    or glyphs[i].component_count < 2 or glyphs[i].flow != direction for i in mains)):
                continue
            pairs = [(i,j) for n,i in enumerate(mains) for j in mains[n+1:]]
            boundaries = tuple({"source_ids": (boxes[i].box_id,boxes[j].box_id),
                "evidence": self.boundary_between(boxes[i],boxes[j],first_glyphs=glyphs[i],
                    second_glyphs=glyphs[j])} for i,j in pairs)
            if (any(not GroupingMixin()._manga_compatible(boxes[i],boxes[j],relaxed=True) for i,j in pairs)
                    or any(row["evidence"]["blocked"]
                           or row["evidence"]["reason"] == "no_resolvable_facing_corridor"
                           for row in boundaries)):
                continue
            rows = [(evidence(i,components[0]), evidence(i,components[1]),
                     evidence(i,components[2],white_only=True)) for i in members]
            if any(old is None or core is None or original is None or old["fill"] >= .55
                   or core["fill"] < .55 or core["area"] >= old["area"]
                   for old,core,original in rows):
                continue
            if any(len({row[column]["label"] for row in rows}) != 1 for column in range(3)):
                continue
            # Positive IDs remain the legacy eight-connected labels. Negative
            # IDs name separate four-connected components without collisions.
            region_id = -rows[0][1]["label"]
            for i,(_,core,_) in zip(members,rows):
                regions[i] = (region_id,core["overlap"])
            repairs.append({"method": "enclosed_complementary_connectivity_v1",
                "main_ids": tuple(boxes[i].box_id for i in mains),
                "member_ids": tuple(boxes[i].box_id for i in members), "region_id": region_id,
                "main_glyphs": tuple({"source_id": boxes[i].box_id, **asdict(glyphs[i])} for i in mains),
                "boundary_checks": boundaries,
                "source_evidence": tuple({"source_id": boxes[i].box_id,
                    "production8": old, "complementary4": core, "unmasked8": original}
                    for i,(old,core,original) in zip(members,rows))})
        return regions, tuple(repairs)

    def glyphs(self, box):
        a, b, c, d = box.bbox
        local = self.ink[b:d, a:c].copy()
        mask = np.zeros_like(local)
        cv2.fillPoly(mask, [np.rint(box.poly - np.array([a,b])).astype(np.int32)], 255)
        local = cv2.bitwise_and(local, mask)
        count, _, stats, centers = cv2.connectedComponentsWithStats(local, connectivity=8)
        minor = min(c-a, d-b)
        # Exclude specks and long outline/connector strokes. This is glyph-scale
        # support, not OCR or a proof that a connected component is a character.
        components = [i for i in range(1,count) if stats[i,4] >= 2
                      and max(stats[i,2],stats[i,3]) <= 1.8*minor
                      and min(stats[i,2],stats[i,3]) >= 1]
        if not components:
            return GlyphEvidence(0, 0, 0., "unknown", ())
        sizes = [max(stats[i,2],stats[i,3]) for i in components]
        scale = float(np.percentile(sizes,75))
        large = [i for i in components if max(stats[i,2],stats[i,3]) >= .7*scale]
        xs, ys = [centers[i,0] for i in large], [centers[i,1] for i in large]
        dx, dy = max(xs)-min(xs), max(ys)-min(ys)
        flow = ("horizontal" if dx > max(scale,1.5*dy) else
                "vertical" if dy > max(scale,1.5*dx) else "unknown")
        bounds = (a+min(stats[i,0] for i in large), b+min(stats[i,1] for i in large),
                  a+max(stats[i,0]+stats[i,2] for i in large), b+max(stats[i,1]+stats[i,3] for i in large))
        return GlyphEvidence(len(components), int(sum(stats[i,4] for i in components)), scale, flow,
                             tuple(int(v) for v in bounds))

    def _included_row_glyph_component(self, labels, stats, label, scan, corridor,
                                      included_members, nonmember_boxes):
        """Attribute one complete original component, never clear a rectangle/ink mask."""
        sx,sy,ex,ey = scan
        x,y,w,h,area = (int(v) for v in stats[label])
        # A scan-edge fragment is not an isolated component, even if an OCR box
        # contains the fragment. Global labels below also check that it is whole.
        if x <= 0 or y <= 0 or x+w >= ex-sx or y+h >= ey-sy:
            return None
        pixels = labels[y:y+h,x:x+w] == label
        yy,xx = np.nonzero(pixels)
        gx,gy = sx+x,sy+y
        if not hasattr(self,"_boundary_dark_components"):
            _,global_labels,global_stats,_ = cv2.connectedComponentsWithStats(self.ink,connectivity=8)
            self._boundary_dark_components = global_labels,global_stats
        global_labels,global_stats = self._boundary_dark_components
        original_label = int(global_labels[gy+int(yy[0]),gx+int(xx[0])])
        if tuple(int(v) for v in global_stats[original_label]) != (gx,gy,w,h,area):
            return None
        # Conservative reuse of original glyph-shape guards: bars and isolated
        # outlines/rings are not evidence for suppressing a divider.
        row_runs = int(np.max(np.sum(pixels & ~np.pad(pixels[:,:-1],((0,0),(1,0))),axis=1)))
        column_runs = int(np.max(np.sum(pixels & ~np.pad(pixels[:-1,:],((1,0),(0,0))),axis=0)))
        if min(row_runs,column_runs) < 2:
            return None
        contours,_ = cv2.findContours(pixels.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        silhouette = np.zeros_like(pixels,dtype=np.uint8)
        cv2.drawContours(silhouette,contours,-1,1,cv2.FILLED)
        solid = silhouette != 0
        # An isolated outline enclosing other original ink is not a glyph proof.
        if np.any(solid & ~pixels & (self.ink[gy:gy+h,gx:gx+w] != 0)):
            return None
        outer_rows = int(np.max(np.sum(solid & ~np.pad(solid[:,:-1],((0,0),(1,0))),axis=1)))
        outer_columns = int(np.max(np.sum(solid & ~np.pad(solid[:-1,:],((1,0),(0,0))),axis=0)))
        if max(outer_rows,outer_columns) < 2:
            return None
        # Actual nonmember pixels, not overlapping rectangles, veto exclusion.
        for box in nonmember_boxes:
            if box.bbox[2] < gx or box.bbox[0] > gx+w or box.bbox[3] < gy or box.bbox[1] > gy+h:
                continue
            polygon = box.poly.astype(np.float32)
            if any(cv2.pointPolygonTest(polygon,(float(gx+a),float(gy+b)),False) >= 0 for a,b in zip(xx,yy)):
                return None
        for box,glyphs in included_members:
            if (glyphs.component_count < 2 or glyphs.flow != "horizontal" or not glyphs.bounds
                    or glyphs.scale <= 0):
                continue
            a,b,r,s = glyphs.bounds
            # Only an intervening core in the unextended facing corridor. Bounds
            # are half-open pixel extents: touching an extent is not crossing it.
            between = corridor[1] <= b < s <= corridor[3]
            minor = min(box.bbox[2]-box.bbox[0],box.bbox[3]-box.bbox[1])
            if (not between or not (a <= gx and b <= gy and gx+w <= r and gy+h <= s)
                    or area < 2 or max(w,h) > 1.8*minor):
                continue
            if not all(cv2.pointPolygonTest(box.poly.astype(np.float32),(float(gx+u),float(gy+v)),False) > 0 for u,v in zip(xx,yy)):
                continue
            return {"member_id":box.box_id,"original_component_id":original_label,
                    "component_bounds":(gx,gy,gx+w,gy+h),"component_area":area,
                    "supported_core":glyphs.bounds,"row_runs":row_runs,"column_runs":column_runs,
                    "silhouette_row_runs":outer_rows,"silhouette_column_runs":outer_columns}
        return None

    def boundary_between(self, first, second, *, first_glyphs=None, second_glyphs=None,
                         included_members=(), nonmember_boxes=()):
        """Veto long original dark strokes in the facing corridor, even with a shared region.

        No OCR-mask removal: blank detections cannot erase the boundary. Use
        supported main-glyph cores when detection rectangles include annotations.
        A missing/open corridor is inconclusive, never positive ownership proof.
        Only the automatic merge guard supplies already-included intervening mains.
        Exclusion is scoped to the evidenced nonadjacent horizontal-row case;
        vertical corridors and other callers retain the original veto. Components
        are excluded only from the decision, never self.ink or the original image.
        """
        a = first_glyphs.bounds if first_glyphs and first_glyphs.bounds else first.bbox
        b = second_glyphs.bounds if second_glyphs and second_glyphs.bounds else second.bbox
        dx = abs((a[0]+a[2])-(b[0]+b[2]))
        dy = abs((a[1]+a[3])-(b[1]+b[3]))
        separated_columns = max(a[0],b[0]) >= min(a[2],b[2]) and max(a[1],b[1]) < min(a[3],b[3])
        if separated_columns or dx >= dy:
            left, right = (a,b) if a[0]+a[2] <= b[0]+b[2] else (b,a)
            bounds = (left[2], max(a[1],b[1]), right[0], min(a[3],b[3]))
            axis = "vertical"
            span = bounds[3]-bounds[1]
        else:
            top, bottom = (a,b) if a[1]+a[3] <= b[1]+b[3] else (b,a)
            bounds = (max(a[0],b[0]), top[3], min(a[2],b[2]), bottom[1])
            axis = "horizontal"
            span = bounds[2]-bounds[0]
        x1,y1,x2,y2 = (int(v) for v in bounds)
        evidence = {"corridor_bounds": (x1,y1,x2,y2), "boundary_axis": axis,
                    "method": "original_otsu_dark_stroke_v1", "blocked": False}
        if x2-x1 < 2 or y2-y1 < 2:
            evidence["reason"] = "no_resolvable_facing_corridor"
            return evidence
        sizes = [g.scale for g in (first_glyphs,second_glyphs) if g and g.scale > 0]
        scale = min(sizes) if sizes else min(min(a[2]-a[0],a[3]-a[1]), min(b[2]-b[0],b[3]-b[1]))
        threshold = max(8.,2*scale,.25*span)
        # Connected/white balloon interiors can be divided only near a tail or
        # connector beyond the glyph ends. Probe two glyph scales there, without
        # widening the inter-column corridor or changing any recognition crop.
        endpoint_probe = int(np.ceil(2*scale))
        if axis == "vertical":
            y1,y2 = max(0,y1-endpoint_probe),min(self.ink.shape[0],y2+endpoint_probe)
        else:
            x1,x2 = max(0,x1-endpoint_probe),min(self.ink.shape[1],x2+endpoint_probe)
        evidence.update(scan_bounds=(x1,y1,x2,y2), endpoint_probe_pixels=endpoint_probe)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(self.ink[y1:y2,x1:x2], connectivity=8)
        lengths,original_lengths,ignored = [],[],[]
        for i in range(1,count):
            length = int(stats[i,3] if axis == "vertical" else stats[i,2])
            if stats[i,4] < max(4,.5*length):
                continue
            original_lengths.append(length)
            proof = (self._included_row_glyph_component(labels,stats,i,(x1,y1,x2,y2),bounds,
                     included_members,nonmember_boxes) if included_members and axis == "horizontal" and length >= threshold else None)
            if proof is not None:
                ignored.append(proof)
            else:
                lengths.append(length)
        evidence.update(longest_dark_stroke=max(lengths,default=0), threshold=threshold)
        if ignored:
            evidence.update(original_longest_dark_stroke=max(original_lengths,default=0),
                            ignored_member_components=tuple(ignored))
        evidence["blocked"] = any(length >= threshold for length in lengths)
        evidence["reason"] = "original_boundary_stroke" if evidence["blocked"] else "no_long_boundary_stroke_observed"
        return evidence
