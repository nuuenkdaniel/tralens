"""Opt-in original-enclosure space hypotheses; never cleanup or semantic truth.

Only a previously validated flat native erase mask can supply missing lettering
space. Neither a white component nor its filled contour authorizes erasure.
"""
import cv2
import numpy as np

from . import natural as n, preparation as p


POLICY = 'closed_original_flat_space_retry_v1'
MAX_ENCLOSURE_PIXELS = 262144


class Enclosures:
    def __init__(self, rgb, detections, geometry):
        n.budget_check(rgb.mode == 'RGB' and rgb.width*rgb.height <= p.MAX_PIXELS
                       and max(rgb.size) <= p.MAX_SIDE, 'enclosure page budget')
        self.pixels = np.asarray(rgb).copy()
        n.require(np.array_equal(self.pixels, geometry.pixels), 'enclosure original pixels changed')
        geometry.bind_geometry(detections)
        self.geometry = geometry
        self.source_sha256 = n.digest(self.pixels.tobytes())
        light = (self.pixels.min(axis=2) >= 180).astype(np.uint8)
        self.white = []
        for connectivity in (4, 8):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=connectivity)
            n.budget_check(count <= 8192, 'enclosure original component budget')
            self.white.append((labels, stats))

    def expand(self, rgb, detections, member_ids, plan, reserved):
        from .automatic import AutomaticPlan, check
        n.require(np.array_equal(self.pixels, np.asarray(rgb)), 'enclosure original pixels changed')
        self.geometry.bind_geometry(detections)
        n.require(type(member_ids) is list and 1 <= len(member_ids) <= 32
            and all(type(i) in (str, int) and i != '' for i in member_ids)
            and len(set(member_ids)) == len(member_ids), 'typed enclosure members required')
        n.budget_check(len(member_ids)*rgb.width*rgb.height <= n.MAX_UNIT_PIXELS, 'enclosure member work budget')
        keys = {(type(i).__name__, i) for i in member_ids}
        n.require(keys <= self.geometry.source_polygons.keys(), 'foreign enclosure member')
        shape = self.pixels.shape[:2]
        n.require(reserved.shape == shape and reserved.dtype == bool
            and all(getattr(plan, name).shape == shape and getattr(plan, name).dtype == bool
                    for name in ('foreground', 'erase', 'layout', 'obstacles')), 'enclosure mask shape/type')
        check(plan.restored_pixels is None and plan.background is not None, 'enclosure_flat_only',
              'enclosure retry requires an existing native flat cleanup, not reconstruction')
        color = np.asarray(plan.background)
        n.require(color.shape == (3,) and np.isfinite(color).all()
            and (color >= 180).all() and (color <= 255).all() and np.ptp(color) <= 16,
            'enclosure needs a measured light neutral flat color')
        check(plan.foreground.any() and not (plan.foreground & ~plan.erase).any(),
              'enclosure_native_support', 'complete native foreground/erase required')
        masks = [np.asarray(n.polygon_mask(self.geometry.source_polygons[key], rgb.size, evidence=True)) != 0
                 for key in sorted(keys)]
        own_counts = np.zeros(shape, np.uint16)
        regions = []
        votes_evidence = []
        for labels, stats in self.white:
            choices = []
            for mask in masks:
                counts = np.bincount(labels[mask], minlength=len(stats))
                positive = int(counts[1:].sum())
                check(positive >= .5*int(mask.sum()), 'enclosure_light_support', 'source lacks original light support')
                label = int(np.argmax(counts[1:])+1)
                check(counts[label] >= .9*positive, 'enclosure_ambiguous', 'source spans ambiguous original regions')
                x, y, w, h, area = map(int, stats[label])
                check(x > 0 and y > 0 and x+w < rgb.width and y+h < rgb.height,
                      'enclosure_open', 'original region is open or page-connected')
                choices.append(label)
                votes_evidence.append({'label': label, 'light_fraction': positive/int(mask.sum()),
                                       'dominant_fraction': int(counts[label])/positive})
            check(len(set(choices)) == 1, 'enclosure_distinct', 'members belong to different original enclosures')
            regions.append(labels == choices[0])
        envelopes = [p.filled_outer(region) for region in regions]
        interiors = [p.erode(envelope, 3) for envelope in envelopes]
        # Diagonal antialias fringe/counters may differ at the original outline.
        # The complete usable cores must agree exactly, before any ink is erased.
        check(np.array_equal(*interiors), 'enclosure_connectivity', '4/8-connected usable original cores disagree')
        envelope = envelopes[0]
        area = int(envelope.sum())
        for mask in masks:
            own_counts += mask
        check(64 <= area <= min(MAX_ENCLOSURE_PIXELS, rgb.width*rgb.height//3, int((own_counts != 0).sum())*30),
              'enclosure_extent', 'original enclosure area is unsupported')
        nonmembers = self.geometry.source_counts > own_counts
        check(not (envelope & nonmembers).any(), 'enclosure_neighbor',
              'another source/annotation intersects the enclosure; do not infer shared ownership')
        outer_ring = p.dilate(envelope, 1) & ~envelope
        outline_support = float((self.pixels[outer_ring].min(axis=1) < 180).mean()) if outer_ring.any() else 0.
        check(outline_support >= .9, 'enclosure_outline', 'closed region lacks independently visible dark outline support')
        interior = interiors[0]
        check(not (plan.erase & ~interior).any(), 'enclosure_clearance', 'native cleanup lacks full boundary clearance')
        residual = np.abs(self.pixels.astype(np.int16)-color.astype(np.int16)).max(axis=2)
        unexplained = interior & (residual > 2) & ~plan.erase
        check(not unexplained.any(), 'enclosure_unexplained_ink',
              'unexplained original ink/art/texture inside enclosure', n.offending_pixels(unexplained))
        check(not (plan.erase & reserved).any(), 'enclosure_reserved', 'cleanup touches prior committed output')
        obstacles = ~interior | nonmembers | reserved
        candidate = p.erode(interior, 2) & ~p.dilate(nonmembers | reserved, 2)
        check(not (plan.foreground & ~candidate).any(), 'enclosure_seed', 'complete source ink loses supported lettering space')
        count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
        n.budget_check(count <= 8192, 'enclosure layout component budget')
        chosen = np.unique(labels[plan.foreground])
        check(len(chosen) == 1 and chosen[0] != 0, 'enclosure_disconnected', 'source ink spans disconnected free pockets')
        layout = labels == chosen[0]
        check(int(layout.sum()) > int(plan.layout.sum()), 'enclosure_no_gain', 'no additional supported lettering space')
        n.require(not (layout & obstacles).any() and not (plan.erase & obstacles).any(), 'enclosure containment failure')
        x, y, w, h = cv2.boundingRect(envelope.astype(np.uint8))
        evidence = {**plan.evidence, 'enclosure_layout': {
            'policy': POLICY, 'source_pixels_sha256': self.source_sha256,
            'geometry_revision': self.geometry.geometry_revision, 'member_ids': member_ids,
            'original_bounds': [x, y, x+w, y+h], 'enclosure_pixels': area,
            'connectivities': [4, 8], 'member_votes': votes_evidence,
            'boundary_fringe_disagreement_pixels': int(np.count_nonzero(envelopes[0] != envelopes[1])),
            'usable_core_disagreement_pixels': 0,
            'outline_support_fraction': outline_support, 'boundary_inset': 3, 'layout_inset': 2,
            'background_rgb': plan.background, 'unexplained_pixels': 0,
            'old_layout_pixels': int(plan.layout.sum()), 'new_layout_pixels': int(layout.sum()),
            'foreground_changed_pixels': 0, 'erase_changed_pixels': 0,
            'semantic_ownership': 'unverified; bounded original flat-space hypothesis only'}}
        return AutomaticPlan(plan.foreground.copy(), plan.erase.copy(), layout, obstacles,
                             plan.background, evidence)
