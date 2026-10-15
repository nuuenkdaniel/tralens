"""Known-member boundary attribution only; original pixels, no engines or models."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tests.local_evidence import local_evidence
from tralens.ocr.automatic_evidence import OriginalInkEvidence, GlyphEvidence
from tralens.ocr import automatic_units as a
from tralens.ocr.automatic_units import propose_automatic_units
from tralens.ocr.recognition_units import PageProvenance, RawDetection

ROOT = Path(__file__).resolve().parents[2]
PAGE = "images/ubunchu-ja-test/ubunchu01_04.png"
BASELINE = "test-results/translation-coverage-diagnosis-v1/baseline-preview.json"


def frozen():
    document = json.loads((ROOT / BASELINE).read_bytes())
    batch = document["ocr"]["batches"][0]
    page = PageProvenance(**batch["page"])
    raw = tuple(RawDetection(**{**row, "page": page}) for row in batch["detections"])
    with Image.open(ROOT / PAGE) as source:
        image = source.convert("RGB")
    boxes = {}
    for row in raw:
        poly = np.array(row.polygon, dtype=np.float64)
        boxes[row.source_id] = SimpleNamespace(box_id=row.source_id, poly=poly, text=row.text,
            bbox=(int(np.floor(poly[:,0].min())),int(np.floor(poly[:,1].min())),
                  int(np.ceil(poly[:,0].max())),int(np.ceil(poly[:,1].max()))))
    return image, raw, boxes


class FrozenMemberBoundaryTests(unittest.TestCase):
    @local_evidence(files=(PAGE, BASELINE))
    def test_frozen_pair_ignores_only_complete_member19_component(self):
        image, _, boxes = frozen()
        self.addCleanup(image.close)
        ink = OriginalInkEvidence(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        member = ink.glyphs(boxes[19])
        result = ink.boundary_between(boxes[14], boxes[21],first_glyphs=ink.glyphs(boxes[14]),
            second_glyphs=ink.glyphs(boxes[21]), included_members=((boxes[19],member),),
            nonmember_boxes=tuple(box for i,box in boxes.items() if i not in (14,19,21,22)))
        self.assertFalse(result["blocked"], {"result":result,"member_glyphs":asdict(member),"member_polygon":boxes[19].poly.tolist()})
        self.assertEqual(result["ignored_member_components"][0]["member_id"],19)
        self.assertEqual(result["ignored_member_components"][0]["component_area"],273)
        self.assertEqual(result["threshold"],36.5)
        self.assertEqual(result["original_longest_dark_stroke"],37)

    @local_evidence(files=(PAGE, BASELINE))
    def test_frozen_desktop_member_glyph_does_not_split_known_rows(self):
        image, raw, boxes = frozen()
        self.addCleanup(image.close)
        before = image.tobytes()
        ink = OriginalInkEvidence(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        boundary = ink.boundary_between(boxes[14], boxes[21],
            first_glyphs=ink.glyphs(boxes[14]), second_glyphs=ink.glyphs(boxes[21]))
        self.assertTrue(boundary["blocked"])
        self.assertEqual(boundary["longest_dark_stroke"], 37)
        self.assertEqual(boundary["threshold"], 36.5)
        batch = propose_automatic_units(raw, image, complementary_ownership=True, regroup_companions=True)
        groups = [p.main_ids for p in batch.proposals if {14,19,21,22}.intersection(p.main_ids)]
        self.assertEqual(groups, [(14,19,21,22)])
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(hashlib.sha256((ROOT / PAGE).read_bytes()).hexdigest(), raw[0].page.image_sha256)

    @local_evidence(files=(PAGE, BASELINE))
    def test_frozen_nonmember_sfx_and_overlapping_rows_remain_vetoed(self):
        image, _, boxes = frozen()
        self.addCleanup(image.close)
        ink = OriginalInkEvidence(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
        before = ink.ink.tobytes()
        result = ink.boundary_between(boxes[14],boxes[21],first_glyphs=ink.glyphs(boxes[14]),second_glyphs=ink.glyphs(boxes[21]),
            included_members=tuple((boxes[i],ink.glyphs(boxes[i])) for i in (24,30)), nonmember_boxes=(boxes[19],))
        self.assertTrue(result["blocked"])
        self.assertNotIn("ignored_member_components",result)
        overlap = ink.boundary_between(boxes[13],boxes[18],first_glyphs=ink.glyphs(boxes[13]),second_glyphs=ink.glyphs(boxes[18]),
            included_members=((boxes[20],ink.glyphs(boxes[20])),))
        self.assertEqual(overlap["reason"],"no_resolvable_facing_corridor")
        self.assertNotIn("ignored_member_components",overlap)
        self.assertEqual(ink.ink.tobytes(),before)

    @local_evidence(files=(PAGE, BASELINE))
    def test_merge_attempt_context_cannot_leak_through_pair_cache(self):
        image,raw,_ = frozen()
        self.addCleanup(image.close)
        original = a.group_manga_boxes
        observed = []
        def check(boxes, **kwargs):
            by_id = {b.box_id:b for b in boxes}
            guard = kwargs["merge_guard"]
            observed.append(guard([by_id[14],by_id[19]],[by_id[21],by_id[22]]))
            observed.append(guard([by_id[14]],[by_id[21]]))
            return original(boxes,**kwargs)
        with patch.object(a,"group_manga_boxes",side_effect=check):
            propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
        self.assertEqual(observed,[True,False])


def rectangle(identifier, bounds):
    x,y,r,s = bounds
    return SimpleNamespace(box_id=identifier,bbox=bounds,poly=np.array([(x,y),(r,y),(r,s),(x,s)],dtype=np.float64))


class IsolatedComponentTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((140,220,3),255,np.uint8)
        self.first = rectangle("top",(35,15,180,40))
        self.second = rectangle("bottom",(35,100,180,125))
        self.member = rectangle("member",(35,45,180,90))
        self.top = GlyphEvidence(3,100,12.,"horizontal",(40,20,175,35))
        self.bottom = GlyphEvidence(3,100,12.,"horizontal",(40,105,175,120))
        self.core = GlyphEvidence(3,300,12.,"horizontal",(60,47,160,86))

    def glyph(self, offset=0):
        # A complex, isolated K-shaped component; no font/runtime/model needed.
        for left,right in (((75,52),(75,80)),((115,52),(75,66)),((75,66),(115,80))):
            cv2.line(self.image,(left[0]+offset,left[1]),(right[0]+offset,right[1]),(0,0,0),2)

    def boundary(self, *, included=True, core=None, nonmembers=()):
        ink = OriginalInkEvidence(self.image)
        original = ink.ink.tobytes()
        result = ink.boundary_between(self.first,self.second,first_glyphs=self.top,second_glyphs=self.bottom,
            included_members=((self.member,core or self.core),) if included else (), nonmember_boxes=nonmembers)
        self.assertEqual(ink.ink.tobytes(),original)
        return result

    def test_isolated_intervening_glyph_can_only_be_excluded_with_member_proof(self):
        self.glyph()
        self.assertTrue(self.boundary(included=False)["blocked"])
        result = self.boundary()
        self.assertFalse(result["blocked"])
        self.assertEqual(result["ignored_member_components"][0]["member_id"],"member")
        self.assertEqual(result["threshold"],33.75)

    def test_straight_divider_inside_supported_core_is_not_ignored(self):
        cv2.line(self.image,(70,65),(125,65),(0,0,0),2)
        result = self.boundary()
        self.assertTrue(result["blocked"])
        self.assertNotIn("ignored_member_components",result)

    def test_closed_outline_inside_core_is_not_ignored(self):
        cv2.rectangle(self.image,(70,51),(125,81),(0,0,0),2)
        self.assertTrue(self.boundary()["blocked"])

    def test_divider_between_boxes_remains_even_when_member_glyph_is_ignored(self):
        self.glyph()
        cv2.line(self.image,(50,96),(170,96),(0,0,0),2)
        result = self.boundary()
        self.assertTrue(result["blocked"])
        self.assertTrue(result.get("ignored_member_components"))
        self.assertGreaterEqual(result["longest_dark_stroke"],result["threshold"])

    def test_component_crossing_core_or_scan_boundary_is_never_ignored(self):
        self.glyph()
        crossing = GlyphEvidence(3,300,12.,"horizontal",(90,47,160,86))
        self.assertTrue(self.boundary(core=crossing)["blocked"])
        self.image.fill(255)
        self.glyph(offset=90)
        self.member = rectangle("member",(140,45,219,90))
        outside_scan = GlyphEvidence(3,300,12.,"horizontal",(145,47,218,86))
        self.assertTrue(self.boundary(core=outside_scan)["blocked"])

    def test_actual_nonmember_ink_overlap_vetoes_exclusion(self):
        self.glyph()
        for name in ("SFX24","SFX30"):
            result = self.boundary(nonmembers=(rectangle(name,(70,49,125,84)),))
            self.assertTrue(result["blocked"])
            self.assertNotIn("ignored_member_components",result)

    def test_unsupported_or_nonintervening_member_cannot_supply_exclusion(self):
        self.glyph()
        for core in (GlyphEvidence(1,300,12.,"horizontal",self.core.bounds),
                     GlyphEvidence(3,300,12.,"unknown",self.core.bounds),
                     GlyphEvidence(3,300,12.,"horizontal",(60,25,160,115))):
            self.assertTrue(self.boundary(core=core)["blocked"])

    def test_vertical_corridor_is_outside_this_narrow_row_repair(self):
        self.glyph()
        image = self.image.transpose(1,0,2).copy()
        ink = OriginalInkEvidence(image)
        first,second = rectangle("left",(15,35,40,180)),rectangle("right",(100,35,125,180))
        member = rectangle("member",(45,35,90,180))
        left = GlyphEvidence(3,100,12.,"vertical",(20,40,35,175))
        right = GlyphEvidence(3,100,12.,"vertical",(105,40,120,175))
        core = GlyphEvidence(3,300,12.,"vertical",(47,60,86,160))
        original = ink.boundary_between(first,second,first_glyphs=left,second_glyphs=right)
        scoped = ink.boundary_between(first,second,first_glyphs=left,second_glyphs=right,included_members=((member,core),))
        self.assertTrue(original["blocked"])
        self.assertEqual(scoped,original)
