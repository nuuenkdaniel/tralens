"""Production-mask shape regressions; no models, temporary files or review claims.

Shape acceptance is deliberately not labelled as correct balloon ownership. An
enclosed artwork pocket and a dialogue balloon can have the same raster shape.
"""
import json
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from tralens.ocr.automatic_evidence import OriginalInkEvidence
from tralens.ocr.automatic_units import propose_automatic_units
from tralens.ocr.automatic_recognition import prepare_automatic_crop
from tralens.ocr.grouping import analyze_manga_regions
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tools.diagnostics import load_unit_page
from tests.local_evidence import local_evidence


def rectangle(identifier, bounds):
    x1, y1, x2, y2 = bounds
    return SimpleNamespace(box_id=identifier, bbox=bounds, text="字", confidence=.99,
        poly=np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], dtype=np.float64))


def production_components(image, boxes):
    """Independent reproduction, including truncation and inclusive polygon masks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_mask = np.zeros_like(gray)
    for box in boxes:
        cv2.fillPoly(text_mask, [box.poly.astype(np.int32)], 255)
    free = cv2.bitwise_not(cv2.bitwise_and(ink, cv2.bitwise_not(text_mask)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    rows = []
    height, width = gray.shape
    for box in boxes:
        x1,y1,x2,y2 = box.bbox
        mask = np.zeros((y2-y1+1,x2-x1+1), dtype=np.uint8)
        cv2.fillPoly(mask, [box.poly.astype(np.int32)-np.array([x1,y1])], 255)
        votes = np.bincount(labels[y1:y2+1,x1:x2+1][mask != 0], minlength=count)
        label = int(np.argmax(votes[1:])+1) if count > 1 else 0
        x,y,w,h,area = (int(v) for v in stats[label])
        overlap = float(votes[label]/np.count_nonzero(mask))
        box_area = (x2-x1)*(y2-y1)
        checks = (overlap >= .9, x > 0 and y > 0 and x+w < width and y+h < height,
                  area >= box_area, area <= 30*box_area, area/(w*h) >= .55)
        rows.append(dict(label=label, area=area, bbox=(x,y,w,h), fill=area/(w*h),
                         overlap=overlap, checks=checks))
    return labels, rows


class ComponentShapeTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((240,300,3),255,dtype=np.uint8)
        self.boxes = [rectangle("left",(110,90,126,150)),
                      rectangle("right",(140,90,156,150))]

    def check_policy(self, expected):
        before = self.image.tobytes()
        _, rows = production_components(self.image,self.boxes)
        actual,_ = analyze_manga_regions(self.boxes,self.image)
        self.assertEqual(set(actual),{i for i,row in enumerate(rows) if all(row["checks"])})
        self.assertEqual(set(actual),set(expected),rows)
        self.assertEqual(self.image.tobytes(),before)
        return rows

    def test_compact_ellipse_rectangle_and_rounded_balloon(self):
        for shape in ("ellipse","rectangle","rounded"):
            with self.subTest(shape=shape):
                self.image.fill(255)
                if shape == "ellipse":
                    cv2.ellipse(self.image,(135,120),(75,95),0,0,360,(0,0,0),3)
                elif shape == "rectangle":
                    cv2.rectangle(self.image,(60,25),(210,215),(0,0,0),3)
                else:
                    cv2.rectangle(self.image,(80,25),(190,215),(0,0,0),-1)
                    cv2.rectangle(self.image,(60,45),(210,195),(0,0,0),-1)
                    for x in (80,190):
                        for y in (45,195):
                            cv2.circle(self.image,(x,y),20,(0,0,0),-1)
                    cv2.rectangle(self.image,(80,29),(190,211),(255,255,255),-1)
                    cv2.rectangle(self.image,(64,45),(206,195),(255,255,255),-1)
                    for x in (80,190):
                        for y in (45,195):
                            cv2.circle(self.image,(x,y),16,(255,255,255),-1)
                for box in self.boxes:
                    cv2.fillPoly(self.image,[box.poly.astype(np.int32)],(0,0,0))
                rows = self.check_policy({0,1})
                self.assertEqual(rows[0]["label"],rows[1]["label"])
                self.assertEqual(rows[0]["overlap"],1.)

    def test_slanted_diamond_fails_fill_despite_enclosure_and_full_overlap(self):
        cv2.polylines(self.image,[np.array([(135,20),(240,120),(135,220),(30,120)])],
                      True,(0,0,0),3)
        rows = self.check_policy(set())
        for row in rows:
            self.assertTrue(all(row["checks"][:4]))
            self.assertLess(row["fill"],.55)

    def test_long_thin_tail_can_reject_otherwise_compact_dialogue(self):
        outline = np.array([(65,65),(185,65),(185,165),(180,165),
                            (280,230),(130,165),(65,165)])
        cv2.polylines(self.image,[outline],True,(0,0,0),2)
        rows = self.check_policy(set())
        self.assertTrue(all(all(row["checks"][:4]) for row in rows))

    def test_slanted_artwork_pocket_has_high_hull_fill_but_low_rectangle_fill(self):
        cv2.ellipse(self.image,(135,120),(100,35),45,0,360,(0,0,0),2)
        self.boxes = [rectangle("label",(120,105,140,135))]
        labels,rows = production_components(self.image,self.boxes)
        self.check_policy(set())
        component = np.uint8(labels == rows[0]["label"])
        contours,_ = cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(max(contours,key=cv2.contourArea)))
        self.assertGreater(rows[0]["area"]/hull_area,.95)
        self.assertLess(rows[0]["fill"],.55)
        self.assertTrue(all(rows[0]["checks"][:4]))

    def test_open_and_page_edge_interiors_never_gain_region_support(self):
        for edge in (False,True):
            with self.subTest(edge=edge):
                self.image.fill(255)
                if edge:
                    cv2.rectangle(self.image,(0,25),(210,215),(0,0,0),3)
                    self.image[100:110,:4] = 255
                else:
                    cv2.ellipse(self.image,(135,120),(75,95),0,15,345,(0,0,0),3)
                rows = self.check_policy(set())
                self.assertTrue(all(not row["checks"][1] for row in rows))

    def test_large_enclosed_panel_fails_area_for_small_detection(self):
        cv2.rectangle(self.image,(5,5),(295,235),(0,0,0),2)
        self.boxes = [rectangle("small",(110,90,120,110))]
        rows = self.check_policy(set())
        self.assertTrue(rows[0]["checks"][1])
        self.assertFalse(rows[0]["checks"][3])

    def test_enclosed_artwork_can_have_identical_accepted_shape(self):
        # A white clothing/paper pocket has no shape-only semantic distinction.
        cv2.rectangle(self.image,(60,25),(210,215),(0,0,0),3)
        rows = self.check_policy({0,1})
        self.assertGreater(rows[0]["fill"],.9)
        self.assertEqual(rows[0]["overlap"],1.)

    def test_connected_balloon_negative_retains_original_divider_veto(self):
        cv2.rectangle(self.image,(60,25),(210,215),(0,0,0),3)
        cv2.line(self.image,(133,27),(133,180),(0,0,0),2)
        rows = self.check_policy({0,1})
        self.assertEqual(rows[0]["label"],rows[1]["label"])
        evidence = OriginalInkEvidence(self.image).boundary_between(*self.boxes)
        self.assertTrue(evidence["blocked"])

    def test_ocr_polygon_removal_can_erase_a_real_divider_but_original_veto_survives(self):
        cv2.rectangle(self.image,(60,25),(210,215),(0,0,0),3)
        cv2.line(self.image,(133,27),(133,212),(0,0,0),2)
        # An oversized third detection erases part of the dark dividing outline.
        self.boxes.append(rectangle("eraser",(129,100,137,130)))
        rows = self.check_policy({0,1})
        self.assertEqual(rows[0]["label"],rows[1]["label"])
        evidence = OriginalInkEvidence(self.image).boundary_between(*self.boxes[:2])
        self.assertTrue(evidence["blocked"])

    def test_connected_balloon_annotation_candidate_cannot_cross_original_outline(self):
        cv2.rectangle(self.image,(90,60),(165,165),(0,0,0),3)
        cv2.line(self.image,(137,62),(137,155),(0,0,0),2)
        self.boxes = [rectangle("main",(110,70,130,150)),rectangle("small_dialogue",(140,80,148,115))]
        self.boxes[1].text = "かな"
        regions,_ = analyze_manga_regions(self.boxes,self.image)
        self.assertEqual(len({r[0] for r in regions.values()}),1)
        self.assertEqual(len(regions),2)
        image = Image.fromarray(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB))
        page = PageProvenance("fixture","/original.png","1"*64,"2"*64,image_fingerprint(image))
        batch = propose_automatic_units(snapshot_detections(self.boxes,page),image)
        self.assertEqual(batch.coverage["tentative_annotations"],0)
        self.assertEqual(batch.coverage["main_detections"],2)
        rejected = [r for r in batch.relationships if r.source_id == "small_dialogue"]
        self.assertTrue(any("annotation_crosses_original_boundary" in r.reasons for r in rejected))


class SavedComponentShapeTests(unittest.TestCase):
    @local_evidence(comparisons=("test-results/crop-isolation-v2/ubunchu02_04-comparison.json",),
                    files=("test-results/ownership-pixel-investigation-parent-v1/rejected-region-statistics.json",))
    def test_saved_rejections_reproduce_exact_production_mask_not_hull(self):
        root = Path(__file__).resolve().parents[2]
        _,image,_,raw = load_unit_page(root/"test-results/crop-isolation-v2/ubunchu02_04-comparison.json",root)
        boxes = [SimpleNamespace(box_id=d.source_id,text=d.text,confidence=d.confidence,
            poly=np.array(d.polygon),bbox=(int(min(x for x,y in d.polygon)),int(min(y for x,y in d.polygon)),
                int(max(x for x,y in d.polygon)),int(max(y for x,y in d.polygon)))) for d in raw]
        bgr = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        # All detections on this page have valid in-image integer bounds.
        labels,rows = production_components(bgr,boxes)
        actual,_ = analyze_manga_regions(boxes,bgr)
        saved = json.loads((root/"test-results/ownership-pixel-investigation-parent-v1/rejected-region-statistics.json").read_text())
        for expected in saved:
            index = next(i for i,b in enumerate(boxes) if b.box_id == expected["source_id"])
            row = rows[index]
            with self.subTest(source_id=expected["source_id"]):
                self.assertEqual(row["bbox"],tuple(expected["component_bbox"]))
                self.assertEqual(row["area"],expected["component_area"])
                self.assertAlmostEqual(row["fill"],expected["fill"],places=12)
                self.assertEqual(row["overlap"],1.)
                self.assertEqual(row["checks"],(True,True,True,True,False))
                self.assertNotIn(index,actual)
                # Both saved white components extend outside the dialogue crop:
                # original overlays show those extensions enter surrounding art.
                members = [b for i,b in enumerate(boxes) if rows[i]["label"] == row["label"]]
                x1,y1 = min(b.bbox[0] for b in members),min(b.bbox[1] for b in members)
                x2,y2 = max(b.bbox[2] for b in members),max(b.bbox[3] for b in members)
                union = np.zeros(labels.shape,dtype=bool)
                union[y1:y2+1,x1:x2+1] = True
                self.assertGreater(np.count_nonzero((labels == row["label"]) & ~union),0)

    @local_evidence(manifests=("test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json",))
    def test_seven_saved_pages_retain_exact_partition_order_regions_and_conflicts(self):
        root = Path(__file__).resolve().parents[2]
        saved = json.loads((root/"test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json").read_text())
        totals = Counter()
        for page in saved["pages"]:
            with self.subTest(page=page["page"]["page_id"]):
                _,image,provenance,raw = load_unit_page(page["comparison_path"],root)
                batch = propose_automatic_units(raw,image)
                # Includes immutable raw evidence, relationship parents/rejections,
                # ordered membership, regions, crop geometry and exact conflicts.
                expected = dict(page["batch"],ownership_policy="production8",ownership_repairs=[],
                                companion_policy="disabled",companion_regroupings=[],companion_rejections=[])
                self.assertEqual(json.loads(json.dumps(asdict(batch))),expected)
                self.assertEqual(batch.coverage,page["coverage"])
                self.assertEqual(batch.page,provenance)
                totals.update(batch.coverage)
        self.assertEqual(len(saved["pages"]),7)
        self.assertEqual(totals["raw_detections"],268)
        self.assertEqual(totals["proposed_units"],136)
        self.assertEqual(totals["unsafe_crops"],24)
        self.assertEqual(totals["tentative_annotations"],14)

    @local_evidence(manifests=("test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json",))
    def test_seven_saved_pages_retain_exact_model_free_eligibility_and_routing(self):
        root = Path(__file__).resolve().parents[2]
        saved = json.loads((root/"test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json").read_text())
        blockers = Counter()
        eligible = 0
        for page in saved["pages"]:
            _,image,provenance,raw = load_unit_page(page["comparison_path"],root)
            original = Path(provenance.image_path).read_bytes()
            batch = propose_automatic_units(raw,image)
            self.assertEqual(len(batch.proposals),len(page["units"]))
            for proposal,record in zip(batch.proposals,page["units"]):
                with self.subTest(page=provenance.page_id,members=proposal.member_ids):
                    crop,evidence,error = prepare_automatic_crop(proposal,batch,image,original_image_bytes=original)
                    expected = record["eligibility_preflight"]
                    self.assertEqual(error,expected["reason"])
                    self.assertEqual(error is None,expected["eligible"])
                    self.assertEqual(json.loads(json.dumps(asdict(evidence))) if evidence else None,expected["crop"])
                    if crop is not None:
                        self.assertEqual(crop.tobytes(),image.crop(proposal.crop_bounds).tobytes())
                    if error:
                        blockers[error] += 1
                    else:
                        eligible += 1
        self.assertEqual(eligible,52)
        self.assertEqual(blockers,{
            "unknown_automated_region_ownership":31,"unsafe_automatic_crop":24,
            "unknown_automatic_reading_orientation":9,"unsupported_script_for_manga_ocr":9,
            "mixed_script_not_routed_to_manga_ocr":7,"script_inconclusive_no_lexical_evidence":4})


if __name__ == "__main__":
    unittest.main()
