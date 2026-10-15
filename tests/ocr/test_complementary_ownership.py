"""Complementary topology fixtures, original glyphs and offline saved evidence."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tralens.ocr.automatic_evidence import OriginalInkEvidence
from tralens.ocr.grouping import analyze_manga_regions
from tralens.ocr.automatic_units import propose_automatic_units
from tralens.ocr.automatic_recognition import prepare_automatic_crop
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tools.diagnostics import load_unit_page
from tests.local_evidence import local_evidence
from tools import diagnostics


def box(source_id, bounds):
    x1,y1,x2,y2 = bounds
    return SimpleNamespace(box_id=source_id,bbox=bounds,text="字",confidence=.99,
        poly=np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],dtype=np.float64))


class ComplementaryOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((240,260,3),dtype=np.uint8)
        cv2.rectangle(self.image,(30,40),(100,140),(255,255,255),-1)
        cv2.rectangle(self.image,(150,160),(210,200),(255,255,255),-1)
        cv2.line(self.image,(100,140),(150,190),(255,255,255),1)
        self.boxes = [box("left",(48,55,60,125)),box("right",(75,55,87,125))]
        for x in (50,77):
            for y in (65,85,105):
                cv2.rectangle(self.image,(x,y),(x+6,y+6),(0,0,0),-1)

    def repair(self, groups=(((0,1),()),)):
        before = self.image.tobytes()
        regions,_ = analyze_manga_regions(self.boxes,self.image)
        result = OriginalInkEvidence(self.image).complementary_ownership(self.boxes,groups,regions)
        self.assertEqual(self.image.tobytes(),before)
        return regions,result

    def test_diagonal_protrusion_is_separated_without_changing_any_pixels(self):
        old,(regions,repairs) = self.repair()
        self.assertEqual(old,{})
        self.assertEqual(set(regions),{0,1})
        self.assertEqual(regions[0][0],regions[1][0])
        self.assertLess(regions[0][0],0)
        self.assertEqual(len(repairs),1)
        for row in repairs[0]["source_evidence"]:
            self.assertLess(row["production8"]["fill"],.55)
            self.assertGreaterEqual(row["complementary4"]["fill"],.55)
            self.assertEqual(row["complementary4"]["overlap"],1.)

    def test_resolvable_white_neck_is_not_pruned_or_closed(self):
        cv2.line(self.image,(100,140),(150,190),(255,255,255),3)
        old,(regions,repairs) = self.repair()
        self.assertEqual(old,{})
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())

    def test_horizontal_dialogue_uses_the_same_topology_and_original_glyph_checks(self):
        height = self.image.shape[0]
        self.image = cv2.rotate(self.image,cv2.ROTATE_90_CLOCKWISE)
        self.boxes = [box(b.box_id,(height-1-b.bbox[3],b.bbox[0],
                                  height-1-b.bbox[1],b.bbox[2])) for b in self.boxes]
        old,(regions,repairs) = self.repair()
        self.assertEqual(old,{})
        self.assertEqual(set(regions),{0,1})
        self.assertEqual(len(repairs),1)

    def test_artwork_ink_without_dialogue_glyph_flow_does_not_supply_ownership(self):
        self.image[40:141,30:101] = 255
        for x in (49,53,57,76,80,84):
            cv2.rectangle(self.image,(x,85),(x+1,91),(0,0,0),-1)
        _,(regions,repairs) = self.repair()
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())

    def test_shared_white_component_does_not_override_original_balloon_divider(self):
        cv2.line(self.image,(68,40),(68,125),(0,0,0),2)
        _,(regions,repairs) = self.repair()
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())

    def test_single_label_or_unsupported_artwork_is_not_dialogue_support(self):
        _,(regions,repairs) = self.repair((((0,),()),))
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())
        self.image[40:141,30:101] = 255
        _,(regions,repairs) = self.repair()
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())

    def test_partial_detection_cannot_be_trimmed_to_fit_a_core(self):
        self.boxes[1] = box("right",(75,55,110,125))
        _,(regions,repairs) = self.repair()
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())

    def test_duplicate_footprints_do_not_manufacture_multi_main_support(self):
        self.boxes[1] = box("duplicate",self.boxes[0].bbox)
        _,(regions,repairs) = self.repair()
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())

    def test_original_open_background_cannot_gain_enclosure_from_polygon_removal(self):
        cv2.line(self.image,(30,80),(0,80),(255,255,255),1)
        _,(regions,repairs) = self.repair()
        self.assertEqual(regions,{})
        self.assertEqual(repairs,())

    @local_evidence(comparisons=("test-results/crop-isolation-v2/ubunchu02_04-comparison.json",))
    def test_saved_page_has_actual_supported_complementary_components(self):
        root = Path(__file__).resolve().parents[2]
        _,image,_,raw = load_unit_page(root/"test-results/crop-isolation-v2/ubunchu02_04-comparison.json",root)
        batch = propose_automatic_units(raw,image)
        boxes = [box(d.source_id,(int(min(x for x,y in d.polygon)),int(min(y for x,y in d.polygon)),
            int(max(x for x,y in d.polygon)),int(max(y for x,y in d.polygon)))) for d in raw]
        for b,d in zip(boxes,raw):
            b.poly = np.array(d.polygon)
        indices = {b.box_id:i for i,b in enumerate(boxes)}
        groups = [(tuple(indices[i] for i in p.main_ids),tuple(indices[i] for i in p.annotation_ids))
                  for p in batch.proposals]
        known = {indices[i]:(region,confidence) for i,region,confidence in batch.regions}
        bgr = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        regions,repairs = OriginalInkEvidence(bgr).complementary_ownership(boxes,groups,known)
        self.assertTrue(repairs,"No saved component satisfies the conservative complementary fallback")
        self.assertEqual([r["main_ids"] for r in repairs],[(14,13),(12,11,10)])
        self.assertTrue(set(known) <= set(regions))
        # The other reported block is NOT forced through: its entire
        # four-connected component still fails the unchanged fill requirement.
        ink = OriginalInkEvidence(bgr).ink
        text_mask = np.zeros_like(ink)
        for b in boxes:
            cv2.fillPoly(text_mask,[b.poly.astype(np.int32)],255)
        free = cv2.bitwise_not(cv2.bitwise_and(ink,cv2.bitwise_not(text_mask)))
        count,labels,stats,_ = cv2.connectedComponentsWithStats(free,connectivity=4)
        for source_id in (23,22,21):
            polygon_mask = np.zeros_like(ink)
            cv2.fillPoly(polygon_mask,[boxes[indices[source_id]].poly.astype(np.int32)],255)
            votes = np.bincount(labels[polygon_mask != 0],minlength=count)
            label = int(np.argmax(votes[1:])+1)
            x,y,w,h,area = stats[label]
            self.assertLess(area/(w*h),.55,(source_id,tuple(stats[label])))

    def batches(self):
        image = Image.fromarray(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB))
        stream = BytesIO()
        image.save(stream,format="PNG")
        content = stream.getvalue()
        page = PageProvenance("fixture","/original.png",hashlib.sha256(content).hexdigest(),
                              "2"*64,image_fingerprint(image))
        raw = snapshot_detections(self.boxes,page)
        return image,content,propose_automatic_units(raw,image),propose_automatic_units(
            raw,image,complementary_ownership=True)

    def test_explicit_opt_in_recovers_eligibility_with_exact_original_crop_and_raw_evidence(self):
        image,content,baseline,repaired = self.batches()
        self.assertEqual(baseline.ownership_repairs,())
        self.assertEqual(repaired.detections,baseline.detections)
        self.assertEqual(repaired.evidence_sha256,baseline.evidence_sha256)
        self.assertEqual(repaired.relationships,baseline.relationships)
        self.assertEqual(repaired.proposals[0].member_ids,baseline.proposals[0].member_ids)
        self.assertEqual(repaired.proposals[0].crop_bounds,baseline.proposals[0].crop_bounds)
        self.assertEqual(prepare_automatic_crop(baseline.proposals[0],baseline,image,
            original_image_bytes=content)[2],"unknown_automated_region_ownership")
        crop,evidence,error = prepare_automatic_crop(repaired.proposals[0],repaired,image,
            original_image_bytes=content)
        self.assertIsNone(error)
        self.assertEqual(crop.tobytes(),image.crop(baseline.proposals[0].crop_bounds).tobytes())
        self.assertEqual(evidence.completeness,"unverified_full_utterance")
        self.assertFalse(repaired.reviewed)
        self.assertFalse(repaired.proposals[0].reviewed)

    def test_repaired_ownership_does_not_override_crop_conflict(self):
        intruder = box("intruder",(65,80,70,85))
        intruder.text = "!"
        self.boxes.append(intruder)
        image,content,baseline,repaired = self.batches()
        proposal = next(p for p in repaired.proposals if len(p.main_ids) == 2)
        self.assertTrue(repaired.ownership_repairs)
        self.assertFalse(proposal.crop_safe)
        self.assertEqual([c.source_id for c in proposal.crop_conflicts],["intruder"])
        self.assertEqual(prepare_automatic_crop(proposal,repaired,image,
            original_image_bytes=content)[2],"unsafe_automatic_crop")

    def test_latin_mixed_and_sfx_routing_remain_separate_from_shape_support(self):
        for text,reason in (("Latin","unsupported_script_for_manga_ocr"),
                            ("字Latin","mixed_script_not_routed_to_manga_ocr"),
                            ("!?","script_inconclusive_no_lexical_evidence")):
            with self.subTest(text=text):
                for b in self.boxes:
                    b.text = text
                image,content,_,batch = self.batches()
                self.assertTrue(batch.ownership_repairs)
                self.assertEqual(prepare_automatic_crop(batch.proposals[0],batch,image,
                    original_image_bytes=content)[2],reason)

    def test_ownership_trace_and_policy_drift_are_rejected(self):
        image,content,_,batch = self.batches()
        altered = copy.deepcopy(batch)
        altered.ownership_repairs[0]["source_evidence"][0]["complementary4"]["fill"] = -.1
        for changed in (altered,replace(batch,ownership_policy="production8")):
            self.assertEqual(prepare_automatic_crop(changed.proposals[0],changed,image,
                original_image_bytes=content)[2],"automatic_proposal_or_structural_evidence_drift")
        changed = replace(batch,ownership_policy="invented")
        self.assertEqual(prepare_automatic_crop(changed.proposals[0],changed,image,
            original_image_bytes=content)[2],"invalid_automatic_ownership_policy")

    def test_diagnostic_cli_explicit_opt_in_is_model_free_and_exports_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/"original.png"
            Image.fromarray(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)).save(original)
            comparison = root/"page-comparison.json"
            comparison.write_text(json.dumps({"input":str(original),"boxes":[
                {"id":b.box_id,"polygon":b.poly.tolist(),"paddle":b.text,"score":b.confidence}
                for b in self.boxes]}))
            with patch.object(diagnostics.importlib,"import_module",side_effect=AssertionError("models forbidden")):
                self.assertEqual(diagnostics.main(["--automatic-proposals","--complementary-ownership",
                    "--comparison",str(comparison),"--output-dir",str(root/"output")]),0)
            report = json.loads((root/"output/automatic-proposals.json").read_text())
            self.assertEqual(report["eligible_preflight_proposals"],1)
            self.assertEqual(report["recognition_calls"],0)
            self.assertEqual(report["live_model_calls"],0)
            self.assertTrue(report["settings"]["complementary_ownership"])
            self.assertTrue(report["pages"][0]["batch"]["ownership_repairs"])
            with self.assertRaises(SystemExit),patch("sys.stderr",new=StringIO()):
                # Parser rejects a repair flag outside the automatic mode.
                diagnostics.main(["--complementary-ownership","--output-dir",str(root/"unused")])

    @local_evidence(manifests=("test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json",))
    def test_identical_seven_page_inputs_preserve_membership_order_and_isolation(self):
        root = Path(__file__).resolve().parents[2]
        saved = json.loads((root/"test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json").read_text())
        counts = Counter()
        expected_repairs = {"ubunchu01_04-comparison":[(14,13,12),(37,36,35)],
                            "ubunchu02_04-comparison":[(14,13),(12,11,10)],
                            "ubunchu03_0002-comparison":[(22,21,26)]}
        for page in saved["pages"]:
            _,image,provenance,raw = load_unit_page(page["comparison_path"],root)
            content = Path(provenance.image_path).read_bytes()
            baseline = propose_automatic_units(raw,image)
            repaired = propose_automatic_units(raw,image,complementary_ownership=True)
            single_page = provenance.page_id == "lmgJZ-comparison"
            changed_children = {38,43} if single_page else set()
            self.assertEqual(repaired.detections,baseline.detections)
            self.assertEqual(tuple(r for r in repaired.relationships if r.source_id not in changed_children),
                             tuple(r for r in baseline.relationships if r.source_id not in changed_children))
            self.assertEqual(repaired.rejected_merges,baseline.rejected_merges)
            self.assertEqual(repaired.unassigned,baseline.unassigned)
            expected_coverage = dict(baseline.coverage)
            if single_page:
                for field,delta in (("main_detections",-2),("tentative_annotations",2),("proposed_units",-2),
                                    ("relationship_conflicts",-2),("unsafe_crops",-3)):
                    expected_coverage[field] += delta
            self.assertEqual(repaired.coverage,expected_coverage)
            self.assertEqual([r["main_ids"] for r in repaired.ownership_repairs],
                             expected_repairs.get(provenance.page_id,[]))
            previous_regions = {i:(r,c) for i,r,c in baseline.regions}
            current_regions = {i:(r,c) for i,r,c in repaired.regions if r > 0}
            self.assertEqual({i:current_regions[i] for i in previous_regions},previous_regions)
            self.assertEqual(set(current_regions)-set(previous_regions),changed_children)
            old_units = {p.main_ids:p for p in baseline.proposals}
            self.assertEqual(set(old_units)-{p.main_ids for p in repaired.proposals},
                             {(38,),(43,)} if single_page else set())
            for old in baseline.proposals:
                if prepare_automatic_crop(old,baseline,image,original_image_bytes=content)[2] is None:
                    counts["eligible_before"] += 1
            for new in repaired.proposals:
                old = old_units[new.main_ids]
                single_parent = single_page and bool(set(new.main_ids)&{37,42})
                if not single_parent:
                    self.assertEqual((new.member_ids,new.main_ids,new.annotation_ids,new.annotation_parents,new.direction,
                        new.crop_bounds,new.crop_conflicts,new.crop_safe,new.word_reasons,new.completeness),
                        (old.member_ids,old.main_ids,old.annotation_ids,old.annotation_parents,old.direction,
                        old.crop_bounds,old.crop_conflicts,old.crop_safe,old.word_reasons,old.completeness))
                _,_,before = prepare_automatic_crop(old,baseline,image,original_image_bytes=content)
                crop,_,after = prepare_automatic_crop(new,repaired,image,original_image_bytes=content)
                if before != after:
                    if single_parent:
                        self.assertEqual(before,"unsafe_automatic_crop")
                        self.assertEqual(new.annotation_parents,((38,37),))
                    else:
                        self.assertEqual(before,"unknown_automated_region_ownership")
                        self.assertTrue(any(tuple(r["member_ids"]) == new.member_ids for r in repaired.ownership_repairs))
                    counts["changed_eligibility"] += 1
                if before is None:
                    self.assertIsNone(after)
                if after is None:
                    counts["eligible_after"] += 1
                    self.assertEqual(crop.tobytes(),image.crop(new.crop_bounds).tobytes())
            counts["repaired_proposals"] += len(repaired.ownership_repairs)
            counts["raw"] += len(raw)
        self.assertEqual(counts["raw"],268)
        self.assertEqual(counts["eligible_before"],52)
        self.assertEqual(counts["eligible_after"],57)
        self.assertEqual(counts["repaired_proposals"],5)
        self.assertEqual(counts["changed_eligibility"],5)


if __name__ == "__main__":
    unittest.main()
