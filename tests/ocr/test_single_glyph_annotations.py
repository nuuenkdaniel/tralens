"""Original-image single-glyph role proposals; provisional roles are not text truth."""
import copy
import hashlib
import json
import unittest
from collections import Counter
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tralens.ocr import automatic_units as a
from tralens.ocr.automatic_recognition import prepare_automatic_crop
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tools.diagnostics import load_unit_page
from tests.local_evidence import local_evidence


def box(identifier,bounds,text="漢字漢字"):
    x,y,r,s = bounds
    return {"id":identifier,"polygon":[[x,y],[r,y],[r,s],[x,s]],"paddle":text,"score":.3,"status":"review"}


class SingleGlyphGeometryTests(unittest.TestCase):
    pattern = ("00100000","11111100","00100100","00100100",
               "01111110","01000100","00111100","00010000")

    def setUp(self):
        self.pixels = np.full((160,170,3),255,dtype=np.uint8)
        cv2.rectangle(self.pixels,(15,10),(155,145),(0,0,0),2)
        for y in (30,55,80,105):
            cv2.rectangle(self.pixels,(40,y),(55,y+15),(0,0,0),-1)
        self.draw_pattern(58,32)
        self.boxes = [box("main",(32,22,66,125)),box("reading",(56,28,69,48),"4")]

    def draw_pattern(self,x,y,scale=1):
        for row,line in enumerate(self.pattern):
            for column,value in enumerate(line):
                if value == "1":
                    self.pixels[y+row*scale:y+(row+1)*scale,x+column*scale:x+(column+1)*scale] = 0

    def propose(self, enabled=True):
        image = Image.fromarray(cv2.cvtColor(self.pixels,cv2.COLOR_BGR2RGB))
        stream = BytesIO()
        image.save(stream,format="PNG")
        content = stream.getvalue()
        page = PageProvenance("fixture","/original.png",hashlib.sha256(content).hexdigest(),"2"*64,image_fingerprint(image))
        before = copy.deepcopy(self.boxes)
        pixels = image.tobytes()
        batch = a.propose_automatic_units(snapshot_detections(self.boxes,page),image,complementary_ownership=enabled)
        self.assertEqual(before,self.boxes)
        self.assertEqual(image.tobytes(),pixels)
        mains = [i for p in batch.proposals for i in p.main_ids]
        annotations = [i for p in batch.proposals for i in p.annotation_ids]
        self.assertCountEqual(mains+annotations+[i for i,_ in batch.unassigned],[b["id"] for b in self.boxes])
        self.assertEqual(len(mains+annotations),len(set(mains+annotations)))
        return image,content,batch

    def assert_unattached(self):
        _,_,batch = self.propose()
        self.assertFalse(any("reading" in p.annotation_ids for p in batch.proposals))
        return batch

    def test_noisy_single_reading_requires_explicit_opt_in_and_retains_raw_evidence(self):
        _,_,legacy = self.propose(False)
        self.assertFalse(any("reading" in p.annotation_ids for p in legacy.proposals))
        for text in ("4","中","x"):
            self.boxes[1]["paddle"] = text
            image,content,batch = self.propose()
            proposal = next(p for p in batch.proposals if "main" in p.main_ids)
            self.assertEqual(proposal.annotation_parents,(("reading","main"),))
            self.assertEqual(proposal.main_ids,("main",))
            self.assertIn("annotation_wording_corroboration_missing",proposal.word_reasons)
            self.assertFalse(proposal.reviewed)
            self.assertEqual(proposal.completeness,"unverified_full_utterance")
            self.assertEqual(next(d.text for d in batch.detections if d.source_id == "reading"),text)
            if text == "4":
                self.assertEqual(batch.detections,legacy.detections)
            crop,_,error = prepare_automatic_crop(proposal,batch,image,original_image_bytes=content)
            self.assertIsNone(error)
            self.assertEqual(crop.tobytes(),image.crop(proposal.crop_bounds).tobytes())

    def test_short_dialogue_outside_parent_margin_stays_main(self):
        self.pixels[28:49,56:70] = 255
        self.draw_pattern(70,32)
        self.boxes[1] = box("reading",(68,28,82,48),"4")
        self.assert_unattached()

    def test_single_kana_dialogue_is_not_reclassified_by_new_noisy_reading_rule(self):
        # Existing kana wording corroboration is a separate legacy hypothesis;
        # a kana-only parent provides no kanji/leading-glyph support here.
        self.boxes[0]["paddle"] = "はいはい"
        self.boxes[1]["paddle"] = "4"
        self.assert_unattached()

    def test_speck_connected_hatching_and_ring_do_not_supply_glyph_position_proof(self):
        for shape in ("speck","hatching","ring"):
            with self.subTest(shape=shape):
                self.pixels[28:49,56:70] = 255
                if shape == "speck":
                    cv2.rectangle(self.pixels,(60,34),(61,35),(0,0,0),-1)
                elif shape == "hatching":
                    cv2.line(self.pixels,(58,32),(64,32),(0,0,0),1)
                    for x in (58,61,64):
                        cv2.line(self.pixels,(x,32),(x,39),(0,0,0),1)
                else:
                    cv2.circle(self.pixels,(62,36),3,(0,0,0),1)
                self.assert_unattached()

    def test_punctuation_readings_do_not_become_furigana(self):
        for text in ("!","?","。","ー"):
            self.boxes[1]["paddle"] = text
            self.assert_unattached()

    def test_comparable_scale_text_and_clipped_parent_stroke_do_not_attach(self):
        self.pixels[28:49,56:70] = 255
        self.draw_pattern(58,32,scale=2)
        self.boxes[1] = box("reading",(56,28,76,52),"4")
        self.assert_unattached()
        self.boxes[1] = box("reading",(43,32,51,40),"4")
        self.assert_unattached()

    def test_duplicate_reading_footprints_do_not_create_two_annotations(self):
        self.boxes.append(dict(self.boxes[1],id="duplicate"))
        self.assert_unattached()

    def test_competing_main_parents_remain_ambiguous(self):
        self.boxes.append(dict(self.boxes[0],id="other-main"))
        batch = self.assert_unattached()
        self.assertTrue(any(r.source_id == "reading" and "ambiguous_annotation_parents" in r.reasons
                            for r in batch.relationships))

    def test_original_balloon_boundary_is_not_erased_by_overlapping_ocr_boxes(self):
        cv2.line(self.pixels,(57,12),(57,142),(0,0,0),1)
        self.boxes[1] = box("reading",(58,28,71,48),"4")
        batch = self.assert_unattached()
        self.assertTrue(any(r.source_id == "reading" and "annotation_crosses_original_boundary" in r.reasons
                            for r in batch.relationships))

    def test_mark_beside_later_main_glyph_does_not_match_leading_glyph_rule(self):
        self.pixels[28:49,56:70] = 255
        self.draw_pattern(58,60)
        self.boxes[1] = box("reading",(56,56,69,76),"4")
        self.assert_unattached()

    def test_nonmember_isolation_is_still_enforced_after_tentative_attachment(self):
        self.boxes.append(box("intruder",(34,45,38,50),"!"))
        image,content,batch = self.propose()
        proposal = next(p for p in batch.proposals if "main" in p.main_ids)
        self.assertEqual(proposal.annotation_parents,(("reading","main"),))
        self.assertFalse(proposal.crop_safe)
        self.assertEqual(prepare_automatic_crop(proposal,batch,image,original_image_bytes=content)[2],"unsafe_automatic_crop")

    def test_mutated_single_glyph_evidence_is_rejected_by_fresh_validation(self):
        image,content,batch = self.propose()
        altered = copy.deepcopy(batch)
        relation = next(r for r in altered.relationships if r.parent_id == "main")
        relation.image_evidence[0]["single_glyph_position"]["original_ink_gap"] = 999
        self.assertEqual(prepare_automatic_crop(altered.proposals[0],altered,image,
            original_image_bytes=content)[2],"automatic_proposal_or_structural_evidence_drift")
        old_policy = replace(batch,ownership_policy="enclosed_complementary_connectivity_v1")
        self.assertEqual(prepare_automatic_crop(old_policy.proposals[0],old_policy,image,
            original_image_bytes=content)[2],"invalid_automatic_ownership_policy")

    def test_mixed_main_routing_is_not_overridden_by_a_single_glyph_role(self):
        self.boxes[0]["paddle"] = "漢字Latin"
        image,content,batch = self.propose()
        proposal = next(p for p in batch.proposals if "main" in p.main_ids)
        self.assertEqual(proposal.annotation_parents,(("reading","main"),))
        self.assertEqual(prepare_automatic_crop(proposal,batch,image,
            original_image_bytes=content)[2],"mixed_script_not_routed_to_manga_ocr")


class SavedSingleGlyphTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[2]

    @local_evidence(comparisons=("test-results/crop-isolation-v2/lmgJZ-comparison.json",))
    def test_original_page_provisional_single_glyph_roles_require_pixel_evidence(self):
        _,image,_,raw = load_unit_page(self.root/"test-results/crop-isolation-v2/lmgJZ-comparison.json",self.root)
        batch = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
        actual = {(r.source_id,r.parent_id) for r in batch.relationships if r.parent_id is not None}
        for source_id,parent_id in ((38,37),(43,42)):
            rows = [row for r in batch.relationships if r.source_id == source_id for row in r.image_evidence
                    if row["parent_id"] == parent_id]
            with self.subTest(source_id=source_id):
                self.assertIn((source_id,parent_id),actual)
                proof = rows[0]["single_glyph_position"]
                self.assertTrue(proof["supported"])
                self.assertEqual(proof["leading_axis_overlap"],1.)
                self.assertEqual(proof["original_ink_gap"],2)
        self.assertFalse(batch.reviewed)
        self.assertTrue(all(not proposal.reviewed for proposal in batch.proposals))
        reversed_batch = a.propose_automatic_units(tuple(reversed(raw)),image,
            complementary_ownership=True,regroup_companions=True)
        self.assertEqual(batch.proposals,reversed_batch.proposals)
        self.assertEqual(batch.relationships,reversed_batch.relationships)

    @local_evidence(manifests=("test-results/ownership-complementary-parent-v1/automatic-proposals.json",))
    def test_identical_seven_page_roles_crop_isolation_and_preflight_deltas(self):
        # Isolate the single-glyph policy from subsequent companion extensions.
        saved = json.loads((self.root/"test-results/ownership-complementary-parent-v1/automatic-proposals.json").read_text())
        counts = Counter()
        for page in saved["pages"]:
            _,image,provenance,raw = load_unit_page(page["comparison_path"],self.root)
            batch = a.propose_automatic_units(raw,image,complementary_ownership=True)
            self.assertEqual(json.loads(json.dumps([asdict(d) for d in batch.detections])),page["batch"]["detections"])
            previous = {tuple(p["main_ids"]):p for p in page["batch"]["proposals"]}
            old_parents = {tuple(pair) for p in previous.values() for pair in p["annotation_parents"]}
            new_parents = {pair for p in batch.proposals for pair in p.annotation_parents}
            additions = {(38,37),(43,42)} if provenance.page_id == "lmgJZ-comparison" else set()
            self.assertEqual(new_parents,old_parents|additions)
            if additions:
                self.assertTrue({(8,7),(12,11),(18,17),(21,14),(22,19)} <= new_parents)
                self.assertTrue({13,16} <= {i for p in batch.proposals for i in p.main_ids})
            for proposal in batch.proposals:
                old = previous[proposal.main_ids]
                if not additions or not set(proposal.main_ids)&{37,42}:
                    actual = json.loads(json.dumps(asdict(proposal)))
                    actual.pop("proposal_id")
                    expected = dict(old)
                    expected.pop("proposal_id")
                    self.assertEqual(actual,expected)
                elif proposal.main_ids == (37,):
                    self.assertEqual(proposal.member_ids,(37,38))
                    self.assertEqual(proposal.annotation_parents,((38,37),))
                    self.assertEqual(proposal.crop_bounds,(426,953,464,1075))
                    self.assertTrue(proposal.crop_safe)
                else:
                    self.assertEqual(proposal.main_ids,(42,41))
                    self.assertEqual(proposal.member_ids,(42,41,43))
                    self.assertEqual(proposal.annotation_parents,((43,42),))
                    self.assertEqual(proposal.crop_bounds,(518,979,578,1082))
                    self.assertFalse(proposal.crop_safe)
                    self.assertEqual(tuple(c.source_id for c in proposal.crop_conflicts),(44,))
                    self.assertEqual(proposal.crop_conflicts[0].intersection_area,404.)
                crop,_,error = prepare_automatic_crop(proposal,batch,image,
                    original_image_bytes=Path(provenance.image_path).read_bytes())
                if error is None:
                    counts["eligible_after"] += 1
                    self.assertEqual(crop.tobytes(),image.crop(proposal.crop_bounds).tobytes())
                counts["unsafe_after"] += not proposal.crop_safe
                self.assertEqual(proposal.completeness,"unverified_full_utterance")
            counts["raw"] += len(raw)
            counts["units_before"] += len(page["units"])
            counts["units_after"] += len(batch.proposals)
            counts["annotations_before"] += page["coverage"]["tentative_annotations"]
            counts["annotations_after"] += batch.coverage["tentative_annotations"]
            counts["unsafe_before"] += page["coverage"]["unsafe_crops"]
            counts["conflicts_before"] += page["coverage"]["relationship_conflicts"]
            counts["conflicts_after"] += batch.coverage["relationship_conflicts"]
            counts["eligible_before"] += sum(entry["eligibility_preflight"]["eligible"] for entry in page["units"])
        self.assertEqual(counts,{"raw":268,"units_before":136,"units_after":134,
            "annotations_before":14,"annotations_after":16,"unsafe_before":24,"unsafe_after":21,
            "conflicts_before":13,"conflicts_after":11,"eligible_before":56,"eligible_after":57})


if __name__ == "__main__":
    unittest.main()
