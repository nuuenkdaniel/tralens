"""Bounded companion runs: actual pixels, nonmember polygons, no models."""
import copy
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from tralens.ocr import automatic_units as a
from tralens.ocr.automatic_evidence import OriginalInkEvidence
from tralens.ocr.automatic_recognition import prepare_automatic_crop
from tralens.ocr.detection import OCR
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tralens.ocr.types import Text_Box, Text_Group
from tools import diagnostics
from tools.diagnostics import load_unit_page
from tests.local_evidence import local_evidence


def box(source_id, bounds):
    x1,y1,x2,y2 = bounds
    return SimpleNamespace(box_id=source_id,bbox=bounds,text="字",confidence=.99,
        poly=np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],dtype=np.float64))


class CompanionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((230,290,3),255,dtype=np.uint8)
        cv2.rectangle(self.image,(20,20),(275,210),(0,0,0),2)
        self.boxes = [box(str(x),(x,40,x+20,160)) for x in (45,75,105,145,175)]
        for b in self.boxes:
            for y in (50,80,110):
                cv2.rectangle(self.image,(b.bbox[0]+3,y),(b.bbox[0]+15,y+12),(0,0,0),-1)
        self.groups = [Text_Group(list(reversed(self.boxes[3:])),direction="vertical"),
                       Text_Group(list(reversed(self.boxes[:3])),direction="vertical")]

    def regroup(self, annotations=None, owners=None):
        image = Image.fromarray(self.image)
        page = PageProvenance("fixture","/original.png","1"*64,"2"*64,image_fingerprint(image))
        raw = snapshot_detections(self.boxes,page)
        by_id = {b.box_id:(d,b,i) for i,(d,b) in enumerate(zip(raw,self.boxes))}
        ink = OriginalInkEvidence(self.image)
        glyphs = {b.box_id:ink.glyphs(b) for b in self.boxes}
        before = self.image.tobytes()
        regions = {i:(-7,1.) for i in range(len(raw))} if owners is None else owners
        result = a._regroup_companions(self.groups,regions,
            by_id,glyphs,ink,annotations or {},raw)
        self.assertEqual(self.image.tobytes(),before)
        return result

    def test_bounded_clear_vertical_run_orders_right_to_left_without_completeness_claim(self):
        groups,accepted,rejected = self.regroup()
        self.assertEqual([[b.box_id for b in g.group] for g in groups],[["175","145","105","75","45"]])
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]["completeness"],"unverified_full_utterance")
        self.assertEqual(accepted[0]["crop_conflicts"],())
        self.assertEqual(rejected,())

    def test_connected_balloon_original_divider_blocks_same_region_join(self):
        cv2.line(self.image,(135,22),(135,180),(0,0,0),2)
        groups,accepted,rejected = self.regroup()
        self.assertEqual(len(groups),2)
        self.assertEqual(accepted,())
        self.assertIn("original_boundary_blocks_companion_seam",rejected[0]["reasons"])

    def test_clear_text_corridor_does_not_override_divider_beyond_glyph_ends(self):
        for b in self.boxes:
            cv2.rectangle(self.image,(b.bbox[0]+3,147),(b.bbox[0]+15,159),(0,0,0),-1)
        cv2.line(self.image,(135,159),(135,208),(0,0,0),2)
        groups,accepted,rejected = self.regroup()
        self.assertEqual(accepted,())
        self.assertEqual(len(groups),2)
        self.assertIn("original_boundary_blocks_companion_seam",rejected[0]["reasons"])

    def test_artwork_or_narrow_white_neck_is_not_clear_dialogue_bridge(self):
        self.image[40:160,126:145] = 0
        self.image[85:95,126:145] = 255
        _,accepted,rejected = self.regroup()
        self.assertEqual(accepted,())
        self.assertIn("original_companion_corridor_not_clear_or_local",rejected[0]["reasons"])

    def test_complete_span_still_prevents_long_same_region_chain(self):
        for x in (205,235):
            b = box(str(x),(x,40,x+20,160))
            self.boxes.append(b)
            for y in (50,80,110):
                cv2.rectangle(self.image,(x+3,y),(x+15,y+12),(0,0,0),-1)
        self.groups[0] = Text_Group(list(reversed(self.boxes[3:])),direction="vertical")
        _,accepted,rejected = self.regroup()
        self.assertEqual(accepted,())
        self.assertIn("companion_complete_span_or_alignment_limit",rejected[0]["reasons"])

    def test_interleaved_columns_do_not_supply_a_contiguous_seam(self):
        self.groups = [Text_Group(self.boxes[::2],direction="vertical"),
                       Text_Group(self.boxes[1::2],direction="vertical")]
        _,accepted,rejected = self.regroup()
        self.assertEqual(accepted,())
        self.assertIn("no_contiguous_resolvable_companion_seam",rejected[0]["reasons"])

    def test_iterative_three_group_join_rechecks_complete_span_after_first_merge(self):
        b = box("205",(205,40,225,160))
        self.boxes.append(b)
        for y in (50,80,110):
            cv2.rectangle(self.image,(208,y),(220,y+12),(0,0,0),-1)
        self.groups = [Text_Group(list(reversed(self.boxes[4:])),direction="vertical"),
                       Text_Group(list(reversed(self.boxes[2:4])),direction="vertical"),
                       Text_Group(list(reversed(self.boxes[:2])),direction="vertical")]
        groups,accepted,rejected = self.regroup()
        self.assertEqual(len(accepted),1)
        self.assertEqual(sorted(len(g.group) for g in groups),[2,4])
        self.assertTrue(any("companion_complete_span_or_alignment_limit" in row["reasons"] for row in rejected))
        self.assertCountEqual([b.box_id for g in groups for b in g.group],[b.box_id for b in self.boxes])

    def test_rotated_nonmember_blocks_actual_union_rectangle(self):
        intruder = box("intruder",(126,85,134,93))
        intruder.poly = np.array([(130,85),(134,89),(130,93),(126,89)],dtype=np.float64)
        self.boxes.append(intruder)
        _,accepted,rejected = self.regroup()
        self.assertEqual(accepted,())
        self.assertIn("companion_union_has_nonmember_conflicts",rejected[0]["reasons"])
        self.assertEqual(rejected[0]["crop_conflicts"][0]["source_id"],"intruder")

    def test_annotations_keep_their_parent_and_do_not_become_main_columns(self):
        child = box("reading",(62,55,67,75))
        self.boxes.append(child)
        annotations = {"reading":"45"}
        before = dict(annotations)
        groups,accepted,_ = self.regroup(annotations)
        self.assertEqual(len(accepted),1)
        self.assertEqual(len(groups[0].group),5)
        self.assertNotIn("reading",accepted[0]["main_ids"])
        self.assertEqual(annotations,before)

    def test_same_region_alone_or_unsupported_orientation_never_qualifies(self):
        for owners in ({i:(7,1.) for i in range(5)},
                       {i:(-7 if i < 3 else -8,1.) for i in range(5)}):
            _,accepted,_ = self.regroup(owners=owners)
            self.assertEqual(accepted,())
        for b in self.boxes:
            b.bbox = (b.bbox[0],40,b.bbox[0]+20,55)
        _,accepted,rejected = self.regroup()
        self.assertEqual(accepted,())
        self.assertIn("unsupported_companion_glyph_flow_or_orientation",rejected[0]["reasons"])


class SavedCompanionTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[2]

    def page(self):
        return load_unit_page(self.root/"test-results/crop-isolation-v2/ubunchu02_04-comparison.json",self.root)

    @local_evidence(comparisons=("test-results/crop-isolation-v2/ubunchu02_04-comparison.json",))
    def test_saved_split_is_complete_link_span_not_positive_ownership_or_completeness(self):
        _,image,page,raw = self.page()
        before = a.propose_automatic_units(raw,image,complementary_ownership=True)
        batch = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
        self.assertEqual(len(batch.companion_regroupings),1,batch.companion_rejections)
        merged = next(p for p in batch.proposals if len(p.main_ids) == 5)
        self.assertEqual(merged.main_ids,(14,13,12,11,10))
        self.assertEqual(merged.crop_bounds,(716,576,965,788))
        self.assertEqual(merged.completeness,"unverified_full_utterance")
        self.assertFalse(merged.reviewed)
        self.assertEqual(batch.detections,before.detections)
        self.assertEqual(batch.relationships,before.relationships)
        self.assertEqual(batch.regions,before.regions)
        self.assertEqual(batch.ownership_repairs,before.ownership_repairs)
        reversed_batch = a.propose_automatic_units(tuple(reversed(raw)),image,
            complementary_ownership=True,regroup_companions=True)
        self.assertEqual(batch.proposals,reversed_batch.proposals)
        self.assertEqual(batch.companion_regroupings,reversed_batch.companion_regroupings)
        self.assertEqual(len(batch.proposals),len(before.proposals)-1)
        self.assertTrue(any(p.main_ids == (23,22,21) for p in batch.proposals))
        content = Path(page.image_path).read_bytes()
        crop,evidence,error = prepare_automatic_crop(merged,batch,image,original_image_bytes=content)
        self.assertIsNone(error)
        self.assertEqual(crop.tobytes(),image.crop(merged.crop_bounds).tobytes())
        self.assertEqual(evidence.completeness,"unverified_full_utterance")

    @local_evidence(comparisons=("test-results/crop-isolation-v2/ubunchu02_04-comparison.json",))
    def test_facade_exposes_both_opt_ins_without_changing_legacy_groups_or_raw_boxes(self):
        _,_,page,raw = self.page()
        boxes = [Text_Box(d.source_id,d.confidence,d.polygon,d.text) for d in raw]
        ocr = OCR.__new__(OCR)
        ocr.image_path = page.image_path
        ocr.predict = Mock(return_value=[boxes])
        ocr.review_boxes = Mock()
        before = [(b.paddle_box_id,b.paddle_text,b.paddle_poly.tolist()) for b in boxes]
        default = ocr.process_images()
        opted = ocr.process_images(propose_units=True,complementary_ownership=True,regroup_companions=True)
        self.assertEqual([[b.box_id for b in g.group] for g in default[0]],
                         [[b.box_id for b in g.group] for g in opted[0]])
        self.assertEqual(before,[(b.paddle_box_id,b.paddle_text,b.paddle_poly.tolist()) for b in boxes])
        self.assertEqual(len(ocr.automatic_unit_proposals[0].companion_regroupings),1)
        for flags in ({"complementary_ownership":True},{"regroup_companions":True},
                      {"propose_units":True,"regroup_companions":True},
                      {"propose_units":True,"complementary_ownership":"yes"}):
            ocr.predict.reset_mock()
            with self.assertRaises(ValueError):
                ocr.process_images(**flags)
            ocr.predict.assert_not_called()

    @local_evidence(comparisons=("test-results/crop-isolation-v2/ubunchu02_04-comparison.json",))
    def test_policy_and_regrouping_trace_drift_fail_fresh_validation(self):
        _,image,page,raw = self.page()
        batch = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
        merged = next(p for p in batch.proposals if len(p.main_ids) == 5)
        altered = copy.deepcopy(batch)
        altered.companion_regroupings[0]["corridor"]["white_fraction"] = 0.
        for changed in (altered,replace(batch,companion_policy="disabled")):
            self.assertEqual(prepare_automatic_crop(merged,changed,image,
                original_image_bytes=Path(page.image_path).read_bytes())[2],"automatic_proposal_or_structural_evidence_drift")

    @local_evidence(comparisons=("test-results/crop-isolation-v2/ubunchu02_04-comparison.json",))
    def test_regrouping_adds_only_local_seam_scan_and_preflight_still_recomputes(self):
        _,image,page,raw = self.page()
        with patch.object(cv2,"connectedComponentsWithStats",wraps=cv2.connectedComponentsWithStats) as components:
            a.propose_automatic_units(raw,image,complementary_ownership=True)
            previous_calls = components.call_count
            page_shape = (image.height,image.width)
            previous_page_calls = sum(call.args[0].shape == page_shape for call in components.call_args_list)
            components.reset_mock()
            batch = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
            regrouped_calls = components.call_count
            self.assertEqual(sum(call.args[0].shape == page_shape for call in components.call_args_list),previous_page_calls)
            self.assertEqual(previous_page_calls,4)
            self.assertEqual(regrouped_calls,previous_calls+1)
            components.reset_mock()
            merged = next(p for p in batch.proposals if len(p.main_ids) == 5)
            self.assertIsNone(prepare_automatic_crop(merged,batch,image,
                original_image_bytes=Path(page.image_path).read_bytes())[2])
            self.assertEqual(components.call_count,regrouped_calls)
            self.assertGreater(previous_calls,0)

    @local_evidence(comparisons=("test-results/crop-isolation-v2/ubunchu02_04-comparison.json",))
    def test_cli_exports_regrouping_and_unverified_completeness_without_models(self):
        comparison,_,_,_ = self.page()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"output"
            with patch.object(diagnostics.importlib,"import_module",side_effect=AssertionError("models forbidden")):
                self.assertEqual(diagnostics.main(["--automatic-proposals","--complementary-ownership",
                    "--regroup-companions","--comparison",comparison["comparison_path"],
                    "--output-dir",str(output)]),0)
            report = json.loads((output/"automatic-proposals.json").read_text())
            self.assertTrue(report["settings"]["regroup_companions"])
            self.assertEqual(len(report["pages"][0]["batch"]["companion_regroupings"]),1)
            self.assertEqual(report["recognition_calls"],0)
            self.assertEqual(report["live_model_calls"],0)
            self.assertTrue(all(unit["proposal"]["completeness"] == "unverified_full_utterance"
                                for unit in report["pages"][0]["units"]))

    @local_evidence(manifests=("test-results/ownership-complementary-parent-v1/automatic-proposals.json",
                               "test-results/companion-regroup-parent-v1/automatic-proposals.json"))
    def test_identical_seven_pages_change_only_the_supported_companion_partition(self):
        saved = json.loads((self.root/"test-results/ownership-complementary-parent-v1/automatic-proposals.json").read_text())
        replay = json.loads((self.root/"test-results/companion-regroup-parent-v1/automatic-proposals.json").read_text())
        expected_batches = {page["page"]["page_id"]:page["batch"] for page in replay["pages"]}
        counts = Counter()
        for page in saved["pages"]:
            _,image,provenance,raw = load_unit_page(page["comparison_path"],self.root)
            old = a.propose_automatic_units(raw,image,complementary_ownership=True)
            new = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
            saved_batch = expected_batches[provenance.page_id]
            self.assertEqual(json.loads(json.dumps([asdict(d) for d in new.detections])),saved_batch["detections"])
            historical = {tuple(p["main_ids"]):p for p in saved_batch["proposals"]}
            for proposal in new.proposals:
                if provenance.page_id == "lmgJZ-comparison" and set(proposal.main_ids)&{37,42,44}:
                    continue  # Role and singleton deltas are checked in their focused suites.
                actual = asdict(proposal)
                actual.pop("proposal_id")
                expected = dict(historical[proposal.main_ids])
                expected.pop("proposal_id")
                self.assertEqual(json.loads(json.dumps(actual)),expected)
            content = Path(provenance.image_path).read_bytes()
            self.assertEqual(old.detections,new.detections)
            self.assertEqual(old.evidence_sha256,new.evidence_sha256)
            self.assertEqual(old.relationships,new.relationships)
            old_regions = {i:(r,c) for i,r,c in old.regions}
            new_regions = {i:(r,c) for i,r,c in new.regions}
            self.assertEqual({i:new_regions[i] for i in old_regions},old_regions)
            self.assertEqual(set(new_regions)-set(old_regions),{45,46} if provenance.page_id == "lmgJZ-comparison" else set())
            self.assertEqual(old.ownership_repairs,new.ownership_repairs)
            self.assertEqual(old.rejected_merges,new.rejected_merges)
            self.assertEqual(old.unassigned,new.unassigned)
            self.assertEqual(new.coverage["unsafe_crops"],old.coverage["unsafe_crops"]-(2 if provenance.page_id == "lmgJZ-comparison" else 0))
            previous = {(p.main_ids,p.annotation_parents):p for p in old.proposals}
            current = {(p.main_ids,p.annotation_parents):p for p in new.proposals}
            if provenance.page_id == "ubunchu02_04-comparison":
                self.assertEqual(set(previous)-set(current),{((14,13),()),((12,11,10),())})
                self.assertEqual(set(current)-set(previous),{((14,13,12,11,10),())})
            elif provenance.page_id == "lmgJZ-comparison":
                self.assertEqual(set(previous)-set(current),{((42,41),((43,42),)),((44,),((45,44),(46,44)))})
                self.assertEqual(set(current)-set(previous),{((44,42,41),((43,42),(45,44),(46,44)))})
            else:
                self.assertEqual(set(previous),set(current))
            for key in set(previous)&set(current):
                x,y = previous[key],current[key]
                self.assertEqual(replace(x,proposal_id=y.proposal_id),y)
            eligible_members = {}
            for batch,name in ((old,"before"),(new,"after")):
                eligible_members[name] = set()
                counts[f"proposals_{name}"] += len(batch.proposals)
                counts[f"unsafe_{name}"] += batch.coverage["unsafe_crops"]
                for proposal in batch.proposals:
                    crop,_,error = prepare_automatic_crop(proposal,batch,image,original_image_bytes=content)
                    if error is None:
                        counts[f"eligible_{name}"] += 1
                        eligible_members[name].update(proposal.member_ids)
                        self.assertEqual(crop.tobytes(),image.crop(proposal.crop_bounds).tobytes())
                    self.assertEqual(proposal.completeness,"unverified_full_utterance")
            self.assertEqual(eligible_members["after"],eligible_members["before"]|
                             ({41,42,43,44,45,46} if provenance.page_id == "lmgJZ-comparison" else set()))
            counts["raw"] += len(raw)
            counts["regroupings"] += len(new.companion_regroupings)
        self.assertEqual(counts,{"raw":268,"regroupings":2,"proposals_before":134,"proposals_after":132,
                                  "unsafe_before":21,"unsafe_after":19,"eligible_before":57,"eligible_after":57})


if __name__ == "__main__":
    unittest.main()
