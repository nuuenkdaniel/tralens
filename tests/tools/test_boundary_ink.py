"""External boundary-ink observations cannot identify missing glyphs or completeness."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tralens.ocr.automatic_evidence import OriginalInkEvidence
from tralens.ocr.automatic_units import propose_automatic_units
from tralens.ocr.automatic_recognition import prepare_automatic_crop
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tools import diagnostics
from tools.diagnostics import load_unit_page
from tests.local_evidence import local_evidence


class BoundaryInkObservationTests(unittest.TestCase):
    def fixture(self, *, complete=False, neighbor=False, art=False, holes=False, horizontal=False,
                open_region=False, speed_lines=()):
        image = np.full((160,160,3),255,dtype=np.uint8)
        cv2.rectangle(image,(10,10),(145,145),(0,0,0),2)
        for y in (40,65):
            cv2.rectangle(image,(45,y),(55,y+10),(0,0,0),1 if holes else -1)
        cv2.rectangle(image,(45,88),(55,92 if complete else 100),(0,0,0),-1)
        polygon = np.array([(40,30),(60,30),(60,95),(40,95)])
        if art:
            cv2.line(image,(55,100),(145,140),(0,0,0),2)
        for x1,y1,x2,y2 in speed_lines:
            cv2.line(image,(x1,y1),(x2,y2),(0,0,0),1)
        if open_region:
            image[55:60,:12] = 255
        if horizontal:
            image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
            polygon = np.array([(159-y,x) for x,y in polygon])
        source = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        stream = BytesIO()
        source.save(stream,format="PNG")
        content = stream.getvalue()
        page = PageProvenance("fixture","/original.png",hashlib.sha256(content).hexdigest(),"2"*64,image_fingerprint(source))
        boxes = [{"id":"main","polygon":polygon.tolist(),"paddle":"字字","score":.99,"status":"accepted"}]
        if neighbor:
            boxes.append({"id":"neighbor","polygon":[[45,96],[56,96],[56,104],[45,104]],"paddle":"!","score":.99})
        raw = snapshot_detections(boxes,page)
        batch = propose_automatic_units(raw,source)
        return source,content,batch,image,boxes

    def assessment(self, batch, evidence):
        results = diagnostics.observe_boundary_ink(batch.proposals,batch.detections,evidence)
        return next(result for proposal,result in zip(batch.proposals,results) if "main" in proposal.main_ids)

    def test_crossing_ink_is_ambiguous_without_changing_pixels_bounds_or_eligibility(self):
        image,content,batch,evidence,_ = self.fixture()
        before = copy.deepcopy(asdict(batch))
        pixels = image.tobytes()
        assessment = self.assessment(batch,evidence)
        self.assertEqual(assessment["status"],"ambiguous_boundary_crossing_ink")
        self.assertEqual(assessment["components"][0]["bounds"],(45,88,56,101))
        self.assertEqual(assessment["components"][0]["outside_crop_ink_pixels"],66)
        self.assertEqual(assessment["crop_bounds"],(40,30,60,95))
        self.assertEqual(assessment["completeness"],"unverified_full_utterance")
        self.assertFalse(assessment["is_completeness_assessment"])
        self.assertTrue(assessment["artwork_contamination_possible"])
        self.assertEqual(assessment["glyph_ownership"],"unresolved")
        self.assertEqual(asdict(batch),before)
        self.assertEqual(image.tobytes(),pixels)
        crop,_,error = prepare_automatic_crop(batch.proposals[0],batch,image,original_image_bytes=content)
        self.assertIsNone(error)
        self.assertEqual(crop.tobytes(),image.crop((40,30,60,95)).tobytes())

    def test_horizontal_boundary_observations_remain_ambiguous(self):
        _,_,batch,evidence,_ = self.fixture(horizontal=True)
        self.assertEqual(self.assessment(batch,evidence)["status"],"ambiguous_boundary_crossing_ink")

    def assert_speed_lines_are_not_missing_glyph_evidence(self, lines):
        reference,_,_,_,_ = self.fixture(complete=True)
        image,_,batch,bgr,_ = self.fixture(complete=True,speed_lines=lines)
        before = image.tobytes()
        observation = self.assessment(batch,bgr)
        self.assertEqual(observation["status"],"ambiguous_boundary_crossing_ink")
        self.assertEqual(len(observation["components"]),len(lines))
        self.assertTrue(observation["artwork_contamination_possible"])
        self.assertFalse(observation["is_completeness_assessment"])
        self.assertEqual(observation["glyph_ownership"],"unresolved")
        self.assertIn("speed-line",observation["limitations"])
        self.assertEqual(observation["completeness"],"unverified_full_utterance")
        # All actual text pixels are contained and unchanged. The only observed
        # components are additional speed lines, not missing parts of the text.
        x1,y1,x2,y2 = batch.proposals[0].crop_bounds
        text_pixels = np.zeros((image.height,image.width),dtype=bool)
        reference_dark = np.asarray(reference)[:,:,0] == 0
        text_pixels[y1:y2,x1:x2] = reference_dark[y1:y2,x1:x2]
        self.assertTrue(np.all(np.asarray(image)[text_pixels] == np.asarray(reference)[text_pixels]))
        for component in observation["components"]:
            x,y,r,s = component["bounds"]
            self.assertFalse(np.any(text_pixels[y:s,x:r]))
        self.assertEqual(image.tobytes(),before)
        self.assertEqual(batch.proposals[0].crop_bounds,(40,30,60,95))

    def test_parallel_speed_lines_cross_crop_while_main_text_is_fully_contained(self):
        self.assert_speed_lines_are_not_missing_glyph_evidence(((56,55,64,59),(56,80,65,84)))

    def test_terminal_artwork_stroke_is_ambiguous_despite_contained_text(self):
        self.assert_speed_lines_are_not_missing_glyph_evidence(((41,89,43,101),))

    def test_contained_ink_and_glyph_holes_do_not_establish_completeness(self):
        for holes in (False,True):
            _,_,batch,evidence,_ = self.fixture(complete=True,holes=holes)
            assessment = self.assessment(batch,evidence)
            self.assertNotEqual(assessment["status"],"ambiguous_boundary_crossing_ink")
            self.assertEqual(assessment["components"],())
            self.assertEqual(assessment["completeness"],"unverified_full_utterance")
            self.assertFalse(assessment["is_completeness_assessment"])

    def test_nonmember_text_connected_to_the_same_ink_is_not_claimed_as_missing_punctuation(self):
        _,_,batch,evidence,_ = self.fixture(neighbor=True)
        self.assertTrue(next(p for p in batch.proposals if "main" in p.main_ids).crop_safe)
        self.assertEqual(self.assessment(batch,evidence)["status"],"no_boundary_crossing_observed")

    def test_large_connected_artwork_and_open_background_do_not_supply_an_observation(self):
        for options in ({"art":True},{"open_region":True}):
            _,_,batch,evidence,_ = self.fixture(**options)
            self.assertNotEqual(self.assessment(batch,evidence)["status"],"ambiguous_boundary_crossing_ink")

    def test_disconnected_undetected_dot_is_not_assigned_to_an_utterance(self):
        image,_,batch,evidence,_ = self.fixture(complete=True)
        bgr = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr,(48,103),(51,106),(0,0,0),-1)
        assessment = self.assessment(batch,bgr)
        self.assertEqual(assessment["status"],"no_boundary_crossing_observed")
        self.assertIn("disconnected",assessment["limitations"])
        self.assertEqual(assessment["completeness"],"unverified_full_utterance")
        self.assertFalse(assessment["is_completeness_assessment"])

    def test_invalid_geometry_is_unassessed_not_silently_complete(self):
        _,_,batch,evidence,_ = self.fixture()
        bad = replace(batch.detections[0],polygon=((40,30),(60,95),(40,95),(60,30)))
        assessment = diagnostics.observe_boundary_ink(batch.proposals,(bad,),evidence)[0]
        self.assertEqual(assessment["status"],"not_assessed")
        self.assertEqual(assessment["reason"],"invalid_or_clipped_detection_geometry")

    def test_existing_diagnostics_exports_external_observation_and_unchanged_original_crop(self):
        image,_,_,_,boxes = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/"original.png"
            image.save(original)
            comparison = root/"page-comparison.json"
            comparison.write_text(json.dumps({"input":str(original),"boxes":boxes}))
            with patch.object(diagnostics.importlib,"import_module",side_effect=AssertionError("models forbidden")):
                self.assertEqual(diagnostics.main(["--automatic-proposals","--observe-boundary-ink",
                    "--comparison",str(comparison),"--output-dir",str(root/"output")]),0)
            report = json.loads((root/"output/automatic-proposals.json").read_text())
            self.assertEqual(report["boundary_ink_observation_counts"],{"ambiguous_boundary_crossing_ink":1})
            self.assertNotIn("clipped_ink_assessment_counts",report)
            self.assertEqual(report["recognition_calls"],0)
            self.assertEqual(report["live_model_calls"],0)
            entry = report["pages"][0]["units"][0]
            self.assertEqual(len(entry["image_paths"]),4)
            with Image.open(entry["image_paths"][0]) as crop:
                self.assertEqual(crop.tobytes(),image.crop(entry["proposal"]["crop_bounds"]).tobytes())
            with Image.open(entry["image_paths"][2]) as context:
                self.assertEqual(context.tobytes(),image.crop(entry["boundary_ink_observation"]["scan_bounds"]).tobytes())
            self.assertNotIn("clipped_ink_assessment",entry)
            with patch.object(diagnostics,"observe_boundary_ink",side_effect=AssertionError("disabled diagnostic must not run")):
                baseline = json.loads(json.dumps(diagnostics.run_automatic_proposals([comparison],root/"baseline")))
            self.assertEqual(baseline["pages"][0]["batch"],report["pages"][0]["batch"])
            self.assertEqual(baseline["pages"][0]["units"][0]["eligibility_preflight"],entry["eligibility_preflight"])
            self.assertEqual(baseline["pages"][0]["units"][0]["recognition"],entry["recognition"])
            self.assertNotIn("boundary_ink_observation",baseline["pages"][0]["units"][0])


class SavedBoundaryInkTests(unittest.TestCase):
    @local_evidence(comparisons=("test-results/crop-isolation-v2/lmgJZ-comparison.json",))
    def test_warning_original_components_against_detection_union(self):
        root = Path(__file__).resolve().parents[2]
        _,image,_,raw = load_unit_page(root/"test-results/crop-isolation-v2/lmgJZ-comparison.json",root)
        batch = propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
        proposal = next(p for p in batch.proposals if 17 in p.main_ids)
        ink = OriginalInkEvidence(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)).ink
        count,labels,stats,_ = cv2.connectedComponentsWithStats(ink,connectivity=8)
        x1,y1,x2,y2 = proposal.crop_bounds
        crop = np.zeros_like(ink,dtype=bool)
        crop[y1:y2,x1:x2] = True
        rows = []
        for label in range(1,count):
            x,y,w,h,area = (int(v) for v in stats[label])
            if x >= 458 and y >= 301 and x+w <= 577 and y+h <= 610:
                outside = int(np.count_nonzero((labels == label) & ~crop))
                if outside:
                    rows.append({"bounds":(x,y,x+w,y+h),"area":area,"outside_crop":outside})
        self.assertEqual(proposal.main_ids,(19,17))
        self.assertEqual(proposal.member_ids,(19,17,18,22))
        self.assertEqual(proposal.crop_bounds,(460,303,575,538))
        self.assertTrue(rows)
        # Larger review bounds are not themselves omitted-punctuation evidence.
        # Isolated components wholly inside that extent but beyond this crop
        # are specks; connected artwork can extend beyond the reviewed extent.
        self.assertTrue(all(row["area"] <= 2 for row in rows),rows)
        bgr = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        assessment = next(result for p,result in zip(batch.proposals,diagnostics.observe_boundary_ink(
            batch.proposals,batch.detections,bgr)) if p == proposal)
        self.assertEqual(assessment["status"],"no_boundary_crossing_observed",assessment)
        self.assertEqual(assessment["components"],())
        self.assertFalse(assessment["is_completeness_assessment"])

    @local_evidence(manifests=("test-results/ocr-lifecycle-review-parent-v1/automatic-proposals.json",
                               "test-results/clipped-ink-parent-v1/automatic-proposals.json"))
    def test_seven_saved_batches_are_unchanged_by_diagnostic_only_assessment(self):
        root = Path(__file__).resolve().parents[2]
        replay = json.loads((root/"test-results/ocr-lifecycle-review-parent-v1/automatic-proposals.json").read_text())
        old_report = json.loads((root/"test-results/clipped-ink-parent-v1/automatic-proposals.json").read_text())
        old_observations = {(page["page"]["page_id"],tuple(entry["proposal"]["main_ids"])):entry["clipped_ink_assessment"]
                            for page in old_report["pages"] for entry in page["units"]}
        statuses = Counter()
        warnings = []
        for page in replay["pages"]:
            _,image,_,raw = load_unit_page(page["comparison_path"],root)
            batch = propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
            before = copy.deepcopy(asdict(batch))
            bgr = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            assessments = diagnostics.observe_boundary_ink(batch.proposals,batch.detections,bgr)
            self.assertEqual(asdict(batch),before)
            self.assertEqual(json.loads(json.dumps([asdict(d) for d in batch.detections])),page["batch"]["detections"])
            previous_proposals = {tuple(p["main_ids"]):p for p in page["batch"]["proposals"]}
            for proposal in batch.proposals:
                if batch.page.page_id == "lmgJZ-comparison" and set(proposal.main_ids)&{37,42}:
                    continue  # v2 single-glyph role changes, not observer effects.
                actual = asdict(proposal)
                actual.pop("proposal_id")
                expected = dict(previous_proposals[proposal.main_ids])
                expected.pop("proposal_id")
                self.assertEqual(json.loads(json.dumps(actual)),expected)
            for proposal,assessment in zip(batch.proposals,assessments):
                statuses[assessment["status"]] += 1
                self.assertFalse(assessment["is_completeness_assessment"])
                self.assertTrue(assessment["artwork_contamination_possible"])
                previous = old_observations.get((batch.page.page_id,proposal.main_ids))
                if previous is None:
                    self.assertEqual((batch.page.page_id,proposal.main_ids),("lmgJZ-comparison",(44,42,41)))
                    self.assertEqual(assessment["status"],"no_boundary_crossing_observed")
                    continue  # New union, not a change to the observer's predicates.
                numeric = [{k:v for k,v in component.items() if k != "observation"} for component in assessment["components"]]
                old_numeric = [{k:v for k,v in component.items() if k != "observation"} for component in previous["components"]]
                self.assertEqual(json.loads(json.dumps(numeric)),old_numeric)
                if "scan_bounds" in previous:
                    self.assertEqual(list(assessment["scan_bounds"]),previous["scan_bounds"])
                if assessment["components"]:
                    warnings.append((batch.page.page_id,proposal.main_ids,assessment["components"]))
        self.assertEqual(statuses,{"not_assessed":52,"no_boundary_crossing_observed":76,"ambiguous_boundary_crossing_ink":4})
        self.assertEqual({(page,ids) for page,ids,_ in warnings},{
            ("ubunchu01_02-comparison",(15,)),("ubunchu01_02-comparison",(14,16)),
            ("ubunchu01_04-comparison",(38,)),("ubunchu03_0002-comparison",(15,))})


if __name__ == "__main__":
    unittest.main()
