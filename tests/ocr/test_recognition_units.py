"""Recognition unit tests use injected text-only callbacks, never live OCR."""
import copy
import hashlib
import json
import socket
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from tools import diagnostics as d
from tests.local_evidence import local_evidence
from tralens.ocr import recognition_units as u


def saved(source_id, bounds, text, status="review"):
    a, b, c, e = bounds
    return {"id": source_id, "polygon": [[a, b], [c, b], [c, e], [a, e]],
            "paddle": text, "selected": text, "score": .3, "status": status,
            "reason": "original", "metadata": {"source_box_id": source_id}}


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), "white")
        self.image.putpixel((35, 15), (3, 7, 11))
        self.page = u.PageProvenance("fixture", "/fixture/original.png", "1"*64, "2"*64,
                                     u.image_fingerprint(self.image))
        self.boxes = [saved(7, (30, 10, 40, 60), "漢字"),
                      saved(8, (39, 12, 45, 45), "かんじ", "rejected")]
        self.raw = u.snapshot_detections(self.boxes, self.page)
        self.relations = [u.ReviewedRelationship(8, 7, "furigana", self.page, True, "reviewer")]
        self.owners = [u.ReviewedOwnership(i, "bubble-a", self.page, True, "reviewer") for i in (7, 8)]
        self.request = u.UnitRequest((7,), "vertical", "column", True, True, True, "reviewer")
        self.callback = Mock(return_value="漢字")

    def plan(self, raw=None, relations=None, requests=None, owners=None):
        return u.plan_recognition_units(self.raw if raw is None else raw,
                                        self.relations if relations is None else relations,
                                        [self.request] if requests is None else requests,
                                        self.owners if owners is None else owners)

    def recognize(self, batch):
        unit = batch.units[0]
        u.recognize_unit(unit, batch, self.image, self.callback)
        return unit

    def test_raw_snapshots_are_immutable_and_detached_from_source_polygons(self):
        raw = self.raw[0]
        with self.assertRaises(FrozenInstanceError):
            raw.text = "changed"
        self.boxes[0]["polygon"][0][0] = 99
        self.assertEqual(raw.polygon[0], (30, 10))
        self.assertEqual(raw.text, "漢字")
        self.assertEqual(raw.confidence, .3)
        self.assertEqual(self.raw[1].status, "rejected")

    def test_explicit_furigana_parent_and_original_unit_crop_no_member_duplication(self):
        before = copy.deepcopy(self.boxes)
        batch = self.plan()
        unit = self.recognize(batch)
        self.assertEqual(unit.plan.annotation_parents, ((8, 7),))
        self.assertEqual(unit.plan.member_ids, (7, 8))
        self.assertEqual(dict(unit.plan.roles)[8], u.MemberRole.ANNOTATION)
        self.assertEqual(unit.crop.bounds, (28, 8, 47, 62))
        crop = self.callback.call_args.args[0]
        self.assertEqual(crop.tobytes(), self.image.crop(unit.crop.bounds).tobytes())
        self.assertEqual(self.callback.call_args.kwargs, {"source_unit_id": unit.plan.unit_id,
                         "member_ids": (7, 8), "crop_bounds": (28, 8, 47, 62)})
        self.assertEqual(unit.status, u.UnitStatus.ACCEPTED)
        self.assertEqual(len(unit.candidates), 1)
        self.assertEqual(batch.fallback_source_ids, (7, 8))
        self.assertEqual(unit.candidates[0].reference_reading, "漢字")
        self.assertEqual(self.boxes, before)
        self.assertFalse(hasattr(self.raw[0], "unit_candidate"))

    def test_stable_unit_id_independent_of_record_order(self):
        raw = self.raw + (replace(self.raw[1], source_id=9, polygon=((41, 46), (44, 46), (44, 50), (41, 50))),)
        relations = self.relations + [u.ReviewedRelationship(9, 7, "furigana", self.page, True, "reviewer")]
        owners = self.owners + [u.ReviewedOwnership(9, "bubble-a", self.page, True, "reviewer")]
        a = self.plan(raw, relations, owners=owners)
        b = self.plan(raw, list(reversed(relations)), owners=list(reversed(owners)))
        self.assertEqual(a.units[0].plan.unit_id, b.units[0].plan.unit_id)

    def test_small_ordinary_dialogue_is_not_inferred_as_furigana(self):
        batch = self.plan(relations=[])
        self.assertEqual(batch.units[0].plan.member_ids, (7,))
        self.assertEqual(batch.units[0].plan.annotation_parents, ())
        self.assertIn(8, batch.fallback_source_ids)
        unit = self.recognize(batch)
        self.callback.assert_not_called()
        self.assertEqual(unit.reason, "unrelated_neighbor_glyphs")
        request = u.UnitRequest((8,), "vertical", "column", True, True, False, "reviewer")
        ordinary = self.plan(relations=[], requests=[request])
        self.assertEqual(dict(ordinary.units[0].plan.roles)[8], u.MemberRole.MAIN)

    def test_companions_use_explicit_reviewed_order_and_join_main_readings_only(self):
        boxes = [saved(16, (30, 10, 40, 60), "右"), saved(13, (10, 10, 20, 60), "左")]
        raw = u.snapshot_detections(boxes, self.page)
        relationship = u.ReviewedRelationship(13, 16, "same_text_block", self.page, True, "reviewer")
        owners = [u.ReviewedOwnership(i, "bubble-a", self.page, True, "reviewer") for i in (16, 13)]
        request = u.UnitRequest((16, 13), "vertical", "block", True, True, True, "reviewer")
        batch = self.plan(raw, [relationship], [request], owners)
        self.callback.return_value = "右左"
        unit = self.recognize(batch)
        self.assertEqual(unit.plan.main_ids, (16, 13))
        self.assertEqual(unit.plan.reference_reading, "右左")
        self.assertEqual(unit.status, u.UnitStatus.ACCEPTED)
        self.assertEqual(boxes[0]["paddle"], "右")
        self.assertEqual(boxes[1]["paddle"], "左")

    def test_different_granularity_cannot_accept_a_matching_member_reading(self):
        raw = (self.raw[0], replace(self.raw[0], source_id=13, text="別", polygon=((10,10),(20,10),(20,60),(10,60))))
        relation = u.ReviewedRelationship(13, 7, "same_text_block", self.page, True, "reviewer")
        owners = [u.ReviewedOwnership(i, "bubble-a", self.page, True, "reviewer") for i in (7,13)]
        request = u.UnitRequest((7,13), "vertical", "column", True, True, True, "reviewer")
        batch = self.plan(raw, [relation], [request], owners)
        unit = self.recognize(batch)
        self.assertEqual(unit.status, u.UnitStatus.REVIEW)
        self.assertIsNone(unit.candidates[0].reference_reading)
        self.assertEqual(unit.reason, "incomparable_granularity_or_main_readings")

    def test_unreviewed_granularity_and_blank_main_recovery_remain_review(self):
        batch = self.plan(requests=[replace(self.request, granularity_reviewed=False)])
        self.assertEqual(self.recognize(batch).status, u.UnitStatus.REVIEW)
        raw = (replace(self.raw[0], text=""), self.raw[1])
        batch = self.plan(raw=raw)
        self.assertEqual(self.recognize(batch).status, u.UnitStatus.REVIEW)

    def test_unreviewed_or_uncertain_model_relationships_never_authorize_units(self):
        for relation in (replace(self.relations[0], reviewed=False),
                         replace(self.relations[0], reviewed="true"),
                         replace(self.relations[0], relationship=u.UnitRelationship.UNCERTAIN)):
            batch = self.plan(relations=[relation])
            self.assertFalse(batch.units[0].approved)
            self.recognize(batch)
            self.assertEqual(batch.fallback_source_ids, (7, 8))
        self.callback.assert_not_called()

    def test_foreign_ids_page_mismatch_and_invalid_ids_are_not_approved(self):
        foreign = replace(self.relations[0], source_id=99)
        wrong_page = replace(self.relations[0], page=replace(self.page, image_sha256="3"*64))
        for relation in (foreign, wrong_page):
            self.assertFalse(self.plan(relations=[relation]).units[0].approved)
        for source_id in (True, 7.0, ""):
            self.assertRaises(ValueError, u.ReviewedRelationship, source_id, 7, "furigana", self.page)
        self.assertFalse(self.plan(requests=[replace(self.request, main_ids=(999,))]).units[0].approved)

    def test_duplicate_multiple_parent_and_cycle_records_block_all_affected_units(self):
        cases = [self.relations*2,
                 self.relations + [replace(self.relations[0], parent_id=8)],
                 self.relations + [u.ReviewedRelationship(7, 8, "furigana", self.page, True, "reviewer")]]
        for records in cases:
            batch = self.plan(relations=records)
            self.assertFalse(batch.units[0].approved)
            self.recognize(batch)
        self.callback.assert_not_called()

    def test_mixed_unknown_ownership_and_multiple_unit_memberships_remain_unresolved(self):
        for owners in ([], self.owners[:1], [self.owners[0], replace(self.owners[1], bubble_id="other")]):
            self.assertFalse(self.plan(owners=owners).units[0].approved)
        batch = self.plan(requests=[self.request, self.request])
        self.assertTrue(all(not unit.approved for unit in batch.units))
        self.assertEqual(batch.fallback_source_ids, (7,8))

    def test_unrelated_substantial_neighbor_between_companions_blocks_crop(self):
        raw = (self.raw[0], replace(self.raw[0], source_id=13, text="左", polygon=((10,10),(20,10),(20,60),(10,60))),
               replace(self.raw[0], source_id=20, text="unrelated", polygon=((22,15),(28,15),(28,50),(22,50))))
        relations = [u.ReviewedRelationship(13,7,"same_text_block",self.page,True,"reviewer")]
        owners = [u.ReviewedOwnership(i,"bubble-a",self.page,True,"reviewer") for i in (7,13,20)]
        request = replace(self.request, main_ids=(7,13), granularity=u.ReadingGranularity.BLOCK)
        batch = self.plan(raw, relations, [request], owners)
        unit = self.recognize(batch)
        self.callback.assert_not_called()
        self.assertEqual(unit.reason, "unrelated_neighbor_glyphs")
        owners[-1] = replace(owners[-1], bubble_id="other")
        self.assertEqual(self.recognize(self.plan(raw, relations, [request], owners)).reason,
                         "mixed_bubble_ownership")

    def test_padding_is_trimmed_to_exclude_unrelated_neighbor_without_masking(self):
        outside = replace(self.raw[0], source_id=99, polygon=((46,10),(55,10),(55,60),(46,60)))
        batch = self.plan(raw=self.raw+(outside,))
        unit = self.recognize(batch)
        self.assertEqual(unit.crop.bounds, (28,8,46,62))
        self.assertEqual(self.callback.call_args.args[0].tobytes(), self.image.crop(unit.crop.bounds).tobytes())

    def test_unknown_external_bubble_and_disjoint_polygon_crop_contamination_block(self):
        external = replace(self.raw[0], source_id=99, polygon=((32,20),(37,20),(37,40),(32,40)))
        batch = self.plan(raw=self.raw+(external,))
        self.assertEqual(self.recognize(batch).reason, "unknown_bubble_ownership")
        self.callback.assert_not_called()

    def test_invalid_geometry_clipping_and_image_provenance_never_call_recognizer(self):
        for polygon in (((30,10),(40,60),(30,60),(40,10)),
                        ((-1,10),(40,10),(40,60),(-1,60)),
                        ((30,10),(40,10),(float("nan"),60),(30,60))):
            batch = self.plan(raw=(replace(self.raw[0], polygon=polygon),self.raw[1]))
            self.assertEqual(self.recognize(batch).status, u.UnitStatus.REVIEW)
        batch = self.plan()
        altered = self.image.copy()
        altered.putpixel((0,0),(0,0,0))
        u.recognize_unit(batch.units[0], batch, altered, self.callback)
        self.assertEqual(batch.units[0].reason, "original_image_provenance_mismatch")
        self.callback.assert_not_called()

    def test_disagreement_spans_one_bounded_alternate_and_no_raw_updates(self):
        original = copy.deepcopy(self.boxes)
        batch = self.plan()
        unit = batch.units[0]
        self.callback.side_effect = ["別字", "漢字"]
        u.recognize_unit(unit,batch,self.image,self.callback)
        self.assertEqual(unit.reason, "unit_engine_disagreement")
        self.assertTrue(unit.candidates[0].differing_spans)
        u.recognize_unit(unit,batch,self.image,self.callback,alternate=True)
        u.recognize_unit(unit,batch,self.image,self.callback,alternate=True)
        u.recognize_unit(unit,batch,self.image,self.callback)
        self.assertEqual(self.callback.call_count,2)
        self.assertEqual(unit.alternate_attempts,1)
        self.assertEqual(unit.status,u.UnitStatus.REVIEW)
        self.assertEqual(unit.reason,"alternate_unit_candidates_disagree")
        self.assertEqual(unit.candidates[-1].crop.bounds,(30,10,45,60))
        self.assertEqual(self.boxes,original)

    def test_callback_errors_and_scores_do_not_overwrite_raw_confidence(self):
        batch = self.plan()
        self.callback.return_value = ("text", .99)
        self.assertRaises(TypeError,self.recognize,batch)
        self.assertEqual(batch.units[0].attempts,1)
        self.assertEqual(self.raw[0].confidence,.3)
        u.recognize_unit(batch.units[0],batch,self.image,self.callback)
        self.assertEqual(self.callback.call_count,1)


class UnitExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        Image.new("RGB",(100,100),"white").save(self.root / "page.png")
        self.boxes = [saved(7,(30,10,40,60),"漢字"), saved(8,(39,12,45,45),"かんじ","rejected")]
        self.comparison = self.root / "fixture.json"
        self.comparison.write_text(json.dumps({"input":"page.png","boxes":self.boxes}))
        _, _, page, _ = d.load_unit_page(self.comparison,self.root)
        self.review = self.root / "reviewed.json"
        self.spec = {"pages":[{"page":asdict(page),
            "relationships":[{"source_id":8,"parent_id":7,"relationship":"furigana","reviewed":True,"reviewer":"human"}],
            "ownerships":[{"source_id":i,"bubble_id":"bubble-a","reviewed":True,"reviewer":"human"} for i in (7,8)],
            "units":[{"main_ids":[7],"direction":"vertical","granularity":"column","reviewed":True,
                      "order_reviewed":True,"granularity_reviewed":True,"reviewer":"human"}]}]}
        self.review.write_text(json.dumps(self.spec))

    def test_optional_api_invoked_via_saved_page_fixture_callback_only(self):
        before = self.comparison.read_bytes()
        callback = Mock(return_value="漢字")
        with patch.object(socket.socket,"connect",side_effect=AssertionError("network forbidden")):
            result = d.run_unit_experiment([self.comparison],self.root/"units",image_root=self.root,
                reviewed_input=self.review,run_model=True,recognizer=callback,
                recognizer_identity="offline-fixture-only",max_attempts=1)
        self.assertEqual(callback.call_count,1)
        self.assertEqual(result["live_model_calls"],0)
        self.assertEqual(result["settings"]["recognizer_kind"],"test_only")
        self.assertEqual(result["units"][0]["unit"]["plan"]["annotation_parents"],((8,7),))
        self.assertEqual(self.comparison.read_bytes(),before)
        self.assertTrue((self.root/"units"/"units.json").exists())
        self.assertTrue(all(Path(p).is_file() for p in result["review_sheets"]))

    def test_preflight_exports_hash_bound_review_template_without_models(self):
        with patch.object(d.importlib,"import_module",side_effect=AssertionError("model import forbidden")):
            result = d.run_unit_experiment([self.comparison],self.root/"preflight",image_root=self.root)
        self.assertEqual(result["callback_attempts"],0)
        template = json.loads((self.root/"preflight"/"review-template.json").read_text())
        self.assertEqual(template["pages"][0]["relationships"],[])
        self.assertEqual(template["pages"][0]["page"],self.spec["pages"][0]["page"])

    def test_unit_cli_planning_is_model_free_and_global_attempt_cap_includes_alternate(self):
        with patch.object(d.importlib,"import_module",side_effect=AssertionError("model import forbidden")):
            d.main(["--unit-experiment","--comparison",str(self.comparison),"--image-root",str(self.root),
                    "--unit-review",str(self.review),"--output-dir",str(self.root/"cli-units")])
        result=json.loads((self.root/"cli-units"/"units.json").read_text())
        self.assertEqual(result["live_model_calls"],0)
        callback=Mock(return_value="different")
        result=d.run_unit_experiment([self.comparison],self.root/"capped",image_root=self.root,
            reviewed_input=self.review,run_model=True,recognizer=callback,
            recognizer_identity="test-only",max_attempts=1,alternate=True)
        self.assertEqual(callback.call_count,1)
        self.assertEqual(result["callback_attempts"],1)

    def test_crop_blocked_live_opt_in_still_does_not_import_optional_model(self):
        self.spec["pages"][0]["ownerships"]=[]
        self.review.write_text(json.dumps(self.spec))
        with patch.object(d.importlib,"import_module",side_effect=AssertionError("no validated crop")):
            result=d.run_unit_experiment([self.comparison],self.root/"blocked-live",image_root=self.root,
                reviewed_input=self.review,run_model=True,model_path="not-opened")
        self.assertEqual(result["live_model_calls"],0)

    def test_advisory_outputs_and_foreign_page_reviews_do_not_promote_units(self):
        self.spec["pages"][0]["relationships"][0].pop("reviewed")
        self.review.write_text(json.dumps(self.spec))
        callback=Mock(return_value="ignored")
        result=d.run_unit_experiment([self.comparison],self.root/"unreviewed",image_root=self.root,
            reviewed_input=self.review,run_model=True,recognizer=callback,recognizer_identity="test-only")
        callback.assert_not_called()
        self.assertIn("unreviewed_relationship",result["units"][0]["unit"]["reason"])
        self.spec["pages"][0]["page"]["image_sha256"]="0"*64
        self.review.write_text(json.dumps(self.spec))
        result=d.run_unit_experiment([self.comparison],self.root/"wrong-page",image_root=self.root,
            reviewed_input=self.review)
        self.assertEqual(len(result["page_errors"]),1)

    @local_evidence(comparisons=("test-results/crop-isolation-v2/lmgJZ-comparison.json",))
    def test_real_parent_saved_page_preflight_uses_external_associations_only(self):
        comparison=Path(__file__).resolve().parents[2]/"test-results/crop-isolation-v2/lmgJZ-comparison.json"
        root=Path(__file__).resolve().parents[2]
        _,_,page,raw=d.load_unit_page(comparison,root)
        # Parent-reviewed associations are EXTERNAL test evidence, not production rules.
        pairs=[(8,7),(12,11),(18,17),(21,14),(22,19)]
        relations=[u.ReviewedRelationship(a,b,"furigana",page,True,"parent-reviewed evidence") for a,b in pairs]
        relations.append(u.ReviewedRelationship(16,13,"same_text_block",page,True,"parent-reviewed evidence"))
        requests=[u.UnitRequest((b,),"vertical","column",True,True,False,"parent") for _,b in pairs]
        requests.append(u.UnitRequest((13,16),"vertical","block",True,True,False,"parent"))
        batch=u.plan_recognition_units(raw,relations,requests,[])
        self.assertEqual(batch.units[0].plan.annotation_parents,((8,7),))
        self.assertTrue(all(not unit.approved for unit in batch.units))
        self.assertTrue(all("mixed_or_unknown_bubble_ownership" in unit.plan.errors for unit in batch.units))
        self.assertEqual(len(batch.fallback_source_ids),len(raw))
        spec={"pages":[{"page":asdict(page),"relationships":[asdict(r) for r in relations],
                        "ownerships":[],"units":[asdict(r) for r in requests]}]}
        reviewed=self.root/"parent-evidence-reviewed.json"
        reviewed.write_text(json.dumps(spec))
        before=comparison.read_bytes()
        result=d.run_unit_experiment([comparison],self.root/"parent-saved-experiment",image_root=root,
                                     reviewed_input=reviewed)
        self.assertEqual(len(result["units"]),6)
        self.assertEqual(result["callback_attempts"],0)
        self.assertEqual(result["live_model_calls"],0)
        self.assertEqual(comparison.read_bytes(),before)


class UtteranceExtentTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), "white")
        # Synthetic trailing punctuation outside every detection footprint.
        for y in range(61, 76):
            self.image.putpixel((35, y), (3, 7, 11))
        self.image.putpixel((35, 79), (13, 17, 19))
        self.page = u.PageProvenance("extent-fixture", "/fixture/original.png", "1"*64, "2"*64,
                                     u.image_fingerprint(self.image))
        self.boxes = [saved("main", (30, 10, 40, 60), "word"),
                      saved("annotation", (40, 12, 45, 45), "reading")]
        self.raw = u.snapshot_detections(self.boxes, self.page)
        self.relations = [u.ReviewedRelationship("annotation", "main", "furigana", self.page, True, "human")]
        self.owners = [u.ReviewedOwnership(i, "bubble-a", self.page, True, "human")
                       for i in ("main", "annotation")]
        self.request = u.UnitRequest(("main",), "vertical", "column", True, True, True, "human")
        legacy = self.batch()
        self.extent = u.ReviewedUtteranceExtent(legacy.units[0].plan.unit_id,
            legacy.units[0].plan.member_ids, self.page, (28, 8, 47, 82),
            "human original-page punctuation review", True, "human")

    def batch(self, extents=(), raw=None, owners=None):
        return u.plan_recognition_units(self.raw if raw is None else raw, self.relations,
            [self.request], self.owners if owners is None else owners, utterance_extents=extents)

    def test_complete_punctuation_original_pixels_and_unchanged_legacy_crop(self):
        before = copy.deepcopy(self.boxes)
        legacy = self.batch()
        old, provenance, error = u.prepare_unit_crop(legacy.units[0], legacy, self.image)
        self.assertIsNone(error)
        self.assertEqual(provenance.bounds, (28, 8, 47, 62))
        self.assertEqual(old.tobytes(), self.image.crop((28, 8, 47, 62)).tobytes())
        batch = self.batch([self.extent])
        crop, provenance, error = u.prepare_unit_crop(batch.units[0], batch, self.image, padding=4)
        self.assertIsNone(error)
        self.assertEqual(provenance.bounds, self.extent.bounds)
        self.assertEqual(provenance.detection_union_bounds, (30, 10, 45, 60))
        self.assertEqual(provenance.reviewed_extent, self.extent)
        self.assertEqual(provenance.padding, 0)
        self.assertEqual(crop.tobytes(), self.image.crop(self.extent.bounds).tobytes())
        self.assertEqual(crop.getpixel((7, 71)), (13, 17, 19))
        self.assertEqual(self.boxes, before)
        self.assertEqual(batch.detections, self.raw)
        self.assertEqual(batch.fallback_source_ids, ("main", "annotation"))
        with self.assertRaises(FrozenInstanceError):
            self.extent.bounds = (0, 0, 1, 1)

    def test_extent_review_page_unit_and_member_bindings_fail_closed(self):
        invalid = [replace(self.extent, reviewed=False), replace(self.extent, reviewer=" "),
                   replace(self.extent, source=""), replace(self.extent, unit_id="foreign"),
                   replace(self.extent, member_ids=("main",)),
                   replace(self.extent, member_ids=tuple(reversed(self.extent.member_ids)))]
        for field in ("page_id", "image_path", "image_sha256", "comparison_sha256", "pixels_sha256"):
            value = "3"*64 if field.endswith("sha256") else "foreign"
            invalid.append(replace(self.extent, page=replace(self.page, **{field: value})))
        callback = Mock(return_value="word")
        for extent in invalid:
            with self.subTest(extent=extent):
                batch = self.batch([extent])
                self.assertFalse(batch.units[0].approved)
                u.recognize_unit(batch.units[0], batch, self.image, callback)
                self.assertEqual(batch.units[0].attempts, 0)
        callback.assert_not_called()
        duplicate = self.batch([self.extent, self.extent])
        self.assertIn("ambiguous_utterance_extent", duplicate.units[0].plan.errors)

    def test_invalid_outside_fractional_and_partial_extents_never_recognize(self):
        cases = [(0, 0, 0, 90), (48, 8, 28, 82), (-1, 8, 47, 82), (28, 8, 101, 82),
                 (28, 8, 47, 101), (28, float("nan"), 47, 82), (28, 8, float("inf"), 82),
                 (True, 8, 47, 82), (28.5, 8, 47, 82), (28, 8, 47), (28, 8, "47", 82),
                 (31, 8, 47, 82), (28, 13, 47, 82), (28, 8, 44, 82), (28, 8, 47, 59)]
        for bounds in cases:
            with self.subTest(bounds=bounds):
                batch = self.batch([replace(self.extent, bounds=bounds)])
                callback = Mock()
                u.recognize_unit(batch.units[0], batch, self.image, callback)
                callback.assert_not_called()
                self.assertEqual(batch.units[0].attempts, 0)
                self.assertEqual(batch.units[0].crop.reviewed_extent.bounds, bounds)

    def test_adjacent_cross_balloon_and_rotated_crop_contamination_not_trimmed(self):
        polygons = [((45, 65), (49, 65), (49, 75), (45, 75)),
                    ((35, 65), (42, 70), (35, 75), (28, 70))]
        for polygon in polygons:
            for bubble in (None, "bubble-a", "bubble-b"):
                with self.subTest(polygon=polygon, bubble=bubble):
                    neighbor = replace(self.raw[0], source_id="neighbor", polygon=polygon)
                    owners = self.owners + ([u.ReviewedOwnership("neighbor", bubble, self.page, True, "human")]
                                            if bubble else [])
                    batch = self.batch([self.extent], self.raw + (neighbor,), owners)
                    callback = Mock()
                    u.recognize_unit(batch.units[0], batch, self.image, callback)
                    callback.assert_not_called()
                    self.assertEqual(batch.units[0].crop.bounds, self.extent.bounds)
                    self.assertIn(batch.units[0].reason, ("unknown_bubble_ownership",
                                  "mixed_bubble_ownership", "unrelated_neighbor_glyphs"))

    def test_rotated_bbox_overlap_alone_does_not_block_safe_extent(self):
        # Bounding box overlaps the rectangle, but the diagonal polygon does not.
        neighbor = replace(self.raw[0], source_id="neighbor",
                           polygon=((46, 90), (55, 80), (60, 85), (51, 95)))
        batch = self.batch([self.extent], self.raw + (neighbor,))
        crop, provenance, error = u.prepare_unit_crop(batch.units[0], batch, self.image)
        self.assertIsNone(error)
        self.assertEqual(crop.tobytes(), self.image.crop(self.extent.bounds).tobytes())

    def test_mixed_member_ownership_still_blocks_extent(self):
        batch = self.batch([self.extent], owners=[self.owners[0], replace(self.owners[1], bubble_id="other")])
        callback = Mock()
        u.recognize_unit(batch.units[0], batch, self.image, callback)
        callback.assert_not_called()
        self.assertIn("mixed_or_unknown_bubble_ownership", batch.units[0].plan.errors)

    def test_extent_still_requires_original_pixels_and_valid_member_polygons(self):
        batch = self.batch([self.extent])
        changed = self.image.copy()
        changed.putpixel((35, 79), (0, 0, 0))
        callback = Mock()
        u.recognize_unit(batch.units[0], batch, changed, callback)
        self.assertEqual(batch.units[0].reason, "original_image_provenance_mismatch")
        malformed = replace(self.raw[0], polygon=((30, 10), (40, 60), (30, 60), (40, 10)))
        invalid = self.batch([self.extent], raw=(malformed, self.raw[1]))
        u.recognize_unit(invalid.units[0], invalid, self.image, callback)
        self.assertEqual(invalid.units[0].reason, "invalid_member_polygon")
        callback.assert_not_called()

    def test_full_utterance_agreement_cannot_accept_and_differences_retained(self):
        for text in ("word", "word!!"):
            batch = self.batch([self.extent])
            unit = batch.units[0]
            callback = Mock(return_value=text)
            u.recognize_unit(unit, batch, self.image, callback)
            self.assertEqual(unit.status, u.UnitStatus.REVIEW)
            self.assertEqual(unit.reason, "full_utterance_reference_completeness_unverified")
            self.assertEqual(unit.candidates[0].reference_reading, "word")
            self.assertEqual(unit.candidates[0].text, text)
            self.assertEqual(bool(unit.candidates[0].differing_spans), text != "word")
            self.assertEqual(unit.crop.comparison_granularity, "full_utterance_without_complete_reference")

    def test_bounded_alternate_preserves_extent_punctuation_and_raw_evidence(self):
        batch = self.batch([self.extent])
        unit = batch.units[0]
        before = asdict(batch.detections[0])
        callback = Mock(side_effect=["word!!", "word!"])
        u.recognize_unit(unit, batch, self.image, callback)
        u.recognize_unit(unit, batch, self.image, callback)
        u.recognize_unit(unit, batch, self.image, callback, alternate=True)
        u.recognize_unit(unit, batch, self.image, callback, alternate=True)
        self.assertEqual(callback.call_count, 2)
        self.assertEqual((unit.attempts, unit.alternate_attempts), (2, 1))
        self.assertEqual(unit.reason, "alternate_unit_candidates_disagree")
        for call in callback.call_args_list:
            self.assertEqual(call.kwargs["crop_bounds"], self.extent.bounds)
            self.assertEqual(call.args[0].tobytes(), self.image.crop(self.extent.bounds).tobytes())
        self.assertEqual(asdict(batch.detections[0]), before)

    def test_alternate_cannot_remove_or_change_prior_reviewed_extent(self):
        for extents in ((), (replace(self.extent, bounds=(28, 8, 47, 80)),)):
            batch = self.batch([self.extent])
            unit = batch.units[0]
            callback = Mock(return_value="word")
            u.recognize_unit(unit, batch, self.image, callback)
            changed = replace(batch, utterance_extents=extents)
            u.recognize_unit(unit, changed, self.image, callback, alternate=True)
            self.assertEqual(callback.call_count, 1)
            self.assertEqual(unit.reason, "alternate_utterance_extent_changed")
            self.assertEqual(unit.crop.bounds, self.extent.bounds)

    def test_diagnostic_schema_and_artifacts_preserve_reviewed_original_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "original.png"
            self.image.save(image_path)
            comparison = root / "fixture-comparison.json"
            comparison.write_text(json.dumps({"input": str(image_path), "boxes": self.boxes}))
            _, image, page, raw = d.load_unit_page(comparison)
            specification = {"page": asdict(page), "relationships": [asdict(replace(r, page=page)) for r in self.relations],
                             "ownerships": [asdict(replace(o, page=page)) for o in self.owners],
                             "units": [asdict(self.request)]}
            batch = d._reviewed_unit_batch(raw, page, specification)
            extent = replace(self.extent, page=page, unit_id=batch.units[0].plan.unit_id)
            specification["utterance_extents"] = [asdict(extent)]
            review = root / "review.json"
            review.write_text(json.dumps({"pages": [specification]}))
            callback = Mock(return_value="word!!")
            result = d.run_unit_experiment([comparison], root / "output", reviewed_input=review,
                run_model=True, recognizer=callback, recognizer_identity="test-only",
                alternate=True, max_attempts=1)
            self.assertEqual(result["callback_attempts"], 1)
            self.assertEqual(result["live_model_calls"], 0)
            record = result["units"][0]
            self.assertEqual(record["unit"]["crop"]["reviewed_extent"], asdict(extent))
            self.assertEqual(len(record["image_paths"]), 2)
            with Image.open(record["image_paths"][0]) as original:
                self.assertEqual(original.tobytes(), image.crop(extent.bounds).tobytes())
            self.assertEqual(record["requested_utterance_extents"], [asdict(extent)])
            # Stale evidence is unresolved, without any fallback/model attempt.
            specification["utterance_extents"][0]["page"]["image_sha256"] = "9"*64
            review.write_text(json.dumps({"pages": [specification]}))
            blocked = d.run_unit_experiment([comparison], root / "blocked", reviewed_input=review,
                run_model=True, recognizer=callback, recognizer_identity="test-only")
            self.assertEqual(blocked["callback_attempts"], 0)
            self.assertEqual(blocked["units"][0]["image_paths"], [])


if __name__ == "__main__":
    unittest.main()
