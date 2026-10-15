"""Automatic structure fixtures only; no engine initialization or human approval."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from tralens.ocr import automatic_units as a
from tralens.ocr import detection
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tralens.ocr.types import Text_Box, MediaType
from tools import diagnostics as d
from tests.local_evidence import local_evidence
from tralens.ocr.grouping import group_manga_boxes


def box(source_id, bounds, text="字", score=.3):
    x1, y1, x2, y2 = bounds
    return {"id": source_id, "polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
            "paddle": text, "selected": text, "score": score, "status": "review"}


class AutomaticUnitTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (160, 140), "white")

    def propose(self, boxes, image=None):
        image = self.image if image is None else image
        page = PageProvenance("fixture", "/original.png", "1"*64, "2"*64, image_fingerprint(image))
        return a.propose_automatic_units(snapshot_detections(boxes, page), image)

    def assert_partition(self, batch):
        mains = [i for p in batch.proposals for i in p.main_ids]
        annotations = [i for p in batch.proposals for i in p.annotation_ids]
        unassigned = [i for i, _ in batch.unassigned]
        ids = mains + annotations + unassigned
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {r.source_id for r in batch.detections})
        parents = [child for p in batch.proposals for child, _ in p.annotation_parents]
        self.assertEqual(len(parents), len(set(parents)))

    def test_vertical_order_right_to_left_with_tentative_annotation(self):
        image = self.image.copy()
        ImageDraw.Draw(image).ellipse((3, 5, 100, 118), outline="black", width=3)
        batch = self.propose([box("left", (20,30,40,95), "左"), box("right", (45,30,65,95), "漢字"),
                              box("reading", (66,35,74,70), "かんじ")], image)
        self.assertEqual(len(batch.proposals), 1)
        unit = batch.proposals[0]
        self.assertEqual(unit.main_ids, ("right", "left"))
        self.assertEqual(unit.annotation_parents, (("reading", "right"),))
        self.assertEqual(unit.direction, "vertical")
        self.assertFalse(unit.reviewed)
        self.assertEqual(unit.completeness, "unverified_full_utterance")
        self.assertTrue(batch.regions)
        self.assert_partition(batch)

    def test_horizontal_row_fragment_order_and_above_annotation(self):
        batch = self.propose([box("second", (10,50,70,62), "二行"), box("first", (10,30,70,42), "漢字"),
                              box("reading", (15,23,40,28), "かんじ")])
        self.assertEqual(batch.proposals[0].main_ids, ("first", "second"))
        self.assertEqual(batch.proposals[0].annotation_parents, (("reading", "first"),))
        self.assertEqual(batch.proposals[0].direction, "horizontal")
        fragments = self.propose([box("b", (40,20,65,30), "後"), box("a", (10,20,35,30), "前")])
        self.assertEqual(fragments.proposals[0].main_ids, ("a", "b"))
        self.assert_partition(batch)

    def test_adjacent_distinct_original_bubbles_never_merge_by_distance(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        draw.ellipse((20,5,55,100), outline="black", width=2)
        draw.ellipse((60,5,95,100), outline="black", width=2)
        batch = self.propose([box(1,(40,25,50,80)), box(2,(65,25,75,80))], image)
        self.assertEqual(len(batch.regions), 2)
        self.assertEqual(len({r[1] for r in batch.regions}), 2)
        self.assertEqual(len(batch.proposals), 2)
        self.assert_partition(batch)

    def test_small_dialogue_sfx_and_dash_only_are_not_size_based_annotations(self):
        batch = self.propose([box("main",(30,15,50,75),"漢字"),
                              box("dialogue",(20,20,26,45),"はい"),
                              box("sfx",(90,80,100,90),"ドン"),
                              box("dash",(51,20,57,50),"ーー")])
        self.assertEqual(batch.coverage["tentative_annotations"], 0)
        self.assertEqual(batch.coverage["main_detections"], 4)
        self.assertTrue(any("missing_kana_and_main_kanji_corroboration" in r.reasons for r in batch.relationships))
        self.assert_partition(batch)

    def test_annotation_placement_cannot_override_distinct_region_evidence(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((15,5,49,105), outline="black", width=1)
        draw.rectangle((50,20,70,80), outline="black", width=1)
        batch = self.propose([box("main",(30,25,45,85),"漢字"),
                              box("reading",(53,30,60,65),"かんじ")], image)
        self.assertEqual(len({r[1] for r in batch.regions}),2)
        self.assertEqual(batch.coverage["tentative_annotations"],0)
        self.assertEqual(batch.coverage["main_detections"],2)
        self.assertTrue(any("annotation_geometry_crosses_distinct_regions" in r.reasons for r in batch.relationships))
        self.assert_partition(batch)

    def test_ambiguous_parents_retained_as_main_without_duplicate_assignment(self):
        batch = self.propose([box("p1",(30,10,50,70),"漢字"), box("p2",(31,12,51,72),"文字"),
                              box("child",(52,20,60,50),"かんじ")])
        relation = next(r for r in batch.relationships if r.source_id == "child")
        self.assertIsNone(relation.parent_id)
        self.assertEqual(relation.possible_parent_ids, ("p1", "p2"))
        self.assertIn("ambiguous_annotation_parents", relation.reasons)
        self.assertEqual(batch.coverage["main_detections"], 3)
        self.assert_partition(batch)

    def test_distant_columns_and_complete_link_anti_bridging(self):
        distant = self.propose([box(1,(10,10,20,90)), box(2,(100,10,110,90))])
        self.assertEqual(len(distant.proposals), 2)
        chain = self.propose([box(i,(5+20*i,10,15+20*i,90)) for i in range(7)])
        self.assertGreater(len(chain.proposals), 1)
        self.assert_partition(chain)

    def test_blank_noisy_readings_and_confidence_do_not_decide_structure(self):
        boxes = [box(1,(20,20,30,85),"",0), box(2,(40,20,50,85),"!?",float("nan"))]
        batch = self.propose(boxes)
        self.assertEqual(batch.coverage["main_detections"], 2)
        self.assertEqual(len(batch.proposals), 1)
        self.assertIn("blank_main_reading", batch.proposals[0].word_reasons)
        other = self.propose([dict(b, score=.99, status="accepted") for b in boxes])
        self.assertEqual([p.proposal_id for p in batch.proposals], [p.proposal_id for p in other.proposals])
        self.assertEqual([p.structural_reasons for p in batch.proposals], [p.structural_reasons for p in other.proposals])
        self.assert_partition(batch)

    def test_invalid_and_clipped_polygons_explicitly_unassigned_and_crop_uncertain(self):
        boxes = [box("valid",(20,20,30,80)), box("clipped",(-5,20,5,60)),
                 dict(box("invalid",(40,20,50,80)), polygon=[[40,20],[50,80],[40,80],[50,20]])]
        batch = self.propose(boxes)
        self.assertEqual(dict(batch.unassigned), {"clipped": "clipped_polygon", "invalid": "invalid_polygon"})
        self.assertFalse(batch.proposals[0].crop_safe)
        self.assertIn("invalid_nonmember_geometry", batch.proposals[0].crop_reasons)
        self.assert_partition(batch)

    def test_rotated_nonmember_intersecting_actual_union_blocks_crop(self):
        boxes = [box("main",(30,10,40,70),"漢字"), box("reading",(41,15,45,45),"かんじ"),
                 dict(box("neighbor",(39,69,47,77)), polygon=[[43,69],[47,73],[43,77],[39,73]])]
        batch = self.propose(boxes)
        unit = next(p for p in batch.proposals if "main" in p.main_ids)
        self.assertFalse(unit.crop_safe)
        self.assertIn("nonmember_polygon_intersects_crop", unit.crop_reasons)
        self.assertIn("punctuation_outside_detections_not_ruled_out", unit.crop_reasons)

    def test_deterministic_ids_order_nonmutation_and_no_review_promotion(self):
        boxes = [box(1,(20,20,30,85),"漢字"), box("1",(40,20,50,85),"文字"),
                 box("r",(51,25,55,55),"もじ")]
        before = copy.deepcopy(boxes)
        a1, a2 = self.propose(boxes), self.propose(list(reversed(boxes)))
        self.assertEqual(a1.proposals, a2.proposals)
        self.assertEqual(a1.relationships, a2.relationships)
        self.assertEqual(boxes, before)
        self.assertFalse(a1.reviewed)
        self.assertTrue(all(not p.reviewed for p in a1.proposals))
        self.assertFalse(any(hasattr(raw, "unit_candidate") for raw in a1.detections))
        self.assert_partition(a1)

    def test_bounds_duplicate_ids_and_image_provenance_fail_without_truncation(self):
        batch = self.propose([box(1,(20,20,30,80)), box(2,(40,20,50,80))])
        with self.assertRaises(ValueError):
            a.propose_automatic_units(batch.detections, self.image, max_detections=1)
        with self.assertRaises(ValueError):
            a.propose_automatic_units(batch.detections + (batch.detections[0],), self.image)
        with self.assertRaises(ValueError):
            a.propose_automatic_units(batch.detections, Image.new("RGB", self.image.size, "black"))

    def test_opt_in_ocr_workflow_preserves_groups_boxes_and_default_return_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"original.png"
            self.image.save(path)
            boxes = [Text_Box(1,.99,[(20,20),(30,20),(30,80),(20,80)],"漢字"),
                     Text_Box(2,.99,[(31,25),(35,25),(35,55),(31,55)],"かんじ")]
            with patch.object(detection, "create_paddle_ocr", side_effect=AssertionError("engine forbidden")):
                ocr = detection.OCR.__new__(detection.OCR)
                ocr.image_path = str(path)
                ocr.predict = Mock(return_value=[boxes])
                legacy = ocr.process_images()
                self.assertFalse(hasattr(ocr, "automatic_unit_proposals"))
                before = copy.deepcopy([(b.paddle_text, b.paddle_poly.tolist(), b.text, b.status) for b in boxes])
                opted = ocr.process_images(propose_units=True)
                self.assertEqual([[b.box_id for b in g.group] for g in opted[0]],
                                 [[b.box_id for b in g.group] for g in legacy[0]])
                self.assertEqual(before, [(b.paddle_text, b.paddle_poly.tolist(), b.text, b.status) for b in boxes])
                self.assertEqual(ocr.automatic_unit_proposals[0].coverage["tentative_annotations"], 1)
                ocr.predict.reset_mock()
                with self.assertRaises(ValueError):
                    ocr.process_images(propose_units=True, media_type=MediaType.OTHER)
                ocr.predict.assert_not_called()

    def test_short_overlapping_annotation_with_blank_ascii_or_kana_reading_uses_original_glyphs(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        for y in (15,35,55,75):
            draw.rectangle((32,y,43,y+11),fill="black")
        for y in (20,25):
            draw.rectangle((50,y,52,y+2),fill="black")
        before = image.tobytes()
        for reading in ("", "a", "かな"):
            with self.subTest(reading=reading):
                batch = self.propose([box("main",(30,10,54,100),"漢字"),
                                      box("short",(49,19,57,29),reading)],image)
                self.assertEqual(batch.proposals[0].annotation_parents,(("short","main"),))
                relation = next(r for r in batch.relationships if r.parent_id == "main")
                self.assertIn("original_small_glyph_scale_and_side_placement",relation.reasons)
                self.assertTrue(relation.image_evidence)
                if reading != "かな":
                    self.assertIn("annotation_wording_corroboration_missing",batch.proposals[0].word_reasons)
                self.assertFalse(batch.reviewed)
                self.assert_partition(batch)
        self.assertEqual(image.tobytes(),before)

    def test_horizontal_short_annotation_overlapping_parent_footprint(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        for x in (20,40,60):
            draw.rectangle((x,40,x+9,49),fill="black")
        for x in (22,27):
            draw.rectangle((x,30,x+2,32),fill="black")
        batch = self.propose([box("main",(15,32,80,54),"字"),box("reading",(21,28,31,36),"?")],image)
        self.assertEqual(batch.proposals[0].annotation_parents,(("reading","main"),))
        self.assert_partition(batch)

    def test_small_dialogue_same_original_glyph_scale_is_not_annotation_even_with_kana(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        for y in (15,35,55,75):
            draw.rectangle((32,y,39,y+9),fill="black")
        for y in (20,35):
            draw.rectangle((50,y,55,y+9),fill="black")
        batch = self.propose([box("main",(30,10,54,100),"漢字"),box("dialogue",(49,19,57,46),"はい")],image)
        self.assertEqual(batch.coverage["tentative_annotations"],0)
        self.assertTrue(any("original_glyph_scale_not_smaller_than_main" in r.reasons for r in batch.relationships))
        self.assert_partition(batch)

    def test_original_small_glyphs_do_not_resolve_two_plausible_parents(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        for y in (15,35,55,75):
            draw.rectangle((32,y,43,y+11),fill="black")
        for y in (20,25):
            draw.rectangle((50,y,52,y+2),fill="black")
        batch = self.propose([box("p1",(30,10,54,100),"字"),box("p2",(31,11,55,101),"文字"),
                              box("reading",(49,19,57,29),"")],image)
        self.assertEqual(batch.coverage["tentative_annotations"],0)
        self.assertTrue(any("ambiguous_annotation_parents" in r.reasons for r in batch.relationships))
        self.assert_partition(batch)

    def test_connected_white_balloon_interiors_do_not_override_original_divider(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((5,5,150,135),outline="black",width=2)
        draw.line((65,7,65,95),fill="black",width=2)  # White connector below divider.
        for x in (38,78):
            for y in (30,50,70):
                draw.rectangle((x,y,x+9,y+9),fill="black")
        batch = self.propose([box("left",(35,25,55,85)),box("right",(75,25,95,85))],image)
        self.assertEqual(len({region[1] for region in batch.regions}),1)
        self.assertEqual(len(batch.proposals),2)
        self.assertTrue(any("original_image_boundary_blocks_column_merge" in r["reasons"] for r in batch.rejected_merges))
        # Legacy helper, without the opt-in guard, retains its previous behavior.
        from types import SimpleNamespace
        plain = [SimpleNamespace(box_id=i,bbox=bounds) for i,bounds in
                 (("left",(35,25,55,85)),("right",(75,25,95,85)))]
        self.assertEqual(len(group_manga_boxes(plain,regions={0:(1,1.),1:(1,1.)})),1)
        self.assert_partition(batch)

    def test_original_panel_boundary_blocks_horizontal_row_merge(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        draw.line((5,49,120,49),fill="black",width=2)
        for x in (15,35,55,75):
            for y in (32,57):
                draw.rectangle((x,y,x+7,y+7),fill="black")
        batch = self.propose([box("above",(10,30,90,42)),box("below",(10,55,90,67))],image)
        self.assertEqual(len(batch.proposals),2)
        self.assertTrue(batch.rejected_merges)
        self.assert_partition(batch)

    def test_blank_noisy_box_cannot_bridge_other_mains_and_cannot_erase_boundary(self):
        for reading in ("", "ASCII!?", "ーー"):
            with self.subTest(reading=reading):
                batch = self.propose([box("left",(10,20,30,100)),box("bridge",(40,20,60,100),reading),
                                      box("right",(70,20,90,100))])
                self.assertFalse(any({"left","right"} <= set(p.main_ids) for p in batch.proposals))
                self.assertTrue(any("unsupported_blank_or_noisy_main_cannot_bridge_group" in r["reasons"]
                                    for r in batch.rejected_merges))
                self.assert_partition(batch)
        image = self.image.copy()
        ImageDraw.Draw(image).line((65,10,65,110),fill="black",width=2)
        batch = self.propose([box("left",(35,20,55,100)),box("bridge",(60,20,80,100),""),
                              box("right",(95,20,115,100))],image)
        self.assertFalse(any({"left","right"} <= set(p.main_ids) for p in batch.proposals))
        self.assert_partition(batch)

    @local_evidence(comparisons=("test-results/crop-isolation-v2/lmgJZ-comparison.json",))
    def test_external_saved_page_pairs_are_evaluation_only_and_remain_candidate_relations(self):
        root = Path(__file__).resolve().parents[2]
        comparison = root/"test-results/crop-isolation-v2/lmgJZ-comparison.json"
        _,image,_,raw = d.load_unit_page(comparison,root)
        before = comparison.read_bytes()
        batch = a.propose_automatic_units(raw,image)
        # External parent associations are evaluation evidence only, never an API input.
        expected = {(8,7),(12,11),(18,17),(21,14),(22,19)}
        candidates = {(r.source_id,parent) for r in batch.relationships for parent in r.possible_parent_ids}
        self.assertTrue(expected <= candidates,expected-candidates)
        self.assertEqual(len(raw),47)
        self.assertEqual(comparison.read_bytes(),before)
        self.assert_partition(batch)
        self.assertFalse(any(set(p.main_ids)&{10,11} and set(p.main_ids)&{14,15} for p in batch.proposals),
                         [(p.main_ids,p.annotation_ids) for p in batch.proposals])


class AutomaticProposalDiagnosticsTests(unittest.TestCase):
    def test_saved_page_cli_original_pixels_coverage_and_new_only_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB",(100,100),"white")
            image.putpixel((25,40),(3,7,11))
            original = root/"original.png"
            image.save(original)
            comparison = root/"page-comparison.json"
            comparison.write_text(json.dumps({"input":str(original), "boxes":[
                box("main",(20,20,30,80),"漢字"), box("r",(31,25,35,55),"かんじ")]}))
            before = comparison.read_bytes()
            with patch.object(detection, "create_paddle_ocr", side_effect=AssertionError("engine forbidden")), \
                 patch.object(d.importlib, "import_module", side_effect=AssertionError("model import forbidden")):
                report = d.run_automatic_proposals([comparison],root/"output")
                self.assertEqual(report["completed_pages"],1)
                self.assertEqual(report["recognition_calls"],0)
                self.assertEqual(report["coverage"]["raw_detections"],2)
                self.assertEqual(report["coverage"]["tentative_annotations"],1)
                entry = report["pages"][0]["units"][0]
                with Image.open(entry["image_paths"][0]) as crop:
                    self.assertEqual(crop.tobytes(),image.crop(entry["proposal"]["crop_bounds"]).tobytes())
                self.assertEqual(d.main(["--automatic-proposals","--comparison",str(comparison),
                                        "--output-dir",str(root/"cli")]),0)
            self.assertEqual(comparison.read_bytes(),before)
            with self.assertRaises(FileExistsError):
                d.run_automatic_proposals([comparison],root/"output")
            with self.assertRaises(ValueError):
                d.run_automatic_proposals([comparison,comparison],root/"duplicates")
            over = d.run_automatic_proposals([comparison],root/"over",max_detections=1)
            self.assertEqual(over["completed_pages"],0)
            self.assertEqual(len(over["page_errors"]),1)
