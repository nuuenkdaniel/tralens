"""Guarded original-ink singleton companions; no engines or reviewed-text claims."""
import copy
import json
import unittest
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tralens.ocr import automatic_units as a
from tralens.ocr.automatic_recognition import prepare_automatic_crop
from tralens.ocr.automatic_evidence import OriginalInkEvidence
from tralens.ocr.grouping import analyze_manga_regions
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tralens.ocr.types import Text_Group
from tools.diagnostics import load_unit_page
from tests.local_evidence import local_evidence


def box(identifier,bounds):
    x,y,r,s = bounds
    return SimpleNamespace(box_id=identifier,bbox=bounds,text="漢字",confidence=.99,
        poly=np.array([(x,y),(r,y),(r,s),(x,s)],dtype=np.float64))


class SingletonCompanionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((200,260,3),255,dtype=np.uint8)
        cv2.rectangle(self.image,(15,15),(245,185),(0,0,0),2)
        self.boxes = [box("left",(36,30,68,150)),box("center",(68,30,104,150)),
                      box("single",(100,30,132,150)),box("host-reading",(90,38,102,64)),
                      box("single-reading",(122,38,138,64))]
        for x in (44,76,108):
            self.draw_main(x)
        for x in (96,128):
            self.draw_reading(x)
        self.groups = [Text_Group([self.boxes[2]],direction="vertical"),
                       Text_Group([self.boxes[1],self.boxes[0]],direction="vertical")]
        self.annotations = {"host-reading":"center","single-reading":"single"}

    def draw_main(self,x,size=14):
        for y in (40,65,90,115):
            cv2.rectangle(self.image,(x,y),(x+size-1,y+size-1),(0,0,0),-1)

    def draw_reading(self,x):
        for y in (44,54):
            cv2.rectangle(self.image,(x,y),(x+3,y+3),(0,0,0),-1)

    def regroup(self):
        image = Image.fromarray(self.image)
        page = PageProvenance("fixture","/original.png","1"*64,"2"*64,image_fingerprint(image))
        raw = snapshot_detections(self.boxes,page)
        before_raw = copy.deepcopy(raw)
        by_id = {b.box_id:(d,b,i) for i,(b,d) in enumerate(zip(self.boxes,raw))}
        regions,_ = analyze_manga_regions(self.boxes,self.image)
        previous_regions = dict(regions)
        ink = OriginalInkEvidence(self.image)
        glyphs = {b.box_id:ink.glyphs(b) for b in self.boxes}
        relationships = tuple(a.RelationshipProposal(child,parent,(parent,),"tentative_furigana",
            ("original_small_glyph_scale_and_side_placement",),"geometry_only") for child,parent in self.annotations.items())
        pixels = self.image.tobytes()
        groups,accepted,rejected = a._regroup_companions(self.groups,regions,by_id,glyphs,ink,
            self.annotations,raw,relationships,ink.single_glyph_context())
        self.assertEqual(self.image.tobytes(),pixels)
        self.assertEqual(raw,before_raw)
        return groups,accepted,rejected,regions,previous_regions

    def test_reciprocal_singleton_with_visible_readings_can_form_one_isolated_unit(self):
        groups,accepted,rejected,regions,previous = self.regroup()
        self.assertEqual([[b.box_id for b in g.group] for g in groups],[["single","center","left"]])
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]["kind"],"singleton_companion")
        self.assertEqual(accepted[0]["crop_conflicts"],())
        self.assertEqual(accepted[0]["corridor"]["white_fraction"],1.)
        self.assertEqual(accepted[0]["completeness"],"unverified_full_utterance")
        self.assertEqual(rejected,())
        self.assertEqual({i:regions[i] for i in previous},previous)
        self.assertEqual(set(regions)-set(previous),{3,4})

    def test_original_independent_balloon_divider_is_not_removed_by_detection_overlap(self):
        cv2.line(self.image,(104,16),(104,184),(0,0,0),1)
        groups,accepted,rejected,_,_ = self.regroup()
        self.assertEqual(accepted,())
        self.assertEqual(len(groups),2)
        self.assertTrue(any("original_boundary_blocks_singleton_seam" in r["reasons"] for r in rejected))

    def test_ordinary_small_dialogue_does_not_match_main_glyph_scale(self):
        self.image[35:145,105:122] = 255
        self.draw_main(108,size=7)
        _,accepted,rejected,_,_ = self.regroup()
        self.assertEqual(accepted,())
        self.assertTrue(any("singleton_complete_span_scale_or_alignment_limit" in r["reasons"] for r in rejected))

    def test_same_region_clear_gap_without_reciprocal_conflict_is_insufficient(self):
        self.boxes[2] = box("single",(105,30,132,150))
        self.groups[0] = Text_Group([self.boxes[2]],direction="vertical")
        _,accepted,rejected,_,_ = self.regroup()
        self.assertEqual(accepted,())
        self.assertTrue(any("no_reciprocal_main_footprint_conflict" in r["reasons"] for r in rejected))

    def test_unannotated_singleton_is_not_blanket_merged(self):
        self.annotations.pop("single-reading")
        _,accepted,_,_,_ = self.regroup()
        self.assertEqual(accepted,())

    def test_terminal_singleton_extension_cannot_absorb_another_column(self):
        self.groups[1].method = "singleton_companion_geometry"
        _,accepted,rejected,_,_ = self.regroup()
        self.assertEqual(accepted,())
        self.assertTrue(any("singleton_extension_is_terminal_not_a_chain" in r["reasons"] for r in rejected))

    def test_duplicate_main_footprints_cannot_manufacture_a_multi_column_host(self):
        self.boxes[0] = box("left",self.boxes[1].bbox)
        self.groups[1] = Text_Group([self.boxes[1],self.boxes[0]],direction="vertical")
        _,accepted,rejected,_,_ = self.regroup()
        self.assertEqual(accepted,())
        self.assertTrue(any("singleton_mains_not_distinct_ink_columns" in r["reasons"] for r in rejected))

    def test_small_artwork_in_visible_seam_blocks_even_without_a_long_divider(self):
        cv2.rectangle(self.image,(103,72),(106,79),(0,0,0),-1)
        _,accepted,rejected,_,_ = self.regroup()
        self.assertEqual(accepted,())
        self.assertTrue(any("original_singleton_corridor_not_clear_or_local" in r["reasons"] for r in rejected))

    def test_rotated_nonmember_is_not_swallowed_by_a_larger_union(self):
        intruder = box("intruder",(62,85,70,93))
        intruder.poly = np.array([(66,85),(70,89),(66,93),(62,89)],dtype=np.float64)
        self.boxes.append(intruder)
        _,accepted,rejected,regions,previous = self.regroup()
        self.assertEqual(accepted,())
        self.assertEqual(regions,previous)
        self.assertTrue(any("singleton_union_has_nonmember_conflicts" in r["reasons"] for r in rejected))

    def test_two_structurally_plausible_groups_stay_ambiguous_not_nearest_selected(self):
        self.boxes[2] = box("single",(100,30,146,150))
        self.boxes.extend([box("right-center",(140,30,172,150)),box("right-edge",(176,30,208,150)),
                           box("right-reading",(162,38,174,64))])
        self.draw_main(148)
        self.draw_main(180)
        self.draw_reading(168)
        self.annotations["right-reading"] = "right-center"
        self.groups = [Text_Group([self.boxes[6],self.boxes[5]],direction="vertical"),
                       Text_Group([self.boxes[2]],direction="vertical"),self.groups[1]]
        groups,accepted,rejected,regions,previous = self.regroup()
        self.assertEqual(len(groups),3)
        self.assertEqual(accepted,())
        self.assertEqual(regions,previous)
        self.assertEqual(sum("ambiguous_singleton_companion_groups" in r["reasons"] for r in rejected),2)


class SavedSingletonCompanionTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[2]

    @local_evidence(comparisons=("test-results/crop-isolation-v2/lmgJZ-comparison.json",))
    def test_original_three_main_balloon_excludes_adjacent_thanks_balloon(self):
        _,image,page,raw = load_unit_page(self.root/"test-results/crop-isolation-v2/lmgJZ-comparison.json",self.root)
        batch = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
        records = [r for r in batch.companion_regroupings if r.get("kind") == "singleton_companion"]
        self.assertEqual(len(records),1,batch.companion_rejections)
        self.assertEqual(records[0]["corridor"]["bounds"],(577,989,579,1062))
        self.assertEqual(records[0]["corridor"]["white_fraction"],1.)
        self.assertEqual(records[0]["boundary"]["longest_dark_stroke"],8)
        self.assertTrue(any(set(boundary["source_ids"]) == {41,44}
            for row in batch.rejected_merges for boundary in row["boundaries"]))
        proposal = next(p for p in batch.proposals if p.main_ids == (44,42,41))
        self.assertEqual(proposal.annotation_parents,((43,42),(45,44),(46,44)))
        self.assertEqual(proposal.member_ids,(44,42,41,43,45,46))
        self.assertEqual(proposal.crop_bounds,(518,979,610,1083))
        self.assertFalse({39,40}.intersection(proposal.member_ids))
        self.assertTrue(any(p.main_ids == (40,39) for p in batch.proposals))
        crop,evidence,error = prepare_automatic_crop(proposal,batch,image,
            original_image_bytes=Path(page.image_path).read_bytes())
        self.assertIsNone(error)
        self.assertTrue(proposal.crop_safe)
        self.assertEqual(proposal.crop_conflicts,())
        self.assertEqual(crop.tobytes(),image.crop(proposal.crop_bounds).tobytes())
        self.assertEqual(evidence.completeness,"unverified_full_utterance")
        self.assertFalse(proposal.reviewed)

    @local_evidence(comparisons=("test-results/crop-isolation-v2/lmgJZ-comparison.json",))
    def test_original_evidence_is_recomputed_and_old_companion_policy_is_rejected(self):
        _,image,page,raw = load_unit_page(self.root/"test-results/crop-isolation-v2/lmgJZ-comparison.json",self.root)
        with patch.object(cv2,"connectedComponentsWithStats",wraps=cv2.connectedComponentsWithStats) as components:
            a.propose_automatic_units(raw,image,complementary_ownership=True)
            page_shape = (image.height,image.width)
            before = sum(call.args[0].shape == page_shape for call in components.call_args_list)
            components.reset_mock()
            batch = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
            self.assertEqual(sum(call.args[0].shape == page_shape for call in components.call_args_list),before)
            self.assertEqual(before,6)
            components.reset_mock()
            proposal = next(p for p in batch.proposals if p.main_ids == (44,42,41))
            self.assertIsNone(prepare_automatic_crop(proposal,batch,image,
                original_image_bytes=Path(page.image_path).read_bytes())[2])
            self.assertEqual(sum(call.args[0].shape == page_shape for call in components.call_args_list),before)
        altered = copy.deepcopy(batch)
        altered.companion_regroupings[0]["annotation_region_support"] = ()
        self.assertEqual(prepare_automatic_crop(altered.proposals[0],altered,image,
            original_image_bytes=Path(page.image_path).read_bytes())[2],"automatic_proposal_or_structural_evidence_drift")
        old = replace(batch,companion_policy="bounded_vertical_companions_v1")
        self.assertEqual(prepare_automatic_crop(old.proposals[0],old,image,
            original_image_bytes=Path(page.image_path).read_bytes())[2],"invalid_automatic_companion_policy")

    @local_evidence(comparisons=("test-results/crop-isolation-v2/lmgJZ-comparison.json",))
    def test_new_join_does_not_override_mixed_script_routing(self):
        _,image,page,raw = load_unit_page(self.root/"test-results/crop-isolation-v2/lmgJZ-comparison.json",self.root)
        changed = tuple(replace(d,text="Ubuntu") if d.source_id == 44 else d for d in raw)
        batch = a.propose_automatic_units(changed,image,complementary_ownership=True,regroup_companions=True)
        proposal = next(p for p in batch.proposals if p.main_ids == (44,42,41))
        self.assertTrue(proposal.crop_safe)
        self.assertEqual(prepare_automatic_crop(proposal,batch,image,
            original_image_bytes=Path(page.image_path).read_bytes())[2],"mixed_script_not_routed_to_manga_ocr")

    @local_evidence(manifests=("test-results/single-glyph-parent-v1/automatic-proposals.json",))
    def test_identical_seven_pages_change_only_one_singleton_partition_and_derived_annotation_regions(self):
        baseline = json.loads((self.root/"test-results/single-glyph-parent-v1/automatic-proposals.json").read_text())
        totals = Counter()
        for page in baseline["pages"]:
            _,image,provenance,raw = load_unit_page(page["comparison_path"],self.root)
            batch = a.propose_automatic_units(raw,image,complementary_ownership=True,regroup_companions=True)
            previous_batch = page["batch"]
            self.assertEqual(json.loads(json.dumps([asdict(d) for d in batch.detections])),previous_batch["detections"])
            self.assertEqual(json.loads(json.dumps([asdict(r) for r in batch.relationships])),previous_batch["relationships"])
            self.assertEqual(json.loads(json.dumps(batch.ownership_repairs)),previous_batch["ownership_repairs"])
            self.assertEqual(json.loads(json.dumps(batch.rejected_merges)),previous_batch["rejected_merges"])
            previous = {(tuple(p["main_ids"]),tuple(map(tuple,p["annotation_parents"]))):p for p in previous_batch["proposals"]}
            current = {(p.main_ids,p.annotation_parents):p for p in batch.proposals}
            if provenance.page_id == "lmgJZ-comparison":
                self.assertEqual(set(previous)-set(current),{((42,41),((43,42),)),((44,),((45,44),(46,44)))})
                self.assertEqual(set(current)-set(previous),{((44,42,41),((43,42),(45,44),(46,44)))})
                expected_added = {45,46}
                self.assertTrue({13,16} <= {i for p in batch.proposals for i in p.main_ids})
                reversed_batch = a.propose_automatic_units(tuple(reversed(raw)),image,
                    complementary_ownership=True,regroup_companions=True)
                self.assertEqual(batch.proposals,reversed_batch.proposals)
                self.assertEqual(batch.companion_regroupings,reversed_batch.companion_regroupings)
            else:
                self.assertEqual(set(previous),set(current))
                expected_added = set()
                self.assertEqual(json.loads(json.dumps(batch.companion_regroupings)),previous_batch["companion_regroupings"])
            old_regions = {i:(r,c) for i,r,c in previous_batch["regions"]}
            new_regions = {i:(r,c) for i,r,c in batch.regions}
            self.assertEqual({i:new_regions[i] for i in old_regions},old_regions)
            self.assertEqual(set(new_regions)-set(old_regions),expected_added)
            for key in set(previous)&set(current):
                expected = dict(previous[key])
                expected.pop("proposal_id")
                actual = asdict(current[key])
                actual.pop("proposal_id")
                self.assertEqual(json.loads(json.dumps(actual)),expected)
            old_eligible = {i for unit in page["units"] if unit["eligibility_preflight"]["eligible"] for i in unit["proposal"]["member_ids"]}
            new_eligible = set()
            for proposal in batch.proposals:
                crop,_,error = prepare_automatic_crop(proposal,batch,image,
                    original_image_bytes=Path(provenance.image_path).read_bytes())
                if error is None:
                    totals["eligible_after"] += 1
                    new_eligible.update(proposal.member_ids)
                    self.assertEqual(crop.tobytes(),image.crop(proposal.crop_bounds).tobytes())
                self.assertFalse(proposal.reviewed)
                self.assertEqual(proposal.completeness,"unverified_full_utterance")
            self.assertEqual(new_eligible,old_eligible|({41,42,43,44,45,46} if expected_added else set()))
            totals["raw"] += len(raw)
            totals["main"] += batch.coverage["main_detections"]
            totals["annotations"] += batch.coverage["tentative_annotations"]
            totals["units_before"] += len(page["units"])
            totals["units_after"] += len(batch.proposals)
            totals["unsafe_before"] += page["coverage"]["unsafe_crops"]
            totals["unsafe_after"] += batch.coverage["unsafe_crops"]
            totals["conflicts"] += batch.coverage["relationship_conflicts"]
            totals["eligible_before"] += sum(unit["eligibility_preflight"]["eligible"] for unit in page["units"])
        self.assertEqual(totals,{"raw":268,"main":252,"annotations":16,"units_before":133,"units_after":132,
                                 "unsafe_before":21,"unsafe_after":19,"conflicts":11,"eligible_before":56,"eligible_after":57})


if __name__ == "__main__":
    unittest.main()
