"""Offline SDK-contract fixtures; no Paddle import, construction or recognition."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from tralens.ocr.detection import OCR
from tralens.ocr.paddle_result import associate_paddle_result, PaddleResultContractError
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tralens.ocr.automatic_units import propose_automatic_units
from tralens.ocr.types import OCRStatus


def poly(x):
    return [[x,10],[x+10,10],[x+10,70],[x,70]]


class PaddleResultTests(unittest.TestCase):
    def setUp(self):
        self.polys = [poly(10),poly(35),poly(60)]

    def parse(self, rec_indices=(0,1,2), **changes):
        result = {"dt_polys":self.polys,"rec_polys":[self.polys[i] for i in rec_indices],
                  "rec_texts":["text-"+str(i) for i in rec_indices],"rec_scores":[.99]*len(rec_indices)}
        result.update(changes)
        return associate_paddle_result(result)

    def test_filtered_first_line_does_not_shift_text_or_remove_raw_boxes(self):
        boxes,report = self.parse((1,2))
        self.assertEqual([b.box_id for b in boxes],[0,1,2])
        self.assertEqual([b.paddle_text for b in boxes],["","text-1","text-2"])
        self.assertEqual(boxes[0].status,OCRStatus.REVIEW)
        self.assertEqual(report["unrecognized_detector_indices"],(0,))
        for i,b in enumerate(boxes):
            self.assertEqual(b.paddle_poly.tolist(),self.polys[i])

    def test_filtered_middle_line_preserves_original_indices(self):
        boxes,_ = self.parse((0,2))
        self.assertEqual([b.paddle_text for b in boxes],["text-0","","text-2"])
        self.assertEqual(boxes[2].ocr_metadata["paddle_association"]["recognition_index"],1)

    def test_reordered_cyclic_reversed_numpy_polygons_bind_by_exact_geometry(self):
        rec = np.array([self.polys[2][1:]+self.polys[2][:1],list(reversed(self.polys[0])),self.polys[1]],dtype=np.float64)
        boxes,report = self.parse((2,0,1),dt_polys=np.array(self.polys,dtype=np.int32),rec_polys=rec,
                                 rec_texts=np.array(["C","A","B"]),rec_scores=np.array([.9,.95,.99],dtype=np.float32))
        self.assertEqual([b.paddle_text for b in boxes],["A","B","C"])
        self.assertEqual([b.paddle_poly.dtype for b in boxes],[np.dtype("int32")]*3)
        self.assertIsInstance(boxes[0].paddle_confidence,np.float32)
        self.assertEqual(report["status"],"aligned")

    def test_no_recognitions_retains_every_detector_only_polygon(self):
        for empty in ([],np.empty((0,4,2),dtype=np.float32)):
            boxes,report = self.parse((),rec_polys=empty,rec_texts=np.array([],dtype=str),rec_scores=np.array([]))
            self.assertEqual(len(boxes),3)
            self.assertTrue(all(b.paddle_text == "" and b.paddle_confidence == 0 and b.status != OCRStatus.ACCEPTED for b in boxes))
            self.assertEqual(report["matched_detector_indices"],())

    def test_balanced_legacy_fixture_has_explicit_full_length_provenance(self):
        boxes,report = associate_paddle_result({"dt_polys":self.polys,"rec_texts":["A","B","C"],"rec_scores":[.9,.91,.92]})
        self.assertEqual([b.paddle_text for b in boxes],["A","B","C"])
        self.assertEqual(report["mode"],"legacy_full_length_alignment")

    def test_legacy_short_or_malformed_arrays_do_not_guess_alignment(self):
        for changes in ({"rec_texts":["C"],"rec_scores":[.99]},
                        {"rec_texts":"ABC","rec_scores":[.99]*3},
                        {"rec_texts":["A","B","C"],"rec_scores":None}):
            boxes,report = associate_paddle_result(dict(dt_polys=self.polys,**changes))
            self.assertTrue(all(b.paddle_text == "" for b in boxes))
            self.assertEqual(report["status"],"unresolved")

    def test_present_but_invalid_rec_polys_never_uses_legacy_fallback(self):
        boxes,report = self.parse(rec_polys=None)
        self.assertTrue(all(b.paddle_text == "" for b in boxes))
        self.assertEqual(report["mode"],"explicit_rec_polys")

    def test_duplicate_detector_geometry_is_ambiguous_even_with_balanced_readings(self):
        boxes,report = self.parse((0,2),dt_polys=[self.polys[0],self.polys[0],self.polys[2]])
        self.assertEqual([b.paddle_text for b in boxes],["","","text-2"])
        self.assertIn("duplicate_detector_geometry",report["issues"])
        legacy,_ = associate_paddle_result({"dt_polys":[self.polys[0]]*2,"rec_texts":["A","A"],"rec_scores":[.99,.99]})
        self.assertTrue(all(b.paddle_text == "" for b in legacy))

    def test_duplicate_recognition_geometry_never_selects_by_confidence_or_agreement(self):
        boxes,report = self.parse((0,0,2),rec_texts=["A","A","C"],rec_scores=[.86,.99,.99])
        self.assertEqual([b.paddle_text for b in boxes],["","","C"])
        self.assertTrue(all("duplicate_recognition_geometry" in row["issues"] for row in report["records"][:2]))

    def test_foreign_and_nearby_geometry_is_retained_but_never_proximity_matched(self):
        foreign = (np.array(self.polys[0],dtype=np.float64)+.00001).tolist()
        boxes,report = self.parse((0,2),rec_polys=[foreign,self.polys[2]])
        self.assertEqual([b.paddle_text for b in boxes],["","","text-2"])
        self.assertEqual(report["raw"]["rec_polys"][0],foreign)
        self.assertIn("foreign_recognition_geometry",report["records"][0]["issues"])

    def test_malformed_recognition_lengths_are_not_zipped_or_truncated(self):
        boxes,report = self.parse(rec_texts=["A"])
        self.assertEqual(len(boxes),3)
        self.assertTrue(all(b.paddle_text == "" for b in boxes))
        self.assertEqual(len(report["raw"]["rec_scores"]),3)

    def test_invalid_scores_are_reviewable_raw_evidence_not_applied_confidence(self):
        for score in (None,float("nan"),float("inf"),-1.,1.1,True,"0.99",10**400,np.array(.99)):
            with self.subTest(score_type=type(score).__name__):
                boxes,report = self.parse((0,2),rec_scores=[score,.99])
                self.assertEqual([b.paddle_text for b in boxes],["","","text-2"])
                self.assertIn("invalid_recognition_score",report["records"][0]["issues"])
                json.dumps(report,allow_nan=False)

    def test_invalid_text_or_polygon_does_not_attach_a_reading(self):
        for changes in ({"rec_texts":[["A"],"B","C"]},
                        {"rec_polys":[None,self.polys[1],self.polys[2]]}):
            boxes,_ = self.parse(**changes)
            self.assertEqual(boxes[0].paddle_text,"")
            self.assertEqual(boxes[2].paddle_text,"text-2" if "rec_polys" in changes else "C")

    def test_high_score_blank_reading_is_not_accepted(self):
        boxes,_ = self.parse((0,),rec_texts=[""])
        self.assertEqual(boxes[0].paddle_confidence,.99)
        self.assertNotEqual(boxes[0].status,OCRStatus.ACCEPTED)

    def test_finite_invalid_detector_geometry_remains_blank_with_original_polygon(self):
        bow = [[10,10],[20,70],[10,70],[20,10]]
        boxes,_ = self.parse((0,2),dt_polys=[bow,self.polys[1],self.polys[2]])
        self.assertEqual(boxes[0].paddle_poly.tolist(),bow)
        self.assertEqual(boxes[0].paddle_text,"")
        self.assertIn("invalid_detector_geometry",boxes[0].ocr_metadata["paddle_association"]["issues"])

    def test_unrepresentable_detector_geometry_fails_page_with_complete_selective_evidence(self):
        result = {"dt_polys":[self.polys[0],[],self.polys[2]],"rec_polys":[self.polys[2]],"rec_texts":["C"],"rec_scores":[.99]}
        with self.assertRaises(PaddleResultContractError) as caught:
            associate_paddle_result(result)
        self.assertEqual(caught.exception.report["raw"],result)
        self.assertEqual(caught.exception.report["status"],"invalid_detector_page")

    def test_source_evidence_is_detached_and_survives_automatic_snapshots(self):
        result = {"dt_polys":copy.deepcopy(self.polys),"rec_polys":[poly(90),self.polys[2]],"rec_texts":["orphan","C"],"rec_scores":[.99,.9]}
        before = copy.deepcopy(result)
        boxes,report = associate_paddle_result(result)
        self.assertEqual(result,before)
        image = Image.new("RGB",(120,100),"white")
        page = PageProvenance("fixture","/original.png","1"*64,"2"*64,image_fingerprint(image))
        batch = propose_automatic_units(snapshot_detections(boxes,page),image)
        self.assertEqual([d.source_id for d in batch.detections],[0,1,2])
        self.assertEqual(json.loads(batch.detections[0].metadata_json)["paddle_page_contract"]["raw"],before)
        report["raw"]["rec_texts"][0] = "changed report only"
        self.assertEqual(boxes[0].ocr_metadata["paddle_page_contract"]["raw"]["rec_texts"][0],"orphan")

    def test_predict_contract_error_preserves_report_and_invalidates_previous_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"image.png"
            Image.new("RGB",(120,100),"white").save(path)
            ocr = OCR.__new__(OCR)
            ocr.image_path = str(path)
            good = {"dt_polys":self.polys,"rec_polys":self.polys,"rec_texts":["A","B","C"],"rec_scores":[.99]*3}
            ocr.ocr = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(res=good)]))
            ocr.review_boxes = Mock()
            ocr.process_images(propose_units=True)
            previous = ocr.automatic_unit_proposals
            ocr.automatic_recognition_results = (("test-only old history",),)
            bad = dict(good,dt_polys=[self.polys[0],None,self.polys[2]])
            ocr.ocr.predict.return_value = [bad]
            with self.assertRaises(PaddleResultContractError):
                ocr.process_images(propose_units=True)
            self.assertFalse(hasattr(ocr,"automatic_unit_proposals"))
            self.assertFalse(hasattr(ocr,"automatic_recognition_results"))
            self.assertFalse(hasattr(ocr,"raw_detections"))
            self.assertEqual(ocr.prediction_contract_reports[0]["raw"]["dt_polys"][1],None)
            self.assertTrue(previous)
