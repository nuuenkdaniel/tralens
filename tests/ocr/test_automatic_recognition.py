"""Automatic candidate handoff uses synthetic callbacks only, never live models."""
import copy
import hashlib
import importlib.util
import json
import socket
import tempfile
import unittest
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from tralens.ocr import automatic_units as a
from tralens.ocr import automatic_recognition as r
from tralens.ocr import detection
from tralens.ocr.recognition_units import PageProvenance, image_fingerprint, snapshot_detections
from tralens.ocr.types import Text_Box
from tools import diagnostics as d
from .test_automatic_units import box


class AutomaticRecognitionTests(unittest.TestCase):
    def fixture(self, *, text="漢字", annotation=False, horizontal=False, two=False, enclosed=True):
        image = Image.new("RGB",(180,130),"white")
        draw = ImageDraw.Draw(image)
        boxes = []
        for index,offset in enumerate((0,95) if two else (0,)):
            if enclosed:
                draw.rectangle((5+offset,5,75+offset,110),outline="black",width=2)
            bounds = (20+offset,30,65+offset,42) if horizontal else (25+offset,20,45+offset,90)
            boxes.append(box(f"main-{index}",bounds,text))
            if horizontal:
                for x in (23,38,53):
                    draw.rectangle((x+offset,32,x+offset+7,39),fill="black")
            else:
                for y in (26,46,66):
                    draw.rectangle((28+offset,y,37+offset,y+9),fill="black")
        if annotation:
            boxes.append(box("reading",(43,27,51,57),"かんじ"))
            for y in (30,35,40,45,50):
                draw.rectangle((46,y,48,y+2),fill="black")
        stream = BytesIO()
        image.save(stream,format="PNG")
        content = stream.getvalue()
        page = PageProvenance("fixture","/original.png",hashlib.sha256(content).hexdigest(),"2"*64,image_fingerprint(image))
        raw = snapshot_detections(boxes,page)
        return image,content,a.propose_automatic_units(raw,image)

    def recognize(self, image,content,batch, callback=None, **kwargs):
        callback = Mock(return_value="漢字！") if callback is None else callback
        results = r.recognize_automatic_batch(batch,image,callback,original_image_bytes=content,
            model_identity="synthetic-test-only",**kwargs)
        return callback,results

    def test_isolated_candidate_is_unverified_exact_pixels_annotation_excluded_nonmutation(self):
        image,content,batch = self.fixture(annotation=True)
        before = copy.deepcopy(asdict(batch))
        self.assertEqual(batch.coverage["tentative_annotations"],1)
        callback,results = self.recognize(image,content,batch,model_settings={"local_only":True})
        self.assertEqual(callback.call_count,1)
        result = results[0]
        candidate = result.candidates[0]
        self.assertEqual(candidate.reference_main_text,"漢字")
        self.assertEqual(candidate.text,"漢字！")
        self.assertTrue(candidate.differing_spans)
        self.assertEqual(candidate.status,"automated_unverified")
        self.assertFalse(candidate.reviewed)
        self.assertEqual(candidate.crop.completeness,"unverified_full_utterance")
        self.assertEqual(candidate.crop.script,"han_only_language_ambiguous")
        self.assertEqual(callback.call_args.args[0].tobytes(),image.crop(candidate.crop.bounds).tobytes())
        self.assertEqual(callback.call_args.kwargs,{"source_proposal_id":batch.proposals[0].proposal_id,
            "member_ids":batch.proposals[0].member_ids,"crop_bounds":batch.proposals[0].crop_bounds})
        self.assertEqual(asdict(batch),before)

    def test_agreement_remains_automated_unverified_and_single_primary_attempt(self):
        image,content,batch = self.fixture(text="こんにちは")
        callback = Mock(return_value="こんにちは")
        _,results = self.recognize(image,content,batch,callback)
        self.assertEqual(results[0].status,"automated_unverified")
        self.assertEqual(results[0].candidates[0].differing_spans,())
        self.recognize(image,content,batch,callback,results=results)
        self.assertEqual(callback.call_count,1)

    def test_horizontal_japanese_eligible_latin_and_mixed_scripts_skip_deterministically(self):
        image,content,batch = self.fixture(text="こんにちは",horizontal=True)
        callback,results = self.recognize(image,content,batch)
        self.assertEqual(callback.call_count,1)
        self.assertEqual(results[0].crop.script,"japanese_kana_evidence")
        for text,reason in (("Ubuntu","unsupported_script_for_manga_ocr"),
                            ("日本abc","mixed_script_not_routed_to_manga_ocr"),
                            ("!!","script_inconclusive_no_lexical_evidence"),
                            ("Привет","unsupported_script_for_manga_ocr")):
            image,content,batch = self.fixture(text=text,horizontal=True)
            callback,results = self.recognize(image,content,batch)
            callback.assert_not_called()
            self.assertEqual(results[0].reason,reason)

    def test_unknown_orientation_and_unreliable_region_ownership_block(self):
        image,content,batch = self.fixture(enclosed=False)
        callback,results = self.recognize(image,content,batch)
        callback.assert_not_called()
        self.assertEqual(results[0].reason,"unknown_automated_region_ownership")
        page = batch.page
        raw = snapshot_detections([box("square",(25,25,45,45),"字")],page)
        square = a.propose_automatic_units(raw,image)
        callback,results = self.recognize(image,content,square)
        callback.assert_not_called()
        self.assertEqual(results[0].reason,"unknown_automatic_reading_orientation")

    def test_stale_original_bytes_pixels_and_page_provenance_block(self):
        image,content,batch = self.fixture()
        wrong = image.copy()
        wrong.putpixel((0,0),(0,0,0))
        for changed_image,changed_content,changed_batch in ((wrong,content,batch),(image,b"stale",batch),
            (image,content,replace(batch,page=replace(batch.page,pixels_sha256="3"*64)))):
            callback,results = self.recognize(changed_image,changed_content,changed_batch)
            callback.assert_not_called()
            self.assertEqual(results[0].attempts,0)

    def test_mutated_proposals_crops_membership_review_flags_and_regions_block(self):
        image,content,batch = self.fixture()
        proposal = batch.proposals[0]
        mutations = [replace(proposal,crop_bounds=(26,20,45,90)),
                     replace(proposal,member_ids=("foreign",)),replace(proposal,member_ids=proposal.member_ids*2),
                     replace(proposal,main_ids=()),replace(proposal,reviewed=True),
                     replace(proposal,direction="horizontal"),replace(proposal,region_ids=(999,)),
                     replace(proposal,structural_reasons=()),replace(proposal,completeness="verified")]
        for changed in mutations:
            with self.subTest(changed=changed):
                callback,results = self.recognize(image,content,replace(batch,proposals=(changed,)))
                callback.assert_not_called()
                self.assertFalse(results[0].candidates)
        callback,results = self.recognize(image,content,replace(batch,regions=(("main-0",999,1.),)))
        callback.assert_not_called()

    def test_foreign_duplicate_raw_ids_and_text_or_geometry_drift_block(self):
        image,content,batch = self.fixture()
        raw = batch.detections[0]
        for detections in ((replace(raw,text="別字"),),(replace(raw,polygon=((26,20),(45,20),(45,90),(26,90))),),
                           (replace(raw,source_id="foreign"),),batch.detections*2,
                           (replace(raw,page=replace(raw.page,comparison_sha256="4"*64)),)):
            callback,results = self.recognize(image,content,replace(batch,detections=detections))
            callback.assert_not_called()
            self.assertEqual(results[0].attempts,0)

    def test_rotated_nonmember_conflict_and_full_crop_are_revalidated_not_crop_safe_flag(self):
        image,content,batch = self.fixture(annotation=True)
        neighbor = replace(batch.detections[0],source_id="neighbor",text="!?",
                           polygon=((49,86),(53,90),(49,94),(45,90)))
        unsafe = a.propose_automatic_units(batch.detections+(neighbor,),image)
        proposal = next(p for p in unsafe.proposals if "main-0" in p.main_ids)
        self.assertFalse(proposal.crop_safe)
        self.assertEqual(proposal.crop_conflicts[0].source_id,"neighbor")
        self.assertGreater(proposal.crop_conflicts[0].intersection_area,0)
        self.assertTrue(proposal.crop_conflicts[0].intersection_polygon)
        result = r.AutomaticRecognitionResult(proposal.proposal_id)
        callback = Mock()
        r.recognize_automatic_proposal(result,proposal,unsafe,image,callback,
            original_image_bytes=content,model_identity="test-only")
        callback.assert_not_called()
        self.assertEqual(result.reason,"unsafe_automatic_crop")
        forged = replace(proposal,crop_safe=True,crop_conflicts=())
        altered = replace(unsafe,proposals=tuple(forged if p == proposal else p for p in unsafe.proposals))
        callback,results = self.recognize(image,content,altered)
        callback.assert_not_called()

    def test_failure_consumes_global_limit_and_no_automatic_retry(self):
        image,content,batch = self.fixture(two=True)
        callback = Mock(side_effect=RuntimeError("synthetic failure"))
        _,results = self.recognize(image,content,batch,callback,max_attempts=1)
        self.assertEqual(callback.call_count,1)
        self.assertEqual(sum(row.attempts for row in results),1)
        self.assertEqual(results[0].status,"error")
        self.assertIn("synthetic failure",results[0].error)
        self.assertEqual(results[1].reason,"automatic_request_limit")
        self.recognize(image,content,batch,callback,max_attempts=1,results=results)
        self.assertEqual(callback.call_count,1)
        self.recognize(image,content,batch,callback,max_attempts=2,results=results)
        self.assertEqual(callback.call_count,2)

    def test_original_pixels_are_rechecked_before_every_attempt(self):
        image,content,batch = self.fixture(two=True)
        def mutate_source(crop,**kwargs):
            image.putpixel((0,0),(0,0,0))
            return "字"
        callback,results = self.recognize(image,content,batch,Mock(side_effect=mutate_source))
        self.assertEqual(callback.call_count,1)
        self.assertEqual(results[1].reason,"original_page_or_image_provenance_mismatch")

    def test_invalid_output_limits_and_retained_state_do_not_retry(self):
        image,content,batch = self.fixture()
        callback,results = self.recognize(image,content,batch,Mock(return_value=("text",.9)))
        self.assertEqual(results[0].attempts,1)
        self.assertEqual(results[0].status,"error")
        for limit in (-1,21,True):
            with self.assertRaises(ValueError):
                self.recognize(image,content,batch,max_attempts=limit)
        with self.assertRaises(ValueError):
            self.recognize(image,content,replace(batch,proposals=batch.proposals*2))
        with self.assertRaises(ValueError):
            self.recognize(image,content,batch,results=[r.AutomaticRecognitionResult("foreign")])

    def test_explicit_production_coordination_preserves_boxes_groups_and_primary_history(self):
        image,content,batch = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"original.png"
            path.write_bytes(content)
            boxes = [Text_Box(raw.source_id,.99,raw.polygon,raw.text) for raw in batch.detections]
            ocr = detection.OCR.__new__(detection.OCR)
            ocr.image_path = str(path)
            ocr.predict = Mock(return_value=[boxes])
            groups = ocr.process_images(propose_units=True)
            before = [(b.text,b.paddle_poly.tolist(),b.status) for b in boxes]
            callback = Mock(return_value="字")
            results = ocr.recognize_automatic_proposals(callback,model_identity="test-only",max_attempts=1)
            self.assertEqual(callback.call_count,1)
            ocr.recognize_automatic_proposals(callback,model_identity="test-only",max_attempts=1)
            self.assertEqual(callback.call_count,1)
            self.assertEqual(results[0][0].status,"automated_unverified")
            self.assertEqual(before,[(b.text,b.paddle_poly.tolist(),b.status) for b in boxes])
            self.assertTrue(groups[0])

    def test_import_has_no_model_network_image_or_file_side_effects(self):
        spec = importlib.util.spec_from_file_location("tralens.ocr.automatic_recognition_probe",Path(r.__file__))
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.open",side_effect=AssertionError("import file side effect")), \
             patch.object(Image,"open",side_effect=AssertionError("import image side effect")), \
             patch.object(socket.socket,"connect",side_effect=AssertionError("network forbidden")), \
             patch.object(socket,"create_connection",side_effect=AssertionError("network forbidden")), \
             patch.object(d.importlib,"import_module",side_effect=AssertionError("model import forbidden")):
            spec.loader.exec_module(module)


class AutomaticFacadeLifecycleTests(unittest.TestCase):
    def setUp(self):
        _,content,_ = AutomaticRecognitionTests().fixture()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)/"original.png"
        path.write_bytes(content)
        self.boxes = [Text_Box("main-0",.99,[(25,20),(45,20),(45,90),(25,90)],"漢字")]
        self.ocr = detection.OCR.__new__(detection.OCR)
        self.ocr.image_path = str(path)
        self.ocr.predict = Mock(return_value=[self.boxes])
        self.ocr.review_boxes = Mock()
        self.ocr.process_images(propose_units=True)
        self.old_proposals = self.ocr.automatic_unit_proposals
        self.callback = Mock(return_value="synthetic first reading")
        self.old_results = self.ocr.recognize_automatic_proposals(self.callback,model_identity="test-only")
        self.history = copy.deepcopy(asdict(self.old_results[0][0]))

    def assert_history_preserved(self):
        self.assertEqual(asdict(self.old_results[0][0]),self.history)
        self.assertEqual(self.boxes[0].paddle_text,"漢字")

    def test_default_new_pass_removes_old_opt_in_proposals_and_candidates(self):
        self.ocr.process_images()
        self.assertFalse(hasattr(self.ocr,"automatic_unit_proposals"))
        self.assertFalse(hasattr(self.ocr,"automatic_recognition_results"))
        with self.assertRaises(ValueError):
            self.ocr.recognize_automatic_proposals(self.callback,model_identity="test-only")
        self.assertEqual(self.callback.call_count,1)
        self.assert_history_preserved()

    def test_same_proposal_ids_in_new_pass_do_not_reuse_previous_candidates(self):
        self.ocr.process_images(propose_units=True)
        self.assertEqual(self.ocr.automatic_unit_proposals,self.old_proposals)
        callback = Mock(return_value="synthetic new reading")
        results = self.ocr.recognize_automatic_proposals(callback,model_identity="test-only")
        self.assertEqual(callback.call_count,1)
        self.assertEqual(results[0][0].candidates[0].text,"synthetic new reading")
        self.assertIsNot(results[0][0],self.old_results[0][0])
        self.ocr.recognize_automatic_proposals(callback,model_identity="test-only")
        self.assertEqual(callback.call_count,1,"same-pass primary attempts must still not retry")
        self.assert_history_preserved()

    def test_new_raw_reading_invalidates_results_with_old_proposal_identity(self):
        self.ocr.predict.return_value = [[Text_Box("main-0",.99,self.boxes[0].paddle_poly,"別字")]]
        self.ocr.process_images(propose_units=True)
        self.assertNotEqual(self.ocr.automatic_unit_proposals,self.old_proposals)
        callback = Mock(return_value="new-pass callback")
        results = self.ocr.recognize_automatic_proposals(callback,model_identity="test-only")
        self.assertEqual(callback.call_count,1)
        self.assertEqual(results[0][0].candidates[0].reference_main_text,"別字")
        self.assert_history_preserved()

    def test_failed_prediction_cannot_leave_previous_page_recognizable(self):
        self.ocr.predict.side_effect = RuntimeError("synthetic prediction failure")
        with self.assertRaises(RuntimeError):
            self.ocr.process_images(propose_units=True)
        self.assertFalse(hasattr(self.ocr,"automatic_unit_proposals"))
        self.assertFalse(hasattr(self.ocr,"automatic_recognition_results"))
        self.assertFalse(hasattr(self.ocr,"raw_detections"))
        with self.assertRaises(ValueError):
            self.ocr.recognize_automatic_proposals(self.callback,model_identity="test-only")
        self.assertEqual(self.callback.call_count,1)
        self.assert_history_preserved()

    def test_failed_later_page_does_not_publish_partial_proposal_set(self):
        self.ocr.predict.return_value = [self.boxes,self.boxes]
        with patch.object(detection,"propose_page_from_boxes",side_effect=[self.old_proposals[0],RuntimeError("synthetic proposal failure")]):
            with self.assertRaises(RuntimeError):
                self.ocr.process_images(propose_units=True)
        self.assertFalse(hasattr(self.ocr,"automatic_unit_proposals"))
        self.assertFalse(hasattr(self.ocr,"automatic_recognition_results"))
        with self.assertRaises(ValueError):
            self.ocr.recognize_automatic_proposals(self.callback,model_identity="test-only")
        self.assertEqual(self.callback.call_count,1)
        self.assert_history_preserved()

    def test_invalid_arguments_do_not_start_or_invalidate_an_existing_pass(self):
        for kwargs in ({"propose_units":"yes"},{"regroup_companions":True},
                       {"propose_units":True,"media_type":"invalid-media"}):
            self.ocr.predict.reset_mock()
            with self.assertRaises(ValueError):
                self.ocr.process_images(**kwargs)
            self.ocr.predict.assert_not_called()
            self.assertIs(self.ocr.automatic_unit_proposals,self.old_proposals)
            self.assertIs(self.ocr.automatic_recognition_results,self.old_results)
        self.assert_history_preserved()


class AutomaticRecognitionDiagnosticsTests(unittest.TestCase):
    def saved(self,root,text="漢字",enclosed=True,name="page"):
        image,content,batch = AutomaticRecognitionTests().fixture(text=text,enclosed=enclosed)
        image_path = root/(name+".png")
        image_path.write_bytes(content)
        comparison = root/(name+"-comparison.json")
        comparison.write_text(json.dumps({"input":str(image_path),"boxes":[{
            "id":raw.source_id,"polygon":raw.polygon,"paddle":raw.text,"selected":raw.text,
            "score":raw.confidence,"status":raw.status} for raw in batch.detections]}))
        return comparison

    def test_default_diagnosis_is_model_free_and_records_exact_conflict_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = self.saved(root)
            with patch.object(d.importlib,"import_module",side_effect=AssertionError("model import forbidden")):
                report = d.run_automatic_proposals([comparison],root/"default")
            self.assertEqual(report["recognition_calls"],0)
            self.assertEqual(report["eligible_preflight_proposals"],1)
            record = report["pages"][0]["units"][0]
            self.assertEqual(record["recognition"]["reason"],"automatic_model_disabled")
            self.assertFalse(record["diagnosis"]["ownership_inferred"])
            image,_,batch = AutomaticRecognitionTests().fixture(annotation=True)
            neighbor = replace(batch.detections[0],source_id="neighbor",polygon=((49,86),(53,90),(49,94),(45,90)))
            unsafe = a.propose_automatic_units(batch.detections+(neighbor,),image)
            proposal = next(p for p in unsafe.proposals if "main-0" in p.main_ids)
            diagnosis = d.diagnose_automatic_proposal(proposal,unsafe)
            self.assertEqual(diagnosis["nonmember_conflicts"][0]["conflict"]["source_id"],"neighbor")
            self.assertIn("rectangular_gap_contamination",diagnosis["categories"])

    def test_global_failure_accounting_over_saved_pages_and_candidate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first,second = self.saved(root,name="first"),self.saved(root,name="second")
            callback = Mock(side_effect=RuntimeError("synthetic failure"))
            report = d.run_automatic_proposals([first,second],root/"limited",run_model=True,max_attempts=1,
                recognizer=callback,recognizer_identity="test-only")
            self.assertEqual(callback.call_count,1)
            self.assertEqual(report["recognition_calls"],1)
            self.assertEqual(report["live_model_calls"],0)
            self.assertEqual(report["pages"][1]["units"][0]["recognition"]["reason"],"automatic_request_limit")
            candidate = d.run_automatic_proposals([first],root/"candidate",run_model=True,
                recognizer=Mock(return_value="漢字！"),recognizer_identity="test-only")
            entry = candidate["pages"][0]["units"][0]
            self.assertEqual(entry["before_text"],"漢字")
            self.assertEqual(entry["after_text"],"漢字！")
            self.assertTrue(entry["recognition"]["candidates"][0]["differing_spans"])

    def test_explicit_live_opt_in_with_no_eligible_crop_does_not_import_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = self.saved(root,text="Latin")
            with patch.object(d.importlib,"import_module",side_effect=AssertionError("model import forbidden")) as importer:
                report = d.run_automatic_proposals([comparison],root/"blocked",run_model=True,model_path=root)
            importer.assert_not_called()
            self.assertEqual(report["recognition_calls"],0)
            self.assertEqual(report["live_model_calls"],0)
            with self.assertRaises(ValueError):
                d.run_automatic_proposals([comparison],root/"invalid",run_model=True,max_attempts=21,model_path=root)
