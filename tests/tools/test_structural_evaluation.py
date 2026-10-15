"""Synthetic saved-artifact fixtures; no OCR/provider imports or calls."""
import ast
import copy
import hashlib
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from tools import structural_evaluation as e


class StructuralEvaluationTests(unittest.TestCase):
    def setUp(self):
        printer = patch("builtins.print")
        printer.start()
        self.addCleanup(printer.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image = Image.new("RGB", (100,100), "white")
        self.image.putpixel((15,15),(3,7,11))
        path = self.root/"original.png"
        self.image.save(path)
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        prefix = b"RGB:100:100:"
        pixels = hashlib.sha256(prefix+self.image.tobytes()).hexdigest()
        self.artifact = {"pages":[],"page_errors":[],"recognition_calls":0}
        for page_index in range(2):
            provenance = {"page_id":f"page-{page_index}","image_path":str(path),"image_sha256":image_hash,
                          "comparison_sha256":"2"*64,"pixels_sha256":pixels}
            page = {"page":provenance,"overlay_path":"existing-overlay.png",
                    "batch":{"page":provenance,"detections":[],"relationships":[],"proposals":[]},"units":[]}
            for index in range(3):
                polygon = [[10+index*25,10],[30+index*25,10],[30+index*25,40],[10+index*25,40]]
                page["batch"]["detections"].append({"source_id":index,"page":provenance,"text":"raw",
                    "status":"accepted","confidence":.99,"polygon":polygon})
                proposal = {"proposal_id":f"proposal-{page_index}-{index}","page":provenance,
                    "member_ids":[index],"main_ids":[index],"annotation_ids":[],"annotation_parents":[],
                    "direction":"horizontal" if index == 1 else "vertical","region_ids":[],
                    "structural_reasons":["unknown_region_ownership"],"crop_safe":index != 0,
                    "crop_bounds":[10+index*25,10,30+index*25,40],"reviewed":False,
                    "structural_status":"proposal_unverified","crop_conflicts":[],
                    "completeness":"unverified_full_utterance"}
                record = {"case_id":proposal["proposal_id"],"source_ids":[index],"proposal":proposal,
                          "diagnosis":{"nonmember_conflicts":[],"script_evidence":"clearly_non_japanese" if index == 1 else "han_only_language_ambiguous"},
                          "eligibility_preflight":{"reason":"unsafe_automatic_crop" if index == 0 else "unknown_automated_region_ownership"},
                          "recognition":{"status":"skipped","attempts":0,"candidates":[]}}
                page["batch"]["proposals"].append(proposal)
                page["units"].append(record)
            self.artifact["pages"].append(page)
        self.path = self.root/"automatic-proposals.json"
        self.path.write_text(json.dumps(self.artifact))

    def run_eval(self,name="output",**kwargs):
        return e.evaluate(self.path,self.root/name,**kwargs)

    def test_full_counts_deterministic_selection_preserves_evidence_and_exact_pixels(self):
        before = self.path.read_bytes()
        report = self.run_eval(max_cases=4,per_page=2)
        self.assertEqual(report["counts"]["proposals"],6)
        self.assertEqual(report["counts"]["selected_cases"],4)
        self.assertEqual(report["category_counts_overlapping"]["unsafe_crop"],2)
        self.assertEqual(report["dominant_saved_blocker"],"unknown_automated_region_ownership")
        self.assertEqual(report["saved_eligibility_blockers"]["unknown_automated_region_ownership"],4)
        evidence = json.loads((self.root/"output/evidence.json").read_text())
        self.assertEqual(evidence["pages"][0]["source_batch"],self.artifact["pages"][0]["batch"])
        for case in evidence["cases"]:
            saved = next(r for p in self.artifact["pages"] for r in p["units"] if r["case_id"] == case["binding"]["proposal_id"])
            self.assertEqual(case["source_record"],saved)
            with Image.open(self.root/"output"/case["evidence"]["actual_crop"]["path"]) as crop:
                self.assertEqual(crop.tobytes(),self.image.crop(case["binding"]["crop_bounds"]).tobytes())
        self.run_eval("repeat",max_cases=4,per_page=2)
        self.assertEqual((self.root/"output/evidence.json").read_bytes(),(self.root/"repeat/evidence.json").read_bytes())
        self.assertEqual(self.path.read_bytes(),before)

    def test_reference_bindings_are_structural_only_and_never_ocr_approval(self):
        self.run_eval(max_cases=2,per_page=1)
        evidence = json.loads((self.root/"output/evidence.json").read_text())
        reference = json.loads((self.root/"output/structural-reference.json").read_text())
        record = reference["records"][0]
        record.update(state="provisional_visual_observation",observer="test-only visual fixture",
                      observed_categories=["ambiguous_ownership"],notes="Synthetic expected unresolved ownership.")
        e.validate_references(reference,evidence["cases"])
        for field,value in (("candidate_reviewed",True),("independent_transcription_reference",True),
                            ("completeness","verified"),("reviewed",True)):
            wrong = copy.deepcopy(reference)
            wrong["records"][0][field] = value
            with self.assertRaises(ValueError):
                e.validate_references(wrong,evidence["cases"])
        reference_path = self.root/"references.json"
        reference_path.write_text(json.dumps(reference))
        report = self.run_eval("referenced",max_cases=2,per_page=1,reference=reference_path)
        self.assertEqual(report["provisional_visual_reference_cases"],1)
        self.assertFalse(report["candidate_review_or_acceptance_performed"])

    def test_stale_foreign_duplicate_references_and_membership_reject(self):
        self.run_eval(max_cases=2,per_page=1)
        cases = json.loads((self.root/"output/evidence.json").read_text())["cases"]
        reference = json.loads((self.root/"output/structural-reference.json").read_text())
        foreign,stale = copy.deepcopy(reference),copy.deepcopy(reference)
        foreign["records"][0]["case_id"] = "foreign"
        stale["records"][0]["binding"]["crop_bounds"] = [0,0,1,1]
        for wrong in (dict(reference,records=reference["records"]*2),foreign,stale):
            with self.assertRaises(ValueError):
                e.validate_references(wrong,cases)
        self.artifact["pages"][0]["units"][0]["proposal"]["member_ids"] = [True]
        self.path.write_text(json.dumps(self.artifact))
        with self.assertRaises(ValueError):
            self.run_eval("invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_stale_images_invalid_bounds_and_new_only_output(self):
        self.run_eval()
        with self.assertRaises(FileExistsError):
            self.run_eval()
        for max_cases,per_page in ((0,3),(65,3),(21,0),(True,3)):
            with self.assertRaises(ValueError):
                self.run_eval("invalid-settings",max_cases=max_cases,per_page=per_page)
        Image.new("RGB",(100,100),"black").save(self.root/"original.png")
        with self.assertRaisesRegex(ValueError,"stale original"):
            self.run_eval("stale")
        self.assertFalse((self.root/"stale").exists())

    def test_cli_and_no_production_provider_imports_or_network(self):
        tree = ast.parse(Path(e.__file__).read_text())
        modules = [node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any("tralens" in module or "diagnostics" in module for module in modules))
        with patch.object(socket.socket,"connect",side_effect=AssertionError("network forbidden")), \
             patch.object(socket,"create_connection",side_effect=AssertionError("network forbidden")):
            self.assertEqual(e.main(["--artifact",str(self.path),"--output-dir",str(self.root/"cli"),
                                     "--max-cases","2","--per-page","1"]),0)
        self.assertEqual(json.loads((self.root/"cli/evaluation.json").read_text())["provider_calls"],0)
