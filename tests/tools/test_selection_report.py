"""Saved-artifact review metrics are synthetic offline checks, not measurements."""
import hashlib
import importlib.util
import io
import json
import socket
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from tools import diagnostics as d
from tralens.ocr import transcription_selection as s


class SelectionReportTests(unittest.TestCase):
    def setUp(self):
        for target, name in ((socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo")):
            guard = patch.object(target, name, side_effect=AssertionError("network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = Image.new("RGB", (100, 100), "white")
        self.image.putpixel((20, 30), (3, 7, 11))
        path = self.root / "original.png"
        self.image.save(path)
        page = d.units.PageProvenance("fixture", str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "2"*64, d.units.image_fingerprint(self.image))
        self.artifact = {"page_errors": [], "units": []}
        for index, text in enumerate(("ゃー！", "word!!", "uncertainーー")):
            unit_id = f"unit-{index}"
            crop = {"page": asdict(page), "unit_id": unit_id, "member_ids": [index], "bounds": [10, 20, 40, 60]}
            self.artifact["units"].append({"unit": {"plan": {"unit_id": unit_id, "member_ids": [index],
                "page": asdict(page), "errors": []}, "candidates": [{"text": text, "engine": "manga", "crop": crop}]},
                "raw_detections": [{"source_id": index, "polygon": [[12, 22], [30, 22], [30, 50], [12, 50]]}]})
        self.path = self.root / "units.json"
        self.path.write_text(json.dumps(self.artifact, ensure_ascii=False))

    def checked(self, template, **changes):
        record = dict(template, reviewed=True, wording_checked=True, order_checked=True,
                      full_utterance_checked=True, wording_uncertain=False, order_uncertain=False,
                      coverage_uncertain=False, punctuation_uncertain=False, reviewer="PM", source="visual original")
        record.update(changes)
        return record

    def preflight(self):
        result = d.run_selection_report(self.path, self.root / "preflight")
        template = json.loads((self.root / "preflight/selection-review-template.json").read_text())
        return result, template

    def run_review(self, document, name="reviewed"):
        review = self.root / (name + "-input.json")
        review.write_text(json.dumps(document, ensure_ascii=False))
        return d.run_selection_report(self.path, self.root / name, reviewed_input=review)

    def test_preflight_templates_counts_no_metrics_and_original_crop_equality(self):
        before = self.path.read_bytes()
        report, template = self.preflight()
        self.assertEqual(report["counts"]["reviewed_units"], 0)
        self.assertEqual(report["counts"]["unresolved_units"], 3)
        self.assertIsNone(report["metrics"]["cer"])
        self.assertEqual(report["metrics"]["reference_codepoint_denominator"], 0)
        self.assertFalse(template["reviews"][0]["reviewed"])
        self.assertTrue(template["reviews"][0]["punctuation_uncertain"])
        with Image.open(report["units"][0]["image_paths"][0]) as image:
            self.assertEqual(image.tobytes(), self.image.crop((10, 20, 40, 60)).tobytes())
        self.assertEqual(self.path.read_bytes(), before)

    def test_provisional_corrections_dash_uncertainty_and_independent_metrics_separate(self):
        before = self.path.read_bytes()
        _, document = self.preflight()
        document["reviews"][0] = self.checked(document["reviews"][0], mode="corrected_text", corrected_text="やー！",
            review_kind="independent_reference", independence_confirmed=True,
            reference_source="independent blind annotation", source="independent annotator")
        document["reviews"][1] = self.checked(document["reviews"][1], mode="corrected_text", corrected_text="word!")
        document["reviews"][2] = self.checked(document["reviews"][2], punctuation_uncertain=True)
        report = self.run_review(document)
        self.assertEqual(report["counts"], {"total_units": 3, "reviewed_units": 2, "unresolved_units": 1,
            "provisional_visual_units": 1, "independent_reference_units": 1})
        self.assertEqual(report["completeness"], {"fully_checked_units": 2, "coverage_claimed_units": 3,
                                                 "unverified_units": 1})
        self.assertEqual(report["metrics"]["total_edit_distance"], 1)
        self.assertEqual(report["metrics"]["reference_codepoint_denominator"], 3)
        self.assertEqual(report["metrics"]["cer"], 1/3)
        self.assertEqual(report["metrics"]["unit_coverage"], 1/3)
        self.assertEqual(report["units"][1]["selected"]["candidate"]["text"], "word!!")
        self.assertEqual(report["units"][1]["selected"]["text"], "word!")
        self.assertIsNone(report["units"][1]["metric"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_exact_metrics_preserve_punctuation_kana_dash_width_and_whitespace(self):
        for hypothesis, reference in (("ゃ", "や"), ("!", "！"), ("ーー", "ー"), ("a ", "a"),
                                      ("é", "e\u0301")):
            self.assertGreater(d.exact_edit_distance(hypothesis, reference), 0)
        self.assertEqual(d.exact_edit_distance("same", "same"), 0)

    def test_stale_artifact_hash_and_foreign_records_block_all_without_partial_selection(self):
        _, document = self.preflight()
        document["reviews"] = [self.checked(r) for r in document["reviews"]]
        for name, change in (("stale", {"units_artifact_sha256": "0"*64}),
                             ("foreign", {"reviews": document["reviews"] + [{"unit_id": "foreign"}]})):
            report = self.run_review(dict(document, **change), name)
            self.assertEqual(report["counts"]["reviewed_units"], 0)
            self.assertTrue(report["global_errors"])

    def test_duplicate_and_malformed_unit_reviews_remain_unresolved(self):
        _, document = self.preflight()
        record = self.checked(document["reviews"][0])
        document["reviews"] = [record, dict(record, reviewed="true")]
        report = self.run_review(document)
        self.assertEqual(report["counts"]["reviewed_units"], 0)
        document["reviews"] = [dict(record, unexpected="value")]
        malformed = self.run_review(document, "malformed")
        self.assertIn("invalid_review_record", malformed["units"][0]["error"])

    def test_saved_candidate_and_original_image_drift_are_rejected(self):
        _, document = self.preflight()
        document["reviews"] = [self.checked(document["reviews"][0])]
        self.artifact["units"][0]["unit"]["candidates"][0]["text"] = "changed"
        self.path.write_text(json.dumps(self.artifact))
        # Even rebinding the whole-file hash cannot fix the stale candidate ID.
        document["units_artifact_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        report = self.run_review(document)
        self.assertIsNone(report["units"][0]["selected"])
        Image.new("RGB", (100, 100), "black").save(self.root / "original.png")
        drift = self.run_review(document, "image-drift")
        self.assertTrue(all("stale_original_image_sha256" in r["error"] for r in drift["units"]))

    def test_bounds_existing_directory_cli_and_no_model_imports(self):
        with self.assertRaises(ValueError):
            d.run_selection_report(self.path, self.root / "limit", max_units=2)
        self.assertFalse((self.root / "limit").exists())
        self.assertEqual(d.main(["--selection-report", "--units-artifact", str(self.path),
                                  "--output-dir", str(self.root / "cli")]), 0)
        with self.assertRaises(FileExistsError):
            d.run_selection_report(self.path, self.root / "cli")
        with self.assertRaises(SystemExit), patch("sys.stderr", new=io.StringIO()):
            d.main(["--selection-report", "--units-artifact", str(self.path), "--output-dir", str(self.root / "bad"),
                    "--run-unit-model"])
        spec = importlib.util.spec_from_file_location("tralens.ocr.selection_import_probe", Path(s.__file__))
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.open", side_effect=AssertionError("import file writes/reads forbidden")), \
             patch.object(d.importlib, "import_module", side_effect=AssertionError("model import forbidden")):
            spec.loader.exec_module(module)

    def test_independent_metric_uses_original_candidate_not_corrected_output(self):
        _, document = self.preflight()
        document["reviews"] = [self.checked(document["reviews"][1], mode="corrected_text", corrected_text="word!",
            review_kind="independent_reference", independence_confirmed=True, reference_source="blind reference")]
        report = self.run_review(document)
        self.assertEqual(report["metrics"]["total_edit_distance"], 1)
        self.assertEqual(report["metrics"]["reference_codepoint_denominator"], 5)
        self.assertEqual(report["metrics"]["cer"], 0.2)
