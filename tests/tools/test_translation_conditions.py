"""Source-only A/B/C experiments with synthetic pixels and fake transports."""
import copy
import hashlib
import json
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.translation.test_previews import fixture, response
from tests.ocr import test_automatic_recognition as recognition_tests
from tralens.ocr.automatic_units import detection_evidence_sha256
from tralens.translation import CapabilityProfile, PreviewTranslator, OllamaRuntime, TRANSLATEGEMMA
from tralens.translation.contracts import TranslationRequest, canonical
from tralens.translation.runtime import compile_source, compact_unit
from tools import translation_diagnostics as d


def ordered_manifest():
    batch, _, _ = fixture()
    detections, proposals, regions = [], [], []
    # Deliberately nonlexicographic IDs; text neighbours must follow spatial order.
    for index, name in enumerate(("z", "b", "y", "a", "x")):
        raw = replace(batch.detections[0], source_id=index, text="読み" + str(index))
        bounds = (index * 50, 20, index * 50 + 20, 90)
        p = replace(batch.proposals[0], proposal_id=name, member_ids=(index,), main_ids=(index,),
                    annotation_ids=(), crop_bounds=bounds)
        detections.append(raw)
        proposals.append(p)
        regions.append((index, "bubble", "synthetic"))
    batch = replace(batch, detections=tuple(detections), proposals=tuple(proposals), regions=tuple(regions),
                    evidence_sha256=detection_evidence_sha256(detections))
    return {"pages": [{"batch": asdict(batch)}]}


class ContextTests(unittest.TestCase):
    def test_saved_order_and_explicit_selection_order_not_hash_order(self):
        manifest = ordered_manifest()
        before = copy.deepcopy(manifest)
        requests, _ = d.preflight(manifest, max_units=5)
        self.assertEqual([r.target_ids[0] for r in requests], ["z", "b", "y", "a", "x"])
        requests, _ = d.preflight(manifest, selected_ids=("x", "b", "z"), max_units=3, mode="B")
        self.assertEqual([r.target_ids[0] for r in requests], ["x", "b", "z"])
        self.assertEqual(requests[0].page_order, ("z", "b", "y", "a", "x"))
        self.assertIn("unverified", requests[0].page_order_status)
        self.assertEqual(manifest, before)

    def test_neighbour_priority_compact_budget_and_preserved_output_order(self):
        manifest = ordered_manifest()
        request = d.preflight(manifest, selected_ids=("y",), max_units=1, mode="B", max_context_chars=0)[0][0]
        all_units = d.preflight(manifest, max_units=5)[0]
        neighbour = all_units[1].units[0]
        budget = len(canonical([compact_unit(neighbour, request.page_order)]))
        request = d.preflight(manifest, selected_ids=("y",), mode="B", max_context_chars=budget)[0][0]
        self.assertEqual(request.context_ids, ("b",))
        request = d.preflight(manifest, selected_ids=("y",), mode="B", max_context_chars=budget * 2 + 2)[0][0]
        self.assertEqual(request.context_ids, ("b", "a"))
        compact = compile_source(request)
        self.assertEqual([u["id"] for u in compact["utterances"]], ["b", "y", "a"])
        text = canonical(compact)
        for field in ("polygon", "evidence_json", "model_settings_json", "image_path", "confidence"):
            self.assertNotIn(field, text)
        self.assertIn('"speaker":"unknown"', text)
        self.assertIn('"gender":"unknown"', text)
        self.assertIn("lexical_uncertainty", text)
        self.assertLess(len(text), len(canonical(asdict(request))))

    def test_alternatives_remain_compact_and_explicit_order_is_bound(self):
        _, _, units = fixture()
        compact = compact_unit(units[0])
        self.assertEqual([r["text"] for r in compact["readings"]], [r.text for r in units[0].readings])
        manifest = ordered_manifest()
        manifest["pages"][0]["translation_page_order"] = {
            "unit_ids": ["x", "a", "y", "b", "z"], "status": "parent_supplied_order_uncertain"}
        request = d.preflight(manifest, max_units=1)[0][0]
        self.assertEqual(request.target_ids, ("x",))
        self.assertEqual(request.page_order_status, "parent_supplied_order_uncertain")
        self.assertNotEqual(request.revision, replace(request, page_order=tuple(reversed(request.page_order))).revision)

    def test_seven_page_matched_selection_is_identical_across_conditions(self):
        pages, selection = [], []
        for index in range(7):
            page = ordered_manifest()["pages"][0]
            batch = page["batch"]
            for p in batch["proposals"]:
                p["proposal_id"] = str(index) + p["proposal_id"]
            pages.append(page)
            selection.append(str(index) + "y")
        manifest = {"pages": pages}
        for mode in ("A", "B", "C"):
            requests, _ = d.preflight(manifest, selected_ids=tuple(reversed(selection)), max_units=7, mode=mode)
            self.assertEqual([r.target_ids[0] for r in requests], list(reversed(selection)))


class EvidenceTests(unittest.TestCase):
    def run_fixture(self, directory, transport, **options):
        root = Path(directory)
        manifest = root / "saved.json"
        manifest.write_text(json.dumps(ordered_manifest()), encoding="utf-8")
        runtime = OllamaRuntime("http://127.0.0.1:11434", transport=transport)
        summary = d.run(manifest, root / "out", runtime=runtime, run_local_model=True,
                        model_digest="synthetic", model_details="synthetic", max_units=1, **options)
        record = json.loads((root / "out" / "results.jsonl").read_text())
        return root, summary, record

    def test_metrics_and_raw_response_retained_without_semantic_score(self):
        raw = {"done": True, "message": {"content": response(("z",))}, "load_duration": 11,
               "prompt_eval_duration": 12, "eval_duration": 13, "total_duration": 14,
               "prompt_eval_count": 15, "eval_count": 16}
        with tempfile.TemporaryDirectory() as directory:
            root, summary, record = self.run_fixture(directory, Mock(return_value=raw))
            evidence = record["generation_evidence"][0]
            self.assertEqual(evidence["generation_metadata"]["eval_count"], 16)
            self.assertEqual(evidence["generation_metadata"]["load_duration"], 11)
            self.assertEqual(json.loads((root / "out" / evidence["raw_response_path"]).read_text()), raw)
            self.assertFalse(summary["semantic_quality_assessed"])

    def test_raw_retention_bound_invalid_output_and_timeout(self):
        raw = {"done": True, "message": {"content": "invalid output " * 500}, "eval_count": 100}
        with tempfile.TemporaryDirectory() as directory:
            root, _, record = self.run_fixture(directory, Mock(return_value=raw), max_response_bytes=1024)
            self.assertEqual(record["failure"], "invalid_output")
            evidence = record["generation_evidence"][0]
            self.assertTrue(evidence["truncated"])
            self.assertEqual((root / "out" / evidence["raw_response_path"]).stat().st_size, 1024)
        with tempfile.TemporaryDirectory() as directory:
            _, summary, record = self.run_fixture(directory, Mock(side_effect=TimeoutError("synthetic")))
            self.assertEqual((record["failure"], summary["attempts"]), ("timeout", 1))
            self.assertEqual(record["generation_evidence"][0]["status"], "timeout")

    def test_provider_incomplete_response_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, record = self.run_fixture(directory, Mock(return_value={"done": False, "eval_count": 2}))
            self.assertEqual(record["failure"], "provider")
            self.assertEqual(record["generation_evidence"][0]["generation_metadata"]["eval_count"], 2)

    def test_late_generation_retains_metadata_but_not_translations(self):
        _, _, units = fixture()
        request = TranslationRequest(units, (units[0].unit_id,))
        raw = {"done": True, "message": {"content": response(request.target_ids)}, "eval_count": 5}
        runtime = OllamaRuntime("http://127.0.0.1:11434", transport=Mock(return_value=raw))
        events = []
        with patch("tralens.translation.runtime.time.monotonic", side_effect=[0, 0, 61, 61]):
            result = PreviewTranslator(runtime).translate(request, observe=events.append)
        self.assertEqual((result.failure, result.translations), ("timeout", ()))
        self.assertEqual(json.loads(events[0]["response_json"])["eval_count"], 5)

    def test_preload_raw_response_and_load_timing_are_separate(self):
        raw = {"done": True, "load_duration": 30}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "saved.json"
            manifest.write_text(json.dumps(ordered_manifest()), encoding="utf-8")
            runtime = OllamaRuntime("http://127.0.0.1:11434", transport=Mock(return_value=raw))
            summary = d.run(manifest, root / "out", runtime=runtime, run_local_model=True,
                model_digest="test", model_details="test", max_units=1, max_attempts=1, preload=True)
            records = [json.loads(line) for line in (root / "out" / "results.jsonl").read_text().splitlines()]
            self.assertEqual(summary["attempts"], 1)
            self.assertEqual(records[0]["timing"]["load_duration_ns"], 30)
            self.assertEqual(records[0]["generation_evidence"]["generation_metadata"]["load_duration"], 30)
            self.assertEqual(records[1]["status"], "budget_exhausted")

    def test_cli_saved_model_and_selection_records_preserve_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "saved.json").write_text(json.dumps(ordered_manifest()), encoding="utf-8")
            (root / "selection.json").write_text('["x","b"]', encoding="utf-8")
            (root / "model.json").write_text(json.dumps({"model": "qwen3.8:latest", "digest": "a" * 64,
                "capabilities": ["vision"]}), encoding="utf-8")
            self.assertEqual(d.main(["--manifest", str(root / "saved.json"), "--output", str(root / "out"),
                "--selection", str(root / "selection.json"), "--model-record", str(root / "model.json"),
                "--max-units", "2"]), 0)
            plan = json.loads((root / "out" / "requests.json").read_text())
            self.assertEqual(plan["selected_ids"], ["x", "b"])
            self.assertEqual(plan["model_digest_parent_asserted"], "a" * 64)


class ImageTests(unittest.TestCase):
    def fixture(self, root):
        helper = recognition_tests.AutomaticRecognitionTests()
        image, content, batch = helper.fixture(text="こんにちは")
        path = root / "original.png"
        path.write_bytes(content)
        manifest = {"pages": [{"batch": asdict(batch)}]}
        request = d.preflight(manifest, mode="C", max_units=1)[0][0]
        return path, content, manifest, request

    def test_original_and_presentation_hashes_resize_and_nonmutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, content, _, request = self.fixture(Path(directory))
            bound, encoded = d.image_presentation(request, path, max_side=64)
            evidence = json.loads(bound.image_evidence_json)
            self.assertEqual(max(evidence["presentation_size"]), 64)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(request.image_evidence_json, "")
            self.assertNotEqual(bound.revision, request.revision)
            transport = Mock(return_value={"done": True, "message": {"content": response(request.target_ids)}})
            runtime = OllamaRuntime("http://127.0.0.1:11434", transport=transport)
            profile = CapabilityProfile("synthetic_vision", True, True, True)
            self.assertEqual(PreviewTranslator(runtime, profile=profile).translate(bound, images=(encoded,)).status, "preview")
            self.assertEqual(PreviewTranslator(runtime, profile=profile).translate(bound, images=("YmFk",)).failure, "capability")
            self.assertEqual(PreviewTranslator(runtime).translate(bound, images=(encoded,)).failure, "capability")
            path.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "file hash"):
                d.image_presentation(request, path)

    def test_C_preflight_requires_explicit_capability_and_original_then_binds_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, content, manifest, request = self.fixture(root)
            saved = root / "saved.json"
            saved.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image-capable"):
                d.run(saved, root / "out", mode="C")
            profile = CapabilityProfile("synthetic_vision", True, True, True)
            with self.assertRaisesRegex(ValueError, "explicit original"):
                d.run(saved, root / "out", mode="C", profile=profile)
            page_id = json.loads(request.units[0].page_json)["page_id"]
            d.run(saved, root / "out", mode="C", profile=profile, images={page_id: path})
            plan = json.loads((root / "out" / "requests.json").read_text())
            self.assertTrue(plan["requests"][0]["request"]["image_evidence_json"])
            image_path = next(iter(plan["image_presentations"].values()))
            self.assertTrue((root / "out" / image_path).is_file())
            self.assertEqual(path.read_bytes(), content)

    def test_blocked_C_never_reads_original_and_pixel_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, manifest, request = self.fixture(root)
            page = json.loads(request.units[0].page_json)
            page["pixels_sha256"] = "0" * 64
            unit = replace(request.units[0], page_json=canonical(page))
            with self.assertRaisesRegex(ValueError, "pixel hash"):
                d.image_presentation(replace(request, units=(unit,)), path)
            for proposal in manifest["pages"][0]["batch"]["proposals"]:
                proposal["crop_safe"] = False
            saved = root / "saved.json"
            saved.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(d, "image_presentation", side_effect=AssertionError("no image read")):
                summary = d.run(saved, root / "out", mode="C",
                                profile=CapabilityProfile("synthetic_vision", True, True, True))
            self.assertEqual(summary["counts"], {"blocked": 1})


class SpecialistTests(unittest.TestCase):
    def test_parent_supplied_literal_template_identity_and_no_recursive_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.json"
            template = "synthetic ONLY ${source_lang_code}->${target_lang_code}: ${text}"
            spec = {"schema_version": 1, "encoding": "literal_substitution_v1", "model": "test",
                    "model_digest": "digest", "verified_by": "parent", "verification_source": "synthetic test",
                    "template": template, "template_sha256": hashlib.sha256(template.encode()).hexdigest()}
            path.write_text(json.dumps(spec), encoding="utf-8")
            encoder, loaded = d.load_specialist_template(path, model="test", model_digest="digest")
            self.assertEqual(encoder(source_lang_code="ja", target_lang_code="en", text="${text}"),
                             "synthetic ONLY ja->en: ${text}")
            with self.assertRaises(ValueError):
                d.load_specialist_template(path, model="other", model_digest="digest")
            saved = Path(directory) / "saved.json"
            saved.write_text(json.dumps(ordered_manifest()), encoding="utf-8")
            transport = Mock(return_value={"done": True, "response": "Hello"})
            def runtime_factory(endpoint, *, specialist_template):
                return OllamaRuntime(endpoint, transport=transport, specialist_template=specialist_template)
            with patch.object(d, "OllamaRuntime", side_effect=runtime_factory):
                summary = d.run(saved, Path(directory) / "out", run_local_model=True, model="test",
                    model_digest="digest", model_details="synthetic", profile=TRANSLATEGEMMA,
                    specialist_template=path, max_units=1)
            self.assertEqual(summary["counts"], {"preview": 1})
            self.assertEqual(transport.call_args.args[0], "/api/generate")
            self.assertIn("synthetic ONLY ja->en:", transport.call_args.args[1]["prompt"])
