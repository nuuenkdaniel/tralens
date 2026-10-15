"""One-image workflow with synthetic pixels, local fake exports and injected providers only."""
import ast
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.ocr import test_automatic_recognition as recognition_tests
from tralens.ocr.automatic_units import propose_page_from_boxes
from tralens.ocr.automatic_recognition import recognize_automatic_batch
from tralens.ocr.types import Text_Box
from tralens.translation.contracts import canonical
from tralens.translation.runtime import Generation, GenerationError, OllamaRuntime
from tralens.translation import workflow as w


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "models"
        for name in ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"):
            directory = self.weights / name
            directory.mkdir(parents=True)
            (directory / "inference.yml").write_text("Global:\n  model_name: " + name + "\n")
            (directory / "inference.json").write_text("{}")
            (directory / "inference.pdiparams").write_bytes(b"fake export, never loaded")
        helper = recognition_tests.AutomaticRecognitionTests()
        image, content, batch = helper.fixture(text="こんにちは", two=True)
        self.image = self.root / "original.png"
        self.image.write_bytes(content)
        self.content = content
        image.close()
        self.boxes = [Text_Box(d.source_id, d.confidence, d.polygon, d.text) for d in batch.detections]
        self.batch = propose_page_from_boxes(self.boxes, str(self.image), complementary_ownership=True, regroup_companions=True)
        self.engine = SimpleNamespace(raw_detections=[self.boxes], automatic_unit_proposals=[self.batch],
            paddle_provenance={"profile": "v6_medium", "synthetic": True},
            prediction_contract_reports=[{"status": "aligned"}], process_images=Mock(), ocr=SimpleNamespace(close=Mock()))
        self.factory = Mock(return_value=self.engine)

    def run_flow(self, name="out", **kwargs):
        return w.run_preview(self.image, self.root / name, device="cpu", model_root=self.weights,
            run_local_ocr=True, ocr_factory=self.factory, **kwargs)

    def provider(self, **kwargs):
        self.engine.ocr.close.assert_called_once()
        data = json.loads(kwargs["payload"]["messages"][1]["content"])
        text = canonical({"translations": [{"id": i, "translation": "Hello", "disposition": "preview",
            "reasons": [], "source_interpretation": "unverified"} for i in data["target_ids"]]})
        return Generation(text, canonical({"done": True, "message": {"content": text}, "eval_count": 7, "load_duration": 8}))

    def test_ocr_only_evidence_all_units_zero_model_construction_and_source_immutability(self):
        before = w._snapshot([w._raw_box(b) for b in self.boxes])
        batch_before = copy.deepcopy(asdict(self.batch))
        with patch("tralens.translation.runtime.OllamaRuntime", side_effect=AssertionError("no model runtime")), \
                patch.object(Text_Box, "transformed", side_effect=AssertionError("no display-box translation")):
            state = self.run_flow()
        self.assertEqual((state["status"], state["attempts"]), ("complete", 0))
        self.assertEqual((state["settings"]["model"], state["settings"]["profile"]), ("translategemma:27b", "translategemma"))
        self.assertEqual(state["unit_status_counts"], {"ocr_only": 2})
        self.assertEqual(state["ocr"]["raw_detections"][0], before)
        self.assertEqual(before, w._snapshot([w._raw_box(b) for b in self.boxes]))
        self.assertEqual(batch_before, asdict(self.batch))
        self.assertEqual(self.image.read_bytes(), self.content)
        self.assertEqual(state, json.loads((self.root / "out" / "preview.json").read_text()))
        self.engine.process_images.assert_called_once_with(propose_units=True, complementary_ownership=True, regroup_companions=True)
        self.factory.assert_called_once_with(str(self.image), device="cpu", model_root=str(self.weights),
                                            model_profile="v6_medium", enable_manga_ocr=False)
        self.engine.ocr.close.assert_called_once()

    def test_injected_generalist_results_bound_raw_retained_and_no_render(self):
        provider = Mock(side_effect=self.provider)
        with patch.object(Text_Box, "transformed", side_effect=AssertionError("never transform")):
            state = self.run_flow(run_local_model=True, model="qwen3.8:latest", profile="generalist", runtime=SimpleNamespace(generate=provider))
        self.assertEqual((state["status"], state["attempts"]), ("complete", 2))
        self.assertEqual(provider.call_count, 2)
        for record in state["units"]:
            self.assertEqual(record["status"], "preview")
            self.assertEqual(record["result"]["request_revision"], record["request_revision"])
            self.assertEqual(record["result"]["outcomes"][0]["reading_revision"], record["unit"]["selected_revision"])
            evidence = record["generation_evidence"][0]
            self.assertEqual(evidence["metadata"]["eval_count"], 7)
            self.assertTrue((self.root / "out" / evidence["raw_response_path"]).is_file())
            self.assertTrue(record["binding_current"])
        self.assertEqual(len((self.root / "out" / "results.jsonl").read_text().splitlines()), 2)
        tree = ast.parse(Path(w.__file__).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("tools" in name or "render" in name or "adapter" in name for name in imports))

    def test_omitted_model_and_profile_dispatch_native_27b_with_global_budget(self):
        transport = Mock(return_value={"done": True, "message": {"content": "Hello"}, "model": "translategemma:27b"})
        runtime = OllamaRuntime("http://127.0.0.1:11434", transport=transport)
        with patch("tralens.translation.runtime.OllamaRuntime", return_value=runtime) as factory:
            state = self.run_flow(run_local_model=True, max_attempts=1)
        self.assertEqual((state["status"], state["attempts"]), ("complete", 1))
        factory.assert_called_once_with("http://127.0.0.1:11434")
        path, body, _ = transport.call_args.args
        self.assertEqual((path, body["model"]), ("/api/chat", "translategemma:27b"))
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertNotIn("format", body)
        self.assertNotIn("raw", body)
        self.assertEqual(state["units"][0]["result"]["model"], "translategemma:27b")
        self.assertEqual(state["units"][1]["status"], "budget_exhausted")

    def test_workflow_default_does_not_change_legacy_perbox_generalist_default(self):
        from tralens.translation import Translate
        client = Mock()
        legacy = Translate(client=client)
        self.assertEqual(legacy.model, "qwen3.8:latest")
        client.chat.assert_not_called()

    def test_failure_spends_global_budget_next_unit_visible_and_no_retry(self):
        provider = Mock(side_effect=GenerationError("synthetic provider error", '{"done":false,"error":"synthetic"}'))
        state = self.run_flow(run_local_model=True, model="qwen3.8:latest", profile="generalist", max_attempts=1,
                              runtime=SimpleNamespace(generate=provider))
        self.assertEqual((state["status"], state["attempts"]), ("complete_with_failures", 1))
        self.assertEqual([r["status"] for r in state["units"]], ["failed", "budget_exhausted"])
        provider.assert_called_once()
        self.assertEqual(state["units"][0]["result"]["translations"], [])
        self.assertEqual(state["units"][0]["generation_evidence"][0]["status"], "provider_error")

    def test_selection_unchosen_and_native_specialist_exact_path(self):
        provider = Mock(return_value=Generation("Hello", '{"done":true,"message":{"content":"Hello"}}'))
        chosen = self.batch.proposals[1].proposal_id
        state = self.run_flow(run_local_model=True, model="translategemma:12b",
            selected_ids=(chosen,), runtime=SimpleNamespace(generate=provider))
        self.assertEqual([r["status"] for r in state["units"]], ["not_selected", "preview"])
        self.assertEqual(state["attempts"], 1)
        payload = provider.call_args.kwargs["payload"]
        self.assertEqual(len(payload["messages"]), 1)
        self.assertNotIn("format", payload)
        self.assertTrue(state["units"][1]["result"]["review_needed"])
        self.assertEqual(state["units"][1]["result"]["model"], "translategemma:12b")

    def test_blocked_blank_and_budget_zero_do_not_disappear_or_construct_runtime(self):
        self.boxes[0].paddle_text = self.boxes[0].text = ""
        from tralens.ocr.recognition_units import snapshot_detections
        from tralens.ocr.automatic_units import detection_evidence_sha256
        raw = snapshot_detections(self.boxes, self.batch.page)
        self.batch = replace(self.batch, detections=raw, evidence_sha256=detection_evidence_sha256(raw))
        self.engine.automatic_unit_proposals = [self.batch]
        with patch("tralens.translation.runtime.OllamaRuntime", side_effect=AssertionError("no eligible calls")):
            state = self.run_flow(run_local_model=True, model="translategemma:12b", profile="translategemma", max_attempts=0)
        blocked = next(r for r in state["units"] if self.boxes[0].paddle_box_id in r["unit"]["member_ids"])
        self.assertEqual(sorted(r["status"] for r in state["units"]), ["blocked", "budget_exhausted"])
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("blank_main_reading", blocked["blocking_reasons"])
        self.assertEqual(state["attempts"], 0)

    def test_page_drift_and_changed_input_during_generation_fail_closed(self):
        self.engine.automatic_unit_proposals = [replace(self.batch,
            page=replace(self.batch.page, image_sha256="0" * 64))]
        runtime = Mock()
        state = self.run_flow("wrong-page", run_local_model=True, model="qwen3.8:latest", profile="generalist", runtime=runtime)
        self.assertEqual(state["error"]["type"], "SourceChanged")
        runtime.generate.assert_not_called()
        self.engine.automatic_unit_proposals = [self.batch]
        self.engine.ocr.close.reset_mock()
        def mutate(**kwargs):
            result = self.provider(**kwargs)
            self.image.write_bytes(b"externally changed source")
            return result
        state = self.run_flow("drift", run_local_model=True, model="qwen3.8:latest", profile="generalist", runtime=SimpleNamespace(generate=mutate))
        self.assertEqual((state["status"], state["attempts"]), ("failed", 1))
        self.assertTrue(all(r["status"] == "stale_source" and not r["binding_current"] for r in state["units"]))
        self.assertIsNotNone(state["units"][0]["result"])

    def test_ocr_failure_retains_available_evidence_and_closes_resources(self):
        self.engine.process_images.side_effect = RuntimeError("synthetic prediction failure")
        state = self.run_flow(run_local_model=True, model="qwen3.8:latest", profile="generalist", runtime=Mock())
        self.assertEqual((state["status"], state["attempts"]), ("failed", 0))
        self.assertTrue(state["ocr"]["raw_detections"][0])
        self.engine.ocr.close.assert_called_once()

    def test_deleted_source_after_provider_invalidates_retained_result_binding(self):
        def remove_source(**kwargs):
            self.image.unlink()
            return Generation("Hello", '{"done":true,"message":{"content":"Hello"}}')
        state = self.run_flow(run_local_model=True, runtime=SimpleNamespace(generate=remove_source))
        self.assertEqual((state["status"], state["attempts"]), ("failed", 1))
        self.assertEqual(state["error"]["type"], "SourceChanged")
        self.assertTrue(all(r["status"] == "stale_source" and not r["binding_current"] for r in state["units"]))
        self.assertIsNotNone(state["units"][0]["result"])

    def test_interrupt_during_provider_counts_started_call_and_preserves_all_units(self):
        runtime = SimpleNamespace(generate=Mock(side_effect=KeyboardInterrupt()))
        state = self.run_flow(run_local_model=True, model="qwen3.8:latest", profile="generalist", runtime=runtime)
        self.assertEqual((state["status"], state["attempts"]), ("interrupted", 1))
        self.assertEqual([r["status"] for r in state["units"]], ["interrupted", "not_executed"])
        self.assertEqual(len(state["units"]), 2)
        self.engine.ocr.close.assert_called_once()

    def test_constructor_failure_artifact_and_argument_rejection_before_factory(self):
        for kwargs in ({"max_attempts": 21}, {"timeout": float("nan")}, {"model": "qwen3.8:latest"},
                       {"profile": "generalist"}, {"model": None}, {"endpoint": "http://example.com"}, {"num_predict": 0}):
            with self.assertRaises(ValueError):
                self.run_flow(**kwargs)
            self.factory.assert_not_called()
            self.assertFalse((self.root / "out").exists())
        self.factory.side_effect = RuntimeError("synthetic constructor failure")
        state = self.run_flow()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"]["type"], "RuntimeError")
        self.assertEqual(json.loads((self.root / "out" / "preview.json").read_text())["status"], "failed")

    def test_local_weights_device_and_ocr_optin_rejected_before_initialization(self):
        for options in ({"run_local_ocr": False, "device": "cpu", "model_root": self.weights},
                        {"run_local_ocr": True, "device": "automatic", "model_root": self.weights},
                        {"run_local_ocr": True, "device": "cpu", "model_root": self.root / "missing-models"}):
            with self.assertRaises((ValueError, OSError)):
                w.run_preview(self.image, self.root / "out", ocr_factory=self.factory, **options)
            self.factory.assert_not_called()
            self.assertFalse((self.root / "out").exists())

    def test_engine_can_release_raw_attributes_on_close_before_model_use(self):
        def close():
            del self.engine.raw_detections
            del self.engine.automatic_unit_proposals
        self.engine.ocr.close.side_effect = close
        state = self.run_flow(run_local_model=True, model="qwen3.8:latest", profile="generalist", max_attempts=1,
            runtime=SimpleNamespace(generate=self.provider))
        self.assertEqual(state["status"], "complete")
        self.assertTrue(state["ocr"]["raw_detections"])
        self.assertEqual(state["attempts"], 1)

    def test_exclusive_output_and_atomic_json_do_not_overwrite_existing_artifact(self):
        destination = self.root / "out"
        destination.mkdir()
        (destination / "keep.txt").write_text("user data")
        with self.assertRaises(FileExistsError):
            self.run_flow()
        self.factory.assert_not_called()
        self.assertEqual((destination / "keep.txt").read_text(), "user data")
        path = destination / "preview.json"
        w._atomic_json(path, {"original": True})
        with self.assertRaises(FileExistsError):
            w._atomic_json(path, {"overwrite": True})
        self.assertEqual(json.loads(path.read_text()), {"original": True})
        self.assertEqual(list(destination.glob(".preview-*")), [])

    def test_empty_ocr_and_unknown_selection_are_reviewable_without_model_calls(self):
        self.engine.raw_detections = [[]]
        self.engine.automatic_unit_proposals = [None]
        state = self.run_flow("empty", run_local_model=True, runtime=Mock())
        self.assertEqual((state["status"], state["units"], state["attempts"]), ("complete", [], 0))
        self.engine.raw_detections = [self.boxes]
        self.engine.automatic_unit_proposals = [self.batch]
        state = self.run_flow("selection", selected_ids=("foreign",))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(len(state["units"]), 2)

    def test_candidate_alternatives_are_retained_without_mutating_or_selecting_them(self):
        from PIL import Image
        with Image.open(self.image) as source:
            rgb = source.convert("RGB")
        results = recognize_automatic_batch(self.batch, rgb, lambda *a, **k: "こんばんは",
            original_image_bytes=self.content, model_identity="synthetic callback only")
        rgb.close()
        self.engine.automatic_recognition_results = (results,)
        before = w._snapshot(results)
        state = self.run_flow()
        self.assertEqual(w._snapshot(results), before)
        self.assertEqual(state["ocr"]["automatic_recognition_results"][0], before)
        self.assertTrue(all(len(r["unit"]["readings"]) == 2 for r in state["units"]))
        self.assertTrue(all(r["unit"]["readings"][0]["text"] == "こんにちは" for r in state["units"]))

    def test_raw_retention_bound_is_explicit_and_input_unchanged(self):
        def generate(**kwargs):
            normal = self.provider(**kwargs)
            raw = json.loads(normal.response_json)
            raw["padding"] = "x" * (1024 * 1024)
            return Generation(normal.text, canonical(raw))
        state = self.run_flow(run_local_model=True, model="qwen3.8:latest", profile="generalist", max_attempts=1,
            runtime=SimpleNamespace(generate=generate))
        evidence = state["units"][0]["generation_evidence"][0]
        self.assertTrue(evidence["truncated"])
        self.assertEqual((self.root / "out" / evidence["raw_response_path"]).stat().st_size, 1024 * 1024)
        self.assertEqual(self.image.read_bytes(), self.content)


class InertCLITests(unittest.TestCase):
    def test_cli_omitted_model_profile_and_explicit_overrides_dispatch(self):
        import main
        base = ["--image", "input.png", "--output", "fresh", "--device", "cpu", "--model-root", "local",
                "--run-local-ocr", "--run-local-model", "--max-attempts", "2"]
        for extra, model, profile in (([], "translategemma:27b", "translategemma"),
                (["--model", "translategemma:12b"], "translategemma:12b", "translategemma"),
                (["--model", "qwen3.8:latest", "--profile", "generalist"], "qwen3.8:latest", "generalist")):
            with patch.object(w, "run_preview", return_value={"status": "complete", "attempts": 2}) as run:
                self.assertEqual(main.main(base + extra), 0)
            self.assertEqual(run.call_args.kwargs["model"], model)
            self.assertEqual(run.call_args.kwargs["profile"], profile)
            self.assertTrue(run.call_args.kwargs["run_local_model"])

    def test_cli_incompatible_default_override_rejects_before_weight_or_engine_setup(self):
        import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            base = ["--image", "unused.png", "--output", str(output), "--device", "cpu", "--model-root", "unused",
                    "--run-local-ocr", "--run-local-model"]
            for extra in (["--model", "qwen3.8:latest"], ["--profile", "generalist"]):
                with patch("tralens.ocr.paddle_profiles.resolve_profile", side_effect=AssertionError("too late to reject")) as resolve:
                    with self.assertRaises(SystemExit) as rejected:
                        main.main(base + extra)
                self.assertEqual(rejected.exception.code, 2)
                resolve.assert_not_called()
                self.assertFalse(output.exists())

    def test_import_help_and_missing_optins_cannot_import_ocr_models_or_renderer(self):
        source = str(Path(__file__).resolve().parents[2] / "src")
        script = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tralens', 'paddleocr', 'paddlex', 'ollama', 'cv2', 'PIL', 'tools'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
import main
for args, expected in [(['--help'], 0),
    (['--image','unused','--output','unused','--device','cpu','--model-root','unused'], 2),
    (['--image','unused','--output','unused','--device','cpu','--model-root','unused','--run-local-model'], 2)]:
    try: main.main(args)
    except SystemExit as error: assert error.code == expected, error.code
    else: raise AssertionError('expected parser exit')
'''
        run = subprocess.run([sys.executable, "-I", "-c", script, source], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_cli_routes_explicit_flags_to_workflow_without_real_initialization(self):
        import main
        with patch.object(w, "run_preview", return_value={"status": "complete", "attempts": 0}) as run:
            result = main.main(["--image", "input.png", "--output", "fresh", "--device", "cpu", "--model-root", "local",
                "--run-local-ocr", "--model", "translategemma:12b", "--profile", "translategemma", "--max-attempts", "2"])
        self.assertEqual(result, 0)
        self.assertFalse(run.call_args.kwargs["run_local_model"])
        self.assertEqual(run.call_args.kwargs["max_attempts"], 2)
