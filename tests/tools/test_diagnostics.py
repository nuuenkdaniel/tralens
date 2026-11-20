"""Saved synthetic inputs and injected transport only; no live model requests."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from PIL import Image
from tools import diagnostics as d


def saved_box(source_id, bounds, text="text", status="review"):
    a, b, c, e = bounds
    return {"id": source_id, "polygon": [[a, b], [c, b], [c, e], [a, e]],
            "paddle": text, "selected": text, "score": .3, "status": status,
            "reason": "synthetic", "metadata": {"source_box_id": source_id}}


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        for name in ("connect",):
            guard = patch.object(socket.socket, name, side_effect=AssertionError("network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        for name in ("create_connection", "getaddrinfo"):
            guard = patch.object(socket, name, side_effect=AssertionError("network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image_path = self.root / "original.png"
        image = Image.new("RGB", (100, 80), "white")
        image.putpixel((11, 12), (17, 42, 99))
        image.save(self.image_path)
        self.boxes = [saved_box(-2, (10, 10, 20, 40), "A", "accepted"),
                      saved_box(4, (25, 10, 35, 40), "B", "review"),
                      saved_box(9, (60, 10, 75, 40), "", "rejected")]
        self.comparison = self.root / "page-comparison.json"
        self.comparison.write_text(json.dumps({"summary": {"image": "original.png"},
                                               "boxes": self.boxes}), encoding="utf-8")
        self.original_json = self.comparison.read_bytes()
        self.original_image = self.image_path.read_bytes()

    def run_review(self, name="review", **kwargs):
        return d.run_review([self.comparison], self.root / name, image_root=self.root,
                            max_cases=3, **kwargs)

    def transport(self, relation="independent", ids=None):
        def request(**kwargs):
            # Infer only the IDs from the generated prompt, never classify images.
            ids_from_prompt = json.loads(kwargs["prompt"].split("\nSupplied saved evidence:\n")[1])
            pair = ids if ids is not None else [entry["source_id"] for entry in ids_from_prompt]
            return {"model": "test-only", "message": {"content": json.dumps(
                {"source_ids": pair, "relationship": relation, "reason": "synthetic response"})}}
        return Mock(side_effect=request)

    def test_import_has_no_pipeline_network_file_or_gui_side_effects(self):
        spec = importlib.util.spec_from_file_location("tools.diagnostics_import_probe", Path(d.__file__))
        module = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in ("tralens.ocr.detection", "paddleocr", "manga_ocr", "ollama",
                        "tralens.translation.translator", "tralens.rendering.renderer"):
                raise AssertionError(f"pipeline import forbidden: {name}")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import), \
             patch.object(Path, "open", side_effect=AssertionError("file side effect")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("directory side effect")), \
             patch.object(Image, "open", side_effect=AssertionError("image side effect")), \
             patch.object(Image.Image, "show", side_effect=AssertionError("GUI forbidden")):
            spec.loader.exec_module(module)
        self.assertEqual(module.Relationship.INDEPENDENT, "independent")

    def test_pipeline_modules_do_not_import_diagnostics_or_keep_demo_entrypoints(self):
        source = Path(d.__file__).resolve().parents[1] / "src" / "tralens"
        for path in source.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all("diagnostics" not in alias.name for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("diagnostics", node.module or "")
                if isinstance(node, ast.If):
                    self.assertNotIn("__name__", ast.unparse(node.test))
        tree = ast.parse((source / "rendering" / "renderer.py").read_text())
        self.assertEqual([node.name for node in tree.body if isinstance(node, ast.FunctionDef)],
                         ["cover_text", "draw_text"])

    def test_standalone_visualization_saves_headless_without_mutating_groups(self):
        boxes = [SimpleNamespace(box_id=b["id"], poly=b["polygon"], status=b["status"],
                                 text=b["selected"], reason=b["reason"]) for b in self.boxes]
        groups = [SimpleNamespace(group=boxes), SimpleNamespace(group=[])]
        before = copy.deepcopy(groups)
        with patch.object(Image.Image, "show", side_effect=AssertionError("GUI forbidden")):
            d.visualize_groups(self.image_path, groups, self.root / "rectangles.png")
            d.visualize_groups_polygon(self.image_path, groups, self.root / "polygons.png")
        self.assertEqual(groups, before)
        for filename in ("rectangles.png", "polygons.png"):
            with Image.open(self.root / filename) as image:
                self.assertEqual(image.size, (100, 80))
                self.assertEqual(image.mode, "RGB")
        self.assertEqual(self.image_path.read_bytes(), self.original_image)
        self.assertEqual(d.debug_color("accepted"), "green")
        self.assertEqual(d.debug_color("review"), "orange")
        self.assertEqual(d.debug_color("rejected"), "red")

    def test_visualization_refuses_input_or_existing_output_overwrite(self):
        with self.assertRaises(FileExistsError):
            d.visualize_groups(self.image_path, [self.boxes], self.image_path)
        self.assertEqual(self.image_path.read_bytes(), self.original_image)

    def test_default_generates_original_annotated_crops_sheets_without_requests(self):
        request = Mock(side_effect=AssertionError("model not enabled"))
        manifest = self.run_review(requester=request)
        request.assert_not_called()
        self.assertEqual(manifest["request_count"], 0)
        self.assertEqual(manifest["case_count"], 3)
        self.assertTrue(manifest["advisory_only"])
        for record in manifest["cases"]:
            self.assertEqual(record["resolution"], "unresolved")
            self.assertEqual(record["error"], "model_disabled")
            self.assertEqual(len(record["image_paths"]), 2)
            bounds = record["crop_bounds"]
            with Image.open(self.image_path) as original, Image.open(record["image_paths"][0]) as crop:
                self.assertEqual(crop.tobytes(), original.crop(bounds).tobytes())
        self.assertTrue(all(Path(path).is_file() for path in manifest["review_sheets"]))
        self.assertEqual(self.comparison.read_bytes(), self.original_json)

    def test_request_bound_provenance_and_negative_relationships_are_retained(self):
        request = self.transport()
        manifest = self.run_review(run_model=True, requester=request, max_requests=1,
                                   endpoint="http://localhost:11434", model="test-only")
        self.assertEqual(manifest["request_count"], 1)
        self.assertEqual(request.call_count, 1)
        first = manifest["cases"][0]
        self.assertEqual(first["source_ids"], [-2, 4])
        self.assertEqual(first["relationship"], "independent")
        self.assertEqual(first["resolution"], "advisory")
        self.assertEqual(first["source_boxes"], self.boxes[:2])
        self.assertEqual(first["comparison_sha256"], hashlib.sha256(self.original_json).hexdigest())
        self.assertEqual(first["original_image_sha256"], hashlib.sha256(self.original_image).hexdigest())
        self.assertEqual(first["model"], "test-only")
        self.assertEqual(first["endpoint"], "http://localhost:11434")
        self.assertIs(first["think"], False)
        self.assertIs(manifest["settings"]["think"], False)
        self.assertEqual(first["raw_response"]["model"], "test-only")
        self.assertIn("uncertain", first["prompt"])
        self.assertEqual(first["options"]["temperature"], 0)
        self.assertGreaterEqual(first["timings"]["request_seconds"], 0)
        self.assertEqual([r["error"] for r in manifest["cases"][1:]], ["request_limit", "request_limit"])
        ledger = [json.loads(line) for line in (self.root / "review" / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(ledger), 3)
        self.assertEqual(ledger[0]["raw_response"], first["raw_response"])
        self.assertTrue(all(row["think"] is False for row in ledger))
        self.assertEqual(self.comparison.read_bytes(), self.original_json)
        self.assertEqual(self.image_path.read_bytes(), self.original_image)

    def test_ollama_payload_disables_thinking_at_top_level_without_network(self):
        opener = MagicMock()
        response = Mock(status=200)
        response.read.return_value = b'{"message":{"content":"synthetic"}}'
        opener.open.return_value.__enter__.return_value = response
        options = {"temperature": 0, "seed": 0, "num_predict": 256, "num_ctx": 4096}
        before = copy.deepcopy(options)
        with patch.object(d, "build_opener", return_value=opener):
            result = d._ollama_request(
                endpoint="http://localhost:11434", model="test-only", prompt="synthetic prompt",
                image_paths=[str(self.image_path)], options=options, timeout=7)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertIs(payload["think"], False)
        self.assertNotIn("think", payload["options"])
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["model"], "test-only")
        self.assertEqual(payload["options"], before)
        self.assertEqual(options, before)
        self.assertEqual(payload["messages"][0]["content"], "synthetic prompt")
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 7)
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["body"], response.read.return_value.decode())

    def test_all_relationship_enums_and_uncertain_resolution(self):
        for relation in d.Relationship:
            response = json.dumps({"source_ids": [-2, 4], "relationship": relation.value})
            self.assertEqual(d.validate_relationship_response(response, [-2, 4])["relationship"], relation.value)
        manifest = self.run_review(run_model=True, requester=self.transport("uncertain"), max_requests=1,
                                   endpoint="http://localhost:11434", model="test-only")
        first = manifest["cases"][0]
        self.assertEqual(first["resolution"], "unresolved")
        self.assertIsNone(first["error"])

    def test_invalid_missing_foreign_and_coerced_model_ids_or_enum_rejected(self):
        invalid = ["not json", '```json\n{}\n```', "[]", "{}",
                   '{"source_ids": [-2,4]}', '{"relationship":"furigana"}',
                   '{"source_ids": [-2,999],"relationship":"duplicate"}',
                   '{"source_ids": [4,-2],"relationship":"duplicate"}',
                   '{"source_ids": ["-2",4],"relationship":"duplicate"}',
                   '{"source_ids": [-2.0,4],"relationship":"duplicate"}',
                   '{"source_ids": [true,4],"relationship":"duplicate"}',
                   '{"source_ids": [-2,4,9],"relationship":"duplicate"}',
                   '{"source_ids": [-2,4],"relationship":"noise"}',
                   '{"source_ids": [-2,4],"relationship":"independent","reason":42}',
                   '{"source_ids": [-2,4],"relationship":"duplicate","source_ids":[4,9]}',
                   '{"source_ids": [-2,4],"relationship":"duplicate","extra":"yes"}']
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                d.validate_relationship_response(response, [-2, 4])

    def test_invalid_model_json_is_stored_raw_and_unresolved(self):
        response = {"message": {"content": '{"source_ids":[-2,99],"relationship":"furigana"}'}}
        manifest = self.run_review(run_model=True, requester=Mock(return_value=response), max_requests=1,
                                   endpoint="http://localhost:11434", model="test-only")
        first = manifest["cases"][0]
        self.assertEqual(first["raw_response"], response)
        self.assertEqual(first["relationship"], "uncertain")
        self.assertEqual(first["resolution"], "unresolved")
        self.assertIn("IDs", first["error"])

    def test_transport_errors_count_toward_bound_and_remain_unresolved(self):
        request = Mock(side_effect=TimeoutError("synthetic timeout"))
        manifest = self.run_review(run_model=True, requester=request, max_requests=1,
                                   endpoint="http://localhost:11434", model="test-only")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(manifest["request_count"], 1)
        self.assertIn("TimeoutError", manifest["cases"][0]["error"])
        self.assertEqual(manifest["cases"][0]["resolution"], "unresolved")

    def test_zero_request_bound_never_calls_injected_transport_even_when_enabled(self):
        request = self.transport()
        manifest = self.run_review(run_model=True, requester=request, max_requests=0,
                                   endpoint="http://localhost:11434", model="test-only")
        request.assert_not_called()
        self.assertEqual(manifest["request_count"], 0)
        self.assertTrue(all(row["error"] == "request_limit" for row in manifest["cases"]))

    def test_saved_duplicate_or_inconsistent_source_ids_fail_before_output(self):
        for boxes in ([self.boxes[0], self.boxes[0]],
                      [dict(self.boxes[0], metadata={"source_box_id": "-2"})]):
            self.comparison.write_text(json.dumps({"input": "original.png", "boxes": boxes}))
            with self.assertRaises(ValueError):
                self.run_review()
        self.assertFalse((self.root / "review").exists())

    def test_http_invalid_json_and_error_responses_preserved(self):
        for index, raw in enumerate(({"http_status": 500, "body": "model error"},
                                      {"http_status": 200, "body": "not json"},
                                      {"error": "missing model"}, {"message": {}})):
            manifest = self.run_review(name=f"http-{index}", run_model=True, requester=Mock(return_value=raw),
                                       max_requests=1, endpoint="http://localhost:11434", model="test-only")
            first = manifest["cases"][0]
            self.assertEqual(first["raw_response"], raw)
            self.assertEqual(first["resolution"], "unresolved")
            self.assertIsNotNone(first["error"])

    def test_explicit_selection_retains_rejected_negative_case_and_ids(self):
        manifest = self.run_review(selections=["page-comparison:-2:9"], context_padding=0)
        self.assertEqual(manifest["case_count"], 1)
        first = manifest["cases"][0]
        self.assertEqual(first["source_ids"], [-2, 9])
        self.assertEqual(first["source_boxes"][1]["status"], "rejected")
        self.assertEqual(first["crop_bounds"], (10, 10, 75, 40))

    def test_saved_input_schema_and_round_robin_multiple_pages(self):
        second = self.root / "second.json"
        second.write_text(json.dumps({"input": "original.png", "boxes": self.boxes}))
        pages = d.load_comparisons([self.comparison, second], self.root)
        cases = d.select_cases(pages, [], 3)
        self.assertEqual([page["name"] for page, _ in cases], ["page-comparison", "second", "page-comparison"])
        self.assertEqual(pages[1]["image_path"], str(self.image_path))

    def test_output_cannot_overwrite_input_or_existing_artifacts(self):
        with self.assertRaises(FileExistsError):
            d.run_review([self.comparison], self.root, image_root=self.root)
        self.run_review()
        previous = (self.root / "review" / "review.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_review()
        self.assertEqual((self.root / "review" / "review.json").read_bytes(), previous)
        self.assertEqual(self.comparison.read_bytes(), self.original_json)

    def test_empty_missing_image_crop_error_is_unresolved_without_requests(self):
        self.image_path.unlink()
        request = self.transport()
        manifest = self.run_review(run_model=True, requester=request,
                                   endpoint="http://localhost:11434", model="test-only")
        request.assert_not_called()
        self.assertEqual(manifest["request_count"], 0)
        self.assertTrue(all("FileNotFoundError" in row["error"] for row in manifest["cases"]))
        self.assertEqual(manifest["cases"][0]["source_ids"], [-2, 4])

    def test_explicit_endpoint_model_and_hard_bounds_required(self):
        for kwargs in ({"run_model": True}, {"max_requests": 65}, {"max_requests": -1},
                       {"max_cases": 0}, {"max_cases": 257}, {"context_padding": 129},
                       {"endpoint": "http://user:password@localhost:11434"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                d.run_review([self.comparison], self.root / "never-created", image_root=self.root, **kwargs)
        self.assertFalse((self.root / "never-created").exists())

    def test_cli_default_is_headless_and_model_disabled(self):
        with patch.object(d, "_ollama_request", side_effect=AssertionError("model not enabled")), \
             patch.object(Image.Image, "show", side_effect=AssertionError("GUI forbidden")):
            self.assertEqual(d.main(["--comparison", str(self.comparison), "--image-root", str(self.root),
                                     "--output-dir", str(self.root / "cli"), "--max-cases", "1"]), 0)
        manifest = json.loads((self.root / "cli" / "review.json").read_text())
        self.assertEqual(manifest["request_count"], 0)


if __name__ == "__main__":
    unittest.main()
