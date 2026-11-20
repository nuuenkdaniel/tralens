"""Offline HTML report: portable real workflow contracts plus optional saved-page CLI cases."""
import copy
from dataclasses import replace
import hashlib
from html.parser import HTMLParser
from html import unescape
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from tests.translation import test_workflow as fixtures
from tests.translation.test_ocr_review import verdict
from tools import translation_review as report
from tralens.translation import workflow
from tralens.translation.contracts import canonical
from tralens.translation.runtime import Generation


ROOT = Path(__file__).resolve().parents[2]
ATTACK = '\"><script>alert(1)</script><svg onload="evil()">&'


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def cli(preview, image, output):
    # Deny network, engines, browser and subprocess creation inside the actual CLI process.
    code = '''
import runpy, socket, sys, unittest.mock as m, urllib.request, webbrowser
sys.path.insert(0, "src")
from tralens.translation.runtime import OllamaRuntime
def denied(*args, **kwargs): raise AssertionError("offline report attempted external operation")
with m.patch.object(socket, "socket", denied), m.patch.object(urllib.request, "urlopen", denied), \\
     m.patch.object(OllamaRuntime, "__init__", denied), m.patch.object(webbrowser, "open", denied), \\
     m.patch("subprocess.Popen", denied):
    sys.argv = ["tools/translation_review.py"] + sys.argv[1:]
    runpy.run_path("tools/translation_review.py", run_name="__main__")
'''
    return subprocess.run([sys.executable, "-c", code, "--preview", str(preview), "--image", str(image),
                           "--output", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=60)


class TranslationReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.n = 0

    def state(self, schema=1, **kwargs):
        self.n += 1
        self.fixture.engine.ocr.close.reset_mock()
        if schema >= 2:
            self.fixture.engine.automatic_unit_proposals = [replace(self.fixture.batch,
                proposals=tuple(replace(p, crop_safe=False) for p in self.fixture.batch.proposals))]
        def review(**kw):
            text = canonical(verdict(kw["payload"]))
            return Generation(text, canonical({"message": {"content": text}}))
        options = {"run_local_model": True, "max_attempts": 8,
                   "runtime": SimpleNamespace(generate=Mock(return_value=Generation("English " + ATTACK, '{"done":true}')))}
        if schema >= 2:
            options["allow_fragment_previews"] = True
        if schema == 3:
            options.update(review_ocr=True, review_runtime=SimpleNamespace(generate=review))
        options.update(kwargs)
        state = self.fixture.run_flow("flow-" + str(self.n), **options)
        self.assertEqual(state["schema_version"], schema)
        return state

    def validate(self, state):
        with Image.open(self.fixture.image) as source:
            rgb = source.convert("RGB")
        try:
            return report.validate(state, self.fixture.image, rgb,
                hashlib.sha256(self.fixture.content).hexdigest(), self.fixture.content)
        finally:
            rgb.close()

    def reject(self, state, mutate):
        damaged = copy.deepcopy(state)
        mutate(damaged)
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.validate(damaged)

    def test_all_schemas_actual_saved_cli_and_source_immutability(self):
        for schema in (1, 2, 3):
            with self.subTest(schema=schema):
                self.state(schema)
                path = self.root / ("flow-" + str(self.n)) / "preview.json"
                before = path.read_bytes()
                output = self.root / ("report-" + str(schema))
                run = cli(path, self.fixture.image, output)
                self.assertEqual(run.returncode, 0, run.stderr)
                summary = json.loads(run.stdout)
                self.assertEqual(summary["raw_detections"], 2)
                self.assertFalse(summary["missing_source_ids"])
                self.assertFalse(summary["rendering_approved"])
                html = (output / "review.html").read_text()
                parser = Tags()
                parser.feed(html)
                self.assertNotIn("script", [tag for tag, _ in parser.tags])
                self.assertFalse(any(key.lower().startswith("on") for _, attrs in parser.tags for key in attrs))
                self.assertIn("&lt;script&gt;", html)
                self.assertIn("Content-Security-Policy", html)
                for tag, attrs in parser.tags:
                    if "src" in attrs:
                        self.assertTrue(attrs["src"].startswith("data:image/png;base64,"))
                    if "href" in attrs:
                        self.assertTrue(attrs["href"].startswith("#unit-"))
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(self.fixture.image.read_bytes(), self.fixture.content)

    def test_import_inert_and_no_sidecar_paths_opened(self):
        code = '''
import pathlib, socket, sys, unittest.mock as m, urllib.request, webbrowser
sys.path.insert(0, "src")
def denied(*a, **kw): raise AssertionError("import performed I/O")
with m.patch.object(socket, "socket", denied), m.patch.object(urllib.request, "urlopen", denied), \\
     m.patch.object(pathlib.Path, "open", denied), m.patch.object(webbrowser, "open", denied):
    import tools.translation_review
assert "tralens.ocr.detection" not in sys.modules
assert "paddleocr" not in sys.modules
'''
        imported = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        state = self.state(3)
        saved = self.root / "saved.json"
        saved.write_text(canonical(state))
        opened = []
        real = Path.open
        def tracked(path, *args, **kwargs):
            opened.append(Path(path))
            return real(path, *args, **kwargs)
        with patch.object(Path, "open", tracked), patch("socket.socket", side_effect=AssertionError("network")), \
                patch.object(report.visual, "OllamaRuntime", side_effect=AssertionError("model")):
            report.run(saved, self.fixture.image, self.root / "report")
        reads = [p for p in opened if p not in {saved, self.fixture.image} and p.parent != self.root / "report"]
        self.assertEqual(reads, [])
        # Views and raw-response sidecars are deliberately absent next to saved.json.
        self.assertFalse((self.root / state["ocr_review"]["units"][0]["views"][0]["path"]).exists())

    def test_compact_summary_page_first_and_closed_metadata_preserve_evidence(self):
        state = self.state(3)
        before = copy.deepcopy(state)
        entries, summary = self.validate(state)
        with Image.open(self.fixture.image) as source:
            rgb = source.convert("RGB")
        try:
            html = report.build_html(state, entries, summary, rgb)
        finally:
            rgb.close()
        parser = Tags()
        parser.feed(html)
        self.assertTrue(all("open" not in attrs for tag, attrs in parser.tags if tag == "details"))
        self.assertIn('id="quick-summary"', html)
        page_position = html.index('<div class="page">')
        top = html[:page_position]
        self.assertIn("Workflow: complete", top)
        self.assertIn("Raw IDs: 2 · Units: 2", top)
        self.assertIn("Outputs — strict: 0 · fragment: 2 · alternate: 2", top)
        self.assertIn("Translation calls: 4/8", top)
        self.assertIn("Review calls: 2/4", top)
        self.assertIn("Release calls: 0 — caller_responsibility; no unload dispatched", top)
        self.assertIn('id="geometry"', top)
        self.assertNotIn("<pre>", top)
        for identifier in ("execution-details", "coverage-details"):
            self.assertLess(page_position, html.index('<details id="' + identifier + '">'))

        def retained_json(identifier):
            block = html.split('<details id="' + identifier + '">', 1)[1].split('</details>', 1)[0]
            return json.loads(unescape(block.split('<pre>', 1)[1].split('</pre>', 1)[0]))

        execution = retained_json("execution-details")
        self.assertEqual(execution["review"], state["ocr_review"])
        self.assertEqual(execution["input_schedule"], state["input_schedule"])
        self.assertEqual(retained_json("coverage-details"), summary)
        self.assertIn('<details><summary>Source readings, revisions and provenance', html)
        self.assertIn('"unit_revision":', html.replace('&quot;', '"'))
        self.assertIn('<h3>Blocking reasons</h3><pre>', html)
        self.assertIn("Original selected OCR (preserved):</p><pre>こんにちは</pre>", html)
        self.assertIn("こんにちは訂</pre>", html)
        self.assertIn("Review needed — no automatic acceptance.", html)
        self.assertIn('href="#unit-1"', html)
        self.assertIn('max-width:1300px;overflow-wrap:anywhere', html)
        self.assertEqual(state, before)

    def test_stale_file_pixels_path_batch_and_original_unit(self):
        state = self.state()
        mutations = [
            lambda s: s["input"].update(sha256="0" * 64),
            lambda s: s["input"].update(pixels_sha256="0" * 64),
            lambda s: s["input"].update(path="../../elsewhere.png"),
            lambda s: s["ocr"]["batches"][0].update(evidence_sha256="0" * 64),
            lambda s: s["ocr"]["batches"][0]["page"].update(comparison_sha256="0" * 64),
            lambda s: s["units"][0]["unit"].update(selected_revision="0" * 64),
            lambda s: s["units"][0].update(unit_revision="0" * 64),
            lambda s: s["units"][0]["unit"].update(member_ids=["foreign"]),
            lambda s: s["units"][0]["unit"].update(order=[True]),
            lambda s: s["ocr"]["raw_detections"][0][0].update(text="changed"),
            lambda s: s["units"].pop(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                self.reject(state, mutate)

    def test_request_result_and_counter_tampering(self):
        state = self.state()
        mutations = [
            lambda s: s["units"][0]["request"].update(mode="B"),
            lambda s: s["units"][0].update(request_revision="0" * 64),
            lambda s: s["units"][0]["result"].update(request_revision="0" * 64),
            lambda s: s["units"][0]["result"]["outcomes"][0].update(reading_revision="0" * 64),
            lambda s: s["units"][0].update(status="blocked"),
            lambda s: s["units"][0].update(binding_current=1),
            lambda s: s.update(attempts=True),
            lambda s: s.update(strict_attempts=0),
            lambda s: s.update(strict_preview_count=99),
            lambda s: s.update(unit_status_counts={"preview": 99}),
            lambda s: s["units"][0]["result"].update(preview_only=1),
            lambda s: s["units"][0]["result"].update(review_needed=1),
            lambda s: s["units"][0]["result"].update(model="foreign-model"),
            lambda s: s["units"][0]["result"].update(execution_metadata_json="{}"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                self.reject(state, mutate)

    def test_fragment_typed_bindings_and_blockers(self):
        state = self.state(2)
        self.validate(state)
        for field in ("input_revision", "request_revision"):
            self.reject(state, lambda s: s["units"][0]["fragment_previews"][0].update({field: "stale"}))
        self.reject(state, lambda s: s["units"][0]["fragment_previews"][0]["input"].update(source_ids=["foreign"]))
        self.reject(state, lambda s: s["units"][0]["fragment_previews"][0]["result"].update(review_needed=False))
        self.reject(state, lambda s: s["units"][0]["fragment_previews"][0]["result"].update(source_binding_json="{}"))
        self.reject(state, lambda s: s["units"][0].update(blocking_reasons=[]))
        self.reject(state, lambda s: s["units"][0].update(status="preview"))
        self.reject(state, lambda s: s.update(fragment_candidate_inputs=0))
        self.reject(state, lambda s: s["settings"].update(allow_fragment_previews=False))

    def test_visual_review_and_alternate_bindings(self):
        state = self.state(3)
        self.validate(state)
        for field in ("request_revision", "verdict_revision", "requested_http_body_sha256"):
            def mutate(s):
                s["ocr_review"]["units"][0][field] = "0" * 64
                s["units"][0]["ocr_review"][field] = "0" * 64
            self.reject(state, mutate)
        for field, value in (("source_ids", ["foreign"]), ("original_text", "changed"),
                             ("target_id", "foreign"), ("uncertainty", "confident")):
            def mutate(s):
                for row in (s["ocr_review"]["units"][0], s["units"][0]["ocr_review"]):
                    row["verdict"][field] = value
            self.reject(state, mutate)
        for field in ("context_bounds", "view_sha256", "view_pixels_sha256", "context_only_neighbor_ids"):
            def mutate(s):
                for row in (s["ocr_review"]["units"][0], s["units"][0]["ocr_review"]):
                    binding = json.loads(row["request"]["binding_json"])
                    binding[field] = ["foreign"] if field == "context_only_neighbor_ids" else []
                    row["request"]["binding_json"] = canonical(binding)
            self.reject(state, mutate)
        self.reject(state, lambda s: s["ocr_review"].update(attempts=0))
        self.reject(state, lambda s: s["ocr_review"].update(control_calls=1))
        child = lambda s: s["units"][0]["visual_alternates"][0]
        self.reject(state, lambda s: child(s).update(input_revision="stale"))
        self.reject(state, lambda s: child(s)["result"].update(request_revision="stale"))
        self.reject(state, lambda s: child(s)["input"]["verdict"].update(proposed_text="replaced"))
        self.reject(state, lambda s: child(s)["result"]["outcomes"][0].update(reading_revision="stale"))
        self.reject(state, lambda s: s["units"][0].update(visual_alternates=[]))
        self.reject(state, lambda s: s["input_schedule"][0].update(request_revision="stale"))

    def test_mixed_typed_ids_and_svg_text_are_preserved_and_escaped(self):
        for ids in ((1, "1"), (1, ATTACK)):
            for box, identifier in zip(self.fixture.boxes, ids):
                box.paddle_box_id = box.box_id = identifier
                box.paddle_text = box.text = ATTACK
                box.ocr_metadata = {"candidates": {"paddle": ATTACK}, "fallback_attempts": 0}
            self.fixture.batch = fixtures.propose_page_from_boxes(self.fixture.boxes, str(self.fixture.image),
                complementary_ownership=True, regroup_companions=True)
            self.fixture.engine.automatic_unit_proposals = [self.fixture.batch]
            state = self.state()
            state["error"] = {"message": ATTACK}
            state["status"], state["stage"] = "failed", "translation"
            entries, summary = self.validate(state)
            with Image.open(self.fixture.image) as source:
                rgb = source.convert("RGB")
            try:
                html = report.build_html(state, entries, summary, rgb, crops=True)
            finally:
                rgb.close()
            self.assertEqual(summary["raw_detections"], 2)
            self.assertIn(ids[0], summary["main_source_ids"])
            self.assertIn(ids[1], summary["main_source_ids"])
            parser = Tags()
            parser.feed(html)
            self.assertNotIn("script", [tag for tag, _ in parser.tags])
            self.assertFalse(any(k.startswith("on") for _, attrs in parser.tags for k in attrs))
            self.assertNotIn(ATTACK, html)
            self.assertIn("&lt;script&gt;", html)

    def test_visual_dispositions_failed_cancelled_and_release_failure(self):
        for decision in ("unchanged", "uncertain", "unreadable"):
            def review(**kw):
                text = canonical(verdict(kw["payload"], decision))
                return Generation(text, canonical({"message": {"content": text}}))
            state = self.state(3, review_runtime=SimpleNamespace(generate=review), max_review_attempts=1)
            self.validate(state)
            self.assertEqual(state["ocr_review"]["units"][0]["status"], "review_" + decision)
        failed = self.state(3, review_runtime=SimpleNamespace(generate=Mock(side_effect=TimeoutError("synthetic"))))
        self.validate(failed)
        flag = {"cancel": False}
        def cancel(**kw):
            flag["cancel"] = True
            return "invalid output"
        cancelled = self.state(3, review_runtime=SimpleNamespace(generate=cancel), cancelled=lambda: flag["cancel"])
        self.validate(cancelled)
        def correction(**kw):
            text = canonical(verdict(kw["payload"]))
            return Generation(text, canonical({"message": {"content": text}}))
        provider = SimpleNamespace(generate=correction, unload=Mock(side_effect=TimeoutError("synthetic")))
        with patch.object(report.visual, "OllamaRuntime", return_value=provider):
            release_failed = self.state(3, review_runtime=None)
        self.validate(release_failed)
        self.assertEqual(release_failed["ocr_review"]["status"], "release_failed")

    def test_schema_types_geometry_duplicate_missing_source_ids(self):
        state = self.state()
        for value in (True, 0, 4, "1", None):
            self.reject(state, lambda s: s.update(schema_version=value))
        for value in (True, None, [], ""):
            self.reject(state, lambda s: s["ocr"]["raw_detections"][0][0].update(source_id=value))
        self.reject(state, lambda s: s["ocr"]["raw_detections"][0].append(copy.deepcopy(s["ocr"]["raw_detections"][0][0])))
        self.reject(state, lambda s: s["ocr"]["raw_detections"][0].pop())
        for value in (float("nan"), float("inf"), True, "0\" onload=\"evil()"):
            self.reject(state, lambda s: s["ocr"]["raw_detections"][0][0]["polygon"][0].__setitem__(0, value))
        for value in (True, "yes", 1):
            if value is True:
                continue
            self.reject(state, lambda s: s["ocr"]["batches"][0]["proposals"][0].update(crop_safe=value))

    def test_duplicate_json_keys_limits_and_exclusive_output(self):
        path = self.root / "saved.json"
        path.write_text('{"schema_version":1,"schema_version":2}')
        with self.assertRaises(ValueError):
            report.run(path, self.fixture.image, self.root / "report")
        state = self.state()
        path.write_text(canonical(state))
        output = self.root / "report"
        output.mkdir()
        with self.assertRaises(ValueError):
            report.run(path, self.fixture.image, output)
        link = self.root / "symlink"
        link.symlink_to(self.root / "absent")
        with self.assertRaises(ValueError):
            report.run(path, self.fixture.image, link)
        self.assertFalse((self.root / "absent").exists())
        self.reject(state, lambda s: s.update(unused=[0] * 8193))
        with self.assertRaises(ValueError):
            report.read_bounded(path, 1)
        for value in ("../secret", "/tmp/secret", "https://invalid/x", "a\\b", ".", ".."):
            self.reject(state, lambda s: s["units"][0]["generation_evidence"][0].update(raw_response_path=value))

    def test_failed_partial_cancelled_budget_and_nonselected_states(self):
        failure = self.state(runtime=SimpleNamespace(generate=Mock(side_effect=TimeoutError("synthetic"))))
        self.validate(failure)
        interrupted = self.state(runtime=SimpleNamespace(generate=Mock(side_effect=KeyboardInterrupt())))
        self.validate(interrupted)
        budget = self.state(max_attempts=0)
        self.validate(budget)
        selected = self.state(selected_ids=(self.fixture.batch.proposals[1].proposal_id,))
        self.validate(selected)
        self.fixture.engine.process_images.side_effect = RuntimeError("synthetic retained OCR failure")
        partial = self.state()
        entries, summary = self.validate(partial)
        self.assertEqual(len(entries), 2)
        self.assertTrue(summary["source_only_failure"])

    def test_no_detections_and_missing_evidence(self):
        self.fixture.engine.raw_detections = [[]]
        self.fixture.engine.automatic_unit_proposals = [None]
        state = self.state()
        entries, summary = self.validate(state)
        self.assertEqual(entries, [])
        self.assertEqual(summary["raw_detections"], 0)
        self.reject(state, lambda s: s.update(ocr=None))
        self.reject(state, lambda s: s.update(attempts=1))


class SavedTranslationReviewTests(unittest.TestCase):
    def check_saved(self, directory, schema):
        preview = ROOT / "test-results" / directory / "preview.json"
        image = ROOT / "images/ubunchu-ja-test/ubunchu01_04.png"
        if not preview.is_file() or not image.is_file():
            self.skipTest("optional local evidence unavailable; saved schema-" + str(schema) + " report case")
        with tempfile.TemporaryDirectory() as temporary:
            run = cli(preview, image, Path(temporary) / "report")
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads(run.stdout)
            self.assertFalse(summary["missing_source_ids"])
            self.assertGreater(summary["unit_count"], 0)

    def test_actual_saved_schema1(self):
        self.check_saved("workflow-fullpage-parent-v1", 1)

    def test_actual_saved_schema2(self):
        self.check_saved("workflow-fragments-priority-parent-v1", 2)

    def test_actual_saved_schema3(self):
        self.check_saved("workflow-visual-review-parent-v2", 3)


if __name__ == "__main__":
    unittest.main()
