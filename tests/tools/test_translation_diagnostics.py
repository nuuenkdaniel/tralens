"""Saved synthetic evidence only; injected translation callbacks, never live I/O."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.translation.test_previews import fixture, response
from tools import translation_diagnostics as d


class DiagnosticsTests(unittest.TestCase):
    def manifest(self, *, blocked=False):
        batch, candidates, _ = fixture(two=True)
        if blocked:
            batch = replace(batch, proposals=tuple(replace(p, crop_safe=False) for p in batch.proposals))
            candidates = ()
        return {"pages": [{"batch": asdict(batch), "units": [{"recognition": {
            "candidates": [asdict(c) for c in candidates]}}]}]}

    def test_preflight_default_no_runtime_new_output_only_and_bound_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "saved.json"
            source.write_text(json.dumps(self.manifest()), encoding="utf-8")
            runtime = Mock()
            with patch.object(d, "OllamaRuntime", side_effect=AssertionError("no runtime")):
                summary = d.run(source, root / "out", runtime=runtime, max_units=1)
            runtime.generate.assert_not_called()
            self.assertEqual(summary["attempts"], 0)
            plan = json.loads((root / "out" / "requests.json").read_text())
            self.assertTrue(plan["preview_only"])
            self.assertEqual(len(plan["requests"]), 1)
            self.assertEqual(plan["dispositions"][0]["disposition"], "preview")
            self.assertEqual(len((root / "out" / "results.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError):
                d.run(source, root / "out")

    def test_global_error_budget_includes_preload_and_append_each_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "saved.json"
            source.write_text(json.dumps(self.manifest()), encoding="utf-8")
            runtime = Mock()
            runtime.preload.side_effect = RuntimeError("failure")
            runtime.generate.side_effect = RuntimeError("failure")
            summary = d.run(source, root / "out", runtime=runtime, run_local_model=True,
                model_digest="test", model_details="synthetic", max_units=2, max_attempts=2, preload=True)
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(runtime.generate.call_count, 1)
            records = [json.loads(line) for line in (root / "out" / "results.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in records], ["failed", "failed", "budget_exhausted"])

    def test_deterministic_selection_context_budget_and_blocked_disposition(self):
        manifest = self.manifest()
        requests, _ = d.preflight(manifest, max_units=1, mode="B", max_context_chars=0)
        self.assertEqual(requests[0].context_ids, ())
        requests2, _ = d.preflight(manifest, max_units=1, mode="B", max_context_chars=8000)
        self.assertEqual(requests[0].target_ids, requests2[0].target_ids)
        self.assertTrue(requests2[0].context_ids)
        _, dispositions = d.preflight(self.manifest(blocked=True), max_units=2)
        self.assertTrue(all(r["disposition"] == "blocked" for r in dispositions))
        for options in ({"selected_ids": ("foreign",)}, {"max_units": 21}, {"mode": "D"}):
            with self.assertRaises(ValueError):
                d.preflight(manifest, **options)

    def test_live_requires_explicit_parent_identity_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                d.run(root / "absent.json", root / "out", run_local_model=True)
            self.assertFalse((root / "out").exists())

    def test_cli_model_free_preflight_and_injected_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "saved.json"
            source.write_text(json.dumps(self.manifest()), encoding="utf-8")
            with patch.object(d, "OllamaRuntime", side_effect=AssertionError("no live construction")):
                self.assertEqual(d.main(["--manifest", str(source), "--output", str(root / "preflight"),
                                         "--max-units", "1", "--max-attempts", "0"]), 0)
            requests, _ = d.preflight(self.manifest(), max_units=1)
            runtime = Mock()
            runtime.generate.return_value = response(requests[0].target_ids)
            summary = d.run(source, root / "success", runtime=runtime, run_local_model=True,
                            model_digest="synthetic", model_details="test only", max_units=1, max_attempts=1)
            self.assertEqual(summary["counts"], {"preview": 1})
            result = json.loads((root / "success" / "results.jsonl").read_text())
            self.assertEqual(result["request_revision"], requests[0].revision)
            self.assertEqual(result["translations"][0][1], "Hello")
