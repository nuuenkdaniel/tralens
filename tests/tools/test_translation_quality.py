"""Clean-text helper tests with invented corpus and fake runtimes; no OCR or model I/O."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tralens.translation import Generation, GenerationOptions
from tralens.translation.contracts import canonical
from tralens.translation.runtime import GENERALIST
from tools import translation_quality as q


def cases():
    return [{"id": i, "source": text, "reference": "HUMAN_REFERENCE_SENTINEL_" + i,
             "split": "test", "origin": "synthetic unittest only"}
            for i, text in (("z-last-sort", "こんにちは"), ("b-blank", ""), ("a-first-sort", "ありがとう"))]


def generalist_response(payload, text="Hello"):
    source = json.loads(payload["messages"][1]["content"])
    raw = {"done": True, "load_duration": 10, "prompt_eval_duration": 20, "eval_duration": 30,
           "prompt_eval_count": 40, "eval_count": 50, "message": {"content": canonical({"translations": [
               {"id": i, "translation": text, "disposition": "preview", "reasons": [], "source_interpretation": "unverified"}
               for i in source["target_ids"]]})}}
    return Generation(raw["message"]["content"], canonical(raw))


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "cases.json"
        self.data = cases()
        self.source.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        self.identity = self.root / "model.json"
        self.identity.write_text(json.dumps({"model": "qwen3.8:latest", "digest": "a" * 64, "details": {"test_only": True}}), encoding="utf-8")

    def run_helper(self, name="run", **kwargs):
        return q.run(self.source, self.root / name, model="qwen3.8:latest", model_record=self.identity, **kwargs)

    def records(self, name="run"):
        return [json.loads(line) for line in (self.root / name / "results.jsonl").read_text().splitlines()]

    def test_default_is_model_free_full_denominator_and_exclusive_output(self):
        runtime = Mock()
        with patch.object(q, "OllamaRuntime", side_effect=AssertionError("must not construct runtime")):
            summary = self.run_helper(runtime=runtime)
        runtime.generate.assert_not_called()
        self.assertEqual((summary["denominator"], summary["record_count"], summary["attempts"]), (3, 3, 0))
        self.assertEqual([r["case_id"] for r in self.records()], [c["id"] for c in self.data])
        self.assertTrue(all(r["candidate"] == "" for r in self.records()))
        with self.assertRaises(FileExistsError):
            self.run_helper()

    def test_reference_and_corpus_metadata_never_enter_request_or_prompt(self):
        case = self.data[0]
        request, frozen = q.frozen_request(case, GENERALIST, GenerationOptions(num_ctx=4096, num_predict=1024, keep_alive=60))
        serialized = canonical(frozen)
        self.assertNotIn(case["reference"], serialized)
        self.assertNotIn(case["origin"], serialized)
        self.assertNotIn('"reference"', serialized)
        self.assertIn("synthetic_text_only_benchmark_not_OCR", request.units[0].page_json)
        self.assertNotIn("polygon", serialized)
        changed_reference = {**case, "reference": "OTHER_REFERENCE", "split": "dev", "origin": "different metadata"}
        self.assertEqual(q.text_request(changed_reference).revision, request.revision)
        self.assertNotEqual(q.text_request({**case, "source": "違う"}).revision, request.revision)

    def test_live_fake_preserves_raw_metadata_failures_blanks_and_single_attempts(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            self.assertNotIn("HUMAN_REFERENCE_SENTINEL", canonical(kwargs["payload"]))
            return generalist_response(kwargs["payload"]) if len(calls) == 1 else Generation("malformed", '{"text":"malformed"}')
        runtime = SimpleNamespace(generate=generate)
        summary = self.run_helper(runtime=runtime, run_local_model=True)
        records = self.records()
        self.assertEqual(summary["statuses"], {"blocked": 1, "failed": 1, "preview": 1})
        self.assertEqual((summary["denominator"], summary["attempts"]), (3, 2))
        self.assertEqual([r["candidate"] for r in records], ["Hello", "", ""])
        self.assertEqual(records[2]["result"]["failure"], "invalid_output")
        self.assertEqual(records[0]["generation_evidence"][0]["generation_metadata"]["eval_count"], 50)
        for record in (records[0], records[2]):
            evidence = record["generation_evidence"][0]
            self.assertFalse(evidence["truncated"])
            self.assertTrue((self.root / "run" / evidence["raw_response_path"]).is_file())
            self.assertGreaterEqual(record["wall_seconds"], 0)
        for call in calls:
            self.assertEqual(call["payload"]["options"], {"temperature": 0, "num_ctx": 4096, "num_predict": 1024})
            self.assertEqual(call["payload"]["keep_alive"], 60)
            self.assertTrue(0 < call["timeout"] <= 120)
        review = json.loads((self.root / "run" / "review.json").read_text())
        self.assertEqual(len(review["cases"]), 3)
        self.assertEqual([c["candidates"][0]["text"] for c in review["cases"]], ["Hello", "", ""])
        self.assertIn(self.data[0]["reference"], canonical(review))
        self.assertNotIn("qwen3.8", canonical(review))
        self.assertNotIn("invalid_output", canonical(review))
        mapping = json.loads((self.root / "run" / "model-mapping.json").read_text())
        self.assertEqual(next(iter(mapping.values()))["model"], "qwen3.8:latest")

    def test_frozen_prompt_equals_actual_payload_and_order_does_not_sort_ids(self):
        captured = []
        def generate(**kwargs):
            captured.append(kwargs["payload"])
            return generalist_response(kwargs["payload"])
        self.run_helper(runtime=SimpleNamespace(generate=generate), run_local_model=True, max_cases=1)
        plan = json.loads((self.root / "run" / "run.json").read_text())
        self.assertEqual(plan["selected_ids"], ["z-last-sort"])
        self.assertEqual(plan["requests"][0]["payload"], captured[0])
        self.assertEqual(plan["requests"][0]["request_revision"], self.records()[0]["result"]["request_revision"])
        self.assertEqual(plan["max_attempts_per_case"], 1)
        self.assertEqual(plan["preload_calls"], 0)

    def test_abstentions_and_timeouts_remain_blank_without_retries(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls) > 1:
                raise TimeoutError("synthetic timeout")
            data = json.loads(kwargs["payload"]["messages"][1]["content"])
            text = canonical({"translations": [{"id": data["target_ids"][0], "translation": "",
                "disposition": "abstained", "reasons": ["synthetic reason"], "source_interpretation": "ambiguous"}]})
            return Generation(text, canonical({"message": {"content": text}}))
        summary = self.run_helper(run_local_model=True, runtime=SimpleNamespace(generate=generate))
        self.assertEqual(summary["statuses"], {"abstained": 1, "blocked": 1, "failed": 1})
        self.assertEqual((summary["attempts"], summary["denominator"]), (2, 3))
        self.assertTrue(all(r["candidate"] == "" for r in self.records()))
        self.assertEqual(self.records()[2]["generation_evidence"][0]["status"], "timeout")

    def test_pair_comparison_exact_counts_blinded_mapping_and_tamper_rejection(self):
        self.run_helper("one")
        self.run_helper("two")
        summary = q.compare(self.source, self.root / "paired", [self.root / "one", self.root / "two"])
        self.assertEqual((summary["paired_cases"], summary["records_per_run"]), (3, [3, 3]))
        review = json.loads((self.root / "paired" / "review.json").read_text())
        self.assertTrue(all(len(c["candidates"]) == 2 for c in review["cases"]))
        records = self.records("two")
        records.pop()
        (self.root / "two" / "results.jsonl").write_text("\n".join(canonical(r) for r in records) + "\n")
        with self.assertRaisesRegex(ValueError, "counts"):
            q.compare(self.source, self.root / "bad", [self.root / "one", self.root / "two"])
        self.assertFalse((self.root / "bad").exists())

    def test_specialist_uses_its_own_prompt_and_pairs_same_sources_with_generalist(self):
        self.run_helper("generalist")
        identity = self.root / "gemma.json"
        identity.write_text(json.dumps({"model": "translategemma:12b", "digest": "b" * 64}))
        captured = []
        def generate(**kwargs):
            captured.append(kwargs["payload"])
            return Generation("Thank you", '{"text":"Thank you"}')
        q.run(self.source, self.root / "specialist", model="translategemma:12b", model_record=identity,
              profile="translategemma", run_local_model=True, runtime=SimpleNamespace(generate=generate))
        self.assertEqual(len(captured), 2)
        for payload in captured:
            self.assertEqual(len(payload["messages"]), 1)
            self.assertEqual(payload["messages"][0]["role"], "user")
            self.assertNotIn("format", payload)
            self.assertNotIn("HUMAN_REFERENCE_SENTINEL", canonical(payload))
        summary = q.compare(self.source, self.root / "paired", [self.root / "generalist", self.root / "specialist"])
        self.assertEqual(summary["records_per_run"], [3, 3])

    def test_compare_rejects_stale_source_binding_or_changed_candidate(self):
        runtime = SimpleNamespace(generate=lambda **kw: generalist_response(kw["payload"]))
        self.run_helper(runtime=runtime, run_local_model=True)
        original = self.records()
        for key, value in (("request_revision", "stale"), ("candidate", "forged")):
            records = copy.deepcopy(original)
            records[0][key] = value
            (self.root / "run" / "results.jsonl").write_text("\n".join(canonical(r) for r in records) + "\n")
            with self.assertRaises(ValueError):
                q.compare(self.source, self.root / "bad", [self.root / "run"])

    def test_optional_metric_unavailable_never_imports_or_installs(self):
        records = [{"case_id": c["id"], "status": "failed", "candidate": ""} for c in self.data]
        with patch.object(q.importlib.util, "find_spec", return_value=None), patch.object(q.importlib, "import_module") as imported:
            score = q.score_records(self.data, records)
        imported.assert_not_called()
        self.assertFalse(score["available"])
        self.assertEqual(score["denominator"], 3)

    def test_installed_metric_helper_keeps_blank_failures_in_denominator(self):
        records = [{"case_id": c["id"], "status": "preview" if i == 0 else "failed",
                    "candidate": "Hello" if i == 0 else ""} for i, c in enumerate(self.data)]
        metric = Mock(return_value=SimpleNamespace(score=12.5))
        with patch.object(q.importlib.util, "find_spec", return_value=object()), patch.object(q.importlib, "import_module",
                return_value=SimpleNamespace(corpus_bleu=metric, __version__="synthetic")):
            result = q.score_records(self.data, records)
        self.assertEqual(metric.call_args.args, (["Hello", "", ""], [[c["reference"] for c in self.data]]))
        self.assertEqual(result["denominator"], 3)
        self.assertFalse(result["semantic_quality_claimed"])
        with patch.object(q.importlib.util, "find_spec", return_value=object()), patch.object(q.importlib, "import_module",
                side_effect=RuntimeError("synthetic broken installation")):
            unavailable = q.score_records(self.data, records)
        self.assertFalse(unavailable["available"])

    def test_caps_case_schema_model_record_and_endpoint_fail_before_output(self):
        for kwargs in ({"max_cases": 151}, {"max_cases": 0}, {"endpoint": "http://example.com"},
                       {"num_predict": 0}, {"keep_alive": -1}):
            with self.assertRaises(ValueError):
                self.run_helper(**kwargs)
            self.assertFalse((self.root / "run").exists())
        self.identity.write_text('{"model":"other","digest":"' + "a" * 64 + '"}')
        with self.assertRaisesRegex(ValueError, "model record"):
            self.run_helper()
        self.source.write_text(json.dumps([self.data[0], self.data[0]]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            q.load_cases(self.source, 150)

    def test_cli_default_offline_and_read_only_metric_availability(self):
        with patch.object(q, "OllamaRuntime", side_effect=AssertionError("no live default")):
            self.assertEqual(q.main(["--cases", str(self.source), "--output", str(self.root / "cli"),
                "--model", "qwen3.8:latest", "--model-record", str(self.identity), "--max-cases", "2"]), 0)
        self.assertEqual(len(self.records("cli")), 2)
        print("Installed-only metric availability: " + canonical(q.metric_availability()))
