"""Release hardening regressions: synthetic source, fake transports, no network."""
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from tests.translation.test_previews import fixture, response
from tests.tools.test_translation_conditions import ordered_manifest
from tralens.ocr.automatic_units import detection_evidence_sha256
from tralens.translation import (PreviewTranslator, OllamaRuntime, TranslationRequest,
    Style, automatic_units, TRANSLATEGEMMA, GENERALIST, UnitOutcome)
from tralens.translation.contracts import canonical, strict_json, revision
from tralens.translation.runtime import parse_outcomes, local_endpoint, make_payload
from tools import translation_diagnostics as diagnostics


def outcome(unit_id, disposition="preview", translation="Hello", interpretation="unverified", reasons=()):
    return {"id": unit_id, "translation": translation, "disposition": disposition,
            "reasons": list(reasons), "source_interpretation": interpretation}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        _, _, self.units = fixture(two=True)
        self.request = TranslationRequest(self.units, tuple(u.unit_id for u in self.units))

    def test_generalist_prompt_translates_readable_fragments_without_inventing_context(self):
        payload = make_payload(self.request, GENERALIST)
        prompt = payload["messages"][0]["content"]
        for control in (
            "fragments, ellipsis, conjunctions and omitted or unknown subjects are normal",
            "translate their readable meaning without inventing missing context",
            "Lack of speaker, object, recipient or narrative context, or generic unverified metadata alone, "
            "never justifies abstention or an unreadable label",
            "Use a review translation for uncertain but readable meaning",
            "only for genuinely unrecoverable source; do not force translation of garbled source",
            "'そしてありがとう!' -> 'And thank you!' is a valid complete preview without inferring a recipient",
            "'……ありがとう。' -> '...Thank you.' preserves ellipsis; an unknown speaker is not unreadability",
        ):
            self.assertIn(control, prompt)
        item_schema = payload["format"]["properties"]["translations"]["items"]
        self.assertEqual(item_schema["required"], ["id", "translation", "disposition", "reasons", "source_interpretation"])
        self.assertFalse(item_schema["additionalProperties"])

    def test_partial_abstention_retains_every_target_and_no_fabricated_translation(self):
        ids = self.request.target_ids
        runtime = Mock()
        runtime.generate.return_value = canonical({"translations": [outcome(ids[0]),
            outcome(ids[1], "abstained", "", "unreadable", ("source cannot be interpreted without guessing",))]})
        result = PreviewTranslator(runtime).translate(self.request)
        self.assertEqual(result.status, "preview")
        self.assertEqual(result.translations, ((ids[0], "Hello"),))
        self.assertEqual(tuple(o.unit_id for o in result.outcomes), ids)
        self.assertEqual(result.outcomes[1].disposition, "abstained")
        self.assertEqual(result.outcomes[1].translation, "")
        self.assertTrue(result.review_needed)
        self.assertEqual(result.outcomes[0].reading_revision, self.units[0].selected_revision)

    def test_all_abstained_is_valid_explicit_result_not_provider_failure_or_empty_preview(self):
        runtime = Mock()
        runtime.generate.return_value = canonical({"translations": [outcome(i, "abstained", "", "ambiguous",
            ("ambiguous wording",)) for i in self.request.target_ids]})
        result = PreviewTranslator(runtime).translate(self.request)
        self.assertEqual((result.status, result.failure, result.translations, result.attempts), ("abstained", "", (), 1))
        self.assertEqual(len(result.outcomes), 2)

    def test_ambiguous_source_cannot_be_downgraded_by_model_to_nonreview_preview(self):
        runtime = Mock()
        runtime.generate.return_value = response(self.request.target_ids)
        result = PreviewTranslator(runtime).translate(self.request)
        self.assertTrue(result.review_needed)
        for item in result.outcomes:
            self.assertEqual((item.disposition, item.source_interpretation), ("review", "ambiguous"))
            self.assertIn("source_reading_uncertain", item.reasons)

    def test_specialist_review_flag_visible_without_changing_official_prompt(self):
        unit = self.units[0]
        runtime = Mock()
        runtime.generate.return_value = "Hello"
        result = PreviewTranslator(runtime, model="translategemma:12b", profile=TRANSLATEGEMMA).translate(
            TranslationRequest((unit,), (unit.unit_id,)))
        self.assertTrue(result.review_needed)
        self.assertEqual(result.outcomes[0].disposition, "review")
        self.assertIn("specialist_reduced_capability", result.outcomes[0].reasons)
        self.assertEqual(len(runtime.generate.call_args.kwargs["payload"]["messages"]), 1)

    def test_text_only_historical_schema_is_not_silently_accepted_as_semantic_v2(self):
        runtime = Mock()
        runtime.generate.return_value = canonical({"translations": [{"id": i, "translation": "Hello"}
            for i in self.request.target_ids]})
        result = PreviewTranslator(runtime).translate(self.request)
        self.assertEqual((result.failure, result.outcomes), ("invalid_output", ()))

    def test_generic_unverified_completeness_never_requires_review_gate_or_blocks_call(self):
        unit = self.units[0]
        unit = replace(unit, readings=(unit.selected,), lexical_reasons=("wording_not_human_checked",),
                       structural_reasons=("automatic_not_human_reviewed",))
        request = TranslationRequest((unit,), (unit.unit_id,))
        runtime = Mock()
        runtime.generate.return_value = response(request.target_ids)
        result = PreviewTranslator(runtime).translate(request)
        self.assertEqual((result.status, result.attempts), ("preview", 1))
        self.assertEqual(result.outcomes[0].source_interpretation, "unverified")
        self.assertFalse(result.review_needed)

    def test_typed_outcome_rejects_invalid_abstention_reason_and_interpretation(self):
        bad = [outcome("a", "abstained", "invented", "ambiguous", ("uncertain",)),
               outcome("a", "abstained", "", "ambiguous"), outcome("a", "review"),
               outcome("a", interpretation="confident"), outcome("a", interpretation="unreadable"),
               outcome("a", reasons=("x" * 513,)), outcome("a", reasons=("reason",) * 13)]
        for item in bad:
            with self.assertRaises(ValueError):
                parse_outcomes(canonical({"translations": [item]}), ("a",))

    def test_result_staleness_rejected_when_source_candidate_or_style_changes(self):
        runtime = Mock()
        runtime.generate.return_value = response(self.request.target_ids)
        result = PreviewTranslator(runtime).translate(self.request)
        result.assert_current(self.request)
        with self.assertRaisesRegex(ValueError, "stale"):
            result.assert_current(replace(self.request, style=Style(target_language="French")))
        changed = replace(self.units[0], selected_revision=self.units[0].readings[1].revision)
        with self.assertRaisesRegex(ValueError, "stale"):
            result.assert_current(replace(self.request, units=(changed, self.units[1])))
        historical = asdict(self.request)
        historical.pop("contract_version")
        self.assertNotEqual(result.request_revision, revision(historical))
        with self.assertRaisesRegex(ValueError, "stale"):
            replace(result, request_revision=revision(historical)).assert_current(self.request)
        wrong_binding = replace(result.outcomes[0], reading_revision=self.units[0].readings[1].revision)
        with self.assertRaisesRegex(ValueError, "binding"):
            replace(result, outcomes=(wrong_binding, result.outcomes[1])).assert_current(self.request)


class EvidenceAndBoundaryTests(unittest.TestCase):
    def test_blank_main_stays_explicitly_blocked_not_dropped_or_repaired_by_alternative(self):
        batch, _, _ = fixture()
        raw = replace(batch.detections[0], text="", selected_text="")
        batch = replace(batch, detections=(raw,), evidence_sha256=detection_evidence_sha256((raw,)))
        units = automatic_units(batch)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].selected.text, "")
        self.assertIn("blank_main_reading", units[0].blocking_reasons)
        runtime = Mock()
        result = PreviewTranslator(runtime).translate(TranslationRequest(units, (units[0].unit_id,)))
        self.assertEqual((result.status, result.attempts), ("blocked", 0))
        self.assertIn("blank_main_reading", result.execution_metadata_json)
        runtime.generate.assert_not_called()

    def test_conflicting_owners_and_string_safety_flag_rejected(self):
        batch, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "ownership"):
            automatic_units(replace(batch, regions=batch.regions + (batch.regions[0],)))
        with self.assertRaisesRegex(ValueError, "boolean"):
            automatic_units(replace(batch, proposals=(replace(batch.proposals[0], crop_safe="false"),)))

    def test_endpoint_scope_rejects_public_or_dns_hosts_and_pins_localhost_without_lookup(self):
        for endpoint in ("http://example.com:11434", "http://8.8.8.8:11434", "http://0.0.0.0:11434",
                         "http://169.254.169.254", "http://server.internal", "http://2130706433"):
            with self.assertRaises(ValueError):
                local_endpoint(endpoint)
        with patch("socket.getaddrinfo", side_effect=AssertionError("no DNS")):
            self.assertEqual(local_endpoint("http://localhost:11434"), "http://127.0.0.1:11434")
            self.assertEqual(local_endpoint("http://100.121.133.88:11434"), "http://100.121.133.88:11434")
            self.assertEqual(local_endpoint("http://[::1]:11434/"), "http://[::1]:11434")

    def test_observer_write_failure_counts_attempt_and_stops_retry_without_output(self):
        _, _, units = fixture()
        request = TranslationRequest(units, (units[0].unit_id,))
        runtime = Mock()
        runtime.generate.return_value = response(request.target_ids)
        result = PreviewTranslator(runtime, max_attempts=3).translate(request,
            observe=Mock(side_effect=OSError("synthetic disk failure")))
        self.assertEqual((result.failure, result.attempts, result.translations), ("evidence_retention", 1, ()))
        self.assertEqual(runtime.generate.call_count, 1)

    def test_each_retry_raw_response_retained_with_distinct_attempts(self):
        _, _, units = fixture()
        request = TranslationRequest(units, (units[0].unit_id,))
        runtime = Mock()
        runtime.generate.side_effect = ["malformed", response(request.target_ids)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = []
            def observe(event):
                evidence.append(diagnostics.retain_response(root, f"attempt-{event['attempt']}", event["response_json"], 1024))
            result = PreviewTranslator(runtime, max_attempts=2).translate(request, observe=observe)
            self.assertEqual((result.status, result.attempts), ("preview", 2))
            self.assertEqual(len(evidence), 2)
            self.assertNotEqual(evidence[0]["raw_response_path"], evidence[1]["raw_response_path"])
            self.assertIn("malformed", (root / evidence[0]["raw_response_path"]).read_text())

    def test_diagnostics_partial_evidence_failure_spends_global_budget_and_appends_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / "saved.json"
            saved.write_text(json.dumps(ordered_manifest()), encoding="utf-8")
            runtime = Mock()
            runtime.generate.return_value = response(("z",))
            with patch.object(diagnostics, "retain_response", side_effect=OSError("synthetic")):
                summary = diagnostics.run(saved, root / "out", runtime=runtime, run_local_model=True,
                    model_digest="asserted", model_details="synthetic", max_attempts=1, max_units=2)
            records = [json.loads(line) for line in (root / "out" / "results.jsonl").read_text().splitlines()]
            self.assertEqual(summary["attempts"], 1)
            self.assertEqual(records[0]["failure"], "evidence_retention")
            self.assertEqual(records[1]["status"], "budget_exhausted")

    def test_finite_json_and_direct_api_input_budget(self):
        for text in ("1e400", "-1e400", "NaN", "Infinity"):
            with self.assertRaises(ValueError):
                strict_json(text)
        _, _, units = fixture()
        request = TranslationRequest(units, (units[0].unit_id,), style=Style(instructions="x" * 65536))
        runtime = Mock()
        result = PreviewTranslator(runtime).translate(request)
        self.assertEqual((result.failure, result.attempts), ("capability", 0))
        runtime.generate.assert_not_called()

    def test_bounded_file_read_and_nonregular_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / "evidence.json"
            saved.write_bytes(b"x" * 1025)
            with self.assertRaisesRegex(ValueError, "budget"):
                diagnostics.read_bounded(saved, 1024)
            with self.assertRaisesRegex(ValueError, "regular file"):
                diagnostics.read_bounded(root, 1024)

    def test_length_terminated_reasoning_like_content_never_becomes_translation(self):
        _, _, units = fixture()
        request = TranslationRequest(units, (units[0].unit_id,))
        raw = {"done": True, "done_reason": "length", "message": {"content": "Let me reason about the source"}}
        runtime = OllamaRuntime("http://127.0.0.1:11434", transport=Mock(return_value=raw))
        events = []
        result = PreviewTranslator(runtime, model="qwen3:4b").translate(request, observe=events.append)
        self.assertEqual((result.failure, result.translations), ("provider", ()))
        self.assertEqual(json.loads(events[0]["response_json"])["done_reason"], "length")
