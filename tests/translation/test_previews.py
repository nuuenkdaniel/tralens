"""Synthetic utterance contracts and injected runtimes only; no models/network."""
import copy
import base64
import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from unittest.mock import Mock, patch

from tests.ocr import test_automatic_recognition as recognition_tests
from tralens.translation import (Style, TranslationRequest, automatic_units, PreviewTranslator,
                                 OllamaRuntime, CapabilityProfile, GENERALIST, TRANSLATEGEMMA, Translate)
from tralens.translation.contracts import canonical
from tralens.translation.runtime import parse_translations


def fixture(two=False):
    helper = recognition_tests.AutomaticRecognitionTests()
    image, content, batch = helper.fixture(text="こんにちは", two=two)
    _, results = helper.recognize(image, content, batch)
    candidates = tuple(c for r in results for c in r.candidates)
    return batch, candidates, automatic_units(batch, candidates)


def response(ids, text="Hello"):
    return json.dumps({"translations": [{"id": i, "translation": text, "disposition": "preview",
        "reasons": [], "source_interpretation": "unverified"} for i in ids]})


class ContractTests(unittest.TestCase):
    def test_automatic_uncertainty_preserved_without_mandatory_review_or_mutation(self):
        batch, candidates, units = fixture()
        before = copy.deepcopy((asdict(batch), [asdict(c) for c in candidates]))
        unit = units[0]
        self.assertEqual(unit.disposition, "preview")
        self.assertEqual(unit.completeness, "unverified_full_utterance")
        self.assertIn("paddle_wording_unresolved", unit.lexical_reasons)
        self.assertEqual(unit.selected.text, "こんにちは")
        self.assertEqual(len(unit.readings), 2)
        self.assertEqual(unit.readings[1].text, candidates[0].text)
        with self.assertRaises(FrozenInstanceError):
            unit.unit_id = "changed"
        self.assertEqual(automatic_units(json.loads(canonical(asdict(batch))),
            json.loads(canonical([asdict(c) for c in candidates]))), units)
        self.assertEqual(before, (asdict(batch), [asdict(c) for c in candidates]))

    def test_binding_changes_with_page_order_members_candidate_style_and_context(self):
        _, _, units = fixture(two=True)
        unit = units[0]
        request = TranslationRequest((unit,), (unit.unit_id,))
        variants = [replace(unit, page_json=canonical({"page": "changed"})),
                    replace(unit, member_ids=unit.member_ids + ("new",)),
                    replace(unit, selected_revision=unit.readings[1].revision),
                    replace(unit, evidence_revision="new"),
                    replace(unit, order=unit.order + ("new",), member_ids=unit.member_ids + ("new",))]
        for changed in variants:
            self.assertNotEqual(request.revision, replace(request, units=(changed,)).revision)
        self.assertNotEqual(request.revision, replace(request, style=Style(target_language="French")).revision)
        context_request = TranslationRequest(units, (units[0].unit_id,), (units[1].unit_id,), mode="B")
        self.assertNotEqual(request.revision, context_request.revision)
        with self.assertRaises(ValueError):
            TranslationRequest(units, (units[0].unit_id,), (units[0].unit_id,), mode="B")

    def test_candidate_and_raw_drift_rejected(self):
        batch, candidates, _ = fixture()
        with self.assertRaisesRegex(ValueError, "raw evidence"):
            automatic_units(replace(batch, evidence_sha256="bad"))
        for field, value in (("member_ids", ("foreign",)), ("proposal_sha256", "bad"),
                             ("batch_evidence_sha256", "bad"), ("main_ids", ("foreign",))):
            candidate = replace(candidates[0], crop=replace(candidates[0].crop, **{field: value}))
            with self.assertRaisesRegex(ValueError, "candidate provenance"):
                automatic_units(batch, (candidate,))

    def test_known_structural_fault_blocks_but_generic_uncertainty_does_not(self):
        batch, _, _ = fixture()
        p = batch.proposals[0]
        for fault in ("mixed_region_ownership", "unknown_reading_orientation",
                      "ambiguous_annotation_parents", "crop_isolation_unresolved"):
            changed = replace(p, structural_reasons=p.structural_reasons + (fault,))
            unit = automatic_units(replace(batch, proposals=(changed,)))[0]
            runtime = Mock()
            result = PreviewTranslator(runtime).translate(TranslationRequest((unit,), (unit.unit_id,)))
            self.assertEqual((result.status, result.attempts), ("blocked", 0))
            runtime.generate.assert_not_called()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        _, _, units = fixture()
        self.request = TranslationRequest(units, (units[0].unit_id,))

    def test_lazy_constructors_and_strict_legacy_failure_safeguards(self):
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network")):
            Translate()
            OllamaRuntime("http://127.0.0.1:11434")
            PreviewTranslator(Mock())
        legacy = Translate(client=Mock())
        for raw in ('prefix [{"id":1,"translation":"x"}]',
                    '[{"id":1,"id":2,"translation":"x"}]',
                    '[{"id":true,"translation":"x"}]', '[{"id":1,"translation":""}]'):
            self.assertEqual(legacy._extract_json_lists(raw), [])

    def test_full_json_schema_bijection_and_duplicate_key_rejection(self):
        valid = response(("a", "b"))
        self.assertEqual(parse_translations(valid, ("b", "a")), (("b", "Hello"), ("a", "Hello")))
        bad = ["```json\n" + valid + "\n```", valid + " trailing", "[]",
               '{"translations":[],"translations":[]}',
               '{"translations":[{"id":"a","translation":"x","translation":"y"}]}',
               response(("a", "a")), response(("a", "context")), response(("a",)),
               response(("a", "b"), ""), '{"translations":NaN}',
               '{"translations":[{"id":true,"translation":"x"}]}']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_translations(raw, ("a", "b"))

    def test_retries_typed_errors_and_deadline(self):
        runtime = Mock()
        runtime.generate.side_effect = ["malformed", response(self.request.target_ids)]
        result = PreviewTranslator(runtime, max_attempts=2).translate(self.request)
        self.assertEqual((result.status, result.attempts), ("preview", 2))
        self.assertTrue(result.preview_only)
        self.assertEqual(result.request_revision, self.request.revision)
        self.assertGreater(runtime.generate.call_args.kwargs["timeout"], 0)
        for error, code in ((TimeoutError("timeout"), "timeout"), (RuntimeError("private"), "provider")):
            runtime.generate.side_effect = error
            result = PreviewTranslator(runtime, max_attempts=2).translate(self.request)
            self.assertEqual((result.failure, result.attempts), (code, 2))
        runtime.generate.side_effect = None
        runtime.generate.return_value = response(self.request.target_ids)
        with patch("tralens.translation.runtime.time.monotonic", side_effect=[0, 0, 61, 61]):
            result = PreviewTranslator(runtime).translate(self.request)
        self.assertEqual(result.failure, "timeout")
        self.assertEqual(result.translations, ())

    def test_cancel_before_and_after_call(self):
        runtime = Mock()
        result = PreviewTranslator(runtime).translate(self.request, cancelled=lambda: True)
        self.assertEqual((result.status, result.attempts), ("cancelled", 0))
        runtime.generate.assert_not_called()
        runtime.generate.return_value = response(self.request.target_ids)
        result = PreviewTranslator(runtime).translate(self.request, cancelled=Mock(side_effect=[False, True]))
        self.assertEqual((result.status, result.attempts, result.translations), ("cancelled", 1, ()))

    def test_context_ids_not_output_ids_and_languages_are_configurable(self):
        _, _, units = fixture(two=True)
        request = TranslationRequest(units, (units[0].unit_id,), (units[1].unit_id,),
                                     style=Style(target_language="French", glossary=(("猫", "chat"),)), mode="B")
        runtime = Mock()
        runtime.generate.return_value = response((units[1].unit_id,))
        result = PreviewTranslator(runtime).translate(request)
        self.assertEqual(result.failure, "invalid_output")
        payload = runtime.generate.call_args.kwargs["payload"]
        self.assertIn("French", payload["messages"][1]["content"])
        self.assertIn("chat", payload["messages"][1]["content"])
        self.assertEqual(payload["format"]["properties"]["translations"]["items"]["properties"]["id"]["enum"],
                         list(request.target_ids))

    def test_specialist_template_plaintext_not_arbitrary_json_and_capability_guards(self):
        runtime = Mock()
        runtime.generate.return_value = "Hello."
        result = PreviewTranslator(runtime, model="translategemma:4b", profile=TRANSLATEGEMMA).translate(self.request)
        self.assertEqual(result.translations, ((self.request.target_ids[0], "Hello."),))
        payload = runtime.generate.call_args.kwargs["payload"]
        self.assertNotIn("format", payload)
        self.assertIn("Japanese (ja)", payload["messages"][0]["content"])
        self.assertEqual(len(payload["messages"]), 1)
        result = PreviewTranslator(runtime, model="translategemma:4b", profile=TRANSLATEGEMMA).translate(
            replace(self.request, style=Style(glossary=(("a", "b"),))))
        self.assertEqual(result.failure, "capability")
        result = PreviewTranslator(runtime).translate(replace(self.request, mode="C"), images=("base64",))
        self.assertEqual(result.failure, "capability")
        runtime.generate.return_value = response(self.request.target_ids)
        page = json.loads(self.request.units[0].page_json)
        evidence = {"original_sha256": page["image_sha256"], "original_pixels_sha256": page["pixels_sha256"],
                    "presentation_sha256": hashlib.sha256(b"synthetic").hexdigest()}
        result = PreviewTranslator(runtime, profile=CapabilityProfile("parent_verified_vision", True, True, True)).translate(
            replace(self.request, mode="C", image_evidence_json=canonical(evidence)),
            images=(base64.b64encode(b"synthetic").decode(),))
        self.assertEqual(result.status, "preview")

    def test_ollama_native_specialist_and_optional_raw_encoder(self):
        raw_profile = replace(TRANSLATEGEMMA, name="test_raw", specialist_chat=False)
        transport = Mock(return_value={"done": True, "response": "Hello"})
        runtime = OllamaRuntime("http://127.0.0.1:11434", transport=transport)
        result = PreviewTranslator(runtime, profile=raw_profile).translate(self.request)
        self.assertEqual((result.failure, result.attempts), ("capability", 0))
        transport.assert_not_called()
        encoder = Mock(return_value="synthetic verified template")
        runtime = OllamaRuntime("http://127.0.0.1:11434", transport=transport, specialist_template=encoder)
        result = PreviewTranslator(runtime, profile=raw_profile).translate(self.request)
        self.assertEqual(result.status, "preview")
        self.assertEqual(transport.call_args.args[0], "/api/generate")
        self.assertTrue(transport.call_args.args[1]["raw"])
        self.assertNotIn("format", transport.call_args.args[1])

    def test_ollama_transport_injection_no_network_and_incomplete_failure(self):
        transport = Mock(return_value={"done": True, "message": {"content": response(self.request.target_ids)}})
        runtime = OllamaRuntime("http://127.0.0.1:11434", transport=transport)
        result = PreviewTranslator(runtime).translate(self.request)
        self.assertEqual(result.status, "preview")
        path, body, timeout = transport.call_args.args
        self.assertEqual(path, "/api/chat")
        self.assertFalse(body["stream"])
        self.assertFalse(body["think"])
        transport.return_value = {"done": False}
        self.assertEqual(PreviewTranslator(runtime).translate(self.request).failure, "provider")
        for endpoint in ("http://user:pass@localhost", "https://localhost", "http://localhost/path"):
            with self.assertRaises(ValueError):
                OllamaRuntime(endpoint)
