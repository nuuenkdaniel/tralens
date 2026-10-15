"""Parent-verified public prompt contract; fake transports only, never model I/O."""
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from tests.translation.test_previews import fixture, response
from tests.tools.test_translation_conditions import ordered_manifest
from tralens.translation import (Style, TranslationRequest, PreviewTranslator, OllamaRuntime,
    TRANSLATEGEMMA, GENERALIST, GenerationOptions)
from tralens.translation.runtime import make_payload, TRANSLATEGEMMA_PROVENANCE
from tools import translation_diagnostics as diagnostics


class SpecialistChatTests(unittest.TestCase):
    def setUp(self):
        _, _, units = fixture()
        self.request = TranslationRequest(units, (units[0].unit_id,))

    def test_exact_official_single_user_content_three_newlines(self):
        expected = (
            "You are a professional Japanese (ja) to English (en) translator. "
            "Your goal is to accurately convey the meaning and nuances of the original Japanese text "
            "while adhering to English grammar, vocabulary, and cultural sensitivities. "
            "Produce only the English translation, without any additional explanations or commentary. "
            "Please translate the following Japanese text into English:\n\n\nこんにちは")
        payload = make_payload(self.request, TRANSLATEGEMMA)
        self.assertEqual(payload, {"messages": [{"role": "user", "content": expected}]})
        self.assertNotIn("<start_of_turn>", expected)

    def test_installed_chat_template_used_all_three_models_without_encoder(self):
        options = GenerationOptions(num_ctx=4096, num_predict=256, keep_alive=0)
        for model in ("translategemma:4b", "translategemma:12b", "translategemma:27b"):
            transport = Mock(return_value={"done": True, "message": {"content": "Hello"}, "eval_count": 4})
            runtime = OllamaRuntime("http://100.121.133.88:11434", transport=transport)
            result = PreviewTranslator(runtime, model=model, profile=TRANSLATEGEMMA,
                                       generation_options=options).translate(self.request)
            self.assertEqual(result.status, "preview")
            path, body, timeout = transport.call_args.args
            self.assertEqual(path, "/api/chat")
            self.assertEqual(body["options"], {"temperature": 0, "num_ctx": 4096, "num_predict": 256})
            self.assertEqual(body["keep_alive"], 0)
            self.assertEqual(body["messages"], make_payload(self.request, TRANSLATEGEMMA)["messages"])
            self.assertFalse({"format", "raw", "prompt", "think"} & set(body))

    def test_language_names_and_codes_canonicalized_unknowns_rejected(self):
        for source, target in (("Japanese", "English"), ("ja", "en")):
            request = replace(self.request, style=Style(source_language=source, target_language=target))
            self.assertEqual(make_payload(request, TRANSLATEGEMMA), make_payload(self.request, TRANSLATEGEMMA))
        for code in ("jap", "eng", "xx", "Engish", "ja_JP", "English (en)"):
            request = replace(self.request, style=Style(target_language=code))
            runtime = Mock()
            result = PreviewTranslator(runtime, model="translategemma:4b", profile=TRANSLATEGEMMA).translate(request)
            self.assertEqual((result.failure, result.attempts), ("capability", 0))
            runtime.generate.assert_not_called()

    def test_context_style_images_never_injected_into_specialist(self):
        for request in (replace(self.request, mode="B"), replace(self.request, mode="C"),
                        replace(self.request, style=Style(glossary=(("猫", "cat"),))),
                        replace(self.request, style=Style(instructions="special style")),
                        replace(self.request, style=Style(honorific_policy="remove"))):
            with self.assertRaises(ValueError):
                make_payload(request, TRANSLATEGEMMA)
        with self.assertRaises(ValueError):
            make_payload(self.request, TRANSLATEGEMMA, images=("not supplied",))

    def test_alternatives_reduced_capabilities_template_provenance_and_unknown_quality(self):
        runtime = Mock()
        runtime.generate.return_value = "Hello"
        result = PreviewTranslator(runtime, model="translategemma:4b", profile=TRANSLATEGEMMA).translate(self.request)
        metadata = json.loads(result.execution_metadata_json)
        self.assertFalse(metadata["alternatives_supplied_to_model"])
        self.assertEqual(metadata["source_interpretation"], "uncertain")
        self.assertTrue(metadata["alternative_reading_revisions"])
        self.assertEqual(metadata["template"], TRANSLATEGEMMA_PROVENANCE)
        self.assertIn("honorific_policy_not_enforced", metadata["reduced_capabilities"])
        self.assertFalse(metadata["semantic_quality_assessed"])
        self.assertEqual(result.request_revision, self.request.revision)
        self.assertEqual(len(self.request.units[0].readings), 2)

    def test_malformed_and_empty_or_oversize_specialist_output_never_becomes_preview(self):
        for value in (None, [], {}, 3, True, object(), "", "  ", "x" * 11):
            runtime = Mock()
            runtime.generate.return_value = value
            result = PreviewTranslator(runtime, model="translategemma:4b", profile=TRANSLATEGEMMA,
                generation_options=GenerationOptions(max_output_chars=10)).translate(self.request, observe=lambda event: None)
            self.assertEqual((result.failure, result.translations), ("invalid_output", ()))

    def test_malformed_provider_response_types_retain_raw_error_evidence(self):
        for value in (None, [], ["model"], {"done": True},
                      {"done": True, "message": []}, {"done": True, "message": {"content": 1}},
                      {"done": True, "done_reason": "length", "message": {"content": "partial"}}):
            runtime = OllamaRuntime("http://localhost:11434", transport=Mock(return_value=value))
            events = []
            result = PreviewTranslator(runtime, model="translategemma:4b", profile=TRANSLATEGEMMA).translate(
                self.request, observe=events.append)
            self.assertEqual(result.failure, "provider")
            self.assertEqual(json.loads(events[0]["response_json"]), value)

    def test_profile_model_endpoint_and_control_validation_before_io(self):
        for model, profile in (("qwen3.8:latest", TRANSLATEGEMMA), ("translategemma:4b", GENERALIST),
                               ("translategemma:latest", TRANSLATEGEMMA), ("bad model", GENERALIST),
                               ("http://model", GENERALIST)):
            with self.assertRaises(ValueError):
                PreviewTranslator(Mock(), model=model, profile=profile)
        for endpoint in ("http://localhost:bad", "http://localhost:0", "http://localhost:65536",
                         "http://local host", "http://user:password@localhost", "http://localhost?x=1",
                         "http://localhost/path", "http://local_host", "http://localhost\\path"):
            with self.assertRaises(ValueError):
                OllamaRuntime(endpoint)
        for kwargs in ({"num_ctx": 1}, {"num_ctx": True}, {"num_predict": -1}, {"num_predict": 8193},
                       {"keep_alive": "5m"}, {"keep_alive": -1}, {"max_output_chars": 131073}):
            with self.assertRaises(ValueError):
                GenerationOptions(**kwargs)

    def test_generalist_keeps_its_own_schema_prompt_with_identical_controls(self):
        runtime = Mock()
        runtime.generate.return_value = response(self.request.target_ids)
        options = GenerationOptions(num_ctx=4096, num_predict=256, keep_alive=0)
        result = PreviewTranslator(runtime, generation_options=options).translate(self.request)
        self.assertEqual(result.status, "preview")
        payload = runtime.generate.call_args.kwargs["payload"]
        self.assertIn("format", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["options"], options.transport_fields()["options"])
        self.assertEqual(payload["keep_alive"], 0)

    def test_native_cli_needs_no_spec_and_records_controls_before_parent_live_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "saved.json"
            manifest.write_text(json.dumps(ordered_manifest()), encoding="utf-8")
            transport = Mock(return_value={"done": True, "message": {"content": "Hello"}})
            def runtime_factory(endpoint, *, specialist_template):
                self.assertIsNone(specialist_template)
                return OllamaRuntime(endpoint, transport=transport)
            with patch.object(diagnostics, "OllamaRuntime", side_effect=runtime_factory):
                self.assertEqual(diagnostics.main(["--manifest", str(manifest), "--output", str(root / "out"),
                    "--profile", "translategemma", "--model", "translategemma:4b", "--model-digest", "a" * 64,
                    "--model-details", "synthetic", "--run-local-model", "--max-units", "1",
                    "--num-ctx", "4096", "--num-predict", "256", "--keep-alive", "0"]), 0)
            plan = json.loads((root / "out" / "requests.json").read_text())
            self.assertEqual(plan["native_template_provenance"], TRANSLATEGEMMA_PROVENANCE)
            self.assertEqual(plan["generation_options"]["num_ctx"], 4096)
            self.assertIsNone(plan["specialist_template"])
            self.assertFalse(plan["model_sources"][0]["source"]["alternatives_supplied_to_model"])
            self.assertIn("primary_reading_only", plan["requests"][0]["execution_metadata"]["reduced_capabilities"])
            self.assertEqual(transport.call_args.args[0], "/api/chat")

    def test_invalid_native_request_rejected_before_preload_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "saved.json"
            manifest.write_text(json.dumps(ordered_manifest()), encoding="utf-8")
            for options in ({"style": Style(target_language="Engish")}, {"mode": "B"}, {"num_predict": 10000}):
                runtime = Mock()
                with self.assertRaises(ValueError):
                    diagnostics.run(manifest, root / "out", runtime=runtime, profile=TRANSLATEGEMMA,
                        model="translategemma:4b", model_digest="a" * 64, model_details="test",
                        run_local_model=True, preload=True, max_units=1, **options)
                runtime.preload.assert_not_called()
                runtime.generate.assert_not_called()
                self.assertFalse((root / "out").exists())
