"""Explicit fragment previews use synthetic OCR/provider fixtures only."""
import copy
from dataclasses import asdict, replace, FrozenInstanceError
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.translation import test_workflow as workflow_tests
from tralens.translation import workflow as w
from tralens.translation import (FragmentInput, FragmentRequest, FragmentResult, fragment_plan,
    TranslationRequest, TranslationResult, PreviewTranslator, Style, Reading, automatic_units)
from tralens.translation.contracts import canonical
from tralens.translation.runtime import (Generation, GenerationError, GENERALIST, TRANSLATEGEMMA,
    make_payload, make_fragment_payload, TRANSLATEGEMMA_PROMPT)


class FragmentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = workflow_tests.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.engine.automatic_unit_proposals = [replace(self.fixture.batch,
            proposals=tuple(replace(p,crop_safe=False) for p in self.fixture.batch.proposals))]

    def test_optin_preview_never_promotes_blocked_original(self):
        strict = self.fixture.run_flow("strict",run_local_model=True,runtime=Mock())
        self.assertEqual((strict["attempts"],strict["unit_status_counts"]),(0,{"blocked":2}))
        self.fixture.engine.ocr.close.reset_mock()
        before = copy.deepcopy(strict["units"][0]["unit"])
        provider = Mock(return_value=Generation("Hello",'{"message":{"content":"Hello"}}'))
        with patch("tralens.ocr.types.Text_Box.transformed",side_effect=AssertionError("no fragment box transformation")):
            state = self.fixture.run_flow("fragment",run_local_model=True,allow_fragment_previews=True,max_attempts=1,
                runtime=SimpleNamespace(generate=provider))
        self.assertEqual(state["attempts"],1)
        self.assertEqual(state["units"][0]["unit"],before)
        self.assertEqual(state["units"][0]["status"],"blocked")
        fragment = state["units"][0]["fragment_previews"][0]
        self.assertEqual(fragment["status"],"fragment_preview")
        self.assertTrue(fragment["result"]["review_needed"])
        self.assertIsNone(state["units"][0]["result"])
        self.assertEqual((state["strict_attempts"],state["fragment_attempts"]),(0,1))
        self.assertEqual((state["strict_preview_count"],state["fragment_preview_count"]),(0,1))
        self.assertEqual(state["units"][1]["fragment_previews"][0]["status"],"fragment_budget_exhausted")

    def test_fragment_failure_consumes_shared_budget_and_retains_raw(self):
        provider = Mock(side_effect=GenerationError("synthetic",'{"done":false,"error":"synthetic"}'))
        state = self.fixture.run_flow(run_local_model=True,allow_fragment_previews=True,max_attempts=1,
            runtime=SimpleNamespace(generate=provider))
        self.assertEqual((state["status"],state["attempts"],state["fragment_attempts"]),("complete_with_failures",1,1))
        self.assertTrue(all(r["status"] == "blocked" for r in state["units"]))
        first,second = [r["fragment_previews"][0] for r in state["units"]]
        self.assertEqual((first["status"],second["status"]),("fragment_failed","fragment_budget_exhausted"))
        raw = first["generation_evidence"][0]
        self.assertTrue((self.fixture.root/"out"/raw["raw_response_path"]).is_file())
        provider.assert_called_once()

    def test_strict_inputs_run_first_and_share_error_budget_with_fragments(self):
        self.fixture.engine.automatic_unit_proposals = [replace(self.fixture.batch,proposals=(
            replace(self.fixture.batch.proposals[0],crop_safe=False),self.fixture.batch.proposals[1]))]
        provider = Mock(side_effect=RuntimeError("synthetic strict provider failure"))
        state = self.fixture.run_flow(run_local_model=True,allow_fragment_previews=True,max_attempts=1,
            runtime=SimpleNamespace(generate=provider))
        self.assertEqual((state["strict_attempts"],state["fragment_attempts"],state["attempts"]),(1,0,1))
        self.assertEqual(state["units"][0]["fragment_previews"][0]["status"],"fragment_budget_exhausted")
        self.assertEqual(state["units"][1]["status"],"failed")

    def test_fragment_optin_without_model_optin_is_model_free(self):
        with patch("tralens.translation.runtime.OllamaRuntime",side_effect=AssertionError("no client")):
            state = self.fixture.run_flow(allow_fragment_previews=True)
        self.assertEqual((state["attempts"],state["fragment_candidate_inputs"]),(0,2))
        self.assertTrue(all(r["fragment_previews"][0]["status"] == "fragment_ocr_only" for r in state["units"]))

    def test_interrupt_in_fragment_preserves_budget_and_original_blocks(self):
        state = self.fixture.run_flow(run_local_model=True,allow_fragment_previews=True,
            runtime=SimpleNamespace(generate=Mock(side_effect=KeyboardInterrupt())))
        self.assertEqual((state["status"],state["attempts"],state["fragment_attempts"]),("interrupted",1,1))
        self.assertTrue(all(r["status"] == "blocked" for r in state["units"]))
        self.assertEqual([r["fragment_previews"][0]["status"] for r in state["units"]],
            ["fragment_interrupted","fragment_not_executed"])

    def test_source_drift_invalidates_both_parent_and_fragment_bindings(self):
        def generate(**kwargs):
            self.fixture.image.unlink()
            return Generation("Hello",'{"text":"Hello"}')
        state = self.fixture.run_flow(run_local_model=True,allow_fragment_previews=True,
            runtime=SimpleNamespace(generate=generate))
        self.assertEqual(state["error"]["type"],"SourceChanged")
        self.assertFalse(state["units"][0]["binding_current"])
        fragment = state["units"][0]["fragment_previews"][0]
        self.assertEqual(fragment["status"],"fragment_stale_source")
        self.assertFalse(fragment["binding_current"])

    def test_workflow_boolean_is_strict_before_engine_setup(self):
        for value in (1,"true",None,[]):
            with self.assertRaises(ValueError):
                self.fixture.run_flow(allow_fragment_previews=value)
        self.fixture.factory.assert_not_called()


class FragmentContractRuntimeTests(unittest.TestCase):
    def setUp(self):
        FragmentWorkflowTests.setUp(self)
        self.unit = automatic_units(self.fixture.engine.automatic_unit_proposals[0])[0]
        self.fragment = fragment_plan(self.unit,allow_fragment_previews=True)[0][0]
        self.request = FragmentRequest(self.fragment)

    def translator(self, runtime, **kwargs):
        return PreviewTranslator(runtime,model="translategemma:27b",profile=TRANSLATEGEMMA,**kwargs)

    def test_immutable_original_and_alternatives_and_derived_binding(self):
        alternative = Reading("another reading","synthetic candidate",canonical({"test":True}))
        parent = replace(self.unit,readings=self.unit.readings+(alternative,))
        before = copy.deepcopy(asdict(parent))
        fragment = fragment_plan(parent,allow_fragment_previews=True)[0][0]
        self.assertIs(fragment.original,parent)
        self.assertEqual(fragment.original.blocking_reasons,parent.blocking_reasons)
        self.assertEqual(fragment.source_reading_revision,parent.selected_revision)
        self.assertEqual(fragment.source_ids,parent.member_ids)
        self.assertNotEqual(fragment.selected_revision,parent.selected_revision)
        self.assertEqual(before,asdict(parent))
        with self.assertRaises(FrozenInstanceError): fragment.original_unit_revision = "changed"

    def test_nonwaivable_faults_combined_and_unknown_structure_stay_blocked(self):
        for fault in ("mixed_region_ownership","unknown_reading_orientation","unresolved_annotation_relationship",
                      "ambiguous_annotation_parents","annotation_role_unresolved_not_independent_text_proof",
                      "unknown_new_structure"):
            parent = replace(self.unit,structural_reasons=self.unit.structural_reasons+(fault,))
            fragments,rejected = fragment_plan(parent,allow_fragment_previews=True)
            self.assertEqual(fragments,())
            self.assertTrue(any(fault in reason for reason in rejected))
        combined = replace(self.unit,member_ids=self.unit.member_ids+("other",),order=("other",)+self.unit.order)
        fragments,rejected = fragment_plan(combined,allow_fragment_previews=True)
        self.assertEqual(fragments,())
        self.assertIn("requires_independent_singleton_raw_input",rejected)

    def test_blank_missing_foreign_or_ambiguous_raw_evidence_rejected(self):
        empty = replace(self.unit.selected,text="")
        parent = replace(self.unit,readings=(empty,),selected_revision=empty.revision)
        self.assertFalse(fragment_plan(parent,allow_fragment_previews=True)[0])
        raw = json.loads(self.unit.selected.evidence_json)
        variants = [[],[{**raw[0],"source_id":"foreign"}], [{**raw[0],"status":"rejected"}],
            [{**raw[0],"metadata_json":canonical({"paddle_association":{"status":"ambiguous","issues":["mismatch"]}})}]]
        for evidence in variants:
            selected = replace(self.unit.selected,evidence_json=canonical(evidence))
            parent = replace(self.unit,readings=(selected,),selected_revision=selected.revision)
            self.assertFalse(fragment_plan(parent,allow_fragment_previews=True)[0])

    def test_strict_api_cannot_accept_fragment_and_execution_requires_boolean_optin(self):
        runtime = Mock(return_value="unused")
        translator = self.translator(runtime)
        with self.assertRaises(ValueError): TranslationRequest((self.fragment,),(self.fragment.unit_id,))
        with self.assertRaises(TypeError): translator.translate(self.request)
        with self.assertRaises(TypeError): make_payload(self.request,TRANSLATEGEMMA)
        for value in (False,1,"true",None):
            with self.assertRaises(ValueError):
                translator.translate_fragment(self.request,allow_fragment_previews=value)
        for value in (1,"true",None):
            with self.assertRaises(ValueError): fragment_plan(self.unit,allow_fragment_previews=value)
        runtime.generate.assert_not_called()
        strict = translator.translate(TranslationRequest((self.unit,),(self.unit.unit_id,)))
        self.assertEqual((strict.status,strict.attempts),("blocked",0))
        with self.assertRaises(TypeError): strict.assert_current(self.request)

    def test_native_prompt_unchanged_fragment_output_review_only_and_stale_rejected(self):
        runtime = Mock()
        runtime.generate.return_value = "Hello"
        result = self.translator(runtime).translate_fragment(self.request,allow_fragment_previews=True)
        self.assertIsInstance(result,FragmentResult)
        self.assertNotIsInstance(result,TranslationResult)
        self.assertFalse(hasattr(result,"translations"))
        self.assertEqual((result.status,result.review_needed),("fragment_preview",True))
        metadata = json.loads(result.execution_metadata_json)
        self.assertFalse(metadata["source_readability_verified"])
        self.assertFalse(metadata["complete_utterance_claimed"])
        payload = runtime.generate.call_args.kwargs["payload"]
        expected = TRANSLATEGEMMA_PROMPT.format(source_name="Japanese",source_code="ja",target_name="English",target_code="en",source_text=self.unit.selected.text)
        self.assertEqual(payload["messages"],[{"role":"user","content":expected}])
        self.assertNotIn("format",payload)
        result.assert_current(self.request,self.unit)
        changed = replace(self.unit,evidence_revision="different page/member evidence")
        with self.assertRaises(ValueError): result.assert_current(self.request,changed)
        with self.assertRaises(ValueError): replace(self.fragment,original=changed)
        with self.assertRaises(TypeError): result.assert_current(TranslationRequest((self.unit,),(self.unit.unit_id,)),self.unit)
        with self.assertRaises(ValueError): replace(result,review_needed=False)

    def test_no_images_or_context_and_no_render_permission(self):
        runtime = Mock()
        result = self.translator(runtime).translate_fragment(self.request,allow_fragment_previews=True,images=("data",))
        self.assertEqual((result.status,result.failure,result.attempts),("fragment_failed","capability",0))
        runtime.generate.assert_not_called()
        self.assertIn("no ownership, crop, image, recognition, rendering or erasure permission",self.fragment.provenance["permissions"])
        with self.assertRaises(TypeError): FragmentRequest(self.fragment,mode="C")
        from tralens.translation.runtime import CapabilityProfile
        with self.assertRaises(ValueError):
            make_fragment_payload(self.request,CapabilityProfile("image-capable control",True,True,True),images=("data",))

    def test_generalist_abstention_and_fragment_failure_never_retry(self):
        runtime = Mock()
        identifier = self.request.target_ids[0]
        runtime.generate.return_value = canonical({"translations":[{"id":identifier,"translation":"","disposition":"abstained",
            "reasons":["genuinely unrecoverable source"],"source_interpretation":"unreadable"}]})
        translator = PreviewTranslator(runtime,model="qwen3.8:latest",profile=GENERALIST,max_attempts=3)
        result = translator.translate_fragment(self.request,allow_fragment_previews=True)
        self.assertEqual((result.status,result.fragments),("fragment_abstained",()))
        runtime.generate.side_effect = RuntimeError("synthetic")
        runtime.generate.reset_mock()
        result = translator.translate_fragment(self.request,allow_fragment_previews=True)
        self.assertEqual((result.status,result.attempts),("fragment_failed",1))
        runtime.generate.assert_called_once()

    def test_fragment_cancellation_before_and_after_provider(self):
        runtime = Mock()
        runtime.generate.return_value = "Hello"
        translator = self.translator(runtime)
        result = translator.translate_fragment(self.request,allow_fragment_previews=True,cancelled=lambda:True)
        self.assertEqual((result.status,result.attempts),("fragment_cancelled",0))
        runtime.generate.assert_not_called()
        result = translator.translate_fragment(self.request,allow_fragment_previews=True,cancelled=Mock(side_effect=[False,True]))
        self.assertEqual((result.status,result.attempts,result.fragments),("fragment_cancelled",1,()))

    def test_generalist_full_provenance_budget_rejected_before_provider(self):
        parent = replace(self.unit,unit_id="x"*65536)
        fragment = fragment_plan(parent,allow_fragment_previews=True)[0][0]
        runtime = Mock()
        result = PreviewTranslator(runtime,model="qwen3.8:latest",profile=GENERALIST).translate_fragment(
            FragmentRequest(fragment),allow_fragment_previews=True)
        self.assertEqual((result.status,result.failure,result.attempts),("fragment_failed","capability",0))
        runtime.generate.assert_not_called()


class FragmentCLITests(unittest.TestCase):
    def test_cli_flag_is_explicit_and_preserves_native_default(self):
        import main
        base = ["--image","unused","--output","fresh","--device","cpu","--model-root","local","--run-local-ocr"]
        for extra,expected in (([],False),(["--allow-fragment-previews"],True)):
            with patch.object(w,"run_preview",return_value={"status":"complete","attempts":0}) as run:
                self.assertEqual(main.main(base+extra),0)
            self.assertIs(run.call_args.kwargs["allow_fragment_previews"],expected)
            self.assertEqual(run.call_args.kwargs["model"],"translategemma:27b")
            self.assertFalse(run.call_args.kwargs["run_local_model"])
        with self.assertRaises(SystemExit) as rejected:
            main.main(base+["--allow-fragment-previews=true"])
        self.assertEqual(rejected.exception.code,2)
