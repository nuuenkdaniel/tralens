"""Visual proposal stage: synthetic pixels and fake sequential providers only."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from tests.translation import test_workflow as fixtures
from tralens.translation import workflow as w
from tralens.translation.contracts import automatic_units, canonical, strict_json, TranslationRequest, fragment_plan, FragmentRequest
from tralens.translation.runtime import Generation, GenerationError, OllamaRuntime, PreviewTranslator, TRANSLATEGEMMA
from tralens.translation import ocr_review as v


def verdict(payload, decision="proposed_correction"):
    data = json.loads(payload["messages"][1]["content"])
    return {"target_id":data["target_id"],"source_ids":data["source_ids"],"original_text":data["original_text"],
        "decision":decision,"proposed_text":data["original_text"]+"訂" if decision=="proposed_correction" else "",
        "reasons":["synthetic visual observation, not truth"],"uncertainty":"unverified"}


class VisualReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.events = []
        self.translate = Mock(side_effect=self.translate_source)

    def translate_source(self, **kwargs):
        text = kwargs["payload"]["messages"][0]["content"].split("\n\n\n",1)[1]
        self.events.append(("translate",text))
        self.assertNotIn("images",kwargs["payload"]["messages"][0])
        return Generation("en:"+text,canonical({"done":True,"message":{"content":"en:"+text}}))

    def review_source(self, **kwargs):
        self.events.append(("review",kwargs["model"]))
        self.fixture.engine.ocr.close.assert_called_once()
        self.assertEqual(len(kwargs["payload"]["messages"][1]["images"]),2)
        text = canonical(verdict(kwargs["payload"]))
        return Generation(text,canonical({"done":True,"message":{"content":text},"eval_count":11,"load_duration":12}))

    def run_flow(self,name="out",reviewer=None,**kwargs):
        return self.fixture.run_flow(name,run_local_model=True,runtime=SimpleNamespace(generate=self.translate),
            review_ocr=True,review_runtime=reviewer,**kwargs)

    def prepare(self):
        unit = automatic_units(self.fixture.batch)[0]
        with Image.open(self.fixture.image) as source:
            image = source.convert("RGB")
        raw = w._snapshot([w._raw_box(b) for b in self.fixture.boxes])
        result = v.prepare_input(unit,image,self.fixture.content,raw,"qwen3.8:latest")
        image.close()
        return unit,result[0],result[1]

    def test_disabled_has_no_provider_calls_and_preserves_original_path(self):
        reviewer = Mock()
        with patch.object(v,"OllamaRuntime",side_effect=AssertionError("review disabled")):
            state = self.fixture.run_flow(run_local_model=True,runtime=SimpleNamespace(generate=self.translate),review_runtime=reviewer)
        reviewer.generate.assert_not_called()
        reviewer.unload.assert_not_called()
        self.assertNotIn("ocr_review",state)
        self.assertNotIn("visual_alternates",state["units"][0])
        self.assertEqual((state["schema_version"],state["attempts"]),(1,2))

    def test_owned_reviewer_phase_release_then_translation_proposed_text_separate(self):
        reviewer = SimpleNamespace(generate=Mock(side_effect=self.review_source))
        def release(model,timeout):
            self.events.append(("release",model))
            return {"response_json":'{"done":true,"done_reason":"unload"}',"wall_seconds":0}
        reviewer.unload = Mock(side_effect=release)
        before = copy.deepcopy(w._snapshot([w._raw_box(b) for b in self.fixture.boxes]))
        with patch.object(v,"OllamaRuntime",return_value=reviewer):
            state = self.run_flow(max_attempts=4)
        self.assertEqual([event[0] for event in self.events],["review","review","release","translate","translate","translate","translate"])
        self.assertEqual((state["ocr_review"]["attempts"],state["ocr_review"]["control_calls"]),(2,1))
        reviewer.unload.assert_called_once_with("qwen3.8:latest",30)
        self.assertEqual((state["attempts"],state["strict_attempts"],state["alternate_attempts"]),(4,2,2))
        self.assertEqual(state["alternate_preview_count"],2)
        self.assertTrue(self.events[-1][1].endswith("訂"))
        for row in state["units"]:
            self.assertEqual(row["unit"]["selected_revision"],row["ocr_review"]["request"]["original"]["selected_revision"])
            self.assertEqual(row["result"]["translations"][0][1],"en:こんにちは")
            alternate = row["visual_alternates"][0]
            self.assertTrue(alternate["result"]["review_needed"])
            self.assertEqual(alternate["result"]["alternates"][0][1],"en:こんにちは訂")
            self.assertEqual(row["ocr_review"]["generation_evidence"][0]["metadata"]["eval_count"],11)
            for view in row["ocr_review"]["views"]:
                self.assertTrue((self.fixture.root/"out"/view["path"]).is_file())
        self.assertEqual(before,w._snapshot([w._raw_box(b) for b in self.fixture.boxes]))
        self.assertEqual(self.fixture.image.read_bytes(),self.fixture.content)

    def test_injected_reviewer_not_unloaded_all_dispositions_and_budget(self):
        for n,decision in enumerate(("unchanged","proposed_correction","unreadable","uncertain")):
            self.fixture.engine.ocr.close.reset_mock()
            def generate(**kwargs):
                data = verdict(kwargs["payload"],decision)
                return Generation(canonical(data),canonical({"message":{"content":canonical(data)}}))
            reviewer = SimpleNamespace(generate=Mock(side_effect=generate),unload=Mock())
            state = self.run_flow("decision-"+str(n),reviewer=reviewer,max_review_attempts=1,max_attempts=4)
            reviewer.generate.assert_called_once()
            reviewer.unload.assert_not_called()
            self.assertEqual(state["ocr_review"]["control_calls"],0)
            self.assertEqual(state["ocr_review"]["units"][0]["status"],"review_"+decision)
            self.assertEqual(state["ocr_review"]["units"][1]["skip_reason"],"review_budget_exhausted")
            self.assertEqual(state["alternate_preview_count"],int(decision=="proposed_correction"))

    def test_invalid_options_rejected_before_ocr_and_no_ocr_only_review(self):
        for options in ({"review_ocr":1},{"max_review_attempts":True},{"max_review_attempts":21},
                        {"review_timeout":121},{"review_timeout":float("nan")},{"review_model":"translategemma:27b"}):
            args = {"review_ocr":True,"run_local_model":True,**options}
            with self.assertRaises(ValueError): self.fixture.run_flow(**args)
        with self.assertRaises(ValueError): self.fixture.run_flow(review_ocr=True)
        with self.assertRaises(ValueError): self.fixture.run_flow(review_ocr=True,run_local_model=True,
            model="qwen3.8",profile="generalist",review_model="qwen3.8:latest")
        self.fixture.factory.assert_not_called()
        self.assertFalse((self.fixture.root/"out").exists())

    def test_zero_review_budget_constructs_no_reviewer(self):
        with patch.object(v,"OllamaRuntime",side_effect=AssertionError("zero review calls")):
            state = self.run_flow(max_review_attempts=0)
        self.assertEqual(state["attempts"],2)
        self.assertEqual((state["ocr_review"]["attempts"],state["ocr_review"]["control_calls"]),(0,0))

    def test_malformed_output_types_truncation_timeout_leave_original_path(self):
        failures = [Generation("malformed",'{"text":"malformed"}'),None,{},Generation("x"*32769,'{"text":"oversized"}'),
                    GenerationError("truncated",'{"done":true,"done_reason":"length"}'),TimeoutError("synthetic")]
        for index,failure in enumerate(failures):
            self.fixture.engine.ocr.close.reset_mock()
            provider = Mock(side_effect=failure) if isinstance(failure,Exception) else Mock(return_value=failure)
            state = self.run_flow("failure-"+str(index),reviewer=SimpleNamespace(generate=provider),max_review_attempts=1)
            self.assertEqual(state["ocr_review"]["units"][0]["status"],"review_failed")
            self.assertEqual((state["ocr_review"]["attempts"],state["attempts"]),(1,2))
            self.assertEqual(state["alternate_preview_count"],0)
            self.assertEqual([r["status"] for r in state["units"]],["preview","preview"])

    def test_strict_parser_rejects_ids_echo_duplicates_wrongtypes_empty_correction(self):
        unit,request,views = self.prepare()
        data = verdict(v.review_payload(request,views))
        invalid = [{**data,"target_id":"foreign"},{**data,"source_ids":["foreign"]},
                   {**data,"original_text":"guessed correction"},{**data,"proposed_text":" "},
                   {**data,"proposed_text":[]},{**data,"reasons":[]},{**data,"uncertainty":"confident"},
                   {**data,"confidence":1.0},{**data,"decision":"unchanged"}]
        for value in invalid:
            with self.assertRaises(ValueError): v.parse_verdict(canonical(value),request,"{}")
        for text in ('{"target_id":"a","target_id":"b"}',"```json\n"+canonical(data)+"\n```",canonical(data)+" trailing"):
            with self.assertRaises(ValueError): v.parse_verdict(text,request,"{}")

    def test_changed_source_polygon_crop_file_and_view_pixels_reject(self):
        unit,request,views = self.prepare()
        for key,value in (("original_unit_revision","stale"),("target_id","foreign"),
                          ("context_bounds",[0,0,1,1]),("target_polygon",[[0,0],[1,0],[1,1]]),
                          ("image_sha256","0"*64)):
            data = strict_json(request.binding_json)
            data[key] = value
            with self.assertRaises(ValueError): replace(request,binding_json=canonical(data))
        with self.assertRaises(v.ReviewBindingError): v.review_payload(request,(b"wrong",views[1]))
        with Image.open(self.fixture.image) as image:
            with self.assertRaises(v.ReviewBindingError):
                v.prepare_input(unit,image.convert("RGB"),b"wrong",w._snapshot([w._raw_box(b) for b in self.fixture.boxes]),request.model)

    def test_view_tamper_after_provider_is_fatal_not_best_effort(self):
        def generate(**kwargs):
            next((self.fixture.root/"out").glob("review-*-original.png")).write_bytes(b"tampered")
            return self.review_source(**kwargs)
        state = self.run_flow(reviewer=SimpleNamespace(generate=generate),max_review_attempts=1)
        self.assertEqual(state["error"]["type"],"SourceChanged")
        self.assertEqual(state["attempts"],0)
        self.translate.assert_not_called()

    def test_cancel_or_source_drift_releases_owned_reviewer_and_never_starts_translator(self):
        for index,drift in enumerate((False,True)):
            self.fixture.engine.ocr.close.reset_mock()
            flag = {"cancel":False}
            def generate(**kwargs):
                answer = self.review_source(**kwargs)
                if drift: self.fixture.image.write_bytes(b"changed image")
                else: flag["cancel"] = True
                return answer
            reviewer = SimpleNamespace(generate=generate,unload=Mock(return_value={"response_json":'{"done":true}',"wall_seconds":0}))
            with patch.object(v,"OllamaRuntime",return_value=reviewer):
                state = self.run_flow("abort-"+str(index),cancelled=lambda:flag["cancel"])
            reviewer.unload.assert_called_once()
            self.assertEqual(state["attempts"],0)
            self.assertEqual(state["status"],"failed" if drift else "interrupted")
            self.translate.assert_not_called()
            self.fixture.image.write_bytes(self.fixture.content)

    def test_release_failure_prevents_large_translator_overlap(self):
        reviewer = SimpleNamespace(generate=self.review_source,unload=Mock(side_effect=GenerationError("release failed",'{"error":"synthetic"}')))
        with patch.object(v,"OllamaRuntime",return_value=reviewer): state = self.run_flow(max_review_attempts=1)
        self.assertEqual((state["status"],state["attempts"]),("failed",0))
        self.assertEqual(state["ocr_review"]["residency"]["release"],"failed")
        self.assertEqual(state["ocr_review"]["control_calls"],1)
        self.assertTrue(state["ocr_review"]["residency"]["evidence"]["raw_response_path"])
        self.translate.assert_not_called()

    def test_blocked_correction_requires_fragment_toggle_and_does_not_clear_flags(self):
        self.fixture.engine.automatic_unit_proposals = [replace(self.fixture.batch,
            proposals=tuple(replace(p,crop_safe=False) for p in self.fixture.batch.proposals))]
        reviewer = SimpleNamespace(generate=Mock(side_effect=self.review_source))
        blocked = self.run_flow("blocked",reviewer=reviewer,max_attempts=4)
        reviewer.generate.assert_not_called()
        self.assertEqual(blocked["attempts"],0)
        self.fixture.engine.ocr.close.reset_mock()
        state = self.run_flow("fragment",reviewer=reviewer,allow_fragment_previews=True,max_review_attempts=1,max_attempts=2)
        self.assertEqual(state["alternate_preview_count"],1)
        self.assertTrue(all(r["status"] == "blocked" for r in state["units"]))
        row = state["units"][0]
        self.assertIsNone(row["result"])
        self.assertEqual(row["visual_alternates"][0]["result"]["alternates"][0][1],"en:こんにちは訂")
        self.assertEqual(row["fragment_previews"][0]["result"]["fragments"][0][1],"en:こんにちは")
        self.assertIn("unsafe_crop",row["blocking_reasons"])

    def test_explicit_selection_budget_and_neighbor_overlap_skip(self):
        chosen = self.fixture.batch.proposals[1].proposal_id
        reviewer = SimpleNamespace(generate=Mock(side_effect=self.review_source))
        state = self.run_flow(reviewer=reviewer,selected_ids=(chosen,),max_review_attempts=1,max_attempts=2)
        reviewer.generate.assert_called_once()
        self.assertEqual(state["ocr_review"]["units"][0]["skip_reason"],"not_selected")
        unit,request,views = self.prepare()
        raw = w._snapshot([w._raw_box(b) for b in self.fixture.boxes])
        target = next(r for r in raw if r["source_id"] == unit.member_ids[0])
        raw.append({**target,"source_id":"neighbor"})
        with Image.open(self.fixture.image) as image:
            with self.assertRaisesRegex(ValueError,"overlaps_nonmember"):
                v.prepare_input(unit,image.convert("RGB"),self.fixture.content,raw,request.model)

    def test_alternate_api_strict_separation_stale_and_fragment_permission(self):
        unit,request,views = self.prepare()
        answer = v.parse_verdict(canonical(verdict(v.review_payload(request,views))),request,"{}")
        alternate = v.AlternateRequest(v.AlternateInput(TranslationRequest((unit,),(unit.unit_id,)),request,answer))
        provider = Mock(return_value="English")
        translator = PreviewTranslator(SimpleNamespace(generate=provider),model="translategemma:27b",profile=TRANSLATEGEMMA)
        with self.assertRaises(TypeError): translator.translate(alternate)
        result = translator.translate_alternate(alternate,allow_visual_alternates=True)
        self.assertEqual(result.status,"alternate_preview")
        self.assertFalse(hasattr(result,"translations"))
        result.assert_current(alternate,unit)
        with self.assertRaises(ValueError): result.assert_current(alternate,replace(unit,evidence_revision="stale"))
        with self.assertRaises(ValueError): translator.translate_alternate(alternate,allow_visual_alternates=1)
        blocked = replace(unit,blocked_reasons=unit.blocked_reasons+("unsafe_crop",))
        with Image.open(self.fixture.image) as source:
            rgb = source.convert("RGB")
        raw = w._snapshot([w._raw_box(b) for b in self.fixture.boxes])
        req,images = v.prepare_input(blocked,rgb,self.fixture.content,raw,request.model)
        rgb.close()
        proposal = v.parse_verdict(canonical(verdict(v.review_payload(req,images))),req,"{}")
        base = FragmentRequest(fragment_plan(blocked,allow_fragment_previews=True)[0][0])
        fragment_alternate = v.AlternateRequest(v.AlternateInput(base,req,proposal))
        provider.reset_mock()
        with self.assertRaises(ValueError): translator.translate_alternate(fragment_alternate,allow_visual_alternates=True)
        provider.assert_not_called()
        with self.assertRaises(ValueError): v.AlternateInput(TranslationRequest((blocked,),(blocked.unit_id,)),req,proposal)

    def test_cancel_before_review_stops_both_providers_without_loading(self):
        reviewer = SimpleNamespace(generate=Mock(),unload=Mock())
        state = self.run_flow(reviewer=reviewer,cancelled=lambda:True)
        self.assertEqual(state["status"],"interrupted")
        self.assertEqual((state["attempts"],state["ocr_review"]["attempts"]),(0,0))
        reviewer.generate.assert_not_called()
        reviewer.unload.assert_not_called()
        self.translate.assert_not_called()

    def test_provider_cancellation_signal_is_not_best_effort_failure(self):
        state = self.run_flow(reviewer=SimpleNamespace(generate=Mock(side_effect=v.ReviewCancelled("synthetic cancel"))))
        self.assertEqual(state["status"],"interrupted")
        self.assertEqual(state["ocr_review"]["attempts"],1)
        self.assertEqual(state["attempts"],0)
        self.translate.assert_not_called()


class ReviewTransportAndCLITests(unittest.TestCase):
    def test_review_adapter_requests_thinking_false_and_tag_scoped_release(self):
        transport = Mock(return_value={"done":True,"message":{"content":"{}"}})
        runtime = OllamaRuntime("http://127.0.0.1:11434",transport=transport)
        runtime.generate(model="qwen3.8:latest",payload={"messages":[],"format":{"type":"object"},**v.REVIEW_OPTIONS.transport_fields()},timeout=30)
        self.assertFalse(transport.call_args.args[1]["think"])
        transport.return_value = {"done":True,"done_reason":"unload"}
        runtime.unload("qwen3.8:latest",30)
        path,body,_ = transport.call_args.args
        self.assertEqual(path,"/api/generate")
        self.assertEqual(body,{"model":"qwen3.8:latest","prompt":"","stream":False,"keep_alive":0})

    def test_cli_real_toggle_dispatch_and_ocr_only_rejection(self):
        import main
        base = ["--image","unused","--output","fresh","--device","cpu","--model-root","local","--run-local-ocr"]
        with self.assertRaises(SystemExit) as rejected: main.main(base+["--review-ocr"])
        self.assertEqual(rejected.exception.code,2)
        with patch.object(w,"run_preview",return_value={"status":"complete","attempts":0}) as run:
            main.main(base+["--run-local-model","--review-ocr","--max-review-attempts","4"])
        self.assertTrue(run.call_args.kwargs["review_ocr"])
        self.assertEqual(run.call_args.kwargs["review_model"],"qwen3.8:latest")
        self.assertEqual(run.call_args.kwargs["model"],"translategemma:27b")
