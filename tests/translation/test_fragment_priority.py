"""Scheduling hints only: synthetic source/provider data, never model or OCR inference."""
from dataclasses import replace
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tests.translation import test_fragments as fixtures
from tralens.ocr.recognition_units import snapshot_detections
from tralens.ocr.automatic_units import detection_evidence_sha256
from tralens.translation.runtime import Generation, GenerationError
from tralens.translation import workflow as w


class FragmentPriorityTests(unittest.TestCase):
    def setUp(self):
        fixtures.FragmentWorkflowTests.setUp(self)
        self.set_texts(("Ｕｂｕｎｔｕ","ねえ"))

    def set_texts(self,texts):
        by_id = {b.paddle_box_id:b for b in self.fixture.boxes}
        for proposal,text in zip(self.fixture.batch.proposals,texts):
            box = by_id[proposal.main_ids[0]]
            box.paddle_text = box.text = text
            box.ocr_metadata["candidates"]["paddle"] = text
        batch = self.fixture.engine.automatic_unit_proposals[0]
        raw = snapshot_detections(self.fixture.boxes,batch.page)
        self.fixture.engine.automatic_unit_proposals = [replace(batch,detections=raw,evidence_sha256=detection_evidence_sha256(raw))]

    def test_automatic_short_kana_precedes_latin_and_failure_spends_budget(self):
        provider = Mock(side_effect=GenerationError("synthetic",'{"error":"synthetic"}'))
        state = self.fixture.run_flow(run_local_model=True,allow_fragment_previews=True,max_attempts=1,
            runtime=SimpleNamespace(generate=provider))
        self.assertTrue(provider.call_args.kwargs["payload"]["messages"][0]["content"].endswith("ねえ"))
        self.assertEqual(state["attempts"],1)
        self.assertEqual(state["fragment_candidate_inputs"],2)
        self.assertEqual([u["status"] for u in state["units"]],["blocked","blocked"])
        self.assertEqual([u["fragment_previews"][0]["status"] for u in state["units"]],
                         ["fragment_budget_exhausted","fragment_failed"])
        first,second = [u["fragment_previews"][0]["scheduling"] for u in state["units"]]
        self.assertEqual((first["priority_tier"],second["priority_tier"]),(1,0))
        self.assertEqual((first["planned_position"],second["planned_position"]),(2,1))
        self.assertFalse(second["certifies_language_quality_readability_or_dialogue"])

    def test_explicit_selection_preserves_latin_first_and_raw_fullwidth_source(self):
        provider = Mock(return_value=Generation("unverified",'{"text":"unverified"}'))
        selected = tuple(p.proposal_id for p in self.fixture.batch.proposals)
        state = self.fixture.run_flow(run_local_model=True,allow_fragment_previews=True,max_attempts=1,
            selected_ids=selected,runtime=SimpleNamespace(generate=provider))
        self.assertTrue(provider.call_args.kwargs["payload"]["messages"][0]["content"].endswith("Ｕｂｕｎｔｕ"))
        self.assertEqual([u["fragment_previews"][0]["status"] for u in state["units"]],
                         ["fragment_preview","fragment_budget_exhausted"])
        for unit in state["units"]:
            scheduling = unit["fragment_previews"][0]["scheduling"]
            self.assertEqual(scheduling["reason"],"explicit_selected_ids_order")
            self.assertIsNone(scheduling["priority_tier"])

    def test_stable_within_tiers_and_no_source_or_alternative_mutation(self):
        for index,texts in enumerate((("あ","漢"),("Ｕｂｕｎｔｕ","ＡＢＣ"),("ﾈｴ","ドン"))):
            self.set_texts(texts)
            box = self.fixture.boxes[0]
            box.ocr_metadata["candidates"]["other_engine"] = "異なる候補"
            self.set_texts(texts)  # bind the additional retained evidence, not a new reading selection
            before = copy.deepcopy(w._snapshot([w._raw_box(b) for b in self.fixture.boxes]))
            provider = Mock(return_value=Generation("unverified",'{"text":"unverified"}'))
            state = self.fixture.run_flow("stable-"+str(index),run_local_model=True,allow_fragment_previews=True,
                max_attempts=2,runtime=SimpleNamespace(generate=provider))
            self.assertEqual([call.kwargs["payload"]["messages"][0]["content"].split("\n\n\n",1)[1]
                              for call in provider.call_args_list],list(texts))
            self.assertEqual(w._snapshot([w._raw_box(b) for b in self.fixture.boxes]),before)
            self.assertEqual(state["fragment_candidate_inputs"],2)
            self.assertEqual(len(state["units"]),2)

    def test_explicit_order_never_moves_fragment_before_strict_input(self):
        batch = self.fixture.engine.automatic_unit_proposals[0]
        self.fixture.engine.automatic_unit_proposals = [replace(batch,proposals=(batch.proposals[0],self.fixture.batch.proposals[1]))]
        selected = tuple(p.proposal_id for p in batch.proposals)
        state = self.fixture.run_flow(run_local_model=True,allow_fragment_previews=True,max_attempts=1,selected_ids=selected,
            runtime=SimpleNamespace(generate=Mock(side_effect=RuntimeError("synthetic strict failure"))))
        self.assertEqual((state["strict_attempts"],state["fragment_attempts"]),(1,0))
        self.assertEqual(state["input_schedule"][0]["input_kind"],"strict_utterance")
        self.assertEqual(state["units"][0]["fragment_previews"][0]["status"],"fragment_budget_exhausted")

    def test_strict_default_and_zero_budget_never_construct_provider(self):
        with unittest.mock.patch("tralens.translation.runtime.OllamaRuntime",side_effect=AssertionError("no model")):
            strict = self.fixture.run_flow("strict",run_local_model=True)
            inventory = self.fixture.run_flow("inventory",run_local_model=True,allow_fragment_previews=True,max_attempts=0)
        self.assertEqual((strict["attempts"],strict["fragment_candidate_inputs"]),(0,0))
        self.assertNotIn("input_schedule",strict)
        self.assertEqual((inventory["attempts"],inventory["fragment_candidate_inputs"]),(0,2))
        self.assertTrue(all(u["fragment_previews"][0]["status"] == "fragment_budget_exhausted" for u in inventory["units"]))


class ScriptHintTests(unittest.TestCase):
    def test_unicode_short_dialogue_kanji_only_sfx_and_fullwidth(self):
        first = ("あ","ねえ","漢","𠮷","ｶﾀｶﾅ","か\u3099","ドン","ㇰ","𛀁","ＡとＢ","CZん","中文","﨑")
        later = ("hello","Ｕｂｕｎｔｕ","ＡＢＣ","※","22","１２３","fm","한글","","…","ー")
        for text in first+later:
            info = w.fragment_scheduling_info(text)
            self.assertEqual(info["priority_tier"],0 if text in first else 1,text)
            self.assertEqual(info["contains_kana_or_han"],text in first,text)
            self.assertFalse(info["certifies_language_quality_readability_or_dialogue"])
            self.assertEqual(info,w.fragment_scheduling_info(text))

    def test_explicit_mode_only_reports_hint_without_applying_tier(self):
        for text in ("ねえ","ＡＢＣ"):
            info = w.fragment_scheduling_info(text,explicit_selection=True)
            self.assertIsNone(info["priority_tier"])
            self.assertEqual(info["reason"],"explicit_selected_ids_order")
