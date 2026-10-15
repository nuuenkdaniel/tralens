"""Translation status/provenance guards with a mocked provider only."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tralens.ocr import types as m
from tralens.translation import translator as translate_module
from tests.ocr.test_grouping import box


class TranslationGuardTests(unittest.TestCase):
    def setUp(self):
        self.translator = translate_module.Translate.__new__(translate_module.Translate)
        self.translator.model = "test-only-no-provider"
        self.translator.client = SimpleNamespace(chat=Mock(return_value={
            "message": {"content": '[{"id": 1, "translation": "Hello"}]'}}))

    def test_unresolved_excluded_from_prompt_and_output_provenance_preserved(self):
        accepted = box(10, (10, 10, 20, 40), "こんにちは", .4)
        accepted.set_status(m.OCRStatus.ACCEPTED, "normalized_engine_agreement")
        accepted.ocr_metadata["candidates"]["manga"] = "こんにちは"
        accepted.ocr_metadata.update(fallback_attempts=1, crop_bounds=(8, 8, 22, 42))
        review = box(11, (30, 10, 40, 40), "UNRESOLVED", .4)
        rejected = box(12, (50, 10, 60, 40), "REJECTED", .4)
        rejected.set_status(m.OCRStatus.REJECTED, "test")
        group = m.Text_Group([accepted, review, rejected], method="mixed", status="grouped",
                             direction="vertical", metadata={"source_indices": (0, 1, 2)})
        result = self.translator.translate_groups([group], "synthetic-not-read")
        prompt = self.translator.client.chat.call_args.kwargs["messages"][0]["content"]
        self.assertNotIn("UNRESOLVED", prompt)
        self.assertNotIn("REJECTED", prompt)
        self.assertIn("こんにちは", prompt)
        self.assertEqual(len(result[0].group), 1)
        translated = result[0].group[0]
        self.assertEqual(translated.text, "Hello")
        self.assertEqual(translated.paddle_text, accepted.paddle_text)
        self.assertEqual(translated.confidence, .4)
        self.assertEqual(translated.status, accepted.status)
        self.assertEqual(translated.reason, accepted.reason)
        self.assertEqual(translated.selected_engine, accepted.selected_engine)
        self.assertEqual(translated.ocr_metadata["candidates"], accepted.ocr_metadata["candidates"])
        self.assertEqual(translated.ocr_metadata["crop_bounds"], (8, 8, 22, 42))
        self.assertEqual(translated.ocr_metadata["ocr_selected_text"], "こんにちは")
        self.assertTrue(translated.ocr_metadata["translation_preview_only"])
        self.assertTrue(translated.ocr_metadata["translation_review_needed"])
        self.assertEqual(translated.ocr_metadata["translation_semantic_status"], "legacy_unverified")
        self.assertNotIn("translation_semantic_status", accepted.ocr_metadata)
        self.assertTrue(np.array_equal(translated.paddle_poly, accepted.paddle_poly))
        self.assertEqual(translated.paddle_box_id, 10)
        self.assertEqual(result[0].metadata, group.metadata)
        self.assertEqual(result[0].method, group.method)
        self.assertEqual(result[0].direction, group.direction)
        self.assertEqual(accepted.text, "こんにちは")
        self.assertEqual(len(group.group), 3)

    def test_no_accepted_text_means_no_provider_call_or_renderable_output(self):
        groups = [m.Text_Group([box(1, (10, 10, 20, 40), "review", .4),
                               box(2, (30, 10, 40, 40), "", .99)])]
        self.assertEqual(self.translator.translate_groups(groups, "synthetic"), [])
        self.translator.client.chat.assert_not_called()

    def test_malformed_duplicate_missing_or_provider_failure_never_yields_renderable_text(self):
        accepted = box(10, (10, 10, 20, 40), "こんにちは", .99)
        group = m.Text_Group([accepted])
        for raw in ('prefix [{"id":1,"translation":"Hello"}]',
                    '[{"id":1,"translation":"Hello"},{"id":1,"translation":"Other"}]',
                    '[]', '[{"id":1,"translation":""}]',
                    '[{"id":1,"id":2,"translation":"Hello"}]'):
            with self.subTest(raw=raw):
                self.translator.client.chat.return_value = {"message": {"content": raw}}
                self.assertEqual(self.translator.translate_groups([group], "never-read"), [])
                self.assertEqual(accepted.text, "こんにちは")
        self.translator.client.chat.side_effect = RuntimeError("synthetic failure")
        self.assertEqual(self.translator.translate_groups([group], "never-read"), [])
