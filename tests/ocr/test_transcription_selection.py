"""Checked selections are independent outputs; all callbacks/models forbidden."""
import copy
import json
import unittest
from dataclasses import FrozenInstanceError, asdict, replace

from tralens.ocr.recognition_units import PageProvenance
from tralens.ocr import transcription_selection as s


def fixture():
    page = PageProvenance("page", "/original.png", "1"*64, "2"*64, "3"*64)
    crop = {"page": asdict(page), "unit_id": "unit", "member_ids": ["main", "annotation"],
            "bounds": [10, 20, 30, 60], "reviewed_extent": {"source": "original punctuation review"}}
    candidate = s.bind_candidate("unit", ("main", "annotation"), page, 0, "ゃー！", "manga", crop)
    review = s.TranscriptionReview("unit", candidate.member_ids, page, candidate.candidate_id,
        candidate.text, candidate.crop_bounds, "existing_candidate", "provisional_visual", "PM", "visual review",
        True, True, True, True, False, False, False, False)
    return crop, candidate, review


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.crop, self.candidate, self.review = fixture()

    def result(self, review=None, candidates=None):
        return s.select_transcription([self.candidate] if candidates is None else candidates,
                                      [self.review if review is None else review])

    def test_existing_candidate_selection_keeps_exact_text_and_provisional_kind(self):
        result = self.result()
        self.assertEqual(result.errors, ())
        self.assertEqual(result.selected.text, "ゃー！")
        self.assertEqual(result.selected.review_kind, s.ReviewKind.PROVISIONAL_VISUAL)
        self.assertEqual(result.selected.review, self.review)
        self.assertFalse(hasattr(result.selected, "accepted"))
        with self.assertRaises(FrozenInstanceError):
            result.selected.text = "changed"

    def test_corrected_text_is_separate_and_raw_candidate_crop_inputs_unchanged(self):
        before = copy.deepcopy(self.crop)
        review = replace(self.review, mode="corrected_text", corrected_text=" ゃーー！ ")
        result = self.result(review)
        self.assertEqual(result.selected.text, " ゃーー！ ")
        self.assertEqual(result.selected.candidate.text, "ゃー！")
        self.assertEqual(self.crop, before)
        self.crop["bounds"][0] = 0
        self.assertEqual(json.loads(self.candidate.crop_json)["bounds"], [10, 20, 30, 60])

    def test_all_checked_and_uncertainty_flags_strict_and_required(self):
        checks = ("reviewed", "wording_checked", "order_checked", "full_utterance_checked")
        uncertainty = ("wording_uncertain", "order_uncertain", "coverage_uncertain", "punctuation_uncertain")
        for field in checks + uncertainty + ("independence_confirmed",):
            for value in (1, "true", None, [], {}):
                with self.subTest(field=field, value=value):
                    self.assertIsNone(self.result(replace(self.review, **{field: value})).selected)
        for field in checks:
            self.assertIsNone(self.result(replace(self.review, **{field: False})).selected)
        for field in uncertainty:
            self.assertIsNone(self.result(replace(self.review, **{field: True})).selected)

    def test_warning_dash_uncertainty_never_promotes_even_selected_candidate(self):
        result = self.result(replace(self.review, punctuation_uncertain=True))
        self.assertIsNone(result.selected)
        self.assertIn("incomplete_or_uncertain_review", result.errors)

    def test_stale_foreign_ids_pages_crops_and_text_fail_closed(self):
        bad = [replace(self.review, unit_id="foreign"), replace(self.review, member_ids=("main",)),
               replace(self.review, member_ids=(True, "annotation")), replace(self.review, candidate_id="0"*64),
               replace(self.review, candidate_text="やー！"), replace(self.review, crop_bounds=(10, 20, 30, 59)),
               replace(self.review, crop_bounds=(True, 20, 30, 60))]
        for field in ("page_id", "image_path", "image_sha256", "comparison_sha256", "pixels_sha256"):
            bad.append(replace(self.review, page=replace(self.review.page,
                **{field: "4"*64 if field.endswith("sha256") else "foreign"})))
        for review in bad:
            self.assertIsNone(self.result(review).selected)

    def test_candidate_and_crop_evidence_drift_and_identity_are_checked(self):
        for candidate in (replace(self.candidate, text="changed"), replace(self.candidate, engine="other"),
                          replace(self.candidate, crop_bounds=(10, 20, 30, 59)),
                          replace(self.candidate, candidate_index=1),
                          replace(self.candidate, crop_json=self.candidate.crop_json.replace("punctuation", "altered"))):
            self.assertIsNone(self.result(candidates=[candidate]).selected)
        alternate = s.bind_candidate("unit", self.candidate.member_ids, self.candidate.page,
                                     1, "ゃーー！", "manga", self.crop)
        result = self.result(candidates=[self.candidate, alternate])
        self.assertEqual(result.selected.candidate, self.candidate)
        self.assertNotEqual(alternate.candidate_id, self.candidate.candidate_id)

    def test_ambiguous_records_or_candidates_no_partial_selection(self):
        self.assertIsNone(s.select_transcription([self.candidate], [self.review, self.review]).selected)
        self.assertIsNone(s.select_transcription([self.candidate], []).selected)
        self.assertIsNone(s.select_transcription([self.candidate, self.candidate], [self.review]).selected)
        self.assertIsNone(s.select_transcription([], [self.review]).selected)
        self.assertIsNone(s.select_transcription([self.candidate], [{}]).selected)
        self.assertIsNone(self.result(candidates=[replace(self.candidate, candidate_id=[])]).selected)
        self.assertIsNone(self.result(candidates=[{}]).selected)
        collision = s.bind_candidate("unit", self.candidate.member_ids, self.candidate.page,
                                     0, "different", "manga", self.crop)
        self.assertIsNone(self.result(candidates=[self.candidate, collision]).selected)

    def test_invalid_types_modes_reviewers_and_correction_are_unresolved(self):
        for field, value in (("reviewer", " "), ("reviewer", 5), ("source", ""),
                             ("review_kind", "qwen_verified"), ("mode", "auto"),
                             ("corrected_text", "ambiguous"), ("reference_source", None)):
            self.assertIsNone(self.result(replace(self.review, **{field: value})).selected)
        for text in (None, 3, "", " "):
            self.assertIsNone(self.result(replace(self.review, mode="corrected_text", corrected_text=text)).selected)

    def test_independent_reference_requires_separate_transcription_and_source(self):
        independent = replace(self.review, review_kind="independent_reference", mode="corrected_text",
                              corrected_text="ゃーー！", independence_confirmed=True,
                              reference_source="blind independent original-page annotation")
        self.assertEqual(self.result(independent).selected.review_kind, s.ReviewKind.INDEPENDENT_REFERENCE)
        for change in ({"independence_confirmed": False}, {"reference_source": ""},
                       {"mode": "existing_candidate", "corrected_text": None}):
            self.assertIsNone(self.result(replace(independent, **change)).selected)
        self.assertIsNone(self.result(replace(self.review, independence_confirmed=True)).selected)

    def test_binding_rejects_invalid_saved_crop(self):
        for change in ({"unit_id": "other"}, {"member_ids": [True]}, {"bounds": [1, 2, 1, 5]},
                       {"bounds": [1, 2, float("inf"), 5]}, {"page": {}}):
            with self.assertRaises(ValueError):
                s.bind_candidate("unit", self.candidate.member_ids, self.candidate.page, 0,
                                 "text", "manga", dict(self.crop, **change))
