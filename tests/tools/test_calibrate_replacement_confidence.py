"""Scoring-only visual labels never become production source or erase inputs."""
import copy
import json
import unittest
from unittest.mock import patch

from tests.rendering import test_natural as fixtures
from tools import calibrate_replacement_confidence as c
from tralens.rendering import automatic as a


class ConfidenceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommandTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.preview = self.f.root / 'preview.json'
        self.baseline = self.f.root / 'baseline'
        self.report = a.run_automatic(preview=self.preview, image=self.f.image,
            fonts=self.f.root / 'fonts.json', output=self.baseline)
        saved = json.loads(self.preview.read_bytes())
        self.labels = {'schema_version': 1, 'purpose': 'model_visual_replacement_observations',
            'reviewer': 'Synthetic test fixture, not a real visual assessment', 'independent_reference': False,
            'split': 'development', 'source_sha256': saved['input']['sha256'],
            'preview_sha256': c.n.digest(self.preview.read_bytes()),
            'baseline_manifest_sha256': c.n.digest((self.baseline / 'manifest.json').read_bytes()),
            'limitations': 'Synthetic contract test only, no model quality inference.',
            'records': [{'index': index, 'member_ids': record['unit']['member_ids'], 'ocr_match': 'match',
                         'replacement': 'acceptable' if record['status'] == 'preview' else 'uncertain',
                         'notes': '<script>escaped fixture notes</script>'}
                        for index, record in enumerate(saved['units'])]}
        self.path = self.f.root / 'labels.json'
        self.write_labels()

    def write_labels(self):
        self.path.write_text(json.dumps(self.labels))

    def run_comparison(self, name='comparison', **kwargs):
        return c.run(preview=self.preview, image=self.f.image, baseline=self.baseline,
                     labels=self.path, output=self.f.root / name, **kwargs)

    def test_offline_all_case_counts_and_real_candidate_binding(self):
        candidate = self.f.root / 'candidate'
        a.run_automatic(preview=self.preview, image=self.f.image, fonts=self.f.root / 'fonts.json',
                        output=candidate, replacement_policy='confidence-gated')
        before = self.path.read_bytes()
        with patch('socket.socket', side_effect=AssertionError('must be offline')):
            result = self.run_comparison(candidate=candidate)
        self.assertEqual(result['all_unit_denominator'], 2)
        self.assertEqual(result['new_inference_calls'], 0)
        self.assertFalse(result['accuracy_probability_calibrated'])
        self.assertFalse(result['independent_reference'])
        self.assertEqual(result['held_out_cases'], 0)
        self.assertEqual(result['actual_candidate']['decisions']['replace_acceptable'], 1)
        self.assertEqual(sum(result['actual_candidate']['decisions'].values()), 2)
        self.assertEqual(self.path.read_bytes(), before)
        html = (self.f.root / 'comparison/review.html').read_text()
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>', html)
        with self.assertRaises(ValueError):
            self.run_comparison()

    def test_uncertain_ocr_labels_are_explicit_not_correct_or_omitted_cases(self):
        for record in self.labels['records']:
            record['ocr_match'] = 'uncertain'
        self.write_labels()
        result = self.run_comparison()
        self.assertIsNone(result['recommended_minimum'])
        for row in result['grid']:
            self.assertEqual(row['ocr_only']['all_units'], 2)
            self.assertEqual(row['ocr_only']['visual_matches'], 0)
            self.assertIsNone(row['ocr_only']['assessed_visual_precision'])

    def test_foreign_stale_missing_truth_claims_and_bad_labels_reject_before_output(self):
        before = copy.deepcopy(self.labels)
        mutations = [lambda d: d.update(source_sha256='0' * 64),
                     lambda d: d.update(preview_sha256='0' * 64),
                     lambda d: d.update(baseline_manifest_sha256='0' * 64),
                     lambda d: d.update(independent_reference=True),
                     lambda d: d.update(split='held-out'),
                     lambda d: d['records'].pop(),
                     lambda d: d['records'][0].update(index=True),
                     lambda d: d['records'][0].update(member_ids=['foreign']),
                     lambda d: d['records'][0].update(replacement='correct'),
                     lambda d: d['records'][0].update(corrected_source='do not import source corrections')]
        for mutate in mutations:
            self.labels = copy.deepcopy(before)
            mutate(self.labels)
            self.write_labels()
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.run_comparison()
            self.assertFalse((self.f.root / 'comparison').exists())

    def test_decision_counts_keep_bad_and_uncertain_outcomes_in_denominator(self):
        labels = [{'replacement': value} for value in ('acceptable', 'unacceptable', 'uncertain')]
        result = c.decision_counts([True, True, False], labels)
        self.assertEqual(sum(result.values()), 3)
        self.assertEqual(result['replace_unacceptable'], 1)
        self.assertEqual(result['preserve_uncertain'], 1)
        with self.assertRaises(ValueError):
            c.decision_counts([True], labels)


if __name__ == '__main__':
    unittest.main()
