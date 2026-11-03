"""Coverage permissions, exact native inputs and scheduling; no live inference."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.translation import test_workflow as fixtures
from tests.local_evidence import local_evidence
from tralens.translation import coverage as c, workflow as w
from tralens.translation.contracts import automatic_units, canonical, TranslationRequest, Reading
from tralens.translation.runtime import (Generation, PreviewTranslator, GENERALIST, TRANSLATEGEMMA,
    make_payload, TRANSLATEGEMMA_PROMPT)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.WorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.batch = replace(self.f.batch, regions=())
        self.f.engine.automatic_unit_proposals = [self.batch]
        self.units = automatic_units(self.batch)
        self.provider = Mock(return_value=Generation('Hello', '{"done":true,"message":{"content":"Hello"}}'))

    def flow(self, name='out', **kwargs):
        kwargs.setdefault('translation_policy', 'coverage-first')
        return self.f.run_flow(name, run_local_model=True, runtime=SimpleNamespace(generate=self.provider), **kwargs)

    def test_default_attempts_original_blocked_text_without_mutating_ocr(self):
        before = w._snapshot(self.batch)
        # Direct call proves the public default, not a test helper override.
        state = w.run_preview(self.f.image, self.f.root/'out', device='cpu', model_root=self.f.weights,
            run_local_ocr=True, run_local_model=True, ocr_factory=self.f.factory,
            runtime=SimpleNamespace(generate=self.provider), max_attempts=64)
        self.assertEqual((state['schema_version'], state['attempts'], state['strict_attempts'], state['coverage_attempts']), (4, 2, 0, 2))
        self.assertEqual((state['strict_preview_count'], state['coverage_preview_count'], state['fragment_preview_count']), (0, 2, 0))
        self.assertEqual(w._snapshot(self.batch), before)
        self.assertEqual(self.f.image.read_bytes(), self.f.content)
        for record, original in zip(state['units'], self.units):
            self.assertEqual(record['unit'], w._snapshot(original))
            self.assertEqual(original.disposition, 'blocked')
            self.assertEqual(record['translation_input']['blocked_reasons'], [])
            self.assertEqual(record['translation_input']['readings'], record['unit']['readings'])
            self.assertEqual(record['translation_input']['selected_revision'], original.selected_revision)
            self.assertEqual(c.bound_result(record, original)['text'], 'Hello')
            metadata = json.loads(record['result']['execution_metadata_json'])
            self.assertEqual(metadata['output_kind'], c.KIND)
            self.assertFalse(metadata['complete_utterance_claimed'])
            self.assertFalse(metadata['source_readability_verified'])
            self.assertTrue(metadata['review_required'])

    def test_conservative_policy_keeps_blocks_and_fragment_optin_separate(self):
        blocked = self.flow('blocked', translation_policy='conservative')
        self.assertEqual(blocked['unit_status_counts'], {'blocked': 2})
        self.provider.assert_not_called()
        self.f.engine.ocr.close.reset_mock()
        fragments = self.flow('fragments', translation_policy='conservative', allow_fragment_previews=True)
        self.assertEqual((fragments['strict_preview_count'], fragments['coverage_preview_count'], fragments['fragment_preview_count']), (0, 0, 2))
        self.assertTrue(all(r['status'] == 'blocked' for r in fragments['units']))

    def test_strict_calls_first_even_if_coverage_is_earlier_on_page(self):
        first = self.f.batch.proposals[0]
        self.f.engine.automatic_unit_proposals = [replace(self.f.batch,
            proposals=(replace(first, crop_safe=False), self.f.batch.proposals[1]))]
        state = self.flow(max_attempts=1, allow_fragment_previews=True)
        self.assertEqual((state['strict_attempts'], state['coverage_attempts'], state['fragment_attempts']), (1, 0, 0))
        self.assertEqual([r['status'] for r in state['units']], ['budget_exhausted', 'preview'])
        self.assertEqual(state['fragment_candidate_inputs'], 0, 'do not duplicate coverage inputs as fragments')
        self.assertEqual([s['input_kind'] for s in state['input_schedule']], ['strict_utterance', c.KIND])

    def test_selected_order_is_kept_within_kind_and_unselected_units_stay_visible(self):
        ids = tuple(u.unit_id for u in reversed(self.units))
        state = self.flow(selected_ids=ids, max_attempts=1)
        self.assertEqual([s['original_unit_id'] for s in state['input_schedule']], list(ids))
        self.assertEqual([r['status'] for r in state['units']], ['budget_exhausted', 'preview'])
        self.f.engine.ocr.close.reset_mock()
        state = self.flow('one', selected_ids=ids[:1])
        self.assertEqual([r['status'] for r in state['units']], ['not_selected', 'preview'])

    def test_grouped_regions_preserve_raw_order_and_exact_native_template(self):
        a, b = self.units
        raw = json.loads(a.selected.evidence_json)+json.loads(b.selected.evidence_json)
        reading = Reading(a.selected.text+b.selected.text, 'raw_main_reading', canonical(raw))
        grouped = replace(a, member_ids=a.member_ids+b.member_ids, order=a.order+b.order,
                          readings=(reading,), selected_revision=reading.revision,
                          blocked_reasons=tuple(c.WAIVABLE))
        derived = c.derive(grouped)
        request = TranslationRequest((derived,), (derived.unit_id,))
        payload = make_payload(request, TRANSLATEGEMMA)
        expected = TRANSLATEGEMMA_PROMPT.format(source_name='Japanese', source_code='ja',
            target_name='English', target_code='en', source_text=reading.text)
        self.assertEqual(payload, {'messages': [{'role': 'user', 'content': expected}]})
        self.assertEqual(derived.order, grouped.order)
        self.assertEqual(grouped.disposition, 'blocked')
        self.assertNotEqual(derived.revision, grouped.revision)

    def test_hard_structural_faults_and_review_only_inputs_cannot_be_waived(self):
        for reason in ('unresolved_annotation_relationship', 'ambiguous_annotation_parents',
                       'annotation_role_unresolved_not_independent_text_proof', 'new_unknown_fault'):
            for field in ('blocked_reasons', 'structural_reasons'):
                unit = replace(self.units[0], **{field: getattr(self.units[0], field)+(reason,)})
                with self.subTest(reason=reason, field=field), self.assertRaises(ValueError):
                    c.derive(unit)
        from tralens.translation.contracts import fragment_plan
        fragment = fragment_plan(self.units[0], allow_fragment_previews=True)[0][0]
        with self.assertRaises(ValueError):
            c.derive(fragment)

    def test_rejected_ambiguous_malformed_or_changed_raw_evidence_stays_blocked(self):
        unit = self.units[0]
        raw = json.loads(unit.selected.evidence_json)
        variants = [[], [{**raw[0], 'status': 'rejected'}], [{**raw[0], 'source_id': True}],
            [{**raw[0], 'polygon': [[0, 0], [0, 0], [0, 0]]}],
            [{**raw[0], 'metadata_json': canonical({'paddle_association': {'status': 'ambiguous'}})}],
            [{**raw[0], 'metadata_json': canonical({'paddle_association': {'status': 'matched', 'issues': 'bad'}})}],
            [{**raw[0], 'reason': 'unresolved_paddle_association'}], [{**raw[0], 'text': 'changed'}]]
        for evidence in variants:
            reading = replace(unit.selected, evidence_json=canonical(evidence))
            bad = replace(unit, readings=(reading,), selected_revision=reading.revision)
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                c.derive(bad)
        for text in ('', 'x'*1001):
            reading = replace(unit.selected, text=text)
            with self.assertRaises(ValueError):
                c.derive(replace(unit, readings=(reading,), selected_revision=reading.revision))

    def test_every_snapshot_and_review_binding_is_checked(self):
        record = self.flow()['units'][0]
        variants = [dict(input_kind='strict_utterance'), dict(binding_current=False), dict(unit_revision='0'*64),
            dict(coverage={}), dict(translation_input=record['unit']), dict(request_revision='0'*64),
            dict(unit={}), dict(request={}), dict(status='failed')]
        for change in variants:
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.bound_result({**record, **change}, self.units[0])
        for change in (dict(review_needed=False), dict(request_revision='0'*64), dict(translations=[['foreign', 'Hello']]),
                       dict(execution_metadata_json='{}')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.bound_result({**record, 'result': {**record['result'], **change}}, self.units[0])
        bad = copy.deepcopy(record)
        bad['result']['outcomes'][0].update(disposition='preview', reasons=[])
        with self.assertRaises(ValueError):
            c.bound_result(bad, self.units[0])

    def test_generalist_is_review_required_but_can_abstain_on_unrecoverable_text(self):
        unit = c.derive(self.units[0])
        request = TranslationRequest((unit,), (unit.unit_id,))
        for disposition, interpretation, text, reasons in (('preview', 'unverified', 'Hello', []),
                ('abstained', 'unreadable', '', ['unrecoverable OCR'])):
            response = canonical({'translations': [dict(id=unit.unit_id, disposition=disposition,
                source_interpretation=interpretation, translation=text, reasons=reasons)]})
            runtime = SimpleNamespace(generate=Mock(return_value=response))
            result = PreviewTranslator(runtime, model='qwen3.8:latest', profile=GENERALIST).translate(request)
            self.assertEqual(result.status, 'preview' if text else 'abstained')
            self.assertTrue(result.review_needed)
            self.assertEqual(result.outcomes[0].disposition, 'review' if text else 'abstained')
            payload = runtime.generate.call_args.kwargs['payload']
            self.assertIn('Unverified OCR regions', payload['messages'][0]['content'])

    def test_source_drift_and_interruption_keep_started_calls_and_uncertainty_visible(self):
        def drift(**kwargs):
            self.f.image.write_bytes(b'changed')
            return self.provider.return_value
        self.provider.side_effect = drift
        state = self.flow()
        self.assertEqual((state['status'], state['coverage_attempts']), ('failed', 1))
        self.assertTrue(all(r['status'] == 'stale_source' and not r['binding_current'] for r in state['units']))
        self.f.image.write_bytes(self.f.content)
        self.f.engine.ocr.close.reset_mock()
        self.provider.side_effect = KeyboardInterrupt()
        state = self.flow('interrupted')
        self.assertEqual((state['status'], state['coverage_attempts']), ('interrupted', 1))
        self.assertEqual([r['status'] for r in state['units']], ['interrupted', 'not_executed'])

    def test_ocr_only_and_zero_budget_do_not_construct_provider(self):
        with patch('tralens.translation.runtime.OllamaRuntime', side_effect=AssertionError('no inference')):
            state = self.f.run_flow(translation_policy='coverage-first', allow_fragment_previews=True)
            self.assertEqual(state['unit_status_counts'], {'ocr_only': 2})
            self.assertEqual(state['fragment_candidate_inputs'], 0)
            self.f.engine.ocr.close.reset_mock()
            state = self.f.run_flow('zero', translation_policy='coverage-first', run_local_model=True, max_attempts=0)
            self.assertEqual(state['unit_status_counts'], {'budget_exhausted': 2})


SAVED = 'test-results/dev-frontend/7d151a7fa33e473e9901d4e709cb3912/run/preview/preview.json'


class SavedCoverageTests(unittest.TestCase):
    @local_evidence(files=(SAVED,))
    def test_demonstrated_page_expands_from_eight_to_twenty_nine_bound_inputs(self):
        root = Path(__file__).resolve().parents[2]
        preview = json.loads((root/SAVED).read_text())
        units = automatic_units(preview['ocr']['batches'][0])
        strict = [u for u in units if u.disposition == 'preview']
        uncertain = [u for u in units if u.disposition == 'blocked']
        self.assertEqual((len(strict), len(uncertain)), (8, 21))
        self.assertTrue(all(not c.rejection_reasons(u) for u in uncertain))
        for members in ([13, 18, 20], [25, 26, 27], [35]):
            original = next(u for u in uncertain if list(u.member_ids) == members)
            derived = c.derive(original)
            self.assertEqual(derived.selected.text, original.selected.text)
            self.assertEqual(derived.order, original.order)
            self.assertEqual(derived.selected_revision, original.selected_revision)
