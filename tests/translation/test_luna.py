import unittest
from unittest.mock import patch

from tests.translation import test_workflow as fixtures
from tralens.translation import luna
from tralens.devui.server import options, RequestError


class LunaTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.WorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def flow(self, **kwargs):
        return self.f.run_flow(model=luna.MODEL, run_local_model=True, max_attempts=1, **kwargs)

    def test_page_batches_bound_replacement_and_abstention(self):
        decisions = {i: dict(id=i, source='こんにちは', english='Hello' if i == 0 else '',
                             reason='dialogue' if i == 0 else 'duplicate') for i in range(2)}
        with patch.object(luna, 'decide', return_value=decisions) as model:
            state = self.flow(related_pages=(self.f.image,))
        self.assertEqual(state['unit_status_counts'], {'preview': 1, 'abstained': 1})
        self.assertEqual(state['attempts'], 1)
        self.assertEqual(model.call_args.kwargs['image'], self.f.image)
        self.assertEqual(model.call_args.kwargs['related'], (self.f.image,))
        self.assertEqual(len(model.call_args.args[0]), 2)

    def test_isolated_uses_no_images_and_respects_region_budget(self):
        with patch.object(luna, 'decide', return_value={0: dict(id=0, source='こんにちは',
                english='Hello', reason='dialogue')}) as model:
            state = self.flow(luna_mode='isolated')
        self.assertEqual(state['unit_status_counts'], {'preview': 1, 'budget_exhausted': 1})
        self.assertIsNone(model.call_args.kwargs['image'])

    def test_reviewed_mode_rechecks_only_declined_targets(self):
        def decide(rows, **kwargs):
            return {r['id']: dict(id=r['id'], source='かな', reason='reading',
                    english='Boom!' if kwargs.get('panel') or r['id'] == 0 else '') for r in rows}
        with patch('tralens.translation.discovery.propose', return_value=[]), patch.object(luna, 'decide', side_effect=decide) as model:
            state = self.f.run_flow(model="gpt-6.2-sol", run_local_model=True, max_attempts=3)
        self.assertTrue(all(call.kwargs["model"] == "gpt-6.2-sol" for call in model.call_args_list))
        self.assertTrue(all(row["result"]["model"] == "gpt-6.2-sol" for row in state["units"]))
        self.assertEqual(state['unit_status_counts'], {'preview': 2})
        self.assertEqual(state['attempts'], 3)
        self.assertTrue(model.call_args_list[0].kwargs['atlas'])
        self.assertTrue(model.call_args_list[1].kwargs['panel'])
        self.assertEqual(len(model.call_args_list[1].args[0]), 1)

    def test_discovered_regions_join_translation_and_duplicate_boxes_are_skipped(self):
        rows = [dict(box=box, source='ドン', english='Boom!', landmark='test panel')
                for box in ([145,180,245,660], [310,870,530,990], [310,870,530,990])]
        def decide(regions, **kwargs):
            return {r['id']: dict(id=r['id'], source=r['ocr'], english='Boom!', reason='test')
                    for r in regions}
        with patch('tralens.translation.discovery.propose',return_value=rows), patch.object(luna,'decide',side_effect=decide):
            state = self.f.run_flow(model=luna.MODEL, run_local_model=True, max_attempts=2,
                                    translation_policy='coverage-first')
        self.assertEqual(state['status'], 'complete', state['error'])
        report = state['frontier_discovery']
        self.assertEqual(len(report['added_ids']),1)
        self.assertEqual(report['skipped'],[[0,'already_covered'],[2,'already_covered']])
        self.assertEqual(state['attempts'],2)
        identity = report['added_ids'][0]
        record = next(r for r in state['units'] if identity in r['unit']['member_ids'])
        self.assertEqual(record['status'],'preview')

    def test_viewer_context_requires_safe_saved_job_ids_and_visual_mode(self):
        for bad in [{'related_jobs': ['../private']}, {'related_jobs': ['a'*32], 'luna_mode': 'isolated'}]:
            with self.assertRaises(RequestError):
                options(dict(mode='translate', translation_model=luna.MODEL, **bad), render_enabled=True)
