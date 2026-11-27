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

    def test_overlapping_marks_need_an_independently_read_character(self):
        self.assertFalse(luna.supported_mark('っ',['っ2','フハ','フフ','フ']))
        self.assertFalse(luna.supported_mark('つん',['っ2','フハ','フフ','フ']))
        self.assertTrue(luna.supported_mark('コツッ',['?','こつ','…','し']))
        self.assertTrue(luna.supported_mark('ッ',['!','っ！','フ','く']))

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

    def test_malformed_or_duplicate_regions_do_not_invalidate_siblings(self):
        def decision(i):
            return dict(id=i, source='文字', english='Text', reason='legible', category='translation')
        rows = [dict(id=i, ocr='文字') for i in range(4)]
        reply = [decision(1), decision(0), decision(0), decision(19),
                 {**decision(2), 'category': 'duplicate'}, None, decision(True)]
        with patch.object(luna.frontier, 'infer', return_value={'regions': reply}):
            self.assertEqual(luna.decide(rows), {1: decision(1)})

    def test_retry_only_missing_region_and_do_not_append_it_to_reading_order(self):
        def decision(i):
            return {i: dict(id=i, source='文字', english='Text', reason='legible', category='translation')}
        with patch('tralens.translation.discovery.propose', return_value=[]), \
                patch.object(luna, 'decide', side_effect=[decision(1), decision(0)]) as model:
            state = self.f.run_flow(model=luna.MODEL, run_local_model=True, luna_mode='page', max_attempts=3)
        self.assertEqual(state['unit_status_counts'], {'preview': 2})
        self.assertEqual([r['id'] for r in model.call_args.args[0]], [0])
        self.assertEqual([r['attempts_started'] for r in state['units']], [2, 1])
        self.assertTrue(all('luna_page_order' not in r for r in state['units']))

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

    def test_blank_outline_can_be_read_without_duplicate_membership(self):
        from PIL import Image
        from tralens.ocr.types import Text_Box
        from tralens.translation import discovery
        old, kept = self.f.boxes
        empty = Text_Box(old.box_id,0,old.poly,'')
        self.f.engine.raw_detections = [[empty,kept]]
        with Image.open(self.f.image) as im:
            size = im.size
        coords = [round(v*1000/size[i%2]) for i,v in enumerate(empty.bbox)]
        row = dict(source='パタン',english='Clack!',box=coords,landmark='door')
        with patch.object(discovery,'propose',return_value=[row,row]) as scan:
            report = discovery.extend(self.f.engine,self.f.image,size,luna.MODEL,30)
        self.assertEqual(scan.call_args.args[1],[kept.paddle_poly.tolist()])
        self.assertEqual(report['recovered_ids'],[empty.box_id])
        self.assertEqual(report['skipped'],[[1,'already_covered']])
        unchanged,recovered = self.f.engine.raw_detections[0]
        self.assertIs(unchanged,kept)
        self.assertNotEqual(recovered.box_id,empty.box_id)
        self.assertEqual(recovered.ocr_metadata['empty_detection']['text'],'')
        self.assertEqual(recovered.paddle_text,'パタン')

    def test_repeated_beats_can_share_a_corner_but_not_a_center_or_another_word(self):
        from types import SimpleNamespace
        from tralens.translation import discovery
        engine = SimpleNamespace(raw_detections=[[]])
        def glyph(word,x):
            return {'source':word,'english':'beat','box':[x,10,x+100,110],'landmark':'strip'}
        proposals = [glyph('パ',10),glyph('パ',80),glyph('パ',240),glyph('パ',400),
                     glyph('パ',82),glyph('タン',470)]
        with patch.object(discovery,'propose',return_value=proposals), patch('tralens.ocr.automatic_units.propose_page_from_boxes'):
            report = discovery.extend(engine,self.f.image,(1000,1000),luna.MODEL,30)
        self.assertEqual(len(report['added_ids']),4)
        self.assertEqual(report['skipped'],[[4,'already_covered'],[5,'already_covered']])

    def test_second_scan_for_repeated_ink_reserves_the_translation_call(self):
        def scan(*args, **kwargs):
            if kwargs.get('repeated'):
                return {'proposed':2,'added_ids':[12]}
            return {'proposed':5,'added_ids':[8,9,10,11],'repeated_kana':['ド']}
        def word(rows, **_):
            return {r['id']:dict(id=r['id'],source='こんにちは',english='Hi',reason='dialogue') for r in rows}
        for budget,expected in ((2,1),(3,2)):
            with patch('tralens.translation.discovery.extend', side_effect=scan) as detector, patch.object(luna,'decide',side_effect=word):
                state = self.f.run_flow(name='repeated-'+str(budget),model=luna.MODEL,
                    luna_mode='page',run_local_model=True,max_attempts=budget)
            self.assertEqual(detector.call_count,expected)
            self.assertEqual(state['attempts'],budget)
            self.assertEqual(state['unit_status_counts'],{'preview':2})
            if expected==2:
                self.assertEqual(state['frontier_discovery']['proposed'],7)
                self.assertEqual(state['frontier_discovery']['added_ids'],[8,9,10,11,12])

    def test_viewer_context_requires_safe_saved_job_ids_and_visual_mode(self):
        for bad in [{'related_jobs': ['../private']}, {'related_jobs': ['a'*32], 'luna_mode': 'isolated'}]:
            with self.assertRaises(RequestError):
                options(dict(mode='translate', translation_model=luna.MODEL, **bad), render_enabled=True)
