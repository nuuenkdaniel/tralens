"""Portable contract, routing and failure tests. No weights/runtime/network needed."""
from copy import deepcopy
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import reconstruction as rec, aggressive, automatic, natural as n
from . import test_automatic as fixtures


class Contracts(unittest.TestCase):
    def test_broad_box_preserves_visible_branches_and_rejects_ambiguous_ink(self):
        import cv2
        from tralens.rendering import stroke_repair
        truth = np.full((96, 96, 3), (244, 219, 168), np.uint8)
        for endpoint in [(0, 12), (95, 34), (21, 95)]:
            cv2.line(truth, (50, 48), endpoint, (155, 137, 99), 6)
        mask = np.zeros((96, 96), bool)
        mask[25:74, 22:76] = True
        ink = np.zeros(mask.shape, np.uint8)
        cv2.putText(ink, 'H', (30, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1, 5)
        crop = truth.copy()
        crop[ink > 0] = 24
        blurred = cv2.GaussianBlur(truth, (25, 25), 8)
        repaired = stroke_repair.refine(crop, mask, mask, blurred)
        self.assertIsNotNone(repaired)
        np.testing.assert_array_equal(repaired[ink == 0], truth[ink == 0])
        error = lambda img: np.abs(img.astype(float) - truth)[mask].mean()
        self.assertLess(error(repaired), error(blurred))
        antialias = np.zeros(mask.shape, np.uint8)
        cv2.putText(antialias, 'H', (30, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 5, cv2.LINE_AA)
        alpha = antialias[:, :, None] / 255
        printed = (truth * (1 - alpha) + 24 * alpha).round().astype(np.uint8)
        antialiased_repair = stroke_repair.refine(printed, mask, mask, blurred)
        self.assertIsNotNone(antialiased_repair)
        self.assertLess(error(antialiased_repair), error(blurred))
        crop[:, 18:22] = 24  # Surrounding art shares the lettering color.
        self.assertIsNone(stroke_repair.refine(crop, mask, mask, blurred))

    def test_auto_selects_per_region_and_uses_available_models(self):
        models = {key: {'backend': key} for key in ('lama', 'sdxl')}
        router = rec.LocalReconstruction({'backend': 'auto', 'models': models})
        mask = np.zeros((96, 96), bool)
        mask[30:65, 30:65] = True
        smooth = np.full((96, 96, 3), 127, np.uint8)
        self.assertIsNone(rec.region_backend(router, smooth, mask))
        ink = smooth.copy()
        ink[:, 45:48] = 0
        self.assertEqual(rec.region_backend(router, ink, mask).config, models['lama'])
        ink[:, ::3] = 0
        self.assertEqual(rec.region_backend(router, ink, mask).config, models['lama'])
        painted = smooth.copy()
        painted[:, :, 0] = np.arange(96) * 2
        painted[:, :, 2] = 20
        self.assertEqual(rec.region_backend(router, painted, mask).config['prefill'], 'classical_context')
        del models['sdxl']
        self.assertEqual(rec.region_backend(router, painted, mask).config, models['lama'])
        manual = rec.LocalReconstruction(models['lama'])
        self.assertIs(rec.region_backend(manual, smooth, mask), manual)
        self.assertEqual(rec.configuration(None, 'auto')['models'], {})

    def test_broken_line_can_win_a_retry_with_equal_outline_scores(self):
        import cv2
        from tralens.rendering import line_continuity

        truth = np.full((80, 140, 3), 185, np.uint8)
        cv2.line(truth, (0, 62), (139, 17), (38, 38, 38), 3)
        mask = np.zeros((80, 140), bool)
        mask[21:60, 60:78] = True
        crop = truth.copy()
        crop[mask] = 15
        broken = truth.copy()
        broken[mask] = 185
        req = rec.request({'size': [140,80]}, {'member_ids':['a'], 'request_revision':'x', 'text':'A'},
                          [0,0,140,80], crop, mask, mask, np.zeros_like(mask), attempt=0,
                          model={'backend':'sdxl-inpaint'})
        router = rec.LocalReconstruction({'backend':'auto','models':{'lama':{'backend':'lama-onnx-512'}}})
        def complete(retry, *args, **kwargs):
            return {'pixels':truth, 'request_revision':retry['revision']}
        with patch.object(rec, 'outline_strength', return_value=.5), \
             patch.object(rec.LocalReconstruction, 'reconstruct', side_effect=complete):
            budget = rec.Budget(calls=1)
            _, _, answer = rec.review_outline(router, req, {'pixels':broken, 'request_revision':req['revision']},
                                               crop, mask, mask, np.zeros_like(mask), budget, float('inf'))
        self.assertEqual(answer['outline_review']['reason'], 'line_continuity')
        self.assertEqual(budget.calls, 2)
        # A stronger absolute outline vetoes the line winner unless its local
        # contrast is no greater at the old strokes than at both flanks.
        for excess, expected in ((-.2, 'lama-onnx-512'), (.4, 'sdxl-inpaint'), (None, 'sdxl-inpaint')):
            with self.subTest(excess=excess), \
                 patch.object(rec, 'outline_strength', side_effect=[1., 2., excess]), \
                 patch.object(rec.LocalReconstruction, 'reconstruct', side_effect=complete):
                _, _, answer = rec.review_outline(router, req, {'pixels':broken, 'request_revision':req['revision']},
                                                  crop, mask, mask, np.zeros_like(mask), rec.Budget(), float('inf'))
                self.assertEqual(answer['outline_review']['selected'], expected)
        # A larger clearance box hides a longer section of the same line.
        wide = np.zeros_like(mask)
        wide[12:68, 46:94] = True
        covered = truth.copy()
        covered[wide] = 15
        crossings = line_continuity.bridges(covered, wide, wide)
        self.assertTrue(crossings)
        self.assertLess(line_continuity.error(truth, crossings), line_continuity.error(covered, crossings))
        crop[:, 78:] = 185
        self.assertFalse(line_continuity.bridges(crop, mask, mask))

    def test_relative_outline_distinguishes_a_ghost_from_a_continuous_ramp(self):
        crop = np.full((80, 80, 3), 200, np.uint8)
        crop[18:62, 32:45] = 20
        commit = np.zeros((80, 80), bool)
        commit[10:70, 21:56] = True
        ramp = np.broadcast_to((2*np.arange(80, dtype=np.uint8))[None, :, None], crop.shape).copy()
        ghost = ramp.copy()
        ghost[18:62, 32:45] += 15
        self.assertAlmostEqual(rec.outline_strength(crop, ramp, commit, relative=True), 0, places=5)
        self.assertGreater(rec.outline_strength(crop, ghost, commit, relative=True), 1)
        self.assertIsNone(rec.outline_strength(crop, ghost, crop[:, :, 0] == 20, relative=True))

    def test_isoluminant_colored_outline_is_visible_to_review(self):
        mask = np.zeros((50, 50), bool)
        mask[10:40, 21:29] = True
        clean = np.full((50, 50, 3), 120, np.uint8)
        crop = clean.copy()
        crop[mask] = (180, 89, 120)
        ghost = clean.copy()
        ghost[mask] = (140, 110, 120)
        self.assertGreater(rec.outline_strength(crop, ghost, mask), 1.5)
        self.assertEqual(rec.outline_strength(crop, clean, mask), 0)

    def test_auto_outline_challenge_respects_binding_budget_and_improvement(self):
        crop = np.full((64, 64, 3), 180, np.uint8)
        mask = np.zeros(crop.shape[:2], bool)
        mask[20:42, 26:35] = True
        crop[mask] = 20
        protected = np.zeros_like(mask)
        protected[:, :20] = True
        router = rec.LocalReconstruction({'backend': 'auto', 'models': {'lama': {'backend': 'lama-onnx-512'}}})
        req = rec.request({'size': [64, 64]}, {'member_ids': ['a'], 'request_revision': 'x', 'text': 'A'},
                          [0, 0, 64, 64], crop, mask, mask, protected, attempt=0, model={'backend': 'sdxl-inpaint'})
        ghost = crop.copy()
        ghost[mask] = 168
        initial = {'pixels': ghost, 'request_revision': req['revision']}

        def review(budget):
            return rec.review_outline(router, req, initial, crop, mask, mask, protected, budget, float('inf'))

        def clean(retry, *args, **kwargs):
            return {'pixels': np.full_like(crop, 180), 'request_revision': retry['revision']}

        with patch.object(rec.LocalReconstruction, 'reconstruct', side_effect=clean) as worker:
            _, _, kept = review(rec.Budget(maximum_calls=1, calls=1))
            self.assertEqual(kept['outline_review']['status'], 'budget_exhausted')
            worker.assert_not_called()
            budget = rec.Budget(maximum_calls=2, calls=1)
            retry, context, answer = review(budget)
            self.assertEqual(budget.calls, 2)
            self.assertEqual(retry['model']['backend'], 'lama-onnx-512')
            self.assertTrue(np.array_equal(context, mask))
            self.assertTrue((rec.compose(retry,crop,context,mask,protected,answer) == 180).all())
            worker.side_effect = lambda retry, *args, **kwargs: {**initial, 'request_revision': retry['revision']}
            original, _, answer = review(rec.Budget())
            self.assertEqual(original, req)
            self.assertEqual(answer['outline_review']['selected'], 'sdxl-inpaint')
            worker.side_effect = TimeoutError('worker timeout')
            original, _, answer = review(rec.Budget())
            self.assertEqual(answer['outline_review']['status'], 'retry_failed')
            self.assertIs(original, req)
            worker.side_effect = lambda *args, **kwargs: initial
            with self.assertRaises(rec.BindingError):
                review(rec.Budget())

    def test_foreign_results_mask_drift_and_outside_pixels(self):
        crop=np.full((32,40,3),123,np.uint8)
        mask=np.zeros((32,40),bool);mask[10:20,10:20]=True
        protected=np.zeros_like(mask);protected[:4]=True
        req=rec.request({'size':[40,32],'sha256':'f'*64},
            {'member_ids':['a'],'request_revision':'a'*64,'text':'Exact!'},[0,0,40,32],crop,mask,mask,protected,attempt=0)
        result={'request_revision':req['revision'],'pixels':np.zeros_like(crop)}
        output=rec.compose(req,crop,mask,mask,protected,result)
        self.assertTrue(np.array_equal(output[~mask],crop[~mask]))
        for bad in ({}, {**result,'request_revision':'foreign'}, {**result,'pixels':np.zeros((1,1,3),np.uint8)}):
            with self.assertRaises(rec.BindingError):rec.compose(req,crop,mask,mask,protected,bad)
        changed=mask.copy();changed[0,0]=True
        with self.assertRaises(rec.BindingError):rec.compose(req,crop,changed,mask,protected,result)

    def test_outlined_colored_art_contact_and_residual_expansion(self):
        crop=np.full((100,100,3),170,np.uint8);crop[20:80,49:52]=0
        crop[30:60,35:65]=(240,20,20);crop[29,34:66]=255
        own=np.zeros((100,100),bool);own[31:59,37:63]=True
        protected=np.zeros_like(own)
        evidence,ladder,metadata=aggressive.masks(crop,own,protected)
        self.assertFalse(metadata['foreground_is_text_certified'])
        first=ladder[0][1];second=ladder[1][1]
        self.assertGreater(int(second.sum()),int(first.sum()))
        candidate=crop.copy();candidate[first]=170
        self.assertGreater(aggressive.assess(crop,candidate,first,evidence,protected)['outer_residual_pixels'],0)
        self.assertTrue((first<=second).all())


class Integrated(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.AutomaticWorkflowTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture.f
        self.fixture.make_other_strict()
        image=Image.open(self.f.image).convert('RGB')
        draw=ImageDraw.Draw(image)
        draw.line((6,31,70,31),fill=(180,20,70),width=4)
        draw.rectangle((27,25,39,36),outline='white',width=2)
        image.save(self.f.image)
        self.f.preview['input'].update(sha256=n.digest(self.f.image.read_bytes()),
            pixels_sha256=n.digest(b'RGB:180:130:'+image.tobytes()))
        from dataclasses import asdict
        from tralens.ocr.automatic_units import propose_page_from_boxes
        from tralens.ocr.types import Text_Box
        from tralens.translation.contracts import TranslationRequest, TranslationResult, UnitOutcome
        boxes=[Text_Box(d['source_id'],.99,d['polygon'],d['text']) for d in self.f.preview['ocr']['raw_detections'][0]]
        batch=asdict(propose_page_from_boxes(boxes,str(self.f.image)))
        self.f.preview['ocr']={'batches':[batch],'raw_detections':[[{**{k:v for k,v in d.items() if k not in ('page','metadata_json')},'ocr_metadata':{}} for d in batch['detections']]]}
        self.f.preview['units']=[]
        for unit in n.automatic_units(batch):
            req=TranslationRequest((unit,),(unit.unit_id,))
            outcome=UnitOutcome(unit.unit_id,'preview','Hi!\nAV?',(),'unverified',unit.selected_revision)
            result=TranslationResult(req.revision,'preview',((unit.unit_id,outcome.translation),),outcomes=(outcome,))
            self.f.preview['units'].append(dict(unit=asdict(unit),request=asdict(req),unit_revision=unit.revision,
                request_revision=req.revision,status='preview',binding_current=True,input_kind='strict_utterance',result=asdict(result)))
        self.f.write_inputs()

    def run_case(self,name,**kwargs):
        return automatic.run_automatic(**self.fixture.inputs(),output=self.f.root/name,
            min_size=14,max_size=28,removal_policy='aggressive',**kwargs)

    def test_always_show_keeps_aggressive_removal_and_native_output(self):
        with patch('socket.socket',side_effect=AssertionError('portable: no network')):
            native=automatic.run_automatic(**self.fixture.inputs(),output=self.f.root/'native',removal_policy='preserve',min_size=14,max_size=28)
            report=self.run_case('aggressive',visibility_policy='always-show')
        self.assertGreater(report['reconstruction_summary']['attempted'],0)
        self.assertGreater(report['reconstruction_summary']['selected_classical']+report['reconstruction_summary']['selected_fill'],0)
        self.assertGreater(report['display_summary']['aggressive_replacements'],0)
        for layer in ('clean','text'):
            before=np.asarray(Image.open(self.f.root/'native'/f'{layer}.png'))
            after=np.asarray(Image.open(self.f.root/'aggressive'/f'{layer}.png'))
            mask=np.asarray(Image.open(self.f.root/'native/erase-mask.png'))>0
            mask|=np.asarray(Image.open(self.f.root/'native/text.png').getchannel('A'))>0
            self.assertTrue(np.array_equal(before[mask],after[mask]))
        for row in report['units']:
            if row.get('reconstruction',{}).get('status')=='composed_unverified':
                self.assertEqual(row['layout']['original_text'],row['binding']['text'])
                self.assertGreaterEqual(row['layout']['size'],14)
        self.assertEqual(report['pixel_audit']['outside_cleanup_changed_pixels'],0)
        self.assertEqual(len(report['transcript']),2)

    def test_auto_without_configured_models_matches_classical(self):
        classical = self.run_case('manual-classical')
        auto = self.run_case('auto', reconstruction_policy='auto')
        self.assertEqual(auto['coverage'], classical['coverage'])
        for layer in ('clean.png', 'composite.png'):
            self.assertEqual((self.f.root/'manual-classical'/layer).read_bytes(),
                             (self.f.root/'auto'/layer).read_bytes())

    def test_timeout_and_oom_fall_back_but_foreign_output_is_hard(self):
        # Force a nonflat cheap candidate so this contract test exercises the
        # worker even when the synthetic page qualifies for a dominant fill.
        real=aggressive.cheap
        force_classical=lambda *a: (real(*a)[0],'classical_telea_unverified')
        forcing=patch.object(aggressive,'cheap',side_effect=force_classical)
        forcing.start();self.addCleanup(forcing.stop)
        for index,error in enumerate((subprocess.TimeoutExpired('isolated-worker',.1),MemoryError('OOM'))):
            with patch.object(rec,'configuration',return_value={'sha256':rec.MODEL_SHA256}), \
                 patch.object(rec.LocalLama,'reconstruct',side_effect=error):
                report=self.run_case('failure'+str(index),reconstruction_policy='lama',reconstruction_config='explicit.json')
            self.assertEqual(report['reconstruction_summary']['learned_failures'],1)
            self.assertGreater(report['coverage']['rendered'],0)
            self.assertEqual(report['reconstruction_summary']['selected_learned'],0)
        with patch.object(rec,'configuration',return_value={'sha256':rec.MODEL_SHA256}), \
             patch.object(rec.LocalLama,'reconstruct',return_value={'request_revision':'foreign','pixels':None}):
            with self.assertRaises(rec.BindingError):
                self.run_case('foreign',reconstruction_policy='lama',reconstruction_config='explicit.json')
        self.assertFalse((self.f.root/'foreign').exists())

    def test_failed_publication_never_exposes_a_partial_completion_manifest(self):
        from tralens import storage
        import errno
        real=storage.atomic_bytes
        def write(path,content,**kwargs):
            if path.name=='manifest.json':raise OSError(errno.ENOSPC,'injected full disk')
            return real(path,content,**kwargs)
        with patch.object(storage,'atomic_bytes',side_effect=write):
            with self.assertRaises(OSError):self.run_case('partial-publication')
        folder=self.f.root/'partial-publication'
        self.assertFalse((folder/'manifest.json').exists())
        for name in ('source.png','clean.png','text.png','composite.png'):
            with Image.open(folder/name) as image:image.verify()
        self.assertEqual((folder/'original-file.bin').read_bytes(),self.f.image.read_bytes())


if __name__=='__main__':unittest.main()
