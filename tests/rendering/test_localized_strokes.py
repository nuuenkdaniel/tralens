"""Original synthetic localization fixtures; no real model or reference annotations."""
import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import automatic as a, localized_strokes as l, natural as n, best_effort
from tralens.ocr import source_localization as s


def prediction(image, raw, ids):
    bounds=s.crop_bounds(raw,ids,image.size)
    selected=[d for d in raw if d['source_id'] in ids]
    polygons=[[[x-bounds[0],y-bounds[1]] for x,y in d['polygon']] for d in selected]
    return {'bounds':bounds,'crop_pixels_sha256':hashlib.sha256(image.crop(bounds).tobytes()).hexdigest(),
        'prediction':{'dt_polys':copy.deepcopy(polygons),'rec_texts':[d['text'] for d in selected],
            'rec_scores':[.98 for d in selected],'rec_polys':polygons,
            'text_word_region':[[poly] for poly in polygons]}}


def fixture():
    image=Image.new('RGB',(500,400),'white');draw=ImageDraw.Draw(image)
    draw.ellipse((100,20,240,220),outline='black',width=2)
    for y in (80,120,160):draw.rectangle((160,y,169,y+13),fill='black')
    draw.rectangle((240,280,260,300),fill='black')
    raw=[{'source_id':'dialogue','text':'原文','confidence':.99,'polygon':[[150,60],[178,60],[178,185],[150,185]]},
         {'source_id':'effect','text':'不確か','confidence':.2,'polygon':[[168,169],[300,169],[300,350],[168,350]]}]
    return image,raw,prediction(image,raw,['dialogue'])


class LocalizedStrokeTests(unittest.TestCase):
    def test_disputed_fringe_uses_whole_strokes_and_preserves_foreign_ink(self):
        image,raw,record=fixture();before=copy.deepcopy(raw)
        with self.assertRaises(a.AutomaticSkip) as old:a.derive_plan(image,raw,['dialogue'])
        self.assertEqual(old.exception.code,'nonmember_ink')
        plan=l.derive(image,raw,['dialogue'],record)
        self.assertEqual(plan.evidence['coarse_neighbor_rectangles_resolved'],['effect'])
        self.assertEqual(raw,before)
        pixels=np.asarray(image).copy();a.apply_cleanup(pixels,plan)
        self.assertTrue((pixels[80:174,160:170]==255).all())
        self.assertTrue(np.array_equal(pixels[280:301,240:261],np.asarray(image)[280:301,240:261]))
        self.assertFalse((plan.erase & plan.obstacles).any())
        self.assertTrue(plan.obstacles[290,250])

    def test_open_boundary_and_different_enclosures_reject(self):
        image,raw,_=fixture();ImageDraw.Draw(image).rectangle((100,20,240,70),fill='white')
        with self.assertRaises(a.AutomaticSkip):l.derive(image,raw,['dialogue'],prediction(image,raw,['dialogue']))
        image,raw,_=fixture();ImageDraw.Draw(image).ellipse((320,20,420,220),outline='black',width=2)
        raw.append({'source_id':'second','text':'別','confidence':.99,'polygon':[[350,70],[380,70],[380,150],[350,150]]})
        with self.assertRaises(a.AutomaticSkip):l.derive(image,raw,['dialogue','second'],prediction(image,raw,['dialogue','second']))

    def test_coincident_detection_and_enclosed_annotation_reject(self):
        for coincident in (True,False):
            image,raw,record=fixture()
            poly=raw[0]['polygon'] if coincident else [[190,80],[210,80],[210,130],[190,130]]
            raw.append({'source_id':'annotation','text':'注','confidence':.9,'polygon':poly})
            with self.assertRaises(a.AutomaticSkip) as error:l.derive(image,raw,['dialogue'],record)
            self.assertIn(error.exception.code,('localized_ownership_tie','localized_enclosed_neighbor'))

    def test_seed_on_border_connected_ink_rejects_instead_of_clipping(self):
        image,raw,_=fixture();ImageDraw.Draw(image).line((165,80,165,22),fill='black',width=2)
        with self.assertRaises(a.AutomaticSkip) as error:
            l.derive(image,raw,['dialogue'],prediction(image,raw,['dialogue']))
        self.assertIn(error.exception.code,('localized_seed_spill','localized_enclosure_ambiguous','localized_uncovered_seed'))

    def test_flat_gray_art_inside_a_recognized_line_stays_blocked(self):
        for attached in (False,True):
            image,raw,_=fixture()
            ImageDraw.Draw(image).rectangle((157,100,167,110),fill=(110,110,110))
            if attached:ImageDraw.Draw(image).line((162,109,162,120),fill=(110,110,110),width=3)
            with self.assertRaises(a.AutomaticSkip) as error:
                l.derive(image,raw,['dialogue'],prediction(image,raw,['dialogue']))
            self.assertEqual(error.exception.code,'localized_gray_interior')

    def test_unexpected_predicted_text_and_missing_or_reused_lines_reject(self):
        image,raw,record=fixture()
        broken=copy.deepcopy(record);broken['prediction']['rec_texts'][0]='別の原文'
        with self.assertRaises(a.AutomaticSkip):l.derive(image,raw,['dialogue'],broken)
        extra=copy.deepcopy(record)
        for name in ('rec_texts','rec_scores','rec_polys','text_word_region'):
            extra['prediction'][name]+=copy.deepcopy(extra['prediction'][name])
        # Identical proposals for the same source are not independent support.
        with self.assertRaises(a.AutomaticSkip):l.derive(image,raw,['dialogue'],extra)

    def test_stale_crop_pixels_geometry_and_word_seeds_reject(self):
        image,raw,record=fixture()
        changed=image.copy();changed.putpixel((165,120),(20,20,20))
        with self.assertRaises(ValueError):l.derive(changed,raw,['dialogue'],record)
        wrong=copy.deepcopy(record);wrong['bounds'][0]+=1
        with self.assertRaises(ValueError):l.derive(image,raw,['dialogue'],wrong)
        wrong=copy.deepcopy(record);wrong['prediction']['text_word_region'][0]=[[[0,0],[10,0],[10,10],[0,10]]]
        with self.assertRaises(a.AutomaticSkip):l.derive(image,raw,['dialogue'],wrong)

    def test_accepted_output_reservation_is_never_cut(self):
        image,raw,record=fixture();reserved=np.zeros((image.height,image.width),bool);reserved[80,160]=True
        with self.assertRaises(a.AutomaticSkip) as error:l.derive(image,raw,['dialogue'],record,reserved=reserved)
        self.assertEqual(error.exception.code,'localized_cleanup_collision')

    def test_known_text_alias_and_gray_patch_are_distinct_controlled_inputs(self):
        pixels=np.full((50,60,3),255,np.uint8)
        pixels[20:30,20:30]=0
        # A genuinely graded, tiny connected tip with increasing contrast to black.
        for x,color in ((30,50),(31,90),(32,150),(33,210)):
            pixels[24,x]=color
        residual=255-pixels.min(axis=2);foreground=residual>2
        proof=l._single_ink(pixels,foreground,residual)
        self.assertTrue(proof['terminal_checks'])
        pixels[20:27,30:38]=120
        residual=255-pixels.min(axis=2)
        with self.assertRaises(a.AutomaticSkip):l._single_ink(pixels,residual>2,residual)

    def test_policy_cannot_enable_without_existing_confidence_guards(self):
        for policy,render,settings in (('unknown','best-effort',{}),('localized','conservative',{}),('localized','best-effort',None)):
            with self.assertRaises(ValueError):a.cleanup_settings(policy,render,settings)


class LocalizationPipelineTests(unittest.TestCase):
    def setUp(self):
        from tests.test_pipeline import PipelineTests
        from tralens.ocr.types import Text_Box
        from tralens.ocr.automatic_units import propose_page_from_boxes
        self.pipeline=PipelineTests();self.pipeline.setUp();self.addCleanup(self.pipeline.doCleanups)
        self.f=self.pipeline.fixture
        image=Image.new('RGB',(500,400),'white');draw=ImageDraw.Draw(image)
        draw.ellipse((35,20,240,250),outline='black',width=2)
        draw.ellipse((300,20,470,250),outline='black',width=2)
        for x in (130,375):
            for y in (90,120,150):draw.rectangle((x,y,x+(9 if x==130 else 7),y+11),fill='black')
        image.save(self.f.image)
        self.f.boxes=[Text_Box('localized',.98,[[125,82],[145,82],[145,169],[125,169]],'普通の台詞'),
                      Text_Box('strict',.98,[[370,82],[390,82],[390,169],[370,169]],'別')]
        batch=propose_page_from_boxes(self.f.boxes,self.f.image,complementary_ownership=True,regroup_companions=True)
        proposals=tuple(replace(p,structural_reasons=p.structural_reasons+('unknown_region_ownership',))
            if p.member_ids==('localized',) else p for p in batch.proposals)
        self.f.batch=replace(batch,proposals=proposals)
        self.f.engine.raw_detections=[self.f.boxes];self.f.engine.automatic_unit_proposals=[self.f.batch]
        self.f.engine.image_path=str(self.f.image)
        self.f.engine.paddle_provenance.update(local_weights=True)
        self.raw=[{'source_id':b.paddle_box_id,'text':b.paddle_text,'confidence':.98,'polygon':b.paddle_poly.tolist()} for b in self.f.boxes]
        self.predictions={}
        for b in self.f.boxes:
            record=prediction(image,self.raw,[b.paddle_box_id]);bounds=record['bounds']
            self.predictions[image.crop(bounds).tobytes()]=record['prediction']
        def predict(pixels,*,return_word_box):
            self.assertTrue(return_word_box)
            return [self.predictions[pixels[...,::-1].tobytes()]]
        self.f.engine.ocr.predict=Mock(side_effect=predict)
        original=a.derive_plan
        def native(image,raw,ids,**kwargs):
            if ids==['localized']:raise a.AutomaticSkip('fixture_native_failure','exercise opt-in localized route')
            return original(image,raw,ids,**kwargs)
        self.native=native

    def run_case(self,name,enabled):
        with patch.object(a,'derive_plan',side_effect=self.native), \
             patch.object(best_effort,'native_plan',side_effect=a.AutomaticSkip('fixture_native_failure','synthetic coverage fixture')):
            return self.pipeline.run_pipeline(name,render_policy='best-effort',replacement_policy='confidence-gated',source_localization=enabled)

    def test_existing_ocr_produces_evidence_then_render_retries_after_strict_edits(self):
        before=self.run_case('before',False)
        self.assertEqual(before['coverage']['rendered'],1,before)
        self.f.engine.ocr.close.reset_mock()
        after=self.run_case('after',True)
        self.assertIsNone(after['error'],after)
        self.assertEqual(after['coverage']['rendered'],2,after)
        import json
        old=json.loads((self.pipeline.root/'before/rendering/manifest.json').read_bytes())
        new=json.loads((self.pipeline.root/'after/rendering/manifest.json').read_bytes())
        for i,row in enumerate(old['units']):
            if row['status']=='automatic_composed':
                for suffix in ('foreground','applied-erase','text'):
                    name=f'unit-{i:03d}-{suffix}.png'
                    self.assertEqual((self.pipeline.root/'before/rendering'/name).read_bytes(),
                                     (self.pipeline.root/'after/rendering'/name).read_bytes())
        row=next(row for row in new['units'] if row['binding']['member_ids']==['localized'])
        self.assertEqual(row['localized_cleanup_attempt']['status'],'planned')
        self.assertGreaterEqual(row['layout']['size'],16)
        self.assertFalse(new['localized_cleanup_summary']['semantic_ownership_verified'])
        self.assertEqual(self.f.engine.ocr.predict.call_count,2)

    def test_prediction_failure_retains_translation_and_strict_output(self):
        self.f.engine.ocr.predict.side_effect=RuntimeError('synthetic crop inference failure')
        result=self.run_case('failure',True)
        self.assertIsNone(result['error'],result)
        self.assertEqual(result['coverage']['total_translations'],2)
        self.assertEqual(result['coverage']['rendered'],1)

    def test_default_never_calls_crop_prediction(self):
        self.f.engine.ocr.predict.side_effect=AssertionError('not enabled')
        result=self.run_case('default',False)
        self.assertIsNone(result['error'],result)
        self.f.engine.ocr.predict.assert_not_called()

    def test_saved_evidence_source_prediction_and_unit_drift_fail_before_rendering(self):
        import json
        result=self.run_case('valid',True)
        self.assertIsNone(result['error'],result)
        data=json.loads((self.pipeline.root/'valid/preview/preview.json').read_bytes())
        for index,change in enumerate(('pixels','geometry','unit','prediction','call_budget')):
            forged=copy.deepcopy(data);stage=forged['ocr']['source_localization']
            if change=='pixels':stage['source_pixels_sha256']='0'*64
            elif change=='geometry':stage['geometry_revision']='0'*64
            elif change=='unit':stage['records'][0]['unit_revision']='0'*64
            elif change=='prediction':stage['records'][0]['prediction']['rec_texts'][0]='changed'
            else:stage['calls']=13
            path=self.pipeline.root/f'forged-{index}.json';path.write_text(json.dumps(forged))
            with self.assertRaises(ValueError):
                a.run_automatic(preview=path,image=self.f.image,fonts=self.pipeline.fonts,
                    output=self.pipeline.root/f'forged-render-{index}',replacement_policy='confidence-gated',cleanup_policy='localized')

    def test_crop_call_cap_is_independent_of_translation_and_retains_skips(self):
        s.refine(self.f.engine,output=self.pipeline.root,max_calls=1)
        stage=self.f.engine.source_localization
        self.assertEqual(stage['calls'],1)
        self.assertEqual(self.f.engine.ocr.predict.call_count,1)
        self.assertEqual(sum(r['status']=='predicted' for r in stage['records']),1)
        self.assertTrue(any(r['reason']=='localization_call_budget' for r in stage['records']))

    def test_disabled_or_mismatched_source_rejects_before_crop_calls(self):
        with self.assertRaises(ValueError):s.refine(self.f.engine,output=self.pipeline.root,max_calls=13)
        with self.assertRaises(ValueError):s.refine(self.f.engine,output=self.pipeline.root,max_calls=True)
        self.f.image.write_bytes(b'changed source, not an image')
        with self.assertRaises(ValueError):s.refine(self.f.engine,output=self.pipeline.root)
        self.f.engine.ocr.predict.assert_not_called()


if __name__=='__main__':unittest.main()
