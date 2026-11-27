"""Clipped display-line recovery from separate, bound crop predictions."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image,ImageDraw

from tralens.ocr import source_localization as crops
from tralens.rendering import (aggressive,automatic,best_effort,localization_requests,
    natural as n,replacement,reconstruction as rec,source_extents)
from .font_fixture import synthetic_font


class SourceExtents(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(620,500),(220,220,220))
        draw=ImageDraw.Draw(self.image)
        draw.rectangle((180,200,250,280),fill='black')
        draw.rectangle((380,200,410,280),fill='black')
        self.raw=[{'source_id':'line','text':'AB-','confidence':.98,
            'polygon':[[160,180],[350,180],[350,300],[160,300]]}]
        self.unit=SimpleNamespace(unit_id='one',revision='a'*64,member_ids=['line'],order=['line'],
            selected=SimpleNamespace(text='AB-'))
        self.full=[[160,180],[430,180],[430,300],[160,300]]
        self.record=self.prediction()

    def prediction(self,*,texts=None,polygons=None):
        bounds=crops.extent_crop_bounds(self.raw,['line'],self.image.size);x,y,_,_=bounds
        polygons=polygons or [self.full]
        polys=[[[a-x,b-y] for a,b in poly] for poly in polygons]
        prediction={'dt_polys':deepcopy(polys),'rec_texts':texts or ['AB-CD'],
            'rec_scores':[.98]*len(polys),'rec_polys':polys,'text_word_region':[[poly] for poly in polys]}
        return {'unit_id':'one','unit_revision':'a'*64,'member_ids':['line'],'crop_kind':'extent',
            'bounds':bounds,'crop_pixels_sha256':n.digest(self.image.crop(bounds).tobytes()),
            'prediction':prediction,'prediction_revision':n.revision(prediction)}

    def derive(self,record=None):
        return source_extents.derive(self.image,self.raw,self.unit,record or self.record)

    def test_complete_line_is_separate_from_raw_geometry_and_reading(self):
        before=deepcopy(self.raw);record=deepcopy(self.record)
        extent=self.derive()
        self.assertEqual(extent['polygon'],self.full)
        self.assertEqual((extent['original_source_text'],extent['observed_text']),('AB-','AB-CD'))
        self.assertFalse(extent['reading_promoted']);self.assertFalse(extent['translation_reselected'])
        self.assertEqual(self.raw,before);self.assertEqual(self.record,record)

    def test_ambiguous_unrelated_boundary_and_perpendicular_predictions_reject(self):
        for record in (self.prediction(texts=['XY']),self.prediction(texts=['AB-CD','AB-EF'],polygons=[self.full,self.full]),
                self.prediction(polygons=[[[160,48],[430,48],[430,300],[160,300]]]),
                self.prediction(polygons=[[[160,180],[350,180],[350,420],[160,420]]])):
            with self.subTest(record=record),self.assertRaises(ValueError):self.derive(record)

    def test_neighbor_line_cannot_be_absorbed_into_extension(self):
        self.raw.append({'source_id':'other','text':'Other','confidence':.98,
            'polygon':[[380,205],[420,205],[420,275],[380,275]]})
        with self.assertRaisesRegex(ValueError,'unique'):self.derive()

    def test_stale_crop_source_and_prediction_bindings_reject(self):
        for key,value in (('unit_revision','b'*64),('unit_id','foreign'),('member_ids',['foreign']),
                ('crop_pixels_sha256','0'*64),('prediction_revision','0'*64),('bounds',[0,0,500,400])):
            wrong=deepcopy(self.record);wrong[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.derive(wrong)

    def test_planner_retains_legacy_behavior_and_requests_large_open_line(self):
        preview={'ocr':{'batches':[{'proposals':[{'proposal_id':'one','direction':'horizontal',
            'annotation_ids':[],'main_ids':['line'],'member_ids':['line']}]}]}}
        report={'source':{},'fonts_sha256':'c'*64,'font_id':'font','style':{},'render_policy':'best-effort',
            'background_method':'auto','evaluation_cap':None,'replacement_policy':{},
            'units':[{'unit_id':'one','status':'skipped','attempted':True,'reason_code':'replacement_cleanup_confidence'}]}
        plan,_=localization_requests.plan(preview,self.image,self.raw,[self.unit],report)
        self.assertEqual(plan['requests'],['one']);self.assertEqual(plan['records'][0]['crop_kind'],'extent')
        self.assertEqual(plan['records'][0]['bounds'],self.record['bounds'])
        old,_=localization_requests.plan(preview,self.image,self.raw,[self.unit],report,policy=crops.LEGACY_REQUEST_POLICY)
        self.assertEqual(old['requests'],[])
        report['units'][0]['status']='automatic_composed'
        empty,_=localization_requests.plan(preview,self.image,self.raw,[self.unit],report)
        self.assertEqual(empty['requests'],[])
        report['units'][0]['status']='skipped'
        oversized=deepcopy(self.raw);oversized[0]['polygon']=[[100,100],[1150,100],[1150,600],[100,600]]
        with patch.object(localization_requests.strokes,'prerequisites',side_effect=AssertionError('legacy crop budget first')):
            old,_=localization_requests.plan(preview,Image.new('RGB',(1200,700)),oversized,[self.unit],report,
                policy=crops.LEGACY_REQUEST_POLICY)
        self.assertEqual(old['records'][0]['reason'],'localization_crop_budget')
        self.assertEqual(old['prerequisite_work'],0)

    def run_cleanup(self,kind,*,no_placement=False):
        extent=self.derive();shape=(self.image.height,self.image.width)
        old=np.zeros(shape,bool);old[240:245,390:395]=True
        before=np.asarray(self.image).copy();before[old]=[20,30,40]
        text=Image.new('RGBA',self.image.size);text.putpixel((402,240),(8,19,30,255))
        alpha=np.asarray(text.getchannel('A'))!=0
        rows=[{'unit_id':'one','status':'skipped','binding':{'member_ids':['line'],
            'request_revision':'c'*64,'text':'Exact words.'}}]
        font=synthetic_font();calls=[]
        def reconstruct(request,crop,inference,commit,protected,**kwargs):
            rec.validate(request,crop,inference,commit,protected);calls.append(request)
            x,y,r,b=request['bounds']
            self.assertTrue(np.array_equal(crop,np.asarray(self.image)[y:b,x:r]))
            self.assertFalse((commit&protected).any())
            return {'request_revision':request['revision'],'pixels':np.full_like(crop,150)}
        backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=reconstruct))
        shared=(self.image,self.raw,rows,Image.fromarray(before),text,old,alpha,font,n.font_coverage(font),
            {'min_size':16,'max_size':26,'outline':0})
        source={'size':list(self.image.size),'sha256':'f'*64}
        with patch.object(aggressive,'cheap',side_effect=lambda crop,*_:(crop.copy(),'classical_telea_unverified')):
            if kind=='deferred':
                pending=aggressive.recover(self.image,source,*shared[1:],backend=backend,
                    extents={'one':extent},budget=rec.Budget(maximum_calls=0))
                self.assertEqual(pending[0][0]['reconstruction']['reason'],'deferred_to_final_reconstruction_budget')
                self.assertTrue(np.array_equal(np.asarray(pending[1]),before))
                backend.reconstruct.assert_not_called()
                shared=(self.image,self.raw,pending[0],pending[1],pending[2],pending[4],pending[5],
                    font,n.font_coverage(font),shared[-1])
            if kind=='recovery':
                result=aggressive.recover(self.image,source,*shared[1:],backend=backend,extents={'one':extent})
            elif no_placement:
                with patch.object(replacement,'patch_layout',side_effect=ValueError('no room')):
                    result=replacement.finish(*shared,source=source,backend=backend,extents={'one':extent})
            else:result=replacement.finish(*shared,source=source,backend=backend,extents={'one':extent})
        rows,clean,after_text,_,erase,_,_,_=result
        tail=np.zeros(shape,bool);tail[210:230,380:410]=True
        if no_placement:
            self.assertTrue(np.array_equal(np.asarray(clean),before))
        else:
            self.assertTrue((np.asarray(clean)[tail]==150).all())
        self.assertTrue(np.array_equal(np.asarray(clean)[old|alpha],before[old|alpha]))
        self.assertTrue(np.array_equal(np.asarray(after_text)[alpha],np.asarray(text)[alpha]))
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase],before[~erase]))
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['binding']['text'],'Exact words.')
        self.assertLessEqual(max(calls[0]['bounds'][2]-calls[0]['bounds'][0],calls[0]['bounds'][3]-calls[0]['bounds'][1]),512)
        return rows

    def test_aggressive_extent_removes_tail_without_overwriting_edits(self):
        self.assertEqual(self.run_cleanup('recovery')[0]['status'],'automatic_composed')

    def test_failed_placement_preserves_extended_strokes(self):
        self.assertEqual(self.run_cleanup('final',no_placement=True)[0]['status'],'skipped')

    def test_exhausted_aggressive_budget_leaves_fresh_final_reconstruction_available(self):
        self.assertEqual(self.run_cleanup('deferred',no_placement=True)[0]['status'],'skipped')


class ExtentCoordinator(unittest.TestCase):
    def test_existing_crop_pass_runs_once_and_keeps_originals_and_translation(self):
        from tests.test_pipeline import PipelineTests
        from tralens.ocr.types import Text_Box
        from tralens.ocr.automatic_units import propose_page_from_boxes
        p=PipelineTests();p.setUp();self.addCleanup(p.doCleanups);f=p.fixture
        case=SourceExtents();case.setUp();case.image.save(f.image)
        f.boxes=[Text_Box('line',.98,case.raw[0]['polygon'],'AB-')]
        batch=propose_page_from_boxes(f.boxes,f.image,complementary_ownership=True,regroup_companions=True)
        f.batch=replace(batch,proposals=tuple(replace(proposal,structural_reasons=proposal.structural_reasons+
            ('unknown_region_ownership',)) for proposal in batch.proposals))
        f.engine.raw_detections=[f.boxes];f.engine.automatic_unit_proposals=[f.batch]
        f.engine.image_path=str(f.image);f.engine.paddle_provenance.update(local_weights=True)
        def predict(pixels,*,return_word_box):
            self.assertTrue((p.root/'extent/rendering-native/manifest.json').is_file())
            self.assertEqual(p.provider.call_count,1)
            self.assertEqual(pixels[...,::-1].tobytes(),case.image.crop(case.record['bounds']).tobytes())
            return [case.record['prediction']]
        f.engine.ocr.predict=Mock(side_effect=predict)
        with patch.object(automatic,'derive_plan',side_effect=automatic.AutomaticSkip('fixture','open display lettering')), \
             patch.object(best_effort,'native_plan',side_effect=automatic.AutomaticSkip('fixture','open display lettering')):
            result=p.run_pipeline('extent',render_policy='best-effort',replacement_policy='confidence-gated',
                source_localization=True,removal_policy='aggressive')
        self.assertIsNone(result['error'],result)
        folder=p.root/'extent';before=json.loads((folder/'preview/preview.json').read_bytes())
        after=json.loads((folder/'localization/preview.json').read_bytes())
        self.assertEqual(before['units'],after['units']);self.assertEqual(before['ocr']['raw_detections'],after['ocr']['raw_detections'])
        final=json.loads((folder/'rendering/manifest.json').read_bytes())
        self.assertEqual(final['source_extent_summary']['accepted'],1)
        self.assertEqual(final['coverage']['rendered'],1)
        f.engine.ocr.predict.assert_called_once();p.provider.assert_called_once()
        self.assertEqual(result['source_localization']['predictions_consumed'],1)


if __name__=='__main__':unittest.main()
