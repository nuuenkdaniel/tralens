"""Independent queue/source-recovery contracts, synthetic providers only."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

from PIL import Image

import main
from tralens import language_review as lr
from tralens.ocr import source_review as sr
from tralens.translation import source_recovery as recover
from tralens.translation.contracts import canonical,revision


class LanguageReviewTests(unittest.TestCase):
    def setUp(self):
        from tests.test_pipeline import PipelineTests
        self.f=PipelineTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.run_pipeline('baseline',replacement_policy='confidence-gated')
        self.preview=self.f.root/'baseline/rendering/evidence-preview.json'
        self.doc=json.loads(self.preview.read_bytes())
        self.rgb=Image.open(self.f.fixture.image).convert('RGB')
        self.options=dict(preview=self.preview,image=self.f.fixture.image)

    def planned(self,n=1):
        plan=lr.plan(self.doc,self.rgb,max_requests=n)
        uid=plan['requests'][0]
        request,*_=sr.prepare(self.doc,plan,uid,self.rgb)
        return plan,uid,request

    def response(self,request,text=None,state='clear',clipped=False):
        uid=request['binding']['unit_id'];row=next(r for r in self.doc['units']if r['unit']['unit_id']==uid)
        text=text if text is not None else row['unit']['readings'][0]['text']
        lines={}
        for target in request['binding']['targets']:
            xs,ys=zip(*target['presented_polygon'])
            lines[target['key']]={'text':text,'state':state,'extent':[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))],
                'clipped':clipped,'reason':'synthetic original-pixel proposal, not reference truth'}
        return canonical({'lines':lines})

    def run_stage(self,name='review',**kwargs):
        with patch('socket.socket',side_effect=AssertionError('no network in portable tests')):
            return main.run_language_review(**self.options,output=self.f.root/name,**kwargs)

    def test_rendering_success_is_not_a_queue_input_or_rejection(self):
        a=lr.plan(self.doc,self.rgb,max_requests=2)
        changed=deepcopy(self.doc);changed['rendering']={'all_units_rendered':True,'text':'Co-Co Deck'}
        b=lr.plan(changed,self.rgb,max_requests=2)
        self.assertEqual(a['requests'],b['requests'])
        self.assertEqual(a['counts'],b['counts'])
        self.assertTrue(all(not r['rendering_status_consulted']for r in b['records']))
        self.assertEqual(a['counts']['considered'],len(self.doc['units']))

    def test_high_score_strict_units_are_reviewable(self):
        plan,uid,request=self.planned(2)
        self.assertEqual(len(plan['requests']),2)
        for r in plan['records']:
            self.assertEqual(r['status'],'requested')
            self.assertFalse(r['confidence_is_accuracy'])
        self.assertIn('stratified_audit_sample',[reason for r in plan['records']for reason in r['reasons']])

    def test_zero_budget_does_not_displace_or_infer(self):
        provider=Mock();factory=Mock()
        state=self.run_stage(max_reviews=0,run_local_model=True,proposal_runtime=provider,
            translation_runtime=provider,reader_factory=factory)
        provider.generate.assert_not_called();factory.assert_not_called()
        self.assertEqual(state['primary_records'],self.doc['units'])
        self.assertEqual(state['counts']['attempted'],0)
        self.assertEqual(state['plan']['counts']['budget_excluded'],len(self.doc['units']))

    def test_plan_only_never_constructs_readers_or_call_models(self):
        provider=Mock();factory=Mock()
        state=self.run_stage(proposal_runtime=provider,translation_runtime=provider,reader_factory=factory)
        self.assertEqual(state['status'],'planned_no_inference')
        provider.generate.assert_not_called();factory.assert_not_called()

    def test_native_original_and_localization_overlay_are_distinct(self):
        plan,uid,request=self.planned()
        request,native,view,overlay=sr.prepare(self.doc,plan,uid,self.rgb)
        self.assertEqual(request['binding']['native_pixels_sha256'],sr.hashlib.sha256(native.tobytes()).hexdigest())
        self.assertNotEqual(view.tobytes(),overlay.tobytes())
        self.assertFalse(request['binding']['detector_geometry_changed'])
        self.assertFalse(request['binding']['pixel_ownership_inferred'])
        prompt=request['payload']['messages'][0]['content']
        source=next(r for r in plan['records']if r['unit_id']==uid)['source_text']
        self.assertNotIn(source,prompt)
        self.assertNotIn('Hi!',prompt)

    def test_changed_queue_or_pixels_reject_before_review(self):
        plan,uid,request=self.planned();bad=deepcopy(plan);bad['records'][0]['source_text']='stale'
        with self.assertRaises(ValueError):sr.prepare(self.doc,bad,uid,self.rgb)
        changed=self.rgb.copy();changed.putpixel((0,0),(0,0,0))
        with self.assertRaises(ValueError):lr.plan(self.doc,changed)

    def test_unchanged_reread_is_retained_without_retranslation(self):
        plan,uid,request=self.planned();provider=Mock();provider.generate.return_value=self.response(request)
        translator=Mock();factory=Mock()
        state=self.run_stage(max_reviews=1,run_local_model=True,proposal_runtime=provider,
            translation_runtime=translator,reader_factory=factory)
        self.assertEqual(state['source_reviews'][0]['status'],'unchanged_visual_corroboration')
        self.assertEqual(state['counts']['unchanged_corroborations'],1)
        self.assertTrue(state['source_reviews'][0]['translation_reused'])
        translator.generate.assert_not_called();factory.assert_not_called()

    def test_uncertain_unchanged_is_not_mislabeled_corroboration(self):
        _,_,request=self.planned();provider=Mock();provider.generate.return_value=self.response(request,state='uncertain')
        state=self.run_stage(max_reviews=1,run_local_model=True,proposal_runtime=provider,translation_runtime=Mock())
        self.assertEqual(state['counts']['unchanged_corroborations'],0)
        self.assertEqual(state['source_reviews'][0]['status'],'uncertain_candidate_retained')

    def test_invalid_response_or_provider_error_keeps_good_primary(self):
        for j,response in enumerate(('{}',RuntimeError('reader unavailable'))):
            provider=Mock()
            if isinstance(response,Exception):provider.generate.side_effect=response
            else:provider.generate.return_value=response
            state=self.run_stage(f'failure-{j}',max_reviews=1,run_local_model=True,
                proposal_runtime=provider,translation_runtime=Mock())
            self.assertEqual(state['counts']['failed'],1)
            self.assertEqual(state['primary_records'],self.doc['units'])
            self.assertEqual(state['candidate_translations'],[])

    def test_changed_corroborated_source_gets_new_bound_TG_request(self):
        plan,uid,request=self.planned();provider=Mock();provider.generate.return_value=self.response(request,'さようなら')
        reader=Mock(return_value={'text':'さようなら','records':[]});close=Mock();translator=Mock();translator.generate.return_value='Goodbye.'
        original=deepcopy(self.doc)
        state=self.run_stage(max_reviews=1,run_local_model=True,proposal_runtime=provider,translation_runtime=translator,
            reader_factory=lambda:(reader,close))
        self.assertEqual(state['counts']['corroborated_candidates'],1)
        self.assertEqual(state['counts']['translation_calls'],1)
        self.assertEqual(self.doc,original)
        self.assertEqual(state['primary_records'],original['units'])
        record=state['candidate_translations'][0]
        self.assertFalse(record['selected']);self.assertTrue(record['primary_retained'])
        parent=next(r['unit'] for r in original['units'] if r['unit']['unit_id']==uid)
        self.assertEqual(list(record['request']['units'][0]['member_ids']),parent['member_ids'])
        self.assertEqual(translator.generate.call_args.kwargs['model'],'translategemma:27b')
        self.assertIn('さようなら',translator.generate.call_args.kwargs['payload']['messages'][0]['content'])
        self.assertEqual(len(reader.call_args.args),1)
        self.assertIsInstance(reader.call_args.args[0],Image.Image)
        close.assert_called_once()

    def test_reader_disagreement_and_budget_do_not_promote_fluent_candidates(self):
        _,_,request=self.planned();provider=Mock();provider.generate.return_value=self.response(request,'さようなら')
        translator=Mock();factory=Mock(return_value=(Mock(return_value={'text':'違う'}),Mock()))
        state=self.run_stage('disagree',max_reviews=1,run_local_model=True,proposal_runtime=provider,
            translation_runtime=translator,reader_factory=factory)
        self.assertEqual(state['counts']['translation_calls'],0);translator.generate.assert_not_called()
        factory=Mock()
        state=self.run_stage('no-reader-budget',max_reviews=1,max_reader_calls=0,run_local_model=True,
            proposal_runtime=provider,translation_runtime=translator,reader_factory=factory)
        factory.assert_not_called();self.assertEqual(state['source_reviews'][0]['status'],'independent_reader_budget_excluded')

    def test_failed_candidate_translation_does_not_erase_valid_primary(self):
        _,_,request=self.planned();provider=Mock();provider.generate.return_value=self.response(request,'さようなら')
        translator=Mock();translator.generate.side_effect=RuntimeError('optional translator failed')
        state=self.run_stage(max_reviews=1,run_local_model=True,proposal_runtime=provider,translation_runtime=translator,
            reader_factory=lambda:(Mock(return_value={'text':'さようなら'}),Mock()))
        self.assertEqual(state['candidate_translations'][0]['result']['status'],'failed')
        self.assertEqual(state['primary_records'],self.doc['units'])

    def test_legitimate_slashes_are_not_mechanically_rewritten(self):
        for text in ('/usr/bin/tool','AC/DC','Input / output','run /tmp/test'):
            self.assertFalse(lr.possible_menu(text,'パス / 名前'))
        self.assertTrue(lr.possible_menu('Yes! / Okay! / Alright!','はい'))

    def test_normalization_preserves_voicing_and_small_kana(self):
        self.assertNotEqual(sr.normalized('ば'),sr.normalized('は'))
        self.assertNotEqual(sr.normalized('っ'),sr.normalized('つ'))
        self.assertEqual(sr.normalized('ＡＢＣ !'),sr.normalized('ABC！'))

    def test_source_span_outside_original_image_or_target_rejects(self):
        _,_,request=self.planned();answer=json.loads(self.response(request))
        answer['lines']['R0']['extent']=[0,0,1,1]
        with self.assertRaises(ValueError):sr.parse(request,canonical(answer),self.rgb)
        answer=json.loads(self.response(request));answer['lines']['foreign']=answer['lines']['R0']
        with self.assertRaises(ValueError):sr.parse(request,canonical(answer),self.rgb)

    def test_candidate_text_page_and_reader_bindings_cannot_be_replayed_stale(self):
        plan,uid,request=self.planned()
        candidate=sr.parse(request,self.response(request,'さようなら'),self.rgb)
        ok,evidence=sr.corroborate(candidate,self.rgb,lambda crop:{'text':'さようなら'},maximum_calls=8)
        self.assertTrue(ok)
        row=next(r for r in plan['records']if r['unit_id']==uid)
        unit=lr.automatic_units(self.doc['ocr']['batches'][0],())[row['index']]
        def build(c,e):return recover.request_for(unit,row,c,e,preview=self.doc,plan=plan,rgb=self.rgb)
        build(candidate,evidence)
        for field,value in [('text','unsupported extra text'),('clear',False)]:
            bad=deepcopy(candidate);bad[field]=value
            with self.assertRaises(ValueError):build(bad,evidence)
        bad=deepcopy(candidate);bad['binding']['source']['sha256']='0'*64
        with self.assertRaises(ValueError):build(bad,evidence)
        bad=deepcopy(evidence);bad[0]['pixels_sha256']='0'*64
        with self.assertRaises(ValueError):build(candidate,bad)
        bad=deepcopy(evidence);bad[0]['reading']['text']='違う'
        with self.assertRaises(ValueError):build(candidate,bad)

    def test_neighbor_or_annotation_center_inside_extent_prevents_clear_support(self):
        _,_,request=self.planned();request=deepcopy(request)
        target=request['binding']['targets'][0]
        request['binding']['protected_neighbor_geometry']=[{'source_id':'annotation','polygon':target['original_polygon']}]
        request['revision']=revision(request['binding'])
        candidate=sr.parse(request,self.response(request,'さようなら'),self.rgb)
        self.assertFalse(candidate['clear'])
        self.assertEqual(candidate['lines'][0]['protected_extent_conflicts'],['annotation'])
        reader=Mock()
        ok,evidence=sr.corroborate(candidate,self.rgb,reader,maximum_calls=8)
        reader.assert_not_called();self.assertFalse(ok)

    def test_reader_failure_is_retained_with_actual_call_count(self):
        _,_,request=self.planned();provider=Mock();provider.generate.return_value=self.response(request,'さようなら')
        reader=Mock(side_effect=RuntimeError('crop reader unavailable'))
        state=self.run_stage(max_reviews=1,run_local_model=True,proposal_runtime=provider,translation_runtime=Mock(),
            reader_factory=lambda:(reader,Mock()))
        self.assertEqual(state['counts']['corroborating_reader_calls'],1)
        self.assertEqual(state['counts']['reader_failures'],1)
        self.assertIn('error',state['source_reviews'][0]['corroboration'][0])
        self.assertEqual(state['primary_records'],self.doc['units'])

    def test_translation_budget_retains_corroborated_source_without_call(self):
        _,_,request=self.planned();provider=Mock();provider.generate.return_value=self.response(request,'さようなら')
        translator=Mock()
        state=self.run_stage(max_reviews=1,max_translation_calls=0,run_local_model=True,
            proposal_runtime=provider,translation_runtime=translator,
            reader_factory=lambda:(Mock(return_value={'text':'さようなら'}),Mock()))
        translator.generate.assert_not_called()
        self.assertEqual(state['source_reviews'][0]['translation_status'],'candidate_translation_budget_excluded')
        self.assertEqual(state['counts']['translation_calls'],0)
