"""Original synthetic balloon topology/ownership and immutable grouping tests."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.balloons import LightInteriors, mask_revision
from tralens.rendering.balloon_clean import derive
from tralens.ocr import balloon_dialogue as bd
from tralens.translation import balloon_dialogue as td
from tralens.translation.contracts import Reading, Utterance, canonical, revision


def fixture():
    image = Image.new('RGB',(280,300),(160,160,160))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25,15,255,285),fill='white',outline='black',width=3)
    for top in (70,102):
        draw.line([(120,top),(106,top),(106,top+23),(120,top+23)],fill='black',width=2)
        draw.line([(106,top+11),(117,top+11)],fill='black',width=2)
    raw = [{'source_id':0,'polygon':[[101,65],[125,65],[125,131],[101,131]],'text':'あい','confidence':.99}]
    return image,raw


class InteriorTests(unittest.TestCase):
    def test_independent_plain_cell_cleanup_has_exact_footprint(self):
        image,raw = fixture();context = LightInteriors(image,raw);interior = context.locate([0])
        with patch('tralens.rendering.automatic.derive_plan',side_effect=AssertionError('no old mask')):
            plan = derive(context,interior,[0])
        self.assertFalse(plan.evidence['prior_native_mask_required'])
        self.assertEqual(len(plan.evidence['cells']),2)
        self.assertFalse((plan.erase & ~interior.core).any())
        self.assertLess(plan.erase.sum(),interior.envelope.sum()/10)
        pixels = np.asarray(image).copy();before = pixels.copy();pixels[plan.erase] = plan.background
        self.assertTrue(np.array_equal(before[~plan.erase],pixels[~plan.erase]))
        self.assertEqual(plan.evidence['cleanup_mask_revision'],mask_revision(plan.erase))

    def test_gradient_and_colored_or_solid_inbox_art_reject(self):
        image,raw = fixture()
        for y in range(40,260):
            for x in range(60,220):
                if image.getpixel((x,y)) == (255,255,255):image.putpixel((x,y),(230+x//15,)*3)
        with self.assertRaises(ValueError):LightInteriors(image,raw).locate([0])
        image,raw = fixture();ImageDraw.Draw(image).rectangle((106,70,120,93),fill='black')
        c = LightInteriors(image,raw)
        with self.assertRaisesRegex(ValueError,'density'):derive(c,c.locate([0]),[0])
        image,raw = fixture();ImageDraw.Draw(image).line([(106,70),(120,70)],fill=(255,0,0),width=2)
        c = LightInteriors(image,raw)
        with self.assertRaisesRegex(ValueError,'colored'):derive(c,c.locate([0]),[0])

    def test_illustration_inside_and_outside_detector_boxes_reject(self):
        image,raw = fixture();draw = ImageDraw.Draw(image)
        draw.rectangle((100,64,126,132),fill='white')
        draw.ellipse((106,70,120,93),outline='black',width=2)
        draw.ellipse((106,102,120,125),outline='black',width=2)
        c = LightInteriors(image,raw)
        with self.assertRaises(ValueError):derive(c,c.locate([0]),[0])
        image,raw = fixture();ImageDraw.Draw(image).ellipse((170,160,200,210),outline='black',width=3)
        c = LightInteriors(image,raw)
        with self.assertRaisesRegex(ValueError,'unexplained dark'):derive(c,c.locate([0]),[0])

    def test_open_page_region_and_clipped_sources_reject(self):
        image,raw = fixture();draw = ImageDraw.Draw(image)
        draw.rectangle((0,0,279,299),fill='white')
        with self.assertRaisesRegex(ValueError,'open/page'):LightInteriors(image,raw).locate([0])
        image,raw = fixture();raw[0]['polygon'] = [[0,0],[280,0],[280,300],[0,300]]
        c = LightInteriors(image,raw)
        with self.assertRaises(ValueError):derive(c,c.locate([0]),[0])

    def test_neighbor_roles_and_committed_output_remain_protected(self):
        image,raw = fixture()
        raw.append({'source_id':1,'polygon':[[105,69],[121,69],[121,94],[105,94]],'text':'あ'})
        c = LightInteriors(image,raw)
        with self.assertRaisesRegex(ValueError,'shared/unselected'):derive(c,c.locate([0]),[0])
        image,raw = fixture();c = LightInteriors(image,raw);reserved = np.zeros((300,280),bool);reserved[70,110] = True
        with self.assertRaisesRegex(ValueError,'protected'):derive(c,c.locate([0]),[0],reserved=reserved)

    def test_adjacent_balloons_never_share_geometry(self):
        image,raw = fixture();ImageDraw.Draw(image).line([(140,17),(140,284)],fill='black',width=3)
        raw.append({'source_id':1,'polygon':[[171,65],[195,65],[195,131],[171,131]],'text':'うえ'})
        c = LightInteriors(image,raw)
        with self.assertRaisesRegex(ValueError,'distinct interiors'):c.locate([0,1])

    def test_original_geometry_snapshot_and_stale_interior_rejection(self):
        image,raw = fixture();c = LightInteriors(image,raw);interior = c.locate([0])
        raw[0]['text'] = 'mutated caller data'
        self.assertEqual(c.raw[('int',0)]['text'],'あい')
        interior.core[100,100] = ~interior.core[100,100]
        with self.assertRaisesRegex(ValueError,'stale'):derive(c,interior,[0])

    def test_detector_may_not_clip_a_complete_original_glyph(self):
        image,raw = fixture()
        # An ink spur outside the detector's right edge must be protected even
        # though the remaining cropped glyph still resembles a regular cell.
        ImageDraw.Draw(image).line([(117,81),(132,81)],fill='black',width=2)
        c = LightInteriors(image,raw)
        with self.assertRaises(ValueError):derive(c,c.locate([0]),[0])


class GroupContractTests(unittest.TestCase):
    def setUp(self):
        page = canonical({'image_sha256':'a'*64,'pixels_sha256':'b'*64})
        self.units = []
        for i,text in enumerate(('これは','会長')):
            reading = Reading(text,'raw_main_reading',canonical([{'source_id':i,'text':text}]))
            self.units.append(Utterance(f'old-{i}',page,(i,),(i,),(reading,),reading.revision,'raw-'+str(i)))
        self.units = tuple(self.units)
        records = [{'label':f'u{i}','unit_id':u.unit_id,'unit_revision':u.revision} for i,u in enumerate(self.units)]
        evidence = {'units':records}
        self.candidate = {'candidate_id':revision(evidence),'evidence':evidence}
        self.group = {'order':['u0','u1'],'relation':'detached_address'}
        self.proposal = {'candidate_id':self.candidate['candidate_id'],'evidence':evidence,'status':'proposed',
            'proposal':{'surface':'plain_dialogue','groups':[self.group],'reason':'uncertain proposal'}}
        self.batch = {'proposals':[{'proposal_id':u.unit_id,'member_ids':list(u.member_ids),
            'main_ids':list(u.order),'annotation_parents':[]}for u in self.units]}

    def test_derived_source_request_retains_old_units_and_annotation_roles(self):
        old = deepcopy(self.units)
        parent = self.units[0]
        self.units = (replace(parent,member_ids=(0,2)),self.units[1])
        self.candidate['evidence']['units'][0]['unit_revision'] = self.units[0].revision
        self.batch['proposals'][0].update(member_ids=[0,2],annotation_parents=[[2,0]])
        request,evidence = td.derive(self.candidate,self.proposal,self.group,self.units,self.batch)
        derived = request.units[0]
        self.assertEqual(derived.member_ids,(0,2,1));self.assertEqual(derived.order,(0,1))
        self.assertEqual(derived.selected.text,'これは\n会長')
        self.assertNotIn(derived.unit_id,[u.unit_id for u in old])
        self.assertEqual(evidence['member_roles'][1],{'source_id':2,'role':'annotation','annotation_parent':0,'parent_unit_id':'old-0'})
        self.assertEqual(evidence['speaker'],'unknown')
        self.assertEqual(self.units[0].readings,old[0].readings)

    def test_stale_parent_wrong_role_and_missing_partition_reject(self):
        with self.assertRaisesRegex(ValueError,'stale'):
            td.derive(self.candidate,self.proposal,self.group,(replace(self.units[0],evidence_revision='changed'),self.units[1]),self.batch)
        self.batch['proposals'][0]['main_ids'] = [1]
        with self.assertRaisesRegex(ValueError,'roles/order'):td.derive(self.candidate,self.proposal,self.group,self.units,self.batch)
        for groups in ([{'order':['u0'],'relation':'independent'}],[{'order':['u0','u0'],'relation':'continuation'}],
                [{'order':['u0','u1'],'relation':'independent'}]):
            with self.assertRaises(ValueError):bd.parse(self.candidate,canonical({'surface':'plain_dialogue','groups':groups,'reason':''}))

    def test_joined_lobes_and_interjections_are_not_merged_without_a_proposal(self):
        response = bd.parse(self.candidate,canonical({'surface':'plain_dialogue','groups':[
            {'order':['u0'],'relation':'independent'},{'order':['u1'],'relation':'independent'}],'reason':'separate interjections'}))
        self.assertTrue(all(len(g['order']) == 1 for g in response['groups']))
        group = response['groups'][0]
        with self.assertRaises(ValueError):td.derive(self.candidate,self.proposal,group,self.units,self.batch)

    def test_unreadable_or_ambiguous_annotation_source_cannot_be_promoted(self):
        for reason in ('blank_main_reading','ambiguous_annotation_parents'):
            units = (replace(self.units[0],blocked_reasons=(reason,)),self.units[1])
            self.candidate['evidence']['units'][0]['unit_revision'] = units[0].revision
            with self.assertRaisesRegex(ValueError,'unreadable'):td.derive(self.candidate,self.proposal,self.group,units,self.batch)

    def test_translation_is_executed_as_new_source_not_old_english_concatenation(self):
        from tralens.translation.runtime import PreviewTranslator,TRANSLATEGEMMA
        request,_ = td.derive(self.candidate,self.proposal,self.group,self.units,self.batch)
        runtime = Mock();runtime.generate.return_value = 'A fresh combined sentence, Chairman.'
        translator = PreviewTranslator(runtime,model='translategemma:27b',profile=TRANSLATEGEMMA)
        record = td.execute(request,translator)
        self.assertEqual(runtime.generate.call_count,1)
        message = runtime.generate.call_args.kwargs['payload']['messages'][0]['content']
        self.assertIn('これは\n会長',message)
        self.assertEqual(record['result']['translations'][0][1],'A fresh combined sentence, Chairman.')


class AtomicPassTests(unittest.TestCase):
    def setUp(self):
        from tests.test_pipeline import PipelineTests
        from tralens.ocr.types import Text_Box
        from tralens.ocr.automatic_units import propose_page_from_boxes
        self.f = PipelineTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        image,raw = fixture();draw = ImageDraw.Draw(image)
        for top in (190,222):
            draw.line([(120,top),(106,top),(106,top+23),(120,top+23)],fill='black',width=2)
            draw.line([(106,top+11),(117,top+11)],fill='black',width=2)
        image.save(self.f.fixture.image)
        raw.append({'source_id':1,'polygon':[[101,185],[125,185],[125,251],[101,251]],'text':'うえ','confidence':.99})
        boxes = [Text_Box(d['source_id'],d['confidence'],d['polygon'],d['text']) for d in raw]
        self.f.fixture.engine.raw_detections = [boxes]
        self.f.fixture.engine.automatic_unit_proposals = [propose_page_from_boxes(boxes,self.f.fixture.image,
            complementary_ownership=True,regroup_companions=True)]
        self.f.provider.side_effect = ['Hi!','Chairman']
        self.native = self.f.run_pipeline('native',replacement_policy='confidence-gated')
        self.assertIsNone(self.native['error'],self.native)
        self.options = dict(preview=self.f.root/'native/rendering/evidence-preview.json',image=self.f.fixture.image,
            fonts=self.f.fonts,baseline=self.f.root/'native/rendering')

    def run_experiment(self,name,**kwargs):
        from main import run_balloon_experiment
        with patch('socket.socket',side_effect=AssertionError('no live inference')):
            return run_balloon_experiment(**self.options,output=self.f.root/name,**kwargs)

    def runtimes(self,text='Hello, Chairman.'):
        proposer = Mock()
        proposer.generate.return_value = canonical({'surface':'plain_dialogue','groups':[
            {'order':['u0','u1'],'relation':'detached_address'}],'reason':'uncertain detached address proposal'})
        translator = Mock();translator.generate.return_value = text
        return proposer,translator

    def test_atomic_group_replaces_parents_once_and_retains_history(self):
        proposer,translator = self.runtimes()
        state = self.run_experiment('new',run_local_model=True,proposal_runtime=proposer,translation_runtime=translator)
        self.assertEqual((state['proposal_calls'],state['translation_calls']),(1,1))
        manifest = json.loads((self.f.root/'new/rendering/manifest.json').read_bytes())
        self.assertEqual(manifest['new_clean_dialogues'],1,manifest['failures'])
        self.assertEqual(len(manifest['suppressed_historical_indices']),2)
        self.assertEqual(len(manifest['active_units']),1)
        self.assertEqual(len(manifest['historical_units']),2)
        self.assertEqual(manifest['duplicate_rendered_members'],0)
        row = manifest['active_units'][0]
        self.assertEqual(row['binding']['text'],'Hello, Chairman.')
        self.assertGreaterEqual(row['layout']['size'],16)
        self.assertEqual(manifest['outside_declared_changed_pixels'],0)
        replay = self.run_experiment('replayed',replay=self.f.root/'new/experiment.json')
        self.assertEqual(replay['proposal_calls'],0)
        for name in ('clean','text','composite','changed-footprint'):
            self.assertEqual((self.f.root/f'new/rendering/{name}.png').read_bytes(),
                (self.f.root/f'replayed/rendering/{name}.png').read_bytes())
        self.assertEqual(self.f.provider.call_count,2)

    def test_group_overflow_and_budget_do_not_remove_old_success(self):
        proposer,translator = self.runtimes('Chairman '*200)
        state = self.run_experiment('overflow',run_local_model=True,proposal_runtime=proposer,
            translation_runtime=translator,visibility_policy='replacement-only')
        manifest = json.loads((self.f.root/'overflow/rendering/manifest.json').read_bytes())
        self.assertEqual(manifest['new_clean_dialogues'],0)
        self.assertEqual(manifest['suppressed_historical_indices'],[])
        self.assertEqual((self.f.root/'native/rendering/composite.png').read_bytes(),
            (self.f.root/'overflow/rendering/composite.png').read_bytes())
        from tralens.rendering import balloon_pass
        proposer,translator = self.runtimes()
        with patch.object(balloon_pass,'MAX_ATTEMPTS',0):
            self.run_experiment('budget',run_local_model=True,proposal_runtime=proposer,
                translation_runtime=translator,visibility_policy='replacement-only')
        self.assertEqual((self.f.root/'native/rendering/composite.png').read_bytes(),
            (self.f.root/'budget/rendering/composite.png').read_bytes())

    def test_no_opt_in_no_calls_and_old_artifacts_unchanged(self):
        before = {p:p.read_bytes() for p in (self.f.root/'native').rglob('*') if p.is_file()}
        proposer,translator = self.runtimes()
        self.run_experiment('off',proposal_runtime=proposer,translation_runtime=translator,visibility_policy='replacement-only')
        proposer.generate.assert_not_called();translator.generate.assert_not_called()
        self.assertTrue(all(p.read_bytes() == data for p,data in before.items()))

    def test_independent_units_share_supported_frames_without_joining_english(self):
        proposer,translator = self.runtimes()
        proposer.generate.return_value = canonical({'surface':'plain_dialogue','groups':[
            {'order':['u0'],'relation':'independent'},{'order':['u1'],'relation':'independent'}],
            'reason':'keep independent utterances'})
        state = self.run_experiment('coordinated',run_local_model=True,proposal_runtime=proposer,translation_runtime=translator)
        translator.generate.assert_not_called()
        self.assertEqual(state['translation_calls'],0)
        m = json.loads((self.f.root/'coordinated/rendering/manifest.json').read_bytes())
        self.assertEqual(m['suppressed_historical_indices'],[])
        self.assertEqual(m['native_preserved'],self.native['coverage']['rendered'])
        self.assertEqual(m['new_clean_dialogues'],1,m['failures'])
        self.assertEqual(m['accepted'][0]['binding']['text'],'Chairman')
        parent = m['accepted'][0]['parent_indices'][0]
        self.assertEqual(m['accepted'][0]['binding'],m['historical_units'][parent]['binding'])
        self.assertFalse(m['accepted'][0]['pixel_ownership']['utterances_joined'])

    def test_visual_art_proposal_is_not_cleanup_permission(self):
        proposer,translator = self.runtimes()
        proposer.generate.return_value = canonical({'surface':'illustration','groups':[
            {'order':['u0'],'relation':'independent'},{'order':['u1'],'relation':'independent'}],'reason':'illustrated interior'})
        state = self.run_experiment('art',run_local_model=True,proposal_runtime=proposer,
            translation_runtime=translator,visibility_policy='replacement-only')
        self.assertEqual(state['new_clean_dialogues'],0)
        self.assertEqual((self.f.root/'native/rendering/composite.png').read_bytes(),
            (self.f.root/'art/rendering/composite.png').read_bytes())
        translator.generate.assert_not_called()

    def test_new_cleanup_does_not_bypass_configured_source_gate(self):
        proposer,translator = self.runtimes()
        with patch('tralens.rendering.replacement_confidence.assess_source',return_value={'allowed':False}):
            state = self.run_experiment('source-gate',run_local_model=True,proposal_runtime=proposer,
                translation_runtime=translator,visibility_policy='replacement-only')
        self.assertEqual(state['new_clean_dialogues'],0)
        self.assertEqual((self.f.root/'native/rendering/composite.png').read_bytes(),
            (self.f.root/'source-gate/rendering/composite.png').read_bytes())

    def test_stale_replay_and_corrupt_native_layers_are_not_authorization(self):
        self.run_experiment('first')
        document = json.loads((self.f.root/'first/experiment.json').read_bytes());document['candidates'][0]['units'][0]['source'] = 'stale'
        stale = self.f.root/'stale.json';stale.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,'drift'):self.run_experiment('stale-out',replay=stale)
        (self.f.root/'native/rendering/text.png').write_bytes(b'corrupt layer')
        with self.assertRaisesRegex(ValueError,'artifact hash'):self.run_experiment('corrupt')
        self.assertFalse((self.f.root/'corrupt').exists())
