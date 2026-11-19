"""Source-bound reading selection, real workflow/renderer, synthetic readers only."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from PIL import Image

from tests.ocr import test_automatic_recognition as fixtures
from tralens.ocr import crop_readings as c
from tralens.ocr.automatic_recognition import recognize_automatic_batch
from tralens.translation import coverage
from tralens.translation.contracts import automatic_units,canonical


class CropReadings(unittest.TestCase):
    def setUp(self):
        self.image,self.original,self.batch=fixtures.AutomaticRecognitionTests().fixture(text='ネットワー7の')
        self.engine=SimpleNamespace(automatic_unit_proposals=(self.batch,),automatic_recognition_results=())
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.output=Path(tmp.name)

    def refine(self,reader):
        c.refine(self.engine,self.image,self.original,reader=reader,model_identity='synthetic',output=self.output)
        return automatic_units(self.batch,self.engine.automatic_recognition_results[0])[0]

    def test_reflowed_abbreviation_requires_native_bar_and_exact_other_characters(self):
        from PIL import ImageDraw
        from tralens.ocr import upright_latin as latin
        image=Image.new('RGB',(36,110),'white');ink=ImageDraw.Draw(image)
        for i,width in enumerate([13,12,2,12]):
            ink.rectangle((16-width//2,8+i*23,16+width//2,23+i*23),outline='black',width=2)
        record=dict(source='AB-の',direction='vertical',polygons=[[(0,0),(35,0),(35,109),(0,109)]],
            column_texts=['AB-の'],view_policy=latin.POLICY,readings=['ABIの','ABIの','ABーの'])
        views=latin.presentations(image,record,record['column_texts'])
        self.assertEqual(len(views),2)
        self.assertGreater(views[1].width,views[0].width)
        self.assertTrue(c.select_record(record))
        self.assertFalse(c.select_record(dict(record,readings=['ABIの','ABIの','ABーを'])))
        self.assertFalse(c.select_record(dict(record,readings=['ABLの','ABLの','ABーの'])))
        self.assertIsNone(latin.presentations(image,dict(record,direction='horizontal'),['AB-の']))
        self.assertIsNone(latin.presentations(image,record,['ABの']))

    def test_oversized_color_reader_is_bounded_and_requires_a_dominant_ink(self):
        from PIL import ImageDraw
        from tralens.ocr import display_readings as display
        im=Image.new('RGB',(900,320),'white')
        record=dict(direction='horizontal',bounds=[0,0,900,320],
            polygons=[[[2,2],[897,2],[897,318],[2,318]]])
        self.assertIsNone(display.presentations(im,record))
        for x in range(20,750,90):
            ImageDraw.Draw(im).rectangle((x,65,x+48,240),fill='orange')
        views=display.presentations(im,record)
        self.assertEqual(len(views),3)
        self.assertTrue(all(v.width<=720 and v.height<=320 for v in views))
        self.assertIsNone(display.presentations(im,dict(record,direction='vertical')))
        self.assertFalse(display.candidate([0,0,3500,500],(3500,500),'horizontal'))
    def test_disjoint_halves_must_support_the_full_sound_and_its_length(self):
        from tralens.ocr import sound_readings as sound
        motif='ドキ'
        record=dict(source='読めない語',direction='horizontal',polygons=[[]],scores=[.48],
            readings=[motif*52,motif*4,'別の文字'],sound_windows={
                'view_hashes':['a','b','c'],'readings':[motif*4,motif*2,motif*2]})
        self.assertTrue(c.select_record(record))
        self.assertEqual(c.candidate_text(record),motif*4)
        for reads in [[motif*4,motif*2,'トキトキ'],[motif*12,motif*2,motif*2],
                [motif*4,motif,''],[motif*4,motif*2,motif*8]]:
            from copy import deepcopy
            other=deepcopy(record);other['sound_windows']['readings']=reads
            self.assertFalse(sound.matches(other))
        damaged = dict(record, readings=['トキ'+motif*8,'トキ'+motif*7,'不明'])
        self.assertEqual(sound.hint(damaged),motif)
        self.assertTrue(sound.matches(damaged))
        self.assertIsNone(sound.hint(dict(damaged, readings=['トキ'+motif*2]*3)))
        self.assertIsNone(sound.hint(dict(record,scores=[.98])))

    def test_oblique_sound_requires_three_matching_reduplicated_readings(self):
        record = dict(direction='unknown', source='手人手', scores=[.6],
            view_policy=c.TILTED_VIEWS, readings=['パチパチ']*3)
        self.assertTrue(c.select_record(record))
        for reads in (['パチパチ','パチパチ','ピチピチ'], ['パキッ']*3, ['拍手拍手']*3):
            self.assertFalse(c.select_record(dict(record, readings=reads)))
        self.assertTrue(c.select_record(dict(record, readings=['パチパチパチ','パチパチ','パチパチ'])))
        self.assertFalse(c.select_record(dict(record, readings=['パチパチパチパチ','パチパチ','パチパチ'])))
        self.assertFalse(c.select_record(dict(record, scores=[.98])))

    def test_three_views_keep_original_pixels_and_prior_candidate(self):
        prior=recognize_automatic_batch(self.batch,self.image,Mock(return_value='ネットワー7の'),
            original_image_bytes=self.original,model_identity='prior-synthetic')
        self.engine.automatic_recognition_results=(prior,)
        before=canonical(asdict(self.batch));old=canonical(asdict(prior[0]));source=self.image.tobytes()
        reader=Mock(return_value='ネットワークの');unit=self.refine(reader);reading=c.preferred_reading(unit)
        self.assertEqual(unit.selected.text,'ネットワー7の')
        self.assertEqual(reading.text,'ネットワークの');self.assertEqual(len(unit.readings),3)
        self.assertEqual(canonical(asdict(prior[0])),old)
        self.assertEqual(canonical(asdict(self.batch)),before);self.assertEqual(self.image.tobytes(),source)
        self.assertEqual(reader.call_args_list[0].args[0].tobytes(),self.image.crop(self.batch.proposals[0].crop_bounds).tobytes())
        self.assertEqual(len(list(self.output.glob('*.png'))),3)
        candidate=json.loads(reading.evidence_json);c.validate_pixels(candidate,self.image)
        derived=coverage.derive(unit,reading.revision)
        self.assertEqual(derived.selected,reading);self.assertTrue(derived.source_uncertain)
        self.assertFalse(coverage.provenance(unit,reading.revision)['source_readability_verified'])

    def test_no_change_disagreement_or_failure_keeps_existing_unit_revision(self):
        before=automatic_units(self.batch)[0]
        cases=[['ネットワー7の']*3,['ネットワークの','ネットワークを','ネットワークの'],
            ['ネットワークの',RuntimeError('synthetic reader failure')]]
        root=self.output
        for index,values in enumerate(cases):
            self.output=root/('case-'+str(index));self.output.mkdir()
            self.engine.automatic_recognition_results=()
            unit=self.refine(Mock(side_effect=values))
            self.assertEqual(unit,before);self.assertEqual(unit.revision,before.revision)
            self.assertIsNone(c.preferred_reading(unit))
            self.assertEqual(self.engine.crop_readings['calls'],len(values))
            self.assertTrue(self.engine.crop_readings['records'][0]['readings'])

    def test_selection_protects_strong_readings_latin_spelling_direction_and_disagreement(self):
        self.assertTrue(c.selection('Wgetzmake',['Wgetでmake']*3,[.95,.99],'horizontal'))
        self.assertTrue(c.selection('1941Pチ1P4',['パチパチパチ']*3,[.65],'horizontal'))
        for raw,new,score,direction in [('Ubuntu特集','Ubumtu特集',.4,'horizontal'),
                ('不撓の牛王','不挽の牛王',.99,'horizontal'),('智','ぴる',.2,'unknown'),
                ('2','る!',.2,'horizontal'),('元の文字','別の\t文字',.4,'vertical')]:
            with self.subTest(raw=raw):self.assertFalse(c.selection(raw,[new]*3,[score],direction))
        self.assertFalse(c.selection('元の文字',['同じ文字','同じ文字','別の文字'],[.5],'vertical'))

    def test_existing_selected_reading_keeps_its_translation_revision(self):
        from tralens.ocr import manga_dialogue
        previous=self.output/'prior';previous.mkdir()
        manga_dialogue.refine(self.engine,self.image,self.original,reader=Mock(return_value='ネットワークの'),
            model_identity='prior-synthetic',output=previous)
        original=automatic_units(self.batch,self.engine.automatic_recognition_results[0])[0]
        unit=self.refine(Mock(return_value='ネットワークの'))
        self.assertEqual(unit,original);self.assertIsNone(c.preferred_reading(unit))
        self.assertEqual(self.engine.crop_readings['selected'],0)
        self.assertEqual(self.engine.crop_readings['records'][0]['deferred_to_prior_reading_revision'],
            manga_dialogue.preferred_reading(original).revision)

    def test_candidate_tampering_and_pixels_are_revalidated(self):
        unit=self.refine(Mock(return_value='ネットワークの'));reading=c.preferred_reading(unit)
        candidate=json.loads(reading.evidence_json)
        for key in ('bounds','polygons','scores','direction','selected','readings','main_ids','source'):
            changed=deepcopy(candidate);proof=json.loads(changed['model_settings_json']);record=proof['record']
            if key=='bounds':record[key][0]+=1
            elif key=='polygons':record[key][0][0][0]+=1
            elif key=='scores':record[key][0]=.01
            elif key=='direction':record[key]='horizontal'
            elif key=='selected':record[key]=False
            elif key=='readings':record[key][1]='違う'
            elif key=='main_ids':record[key]=['foreign']
            else:record[key]='other'
            changed['model_settings_json']=canonical(proof)
            with self.subTest(key=key),self.assertRaises(ValueError):automatic_units(self.batch,[changed])
        changed=self.image.copy();changed.putpixel((0,0),(1,2,3))
        with self.assertRaisesRegex(ValueError,'pixels'):c.validate_pixels(candidate,changed)
        changed=deepcopy(candidate);proof=json.loads(changed['model_settings_json']);proof['record']['view_hashes'][0]='0'*64
        changed['model_settings_json']=canonical(proof)
        with self.assertRaisesRegex(ValueError,'pixels'):c.validate_pixels(changed,self.image)
        with self.assertRaises(ValueError):c.views(self.image,[-1,0,20,20],[self.batch.detections[0].polygon])

    def test_coordinator_closes_reader_and_rejects_saved_pixel_drift(self):
        from tests.test_pipeline import PipelineTests
        from tralens.rendering import automatic
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.fixture.engine.image_path=str(fixture.fixture.image)
        reader=Mock(return_value='こんにちわ');reader.identity={'synthetic':True}
        generation=fixture.provider.return_value
        def translated(**kwargs):
            reader.close.assert_called_once()
            return generation
        fixture.provider.side_effect=translated
        with patch('tralens.ocr.hayai_adapter.verify_snapshot',return_value=self.output), \
                patch('tralens.ocr.hayai_adapter.HayaiReader',return_value=reader):
            report=fixture.run_pipeline(crop_reader_model=self.output,render_policy='best-effort')
        self.assertEqual(report['status'],'complete',report);self.assertEqual(report['coverage']['rendered'],2)
        path=fixture.root/'job/preview/preview.json';saved=json.loads(path.read_bytes())
        self.assertEqual(saved['ocr']['crop_readings']['calls'],6)
        units=automatic_units(saved['ocr']['batches'][0],saved['ocr']['automatic_recognition_results'][0])
        for unit,record in zip(units,saved['units']):
            self.assertEqual(c.preferred_reading(unit).text,'こんにちわ')
            self.assertEqual(coverage.bound_result(record,unit)['text'],'Hi!')
        candidate=saved['ocr']['automatic_recognition_results'][0][0]['candidates'][-1]
        proof=json.loads(candidate['model_settings_json']);proof['record']['view_hashes'][0]='0'*64
        candidate['model_settings_json']=canonical(proof)
        changed=fixture.root/'changed.json';changed.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError,'crop reading pixels changed'):
            automatic.run_automatic(preview=changed,image=fixture.fixture.image,fonts=fixture.fonts,
                output=fixture.root/'invalid',font_id='dialogue')

    def test_partial_reader_failure_retained_and_reader_closed_before_translation(self):
        import main
        self.engine.image_path=self.output/'original.png';self.engine.image_path.write_bytes(self.original)
        reader=Mock(side_effect=['ネットワークの',RuntimeError('local failure')]);reader.identity={'synthetic':True}
        with patch('tralens.ocr.hayai_adapter.verify_snapshot',return_value=self.output), \
                patch('tralens.ocr.hayai_adapter.HayaiReader',return_value=reader):
            main.crop_reader_refiner(self.output)(self.engine,output=self.output)
        reader.close.assert_called_once()
        self.assertEqual(self.engine.crop_readings['calls'],2)
        self.assertIn('local failure',self.engine.crop_readings['records'][0]['error'])
        self.assertEqual(automatic_units(self.batch,self.engine.automatic_recognition_results[0]),automatic_units(self.batch))

    def test_pinned_asset_mismatch_fails_before_any_model_import(self):
        from tralens.ocr.hayai_adapter import verify_snapshot
        path=self.output/'model.json';path.write_bytes(b'changed')
        with patch('tralens.ocr.hayai_adapter.FILES',{'model.json':'0'*64}), \
                self.assertRaisesRegex(ValueError,'asset mismatch'):
            verify_snapshot(self.output)
        import subprocess,sys
        script='import sys; import main; import tralens.ocr.hayai_adapter; assert not {"torch","transformers","safetensors"}.intersection(sys.modules)'
        subprocess.run([sys.executable,'-c',script],check=True)


if __name__=='__main__':unittest.main()
