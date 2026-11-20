"""Reader evidence isolation, failure preservation and real workflow boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from tests.ocr import test_automatic_recognition as fixtures
from tralens.ocr import crop_readings, reader_presentations as r
from tralens.ocr.automatic_units import propose_automatic_units
from tralens.translation.contracts import automatic_units


class ReaderPresentations(unittest.TestCase):
    def setUp(self):
        self.image,self.original,self.batch=fixtures.AutomaticRecognitionTests().fixture(text='ネットワー7の')
        self.engine=SimpleNamespace(automatic_unit_proposals=(self.batch,), automatic_recognition_results=())
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.output=Path(tmp.name)

    def collect(self, reader, **options):
        r.collect(self.engine,self.image,self.original,reader=reader,model_identity='synthetic',output=self.output,**options)
        return self.engine.reader_presentations

    def saved(self):
        return {'batches':[r.snapshot(self.engine.automatic_unit_proposals[0])],
            'reader_presentations':r.snapshot(self.engine.reader_presentations)}

    def test_prior_selected_readings_and_revisions_survive_all_new_agreement(self):
        previous=self.output/'previous';previous.mkdir()
        crop_readings.refine(self.engine,self.image,self.original,reader=Mock(return_value='ネットワークの'),
            model_identity='prior-synthetic',output=previous)
        prior=self.engine.automatic_recognition_results;before=automatic_units(self.batch,prior[0])
        self.assertEqual(crop_readings.preferred_reading(before[0]).text,'ネットワークの')
        batch=r.snapshot(self.batch);pixels=self.image.tobytes()
        reader=Mock(return_value='別の文字');stage=self.collect(reader)
        self.assertEqual(stage['status'],'complete');self.assertEqual(reader.call_count,3)
        self.assertIs(self.engine.automatic_recognition_results,prior)
        self.assertEqual(automatic_units(self.batch,prior[0]),before)
        self.assertEqual(r.snapshot(self.batch),batch);self.assertEqual(self.image.tobytes(),pixels)
        r.validate_saved(self.saved(),self.image,output=self.output)
        self.assertTrue(all(row['text']=='別の文字' and not row['selected'] for row in stage['results']))

    def test_exact_pixel_dedup_keeps_all_source_and_orientation_associations(self):
        self.image,self.original,self.batch=fixtures.AutomaticRecognitionTests().fixture(two=True)
        self.engine.automatic_unit_proposals=(self.batch,)
        reader=Mock(return_value='同じ文字');stage=self.collect(reader,quarter_turns=(0,1))
        self.assertEqual(len(stage['plan']['records']),12)
        self.assertEqual(reader.call_count,6)
        self.assertEqual(sorted(i for request in stage['plan']['requests'] for i in request['records']),list(range(12)))
        self.assertTrue(all(len(request['records'])==2 for request in stage['plan']['requests']))
        r.validate_saved(self.saved(),self.image,output=self.output)

    def test_empty_rejected_and_nonquad_records_are_retained_without_text_filtering(self):
        first=self.batch.detections[0]
        raw=(replace(first,text='',selected_text='',status='rejected',confidence=0.),
            replace(first,source_id='triangle',polygon=((80,20),(100,20),(90,45))))
        self.batch=propose_automatic_units(raw,self.image);self.engine.automatic_unit_proposals=(self.batch,)
        reader=Mock(return_value='');stage=self.collect(reader)
        self.assertEqual(reader.call_count,3);self.assertEqual(stage['status'],'partial')
        records=stage['plan']['records']
        self.assertEqual(len(records),6)
        self.assertTrue(all(row['status']=='planned' for row in records[:3]))
        self.assertTrue(all(row['status']=='unavailable' and 'quadrilateral' in row['error'] for row in records[3:]))
        self.assertTrue(all(row['text']=='' and row['status']=='read' for row in stage['results']))
        r.validate_saved(self.saved(),self.image,output=self.output)

    def test_failed_and_oversized_responses_survive_and_later_views_still_run(self):
        before=automatic_units(self.batch);long='あ'*513
        stage=self.collect(Mock(side_effect=[RuntimeError('synthetic failure'),long,'残る文字']))
        self.assertEqual([row['status'] for row in stage['results']],['failed','failed','read'])
        self.assertEqual(stage['results'][1]['text'],long)
        self.assertEqual(self.engine.automatic_recognition_results,())
        self.assertEqual(automatic_units(self.batch),before)
        r.validate_saved(self.saved(),self.image,output=self.output)
        with self.assertRaisesRegex(ValueError,'already attempted'):
            self.collect(Mock())

    def test_source_mapping_associations_predictions_and_saved_pixels_are_checked(self):
        self.collect(Mock(return_value='別の文字'));saved=self.saved()
        for kind in ('matrix','association','selection','text','missing','identity'):
            changed=deepcopy(saved);stage=changed['reader_presentations']
            if kind=='matrix':stage['plan']['records'][0]['presentation']['geometry']['view_to_source'][0][2]+=1
            elif kind=='association':stage['plan']['requests'][0]['records']=[]
            elif kind=='selection':stage['results'][0]['selected']=True
            elif kind=='text':stage['results'][0]['text']='改変'
            elif kind=='missing':stage['results'].pop()
            else:stage['plan']['model_identity']='different'
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.validate_saved(changed,self.image,output=self.output)
        source=self.image.copy();source.putpixel((0,0),(1,2,3))
        with self.assertRaisesRegex(ValueError,'source pixels'):r.validate_saved(saved,source)
        path=self.output/saved['reader_presentations']['plan']['requests'][0]['input']['path']
        with Image.open(path) as opened:crop=opened.convert('RGB')
        crop.putpixel((0,0),(1,2,3));crop.save(path)
        with self.assertRaisesRegex(ValueError,'saved artifact'):r.validate_saved(saved,self.image,output=self.output)

    def test_interrupt_keeps_completed_responses_and_full_input_plan(self):
        with self.assertRaises(KeyboardInterrupt):self.collect(Mock(side_effect=['残る文字',KeyboardInterrupt()]))
        saved=json.loads((self.output/r.DIRECTORY/'evidence.json').read_text())
        self.assertEqual(saved['status'],'interrupted');self.assertEqual(len(saved['plan']['requests']),3)
        self.assertEqual(saved['results'][0]['text'],'残る文字')
        self.assertEqual(self.engine.automatic_recognition_results,())

    def test_coordinator_closes_one_reader_before_translation_and_render_reuses_evidence(self):
        from tests.test_pipeline import PipelineTests
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        engine=fixture.fixture.engine;engine.image_path=str(fixture.fixture.image)
        reader=Mock(return_value='こんにちは');reader.identity={'synthetic':True}
        generation=fixture.provider.return_value
        def translated(**kwargs):
            reader.close.assert_called_once()
            r.validate_saved({'batches':engine.automatic_unit_proposals,
                'reader_presentations':engine.reader_presentations},Image.open(engine.image_path).convert('RGB'),
                output=fixture.root/'job/preview')
            return generation
        fixture.provider.side_effect=translated
        with patch('tralens.ocr.hayai_adapter.verify_snapshot',return_value=self.output), \
                patch('tralens.ocr.hayai_adapter.HayaiReader',return_value=reader) as loaded:
            report=fixture.run_pipeline(crop_reader_model=self.output,reader_presentations=True,render_policy='best-effort')
        self.assertEqual(report['status'],'complete',report);self.assertEqual(report['coverage']['rendered'],2)
        loaded.assert_called_once();reader.close.assert_called_once()
        stage=engine.reader_presentations
        self.assertEqual(reader.call_count,6+len(stage['results']))
        self.assertEqual(stage['status'],'complete');self.assertFalse(report['reader_presentations']['selected'])
        preview=json.loads((fixture.root/'job/preview/preview.json').read_bytes())
        self.assertEqual(preview['ocr']['reader_presentations'],stage)
        self.assertEqual((fixture.root/'job/rendering/evidence-preview.json').read_bytes(),
            (fixture.root/'job/preview/preview.json').read_bytes())
        original=automatic_units(fixture.fixture.batch)
        self.assertEqual([row['unit_revision'] for row in preview['units']],[unit.revision for unit in original])

    def test_workflow_rejects_changed_evidence_before_any_translation(self):
        from tests.test_pipeline import PipelineTests
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        def refine(engine,*,output):
            with Image.open(fixture.fixture.image) as source:
                r.collect(engine,source.convert('RGB'),fixture.fixture.content,reader=Mock(return_value='別の文字'),
                    model_identity='synthetic',output=output)
            engine.reader_presentations['plan']['records'][0]['source_id']='foreign'
        report=fixture.run_pipeline(ocr_refine=refine)
        self.assertEqual(report['status'],'failed',report);fixture.provider.assert_not_called()
        self.assertIn('geometry/association',report['error']['message'])
        self.assertFalse((fixture.root/'job/rendering').exists())

    def test_option_validation_precedes_model_loading(self):
        import main
        from tests.test_pipeline import PipelineTests
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        with patch('tralens.ocr.hayai_adapter.verify_snapshot') as verify:
            with self.assertRaisesRegex(ValueError,'boolean'):main.crop_reader_refiner('unused',reader_presentations='yes')
            with self.assertRaisesRegex(ValueError,'explicit local crop reader'):fixture.run_pipeline(reader_presentations=True)
            verify.assert_not_called()
        fixture.fixture.factory.assert_not_called();fixture.provider.assert_not_called()


if __name__=='__main__':unittest.main()
