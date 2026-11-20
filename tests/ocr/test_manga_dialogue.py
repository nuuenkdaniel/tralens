"""Real candidate/translation/render bindings with synthetic local readers."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image

from tests.ocr import test_automatic_recognition as fixtures
from tests import test_pipeline as pipeline
from tralens.ocr import manga_dialogue as m
from tralens.translation import coverage
from tralens.translation.contracts import automatic_units,canonical,flatten_candidates


class MangaDialogue(unittest.TestCase):
    def test_comma_only_view_difference_keeps_character_agreement(self):
        self.assertTrue(m.selection('はいそこてす','はいそこです','はい、そこです'))
        self.assertFalse(m.selection('ネットワー7の','ネットワークの','ネットワークの？'))

    def setUp(self):
        self.image,self.original,self.batch=fixtures.AutomaticRecognitionTests().fixture(text='ネットワー7の')
        self.engine=SimpleNamespace(automatic_unit_proposals=(self.batch,),automatic_recognition_results=())
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.output=Path(tmp.name)

    def refine(self,reader):
        m.refine(self.engine,self.image,self.original,reader=reader,model_identity='synthetic',output=self.output)
        return automatic_units(self.batch,self.engine.automatic_recognition_results[0])[0]

    def test_consistent_candidate_keeps_raw_and_binds_strict_unit_as_uncertain(self):
        before=asdict(self.batch)
        unit=self.refine(Mock(return_value='ネットワークの'))
        reading=m.preferred_reading(unit)
        self.assertEqual(unit.selected.text,'ネットワー7の')
        self.assertEqual(reading.text,'ネットワークの')
        derived=coverage.derive(unit,reading.revision)
        self.assertEqual(derived.selected.text,reading.text)
        self.assertTrue(derived.source_uncertain)
        self.assertEqual(before,asdict(self.batch))
        m.validate_pixels(json.loads(reading.evidence_json),self.image)
        self.assertEqual(len(list(self.output.glob('*.png'))),2)

    def test_disagreement_and_failed_second_read_retain_evidence(self):
        for second in ('ネットワークを',RuntimeError('fixture failure')):
            with self.subTest(second=second),tempfile.TemporaryDirectory() as out:
                self.output=Path(out);self.engine.automatic_recognition_results=()
                unit=self.refine(Mock(side_effect=['ネットワークの',second]))
                self.assertIsNone(m.preferred_reading(unit))
                stage=self.engine.dialogue_readings
                self.assertEqual(stage['calls'],2)
                self.assertEqual(stage['records'][0]['readings'][0],'ネットワークの')
                self.assertEqual(unit.selected.text,'ネットワー7の')

    def test_source_pixels_and_forged_candidate_are_rejected(self):
        unit=self.refine(Mock(return_value='ネットワークの'))
        reading=m.preferred_reading(unit);candidate=json.loads(reading.evidence_json)
        changed=self.image.copy();changed.putpixel((0,0),(1,2,3))
        with self.assertRaises(ValueError):m.validate_pixels(candidate,changed)
        for change in ('crop','text','record','score','selection'):
            altered=deepcopy(candidate)
            if change=='crop':altered['crop']['bounds'][0]+=1
            elif change=='text':altered['text']='ネットワークを'
            else:
                proof=json.loads(altered['model_settings_json'])
                if change=='score':proof['record']['minimum_raw_score']=.01
                elif change=='selection':proof['record']['selected']=False
                else:proof['record']['readings'][0]='ネットワークを'
                altered['model_settings_json']=canonical(proof)
            fake=replace(reading,text=altered['text'],evidence_json=canonical(altered))
            mutated=replace(unit,readings=(unit.selected,fake))
            with self.assertRaises(ValueError):m.preferred_reading(mutated)

    def test_unbounded_crops_and_forged_safe_neighbor_claim_never_reach_reader(self):
        for bounds in ((-1,0,40,50),(0,0,100000,100000),(0,0,40.5,50)):
            with self.assertRaises(ValueError):m.presentations(self.image,bounds)
        original=self.batch.proposals[0]
        neighbor=replace(self.batch.detections[0],source_id=99)
        self.engine.automatic_unit_proposals=(replace(self.batch,detections=self.batch.detections+(neighbor,)),)
        reader=Mock(return_value='ネットワークの')
        self.refine(reader)
        reader.assert_not_called()
        self.assertTrue(original.crop_safe)

    def test_saved_envelopes_and_complete_coordinator_render(self):
        f=pipeline.PipelineTests();f.setUp();self.addCleanup(f.doCleanups)
        def refine(engine,*,output):
            with Image.open(f.fixture.image) as source:
                m.refine(engine,source.convert('RGB'),f.fixture.image.read_bytes(),
                    reader=lambda crop:'こんにちわ',model_identity='synthetic',output=output)
        result=f.run_pipeline(ocr_refine=refine,render_policy='best-effort')
        self.assertEqual(result['status'],'complete',result)
        saved=json.loads((f.root/'job/preview/preview.json').read_bytes())
        units=automatic_units(saved['ocr']['batches'][0],saved['ocr']['automatic_recognition_results'][0])
        self.assertEqual(len(flatten_candidates(saved['ocr']['automatic_recognition_results'][0])),2)
        for unit,record in zip(units,saved['units']):
            self.assertEqual(m.preferred_reading(unit).text,'こんにちわ')
            self.assertEqual(coverage.bound_result(record,unit)['text'],'Hi!')
        self.assertEqual(result['coverage']['rendered'],2)
