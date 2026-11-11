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
        self.assertGreater(report['reconstruction_summary']['selected_classical'],0)
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

    def test_timeout_and_oom_fall_back_but_foreign_output_is_hard(self):
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
