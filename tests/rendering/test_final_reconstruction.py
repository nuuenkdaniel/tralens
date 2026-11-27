"""Removal-only reconstruction must retain binding, budgets and prior edits."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image,ImageDraw

from tralens.rendering import aggressive,natural as n,replacement,reconstruction as rec
from .font_fixture import synthetic_font


class FinalReconstruction(unittest.TestCase):
    def setUp(self):
        pixels=np.full((180,240,3),180,np.uint8)
        pixels[:]=np.arange(240,dtype=np.uint8)[None,:,None]%20+170
        self.image=Image.fromarray(pixels)
        ImageDraw.Draw(self.image).rectangle((96,64,108,112),fill='black')
        self.raw=[{'source_id':1,'polygon':[[90,60],[114,60],[114,116],[90,116]]}]
        self.rows=[{'unit_id':'one','status':'skipped','reason_code':'replacement_layout_confidence',
            'binding':{'member_ids':[1],'request_revision':'a'*64,'text':'Exact words.'}}]
        self.source={'size':list(self.image.size),'sha256':'f'*64}
        self.probability=np.zeros((180,240),np.uint8);self.probability[64:113,96:109]=220
        self.font=synthetic_font();self.coverage=n.font_coverage(self.font)
        self.style={'min_size':16,'max_size':26,'outline':0}

    def test_light_print_on_midtone_panel_respects_border_and_reserved_pixels(self):
        crop = np.full((80,160,3),255,np.uint8)
        crop[5:75,5:150] = 25
        crop[7:73,7:148] = 170
        own = np.zeros(crop.shape[:2],bool)
        own[20:54,20:137] = True
        for x in range(25,130,15):
            crop[23:45,x:x+6] = 255
        blocked = np.zeros_like(own)
        blocked[20:52,21:28] = True
        ink,field,tone = replacement.light_printed_surface(crop,own,blocked)
        self.assertEqual(tone,170)
        self.assertFalse(np.any(ink & blocked))
        self.assertFalse(field[:,5].any())
        self.assertTrue(ink[28,42])
        self.assertIsNone(replacement.light_printed_surface(np.full_like(crop,170),own,blocked))

    def test_caption_strip_preserves_art_and_uses_available_horizontal_space(self):
        crop = np.full((92, 370, 3), [245,223,201], np.uint8)
        own = np.zeros(crop.shape[:2], bool)
        own[28:51,70:270] = True
        for x in range(74, 268, 10):
            crop[30:47,x:x+4] = [40,20,10]
        crop[52:,260:265] = 10
        protected = np.zeros_like(own)
        protected[:,:50] = True
        args = (crop,own,protected,'A new chapter',self.font,self.coverage,self.style)
        ink,tone,tile,(x,y),proof = replacement.flat_caption(*args)
        self.assertTrue(np.array_equal(tone,[245,223,201]))
        self.assertTrue(np.array_equal(ink,crop[:,:,0]==40))
        self.assertFalse(protected[y:y+tile.height,x:x+tile.width][np.asarray(tile.getchannel('A'))>0].any())
        self.assertEqual(proof['draw_text'],'A new chapter')
        crop[:,70:270] = [40,20,10]
        self.assertIsNone(replacement.flat_caption(*args))

    def test_multiline_sign_uses_only_the_connected_paper(self):
        paper = np.zeros((170, 190), bool)
        paper[12:155, 17:174] = True
        paper[50:95, 155:] = False
        target = 'Library hours\nWeekdays\n9 to 4'
        with patch.object(replacement, 'patch_layout', side_effect=ValueError('no rectangle')):
            tile, (x, y), proof = replacement.printed_surface_layout(target, self.font, self.coverage,
                (30, 25, 145, 136), paper, self.style)
        ink = np.asarray(tile.getchannel('A')) != 0
        self.assertTrue(ink.any())
        self.assertTrue(paper[y:y+tile.height, x:x+tile.width][ink].all())
        self.assertEqual(proof['original_text'], target)
        self.assertGreaterEqual(proof['size'], self.style['min_size'])

    def finish(self,backend,*,budget=None,erase=None,clean=None):
        blank=np.zeros((180,240),bool)
        with patch.object(replacement,'patch_layout',side_effect=ValueError('no room')), \
                patch.object(aggressive,'cheap',side_effect=lambda crop,*_:(crop.copy(),'classical_telea_unverified')):
            return replacement.finish(self.image,self.raw,self.rows,clean or self.image.copy(),
                Image.new('RGBA',self.image.size),blank if erase is None else erase,blank,
                self.font,self.coverage,self.style,probability=self.probability,source=self.source,
                backend=backend,budget=budget)

    def test_fresh_request_excludes_prior_edits_and_preserves_source_without_layout(self):
        old=np.zeros((180,240),bool);old[83:87,100:104]=True
        before=np.asarray(self.image).copy();before[old]=[20,30,40]
        calls=[]
        def reconstruct(request,crop,inference,commit,protected,**kwargs):
            rec.validate(request,crop,inference,commit,protected)
            calls.append((request,crop.copy(),commit.copy(),protected.copy()))
            self.assertFalse((commit&protected).any())
            return {'request_revision':request['revision'],'pixels':np.full_like(crop,150)}
        backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=reconstruct))
        rows,clean,text,_,erase,_,artifacts,summary=self.finish(backend,erase=old,clean=Image.fromarray(before))
        self.assertEqual(summary['learned_calls'],1);self.assertEqual(summary['selected_learned'],0)
        self.assertEqual(rows[0]['status'],'skipped');self.assertIsNone(text.getbbox())
        self.assertTrue(np.array_equal(np.asarray(clean),before))
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase],before[~erase]))
        request,crop,commit,protected=calls[0]
        x,y,r,b=request['bounds']
        self.assertTrue(np.array_equal(crop,np.asarray(self.image)[y:b,x:r]))
        self.assertEqual(json.loads(artifacts['recovery-final-000-request.json']),request)
        self.assertFalse(np.any(erase & ~old))
        self.assertTrue(np.array_equal(np.asarray(artifacts['recovery-final-000-protected-mask.png'])!=0,protected))

    def test_backend_failure_and_exhausted_budget_preserve_unplaced_source(self):
        backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=TimeoutError('worker stopped')))
        result=self.finish(backend)
        self.assertEqual(result[-1]['learned_failures'],1)
        self.assertEqual(result[0][0]['removal_fallback']['method'],'local_surface_occlusion')
        self.assertTrue(np.array_equal(result[1],self.image))
        backend.reconstruct.reset_mock()
        result=self.finish(backend,budget=rec.Budget(maximum_calls=0))
        backend.reconstruct.assert_not_called()
        self.assertEqual(result[-1]['removed'],0)
        self.assertEqual(result[-1]['learned_calls'],0)

    def test_neighbor_text_is_hidden_at_native_origin_but_never_committed(self):
        self.raw.append({'source_id':2,'polygon':[[122,65],[138,65],[138,114],[122,114]]})
        self.probability[70:110,126:134]=220
        ImageDraw.Draw(self.image).rectangle((126,70,133,109),fill='black')
        calls=[]
        def reconstruct(request,crop,inference,commit,protected,**kwargs):
            rec.validate(request,crop,inference,commit,protected)
            x,y,r,b=request['bounds'];self.assertGreater(min(x,y),0)
            neighbor=(slice(70-y,110-y),slice(126-x,134-x))
            self.assertTrue(inference[neighbor].all())
            self.assertTrue(protected[neighbor].all())
            self.assertFalse(commit[neighbor].any())
            self.assertTrue(np.array_equal(crop,np.asarray(self.image)[y:b,x:r]))
            calls.append(inference.copy())
            return {'request_revision':request['revision'],'pixels':np.full_like(crop,150)}
        backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=reconstruct))
        _,clean,_,_,erase,_,artifacts,_=self.finish(backend)
        self.assertEqual(len(calls),1)
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase],np.asarray(self.image)[~erase]))
        self.assertTrue(np.array_equal(np.asarray(clean)[70:110,126:134],np.asarray(self.image)[70:110,126:134]))
        self.assertTrue(np.array_equal(np.asarray(artifacts['recovery-final-000-inference-mask.png'])!=0,calls[0]))

    def test_recovery_pass_keeps_inference_only_predictions_out_of_output(self):
        self.rows[0]['binding']['text']='Hi.'
        self.raw.append({'source_id':2,'polygon':[[142,65],[155,65],[155,114],[142,114]]})
        self.probability[70:110,145:151]=220
        ImageDraw.Draw(self.image).rectangle((145,70,150,109),fill='black')
        blank=np.zeros((180,240),bool);calls=[]
        def reconstruct(request,crop,inference,commit,protected,**kwargs):
            rec.validate(request,crop,inference,commit,protected)
            x,y,r,b=request['bounds']
            neighbor=(slice(70-y,110-y),slice(145-x,151-x))
            self.assertTrue(inference[neighbor].all())
            self.assertFalse(commit[neighbor].any())
            self.assertTrue(np.array_equal(crop,np.asarray(self.image)[y:b,x:r]))
            calls.append(inference.copy())
            return {'request_revision':request['revision'],'pixels':np.full_like(crop,150)}
        backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=reconstruct))
        with patch.object(aggressive,'cheap',side_effect=lambda crop,*_:(crop.copy(),'classical_telea_unverified')):
            rows,clean,_,_,erase,_,artifacts,summary=aggressive.recover(self.image,self.source,self.raw,
                self.rows,self.image.copy(),Image.new('RGBA',self.image.size),blank,blank,self.font,
                self.coverage,self.style,probability=self.probability,backend=backend)
        self.assertEqual(len(calls),1);self.assertEqual(summary['selected_learned'],1)
        self.assertEqual(rows[0]['status'],'automatic_composed')
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase],np.asarray(self.image)[~erase]))
        self.assertTrue(np.array_equal(np.asarray(artifacts['recovery-000-0-inference-mask.png'])!=0,calls[0]))

    def test_foreign_result_is_a_hard_error(self):
        backend=SimpleNamespace(config={},reconstruct=Mock(return_value={'request_revision':'foreign','pixels':None}))
        with self.assertRaises(rec.BindingError):self.finish(backend)

    def test_wrong_page_probability_is_not_silently_replaced_by_broad_cleanup(self):
        self.probability=self.probability[1:]
        backend=SimpleNamespace(config={},reconstruct=Mock())
        with self.assertRaisesRegex(rec.BindingError,'probability shape'):
            self.finish(backend)
        backend.reconstruct.assert_not_called()


if __name__=='__main__':unittest.main()
