"""Unused placement clearance cannot pin source ink beside immutable edits."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image,ImageDraw

from tralens.rendering import aggressive,replacement,reconstruction as rec,natural as n,preparation as p
from .font_fixture import synthetic_font


class CleanupClearance(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(240,180),'white')
        ImageDraw.Draw(self.image).rectangle((96,64,108,112),fill='black')
        self.raw=[{'source_id':1,'polygon':[[90,60],[114,60],[114,116],[90,116]]}]
        self.rows=[{'unit_id':'one','status':'skipped','binding':{
            'member_ids':[1],'request_revision':'a'*64,'text':'Hi.'}}]
        self.source={'size':list(self.image.size),'sha256':'f'*64}
        self.old=np.zeros((180,240),bool);self.old[83:87,100:104]=True
        self.before=np.asarray(self.image).copy();self.before[self.old]=[20,30,40]
        self.text=Image.new('RGBA',self.image.size);ImageDraw.Draw(self.text).point((102,93),fill=(8,19,30,255))
        self.alpha=np.asarray(self.text.getchannel('A'))!=0
        self.probability=np.zeros(self.old.shape,np.uint8);self.probability[64:113,96:109]=220
        self.font=synthetic_font();self.coverage=n.font_coverage(self.font)
        self.style={'min_size':16,'max_size':26,'outline':0};self.calls=[]
        def reconstruct(request,crop,inference,commit,protected,**kwargs):
            rec.validate(request,crop,inference,commit,protected)
            self.calls.append((request,commit.copy(),protected.copy()))
            return {'request_revision':request['revision'],'pixels':np.full_like(crop,150)}
        self.backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=reconstruct))

    def run_pass(self,kind,*,placement=True):
        shared=(self.image,self.raw,self.rows,Image.fromarray(self.before),self.text,
            self.old,self.alpha,self.font,self.coverage,self.style)
        if kind=='recovery':
            call=lambda:aggressive.recover(self.image,self.source,*shared[1:],probability=self.probability,backend=self.backend)
        else:
            call=lambda:replacement.finish(*shared,probability=self.probability,source=self.source,backend=self.backend)
        with patch.object(aggressive,'cheap',side_effect=lambda crop,*_:(crop.copy(),'classical_telea_unverified')):
            if placement:return call()
            with patch.object(replacement,'patch_layout',side_effect=ValueError('no room')):return call()

    def check_cleanup(self,result):
        rows,clean,text,_,erase,_,_,_=result
        self.assertEqual(len(self.calls),1)
        committed=self.old|self.alpha;clearance=p.dilate(committed,2)&~committed
        source_ink=np.asarray(self.image)[:,:,0]==0
        released=clearance&source_ink
        self.assertTrue(released.any())
        self.assertTrue((np.asarray(clean)[released]==150).all())
        self.assertTrue(np.array_equal(np.asarray(clean)[committed],self.before[committed]))
        self.assertTrue(np.array_equal(np.asarray(text)[self.alpha],np.asarray(self.text)[self.alpha]))
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase],self.before[~erase]))
        new_alpha=(np.asarray(text.getchannel('A'))!=0)&~self.alpha
        self.assertFalse((new_alpha&p.dilate(committed,2)).any())
        request,commit,protected=self.calls[0];x,y,r,b=request['bounds']
        self.assertFalse((commit&protected).any())
        self.assertTrue(protected[committed[y:b,x:r]].all())
        return rows

    def test_recovery_erases_clearance_ink_while_keeping_edits_and_placement_spacing(self):
        rows=self.check_cleanup(self.run_pass('recovery'))
        self.assertEqual(rows[0]['status'],'automatic_composed')

    def test_failed_placement_keeps_source_and_prior_edits(self):
        rows,clean,text,_,erase,_,_,_=self.run_pass('final',placement=False)
        self.assertEqual(rows[0]['status'],'skipped')
        self.assertTrue(np.array_equal(clean,self.before))
        self.assertTrue(np.array_equal(text,self.text))
        self.assertTrue(np.array_equal(erase,self.old))

    def test_final_tile_stays_outside_prior_layout_clearance(self):
        result=self.run_pass('final');rows=self.check_cleanup(result)
        placement=rows[0]['removal_fallback']
        x,y,r,b=placement.get('backing_bounds') or placement['outlined_tile_bounds']
        self.assertFalse(p.dilate(self.old|self.alpha,2)[y:b,x:r].any())


if __name__=='__main__':unittest.main()
