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

    def finish(self,backend,*,budget=None,erase=None,clean=None):
        blank=np.zeros((180,240),bool)
        with patch.object(replacement,'patch_layout',side_effect=ValueError('no room')), \
                patch.object(aggressive,'cheap',side_effect=lambda crop,*_:(crop.copy(),'classical_telea_unverified')):
            return replacement.finish(self.image,self.raw,self.rows,clean or self.image.copy(),
                Image.new('RGBA',self.image.size),blank if erase is None else erase,blank,
                self.font,self.coverage,self.style,probability=self.probability,source=self.source,
                backend=backend,budget=budget)

    def test_fresh_request_excludes_prior_edits_and_retains_cleanup_without_layout(self):
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
        self.assertEqual(summary['learned_calls'],1);self.assertEqual(summary['selected_learned'],1)
        self.assertEqual(rows[0]['status'],'source_removed');self.assertIsNone(text.getbbox())
        self.assertTrue(np.array_equal(np.asarray(clean)[old],before[old]))
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase],before[~erase]))
        request,crop,commit,protected=calls[0]
        x,y,r,b=request['bounds']
        self.assertTrue(np.array_equal(crop,np.asarray(self.image)[y:b,x:r]))
        self.assertEqual(json.loads(artifacts['recovery-final-000-request.json']),request)
        self.assertTrue(np.array_equal(np.asarray(artifacts['recovery-final-000-commit-mask.png'])!=0,commit))
        self.assertTrue(np.array_equal(np.asarray(artifacts['recovery-final-000-protected-mask.png'])!=0,protected))

    def test_backend_failure_and_exhausted_budget_still_remove_source(self):
        backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=TimeoutError('worker stopped')))
        result=self.finish(backend)
        self.assertEqual(result[-1]['learned_failures'],1)
        self.assertEqual(result[0][0]['removal_fallback']['method'],'local_surface_occlusion')
        self.assertGreater(np.asarray(result[1])[70:110,98:107].min(),0)
        backend.reconstruct.reset_mock()
        result=self.finish(backend,budget=rec.Budget(maximum_calls=0))
        backend.reconstruct.assert_not_called()
        self.assertEqual(result[-1]['removed'],1)
        self.assertEqual(result[-1]['learned_calls'],0)

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
