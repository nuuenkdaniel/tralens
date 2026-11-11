import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image,ImageDraw

from tralens.rendering import natural as n, replacement, aggressive
from .font_fixture import synthetic_font


class RemovalFirst(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(420,300),'white')
        draw=ImageDraw.Draw(self.image)
        for y in range(90,180,15):draw.rectangle((198,y,205,y+8),fill='black')
        self.raw=[{'source_id':1,'polygon':[[196,86],[208,86],[208,183],[196,183]]}]
        self.rows=[{'unit_id':'one','status':'skipped','reason_code':'replacement_layout_confidence',
            'binding':{'member_ids':[1],'text':'A complete sentence needs more than a narrow column.'}}]
        self.font=synthetic_font();self.coverage=n.font_coverage(self.font)
        self.style={'min_size':16,'max_size':26,'outline':0}

    def finish(self,rows=None):
        mask=np.zeros((300,420),bool)
        return replacement.finish(self.image,self.raw,rows or self.rows,self.image.copy(),
            Image.new('RGBA',self.image.size),mask,mask,self.font,self.coverage,self.style)

    def test_narrow_column_is_removed_and_exact_english_is_widened(self):
        rows,clean,text,composite,erase,layout,_,summary=self.finish()
        self.assertEqual(summary['placed'],1)
        self.assertTrue((np.asarray(clean)[90:180,198:206]==255).all())
        proof=rows[0]['layout'];self.assertGreaterEqual(proof['size'],16)
        exact=self.rows[0]['binding']['text']
        for offset in reversed(proof['inserted_break_offsets']):exact=exact[:offset]+'\n'+exact[offset:]
        self.assertEqual(exact,proof['draw_text'])
        self.assertGreater(rows[0]['removal_fallback']['backing_bounds'][2]-rows[0]['removal_fallback']['backing_bounds'][0],12)
        n.verify_pixel_audit(n.pixel_audit(self.image,clean,composite,erase,np.asarray(text.getchannel('A'))!=0,~(erase|layout)))

    def test_placement_failure_cannot_cancel_cleanup(self):
        with patch.object(replacement,'patch_layout',side_effect=ValueError('no room')):
            rows,clean,text,_,erase,_,_,summary=self.finish()
        self.assertEqual(rows[0]['status'],'source_removed')
        self.assertEqual(summary['removed'],1)
        self.assertIsNone(text.getbbox())
        self.assertTrue((np.asarray(clean)[90:180,198:206]==255).all())
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase],np.asarray(self.image)[~erase]))

    def test_large_neighbor_footprint_cannot_veto_source_but_prior_edits_can(self):
        self.raw.append({'source_id':2,'polygon':[[100,20],[300,20],[300,220],[100,220]]})
        reserved=np.zeros((300,420),bool);reserved[95:100,199:202]=True
        bounds,own,blocked=aggressive.Geometry(self.raw,self.image.size).local([1],reserved,source_priority=True)
        x,y,r,b=bounds
        self.assertTrue((blocked[95-y:100-y,199-x:202-x]).all())
        self.assertFalse((own&blocked&~reserved[y:b,x:r]).any())

    def test_duplicate_members_are_not_placed_twice(self):
        rows=self.rows+[dict(self.rows[0],unit_id='duplicate')]
        result=self.finish(rows)
        self.assertEqual(result[-1]['placed'],1)
        self.assertEqual(result[0][1]['removal_fallback']['reason'],'duplicate_source_membership')

    def test_failed_placement_uses_independent_cleanup_without_speculative_model_work(self):
        self.rows[0]['binding']['request_revision']='a'*64
        def reconstruct(request,crop,inference,commit,protected,**kwargs):
            pixels=crop.copy();pixels[commit]=123
            return {'request_revision':request['revision'],'pixels':pixels}
        backend=SimpleNamespace(config={},reconstruct=Mock(side_effect=reconstruct))
        blank=np.zeros((300,420),bool);text=Image.new('RGBA',self.image.size)
        with patch.object(aggressive.typography,'layout_dialogue',side_effect=ValueError('no local fit')), \
                patch.object(aggressive,'cheap',side_effect=lambda crop,*args:(crop.copy(),'classical_telea_unverified')):
            result=aggressive.recover(self.image,{'size':list(self.image.size),'sha256':'f'*64},self.raw,self.rows,
                self.image.copy(),text,blank,blank,self.font,self.coverage,self.style,backend=backend)
        rows,clean,text,_,erase,layout,artifacts,_=result
        self.assertEqual(rows[0]['reconstruction']['status'],'cleanup_candidate_only')
        backend.reconstruct.assert_not_called()
        with patch.object(replacement,'patch_layout',side_effect=ValueError('no fit anywhere')):
            result=replacement.finish(self.image,self.raw,rows,clean,text,erase,layout,self.font,
                self.coverage,self.style)
        self.assertEqual(result[0][0]['status'],'source_removed')
        self.assertTrue((np.asarray(result[1])[90:180,198:206]==255).all())

    def test_fixed_native_crop_reduces_context_instead_of_resizing_source(self):
        raw=[{'source_id':1,'polygon':[[30,20],[510,20],[510,190],[30,190]]}]
        bounds,own,_=aggressive.Geometry(raw,(600,300)).local([1],np.zeros((300,600),bool))
        x,y,r,b=bounds
        self.assertEqual(r-x,512)
        self.assertTrue((own[20-y:191-y,30-x:511-x]).all())
        raw[0]['polygon'][1][0]=560;raw[0]['polygon'][2][0]=560
        with self.assertRaises(ValueError):aggressive.Geometry(raw,(600,300)).local([1],np.zeros((300,600),bool))


if __name__=='__main__':unittest.main()
