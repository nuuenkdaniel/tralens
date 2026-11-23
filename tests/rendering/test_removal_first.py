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

    def test_short_effect_shifts_clear_of_neighbor_without_losing_ink(self):
        polygon = Image.new('L',(160,100))
        ImageDraw.Draw(polygon).polygon([(20,60),(124,22),(137,50),(33,88)],fill=255)
        own = np.asarray(polygon) != 0
        blocked = np.zeros_like(own)
        style = {**self.style,'max_size':16}
        def place():
            return aggressive.slanted_layout('Coo!',self.font,self.coverage,own,blocked,style)
        first,_,_ = place()
        ys,xs = np.nonzero(np.asarray(first.getchannel('A')))
        outer = xs.argmin()
        blocked[ys[outer],xs[outer]] = True
        layer,proof,content = place()
        alpha = np.asarray(layer.getchannel('A')) != 0
        self.assertIn('placement_adjustment',proof)
        self.assertEqual(alpha.sum(),len(xs))
        self.assertFalse((alpha & blocked).any())
        self.assertFalse((alpha & (np.asarray(content)==0)).any())

    def test_handwriting_placement_avoids_rule_on_clean_plate(self):
        mask = Image.new('L',(200,120),255)
        style = {**self.style,'max_size':18}
        layer,proof = n.layout('Hello',self.font,self.coverage,mask,style)
        plate = np.full((120,200,3),255,np.uint8)
        plate[:,99:102] = 0
        allowed = np.ones((120,200),bool)
        moved,metrics = aggressive.place_on_clean_paper(layer,proof,plate,allowed,self.font,self.coverage,style)
        self.assertEqual(metrics['original_text'],'Hello')
        self.assertGreaterEqual(metrics['size'],style['min_size'])
        alpha = np.asarray(moved.getchannel('A')) != 0
        self.assertFalse(alpha[:,99:102].any())

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

    def test_dominant_color_completes_only_matching_owned_ink(self):
        rgb=np.full((150,300,3),[240,221,204],np.uint8)
        own=np.zeros(rgb.shape[:2],bool);own[10:125,10:245]=True
        initial=np.zeros_like(own);initial[15:115,68:245]=True
        for x in [24,80,142,204,268]:
            rgb[35:95,x:x+20]=[245,151,10]
        rgb[12:30,18:40]=[30,48,230]
        protected=np.zeros_like(own);protected[75:100,18:52]=True
        mask,proof=replacement.complete_color_ink(rgb,own,initial,protected)
        self.assertGreaterEqual(proof['supported_components'],2)
        self.assertTrue(mask[45,30])
        self.assertFalse(mask[85,30] or mask[50,275] or mask[16,20])
        self.assertTrue((mask[initial]).all())

    def test_renderable_greek_letter_in_mixed_caption(self):
        text='Measured at 5 μm.'
        n.validate_text(text,set(map(ord,text)))
        with self.assertRaisesRegex(ValueError,'missing glyph coverage'):
            n.validate_text(text,set(map(ord,text))-set(map(ord,'μ')))

    def test_outlined_ink_requires_repeated_enclosed_light_cores(self):
        page=Image.new('RGB',(180,80),(95,155,120))
        pen=ImageDraw.Draw(page)
        for x in range(15,145,15):
            pen.rectangle((x,18,x+8,44),fill='black')
            pen.rectangle((x+3,22,x+5,38),fill='white')
        pen.rectangle((43,10,55,13),fill=(77,83,95))
        pen.rectangle((95,47,109,50),fill=(3,4,2))
        pixels=np.array(page)
        broad=np.zeros((80,180),bool)
        broad[10:51,8:156]=True
        commit,inference,proof=replacement.outlined_ink(pixels,broad)
        self.assertGreaterEqual(proof['enclosed_light_components'],8)
        self.assertFalse((commit&~broad).any())
        self.assertTrue((commit<=inference).all())
        self.assertTrue(commit[20:43,15:24].all())
        self.assertFalse(commit[11,49])
        self.assertTrue(commit[49,105])
        self.assertFalse(commit[11:13,10:150].any())
        self.assertIsNone(replacement.outlined_ink(pixels,broad & (np.indices(broad.shape)[1]<50)))
        self.assertIsNone(replacement.outlined_ink(np.array(page.convert('L').convert('RGB')),broad))

    def test_painted_fallback_preserves_background_inside_english_tile(self):
        painted = Image.new('RGB',self.image.size,(100,175,130))
        ImageDraw.Draw(painted).rectangle((198,94,204,168),fill='black')
        self.image=painted
        rows,clean,text,_,erase,_,_,summary=self.finish()
        self.assertEqual(summary['placed'],1)
        self.assertEqual(rows[0]['cleanup_method'],'removal_with_outlined_target')
        left,top,right,bottom=rows[0]['removal_fallback']['outlined_tile_bounds']
        self.assertTrue((~erase[top:bottom,left:right]).any())
        self.assertIsNotNone(text.getbbox())
        self.assertFalse((np.array(clean)[95:168,199:204]==0).all(axis=2).any())
        self.assertTrue(np.array_equal(np.array(clean)[~erase],np.array(painted)[~erase]))

    def test_single_line_preserves_gray_paper_behind_english(self):
        self.image = Image.new('RGB', self.image.size, (185, 185, 185))
        ImageDraw.Draw(self.image).rectangle((198, 94, 204, 168), fill='black')
        self.rows[0]['binding']['text'] = 'Look!'
        rows, clean, text, _, erase, _, _, summary = self.finish()
        self.assertEqual(summary['placed'], 1)
        x, y, r, b = rows[0]['removal_fallback']['outlined_tile_bounds']
        self.assertTrue((~erase[y:b, x:r]).any())
        self.assertTrue(np.array_equal(np.asarray(clean)[~erase], np.asarray(self.image)[~erase]))
        self.assertIsNotNone(text.getbbox())

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
