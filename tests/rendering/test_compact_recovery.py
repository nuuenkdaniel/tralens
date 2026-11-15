"""A successful but stacked layout may improve without changing cleanup."""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image,ImageDraw

from tralens.rendering import aggressive,replacement,natural as n
from .font_fixture import synthetic_font


class CompactRecovery(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(320,360),'white')
        ImageDraw.Draw(self.image).rectangle((155,90,165,250),fill='black')
        self.raw=[{'source_id':1,'polygon':[[150,85],[170,85],[170,255],[150,255]]}]
        self.blank=np.zeros((360,320),bool)
        self.probability=np.zeros(self.blank.shape,np.uint8);self.probability[90:251,155:166]=220
        self.font=synthetic_font();self.coverage=n.font_coverage(self.font)
        self.style={'min_size':16,'max_size':26,'outline':0}

    def render(self,text,*,probability=True):
        rows=[{'unit_id':'one','status':'skipped','binding':{
            'member_ids':[1],'request_revision':'a'*64,'text':text}}]
        return aggressive.recover(self.image,{'size':list(self.image.size),'sha256':'f'*64},self.raw,
            rows,self.image.copy(),Image.new('RGBA',self.image.size),self.blank,self.blank,
            self.font,self.coverage,self.style,probability=self.probability if probability else None)

    def test_compact_retry_preserves_wording_font_floor_and_cleanup_pixels(self):
        text='One two three four five six.'
        result=self.render(text)
        with patch.object(replacement,'patch_layout',side_effect=ValueError('no better fit')):
            control=self.render(text)
        row=result[0][0];prior=row['reconstruction']['prior_placement'];proof=row['layout']
        self.assertEqual(row['reconstruction']['placement_retry'],'compact_source_centered_rectangle')
        self.assertEqual(proof['original_text'],text)
        self.assertGreaterEqual(proof['size'],prior['size'])
        self.assertGreaterEqual(proof['size'],self.style['min_size'])
        self.assertLess(len(proof['draw_text'].splitlines()),len(prior['draw_text'].splitlines()))
        exact=text
        for offset in reversed(proof['inserted_break_offsets']):exact=exact[:offset]+'\n'+exact[offset:]
        self.assertEqual(proof['draw_text'],exact)
        self.assertTrue(np.array_equal(result[4],control[4]))
        self.assertEqual(result[1].tobytes(),control[1].tobytes())
        self.assertEqual(control[0][0]['status'],'automatic_composed')

    def test_explicit_line_breaks_do_not_trigger_compact_retry(self):
        text='One\ntwo\nthree\nfour\nfive\nsix.'
        with patch.object(replacement,'patch_layout',side_effect=AssertionError('explicit lines')):
            result=self.render(text)
        self.assertEqual(result[0][0]['layout']['draw_text'],text)

    def test_broad_polygon_cleanup_cannot_authorize_compact_retry(self):
        with patch.object(replacement,'patch_layout',side_effect=AssertionError('requires stroke support')):
            result=self.render('One two three four five six.',probability=False)
        self.assertEqual(result[0][0]['status'],'automatic_composed')
        self.assertNotIn('prior_placement',result[0][0]['reconstruction'])


if __name__=='__main__':unittest.main()
