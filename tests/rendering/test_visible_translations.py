"""Opaque display policy, real synthetic layout/contracts; no live readers/models."""
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tralens.rendering import visible_translations as v, automatic, natural as n
from .font_fixture import synthetic_font


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(360,280),(100,100,100))
        self.font=synthetic_font();self.coverage=n.font_coverage(self.font)
        self.style={'min_size':16,'max_size':24,'outline':0}

    def place(self,bounds,text,blocked=None):
        own=np.zeros((280,360),bool);x,y,r,b=bounds;own[y:b,x:r]=True
        blocked=np.zeros_like(own) if blocked is None else blocked
        return v.place(self.image,bounds,own,blocked,text,self.font,self.coverage,self.style)

    def test_replace_all_cleans_source_and_preserves_neighbor_pixels(self):
        art = np.full((280,360,3),247,np.uint8)
        own = np.zeros(art.shape[:2],bool)
        own[70:112,120:198] = True
        art[81:99,128:191:5] = 0
        protected = np.zeros_like(own)
        protected[:,193:197] = True
        art[protected] = 22
        kind, patch_layer, letters, (x,y), proof, _ = v.replace_at_source(
            Image.fromarray(art), (120,70,198,112), own, protected, 'Hello!', self.font,self.coverage,self.style)
        mask = np.asarray(patch_layer.getchannel('A')) != 0
        self.assertEqual(kind,'aggressive')
        self.assertEqual(proof['original_text'],'Hello!')
        self.assertIsNotNone(letters.getchannel('A').getbbox())
        self.assertFalse((mask & protected[y:y+mask.shape[0],x:x+mask.shape[1]]).any())
        self.assertTrue((np.asarray(patch_layer)[81-y:99-y,128-x:191-x:5,:3] == 247).all())

    def test_force_draws_edge_and_ruled_cells_after_debug_budget_exhaustion(self):
        from types import SimpleNamespace
        raw,units,rows = [],[],[]
        art = np.full((144,320,3),244,np.uint8)
        for i,(x,y) in enumerate([(0,0),(116,60),(240,120)]):
            art[y+2:y+9,x+3:x+12] = 0
            raw.append(dict(source_id=i,polygon=[[x,y],[x+18,y],[x+18,y+11],[x,y+11]]))
            units.append(SimpleNamespace(selected=SimpleNamespace(text='元')))
            rows.append(dict(status='skipped',reason_code='ruled_table_cell',
                             binding=dict(member_ids=[i],text=('A'*28 if i==0 else 'Hello there!'))))
        image=Image.fromarray(art)
        with patch.object(v,'MAX_ATTEMPTS',0):
            out,_,_,ink,footprint,tiles,summary = v.display(image,raw,units,rows,image,Image.new('RGBA',image.size),
                image,self.font,self.coverage,self.style,max_displays=0,replace_all=True)
        self.assertEqual(summary['aggressive_replacements'],3)
        self.assertEqual(tiles,{})
        self.assertIsNotNone(ink.getbbox())
        np.testing.assert_array_equal(np.asarray(out)[~footprint],art[~footprint])
        for row in rows:
            self.assertNotIn('patch_artifact',row['display_fallback'])
            self.assertEqual(row['display_fallback']['layout']['original_text'],row['binding']['text'])

    def test_reader_floor_overflows_edge_box_without_covering_other_text(self):
        from types import SimpleNamespace
        raw = [dict(source_id=i, polygon=poly) for i, poly in enumerate((
            [[0,0],[14,0],[14,12],[0,12]], [[56,0],[108,0],[108,22],[56,22]]))]
        target = 'Keep the entire message, including punctuation!'
        row = dict(status='skipped', reason_code='ruled_table_cell',
                   binding=dict(member_ids=[0], text=target))
        units = [SimpleNamespace(selected=SimpleNamespace(text='元'))]
        _,_,_,ink,_,_,summary = v.display(self.image,raw,units,[row],self.image,
            Image.new('RGBA',self.image.size),self.image,self.font,self.coverage,
            dict(min_size=24,max_size=30,readability_floor=24),replace_all=True)
        self.assertEqual(summary['displayed_translations'],1)
        proof = row['display_fallback']['layout']
        self.assertEqual(proof['original_text'],target)
        self.assertGreaterEqual(proof['size'],24)
        self.assertGreater(ink.getbbox()[2],15)
        self.assertIsNone(ink.crop((56,0,109,23)).getbbox())

    def test_forced_word_break_keeps_trailing_punctuation_with_letters(self):
        from io import BytesIO
        from PIL import ImageFont
        from tralens.rendering.compact_lettering import wrap
        font = ImageFont.truetype(BytesIO(self.font), 24)
        for word in ('understand.', 'Yaginari,', 'Wait!'):
            lines = wrap(word, font, font.getlength(word[:-1]))
            self.assertEqual(''.join(lines), word)
            self.assertTrue(lines[-1][0].isalpha())
            self.assertGreater(len(lines[-1]), 1)

    def test_dense_reader_caption_can_use_space_beyond_the_first_search_window(self):
        from tralens.rendering import compact_lettering
        blocked = np.ones((240,1200),bool)
        blocked[30:210,920:1195] = False
        art = np.full((*blocked.shape,3),250,np.uint8)
        target = 'Words stay legible even on a densely lettered page.'
        layer,proof = compact_lettering.letters(art,(440,95,490,110),blocked,target,self.font,
            {'min_size':8,'readability_floor':24})
        self.assertEqual(proof['size'],24)
        self.assertEqual(''.join(proof['lines']),target)
        self.assertFalse((np.asarray(layer.getchannel('A')).astype(bool) & blocked).any())

    def test_cleanplate_preserves_crossing_rules_without_joining_unrelated_ends(self):
        for color in ((249,249,249),(216,239,218)):
            paper=np.full((72,116,3),color,np.uint8)
            paper[35:37,:] = (30,65,34)
            paper[:,57:59] = (85,65,34)
            original=paper.copy()
            original[18:54,40:75:7]=0
            original[41,:32]=original[41,82:]=0
            source=np.zeros(original.shape[:2],bool)
            source[13:58,34:80]=True
            clean,_,method=v.source_cleanplate(original,source,np.zeros_like(source),short_effect=False)
            self.assertEqual(method,'dominant_source_surface')
            np.testing.assert_array_equal(clean[35,40:56],paper[35,40:56])
            np.testing.assert_array_equal(clean[17:34,58],paper[17:34,58])
            np.testing.assert_array_equal(clean[41,35:56],paper[41,35:56])

    def test_previously_cleaned_source_can_receive_letters_over_its_own_erasure(self):
        from types import SimpleNamespace
        rgb = Image.new('RGB',(100,84),'white')
        erase = np.zeros((84,100),bool)
        erase[20:62,22:80] = True
        row = dict(status='skipped',reason_code='coverage_uncertain_geometry',
                   binding=dict(member_ids=[1],text='Yes!'),
                   removal_fallback=dict(status='removed',bounds=[22,20,80,62]))
        raw = [dict(source_id=1,polygon=[[24,22],[78,22],[78,60],[24,60]])]
        units = [SimpleNamespace(selected=SimpleNamespace(text='はい'))]
        _,_,_,letters,_,_,summary=v.display(rgb,raw,units,[row],rgb,Image.new('RGBA',rgb.size),rgb,self.font,
            self.coverage,self.style,native_erase=erase,replace_all=True,
            cleanup_artifacts={'recovery-final-000-commit-mask.png':Image.new('L',(58,42),255)})
        self.assertEqual(row['display_status'],'aggressive')
        self.assertGreater(summary['reused_cleanplate_ink_pixels'],0)
        self.assertEqual(summary['native_cleanup_changed_pixels'],0)
        self.assertFalse((np.asarray(letters.getchannel('A'))[~erase]!=0).any())

    def test_heart_glyphs_are_subject_to_the_same_font_coverage_check(self):
        text = 'Hello ♡♥　★ €5 20°C ½'
        n.validate_text(text,set(map(ord,text)))
        with self.assertRaisesRegex(ValueError,'missing glyph'):
            n.validate_text(text,set(map(ord,text)) - {ord('♡')})

    def test_opaque_patch_does_not_need_light_background_or_native_strokes(self):
        with patch('cv2.inpaint',side_effect=AssertionError('no guessed reconstruction')):
            kind,patch_layer,letters,xy,proof,_=self.place((80,60,210,180),'Hello there!')
        self.assertEqual(kind,'patch')
        self.assertTrue((np.asarray(patch_layer.getchannel('A'))==255).all())
        self.assertGreaterEqual(proof['size'],16)
        self.assertEqual(proof['original_text'],'Hello there!')
        self.assertFalse(((np.asarray(letters.getchannel('A'))!=0)&(np.asarray(patch_layer.getchannel('A'))==0)).any())

    def test_narrow_column_uses_readable_associated_caption(self):
        kind,patch_layer,letters,xy,proof,_=self.place((160,80,168,150),'Chairman')
        self.assertEqual(kind,'caption')
        self.assertEqual(proof['size'],16)
        self.assertIn('leader',proof)
        self.assertEqual(proof['original_text'],'Chairman')

    def test_prior_edits_and_neighbor_text_cannot_be_covered(self):
        blocked=np.zeros((280,360),bool);blocked[70:170,90:190]=True
        kind,backing,_,(x,y),proof,_=self.place((80,60,210,180),'Hello there!',blocked)
        self.assertEqual(kind,'caption')
        mask=np.asarray(backing.getchannel('A'))!=0
        self.assertFalse((mask&blocked[y:y+backing.height,x:x+backing.width]).any())
        with self.assertRaises(ValueError):self.place((80,60,210,180),'Hello there!',np.ones_like(blocked))

    def test_exact_slash_alternatives_spaces_and_punctuation_survive(self):
        text='Hello! / Hi?'
        *_,proof,routes=self.place((80,60,230,190),text)
        rendered=text
        for offset in reversed(proof['inserted_break_offsets']):rendered=rendered[:offset]+'\n'+rendered[offset:]
        self.assertEqual(proof['original_text'],text)
        self.assertEqual(rendered,proof['draw_text'])

    def test_bounds_and_resource_failure_are_real_failures_not_tiny_type(self):
        with patch.object(n,'layout',side_effect=n.ResourceLimit('synthetic bounded search')):
            with self.assertRaises(n.ResourceLimit):self.place((80,60,210,180),'Hello!')
        for bad in (True,{},'unknown'):
            with self.assertRaises(ValueError):v.settings(bad)


class VisiblePipelineTests(unittest.TestCase):
    def setUp(self):
        from tests.test_pipeline import PipelineTests
        self.fixture=PipelineTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root

    def test_low_source_score_does_not_gate_display_and_counts_are_separate(self):
        result=self.fixture.run_pipeline('visible',replacement_policy='confidence-gated',visibility_policy='always-show')
        self.assertIsNone(result['error'],result)
        self.assertEqual(result['coverage']['total_translations'],2)
        self.assertEqual(result['coverage']['rendered'],0)
        self.assertEqual(result['display_summary']['displayed_translations'],2)
        self.assertEqual(result['display_summary']['clean_replacements'],0)
        self.assertEqual(self.fixture.provider.call_count,2)
        m=json.loads((self.root/'visible/rendering/manifest.json').read_bytes())
        for row in m['units']:
            self.assertIn(row['display_status'],('patch','caption'))
            self.assertGreaterEqual(row['display_fallback']['layout']['size'],16)

    def test_conservative_clean_choice_still_displays_bound_uncertain_translations(self):
        from dataclasses import replace
        engine=self.fixture.fixture.engine
        engine.automatic_unit_proposals=[replace(self.fixture.fixture.batch,regions=())]
        result=self.fixture.run_pipeline('coverage',visibility_policy='always-show')
        self.assertIsNone(result['error'],result)
        self.assertEqual(result['coverage']['coverage_translations'],2)
        self.assertEqual(result['coverage']['rendered'],0)
        self.assertEqual(result['display_summary']['displayed_translations'],2)

    def test_mode_off_outputs_are_byte_identical(self):
        a=self.fixture.run_pipeline('default')
        b=self.fixture.run_pipeline('explicit',visibility_policy='replacement-only')
        self.assertIsNone(a['error']);self.assertIsNone(b['error'])
        for path in(self.root/'default/rendering').glob('*.png'):
            self.assertEqual(path.read_bytes(),(self.root/'explicit/rendering'/path.name).read_bytes(),path.name)

    def test_separate_budget_exhaustion_preserves_successes(self):
        before=self.fixture.run_pipeline('native')
        with patch.object(v,'MAX_ATTEMPTS',0):after=self.fixture.run_pipeline('limited',visibility_policy='always-show')
        self.assertEqual(after['coverage']['rendered'],before['coverage']['rendered'])
        self.assertEqual((self.root/'native/rendering/composite.png').read_bytes(),(self.root/'limited/rendering/composite.png').read_bytes())

    def test_optional_display_failure_never_rolls_back_native_success(self):
        with patch.object(v,'display',side_effect=n.ResourceLimit('synthetic pass exhaustion')):
            result=self.fixture.run_pipeline('failure',visibility_policy='always-show')
        self.assertIsNone(result['error'],result)
        self.assertEqual(result['coverage']['rendered'],2)
        self.assertTrue(result['display_summary']['rolled_back'])
        self.assertEqual(result['display_summary']['clean_replacements'],2)

    def test_declared_patch_text_composition_and_pixel_locality(self):
        result=self.fixture.run_pipeline('visible',replacement_policy='confidence-gated',visibility_policy='always-show')
        folder=self.root/'visible/rendering'
        clean=Image.open(folder/'clean.png').convert('RGBA');patch_layer=Image.open(folder/'patch.png').convert('RGBA')
        text=Image.open(folder/'text.png').convert('RGBA');final=Image.open(folder/'composite.png').convert('RGB')
        actual=Image.alpha_composite(Image.alpha_composite(clean,patch_layer),text).convert('RGB')
        self.assertEqual(actual.tobytes(),final.tobytes())
        native=Image.open(folder/'native-composite.png').convert('RGB')
        mask=np.asarray(Image.open(folder/'fallback-mask.png'))!=0
        self.assertTrue(np.array_equal(np.asarray(native)[~mask],np.asarray(final)[~mask]))
        self.assertEqual(result['display_summary']['outside_fallback_changed_pixels'],0)

    def test_changed_original_never_becomes_a_new_fallback_authorization(self):
        self.fixture.run_pipeline('initial',visibility_policy='always-show')
        with Image.open(self.fixture.fixture.image) as source:changed=source.convert('RGB')
        changed.putpixel((0,0),(0,0,0));changed.save(self.fixture.fixture.image)
        with self.assertRaises(ValueError):
            automatic.run_automatic(preview=self.root/'initial/preview/preview.json',image=self.fixture.fixture.image,
                fonts=self.fixture.fonts,output=self.root/'stale',visibility_policy='always-show')
        self.assertFalse((self.root/'stale').exists())
