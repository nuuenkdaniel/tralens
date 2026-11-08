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
