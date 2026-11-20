"""Original synthetic shapes, not manga/reference masks or live models."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import automatic as a, layout_enclosures as e, natural as n
from tralens.ocr.types import Text_Box
from tralens.ocr.automatic_units import propose_page_from_boxes
from tralens.translation.runtime import Generation
from .font_fixture import synthetic_font


def fixture(color=(255,255,255), shape='ellipse'):
    image=Image.new('RGB',(500,400),color)
    draw=ImageDraw.Draw(image)
    if shape=='ellipse':draw.ellipse((35,20,305,255),outline='black',width=2)
    elif shape=='tail':draw.polygon([(35,20),(305,20),(305,220),(210,220),(190,250),(182,220),(35,220)],outline='black',width=2)
    elif shape=='open':pass
    else:draw.rectangle((35,20,305,255),outline='black',width=1)
    for y in (90,120,150):draw.rectangle((160,y,169,y+11),fill='black')
    raw=[{'source_id':'main','polygon':[[155,82],[175,82],[175,169],[155,169]]}]
    plan=a.derive_plan(image,raw,['main'])
    return image,raw,plan


class EnclosureSpaceTests(unittest.TestCase):
    def expand(self,image,raw,plan,ids=None,reserved=None):
        geometry=a.OriginalComponents(image);geometry.bind_geometry(raw)
        return e.Enclosures(image,raw,geometry).expand(image,raw,['main'] if ids is None else ids,plan,
            np.zeros_like(plan.erase) if reserved is None else reserved)

    def test_closed_curved_offwhite_and_nonrectangular_space_preserve_cleanup(self):
        for color,shape in (((255,255,255),'ellipse'),((240,238,232),'ellipse'),((255,255,255),'tail')):
            with self.subTest(color=color,shape=shape):
                image,raw,plan=fixture(color,shape)
                pixels=np.asarray(image).copy()
                expanded=self.expand(image,raw,plan)
                self.assertGreater(int(expanded.layout.sum()),int(plan.layout.sum()))
                self.assertTrue(np.array_equal(expanded.foreground,plan.foreground))
                self.assertTrue(np.array_equal(expanded.erase,plan.erase))
                self.assertFalse((expanded.layout & expanded.obstacles).any())
                text='A longer dialogue can use the supported space.'
                font=synthetic_font();coverage=n.font_coverage(font)
                with self.assertRaisesRegex(ValueError,'overflow'):
                    n.layout(text,font,coverage,Image.fromarray(plan.layout.astype(np.uint8)*255),
                             {'min_size':16,'max_size':28,'outline':0})
                layer,proof=n.layout(text,font,coverage,Image.fromarray(expanded.layout.astype(np.uint8)*255),
                                     {'min_size':16,'max_size':28,'outline':0})
                self.assertGreaterEqual(proof['size'],16)
                self.assertEqual(proof['original_text'],text)
                clean=pixels.copy();a.apply_cleanup(clean,expanded)
                composite=Image.alpha_composite(Image.fromarray(clean).convert('RGBA'),layer).convert('RGB')
                audit=n.pixel_audit(image,Image.fromarray(clean),composite,expanded.erase,
                                   np.asarray(layer.getchannel('A'))!=0,expanded.obstacles)
                n.verify_pixel_audit(audit)
                self.assertTrue(np.array_equal(np.asarray(image),pixels))

    def test_open_or_diagonally_leaking_frame_never_becomes_an_enclosure(self):
        for shape in ('open','square'):
            image,raw,plan=fixture(shape=shape)
            if shape=='square':image.putpixel((35,20),(255,255,255))
            with self.subTest(shape=shape),self.assertRaises(a.AutomaticSkip):self.expand(image,raw,plan)

    def test_annotation_or_other_text_geometry_inside_same_region_is_not_waived(self):
        image,raw,plan=fixture()
        for polygon in ([[80,80],[100,80],[100,100],[80,100]],raw[0]['polygon']):
            with self.assertRaises(a.AutomaticSkip) as skipped:
                self.expand(image,raw+[{'source_id':'annotation','polygon':polygon}],plan)
            self.assertEqual(skipped.exception.code,'enclosure_neighbor')

    def test_unexplained_gray_art_punctuation_and_texture_remain_blocked(self):
        for color in ((0,0,0),(80,80,80),(250,250,250)):
            image,raw,plan=fixture();ImageDraw.Draw(image).rectangle((100,95,105,110),fill=color)
            with self.subTest(color=color),self.assertRaises(a.AutomaticSkip) as skipped:self.expand(image,raw,plan)
            self.assertEqual(skipped.exception.code,'enclosure_unexplained_ink')

    def test_adjacent_balloon_members_are_not_joined_by_background(self):
        image,raw,plan=fixture(shape='square')
        ImageDraw.Draw(image).line((220,20,220,255),fill='black',width=2)
        raw.append({'source_id':'other','polygon':[[240,100],[260,100],[260,140],[240,140]]})
        with self.assertRaises(a.AutomaticSkip) as skipped:self.expand(image,raw,plan,['main','other'])
        self.assertEqual(skipped.exception.code,'enclosure_distinct')

    def test_reserved_source_and_unavailable_boundary_clearance_reject(self):
        image,raw,plan=fixture()
        with self.assertRaises(a.AutomaticSkip) as skipped:self.expand(image,raw,plan,reserved=plan.erase)
        self.assertEqual(skipped.exception.code,'enclosure_reserved')
        erase=plan.erase.copy();erase[20,170]=True
        with self.assertRaises(a.AutomaticSkip) as skipped:self.expand(image,raw,replace(plan,erase=erase))
        self.assertEqual(skipped.exception.code,'enclosure_clearance')

    def test_foreign_source_pixels_geometry_and_members_reject(self):
        image,raw,plan=fixture();geometry=a.OriginalComponents(image);geometry.bind_geometry(raw)
        context=e.Enclosures(image,raw,geometry)
        changed=image.copy();changed.putpixel((0,0),(0,0,0))
        for page,boxes,ids in ((changed,raw,['main']),
            (image,[{**raw[0],'polygon':[[154,82],[175,82],[175,169],[154,169]]}],['main']),
            (image,raw,['foreign'])):
            with self.assertRaises(ValueError):context.expand(page,boxes,ids,plan,np.zeros_like(plan.erase))

    def test_policy_needs_explicit_confidence_gated_best_effort(self):
        for policy,render,settings in (('unknown','best-effort',{}),('enclosed-retry','conservative',{}),('enclosed-retry','best-effort',None)):
            with self.assertRaises(ValueError):a.layout_settings(policy,render,settings)


class EnclosurePipelineTests(unittest.TestCase):
    def setUp(self):
        from tests.test_pipeline import PipelineTests
        self.pipeline=PipelineTests();self.pipeline.setUp();self.addCleanup(self.pipeline.doCleanups)
        self.root=self.pipeline.root

    def test_actual_pipeline_routes_uncertain_layout_and_keeps_exact_target(self):
        image,raw,plan=fixture()
        image.save(self.pipeline.fixture.image)
        box=Text_Box('main',.98,raw[0]['polygon'],'普通の台詞')
        engine=self.pipeline.fixture.engine
        engine.raw_detections=[[box]]
        engine.automatic_unit_proposals=[propose_page_from_boxes([box],self.pipeline.fixture.image,
            complementary_ownership=True,regroup_companions=True)]
        text='A longer dialogue can use the supported space.'
        self.pipeline.provider.return_value=Generation(text,'{"done":true}')
        common={'render_policy':'best-effort','replacement_policy':'confidence-gated'}
        before=self.pipeline.run_pipeline('before',**common)
        after=self.pipeline.run_pipeline('after',layout_policy='enclosed-retry',**common)
        self.assertEqual(before['coverage']['rendered'],0,before)
        self.assertEqual(after['coverage']['rendered'],1,after)
        self.assertEqual(after['layout_policy'],'enclosed-retry')
        manifest=json.loads((self.root/'after/rendering/manifest.json').read_bytes())
        row=manifest['units'][0]
        self.assertEqual(row['binding']['text'],text)
        self.assertGreaterEqual(row['layout']['size'],16)
        self.assertEqual(row['enclosure_layout_attempt']['status'],'planned')
        self.assertTrue(np.array_equal(np.asarray(Image.open(self.root/'after/rendering/unit-000-applied-erase.png'))!=0,plan.erase))

    def test_existing_strict_successes_are_byte_identical_and_skip_retry(self):
        common={'render_policy':'best-effort','replacement_policy':'confidence-gated','min_ocr_confidence':.1}
        before=self.pipeline.run_pipeline('before',**common)
        with patch.object(e,'Enclosures',side_effect=AssertionError('no retry for success')):
            after=self.pipeline.run_pipeline('after',layout_policy='enclosed-retry',**common)
        self.assertEqual(before['coverage']['rendered'],after['coverage']['rendered'])
        for path in (self.root/'before/rendering').glob('*.png'):
            self.assertEqual(path.read_bytes(),(self.root/'after/rendering'/path.name).read_bytes(),path.name)

    def test_bad_policy_fails_preflight_without_model_or_ocr_calls(self):
        with self.assertRaises(ValueError):self.pipeline.run_pipeline(layout_policy='enclosed-retry')
        self.pipeline.provider.assert_not_called()
        self.pipeline.fixture.engine.process_images.assert_not_called()


if __name__=='__main__':unittest.main()
