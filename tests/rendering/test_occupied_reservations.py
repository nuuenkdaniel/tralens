"""Unused layout support is not committed text or new erase permission."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import automatic as a, preparation as p


class OccupancyTests(unittest.TestCase):
    def fixture(self):
        foreground=np.zeros((100,140),bool);foreground[60:70,15:25]=True
        erase=p.dilate(foreground,1)
        layout=np.zeros_like(foreground);layout[5:95,5:135]=True
        plan=a.AutomaticPlan(foreground,erase,layout,~layout,[255,255,255],{})
        tile=Image.new('RGBA',(16,12));ImageDraw.Draw(tile).rectangle((2,2,13,9),fill='black')
        return plan,(tile,(12,75))

    def test_exact_committed_alpha_erase_and_collar_not_unused_whitespace(self):
        plan,tile=self.fixture();before={k:getattr(plan,k).copy() for k in ('foreground','erase','layout','obstacles')}
        reserved=a.occupied_reservation(plan,tile)
        alpha=np.zeros_like(plan.erase);alpha[77:85,14:26]=True
        self.assertTrue(np.array_equal(reserved,p.dilate(plan.erase|alpha,2)))
        self.assertFalse(reserved[10,100])
        self.assertFalse((plan.erase & ~reserved).any())
        for key,value in before.items():self.assertTrue(np.array_equal(getattr(plan,key),value))

    def test_unused_layout_overlap_can_be_reused_without_discarding_source_ink(self):
        previous,tile=self.fixture()
        foreground=np.zeros_like(previous.foreground);foreground[20:30,60:70]=True
        host=replace(previous,foreground=foreground,erase=p.dilate(foreground,1))
        with self.assertRaises(a.AutomaticSkip):a.reserve_layout(host,previous.layout|previous.erase)
        reserved=a.occupied_reservation(previous,tile)
        updated=a.reserve_layout(host,reserved)
        self.assertTrue(np.array_equal(updated.foreground,host.foreground))
        self.assertTrue(np.array_equal(updated.erase,host.erase))
        self.assertFalse((updated.layout & reserved).any())
        self.assertEqual(updated.evidence['layout_reservation']['cleanup_pixels_discarded'],0)
        self.assertEqual(updated.evidence['layout_reservation']['source_foreground_discarded'],0)

    def test_real_committed_cleanup_overlap_remains_blocked(self):
        plan,tile=self.fixture()
        with self.assertRaises(a.AutomaticSkip):a.reserve_layout(plan,a.occupied_reservation(plan,tile))
        with self.assertRaises(ValueError):a.occupied_reservation(plan,(tile[0],(139,95)))
        bad=tile[0].copy()
        with self.assertRaises(a.AutomaticSkip):a.occupied_reservation(plan,(bad,(0,0)))

    def test_default_route_never_uses_occupied_layout_policy(self):
        from tests.test_pipeline import PipelineTests
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        with patch.object(a,'occupied_reservation',side_effect=AssertionError('default route unchanged')):
            result=fixture.run_pipeline(render_policy='best-effort')
        self.assertEqual(result['coverage']['rendered'],2,result)


if __name__=='__main__':unittest.main()
