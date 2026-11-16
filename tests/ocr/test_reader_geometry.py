"""Reader transform orientation, interpolation, borders and immutable bindings."""
from copy import deepcopy
import unittest

import numpy as np
from PIL import Image

from tralens.ocr import reader_geometry as r


class ReaderGeometry(unittest.TestCase):
    def image(self):
        y,x=np.indices((120,160))
        return Image.fromarray(np.stack((x,y,(3*x+7*y)%256),axis=-1).astype(np.uint8))

    def test_axis_aligned_crop_is_exact_and_quarter_turns_are_lossless(self):
        image=self.image();original=image.tobytes();polygon=[[20,30],[60,30],[60,80],[20,80]]
        expected=np.asarray(image)[28:83,18:63]
        for turn in range(4):
            crop,evidence=r.present(image,polygon,quarter_turns=turn)
            np.testing.assert_array_equal(crop,np.rot90(expected,turn))
            np.testing.assert_array_equal(r.replay(image,evidence),crop)
            xs,ys=r.source_coordinates(evidence['geometry'])
            np.testing.assert_allclose(np.asarray(crop)[:,:,0],xs,atol=1e-10)
            np.testing.assert_allclose(np.asarray(crop)[:,:,1],ys,atol=1e-10)
        self.assertEqual(image.tobytes(),original)
        self.assertEqual(polygon,[[20,30],[60,30],[60,80],[20,80]])

    def test_cyclic_and_reversed_boundaries_do_not_mirror_or_rotate_text(self):
        image=self.image();polygon=np.array([[20,20],[110,40],[100,75],[10,55]])
        expected,_=r.present(image,polygon)
        for reverse in (False,True):
            for start in range(4):
                boundary=np.roll(polygon[::-1] if reverse else polygon,start,axis=0)
                crop,evidence=r.present(image,boundary)
                np.testing.assert_array_equal(crop,expected)
                self.assertEqual(evidence['geometry']['source_polygon'],boundary.tolist())

    def test_affine_color_ramp_matches_inverse_coordinates_without_mirroring(self):
        image=self.image();polygon=[[25,25],[115,45],[105,80],[15,60]]
        for view in r.VIEWS:
            for turn in range(4):
                crop,evidence=r.present(image,polygon,view=view,quarter_turns=turn)
                mapping=evidence['geometry'];xs,ys=r.source_coordinates(mapping)
                valid=(xs>=0)&(ys>=0)&(xs<image.width-1)&(ys<image.height-1)
                # A source coordinate ramp is linear, so interpolated red/green
                # recover inverse coordinates within quantization/rounding error.
                np.testing.assert_allclose(np.asarray(crop)[:,:,0][valid],xs[valid],atol=.55)
                np.testing.assert_allclose(np.asarray(crop)[:,:,1][valid],ys[valid],atol=.55)
                q=np.c_[np.asarray(mapping['ordered_polygon']),np.ones(4)]
                mapped=q@np.asarray(mapping['source_to_view']).T;mapped=mapped[:,:2]/mapped[:,2:]
                w,h=mapping['size']
                self.assertTrue(((mapped>=0)&(mapped<np.array([w,h]))).all())

    def test_off_page_context_is_declared_white_without_source_wrapping(self):
        image=self.image();crop,evidence=r.present(image,[[0,0],[20,0],[20,10],[0,10]],view='context')
        mapping=evidence['geometry'];xs,ys=r.source_coordinates(mapping)
        outside=(xs < -1)|(ys < -1)|(xs>image.width)|(ys>image.height)
        self.assertTrue(outside.any())
        self.assertTrue((np.asarray(crop)[outside]==255).all())
        self.assertFalse(mapping['erase_permission'])
        self.assertFalse(mapping['axis_is_writing_direction'])

    def test_perspective_inverse_tracks_nonparallel_edges(self):
        image=self.image();polygon=[[30,20],[120,35],[90,95],[25,75]]
        crop,evidence=r.present(image,polygon,quarter_turns=1)
        mapping=evidence['geometry'];xs,ys=r.source_coordinates(mapping)
        np.testing.assert_allclose(np.asarray(crop)[:,:,0],xs,atol=.55)
        np.testing.assert_allclose(np.asarray(crop)[:,:,1],ys,atol=.55)
        self.assertGreater(abs(mapping['source_to_base'][2][0])+abs(mapping['source_to_base'][2][1]),1e-5)
        original=np.c_[np.asarray(mapping['ordered_polygon']),np.ones(4)]
        destination=original@np.asarray(mapping['source_to_view']).T
        back=destination@np.asarray(mapping['view_to_source']).T
        np.testing.assert_allclose(back[:,:2]/back[:,2:],original[:,:2],atol=1e-9)

    def test_saved_transform_and_source_tampering_are_rejected(self):
        image=self.image();_,evidence=r.present(image,[[20,20],[110,40],[100,75],[10,55]])
        for change in ('source','matrix','view_pixels','selection','runtime'):
            with self.subTest(change=change):
                altered=deepcopy(evidence);source=image.copy()
                if change=='source':source.putpixel((0,0),(1,2,3))
                elif change=='matrix':altered['geometry']['view_to_source'][0][2]+=1
                elif change=='view_pixels':altered['view_pixels_sha256']='0'*64
                elif change=='selection':altered['selected']=True
                else:altered['opencv_version']='other'
                with self.assertRaises(ValueError):r.replay(source,altered)

    def test_bad_or_oversized_geometry_fails_before_sampling(self):
        bad=([[10,10],[20,10],[10,10],[10,20]], # duplicate
            [[10,10],[20,20],[20,10],[10,20]], # crossed
            [[10,10],[20,10],[12,12],[10,20]], # concave
            [[10,10],[20,10],[20,float('nan')],[10,20]],
            [[-1,10],[20,10],[20,20],[10,20]],
            [['10','10'],['20','10'],['20','20'],['10','20']])
        for polygon in bad:
            with self.subTest(polygon=polygon),self.assertRaises(ValueError):r.plan((160,120),polygon)
        with self.assertRaises(ValueError):r.plan((2000,2000),[[0,0],[1800,0],[1800,1800],[0,1800]])
        with self.assertRaises(ValueError):r.plan((160,120),[[0,0],[10,0],[10,10],[0,10]],quarter_turns=True)
        with self.assertRaises(ValueError):r.plan((160,120),[[0,0],[10,0],[10,10],[0,10]],view='expected_text')


if __name__=='__main__':unittest.main()
