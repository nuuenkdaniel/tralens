"""Real sfnt/TTC parsing and rasterization; original synthetic glyphs only."""
import json
from io import BytesIO
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageFont

from tralens.rendering import natural as n, font_faces as f, replacement
from .font_fixture import synthetic_font, pack


def cjk_font():
    data=bytearray(synthetic_font())
    points=[*range(32,127),ord('こ'),ord('ー')]
    sub=pack('HHIII',12,0,16+12*len(points),0,len(points))
    sub+=b''.join(pack('III',cp,cp,cp-31 if cp<127 else 2) for cp in points)
    cmap=pack('HHHHI',0,1,3,10,12)+sub
    count=struct.unpack_from('>H',data,4)[0]
    for i in range(count):
        if data[12+16*i:16+16*i]==b'cmap':
            struct.pack_into('>4sIII',data,12+16*i,b'cmap',f.checksum(cmap),len(data),len(cmap));break
    data.extend(cmap)
    return bytes(data)


def collection(*fonts):
    header=12+4*len(fonts);body=bytearray();offsets=[]
    for font in fonts:
        offset=header+len(body);offsets.append(offset);copy=bytearray(font)
        for i in range(struct.unpack_from('>H',copy,4)[0]):
            pos=12+16*i+8
            struct.pack_into('>I',copy,pos,struct.unpack_from('>I',copy,pos)[0]+offset)
        body.extend(copy);body.extend(b'\0'*((-len(copy))%4))
    return pack('4sII',b'ttcf',0x10000,len(fonts))+b''.join(pack('I',o) for o in offsets)+body


class FontFaces(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        license=self.root/'license';license.write_bytes(b'Original synthetic test font')
        self.manifest={'schema_version':1,'fonts':[], 'fallbacks':{'latin':['cjk']}}
        for identity,data,index in [('latin',synthetic_font(),None),('cjk',collection(synthetic_font(),cjk_font()),1)]:
            path=self.root/identity;path.write_bytes(data)
            record={'id':identity,'role':'dialogue','weight':'regular','path':str(path),'sha256':n.digest(data),
                'license':'synthetic test','license_path':str(license),'license_sha256':n.digest(license.read_bytes())}
            if index is not None:record['face_index']=index
            self.manifest['fonts'].append(record)

    def test_explicit_face_keeps_glyph_metrics_and_checksum(self):
        data=collection(synthetic_font(),cjk_font())
        for i in (0,1):
            extracted=f.standalone(data,i)
            self.assertEqual(f.checksum(extracted),0xb1b0afba)
            original=ImageFont.truetype(BytesIO(data),24,index=i)
            copied=ImageFont.truetype(BytesIO(extracted),24)
            self.assertEqual(original.getmask('AVこー').tobytes() if hasattr(original.getmask('AVこー'),'tobytes')
                else bytes(original.getmask('AVこー')),bytes(copied.getmask('AVこー')))
            self.assertEqual(original.getlength('AVこー'),copied.getlength('AVこー'))
        self.assertNotIn(ord('こ'),n.font_coverage(f.standalone(data,0)))
        self.assertIn(ord('こ'),n.font_coverage(f.standalone(data,1)))

    def test_collection_needs_valid_explicit_index_and_bounded_tables(self):
        data=collection(synthetic_font())
        for index in (None,True,-1,1,'0'):
            with self.subTest(index=index),self.assertRaises(ValueError):f.standalone(data,index)
        with self.assertRaises(ValueError):f.standalone(synthetic_font(),0)
        for changed in (data[:13],data[:8]+pack('I',33)+data[12:],data[:12]+pack('I',len(data))+data[16:]):
            with self.assertRaises(ValueError):f.standalone(changed,0)

    def test_complete_target_fallback_preserves_latin_layout_and_exact_mixed_text(self):
        faces=f.ApprovedFaces(self.manifest,'latin');mask=Image.new('L',(400,180),255)
        style={'min_size':16,'max_size':24,'outline':0}
        for text,expected in [('AV!', 'latin'),('Try "aこPー".', 'cjk'),('AV!', 'latin')]:
            row={'binding':{'text':text}}
            font=faces.resolve(row);layer,proof=n.layout(text,*font,mask,style)
            self.assertEqual(row['font_selection']['selected_id'],expected)
            self.assertEqual(proof['original_text'],text);self.assertGreaterEqual(proof['size'],16)
            exact=text
            for offset in reversed(proof['inserted_break_offsets']):exact=exact[:offset]+'\n'+exact[offset:]
            self.assertEqual(proof['draw_text'],exact)
            if expected=='latin':self.assertEqual(layer.tobytes(),n.layout(text,*faces.primary,mask,style)[0].tobytes())
        for text in ('Missing 字','Bad\tcontrol','Bad\u202etext'):
            with self.assertRaises(ValueError):faces.resolve({'binding':{'text':text}})

    def test_handwritten_columns_require_independent_tilts(self):
        def column(x, lean):
            return {'polygon': [[x, 0], [x+20, 0], [x+20+lean, 80], [x+lean, 80]]}
        self.assertTrue(f.handwritten_columns([column(80, 9), column(30, -2)]))
        self.assertFalse(f.handwritten_columns([column(80, 9), column(30, 9)]))
        self.assertFalse(f.handwritten_columns([column(80, 0), column(30, 1)]))

    def test_source_role_respects_whole_target_coverage_and_cache(self):
        primary = self.manifest['fonts'][0]
        for role, weight in [('handwriting', 'regular'), ('shout', 'bold')]:
            self.manifest['fonts'].append(dict(primary, id=role, role=role, weight=weight))
        faces = f.ApprovedFaces(self.manifest, 'latin')
        for role, text, expected in [(None, 'Hey', 'latin'), ('handwriting', 'Hey', 'handwriting'),
                                     ('shout', 'Hey', 'shout'), ('handwriting', 'Heyこ', 'cjk')]:
            row = {'binding': {'text': text}, 'lettering_role': role}
            faces.resolve(row)
            self.assertEqual(row['font_selection']['selected_id'], expected)

    def test_fallback_is_explicit_and_checks_license_and_font_bytes(self):
        self.manifest['fallbacks']={}
        with self.assertRaisesRegex(ValueError,'glyph coverage'):
            f.ApprovedFaces(self.manifest,'latin').resolve({'binding':{'text':'こ'}})
        self.manifest['fallbacks']={'latin':['missing']}
        with self.assertRaises(ValueError):f.ApprovedFaces(self.manifest,'latin')
        self.manifest['fallbacks']={'latin':['cjk']};self.manifest['fonts'][1]['sha256']='0'*64
        with self.assertRaises(ValueError):f.ApprovedFaces(self.manifest,'latin')

    def test_removal_and_placement_use_same_fallback_and_missing_glyph_keeps_source(self):
        from .test_removal_first import RemovalFirst
        fixture=RemovalFirst();fixture.setUp()
        faces=f.ApprovedFaces(self.manifest,'latin')
        for target,status in [('Try aこPー.','automatic_composed'),('Missing 字','skipped')]:
            fixture.rows[0]['binding']['text']=target;mask=np.zeros((300,420),bool)
            rows,clean,text,composite,erase,layout,_,summary=replacement.finish(fixture.image,fixture.raw,
                fixture.rows,fixture.image.copy(),Image.new('RGBA',fixture.image.size),mask,mask,
                *faces.primary,fixture.style,font_resolver=faces.resolve)
            self.assertEqual(rows[0]['status'],status)
            self.assertEqual(summary['removed'],int(status=='automatic_composed'))
            if status=='skipped':
                self.assertTrue(np.array_equal(clean,fixture.image))
            else:
                self.assertTrue((np.asarray(clean)[90:180,198:206]==255).all())
            self.assertTrue(np.array_equal(np.asarray(clean)[~erase],np.asarray(fixture.image)[~erase]))
            if status=='automatic_composed':
                self.assertEqual(rows[0]['font_selection']['selected_id'],'cjk')
                self.assertEqual(rows[0]['layout']['original_text'],target)

    def test_automatic_pipeline_keeps_per_unit_face_through_recovery(self):
        from tests.test_pipeline import PipelineTests
        from tralens.translation.runtime import Generation
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.fonts.write_text(json.dumps(self.manifest))
        fixture.provider.side_effect=[Generation('こ','{}'),Generation('Hi!','{}')]
        result=fixture.run_pipeline('mixed',font_id='latin',render_policy='best-effort',
            removal_policy='aggressive',visibility_policy='always-show')
        self.assertIsNone(result['error'],result)
        manifest=json.loads((fixture.root/'mixed/rendering/manifest.json').read_bytes())
        self.assertEqual(manifest['display_summary']['available_not_displayed'],0)
        self.assertEqual([r['font_selection']['selected_id'] for r in manifest['units']],['cjk','latin'])


if __name__=='__main__':unittest.main()
