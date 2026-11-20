import unittest
from copy import deepcopy
from PIL import Image
from tools.experimental import english_review as english
from tools.experimental import simple_transcription as source
from tralens.translation.contracts import canonical


class Review(unittest.TestCase):
    def test_english_error_does_not_require_source_change(self):
        req=english.request('はい','No.',binding={'source_sha256':'a'*64,'unit_revision':'b'*64,'primary_result_revision':'c'*64})
        result=english.parse(req,canonical({'translation':'Yes.','uncertainties':[],'assessment':'candidate'}))
        self.assertEqual(result['translation'],'Yes.')
        self.assertFalse(result['source_changed']);self.assertTrue(result['primary_retained'])
        self.assertFalse(result['selected'])
        req['evidence']['data']['target_japanese']='いいえ'
        with self.assertRaises(ValueError):english.parse(req,'{}')

    def test_uncertainty_cannot_silently_replace_primary(self):
        req=english.request('はい','Yes.',binding={'source_sha256':'a','unit_revision':'b','primary_result_revision':'c'})
        with self.assertRaises(ValueError):english.parse(req,canonical({'translation':'No.','uncertainties':[],'assessment':'source_uncertain'}))

    def test_simple_transcription_has_no_original_reader_veto(self):
        req,_=source.request(Image.new('RGB',(80,100),'white'),[0,0,80,100],source={'sha256':'a'},member_ids=[1])
        result=source.parse(req,canonical({'text':'Exact visible letters','uncertain':False}))
        self.assertFalse(req['evidence']['original_reader_required_to_agree'])
        self.assertFalse(result['selected'])
        req['payload']['messages'][0]['content']='changed'
        with self.assertRaises(ValueError):source.parse(req,'{}')
