import json
import importlib.util
import unittest
from unittest.mock import Mock

from tests.translation.test_previews import fixture
from tralens.translation.runtime import PreviewTranslator,TRANSLATEGEMMA
from tralens.translation.contracts import TranslationRequest
from tralens.translation import rendition


class SingleRendition(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('fugashi') and importlib.util.find_spec('unidic_lite'),
                         'optional UniDic readings unavailable')
    def test_missed_transfer_command_and_software_package(self):
        from tralens.translation.contracts import Style
        self.assertEqual(rendition.retry_policy('それをおよこしなさい！','Welcome!',Style()),rendition.COMMAND)
        self.assertIsNone(rendition.retry_policy('おかえりなさい！','Welcome!',Style()))
        self.assertTrue(rendition.candidate_matches(rendition.COMMAND,'Welcome!','Hand it over!'))
        self.assertFalse(rendition.candidate_matches(rendition.COMMAND,'Welcome!','Welcome back!'))
        self.assertEqual(rendition.retry_policy('不要パッケージ','extra baggage',Style()),rendition.LOAN)
        self.assertIsNone(rendition.retry_policy('不要パッケージ','an unused package',Style()))
        self.assertEqual(rendition.retry_policy('不要なドライバー','Baggage',Style()),rendition.LOAN)
        self.assertEqual(rendition.retry_policy('お読みなさい','Hello!',Style()),rendition.COMMAND)
        self.assertIsNone(rendition.retry_policy('プリインストールOSの話','The pre-installed operating system',Style()))
        self.assertIsNone(rendition.retry_policy('荷物','baggage',Style()))

    def test_computing_sense_requires_independent_same_page_evidence(self):
        from types import SimpleNamespace
        def unit(i,text,page):
            return SimpleNamespace(page_json=page,unit_id=i,selected=SimpleNamespace(text=text),selected_revision=str(i))
        target=unit(1,'不要なドライバー','a')
        other=unit(2,'ドライバーをインストールする','a')
        self.assertEqual(rendition.technical_context(target,[target,other])['unit_id'],2)
        self.assertIsNone(rendition.technical_context(target,[unit(3,'ドライバーの道具箱','a'),unit(4,other.selected.text,'b')]))

    def test_success_keeps_original_and_binding_and_retains_both_responses(self):
        _,_,units=fixture();request=TranslationRequest(units,(units[0].unit_id,))
        provider=Mock();provider.generate.side_effect=['Hello! / Hi there!','Hello!']
        events=[]
        result=PreviewTranslator(provider,model='translategemma:27b',profile=TRANSLATEGEMMA,
            single_rendition=True).translate(request,observe=events.append)
        result.assert_current(request)
        self.assertEqual(result.translations[0][1],'Hello!')
        self.assertEqual(len(events),2)
        evidence=json.loads(result.execution_metadata_json)['rendition_retry']
        self.assertEqual(evidence['primary_translation'],'Hello! / Hi there!')
        self.assertEqual(evidence['selected'],'single_rendition')
        self.assertEqual(result.attempts,2)

    def test_failed_or_still_alternative_retry_keeps_primary(self):
        _,_,units=fixture();request=TranslationRequest(units,(units[0].unit_id,))
        for failure in (TimeoutError('server uncertain'),RuntimeError('call budget'),'', 'Hello / Hi'):
            provider=Mock();provider.generate.side_effect=['Hello! / Hi there!',failure]
            result=PreviewTranslator(provider,model='translategemma:27b',profile=TRANSLATEGEMMA,
                single_rendition=True).translate(request)
            self.assertEqual(result.status,'preview')
            self.assertEqual(result.translations[0][1],'Hello! / Hi there!')
            self.assertEqual(provider.generate.call_count,2)

    def test_only_retry_lost_contrast_in_japanese_to_english(self):
        from tralens.translation.contracts import Style
        policy = rendition.retry_policy('行きたいけれど','I want to go.',Style())
        self.assertEqual(policy,rendition.UNFINISHED)
        self.assertTrue(rendition.candidate_matches(policy,'I want to go.', 'I want to go, but…'))
        self.assertFalse(rendition.candidate_matches(policy,'I want to go.', 'I want to go, but I cannot.'))
        self.assertFalse(rendition.candidate_matches(policy,'It costs 30.', 'It costs 300, but…'))
        for source,target in [('行きたいけれど','I want to go, though.'),('どこにいるの','Where are you?'),
                ('アップデートからXが','X stopped.')]:
            self.assertIsNone(rendition.retry_policy(source,target,Style()))

    def test_wait_request_lost_in_translation_can_be_retried(self):
        from tralens.translation.contracts import Style
        src = '扉を開けるのは少しお待ちください。'
        self.assertEqual(rendition.retry_policy(src,'Open the door.',Style()),rendition.WAIT)
        self.assertIsNone(rendition.retry_policy(src,'Wait before opening the door.',Style()))
        self.assertIsNone(rendition.retry_policy('どうぞ拍手でお迎えください！','Please applaud!',Style()))
        self.assertTrue(rendition.candidate_matches(rendition.WAIT,'Open the door.','Please wait before opening the door.'))
        self.assertFalse(rendition.candidate_matches(rendition.WAIT,'Open door 3.','Wait before opening door 2.'))
        self.assertFalse(rendition.candidate_matches(rendition.WAIT,'Open the door.','Please open the door.'))

    def test_literal_slashes_and_single_symbols_do_not_trigger(self):
        for source,target in [('foo / bar','foo / bar'),('了','Done / Finished'),('かな','A/B test'),('かな','Just one sentence.')]:
            self.assertFalse(rendition.needs_retry(source,target))


if __name__=='__main__':unittest.main()
