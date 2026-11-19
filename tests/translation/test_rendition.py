import json
import unittest
from unittest.mock import Mock

from tests.translation.test_previews import fixture
from tralens.translation.runtime import PreviewTranslator,TRANSLATEGEMMA
from tralens.translation.contracts import TranslationRequest
from tralens.translation import rendition


class SingleRendition(unittest.TestCase):
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

    def test_literal_slashes_and_single_symbols_do_not_trigger(self):
        for source,target in [('foo / bar','foo / bar'),('了','Done / Finished'),('かな','A/B test'),('かな','Just one sentence.')]:
            self.assertFalse(rendition.needs_retry(source,target))


if __name__=='__main__':unittest.main()
