import json
import importlib.util
import unittest
from unittest.mock import Mock

from tests.translation.test_previews import fixture
from tralens.translation.runtime import PreviewTranslator,TRANSLATEGEMMA
from tralens.translation.contracts import TranslationRequest
from tralens.translation import rendition


class SingleRendition(unittest.TestCase):
    def test_two_clauses_do_not_acquire_two_hedges(self):
        from tralens.translation.rendition import printed_cadence
        target = 'It might be too late. Everyone probably hates me.'
        self.assertEqual(printed_cadence('手遅れかも、嫌われた…',target),
                         'It might be too late. Everyone hates me…')
        self.assertIn('probably',printed_cadence('手遅れかも、たぶん嫌われた…',target))

    def test_lettering_preserves_model_word_choice(self):
        for source, target in [('ゆらっ', 'Wobbly'), ('健やかで', 'Stay healthy!'),
                               ('ちょーっと待った', 'Wait up!'),
                               ('シス管の名折れ', "It would be a disgrace to my profession.")]:
            self.assertEqual(rendition.printed_cadence(source, target), target)

    def test_neighbor_name_retry_uses_only_the_short_target(self):
        guide = 'Other dialogue: "Orchid". Please translate the following Japanese text: 見てよ'
        original = {'messages':[{'role':'user','content':guide,'images':['scene']}]}
        self.assertTrue(rendition.borrowed_name('見てよ','Look at Orchid.', ['Orchidを準備した']))
        self.assertFalse(rendition.borrowed_name('オーキッドだ','It is Orchid.', ['Orchidを準備した']))
        self.assertFalse(rendition.borrowed_name('見てよ','Look!', ['Orchidを準備した']))
        isolated = rendition.payload(original, rendition.BORROWED)['messages'][0]
        self.assertNotIn('Orchid', isolated['content'])
        self.assertNotIn('images', isolated)
        self.assertTrue(isolated['content'].endswith('見てよ'))
        self.assertIn('images', original['messages'][0])

    def test_sound_retry_accepts_printable_emphasis_and_rejects_descriptions(self):
        from tralens.translation.contracts import Style
        self.assertEqual(rendition.retry_policy('ばっ！','*JUMP!*',Style()),rendition.SOUND_LETTERING)
        self.assertTrue(rendition.candidate_matches(rendition.SOUND_LETTERING,'*JUMP!*','*WHUMP!*','ばっ！'))
        self.assertFalse(rendition.candidate_matches(rendition.SOUND_LETTERING,'*JUMP!*','He lands on the ground.','ばっ！'))

    def test_printer_strips_model_emphasis_but_preserves_source_asterisks(self):
        self.assertEqual(rendition.printed_cadence('ざあっ', '**Splash!**'), 'Splash!')
        self.assertEqual(rendition.printed_cadence('記号＊', '*star*'), '*star*')

    def test_article_only_changes_the_standalone_english_noun(self):
        self.assertEqual(rendition.printed_cadence('無料OSの実力', 'The power of free OS...'),
                         'The power of a free OS...')
        self.assertEqual(rendition.printed_cadence('無料OSのサポート', 'The cost of free OS support.'),
                         'The cost of free OS support.')

    def test_scene_hints_do_not_assign_a_caption_to_embedded_words(self):
        self.assertFalse(rendition.usage_context('ぽとっ'))
        self.assertTrue(rendition.usage_context('はらはら'))
        self.assertFalse(rendition.usage_context('花びらがはらはらと散った。'))
        self.assertTrue(rendition.usage_context('おすこやかで'))
        self.assertFalse(rendition.usage_context('みんな健やかでうれしい。'))
        self.assertNotIn('leav', rendition.usage_context('医者の名折れだ。'))

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

    def test_request_for_speed_and_ocr_explanations(self):
        from tralens.translation.contracts import Style
        self.assertEqual(rendition.retry_policy('そんでなるはや!', 'And so it happened!', Style()),rendition.URGENT)
        self.assertIsNone(rendition.retry_policy('なるはや！','ASAP!',Style()))
        self.assertTrue(rendition.commentary_output('くハな','Without more context, '+ 'these are possible translations. '*5))
        self.assertFalse(rendition.commentary_output('くハな','Unusual.'))
        self.assertEqual(rendition.visual_retry('力千','Rikisen'),rendition.CLICK)
        self.assertIsNone(rendition.visual_retry('千人','A thousand people'))

    def test_visual_retry_is_selective_and_keeps_valid_localizations(self):
        for source, primary, policy in [
                ('トン', 'Ton', rendition.VISUAL_SOUND), ('トン', 'Tap', None),
                ('トン！', 'Click.', rendition.VISUAL_SOUND), ('カチ', 'Click.', None),
                ('3トン', '3 tons', None), ('勝手に投稿して', 'Post it without permission.', rendition.VISUAL_PERMISSION),
                ('投稿してください', 'Please post it.', None), ('ハパーチク！', 'Hapachiku!', rendition.VISUAL_SOUND),
                ('キーボードにのっちゃったから?', 'Because someone got on the keyboard?', None)]:
            with self.subTest(source=source, primary=primary):
                self.assertEqual(rendition.visual_retry(source, primary), policy)
        self.assertFalse(rendition.candidate_matches(rendition.VISUAL_SOUND, 'Ton', 'It is light.', 'トン'))
        self.assertTrue(rendition.candidate_matches(rendition.VISUAL_SOUND, 'Click.', 'Tap!', 'トン'))
        self.assertFalse(rendition.candidate_matches(rendition.VISUAL_PERMISSION, 'Post it without permission.', 'You posted it without asking.', '勝手に投稿して'))
        self.assertTrue(rendition.candidate_matches(rendition.VISUAL_PERMISSION, 'Post it without permission.', 'Posted it without asking…!', '勝手に投稿して'))

    def test_visual_retry_keeps_target_and_records_crop(self):
        from dataclasses import replace
        _,_,units = fixture()
        unit = units[0]
        reading = replace(unit.selected, text='トン')
        unit = replace(unit, readings=(reading,), selected_revision=reading.revision)
        request = TranslationRequest((unit,), (unit.unit_id,))
        provider = Mock()
        provider.generate.side_effect = ['Ton', 'Tap']
        crop = Mock(return_value=(b'example', (0, 0, 30, 30), ''))
        result = PreviewTranslator(provider, model='translategemma:27b', profile=TRANSLATEGEMMA,
            single_rendition=True, context_crop=crop).translate(request)
        result.assert_current(request)
        crop.assert_called_once_with(unit)
        first, second = [call.kwargs['payload']['messages'][0] for call in provider.generate.call_args_list]
        self.assertNotIn('images',first)
        self.assertTrue(second['images'])
        self.assertEqual(result.translations[0][1], 'Tap')
        metadata = json.loads(result.execution_metadata_json)
        self.assertTrue(metadata['images_supplied'])
        self.assertNotIn('no_context_or_images', metadata['reduced_capabilities'])
        self.assertEqual(metadata['rendition_retry']['primary_translation'], 'Ton')

    def test_ticket_gate_retry_preserves_direction_and_leaves_exits_alone(self):
        from tralens.translation.contracts import Style
        for source, modifier in [('中央改札口','Central'), ('南改札口','South')]:
            self.assertEqual(rendition.retry_policy(source, modifier+' exit', Style()), rendition.TICKET_GATE)
            self.assertTrue(rendition.candidate_matches(rendition.TICKET_GATE, '', modifier+' ticket gate', source))
            self.assertFalse(rendition.candidate_matches(rendition.TICKET_GATE, '', 'East ticket gate', source))
            self.assertIsNone(rendition.retry_policy(source, modifier+' ticket gate', Style()))
        self.assertIsNone(rendition.retry_policy('東出口','East exit',Style()))
        self.assertIsNone(rendition.retry_policy('改札口で会いましょう','Meet at the exit.',Style()))

    @unittest.skipUnless(importlib.util.find_spec('fugashi') and importlib.util.find_spec('unidic_lite'),
                         'optional UniDic readings unavailable')
    def test_banner_term_requires_an_independent_same_page_line(self):
        from types import SimpleNamespace
        def unit(i,page,text):
            reading = SimpleNamespace(text=text, origin='raw_main_reading',
                evidence_json=json.dumps([dict(text=text,confidence=.8)]))
            return SimpleNamespace(unit_id=str(i),member_ids=(i,),page_json=page,selected=reading,
                selected_revision=str(i),structural_reasons=('punctuated_horizontal_heading',))
        target=unit(1,'p','ようこそ!イストール・フェスト')
        other=unit(2,'p','インストール')
        self.assertEqual(rendition.banner_term(target,[other])['english'],'install')
        self.assertIsNone(rendition.banner_term(target,[unit(1,'p','インストール')]))
        self.assertIsNone(rendition.banner_term(target,[unit(2,'another','インストール')]))

    def test_banner_retry_keeps_both_greeting_and_title(self):
        policy = rendition.BANNER_TERM
        source = 'ようこそ!イストール・フェスト'
        self.assertFalse(rendition.candidate_matches(policy,'Welcome!', 'Install Fest!',source,term='install'))
        self.assertTrue(rendition.candidate_matches(policy,'Welcome!', 'Welcome to the Install Fest!',source,term='install'))

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
                ('アップデートからXが止まった','X stopped.')]:
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

    def test_dictionary_gloss_retry_keeps_single_attested_word(self):
        from tralens.translation.contracts import Style
        primary = 'gaze, look, glance'
        policy = rendition.retry_policy('まなざし', primary, Style())
        self.assertEqual(policy, rendition.GLOSS)
        self.assertTrue(rendition.candidate_matches(policy, primary, 'Gaze.'))
        self.assertFalse(rendition.candidate_matches(policy, primary, 'Look at me.'))
        for source, target in [('赤、青、白', 'red, blue, white'), ('はやく', 'go, go, go')]:
            self.assertIsNone(rendition.retry_policy(source, target, Style()))

    def test_incomplete_technical_subject_preserves_version_and_symbol(self):
        from tralens.translation.contracts import Style
        source = '今朝のアップデトから8.10の×が'
        primary = 'The update this morning removed the "×" from version 8.10.'
        policy = rendition.retry_policy(source, primary, Style())
        self.assertEqual(policy, rendition.SUBJECT)
        self.assertTrue(rendition.candidate_matches(policy, primary, 'Since this morning’s update, the 8.10 ×', source))
        for candidate in ('Version 8.1’s ×…', 'Version 8.10…', primary.rstrip('.')+'…'):
            self.assertFalse(rendition.candidate_matches(policy, primary, candidate, source))
        self.assertIsNone(rendition.retry_policy('GPUが止まった', 'The GPU stopped.', Style()))
        self.assertIsNone(rendition.retry_policy('新しいGPUが', 'The new GPU…', Style()))

    def test_literal_slashes_and_single_symbols_do_not_trigger(self):
        for source,target in [('foo / bar','foo / bar'),('了','Done / Finished'),('かな','A/B test'),('かな','Just one sentence.')]:
            self.assertFalse(rendition.needs_retry(source,target))


if __name__=='__main__':unittest.main()
