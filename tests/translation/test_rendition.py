import json
import importlib.util
import unittest
from unittest.mock import Mock, patch

from tests.translation.test_previews import fixture
from tralens.translation.runtime import PreviewTranslator,TRANSLATEGEMMA
from tralens.translation.contracts import TranslationRequest
from tralens.translation import rendition


class SingleRendition(unittest.TestCase):
    def test_playful_retraction_needs_the_standalone_interjection_and_neighboring_joke(self):
        self.assertIn('冗談だよ',rendition.playful_retraction('なんちゃって！',['冗談だよ']))
        self.assertFalse(rendition.playful_retraction('なんて素敵だ',['冗談だよ']))
        self.assertFalse(rendition.playful_retraction('なんて',['何の話？']))

    @unittest.skipUnless(importlib.util.find_spec('fugashi') and importlib.util.find_spec('unidic_lite'),'local dictionary unavailable')
    def test_printed_alias_checks_dictionary_spelling_and_limits_the_phonetic_hint_to_names(self):
        self.assertIn('Catherine',rendition.printed_alias_guide('AKAキャサリン',phonetic_fallback=True))
        self.assertEqual(rendition.printed_alias_guide('キャサリンと出かけた'), '')
        self.assertIn('igunuchiusu',rendition.printed_alias_guide('AKA聖イグヌチウス').lower())

    def test_separately_printed_contempt_suffix_is_distinct_from_a_word_ending_in_me(self):
        self.assertTrue(rendition.contemptuous_address('町の悪党め！',['町の','悪党','め！']))
        self.assertFalse(rendition.contemptuous_address('私のため…',['私の','ため…']))
        self.assertFalse(rendition.contemptuous_address('町の悪党め！',['町の','敵','め！']))

    def test_second_category_requires_the_listed_noun_in_both_lines(self):
        first = '静かなエンジンと'
        second = 'そうではない騒々しいエンジンです'
        self.assertIn('first item',rendition.contextual_dialogue_prompt(first,[second]))
        self.assertNotIn('first item',rendition.contextual_dialogue_prompt(first,['そうではない人です']))
        self.assertIn('Continue the list',rendition.contextual_dialogue_prompt(second,[],first))
        self.assertNotIn('Continue the list',rendition.contextual_dialogue_prompt(second,[],'それは本当ですか'))

    def test_item_announcement_requires_an_introduction_of_the_same_object(self):
        source = 'バリアの護符！'
        prompt = rendition.announced_item_effect(source, ['これが私の護符だ。','身を守る効果がある。'])
        self.assertIn('effect', prompt)
        self.assertTrue(prompt.endswith(source))
        self.assertFalse(rendition.announced_item_effect(source, ['これが僕の巻物だ。','身を守る効果がある。']))
        self.assertFalse(rendition.announced_item_effect(source, ['これが私の護符。']))

    def test_interrupted_word_retains_the_source_prefix_and_ellipsis(self):
        self.assertEqual(rendition.interrupted_word('{"japanese_completion":"返信","english":"reply"}','返し…'),'Reply…')
        for invalid in ('{"japanese_completion":"着信","english":"call"}','{"japanese_completion":"返信","english":"返信"}'):
            with self.assertRaises(ValueError):
                rendition.interrupted_word(invalid,'返し…')

    def test_effect_envelope_keeps_only_single_printable_field(self):
        self.assertEqual(rendition.effect_field('{"effect":"Swish!"}'),'Swish!')
        for bad in ('{"effect":"swish","explanation":"a fan"}','{"effect":"first\\nsecond"}','{}',None):
            with self.assertRaises(ValueError):
                rendition.effect_field(bad)
    def test_extracts_only_a_separately_quoted_answer_to_a_translation_request(self):
        source='初めて投稿します。'
        message='Okay, here’s the translation of the post:\n\n“This is my first post.”\n'
        self.assertEqual(rendition.quoted_response_body(source,message),'This is my first post.')
        bold=message.replace('“This is my first post.”','**“This is my first post.”**')
        self.assertEqual(rendition.quoted_response_body(source,bold),'This is my first post.')
        self.assertEqual(rendition.quoted_response_body('これが翻訳です。',message),message)
        self.assertEqual(rendition.quoted_response_body(source,'“This is my first post.”'),'“This is my first post.”')
    def test_footnote_requires_the_same_name_in_nearby_dialogue_to_join_radicals(self):
        source='※理科研究同女子会では"Linux"を使用'
        repaired=rendition.self_contained_prompt(source,['私たちは理科研究同好会の一員だ。'])
        self.assertTrue(repaired.endswith('※理科研究同好会では"Linux"を使用'))
        self.assertTrue(rendition.self_contained_prompt(source,['女子会だよ']).endswith(source))
        self.assertIn('command',rendition.contextual_dialogue_prompt('ミセロ！',[]))
        self.assertFalse(rendition.contextual_dialogue_prompt('キロ',[]))

    def test_direct_commands_resolve_a_single_previous_person(self):
        utterance='どうして迷っているの？起こして、呼んで、連れて来なさい！'
        context=['この人がお姉さん？','同い年に見える。']
        self.assertIn(context[0],rendition.contextual_dialogue_prompt(utterance,context))
        self.assertFalse(rendition.contextual_dialogue_prompt(utterance,context[:1]))
        self.assertFalse(rendition.contextual_dialogue_prompt('電話してね。',context))
        self.assertIn('安く買えてよかった',rendition.contextual_dialogue_prompt('良かったのか？',['安く買えてよかった。']))
    def test_inline_hesitation_and_counted_noun_require_source_evidence(self):
        self.assertIn('stumbles',rendition.printed_word_guide('新しいぷ・ぷりんた？'))
        self.assertFalse(rendition.printed_word_guide('メアリー・スミス'))
        self.assertIn('予約システム',rendition.printed_word_guide('予約システムひとつまともに使えない'))
        self.assertFalse(rendition.printed_word_guide('りんごをひとつ食べた'))

    def test_omitted_object_requires_an_adjacent_same_page_topic(self):
        from types import SimpleNamespace
        def unit(identity, text, page='a'):
            return SimpleNamespace(unit_id=identity, page_json=page, selected=SimpleNamespace(text=text))
        last = unit(1,'新しい部品だよね')
        current = unit(2,'修理してやんねーと')
        self.assertEqual(rendition.preceding_topic(current.selected.text,current,[last,current]), last.selected.text)
        self.assertFalse(rendition.preceding_topic('扉を修理してやんねーと', current,[last,current]))
        self.assertFalse(rendition.preceding_topic(current.selected.text,current,[unit(1,last.selected.text,'b'),current]))

    def test_standalone_mimetic_and_contextual_term(self):
        for term in ('ちょ！Ｗ','ほんとww'):
            self.assertTrue(rendition.chat_laughter(term))
        for term in ('消費電力95W','ゲームW','W杯'):
            self.assertFalse(rendition.chat_laughter(term))
        self.assertTrue(rendition.impact_sound('カタン'))
        self.assertFalse(rendition.impact_sound('ガチ'))
        self.assertFalse(rendition.impact_sound('カチンと来た'))
        self.assertTrue(rendition.cosmetic_practice('エンジェル・メイク'))
        self.assertFalse(rendition.cosmetic_practice('メイク'))
        self.assertIn('mortuary',rendition.usage_context('エンゼルメイク'))
        self.assertIn('fashion',rendition.usage_context('エンゼルメイク'))

    def test_onset_comes_from_independently_printed_columns(self):
        from types import SimpleNamespace
        raw = SimpleNamespace(origin='raw_main_reading',evidence_json=json.dumps(
            [{'source_id':5,'text':'い'},{'source_id':2,'text':'いや'}]))
        u = SimpleNamespace(selected=SimpleNamespace(text='いいや'),readings=(raw,),order=(5,2))
        self.assertTrue(rendition.stuttered_reading(u))
        self.assertEqual(rendition.printed_cadence('いいや','No.',unit=u),'N-No.')
        self.assertEqual(rendition.printed_cadence('いいや','N-no.',unit=u),'N-no.')
        self.assertFalse(rendition.split_onset('いいえ',['いいえ']))

    def test_self_contained_requests_do_not_override_greetings_or_complete_claims(self):
        note=rendition.self_contained_prompt('※動物研究部の“Cat”という名前。',
                                            ['動物研究部ができたよ。','明日は休みだ。'])
        self.assertIn('動物研究部ができたよ',note)
        self.assertNotIn('明日は休み',note)
        self.assertIn('granting access',rendition.self_contained_prompt('うちのシャワーを貸すよ'))
        self.assertFalse(rendition.self_contained_prompt('この傘を貸すよ'))
        self.assertIn('元気',rendition.usage_context('ショボーン…'))
        self.assertIn('外れ',rendition.usage_context('カポッ!'))
        self.assertIn('明るく',rendition.usage_context('ぱあああ'))
        self.assertFalse(rendition.usage_context('かぽっと外れた'))
        self.assertTrue(rendition.self_contained_prompt('じゃこれもよろしく'))
        self.assertTrue(rendition.self_contained_prompt('僕はその中でも'))
        self.assertFalse(rendition.self_contained_prompt('どうぞよろしくお願いします。'))
        self.assertFalse(rendition.self_contained_prompt('僕はその中でも一番弱かった。'))

    def test_print_punctuation_and_lexical_repeats(self):
        self.assertEqual(rendition.printed_cadence('【夜】','【Night】 falls.'),'[Night] falls.')
        self.assertEqual(rendition.printed_cadence('【夜】','【夜】'), '【夜】')
        self.assertEqual(rendition.printed_cadence('-予告-', '-Preview-'), '– Preview –')
        self.assertEqual(rendition.printed_cadence('予告', 'pre-release'), 'pre-release')
        self.assertEqual(rendition.printed_cadence('へへへ','Hehehe!'),'He he he!')
        self.assertEqual(rendition.printed_cadence('笑い声','Hehehe!'),'Hehehe!')
        self.assertEqual(rendition.interface_acronym('GUーの画面',['User', 'Interface']),'GUIの画面')
        self.assertEqual(rendition.interface_acronym('GUーの画面',['Music', 'Editor']),'')

    @unittest.skipUnless(importlib.util.find_spec('jaconv'), 'requires Japanese transliteration')
    def test_spelling_note_must_contrast_the_same_spoken_word(self):
        cue = rendition.quoted_term('こんぴゅた',['※こんぴゅたではなくコンピューターです'])
        self.assertIn('konpyuta',cue)
        self.assertFalse(rendition.quoted_term('こんぴゅた',['コンピューターです']))
        self.assertFalse(rendition.quoted_term('りんご',['※りんごではなくミカンです']))
        self.assertIn('confident offer',rendition.usage_context('私の道具があれば！特別な道具です！'))
        self.assertFalse(rendition.usage_context('あの道具があれば帰れる。'))

    def test_disputed_toolbar_label_needs_independent_readings_and_file_listing(self):
        from types import SimpleNamespace
        from tralens.translation.contracts import Reading,canonical
        def label(raw_text,score=.64):
            raw = Reading(raw_text,'raw_main_reading',canonical([{'confidence':score,'polygon':[[1,1],[28,1],[28,18],[1,18]]}]))
            revised = Reading('設是','consistent_crop_reading_unverified','{}')
            return SimpleNamespace(readings=(raw,revised),selected=revised,member_ids=(8,))
        files = ['daily.pdf','notes.odt']
        self.assertTrue(rendition.disputed_toolbar_label(label('設足'),files))
        self.assertFalse(rendition.disputed_toolbar_label(label('設是'),files))
        self.assertFalse(rendition.disputed_toolbar_label(label('設足',.99),files))
        self.assertFalse(rendition.disputed_toolbar_label(label('設足'),['daily.pdf','daily.pdf']))

    def test_missing_voicing_mark_requires_a_locative_context(self):
        self.assertEqual(rendition.clarified_spelling('村のはすれにある学校'),'村の外れにある学校')
        self.assertFalse(rendition.clarified_spelling('このことはすれ違いでした'))

    def test_hostile_challenge_requires_a_printed_hostile_address(self):
        self.assertEqual(rendition.clarified_spelling('やる気か、コラー！'),'喧嘩を売ってるのか、コラー！')
        self.assertEqual(rendition.clarified_spelling('やる気か'),'')
        self.assertEqual(rendition.clarified_spelling('ちゃんとやるのか'),'')
        self.assertEqual(rendition.clarified_spelling('マトモに働けない'),'ちゃんと働けない')

    def test_only_source_corroborated_separate_beats_repeat_the_selected_word(self):
        from types import SimpleNamespace
        def unit(reason):
            return SimpleNamespace(selected=SimpleNamespace(origin='raw_main_reading',
                evidence_json=json.dumps([{'reason':reason}])))
        raw=SimpleNamespace(origin='raw_main_reading',evidence_json=json.dumps(
            [{'source_id':3,'text':'シヤカ'},{'source_id':5,'text':'シヤカ'}]))
        separate=SimpleNamespace(order=(3,5),readings=(raw,),selected=raw)
        self.assertEqual(rendition.printed_cadence('シャカシャカ','Shake.',unit=separate),'Shake shake.')
        self.assertEqual(rendition.printed_cadence('シャカシャカ','Shake it.',unit=separate),'Shake it.')
        printed = unit('corroborated_repeated_effect')
        self.assertEqual(rendition.printed_cadence('ガタガタッ','clack!',unit=printed),'clack clack!')
        self.assertEqual(rendition.printed_cadence('ガタガタッ','clack clack!',unit=printed),'clack clack!')
        self.assertEqual(rendition.printed_cadence('ガタガタッ','trembles',unit=unit('paddle')),'trembles')
        self.assertEqual(rendition.printed_cadence('ガタッ','clack!',unit=printed),'clack!')

    def test_only_repeated_speech_onset_stammers(self):
        self.assertEqual(rendition.printed_cadence('ちっちがうよ','That’s wrong!'),'Th-That’s wrong!')
        self.assertEqual(rendition.printed_cadence('ちっちがうよ','Th-that’s wrong!'),'Th-that’s wrong!')
        self.assertEqual(rendition.printed_cadence('しっしっ','Shoo!'),'Shoo!')
        self.assertEqual(rendition.printed_cadence('あっあのUSBは？','Where is that USB?'),'W-Where is that USB?')
        self.assertIn('postpone',rendition.usage_context('予約するのをお待ちください。'))
        self.assertFalse(rendition.usage_context('友達を待ってください。'))

    def test_sound_senses_are_isolated_and_keep_shoo_separate_from_shushing(self):
        self.assertEqual(rendition.usage_context('しっしっ'),rendition.usage_context('シッシッ！'))
        self.assertIn('追い払う',rendition.usage_context('しっしっ'))
        self.assertFalse(rendition.usage_context('しーっ'))
        self.assertFalse(rendition.usage_context('コンを選んで'))

    @unittest.skipUnless(importlib.util.find_spec('fugashi') and importlib.util.find_spec('unidic_lite'), 'requires Japanese dictionary')
    def test_emphasized_adjective_and_unfinished_document_reference(self):
        self.assertEqual(rendition.clarified_spelling('これオカしくない？'),'これ可笑しくない？')
        self.assertEqual(rendition.clarified_spelling('日記には一行も…'),'日記には一行も書かれていない…')
        self.assertFalse(rendition.clarified_spelling('日記には一行も残らない'))
        self.assertFalse(rendition.clarified_spelling('ゲームしない？'))

    def test_inference_marker_starts_its_own_printed_line(self):
        from types import SimpleNamespace
        def line(parts):
            return SimpleNamespace(selected=SimpleNamespace(origin='raw_main_reading',text=''.join(parts),
                evidence_json=json.dumps([{'text':t} for t in parts])))
        self.assertEqual(rendition.printed_clause_break(line(['扉が開く','そう、この鍵で'])),
                         '扉が開く。そう、この鍵で')
        self.assertFalse(rendition.printed_clause_break(line(['扉が開くそう、この鍵で'])))
        self.assertFalse(rendition.printed_clause_break(line(['聞いたら','そう、それだと'])))

    def test_colloquial_reading_preserves_address_and_a_cutoff(self):
        self.assertEqual(rendition.forum_reading('おい、モチツケ！'),'おい、落ち着け！')
        self.assertEqual(rendition.forum_reading('まあモチツ'),'まあ落ち着…')
        self.assertFalse(rendition.forum_reading('餅つけ'))
        self.assertFalse(rendition.forum_reading('君に教えて'))

    def test_positive_sigh_accepts_ascii_ellipsis_but_not_negated_praise(self):
        self.assertEqual(rendition.visual_retry('うう...嬉しいよ…','Ugh... I am happy.'),rendition.VISUAL_REACTION)
        self.assertFalse(rendition.positive_reaction('うぅ…嬉しくないよ'))
        self.assertFalse(rendition.positive_reaction('うぅ…嬉しいけど'))
        self.assertFalse(rendition.visual_retry('うぅ…嬉しいよ…','Oh... I am happy.'))

    @unittest.skipUnless(importlib.util.find_spec('fugashi') and importlib.util.find_spec('unidic_lite'), 'requires Japanese dictionary')
    def test_common_noun_retry_leaves_names_and_complete_phrases_alone(self):
        from tralens.translation.contracts import Style
        self.assertEqual(rendition.retry_policy('形勢','Formidable',Style()),rendition.NOUN_LABEL)
        self.assertIsNone(rendition.retry_policy('形勢','An unfavorable situation.',Style()))
        self.assertIsNone(rendition.retry_policy('危険','Dangerous',Style()))
        self.assertTrue(rendition.candidate_matches(rendition.NOUN_LABEL,'Formidable','Situation'))
        self.assertFalse(rendition.candidate_matches(rendition.NOUN_LABEL,'Formidable','Impressive'))

    def test_comparative_uses_only_one_printed_directive_in_the_same_panel(self):
        from types import SimpleNamespace
        def line(source,x,page=1):
            selected = SimpleNamespace(text=source,origin='raw_main_reading',
                evidence_json=json.dumps([{'polygon':[[x,20],[x+4,20],[x+4,37],[x,37]]}]))
            return SimpleNamespace(selected=selected,page_json=page)
        source = line('それなら私より',30)
        recipient = line('先生に頼めよ',50)
        bounds = (10,5,68,50)
        self.assertTrue(rendition.RECIPIENT_COMPARISON.fullmatch(source.selected.text))
        self.assertFalse(rendition.RECIPIENT_COMPARISON.fullmatch('彼女は僕より'))
        restore = lambda others: rendition.comparative_continuation(source,others,bounds)
        self.assertEqual(restore([recipient,line('友達に聞けよ',80)]),'それなら私に頼むより…')
        self.assertIsNone(restore([recipient,line('友達に聞けよ',40)]))
        self.assertIsNone(restore([line('先生に頼めよ',50,page=2),line('ずっと速かった',50)]))

    def test_panel_context_follows_internal_gutters_and_falls_back_on_blank_paper(self):
        from PIL import Image, ImageDraw
        from tralens.translation.workflow import _panel_bounds
        page = Image.new('RGB',(360,240),'white')
        seed = (115,80,128,115)
        fallback = (70,30,180,165)
        self.assertEqual(_panel_bounds(page,seed,fallback),fallback)
        draw = ImageDraw.Draw(page)
        draw.rectangle((12,12,162,225),fill='#aaa')
        draw.rectangle((175,12,348,225),fill='#777')
        left,top,right,bottom = _panel_bounds(page,seed,fallback)
        self.assertLess(left,seed[0])
        self.assertGreater(bottom,seed[3])
        self.assertLess(top,seed[1])
        self.assertTrue(162 < right < 175)

    def test_surprised_shortage_needs_independent_context(self):
        cue = rendition.shortage_exclamation
        self.assertIn('品切れだ', cue('あら、紅茶は品切れじゃないの！', ['それがないと困るわ']))
        self.assertFalse(cue('あら、紅茶は品切れじゃないの！', ['棚を見て']))
        self.assertFalse(cue('水は空じゃないの。', ['それがないと困るわ']))
        self.assertFalse(cue('まあ、赤じゃないの？', ['これがないと困るわ']))

    def test_detached_connective_cannot_borrow_a_sentence(self):
        self.assertTrue(rendition.expanded_short_label('って', "She isn't coming!"))
        self.assertFalse(rendition.expanded_short_label('って', 'But…'))
        self.assertFalse(rendition.expanded_short_label('いないし！', "He isn't here!"))

    def test_multiword_latin_label_needs_accepted_evidence(self):
        from dataclasses import replace
        from tralens.translation.contracts import canonical
        base = fixture()[2][0]
        for source, confidence, status, preserved in [
                ('Community Edition', .94, 'accepted', True), ('Community', .94, 'accepted', False),
                ('Community Edition', .84, 'review', False), ('Community 版', .94, 'accepted', False)]:
            with self.subTest(source=source, confidence=confidence):
                reading = replace(base.selected, origin='raw_main_reading', text=source,
                    evidence_json=canonical([dict(confidence=confidence, status=status, text=source)]))
                unit = replace(base, readings=(reading,), selected_revision=reading.revision)
                request = TranslationRequest((unit,), (unit.unit_id,))
                provider = Mock()
                provider.generate.return_value = 'Community Edition'
                result = PreviewTranslator(provider, model='translategemma:27b', profile=TRANSLATEGEMMA,
                    single_rendition=True, page_context=True, terminology_units=(unit,)).translate(request)
                self.assertEqual(result.status == 'abstained', preserved)
                self.assertEqual(provider.generate.called, not preserved)

    def test_short_reply_uses_only_a_nearby_offer_on_the_same_page(self):
        from types import SimpleNamespace
        def unit(i, text, page='a'):
            return SimpleNamespace(unit_id=i, page_json=page, selected=SimpleNamespace(text=text))
        reply=unit(2,'ここで待っててね')
        offer=unit(1,'何をすればいい？')
        self.assertIn('待っていてください',rendition.reply_to_offer(reply.selected.text,2,[offer,reply]))
        self.assertFalse(rendition.reply_to_offer('僕は待ってて…',2,[offer,reply]))
        self.assertFalse(rendition.reply_to_offer(reply.selected.text,2,[unit(1,offer.selected.text,'b'),reply]))
        self.assertFalse(rendition.reply_to_offer(reply.selected.text,2,[offer,*[unit(i,'そう') for i in range(4,8)],reply]))

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
        self.assertTrue(rendition.expanded_short_label('ルビー', 'Ruby. The dog is probably hiding under the table.'))
        self.assertFalse(rendition.expanded_short_label('ルビー', 'Ruby!'))
        self.assertFalse(rendition.expanded_short_label('犬はテーブルの下に隠れてる', 'The dog is probably hiding under the table.'))
        self.assertIn('Science Weekly',rendition.usage_context('初出：rScience WeeklyJ月刊誌'))
        self.assertFalse(rendition.usage_context('関数rScienceWeeklyJを呼ぶ'))
        self.assertFalse(rendition.usage_context('ぽとっ'))
        self.assertTrue(rendition.usage_context('はらはら'))
        self.assertFalse(rendition.usage_context('花びらがはらはらと散った。'))
        self.assertTrue(rendition.usage_context('おすこやかで'))
        self.assertTrue(rendition.positive_reaction('あぁ…嬉しいね'))
        self.assertTrue(rendition.positive_reaction('うう...うれしいよ'))
        self.assertFalse(rendition.positive_reaction('うう…幸せじゃない'))
        self.assertFalse(rendition.positive_reaction('うぅ…おいしいけど高い'))
        self.assertIn('authority',rendition.usage_context('権力に影響を与えます。'))
        self.assertIn('退勤',rendition.usage_context('退勤までには済む。'))
        self.assertIn('三丁目',rendition.usage_context('三丁目の家です。'))
        self.assertIn('財布',rendition.usage_context('財布も盗まれた。'))
        self.assertTrue(rendition.extra_circumstance('うっかり寝ちゃった！','I fell asleep with the light on.'))
        self.assertFalse(rendition.extra_circumstance('窓を開けたまま寝ちゃった','I fell asleep with the window open.'))
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
                ('ピーッ', 'Sproing!', rendition.VISUAL_TONE), ('ピーッ', 'Whistle!', None),
                ('ドン', 'Thud!', None), ('彼の腕を引く', 'Pull his arm.', None),
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

    def test_part_owner_requires_a_separately_printed_object(self):
        from dataclasses import replace
        from types import SimpleNamespace
        unit = fixture()[2][0]
        reading = replace(unit.selected, text='腕を直せば…')
        unit = replace(unit, readings=(reading,), selected_revision=reading.revision)
        other = SimpleNamespace(unit_id='other', page_json=unit.page_json, selected=SimpleNamespace(text='ロボットが動く'))
        request = TranslationRequest((unit,), (unit.unit_id,))
        for owner, count in [('ロボット',3),('時計',2)]:
            runtime = Mock()
            runtime.generate.side_effect = ['If I fix my arm...', json.dumps({
                'owner_type':'inanimate','owner_noun_ja':owner,'observation':'The arm belongs to a machine.'}),
                "If I fix the robot's arm..."]
            translator = PreviewTranslator(runtime,model='translategemma:27b',profile=TRANSLATEGEMMA,
                single_rendition=True,page_context=True,terminology_units=(unit,other),
                context_crop=lambda unit,**options:(b'panel',[0,0,32,64],''))
            result = translator.translate(request)
            self.assertEqual(runtime.generate.call_count,count)
            self.assertEqual('robot' in result.translations[0][1],count==3)
            if count == 3:
                self.assertTrue(runtime.generate.call_args.kwargs['payload']['messages'][0]['content'].endswith('ロボットの腕を直せば…'))
            result.assert_current(request)

    def test_smaller_name_below_title_survives_as_a_separate_line(self):
        from dataclasses import replace
        from tralens.translation.contracts import canonical
        unit = fixture()[2][0]
        names = [('新しい物語',[[2,2],[93,2],[93,25],[2,25]]),
                 ('田中はな',[[3,26],[56,26],[56,39],[3,39]])]
        reading = replace(unit.selected,origin='raw_main_reading',text=''.join(name for name,_ in names),
            evidence_json=canonical([dict(text=name,polygon=polygon) for name,polygon in names]))
        unit = replace(unit,readings=(reading,),selected_revision=reading.revision)
        request = TranslationRequest((unit,),(unit.unit_id,))
        provider=Mock()
        provider.generate.return_value='A New Story'
        with patch.object(rendition,'name_label',side_effect=lambda line,**kw:('Tanaka','Hana') if kw.get('byline') else None):
            result=PreviewTranslator(provider,model='translategemma:27b',profile=TRANSLATEGEMMA,
                single_rendition=True,page_context=True,terminology_units=(unit,)).translate(request)
        self.assertEqual(result.translations[0][1],'A New Story\nHana Tanaka')
        message=provider.generate.call_args.kwargs['payload']['messages'][0]['content']
        self.assertTrue(message.endswith('新しい物語'))
        self.assertNotIn('田中はな',message)
        result.assert_current(request)

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
