"""One bounded specialist retry for explicit alternate-rendition output.

Never trim a renderer's target or split a translation and pick its first phrase.
The original response survives as evidence and remains the fallback on failure.
"""
import re


POLICY='single_rendition_retry_v1'
INSTRUCTION=('Give one natural translation suitable for dialogue, not a list of alternative phrasings. '
    'Preserve repetitions, hesitations, punctuation, and unfinished sentences. '
    'Do not add explanations, alternatives in parentheses, or words absent from the source. ')


def needs_retry(source, target):
    # Literal slashes in the source (commands, paths, alternatives) are legitimate.
    # A single isolated symbol supplies too little context for this second pass.
    return (len(source.strip())>=2 and '/' not in source and '／' not in source
        and 2<=len(target.split(' / '))<=5
        and all(re.search(r'[A-Za-z]',part) for part in target.split(' / ')))


UNFINISHED = 'missing_final_contrast_v1'
CONTRAST = re.compile(r"\b(?:but|however|though|although|yet|nevertheless|except)\b", re.I)
UNFINISHED_INSTRUCTION = ('This source stops mid-phrase. Translate only the visible words as an unfinished '
    'English phrase ending in an ellipsis. Do not supply a missing verb or object. ')


WAIT = 'missing_request_to_wait_v1'
WAIT_SOURCE = re.compile(r'の[をは](?:少し|しばらく)?お待ちください')
WAIT_TARGET = re.compile(r"\b(?:wait|hold|delay|postpone|refrain|before|until|yet)\b", re.I)
WAIT_INSTRUCTION = ('Preserve the request to wait, including exactly what the listener '
    'is asked to wait for or delay. ')
GLOSS = 'single_caption_gloss_v1'
GLOSS_LIST = re.compile(r'[a-z]+(?:, [a-z]+){2,4}')
GLOSS_INSTRUCTION = ('Give one natural English expression for this caption. Use a single rendition that can '
    'be printed on the page. Preserve any enumeration or repetition in the Japanese. ')
SUBJECT = 'unfinished_technical_subject_v1'
TECH_TOKEN = re.compile(r'[A-Za-z][A-Za-z0-9]*|[0-9]+(?:\.[0-9]+)*|×')
TECH_SUBJECT = re.compile(f'({TECH_TOKEN.pattern})が$')
SUBJECT_INSTRUCTION = ('The visible lettering ends at a Japanese subject marker. Give only the incomplete '
    'noun phrase in English, with its modifiers and trailing ellipsis. Preserve version numbers and symbols. ')
COMMAND = 'untranslated_command_verb_v1'
LOAN = 'technical_loanword_v1'
GREETING = re.compile(r"(?:welcome(?: back)?|hello|hi|good(?: morning| evening| night))\s*[.!…]*",re.I)
TECHNICAL = re.compile(r'(?:不要|必須|依存|既存|旧式|互換|非対応|標準)(?:な|の)?([ァ-ヺー]{3,})')
TICKET_GATE = 'station_ticket_gate_label_v1'
GATE_MODIFIERS = {'中央':'central', '東':'east', '西':'west', '南':'south', '北':'north'}
BANNER_TERM = 'corroborated_banner_loanword_v1'


VISUAL_PERMISSION = 'unauthorized_action_as_command'
VISUAL_SOUND = 'ambiguous_comic_sound'
PERMISSION_COMMAND = re.compile(r"^(?:please\s+)?(post|upload|publish|share|send|delete|use)\b", re.I)
BIRD_SOUND = re.compile(r'ピ[ーイ]?チク|パ[ーア]?チク|ポッポ')
VISUAL_ACTOR = 'unstated_actor_in_question'
URGENT = 'colloquial_urgency'
BAR_CALL = 'bartender_address'
REREAD = 'short_ocr_reread'
CLICK = 'katakana_click_lookalike'
URGENCY = re.compile(r"\b(?:asap|hurry|quick\w*|fast|soon|speed|pronto|on the double|make it snappy)\b",re.I)


def commentary_output(source, english):
    return (len(source.strip()) <= 12 and len(english) > 100 and
            re.search(r"(?:without more context|fragmented sentence|possible translations|a precise translation)",english,re.I))


def short_reading(unit):
    from .contracts import strict_json
    if unit.selected.origin != 'raw_main_reading':
        return None
    evidence = strict_json(unit.selected.evidence_json)
    if not isinstance(evidence,list) or len(evidence) not in (1,2):
        return None
    metadata = [strict_json(row.get('metadata_json','{}')) for row in evidence]
    if len(evidence)==2 and metadata[1].get('short_column_anchor')!=evidence[0]['source_id']:
        return None
    parts = [m.get('corroborated_short_reading') for m in metadata]
    if not all(isinstance(part,str) and re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿]{1,6}',part) for part in parts):
        return None
    return ''.join(parts)


VISUAL_POLICIES = (VISUAL_PERMISSION, VISUAL_SOUND, VISUAL_ACTOR, CLICK)


def visual_guide(policy, source, punctuation):
    if policy == CLICK:
        return ('These glyphs can be OCR lookalikes for katakana カチ. Read the printed shapes. '
                'If this is the sound カチ, use “Click”. Translate only the marked lettering. ')
    if policy == VISUAL_PERMISSION:
        guide = 'Read the scene to distinguish an instruction from a reaction to an unauthorized act. '
        if not punctuation and source.rstrip().endswith(('!?','？！','！？','?!')):
            return ('The Japanese ends in a startled reaction. Keep the action and the surprised punctuation. '
                    'Use a subjectless English fragment, e.g. “Posted it without asking…!?” for posting. '
                    'Avoid guessing a speaker or the person responsible. ')
        if punctuation:
            guide = (f'A separate adjacent punctuation column reads {punctuation}. Treat it as tone only. '
                     'Use a subjectless startled fragment in English, e.g. “Posted it without asking…!?” '
                     'if the action is posting. Do not convert this reaction into a command or add a personal pronoun. ')
        return guide
    if policy == VISUAL_ACTOR:
        return ('Preserve the accidental movement and the causal question. For のっちゃったから？ use '
                '“Because someone got on ...?” when the person is unidentified. Translate the object '
                'of に. Do not invent a thing that someone put there. ')
    if source.strip(' 。！!?？…') == 'トン':
        return ('Japanese lettering convention: トン denotes a single light tap on contact, heavier ドン a thud. '
                'Match the visible kana and action; write “Tap” if the depicted contact is light. ')
    if 'チク' in source:
        return ('This is bird-call lettering. チク in ピーチク or パーチク represents chirping. Render the call '
                'itself as English sound lettering such as “Chirp!” or “Chirrup!”; do not spell out the Japanese syllables. ')
    return ('If this is a pigeon or dove, ポー or ポッポ represents a coo. Use the natural English bird '
            'call, e.g. “Coo!” for the long note. ')


def visual_retry(source, primary):
    if source.strip() in ('力千','力チ','カ千') and primary.strip(' .!?').lower() in ('rikisen','power','strength','thousand'):
        return CLICK
    if re.search(r'(?:勝手に|無断で).+(?:して|って)[。！!?？…]*$', source) and (PERMISSION_COMMAND.match(primary.strip())
            or source.rstrip().endswith(('!?','？！','！？','?!')) and re.match(r'did you ',primary.strip(),re.I)):
        return VISUAL_PERMISSION
    if source.strip(' 。！!?？…') == 'トン' and re.fullmatch(r'(?:a )?(?:ton|tonne|tunnel|click)[.!]?', primary.strip(), re.I):
        return VISUAL_SOUND
    if re.search(r'(?:のっ|乗っ)ちゃったから[?？]$',source) and re.search(r'\b(?:you|I|he|she|we)\b', primary, re.I):
        return VISUAL_ACTOR
    if source.strip(' 。！!?？…') == 'ポッポ' and primary.strip(' .!?').lower() == 'chirp':
        return VISUAL_SOUND
    if (BIRD_SOUND.search(source) and re.search(r'(?:p[aeiou]+[a-z]*ch[ia]k|poppo)', primary, re.I)
            or source.strip(' 。！!?？…') == 'ポー' and primary.strip(' .!?').lower() in ('pooh', 'po', 'poo')):
        return VISUAL_SOUND
    return None


def banner_term(unit, units):
    """A missing kana in a title may be legible in another line on the same page."""
    from .contracts import strict_json
    if 'punctuated_horizontal_heading' not in unit.structural_reasons:
        return None
    terms = re.findall(r'[ァ-ヺー]{4,24}', unit.selected.text)
    matches = {}
    for other in units:
        if (other.page_json != unit.page_json or set(unit.member_ids).intersection(other.member_ids)
                or other.selected.origin != 'raw_main_reading'):
            continue
        lines = strict_json(other.selected.evidence_json)
        if not isinstance(lines,list):
            continue
        for line in lines:
            if not isinstance(line,dict) or not isinstance(line.get('text'),str) or not isinstance(line.get('confidence'),(int,float)):
                continue
            word = line['text'].strip()
            if (line['confidence'] < .7 or not re.fullmatch(r'[ァ-ヺー]{5,25}', word)
                    or not any(len(word) == len(term)+1 and word[0] == term[0] and word[-1] == term[-1]
                               and any(word[:i]+word[i+1:] == term for i in range(1, len(word)-1))
                               for term in terms)):
                continue
            matches[word] = {'word':word, 'unit_id':other.unit_id, 'reading_revision':other.selected_revision}
    if len(matches) != 1:
        return None
    word, proof = next(iter(matches.items()))
    try:
        from fugashi import Tagger
        import unidic_lite
        entries = list(Tagger(f'-d "{unidic_lite.DICDIR}"')(word))
    except (ImportError, RuntimeError):
        return None
    if len(entries) == 1:
        spelling, _, english = (entries[0].feature.lemma or '').partition('-')
        if spelling == word and re.fullmatch(r'[A-Za-z]{3,24}', english):
            return {**proof, 'english':english}
    return None


BORROWED = 'neighbor_name_in_short_line'
SOUND_LETTERING = 'emphatic_sound_as_stage_direction'


def borrowed_name(source, target, nearby):
    # Restrict this check to short Japanese lines without a possible katakana name.
    if not re.fullmatch(r'[ぁ-ゖ一-鿿々]{2,8}[。！？…]*', source):
        return False
    names = {word.casefold() for line in nearby for word in re.findall(r'[A-Za-z]{4,}', line)}
    return bool(names.intersection(re.findall(r'[a-z]+', target.casefold())))


SELECTION = {SOUND_LETTERING:'sound_lettering', BORROWED:'isolated_target', URGENT:"urgency", BAR_CALL:"bar_address", REREAD:"consistent_local_reread", **{key:'visual_context' for key in VISUAL_POLICIES}, TICKET_GATE:'station_label', BANNER_TERM:'page_title_term', GLOSS:'single_caption_gloss', SUBJECT:'unfinished_subject', COMMAND:'command_verb', LOAN:'technical_loanword', POLICY:'single_rendition',
             UNFINISHED:'unfinished_contrast', WAIT:'request_to_wait'}


def lexical_hint(source, policy):
    """Read UniDic's Japanese verb lemma or loanword entry for the actual source."""
    try:
        from fugashi import Tagger
        import unidic_lite
        words = list(Tagger(f'-d "{unidic_lite.DICDIR}"')(source))
    except (ImportError,RuntimeError):
        return None
    if policy==COMMAND:
        for word,ending in zip(words,words[1:]):
            if word.feature.pos1=='動詞' and ending.surface=='なさい':
                return word.feature.lemma
    if policy==LOAN:
        modifiers = set(TECHNICAL.findall(source))
        nouns = []
        for word in words:
            lemma = word.feature.lemma or ''
            spelling,sep,loan = lemma.partition('-')
            if (word.feature.pos1=='名詞' and word.feature.goshu=='外' and sep and spelling==word.surface and word.surface in modifiers
                    and re.fullmatch('[a-zA-Z]{3,24}',loan)):
                nouns.append(loan)
        if len(nouns)==1:
            return nouns[0]
    return None


def technical_context(target, units):
    """Require independent page evidence before choosing the computing sense."""
    terms = TECHNICAL.findall(target.selected.text)
    for other in units:
        if other.page_json!=target.page_json or other.unit_id==target.unit_id:
            continue
        text = other.selected.text
        for term in terms:
            if term in text and (re.search(r'インストール|ソフトウェア|コンピュータ|Linux|Ubuntu',text,re.I)
                    or term+'管理' in text):
                return {'unit_id':other.unit_id,'reading_revision':other.selected_revision,'term':term}
    return None


def contains_loanword(target, term):
    # Compound spelling and inflection can change without losing the term.
    undashed = re.sub(r'(?<=[A-Za-z])[-‐‑–](?=[A-Za-z])','',target)
    return bool(term and re.search(r'\b'+re.escape(term)+r'(?:s|es|ed|ing|d)?\b',undashed,re.I))


def retry_policy(source, target, style):
    if style.source_language in ('Japanese','ja') and style.target_language in ('English','en'):
        if (re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,8}[っッ][！!]+', source)
                and re.fullmatch(r'\*{1,2}[A-Za-z][A-Za-z !.-]*\*{1,2}', target.strip())):
            return SOUND_LETTERING
        if 'なるはや' in source and not URGENCY.search(target):
            return URGENT
        bare = source.strip().rstrip('。！!…')
        if (bare.endswith('改札口') and bare[:-3] in GATE_MODIFIERS
                and not re.search(r'\b(ticket|fare) gates?\b', target, re.I)):
            return TICKET_GATE
        if (bare.endswith('なさい') and bare not in ('おかえりなさい','お帰りなさい','おやすみなさい','お休みなさい')
                and GREETING.fullmatch(target.strip()) and lexical_hint(source,COMMAND)):
            return COMMAND
        if re.fullmatch(r'[ぁ-ゖァ-ヺー]{2,8}', source.strip()) and GLOSS_LIST.fullmatch(target.strip()):
            terms = target.strip().split(', ')
            if len(terms) == len(set(terms)):
                return GLOSS
        if TECH_SUBJECT.search(source.strip()) and re.search(r'(?<!\.)[.!]$', target.strip()):
            return SUBJECT
        if TECHNICAL.search(source):
            term = lexical_hint(source,LOAN)
            if term and not contains_loanword(target,term):
                return LOAN
    if needs_retry(source,target):
        return POLICY
    if (style.source_language in ('Japanese','ja') and style.target_language in ('English','en')
            and WAIT_SOURCE.search(source) and not WAIT_TARGET.search(target)):
        return WAIT
    if (style.source_language in ('Japanese','ja') and style.target_language in ('English','en')
            and len(source.strip()) >= 5 and source.rstrip().endswith(('だが','ですが','けれど','けれども'))
            and target.strip().endswith('.') and not target.strip().endswith('...')
            and not CONTRAST.search(target)):
        return UNFINISHED
    return None


def candidate_matches(policy, primary, candidate, source='', *, term=None):
    if ' / ' in candidate:
        return False
    if policy == SOUND_LETTERING:
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*(?: [A-Za-z'-]+)?[!….]*", printed_cadence(source, candidate)))
    if policy == BORROWED:
        return bool(re.search(r'[A-Za-z]', candidate) and len(candidate.split()) < len(primary.split())
                    and not re.search(r'[ぁ-ヺ一-鿿\n*]', candidate))
    if policy == URGENT:
        return bool(URGENCY.search(candidate))
    if policy == BAR_CALL:
        return bool(re.search(r"\b(?:bar(?:tender|keep)|publican)\b",candidate,re.I))
    if policy == REREAD:
        return (0 < len(candidate.strip()) < 80 and not re.search(r'[ぁ-ヺ一-龯]|[\n*]',candidate)
            and not commentary_output(source,candidate))
    if policy in VISUAL_POLICIES:
        if visual_retry(source, candidate) or re.search(r'[ぁ-ヺ一-龯]', candidate):
            return False
        if policy == VISUAL_PERMISSION:
            verb = PERMISSION_COMMAND.match(re.sub(r"^did you\s+", "", primary.strip(), flags=re.I))
            if not verb:
                return False
            stem = verb[1].lower().removesuffix('e')
            action = r'sen(?:d|t)' if stem == 'send' else re.escape(stem) + r'(?:e|ed|d|ing|s)?'
            return (bool(re.search(r'\b'+action+r'\b', candidate, re.I))
                and bool(re.search(r"without|permission|didn.t ask|never asked|behind .* back", candidate, re.I))
                and not re.search(r"\b(?:he|she|you)\b", candidate, re.I))
        if policy == VISUAL_SOUND and source.strip(' 。！!?？…') == 'トン':
            return bool(re.fullmatch(r'(?:tap|pat|thump|thud|knock|tock)[.!…]*', candidate.strip(), re.I))
        if policy == VISUAL_ACTOR:
            return (candidate.endswith('?') and not re.search(r"\b(?:you|I|he|she|we|put|place)\b", candidate, re.I)
                and ('キーボード' not in source or 'keyboard' in candidate.lower()))
        return bool(re.fullmatch(r"[A-Za-z!-]+(?:[ .,!-]+[A-Za-z!-]+)*[.!…]*", candidate.strip())
            and len(candidate.split()) <= max(5, len(source)))
    if policy == TICKET_GATE:
        modifier = GATE_MODIFIERS.get(source.strip().rstrip('。！!…')[:-3])
        return bool(modifier and re.fullmatch(modifier+r' (ticket|fare) gates?[.!]?',candidate.strip(),re.I))
    if policy == BANNER_TERM:
        if source.startswith('ようこそ') and not re.search(r'\bwelcome\b', candidate, re.I):
            return False
        return contains_loanword(candidate, term) and re.findall(r'\d+', primary) == re.findall(r'\d+', candidate)
    if policy == GLOSS:
        return candidate.strip().rstrip('.').casefold() in primary.strip().split(', ')
    if policy == SUBJECT:
        subject = TECH_SUBJECT.search(source.strip())
        tokens = {word.casefold() for word in TECH_TOKEN.findall(source)}
        words = {word.casefold() for word in TECH_TOKEN.findall(candidate)}
        numbers = r'[0-9]+(?:\.[0-9]+)*'
        return bool(subject and tokens <= words
            and re.findall(numbers, source) == re.findall(numbers, candidate)
            and primary.rstrip(' .!…') != candidate.rstrip(' .!…')
            and (candidate.rstrip().endswith(('...', '…')) or candidate.rstrip().endswith(subject[1])))
    if policy in (LOAN,COMMAND):
        if re.findall(r'\d+',primary)!=re.findall(r'\d+',candidate):
            return False
        if policy==COMMAND:
            return not GREETING.fullmatch(candidate.strip())
        term = lexical_hint(source,LOAN)
        return contains_loanword(candidate,term)
    if policy == WAIT:
        return (bool(WAIT_TARGET.search(candidate))
            and re.findall(r'\d+', primary) == re.findall(r'\d+', candidate))
    if policy != UNFINISHED:
        return True
    # Only take a response that actually restores an open connective and
    # preserves any numeric information from the primary translation.
    return (bool(re.search(r'\b(?:but|though|however|yet)\s*(?:…|\.{3})$',candidate.strip(),re.I))
        and re.findall(r'\d+(?:[.,]\d+)*',primary) == re.findall(r'\d+(?:[.,]\d+)*',candidate))


def payload(original, policy=POLICY, source='', *, term=None, punctuation=''):
    from copy import deepcopy
    result=deepcopy(original)
    messages=result['messages']
    if len(messages)!=1 or messages[0]['role']!='user':
        raise ValueError('single-rendition retry requires one native specialist user message')
    text=messages[0]['content']
    marker='Please translate the following '
    if marker not in text:raise ValueError('unknown specialist prompt structure')
    if policy == BORROWED:
        messages[0].pop('images', None)
        messages[0]['content'] = INSTRUCTION + marker + text.rsplit(marker, 1)[1]
        return result
    if policy == SOUND_LETTERING:
        # Retain the attached scene, while isolating the effect from unrelated dialogue.
        text = marker + text.rsplit(marker, 1)[1]
        guide = ('Letter the audible sound itself as one brief English comic sound effect. '
                 'Use the depicted action. Do not describe it or give a command. ')
    elif policy in VISUAL_POLICIES:
        guide = visual_guide(policy, source, punctuation) + 'Only translate the supplied text. Other lettering is context only. '
    elif policy == REREAD:
        if not term:
            raise ValueError('missing corroborated OCR alternate')
        messages[0]['content'] = text.removesuffix(source) + term
        return result
    elif policy == URGENT:
        guide = ('なるはや is conversational shorthand for なるべく早く, as soon as possible. '
                 'Preserve the request for speed in a concise spoken line. ')
    elif policy == BAR_CALL:
        guide = ('A separate line on this page orders beer. The speaker is calling to the person '
                 'serving the drinks. Use natural English for that form of address. ')
    elif policy == TICKET_GATE:
        guide = ('Translate the whole railway sign as a short English label. 改札口 means ticket gate. '
                 'Keep its location or direction modifier. Use sentence case. ')
    elif policy == BANNER_TERM:
        guide = (f'Translate both lines as one natural banner. The same page independently prints the '
                 f'loanword {term["word"]} ({term["english"]}). Use this spelling for the title. ')
        if source.startswith('ようこそ'):
            guide += 'Use “Welcome to” followed directly by the event name, with punctuation at the end. '
    elif policy in (COMMAND,LOAN):
        hint = lexical_hint(source,policy)
        if not hint:
            raise ValueError('missing source dictionary entry')
        if policy==COMMAND:
            guide = f'The inflected verb before なさい has the Japanese dictionary form {hint}. Translate its action as one natural spoken command. Return only the dialogue, with no alternatives or explanation. '
        else:
            guide = (f'Preserve the computing term as “{hint}” in the English dialogue, '
                'including when the speaker uses it as a joke about a person. ')
    else:
        guide = {GLOSS:GLOSS_INSTRUCTION,SUBJECT:SUBJECT_INSTRUCTION,UNFINISHED:UNFINISHED_INSTRUCTION,WAIT:WAIT_INSTRUCTION,POLICY:INSTRUCTION}[policy]
    messages[0]['content']=text.replace(marker,guide+marker,1)
    return result


def usage_context(source):
    """Supply ambiguous senses or grammatical cues; the translator chooses the English."""
    if '名折れ' in source:
        action = '帰す means letting someone leave. ' if re.search(r'帰し|帰す', source) else ''
        return (action + '名折れ is a loss of personal professional honor. Name the speaker’s own profession; '
                'use direct address if the other speaker is present. Do not introduce an organization.')
    cues = (
        (r'もう[^、。！？]{1,16}かも',
         'Keep each emotional beat in its own sentence. The time adverb and uncertainty '
         'belong to the same clause.'),
        (r'困窮',
         '困窮 specifically describes financial hardship; make the financial aspect explicit.'),
        (r'^(?:お)?(?:すこやか|健やか)で[。…]*$',
         'This is a farewell like a conventional expression said to someone departing; avoid '
         'literal references to health. Use at most three words.'),
        (r'^はらはら[っッ]?[。…]*$',
         'When a character is crying, this depicts falling tears. Use a compact visual caption '
         'such as “tears fall”.'),
        (r'ー+っ?と待った',
         'Transliterate the prolonged vowel into the English delivery. Spell the stressed word '
         'with three vowels (e.g. Hoooold, Waaait). End with an exclamation mark.'),
        (r'人ではな',
         'Use the artwork to count the people described. For a single individual maintain a '
         'singular predicate even when using gender-neutral they.'),
        (r'信じられ.*(?:この|あの|その).+で$',
         'The Japanese stops mid-sentence after a place. Break after the disbelief, then end on '
         'the place name with an ellipsis. Use no indefinite article before the place name.'),
        (r'お目にかけ', 'お目にかける humbly offers to SHOW something TO the listener.'),
        (r'^(?:ゆら|ユラ)[っッ]$',
         'For an isolated motion caption use a short action verb, without punctuation or an '
         'adjective.'),
    )
    return next((note for pattern, note in cues if re.search(pattern, source)), '')


def printed_cadence(source, target):
    """Keep a trailing thought open and avoid duplicating a single uncertainty marker."""
    if not target or not target.strip():
        return target
    target = target.strip()
    emphasis = re.fullmatch(r'(\*{1,2})([^*\n]+)\1', target)
    if emphasis and '*' not in source and '＊' not in source and re.search(r'[A-Za-z]', emphasis[2]):
        target = emphasis[2].strip()
    if source.rstrip().endswith(('…','...')):
        target = re.sub(r'[.。…]+$', '',target).rstrip() + '…' if target[-1] not in '?!' else target
    clauses = re.split(r'かも[、,]',source)
    if (len(clauses)==2 and re.search(r'た[。…]*$',clauses[1])
            and not re.search(r'かも|たぶん|多分|おそらく|だろう|でしょう|らしい|よう|そう|思',clauses[1])):
        # Require the first English clause to already express the uncertainty.
        target = re.sub(r'(\b(?:might|may|maybe)\b[^.!?…;]*[.!…;]+\s+[^.!?…;]*?)\bprobably\s+',
                        r'\1',target,flags=re.I)
    if '無料OS' in source:
        target = re.sub(r'\bof free OS(?=[.!?…]*$)', 'of a free OS', target)
    return target
