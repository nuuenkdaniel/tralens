"""One bounded specialist retry for explicit alternate-rendition output.

Never trim a renderer's target or split a translation and pick its first phrase.
The original response survives as evidence and remains the fallback on failure.
"""
import re
from functools import lru_cache

from ..ocr.types import menu_shortcuts


POLICY='single_rendition_retry_v1'
INSTRUCTION=('Give one natural translation suitable for dialogue, not a list of alternative phrasings. '
    'Preserve repetitions, hesitations, punctuation, and unfinished sentences. '
    'Do not add explanations, alternatives in parentheses, or words absent from the source. ')


def interface_acronym(source,nearby):
    # A vertical Latin I resembles a Japanese long-vowel mark. Require the
    # independently printed interface expansion before interpreting that mark.
    if not re.search(r'(?<![A-Za-z])[CG]U[ー－](?![A-Za-z])',source):
        return ''
    context = ' '.join((source,*nearby))
    english = re.sub(r'[^a-z]','',context.lower())
    if 'ユーザーインターフェース' not in context and 'userinterface' not in english:
        return ''
    return re.sub(r'(?<![A-Za-z])([CG]U)[ー－](?![A-Za-z])',r'\1I',source)


def disputed_toolbar_label(unit, nearby):
    """A tiny label in a file browser can be legible from its icon when OCR disagrees."""
    from .contracts import strict_json
    from ..ocr.types import is_printed_literal
    source = unit.selected.text
    if (len(unit.member_ids)!=1 or not re.fullmatch(r'[一-鿿]{2,4}',source)
            or len({line for line in nearby if is_printed_literal(line) and '.' in line})<2):
        return False
    raw = next((r for r in unit.readings if r.origin=='raw_main_reading'),None)
    if (not raw or raw.text==source or len(raw.text)!=len(source)
            or sum(a!=b for a,b in zip(raw.text,source))!=1):
        return False
    lines = strict_json(raw.evidence_json)
    if len(lines)!=1 or lines[0].get('confidence',1)>=.8 or not lines[0].get('polygon'):
        return False
    xs,ys = zip(*lines[0]['polygon'])
    width,height = max(xs)-min(xs),max(ys)-min(ys)
    return 6<=height<=32 and height<=width<=84


def needs_retry(source, target):
    # Literal slashes in the source (commands, paths, alternatives) are legitimate.
    # A single isolated symbol supplies too little context for this second pass.
    return (len(source.strip())>=2 and '/' not in source and '／' not in source
        and 2<=len(target.split(' / '))<=5
        and all(re.search(r'[A-Za-z]',part) for part in target.split(' / ')))


UNFINISHED = 'missing_final_contrast_v1'
CONTRAST = re.compile(r"\b(?:but|however|though|although|yet|nevertheless|except)\b", re.I)
UNFINISHED_INSTRUCTION = ('The Japanese ends with a contrastive connective and leaves the next clause unspoken. '
    'Keep the completed statement, followed by its contrast word and an ellipsis. Only letter those words. ')


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
VISUAL_OWNER = 'unmarked_part_owner'
VISUAL_TONE = 'sound_source_mismatch'
VISUAL_REACTION = 'spoken_reaction_mismatch'
UNMARKED_PART = re.compile(r'(?<![一-鿿の])(?:目|眼|耳|口|手|足|腕|顔|首|頭|背中|羽|尾|翼)を')
URGENT = 'colloquial_urgency'
BAR_CALL = 'bartender_address'
REREAD = 'short_ocr_reread'
CLICK = 'katakana_click_lookalike'
URGENCY = re.compile(r"\b(?:asap|hurry|quick\w*|fast|soon|speed|pronto|on the double|make it snappy)\b",re.I)


def commentary_output(source, english):
    return (len(source.strip()) <= 12 and len(english) > 100 and
            re.search(r"(?:without more context|fragmented sentence|possible translations|a precise translation)",english,re.I))


def quoted_response_body(source, english):
    """Extract a single quoted answer after an explicit model preamble."""
    if re.search(r'英訳|翻訳|訳文|translation',source,re.I):
        return english
    heading,separator,body=english.strip().partition('\n\n')
    if (not separator or not re.fullmatch(
            r"(?:okay[,!]?\s+|sure[,!]?\s+)?here(?:['’]s| is) the (?:English )?translation\b[^\n]{0,160}:",
            heading,re.I)):
        return english
    body=body.strip()
    if body.startswith(('**','__')) and body.endswith(body[:2]):
        body=body[2:-2].strip()
    if len(body)>2 and (body[0],body[-1]) in (('"','"'),('“','”')):
        return body[1:-1].strip()
    return english


def effect_field(response):
    """Only the printable effect from a structured visual answer enters the page."""
    from .contracts import strict_json
    if not isinstance(response,str):
        raise ValueError('invalid comic effect response')
    data=strict_json(response)
    word=data.get('effect') if isinstance(data,dict) and set(data)=={'effect'} else None
    if not isinstance(word,str) or not word.strip() or not word.isprintable() or len(word)>32 or len(word.split())>4:
        raise ValueError('invalid short comic effect')
    return word.strip()


def reaction_beat(response, part):
    """Keep the word that belongs to this panel of a split exclamation."""
    from .contracts import strict_json
    pair=strict_json(response)
    if not isinstance(pair,dict) or set(pair)!={'first','second'} or not all(
            isinstance(s,str) and re.fullmatch(r"[A-Za-z][A-Za-z'’!?.,…–—-]{0,23}",s.strip()) for s in pair.values()):
        raise ValueError('invalid two-beat reaction')
    return pair[('first','second')[part]].strip()


def interrupted_word(response, source):
    """Accept a short English fragment only when the proposed word starts with the visible kanji."""
    from .contracts import strict_json
    data=strict_json(response)
    prefix=re.match(r'[一-鿿]+',source)
    if (not isinstance(data,dict) or set(data)!={'japanese_completion','english'}
            or not isinstance(data['japanese_completion'],str)
            or (prefix and not data['japanese_completion'].startswith(prefix[0]))):
        raise ValueError('invalid interrupted word')
    text=data['english']
    if not isinstance(text,str) or not re.fullmatch(r"[A-Za-z][A-Za-z’' -]{0,27}(?:\.{1,3}|…)?",text.strip()) or len(text.split())>4:
        raise ValueError('invalid interrupted lettering')
    word=text.strip().rstrip('.… ')
    return word[0].upper()+word[1:]+'…'


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


PRINTED_SPELLING = "printed_latin_spelling"
# A cross inside or directly after a Latin name can be an OCR substitution.
# Exclude numeric dimensions and product names on either side of a crossover.
SPELLING_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{3,}×[a-z]*(?![A-Za-z0-9×])")

VISUAL_POLICIES = (PRINTED_SPELLING, VISUAL_PERMISSION, VISUAL_SOUND, VISUAL_ACTOR, VISUAL_OWNER, VISUAL_TONE, VISUAL_REACTION, CLICK)


def visual_guide(policy, source, punctuation):
    if policy == VISUAL_TONE:
        return ('Identify what makes this sound in the panel. Distinguish an electronic signal, '
                'a call or a whistle from an impact. Letter the actual sound concisely. ')
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
    token = SPELLING_TOKEN.search(source)
    if token and token[0] in primary and len(SPELLING_TOKEN.findall(source))==1:
        return PRINTED_SPELLING
    if positive_reaction(source) and re.match(r'^\W*(?:ugh|yuck|eww?)\b',primary,re.I):
        return VISUAL_REACTION
    if UNMARKED_PART.search(source) and re.search(r'\b(?:my|your|his|her|our|their)\s+(?:eyes?|ears?|mouth|hands?|feet|foot|arms?|face|neck|head|back|wings?|tail)\b',primary,re.I):
        return VISUAL_OWNER
    if (re.fullmatch(r'[ピプビ][ーィイッピプビ]{1,7}[!！…]*',source.strip())
            and re.fullmatch(r'\W*(?:boing|sproing|thud|splat|crash|squeak)\W*',primary,re.I)):
        return VISUAL_TONE
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
NOUN_LABEL = 'common_noun_label'
SINGLE_ADJECTIVE = re.compile(r'[A-Za-z]+(?:able|ous|ful|less|ive)[.!]?', re.I)
SOUND_LETTERING = 'emphatic_sound_as_stage_direction'
SHRIEK = re.compile(r'[きキ][ゃャ][ぁァあアー]+[っッ]?[！!]*')


def detached_connective(source):
    return source.strip(' 。!！?？…') in ('て', 'で', 'って')


RECIPIENT_COMPARISON = re.compile(r'(?:(?:そんなの|そういう(?:こと|話)は|それ(?:は|なら))、?)?(?:俺|オレ|おれ|僕|ボク|ぼく|私|わたし|あたし)より[…。]*')


def comparative_continuation(unit, scene, bounds):
    """Restore an omitted predicate when the same panel redirects an imperative."""
    from .contracts import strict_json
    matches = []
    for other in scene:
        if other.page_json!=unit.page_json or other.selected.origin!='raw_main_reading':
            continue
        # Godan imperatives take the e-row ending. Restore the dictionary
        # form in the comparative clause, e.g. 頼めよ -> 頼むより.
        command = re.search(r'に(?:でも)?([一-鿿]{1,4})([けげせてねべめれ])(?:よ|や)[!！。…]*$',other.selected.text)
        if not command:
            continue
        rows = strict_json(other.selected.evidence_json)
        if not isinstance(rows,list) or not rows or not all(row.get('polygon') for row in rows):
            continue
        if all(bounds[0]<=x<=bounds[2] and bounds[1]<=y<=bounds[3]
               for row in rows for x,y in row['polygon']):
            matches.append(command[1]+command[2].translate(str.maketrans('けげせてねべめれ','くぐすつぬぶむる')))
    if len(matches)==1:
        return unit.selected.text.partition('より')[0]+'に'+matches[0]+'より…'


def expanded_short_label(source, target):
    """A short printed name or a single mark cannot supply a whole extra sentence."""
    source = source.strip(' 。!！?？…')
    if re.fullmatch(r'[あえおうんへぁぇぉ]っ?',source) and len(target.split())>=4:
        return True
    if detached_connective(source) and len(target.split()) >= 3:
        return True
    if re.fullmatch(r'[A-Za-z0-9_.:/-]{1,12}',source):
        return len(target.split()) >= 5 and len(target) > 3*len(source)
    if re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,2}',source) and re.search(r'\d+[.,]\d+',target):
        return True
    if re.fullmatch(r'[ァ-ヺー・]{2,14}', source):
        return len(target.split()) >= 8 and len(target) > 3*len(source)
    return 0 < len(source) <= 2 and len(target.split()) >= 8


def borrowed_name(source, target, nearby):
    # Restrict this check to short Japanese lines without a possible katakana name.
    if not re.fullmatch(r'[ぁ-ゖ一-鿿々]{2,8}[。！？…]*', source):
        return False
    names = {word.casefold() for line in nearby for word in re.findall(r'[A-Za-z]{4,}', line)}
    return bool(names.intersection(re.findall(r'[a-z]+', target.casefold())))


def extra_circumstance(source, target):
    """A single accidental action cannot supply an accompanying event."""
    return bool(re.fullmatch(r'(?:きっと|たぶん|たしか|あっ)?[.…、]*うっかり(?:して)?[一-鿿ぁ-ゖ]{1,8}'
                             r'(?:ちゃった|てしまった|でしまった)(?:んだ|の)?[。.!！…]*',source)
                and not re.search(r'を|まま|ながら|ところ|ので',source)
                and re.search(r'\b(?:with (?:the|an?|my|our|his|her)|while|because|leaving the)\b',target,re.I))


SELECTION = {SOUND_LETTERING:'sound_lettering', BORROWED:'isolated_target', NOUN_LABEL:'common_noun', URGENT:"urgency", BAR_CALL:"bar_address", REREAD:"consistent_local_reread", **{key:'visual_context' for key in VISUAL_POLICIES}, TICKET_GATE:'station_label', BANNER_TERM:'page_title_term', GLOSS:'single_caption_gloss', SUBJECT:'unfinished_subject', COMMAND:'command_verb', LOAN:'technical_loanword', POLICY:'single_rendition',
             UNFINISHED:'unfinished_contrast', WAIT:'request_to_wait'}


def lexical_hint(source, policy):
    """Read UniDic's Japanese verb lemma or loanword entry for the actual source."""
    try:
        from fugashi import Tagger
        import unidic_lite
        words = list(Tagger(f'-d "{unidic_lite.DICDIR}"')(source))
    except (ImportError,RuntimeError):
        return None
    if policy == NOUN_LABEL and len(words) == 1 and not words[0].is_unk:
        entry = words[0].feature
        if (entry.pos1, entry.pos2, entry.pos3) == ('名詞','普通名詞','一般'):
            return entry.lemma
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
        if (SINGLE_ADJECTIVE.fullmatch(target.strip()) and re.fullmatch(r'[一-鿿]{2,6}',source)
                and lexical_hint(source,NOUN_LABEL)):
            return NOUN_LABEL
        if SHRIEK.fullmatch(source) and re.fullmatch(r'[* ]*kya+[!.* ]*',target,re.I):
            return SOUND_LETTERING
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


def verified_spelling(source, primary, response):
    """A visual reread may change only the ambiguous character in the copied word."""
    token = SPELLING_TOKEN.search(source)
    if token and isinstance(response,str):
        word = token[0]
        if response.strip() in (word.replace('×','x'),word.replace('×','X')):
            pattern = r'(?<![A-Za-z0-9])'+re.escape(word)+r'(?![A-Za-z0-9])'
            return re.sub(pattern,response.strip(),primary)
    return primary


def candidate_matches(policy, primary, candidate, source='', *, term=None):
    if ' / ' in candidate:
        return False
    if policy == PRINTED_SPELLING:
        token = SPELLING_TOKEN.search(source)
        return bool(token and candidate!=primary and
            any(verified_spelling(source,primary,token[0].replace('×',x))==candidate for x in ('x','X')))
    if policy == SOUND_LETTERING:
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*(?: [A-Za-z'-]+)?[!….]*", printed_cadence(source, candidate)))
    if policy == NOUN_LABEL:
        return (not SINGLE_ADJECTIVE.fullmatch(candidate.strip())
                and bool(re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z']+){0,4}[.]?",candidate.strip())))
    if policy == BORROWED:
        removes_circumstance = (extra_circumstance(source,primary) and not extra_circumstance(source,candidate)
                                and len(candidate.split()) <= len(primary.split())+2)
        return bool(re.search(r'[A-Za-z]', candidate) and (removes_circumstance or len(candidate.split()) < len(primary.split()))
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
        if policy == VISUAL_OWNER:
            return 0 < len(candidate.split()) <= 2*len(primary.split())+4 and '\n' not in candidate
        if policy == VISUAL_REACTION:
            return '\n' not in candidate and 2<=len(candidate.split())<=2*len(primary.split())+4
        if policy == VISUAL_TONE:
            return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*(?: [A-Za-z'-]+)?[!….]*", printed_cadence(source,candidate)))
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


def payload(original, policy=POLICY, source='', *, term=None, punctuation='', scene=()):
    from copy import deepcopy
    result=deepcopy(original)
    messages=result['messages']
    if len(messages)!=1 or messages[0]['role']!='user':
        raise ValueError('single-rendition retry requires one native specialist user message')
    text=messages[0]['content']
    marker='Please translate the following '
    if marker not in text:raise ValueError('unknown specialist prompt structure')
    if policy == PRINTED_SPELLING:
        word = SPELLING_TOKEN.search(source)[0]
        messages[0]['content'] = (f'Read the spelling of the Latin word corresponding to OCR “{word}” in the '
            'attached lettering. The text may run vertically. Distinguish a letter from a printed symbol. '
            'Return only the word as printed, keeping its case. Japanese context: '+source)
        return result
    if policy == VISUAL_REACTION:
        messages[0]['content'] = ('Render the line in two parts. For the sound, write a pleased English hum or a '
            'touched sigh, and for the speech write only the words following that sound. Preserve repetition. Japanese: '+source)
        result['format'] = {'type':'object','additionalProperties':False,'required':['sound','speech'],
            'properties':{'sound':{'type':'string'},'speech':{'type':'string'}}}
        return result
    if policy == VISUAL_OWNER:
        part = UNMARKED_PART.search(source)[0][:-1]
        messages[0]['content'] = ('Look at the actual action in this comic panel. What physical thing is touched, '
            'and who or what has the named part '+part+'? If it belongs to an inanimate object, name the object in '
            'ordinary Japanese, using a noun from the surrounding dialogue if possible. If ownership is unclear '
            'leave the noun empty. Target lettering '+source+'. Other lettering from this page: '+' / '.join(scene[:32]))
        fields = {key:{'type':'string'} for key in ('observation','owner_noun_ja','owner_type')}
        fields['owner_type']['enum'] = ['inanimate','person','animal','unclear']
        result['format'] = {'type':'object','additionalProperties':False,'properties':fields,'required':list(fields)}
        return result
    if policy in (BORROWED, NOUN_LABEL):
        messages[0].pop('images', None)
        if policy == NOUN_LABEL:
            messages[0]['content'] = ('The printed Japanese is a common noun. Give a compact English noun phrase '
                                      'for the entry. Keep its original meaning. Please translate the following '
                                      'Japanese text into English: '+source)
            result['format'] = {'type':'object','additionalProperties':False,'required':['label'],
                                'properties':{'label':{'type':'string'}}}
        else:
            messages[0]['content'] = INSTRUCTION + marker + text.rsplit(marker, 1)[1]
        return result
    if policy == SOUND_LETTERING and SHRIEK.fullmatch(source):
        messages[0] = {'role':'user','content':(
            'Choose the short cry an English-speaking character would actually utter for this sound in a comic. '
            'Print the cry alone with an exclamation point. A sudden, high-pitched shriek of surprise.')}
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


def printed_commands(unit):
    from .contracts import strict_json
    if unit.selected.origin!='raw_main_reading' or 'command_list' not in unit.selected.evidence_json:
        return ''
    rows = strict_json(unit.selected.evidence_json)
    if not rows or not strict_json(rows[0]['metadata_json']).get('command_list'):
        return ''
    return ' '.join(row['text'] for row in rows)


def printed_clause_break(unit):
    """A new printed line beginning そう、これ… starts a new thought after a finite verb."""
    from .contracts import strict_json
    if unit.selected.origin!='raw_main_reading' or not re.search(r'そう[、,][こそあ]',unit.selected.text):
        return ''
    evidence = strict_json(unit.selected.evidence_json)
    if not isinstance(evidence,list) or not all(isinstance(r,dict) for r in evidence):
        return ''
    lines = [r.get('text','') for r in evidence]
    if ''.join(lines)!=unit.selected.text:
        return ''
    for i,line in enumerate(lines[1:],1):
        if (re.match(r'そう[、,](?:これ|それ|あれ|この|その|あの)',line)
                and re.search(r'(?:[くぐすつぬぶむるた]|ます|です)$',lines[i-1])):
            return ''.join(lines[:i])+'。'+''.join(lines[i:])
    return ''


def small_print_lines(unit):
    """Identify a short, smaller line beneath left-aligned display lettering."""
    from .contracts import strict_json
    if unit.selected.origin != 'raw_main_reading':
        return ()
    rows = strict_json(unit.selected.evidence_json)
    if not isinstance(rows,list) or not 2 <= len(rows) <= 5 or not all(isinstance(r,dict) for r in rows):
        return ()
    lines = [r.get('text','') for r in rows]
    if (''.join(lines) != unit.selected.text or len(lines[-1]) > 14
            or not re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿 A-Za-z・]+',lines[-1])
            or not all(r.get('polygon') for r in rows)):
        return ()
    def box(row):
        xs,ys = zip(*row['polygon'])
        return min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)
    x,y,w,h = box(rows[-2])
    lx,ly,lw,lh = box(rows[-1])
    if (lw >= 1.6*lh and w >= 2*h and .45*h <= lh <= .82*h
            and abs(x-lx) <= .45*h and y+.55*h <= ly <= y+1.5*h):
        return lines
    return ()


def name_label(line, *, byline=False):
    """Suggest a dictionary spelling where the printed kana identify a personal name."""
    match = re.fullmatch(r'(?:[A-Z][A-Z0-9_-]{1,16} *)?([一-鿿]{1,4})[ 　]*([ぁ-ゖァ-ヺー]{1,8})(?:その他[…・.]*)?',line)
    if not match:
        return None
    try:
        from fugashi import Tagger
        import unidic_lite
        import jaconv
        tagger = Tagger(f'-d "{unidic_lite.DICDIR}"')
        words = list(tagger(match[1]))
        if len(words) != 1:
            return None
        surname = words[0].feature
        first_name = any(len(row)==1 and row[0].feature.pos4=='名'
                         for row in tagger.nbestToNodeList(match[2],8))
        has_label = bool(re.match(r'[A-Z][A-Z0-9_-]+',line) or 'その他' in line)
        if not ((surname.pos4=='姓' and (byline or first_name)) or (has_label and first_name)):
            return None
        family = jaconv.kata2alphabet(surname.kana or '').capitalize()
        given = jaconv.kana2alphabet(jaconv.kata2hira(match[2])).capitalize()
        return (family,given) if family and given else None
    except (ImportError,RuntimeError):
        return None


def kana_verb_spelling(source):
    """Clarify a kana-only prohibitive or a verb whose は can be misread as a particle."""
    clause = source.rstrip('。!?！？…')
    if not re.fullmatch(r'(?:または[ぁ-ゖ]{2,20}|[ぁ-ゖ]{3,16}な)',clause):
        return None
    try:
        from fugashi import Tagger
        import unidic_lite
        words = list(Tagger(f'-d "{unidic_lite.DICDIR}"')(clause))
    except (ImportError,RuntimeError):
        return None
    if len(words)==2 and words[-1].surface=='な' and words[-1].feature.pos2=='終助詞':
        verb = words[0]
    elif len(words)>=2 and words[0].surface=='また' and words[1].surface.startswith('は'):
        verb = words[1]
    else:
        return None
    if verb.feature.pos1 != '動詞' or len(verb.surface)<3:
        return None
    base,lemma = verb.feature.orthBase or '',verb.feature.lemma or ''
    if not re.fullmatch(r'[一-鿿]+[ぁ-ゖ]+',lemma):
        return None
    ending = re.search(r'[ぁ-ゖ]+$',lemma).group()
    if not base.endswith(ending) or len(base)<=len(ending):
        return None
    stem = base[:-len(ending)]
    if verb.surface.startswith(stem):
        return source.replace(verb.surface,lemma[:-len(ending)]+verb.surface[len(stem):],1)
    return None


def shortage_exclamation(source, nearby):
    """A surprised じゃないの can affirm an observable shortage mentioned again in the scene."""
    match = re.search(r'(?:が|は)(空(?:っぽ)?|不足|品切れ|売り切れ|在庫切れ)じゃないの[!！?？。…]*$',source)
    if (not match or not re.search(r'おや|あら|まあ|まさか',source[:10])
            or not any(re.search(r'(?:これ|それ|あれ|材料|在庫)がないと',line) for line in nearby)):
        return ''
    observed = source[:match.start(1)]+match[1]+'だ'
    return (f'The speaker is surprised by what they find. A second line reports a consequence of its absence. '
            f'Here the exclamatory じゃないの invites agreement with the observation {observed}. '
            'Translate the affirmative discovery, preserving the speaker’s surprise.')


def annotation_guide(record):
    if not record:
        return ''
    base,gloss=record['main'],record['reading']
    if gloss=='※':
        return f'The printed note marker ※ on {base} points to a footnote. Set an English footnote asterisk immediately after the corresponding term, then continue the sentence. Preserve any unfinished clause.'
    return (f'The author wrote {gloss} in small kana alongside {base}. If it gives the ordinary '
            'pronunciation, translate the main text once. If it supplies a distinct name, allusion or pun, '
            'convey both in one concise English line.')


def repeated_addressee(source, before):
    """Carry an explicitly addressed person through a short run of repeated predicates."""
    predicate=source.rstrip('ーっッ!！…。')
    if not re.fullmatch(r'(?:必要|大切|大事|好き|大好き)(?:だ|です)',predicate):
        return ''
    recent=before[-8:]
    for i in range(len(recent)-1,-1,-1):
        call=recent[i].rstrip('!！…。')
        if not re.fullmatch(r'(君|きみ|あなた|お前|おまえ|あんた)も[、,]?\1も',call):
            continue
        between=recent[i+1:]
        if between and (between[0].rstrip('ーっッ!！…。')!=predicate
                or any(re.search(r'(?:これ|それ|あれ|私|僕|君|あなた|お前|あんた)[をもはが]',v) for v in between)):
            return ''
        return re.match(r'[^も]+',call)[0]+'が'+source
    return ''


def preceding_topic(source, unit, scene):
    """Recover the omitted object of an informal obligation from the previous remark."""
    if ('を' in source or len(source)>26 or not re.search(
            r'[てで](?:や(?:んね[ーえ]と|ら(?:ないと|なきゃ))|あげ(?:ないと|なきゃ))[…。！!]*$',source)):
        return ''
    index = next((i for i,other in enumerate(scene) if other.unit_id==unit.unit_id),0)
    for previous in reversed(scene[max(0,index-2):index]):
        text = previous.selected.text
        if previous.page_json==unit.page_json and re.search(r'[ぁ-ゖァ-ヺ一-鿿](?:だよな|だよね|なんだ|ですよね|だね)[。！!？?…]*$',text):
            return text[:120]
    return ''


def reply_to_offer(source, unit_id, scene):
    """Resolve contracted ていて in a short reply to a nearby offer of help."""
    ending = re.search(r'([でて])て([ねよ。！!?…]*)$',source)
    if not ending or len(source)>24 or re.search(r'私|わたし|僕|ぼく|ボク|俺|おれ|オレ|けど|ので|から',source):
        return ''
    index = next((i for i,u in enumerate(scene) if u.unit_id==unit_id),None)
    if index is None:
        return ''
    neighbors = scene[max(0,index-3):index+4]
    if not any(u.page_json==scene[index].page_json and u.unit_id!=unit_id
               and re.search(r'何(?:から|を).{0,12}(?:えば|れば|ようか|ましょう)',u.selected.text) for u in neighbors):
        return ''
    polite = source[:ending.start()]+ending[1]+'いてください'+ending[2]
    return (f'The other speaker offers to do something. The reply is addressed to that person. '
            f'The polite expansion of this colloquial request is {polite}. Preserve that speech act and its time reference.')


def positive_reaction(source):
    return (re.match(r'^[うくあ][ぅうぁあっッ]*(?:[…、。]|\.{2,})',source)
            and re.search(r'おいしい|美味しい|うれしい|嬉しい|幸せ',source)
            and not re.search(r'ない|なく|なかった|けど|けれど|のに|とは|わけ',source))


@lru_cache(maxsize=256)
def mimetic_adverb(source, *, handwritten=False):
    """Look up a standalone emphatic kana reading before treating it as speech."""
    word = source.strip(' ！!？?…。')
    if re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,6}[りリ][っッ]',word):
        stem = word[:-1]
    elif handwritten and re.fullmatch(r'[ぁ-ゖァ-ヺー]{2,6}',word):
        stem = word
    else:
        return ''
    try:
        from fugashi import Tagger
        import unidic_lite
        tagger = Tagger(f'-d "{unidic_lite.DICDIR}"')
    except (ImportError,RuntimeError):
        return ''
    words = list(tagger(stem))
    if len(words)==1 and not words[0].is_unk and words[0].feature.pos1=='副詞':
        return words[0].feature.lemma or stem
    return ''


def forum_reading(source):
    """Expand a quoted forum nickname or a transposed command into ordinary Japanese."""
    nickname = re.search(r'教えて(?:[くク][んン]|君|ちゃん)',source)
    if nickname:
        return source[:nickname.start()]+'何でも人に聞くやつ'+source[nickname.end():]
    call = re.fullmatch(r'(まあ|まぁ|おい)([ 、,]*)モチ[ツッ](ケ?)([っッ!！…。]*)',source)
    if call:
        return call[1]+call[2]+('落ち着け' if call[3] else '落ち着…')+call[4]
    return ''


def clarified_spelling(source):
    """Restore ordinary Japanese spelling where emphasis or ellipsis obscures the syntax."""
    # A missing voicing mark in はずれ can turn a location into an invented
    # proper noun. Require a place followed immediately by a locative phrase.
    edge = re.search(r'([ァ-ヺー一-鿿々]{1,18}の)はすれ(?=の[森村町山道駅川]|に(?:ある|住|立|着))',source)
    if edge:
        return source[:edge.start()]+edge[1]+'外れ'+source[edge.end():]
    if re.fullmatch(r'(?:どよ[おぉー]*ん|ドヨ[オォー]*ン)[…。!！]*',source):
        return '重苦しい空気…'
    if re.fullmatch(r'(?:むくり|ムクリ)[っッー]?[。.!！…]*',source):
        return 'むくりと起き上がる'
    challenge = re.fullmatch(r'(?:やんのか|やるのか|やる気か)([、, ]*(?:コラ|こら|てめ[えぇー]|テメ[エェー])[ぁァあアえエぇェー!！?？]*)',source)
    if challenge:
        return '喧嘩を売ってるのか'+challenge[1]
    manner = re.search(r'(?:まとも|マトモ|真面)に(?=[見聞読書話遊働使]|でき|出来)',source)
    if manner:
        return source[:manner.start()]+'ちゃんと'+source[manner.end():]
    location = re.search(r'((?:DB|データベース|日記|記録|名簿|記事|説明書|本)に(?:は|も)一(?:言|行|字)も)([.…。!！]+)$',source,re.I)
    if location:
        return source[:location.end(1)]+'書かれていない'+location[2]
    stem = re.search(r'([ァ-ヺ]{2,12})(し[いくかけ])',source)
    if not stem:
        return ''
    try:
        from fugashi import Tagger
        import unidic_lite
        tagger = Tagger(f'-d "{unidic_lite.DICDIR}"')
    except (ImportError,RuntimeError):
        return ''
    # A katakana noun may immediately precede the emphasized adjective.
    for start in range(len(stem[1])-1):
        kana = ''.join(chr(ord(c)-96) for c in stem[1][start:])+stem[2]
        tokens = list(tagger(kana))
        if len(tokens)!=1 or tokens[0].is_unk or tokens[0].feature.pos1!='形容詞':
            continue
        word = tokens[0].feature.lemma or ''
        if re.fullmatch(r'[一-鿿]+しい',word):
            return source[:stem.start()+start]+word[:-1]+stem[2][-1]+source[stem.end():]
    return ''


def preceding_tier(unit, scene):
    """The upper half of a split detector column anchors this aside to a specific balloon."""
    from .contracts import strict_json
    reading = next((v for v in unit.readings if v.origin=='raw_main_reading'),None)
    if reading is None or 'confirmed_tier_column' not in reading.evidence_json:
        return ''
    for row in strict_json(reading.evidence_json):
        if row.get('reason')!='confirmed_tier_column':
            continue
        proof = strict_json(row['metadata_json']).get('tier_split',{})
        if proof.get('part')!=1:
            continue
        preceding = [other.selected.text for other in scene if other.page_json==unit.page_json
                     and other.unit_id!=unit.unit_id and proof.get('original_id') in other.order]
        if len(preceding)==1:
            return preceding[0][:150]
    return ''


def quoted_term(source, nearby):
    match = re.fullmatch(r'[「『]([ぁ-ゖァ-ヺー]{2,14})[」』]',source)
    if not match and re.fullmatch(r'[ぁ-ゖァ-ヺー]{3,12}',source):
        # A pronunciation correction sometimes explains the preceding joke.
        # Require the actual mistaken form and a similar kana alternative.
        correction = re.compile(r'(?:※|注[：:]?|発音は|読み方は)[「『]?' + re.escape(source)
                                + r'[」』]?(?:ではなく|じゃなくて)[「『]?([ァ-ヺー]{2,14})')
        for line in nearby:
            note = correction.search(line)
            if note:
                try:
                    import jaconv
                except ImportError:
                    return ''
                from difflib import SequenceMatcher
                original = jaconv.kata2hira(source)
                if SequenceMatcher(None,original,jaconv.kata2hira(note[1])).ratio()>=.5:
                    return (f'A printed pronunciation note contrasts {source} with {note[1]}. '
                            f'The target is the deliberately different pronunciation {jaconv.kana2alphabet(original)}. '
                            'Letter that pronunciation by itself, keeping the joke for the correction below.')
    if not match or not any(re.search(r'(?:って|という)(?:いう)?(?:言葉|ことば|名前|呼び名)(?:の意味|の由来|は)',line) for line in nearby):
        return ''
    try:
        import jaconv
        reading = jaconv.kana2alphabet(jaconv.kata2hira(match[1]))
    except ImportError:
        reading = match[1]
    return (f'The next dialogue explains what this quoted term means. Preserve its pronunciation '
            f'({reading}) as the term being discussed. Its definition appears in a separate balloon.')


ITEM_INTRODUCTION = re.compile(r'([一-鿿々])があれば[!！].{1,22}\1です[!！。]*$')


COVER_TITLE_GUIDE = ('This is the whole colored title of the work. For a kana name, romanize the '
                     'printed syllables in order, including combinations with small kana. Preserve its punctuation. '
                     'Keep the title distinct from words in the subtitle.')

def contemptuous_address(source, lines):
    if (len(lines)==3 and ''.join(lines)==source and lines[0].endswith('の')
            and re.fullmatch(r'[一-鿿][一-鿿ぁ-ゖァ-ヺー]+',lines[1])
            and re.fullmatch(r'め(?:…+|\.{2,}|[!！])',lines[2])):
        return ('The separately lettered final め is a contemptuous suffix attached to the noun before it. '
                'The speaker is addressing someone with an insult. Translate that noun phrase as spoken English '
                'and preserve the trailing punctuation. Only give the words to print.')
    return ''


def playful_retraction(source, nearby):
    if re.fullmatch(r'な[ーあ]*ん(?:て|ちゃって|つって)[!！~〜…。]*',source):
        comment=next((line for line in nearby if re.search(r'冗談(?:だ|で|よ|さ)|本気にした',line)),None)
        if comment:
            return (f'Translate this Japanese comic interjection. Nearby the speaker says {comment}. '
                    'This is a playful take-back of the preceding claim. Use a short English interjection '
                    'in that tone. Translate only the target. Japanese: '+source)
    return ''


def printed_alias_guide(source, *, phonetic_fallback=False):
    if not re.match(r'^A\.?K\.?A\.?[一-鿿ァ-ヺ]',source,re.I):
        return ''
    name=re.search(r'[ァ-ヺー]{3,}',source)
    if not name:
        return ''
    import jaconv
    pronunciation=jaconv.kana2alphabet(jaconv.kata2hira(name[0]))
    if phonetic_fallback:
        try:
            from fugashi import Tagger
            import unidic_lite
            words=list(Tagger(f'-d "{unidic_lite.DICDIR}"')(name[0]))
        except (ImportError,RuntimeError):
            words=[]
        if len(words)==1:
            word=words[0]
            if word.is_unk:
                return (f'Translate this short Japanese pseudonym label. Use the romanization {pronunciation.capitalize()} '
                        'exactly once for its unfamiliar name. Begin with AKA. Use a natural English title. '
                        'Express the honorific only once. Give only the English label.')
            if word.feature.pos2=='固有名詞':
                spelling,_,latin=(word.feature.lemma or '').partition('-')
                if spelling==name[0] and re.fullmatch(r"[A-Za-z][A-Za-z' -]{1,30}",latin):
                    return f'Translate this nickname label, keeping AKA. The dictionary spelling of {spelling} is {latin}.'
    return (f'This is an alias on a printed name card. The kana in the name are pronounced {pronunciation}. '
            'Use the established alias if known. If uncertain, retain the printed pronunciation, including '
            'deliberate wordplay. Give only the short English label starting with AKA.')


def small_label_guide(first_line):
    import jaconv
    pronunciation = jaconv.kana2alphabet(jaconv.kata2hira(first_line))
    return (f'This small label has a title on its first line, {first_line} (pronounced {pronunciation}), '
            'and the kind of item on its second line. If the title names something, use that pronunciation '
            'without correcting it to a better-known name. Translate ordinary descriptive words normally. '
            'Give only a compact English label.')


def chat_laughter(source):
    import unicodedata
    normalized = unicodedata.normalize('NFKC',source)
    return (re.search(r'[ぁ-ゖァ-ヺ一-鿿][!?]+[wW]{1,6}$',normalized) is not None or
            re.search(r'[ぁ-ゖァ-ヺ一-鿿]w{2,6}$',normalized) is not None)


def impact_sound(source):
    return re.fullmatch(r'(?:[ガカ]チ|[ガカ]タ|[ゴコ]ト)[ーッ]?ン[ッ!！?？…]*',source) is not None


def cosmetic_practice(source):
    return re.fullmatch(r'エン(?:ゼ|ジェ)ル[・ ]?メイク[。！!]*',source) is not None


def printed_word_guide(source):
    # A middle dot can separate a repeated syllable within a sentence. Require
    # a matching restart followed by kana, so katakana names with dots survive.
    restart = re.search(r'([ぁ-ゖァ-ヺ])[・、]\1([ぁ-ゖァ-ヺー]{2,})', source)
    if restart:
        return (f'The speaker stumbles specifically on {restart[0]}. Repeat the onset of its English '
                'equivalent at that point in the line. Keep the other words fluent.')
    counted = re.search(r'([ァ-ヺ一-鿿ー]{2,20})(?:ひとつ|一つ)(?=(?:まとも|マトモ|満足)に)',source)
    if counted:
        return (f'The complete counted noun is {counted[1]}. The speaker says someone cannot '
                'even handle one of those properly. Keep the object and adverb attached to their own words.')
    application = re.search(r'([ァ-ヺー]{3,18})(?:という|な|の)(?:パッケージマネージャー?|ブラウザー?|メールクライアント)',source)
    if application:
        return (f'{application[1]} names the application. Attach its Latin-script product name as a modifier '
                'directly before the type of application, once. Avoid an appositive or repeated adjective.')
    return ''


def announced_item_effect(source, nearby):
    item_name = re.fullmatch(r'([ァ-ヺー]{2,16})の(魔法薬|秘薬|魔法石|巻物|護符)[！!]+', source)
    if item_name and any(re.search(r'効果|可能にな|変貌|変身|身を守',line) for line in nearby):
        item = item_name[2]
        introduction = next((line for line in nearby if re.search(
            r'(?:これ(?:が|は)|ここにあるのは)(?:私の|僕の|俺の|わたしの)?' + item, line)), None)
        if introduction:
            return (f'The preceding dialogue is {introduction}. The noun before の describes the effect being '
                    'bestowed. Use the form "[item] of [effect]!". Return only a JSON string containing the '
                    'name in English. Japanese: '+source)
    return ''


def self_contained_prompt(source, nearby=()):
    if (re.search(r'[てで]おいて(?:ね|よ)?[。！!]*$',source)
            and not re.search(r'勝手に|無断で|くせに|[私僕俺]が|わたしが',source)):
        return ('Translate the standalone Japanese line into natural dialogue. The final te-oite form asks '
                'the listener to do something in preparation. Keep the addressee as the person doing the '
                'work. Preserve any named group or institution that identifies the person, including its '
                'specific field or subject. Use familiar English for technical abbreviations. Make the request '
                'sound like something a person would actually say. Keep the line concise. Give only the English dialogue. '+source)
    if re.search(r'(?:トイレ|お手洗い|シャワー|浴室|台所|キッチン)(?:を|は)?貸す', source):
        return ('Translate the following Japanese comic lettering. The speaker offers the addressee use of a '
                'fixed facility. Treat 貸す as granting access. Then render the invitation in natural spoken '
                'English. Give only the English lettering. '+source)
    if source.startswith('※') and re.search(r'["“「][A-Za-z][A-Za-z0-9 -]{2,24}["”」]',source):
        from difflib import SequenceMatcher
        context = []
        for line in nearby:
            match=SequenceMatcher(None,source,line,autojunk=False).find_longest_match()
            if match.size>=5 and re.search(r'[一-鿿ァ-ヺ]',line[match.b:match.b+match.size]):
                context.append(line)
        hint = (' Other dialogue mentioning the same term: '+' / '.join(context[:2])+'.') if context else ''
        # Two radicals can be read as separate full characters in fine print.
        # Only suggest a joined glyph when the same surrounding name occurs
        # intact elsewhere on the page, including text on both sides of it.
        for split,glyph in (('女子','好'),('女未','妹'),('亻本','体'),('木木','林')):
            if source.count(split)!=1:
                continue
            pos=source.index(split)
            joined=source.replace(split,glyph)
            for line in context:
                match=SequenceMatcher(None,joined,line,autojunk=False).find_longest_match()
                if match.size>=5 and match.a<=pos-2 and match.a+match.size>=pos+2:
                    term=joined[match.a:match.a+match.size]
                    hint+=f' The same name is printed more clearly as {term} in the dialogue. Use that name in the footnote.'
                    source=joined
                    break
        return ('Translate this manga footnote into concise English. Begin with an asterisk and retain the '
                'quoted Latin spelling exactly.'+hint+' Give only the footnote. '+source)
    if cue := printed_word_guide(source):
        return 'Translate this Japanese dialogue into English. '+cue+' Output only the printable dialogue. '+source
    if chat_laughter(source):
        return ('Translate this short Japanese chat reaction. Final w is shorthand for 笑, marking the '
                'speaker laughing at the remark. Convey that playful tone in natural English chat. '
                'Output only the English reaction. '+source)
    if cosmetic_practice(source):
        return ('Name the cosmetic practice in the pictured panel. The Japanese is '+source+
                '. In a mortuary setting this means 葬儀用のメイク. Translate its meaning with a short English label. No commentary.')
    if ITEM_INTRODUCTION.search(source):
        # A neighboring balloon may discuss a consequence of this item. Copying
        # that sentence can reverse the item's actual effect.
        return ('Letter only the two exclamations in English. This speaker is presenting the item they are holding. '
                'Render the first as a With phrase, then give the second as its own exclamation, '
                'translating the printed description exactly. '
                + re.sub(r'([!！])',r'\1\n',source).strip())
    if re.fullmatch(r'(?:じゃあ?|では)?(?:これ|それ|あれ)(?:も|を)?よろしく(?:ね|な|お願いします)?[。！!]*',source):
        return ('Translate only the spoken request. よろしく in this handoff says what the speaker '
                'wants the addressee to do about this item. '+source)
    if re.fullmatch(r'(?:私|わたし|僕|ぼく|俺)(?:は|も)(?:その|この)中でも[…。・]*',source):
        return ('Translate this unfinished introduction as a fragment and end with an ellipsis. '
                'The speaker is singling out their own role within the previously mentioned group. '
                'The sentence has no past-tense verb. '+source)
    return ''


def contextual_dialogue_prompt(source, nearby, previous=''):
    """Use a recent printed question or statement to resolve an omitted argument."""
    listed = re.search(r'([ァ-ヺー]{3,}|[一-鿿]{2,4})と$',source)
    if listed and any(v.startswith(('そうではない','そうでない')) and listed[1] in v for v in nearby):
        return ('Translate the first item in a spoken list. Write one short English noun phrase and end '
                'with the English words “and…” to lead into the next balloon. Output only the English '
                'lettering for this item: '+source)
    if source.startswith(('そうではない','そうでない')) and previous.endswith('と'):
        category = re.search(r'([ァ-ヺー]{3,}|[一-鿿]{2,4})と$',previous)
        if category and category[1] in source:
            return (f'The speaker lists two kinds of the same thing. The first part is {previous}. '
                    'This next balloon gives the contrasting second kind. Continue the list in English, '
                    'keeping both the negative qualification and the description. Letter only the second part. '+source)
    relationship = re.fullmatch(r'(?:オレ|おれ|ボク|ぼく|私|わたし|あたし|俺|僕)の([一-鿿]{2,4})(?:相手|先|元)[にはを]',source)
    if relationship:
        root = relationship[1]
        example = next((line for line in nearby if root+'関係' in line),None)
        if example and any(re.search(r'パッケージ|ソフトウェア|apt-get|リポジトリ',v,re.I) for v in nearby):
            return (f'The page is about software. A nearby line reads {example}. The speaker personifies '
                    'a software relationship. Use the ordinary software noun for the other component. '
                    'Letter only this short possessive noun phrase in English, ending in a dash. '+source)
    if re.search('反芻|反すう|はんすう',source):
        animal=next((v for v in nearby if re.search('牛|山羊|羊|キリン|ラクダ',v)),None)
        if animal and re.search(r'(?:すっか|しよう|しよっか|するか)[ぁあー!！?？。]*$',source):
            return (f'Translate this comic dialogue. The page identifies the animal with {animal}. '
                    'A ruminant is talking about chewing its food again. Keep the agricultural sense and '
                    'the casual tone. Use the conventional English agricultural term in the first person '
                    'for what the animal will do. Give only a short line, at most ten words. '+source)
    technical=any(re.search('不具合|不調|障害|バグ登録|起動ログ',v) for v in nearby)
    if technical:
        if re.fullmatch(r'(?:分|わ)か(?:んない|らない)(?:ワケ|わけ|訳)だ[ぁあァー！!…。]*',source):
            return ('Translate the Japanese into a single English exclamation. In this context ワケだ is equivalent '
                    'to 道理で, and the current explanation accounts for a problem the speaker had in the past. '
                    'Preserve the negative verb as the speaker’s earlier inability to figure it out. '
                    'Only the English sentence. '+source)
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9.+-]{0,24}の(?:バグ|不具合)[….]*[?？]',source):
            return ('Translate this as a short English noun phrase asking about a software defect, with the '
                    'affected software after the word for the defect. Preserve the question mark. '+source)
        if re.fullmatch(r'(?:しばらく|当面|さしあたり|今のところ)は?(?:この|その)まま[.…。]*',source):
            return ('Translate this unfinished quoted Japanese noun phrase into an unfinished English phrase '
                    'ending in dots. Leave the subject and verb unstated. Only write the fragment: '+source)
    if (re.fullmatch(r'[ァ-ヺー一-鿿]{2,12}の(?:ヤツ|奴|やつ)[っッ!！…。]*',previous)
            and re.fullmatch(r'勝手に[一-鿿ぁ-ゖ]{2,14}[てで][…!?！？.]*',source)):
        return (f'The immediately preceding reaction was {previous}. The speaker has just recognized who acted. '
                'Keep that person as the third-person subject of this surprised discovery. '
                'Translate only the current reaction: '+source)
    if source.startswith(('はじめて投稿','初めて投稿')) and '。' in source:
        match=re.search(r'。([A-Za-z][A-Za-z0-9ー‐-]{2,18})(?=[ぁ-ヺ一-鿿])',source)
        if match and any(match[1].replace('ー','-').casefold() in s.casefold() for s in nearby):
            handle=match[1].replace('ー','-')
            return (f'This is a first-person forum post by {handle}. The opening introduces the author. '
                    'Keep the author’s name in the introduction. The following problem concerns the updated system. '
                    'Preserve any legible software names and version digits. Translate only the visible post: '+source)
    if re.search(r'を[一-鿿]{2,4}させて(?:頂|いただ)[ぁ-ゖ…]*$',source) and source.startswith('以下'):
        return ('The writer introduces the material that follows. Keep the object of the polite verb and '
                'the items in the list together. A partially hidden noun before の is incidental. '
                'Give only natural English for the visible line: '+source)
    if re.fullmatch(r'投稿[：:]\s*\d+',source) and any(re.search(r'登録日|登录日|新メンバ|From[:：]',s) for s in nearby):
        return ('This is a profile statistic in a message board sidebar. Use a count label for posts, '
                'followed by the printed number. English label and value only: '+source)
    if re.fullmatch(r'(?:オシエ|ミセ|タスケ|コタエ|ヤメ|ニゲ|タベ)ロ[!！]*',source):
        hiragana=''.join(chr(ord(c)-96) if 'ァ'<=c<='ヶ' else c for c in source)
        return (f'This is the Japanese command {hiragana}, written emphatically in katakana. '
                'The writer is commanding the addressee to act. Give only the brief English command: '+hiragana)
    if source.rstrip('？?') in ('よかったのか','良かったのか') and source.endswith(('？','?')):
        claim=next((s for s in reversed(nearby) if re.search(r'(?:て|で)(?:よ|良)かった(?:よ|ね|な|です)?[。.…！!]*$',s)),None)
        if claim:
            return (f'The previous speaker says {claim}. The reply echoes their claim in surprise. '
                    f'Translate only the short reply, preserving who expressed relief. English lettering only. {source}')
    if len(re.findall(r'[てで][、,]',source))>=2 and re.search(r'(?:なさい|てやれ|てくれ)[！!。]*$',source):
        topic=[s for s in nearby if re.search(r'(?:これ|あれ|この人|あの人)(?:が|は).{1,12}[？?]',s)]
        if len(topic)==1 and any(re.search(r'同い年|何歳|おいくつ',s) for s in nearby):
            if re.match(r'(?:何で|なんで|どうして).{2,24}[？?]',source):
                return (f'Earlier a person was identified in dialogue: {topic[0]}. The speaker is talking directly '
                        'to the listener. Keep the listener as the subject of the opening question and the commands; '
                        'the other person is the object only of the commanded actions. Give only the English dialogue: '+source)
    return ''


def usage_context(source):
    """Supply ambiguous senses or grammatical cues; the translator chooses the English."""
    # Isolated mimetics also describe silent movement and mood. Keep their Japanese
    # senses contextual so the same spelling may still describe a literal sound.
    if cue := printed_word_guide(source):
        return cue
    if re.fullmatch(r'カ[アァー]{1,5}ッ',source):
        return ('This drawn effect can indicate the sudden heat of embarrassment beside a blushing face, '
                'or a call beside a bird. Identify the visible subject and choose one brief English effect.')
    if re.fullmatch(r'[ばバぱパ][っッ][！!…]?',source):
        return ('This drawn effect marks a sudden movement or change. Read the gesture beside it and use '
                'one short English action beat that fits the visible movement.')
    if re.fullmatch(r'[すスサさ][っッ][!！…]?', source):
        return ('A short mimetic beside a moving hand or object can indicate a quick quiet movement. '
                'Inspect the scene and use a compact action effect when the lettering accompanies a gesture.')
    if chat_laughter(source):
        return 'The final w is Japanese online laughter. Carry its tone into the English reaction.'
    if impact_sound(source):
        return ('This standalone katakana with its final nasal is a drawn impact or contact sound. '
                'Use the pictured object or gesture to choose one concise English sound effect.')
    if cosmetic_practice(source):
        return 'Check the scene: in mortuary usage this denotes 葬儀用のメイク（死化粧）. In a fashion setting it can be decorative.'
    beat = source.strip(' 。.!！?？…')
    if re.fullmatch(r'[あえおうんへぁぇぉ][っッ?!！？…]*',beat):
        return 'Only a short interjection is printed here. Keep it to one English beat, with the hesitation or question intact.'
    if re.fullmatch(r'([ぁ-ゖァ-ヺ][ぁ-ゖァ-ヺー]{1,5}(?:りん|リン|ちゃん|たん|くん))\1',beat):
        return ('A repeated kana nickname can be a chant or call. Check the picture for a written name. '
                'Keep its phonetic reading and repetition when the artwork gives no evidence for an English object.')
    if len(beat) <= 8:
        beat = ''.join(chr(ord(c)-96) if 'ァ' <= c <= 'ヶ' else c for c in beat)
        for pattern, sense in (
                (r'どよ[おぉー]*ん', '場に重苦しい気分が漂う様子。声ではなく雰囲気を表す擬態語。'),
                (r'(?:しっ){2,3}', '人や動物を追い払うときに繰り返して言う短い掛け声。'),
                (r'(?:ぐ[いぃ]){2}[っー]?', '力を込めて何度も引っ張ったり押したりする動作を表す。絵で動きの向きを確かめる。'),
                (r'(?:こち[ょよ]){2}[っー]?', '羽根や指などで軽くくすぐる動作を繰り返すときの擬態語。'),
                (r'ぽん[っー]?', '軽い接触、物が外れたり現れたりする瞬間の短い音。文字のすぐそばの動きを確かめる。'),
                (r'かぽ[っー]?', '中が空いたものがはまったり外れたりする軽い音。物がどう動くかを絵で確かめる。'),
                (r'ぱ[あぁー]+', '顔が急に明るくなったり、辺りが明るく開けたりする様子。人物の表情か光景かを絵で区別する。'),
                (r'しょぼ[ーおぉ]*ん', '元気をなくして肩を落とす様子。人物の様子やしぐさに合わせる。'),
                (r'むぐ[っー]?', '口いっぱいに食べ物を含んで口を動かす、または口をふさがれて声がこもる様子。口もとや食べ物を確認する。'),
                (r'ふる[っー]?', '人物のそばに何度も描かれた場合は小さく体を震わせる様子。動きと文脈で普通の動詞の用法と区別する。'),
                (r'きゃ[あぁー]+', '驚いたときなどの甲高い叫び声。英語の短い叫び声で表す。'),
                (r'(?:ふき){2}[っー]?', '物や顔を何度も拭くしぐさ。繰り返す動作に合わせて短く表す。'),
                (r'むくり[っー]?', '横になった人が体を起こしたり立ち上がったりする様子。跳躍ではない。'),
                (r'こん(?:こん)?', '軽く硬いものをたたく音、あるいは狐の鳴き声。描かれた物や動物で区別する。'),
                (r'が[あぁー]+ん', '衝撃を受けて言葉を失う様子、または強く響く音。絵で心の衝撃と実際の衝突を区別する。'),
                (r'[とこ]て[んっ]', '人や物が軽くころんだり倒れたりする様子。')):
            if re.fullmatch(pattern, beat):
                return ('Japanese dictionary sense of this isolated comic effect: '+sense+
                        ' Render the applicable sense as a one- or two-word English comic beat.')
    if re.fullmatch(r'[「『].{1,46}(?:[うくぐすつぬぶむる]|ない)こと[」』]',source):
        return ('This quotation ends in the plain verb plus こと used in directions or a written rule. '
                'When it specifies a task, express the instruction directly. Keep the quotation marks.')
    if ITEM_INTRODUCTION.search(source):
        return ('The following exclamation identifies the very item the speaker has just presented. '
                'があれば describes what the item enables. Keep that confident offer and the following reveal.')
    if re.search(r'(?:そこまで|間近に)来ているの(?:だから|ですから)',source):
        return ('This is an emphatic explanation that the previously named person or event is nearly here. '
                'The の belongs to のだから. Keep it declarative and retain the sense of approaching.')
    if re.search(r'(?:まとも|マトモ|真面)に(?:[見聞読書話遊働使]|でき|出来)',source):
        return ('The に form modifies how the following action is carried out. Keep that adverb on the '
                'verb, including its negation, rather than applying it to the preceding noun as a quality.')
    if re.search(r'(?:はま|ハマ|嵌)(?:る|っ|り|ら|れ)',source):
        return ('The verb はまる can mean fitting into something, getting stuck in a predicament, '
                'or becoming absorbed in an interest. Identify its object in the adjoining dialogue. '
                'Read the target as a response to that situation.')
    if re.fullmatch(r'[ひヒ][ゅュ][るル]{2,8}[っッー〜～]*',source):
        return ('This sound imitates air whistling as an object moves or falls. '
                'Use the pictured movement to choose a concise audible English effect.')
    if re.fullmatch(r'ミ[ー〜～]+ン[ーッ]*',source):
        return ('This drawn-out Japanese sound often represents a cicada buzzing in summer. '
                'Check the scene for an insect call or another sustained sound.')
    address = re.search(r'([〇零一二三四五六七八九十百0-9０-９]+)(丁目|番地)',source)
    if address:
        number = address[1] if re.search(r'[十百]',address[1]) else address[1].translate(str.maketrans('〇零一二三四五六七八九','00123456789'))
        return (f'{address[0]} is a numbered address designation ({number}{address[2]}). Retain its number and locality '
                'when describing where someone or something is.')
    deadline = re.search(r'(登校|下校|出勤|退勤|出国|入国|下船|下車)まで(?:に|には)',source)
    if deadline:
        return (f'The deadline is the event {deadline[1]} itself. Distinguish departure from arrival. '
                'Keep the same endpoint. An unstated onward destination is outside this time expression.')
    concurrent = re.search(r'(.{1,24})ながら(.{1,18}?)しないで[ーっッ!！…。]*$',source)
    defer = re.search(r'([^、。！？]{1,28}(?:する|える|ける|める|見る|読む|行く))のを(?:お)?待(?:ちください|って)',source)
    if defer:
        return (f'The act ending in {defer[1]} is what the listener is asked to postpone. '
                'Express holding off on that action, keeping the original level of politeness.')
    if concurrent:
        return (f'The request forbids {concurrent[2]}する, the main action after ながら. '
                f'{concurrent[1]}ながら describes the simultaneous action. Keep that distinction.')
    if re.search(r'うっかり.{0,20}(?:ちゃった|てしまった|でしまった)',source):
        return ('うっかり describes how the stated event happened. Keep that one accidental action. '
                'Avoid adding circumstances from the drawing or other dialogue to the spoken line.')
    passive = re.search(r'([一-鿿]{1,8})も(?:取られ|盗られ|盗まれ|奪われ|捕らえられ)(?:て[い]?る|た)',source)
    if passive:
        return (f'{passive[1]} is what was taken or affected. The speaker can be adversely affected '
                'without being the thing taken. Preserve the named participant and the list structure.')
    if positive_reaction(source):
        return ('Interpret the opening vocalization with the positive words the speaker actually says. '
                'Look at the speaker before choosing the English sigh. A pleased hum or tearful “Oh…” '
                'can express being moved. Keep the repetition. Reserve “Ugh” for disgust or frustration.')
    if len(menu_shortcuts(source))>=2:
        return ('These are separate software menu labels. Read the labels against the screenshot, '
                'retaining the parenthesized keyboard access key after each one in the same order.')
    if re.search(r'権力(?:に[もは]?|を|者に)影響を与え',source):
        return ('権力 refers to authority or the people exercising it. 与える belongs to 影響を与える. '
                'Preserve the difference between influencing someone and directly controlling them.')
    if re.search(r'全然[っッ!！…。]*$',source):
        return ('The lettering cuts off at the intensifier 全然 with no predicate. Read the reaction '
                'with the neighboring speech. Keep it unfinished if the speaker stops. Do not insert '
                'an assessment of quality, ability or quantity unless it is actually stated.')
    if re.search(r'(それは){2}|(とても){2}|(本当に){2}',source):
        return ('The speaker doubles an intensifier before the description. Carry that extra degree '
                'of emphasis into the English wording.')
    if re.search(r'(?:俺|オレ|おれ|僕|ボク|ぼく|私|わたし|あたし|あなた|君)より[。…]*$',source):
        return ('The visible line ends at the comparative particle より after a pronoun. '
                'Leave the comparison open for the next balloon. Do not invent an ability, action or conclusion.')
    if (re.search(r'具合|調子', source)
            and re.search(r'ターミナル|起動|再起動|端末|ソフト|パソコン|システム|サーバ', source)):
        return ('The target explicitly describes troubleshooting technology. Resolve the omitted subject '
                'of its condition as the device or system. Translate the technical symptom in that sense.')
    if '名折れ' in source:
        action = '帰す means letting someone leave. ' if re.search(r'帰し|帰す', source) else ''
        return (action + '名折れ is a loss of personal professional honor. Name the speaker’s own profession; '
                'use direct address if the other speaker is present. Do not introduce an organization.')
    if re.search(r'(?:姉|妹|兄|弟|父|母|祖父|祖母)さんの.+を[？?]$',source):
        return ('The Japanese ends with an object marker and no verb. Print only a noun phrase and question mark. '
                'Do not add an action, an offer, or a person doing something. Express の with an English '
                'possessive. In direct address a respectful kinship term can mean the listener’s relative. '
                'For example, 亡くなったお父さんの時計を？ can be “Your late father’s watch?”')
    name = name_label(source)
    if name:
        return (f'This label contains the personal name {name[0]} {name[1]} in family-name-first order. '
                'Use the printed given name and retain surrounding labels or qualifiers.')
    imitation = re.search(r'([^。、！？]{1,24})の(.{0,6})(?:マネ|まね|真似)ばっ?か(?:り|し)?じゃない',source)
    if imitation:
        return (f'{imitation[1]} is the model being imitated. The omitted subject does the copying. '
                'The speaker complains that copying is ALL it does. Keep that criticism and the object of imitation.')
    if re.search(r'ばっ?か(?:り|し)?じゃない[!?！？。…]*$',source):
        return ('In colloquial Japanese, a final じゃない can seek agreement: “All it does is …, isn’t it?” '
                'When the source complains about 悪い things, keep the “nothing but” complaint. Do not negate it or switch the agent.')
    if re.search(r'ネットニュース|ニュースグループ',source) and re.search(r'投稿|購読|ユニックス|[Uu]nix',source):
        return ('In historical computing dialogue, check whether the speaker posted a message to a '
                'Netnews or Usenet newsgroup. Distinguish that discussion system from a news publication.')
    document = re.search(r'(オレ|俺|僕|あたし|私|わたし|君|あなた)の([^。、！？]{0,20}(?:DB|データベース|日記|記録|名簿|記事|説明書|本))には?一(?:言|行|字)も',source,re.I)
    if document:
        owner,book = document.groups()
        who = 'the listener' if owner in ('君','あなた') else 'the speaker'
        return (f'{owner} owns the {book}. The Japanese locates the missing mention INSIDE that document. '
                f'The information is absent from the document belonging to {who}. Retain the personal possessive '
                f'on the document itself. Translate the full document description「{book}」including its modifier. '
                'Leave the unprinted topic implicit.')
    if re.search(r'(?:DB|データベース|日記|記録|名簿|記事|説明書|本)に(?:は|も)一(?:言|行|字)も',source,re.I):
        return ('The document or database is the place where information is missing. Preserve the '
                'implied negative and any unfinished ending. The speaker is looking for a mention in that source.')
    if re.search(r'(?:登録|登录|登錄)日\s*[:：]\s*[0-9０-９]{4}[-/年]',source):
        return ('This profile lists a date of registration. Keep the printed date and separate its field '
                'from the creation date of a post and the last login time.')
    if re.search(r'を(?:勤|務|つと)めるべき(?:なの)?は[…。]*$',source):
        return ('A noun before を names the role or function someone or something should fill. '
                'The line ends at は before identifying who or what should serve in that role.')
    if re.search(r'(?:つな|繋)いで[い]?るのは.+と.+(?:です|だ)',source):
        return ('This cleft identifies both ends of one connection. The two items joined by と '
                'are connected to one another. Keep both endpoints and any request for confirmation.')
    if re.fullmatch(r'[こそあ]ういう(?:の|こと|人)って[…。]*',source):
        return ('The quoted topic is unfinished at って. Keep the demonstrative and the topic open '
                'for its predicate in the adjoining line.')
    if re.fullmatch(r'(?:あんた|あなた|君|僕|俺|私|彼女|彼)の(?:物|もの|分)も[…。]*',source):
        return ('A possessive noun phrase ends in additive も. Keep the additional possessor and '
                'leave the action open. A speaker can add another item without asking about ownership.')
    if re.search(r'(?:ちゃん|ちん|さん|くん|君)[ 、,]*ばん(?:ほ|わ)[ー～ぁあっ!！…。]*$',source):
        return ('A familiar nickname is followed by a playful evening greeting, related to こんばんは. '
                'Keep the speaker greeting the person addressed.')
    ordinary = forum_reading(source)
    if ordinary:
        return f'The forum expression here has the ordinary Japanese reading {ordinary}. Keep the informal tone.'
    if re.search(r'(?:ない|ねえ|なかった|なかったんだ)(?:ワケ|わけ|訳)だ[ぁあねなよ。！!…]*$',source):
        return ('Sentence-final わけだ presents the speaker realizing the explanation for something. '
                'Preserve the inference as well as the negative predicate.')
    fandom = re.search(r'([ァ-ヺー]{2,12}|[A-Za-z]{2,16})厨', source)
    if fandom:
        abbreviation = 'In Japanese computing slang リナ can abbreviate Linux. Check the topic of the page. ' if fandom[1]=='リナ' else ''
        return (abbreviation + f'In internet slang the suffix 厨 describes an obsessive fan of the preceding name '
                f'{fandom[1]}. Keep the specific service or fandom name in English, using its printed '
                'nickname if necessary. Preserve the speaker’s attitude.')
    if re.search(r'ググれ|ググって', source):
        return ('Read this as colloquial internet speech. Distinguish an imperative to search from '
                'a description. After a manual, FAQ or other reading material, a slang homophone such '
                'as 嫁 can stand for 読め. Preserve separate commands and insults.')
    # Japanese corner quotes sometimes become an ASCII r/J pair in Latin titles.
    # Require a publication context and matched boundaries before suggesting a reading.
    if re.search(r'初出|掲載|連載|発行|出版|誌名|書名', source):
        quoted = re.search(r'(?:^|[^A-Za-z0-9])r([A-Z][A-Za-z0-9 .&-]{2,60}[a-z0-9])J(?=$|[^A-Za-z0-9])', source)
        if quoted:
            return (f'This publication credit contains a quoted title. The OCR r/J pair resembles Japanese '
                    f'corner quotes around “{quoted[1]}”. Use that title if the context agrees. '
                    'Distinguish the title of the publication from its publisher.')
    pronunciation = re.search(r'[（(]([ぁ-ゖ]+(?:[ 　]+[ぁ-ゖ]+)*)[)）]', source)
    if pronunciation:
        return (f'The parenthetical kana supply the pronunciation {pronunciation[1]}. '
                'Romanize the printed kana exactly, including voicing, and use the same spelling each time the name occurs.')
    spelling = kana_verb_spelling(source)
    if spelling:
        return (f'The dictionary identifies the verb in this kana line. With that same inflection, '
                f'the kanji spelling reads {spelling}. Use this reading if it agrees with the scene.')
    cues = (
        (r'(?:見られ|聞きつけられ|見つけられ|気づかれ|知られ|捕まえられ)ないように', 'This avoidance clause is passive. The person approaching, escaping, or hiding wants to avoid being seen, heard, noticed or caught by someone else. Preserve who observes whom.'),
        (r'(?:で|について|巡って).{0,8}揉め', 'People are disagreeing over the stated issue. Distinguish that interpersonal dispute from a technical malfunction.'),
        (r'^(?:死んで|亡くなって|倒れて|寝て|消えて|逃げて).{0,15}(?:よね|でしょう|ですか)',
         'Preserve the degree of certainty. んですよね presents the state as understood and seeks agreement. '
         'Use a statement with a confirming tag for that form. Keep the referent open unless the scene identifies it.'),
        (r'具合|調子', 'Identify whose condition is discussed. A machine, device, or system can malfunction; do not assume a person is sick.'),
        (r'票.*(?:獲得|集め|投じ)|(?:獲得|集め|投じ).*票', 'In a committee or election, a group directly before 票 names those casting the votes. Distinguish gaining votes from that group from campaigning for its seats or collecting ID cards.'),
        (r'を[？?]$', 'This is a noun phrase with an object marker and a question mark. Render it as an English noun phrase with a question mark. Leave the actor and verb unstated.'),
        (r'[ァ-ヺ一-龯]市長', 'A place name before a civic title can identify the jurisdiction. Check whether the speaker is naming a town or a person.'),
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


def split_onset(source, lines):
    if len(lines)<2 or not re.fullmatch('[ぁ-ゖァ-ヺ]', lines[0]) or len(lines[1])<2:
        return False
    return lines[1].startswith(lines[0]) and source==''.join(lines)


def stuttered_reading(unit):
    from .contracts import strict_json
    raw = next((r for r in unit.readings if r.origin=='raw_main_reading'),None)
    if raw is None or len(unit.order)<2:
        return False
    readings = strict_json(raw.evidence_json)
    if not isinstance(readings,list):
        return False
    by_id = {row.get('source_id'):row.get('text','') for row in readings if isinstance(row,dict)}
    return split_onset(unit.selected.text,[by_id.get(i,'') for i in unit.order])


def printed_cadence(source, target, *, unit=None):
    """Keep a trailing thought open and avoid duplicating a single uncertainty marker."""
    if not target or not target.strip():
        return target
    target = target.strip()
    framed = re.fullmatch(r'([-–—])([^-–—]+)\1',target)
    if framed and re.fullmatch(r'([-–—])[ぁ-ヺ一-鿿々]+\1',source):
        target = '– ' + framed[2].strip() + ' –'
    laugh = re.fullmatch(r'(ha|he|ho|hi)(?:\1){2,}([.!…]*)',target,re.I)
    if laugh and re.fullmatch(r'([はハへヘほホひヒ])\1{2,}',source):
        syllable = target[:2]
        target = ' '.join([syllable]+[syllable.lower()]*(len(source)-1))+laugh[2]
    if (('【' in target or '】' in target) and re.search('[A-Za-z]',target)
            and not re.search('[ぁ-ヺ一-鿿]',target)):
        # English typefaces commonly lack Japanese lenticular brackets. Keep
        # the same emphasis and enclosure in the English transcript and page.
        target = target.replace('【','[').replace('】',']')
    motif = re.fullmatch(r'([ぁ-ゖァ-ヺー]{2,4})\1',source)
    beat = re.fullmatch(r'([A-Za-z]{2,16})([.!…]?)',target)
    if motif and beat and unit is not None and len(getattr(unit,'order',()))==2:
        # The same effect on two distinct printed rows carries two beats.
        # A repeated spelling inside a single word supplies no such count.
        from .contracts import strict_json
        raw = next((r for r in unit.readings if r.origin=='raw_main_reading'),None)
        lines = strict_json(raw.evidence_json) if raw else []
        large = str.maketrans('ゃゅょャュョ','やゆよヤユヨ')
        if (len(lines)==2 and [r.get('source_id') for r in lines]==list(unit.order)
                and all(r.get('text','').translate(large)==motif[1].translate(large) for r in lines)):
            target = beat[1]+' '+beat[1].lower()+beat[2]
    if (unit is not None and source.endswith('ッ') and unit.selected.origin=='raw_main_reading'
            and re.fullmatch(r'[A-Za-z]{2,14}[!…]?',target)):
        from ..ocr.sound_readings import repeat
        from .contracts import strict_json
        motif = repeat(source[:-1])
        if motif and 2<=len(source[:-1])//len(motif)<=4:
            lines = strict_json(unit.selected.evidence_json)
            if (isinstance(lines,list) and len(lines)==1 and
                    lines[0].get('reason')=='corroborated_repeated_effect'):
                count = len(source[:-1])//len(motif)
                word = target.rstrip('!…')
                target = ' '.join([word]*count)+target[len(word):]
    if ((unit is not None and len(source)>=3 and source[0]==source[1] and stuttered_reading(unit)
             or re.match(r'^([ぁ-ゖァ-ヺ])[っッ]\1(?![っッ])[ぁ-ゖァ-ヺ一-鿿](?:[ぁ-ゖァ-ヺ一-鿿]|[A-Z0-9])',source))
            and re.match(r'^[A-Za-z]',target)
            and not re.search(r'^([A-Za-z]+)(?:[-–—,]|\.{2,}|…|\s)\s*\1',target,re.I)):
        # Repeat the onset of the translator's own word. Preserve an existing
        # stutter, including a repeated whole first word.
        first = re.match(r'(?:th|sh|ch|[a-z])',target,re.I)[0]
        target = first + '-' + target
    # OCR often confuses a rotated capital I with the kana length mark. The
    # spelled-out definition independently supplies its initials in English.
    uncertain = re.search(r'(?<![A-Za-z])([A-Z]{2,5})[ーⅠ]とは', source)
    if uncertain:
        stem = uncertain[1]
        definition = re.search(r'\b' + stem + r'\s+(?:stands for|means|is short for)\s+'
                               r'([A-Za-z][A-Za-z -]{2,80})', target, re.I)
        if definition:
            words = re.findall(r'[A-Za-z]+(?:-[a-z]+)?', definition[1])
            initials = ''.join(word[0].upper() for word in words)
            if initials == stem + 'I':
                target = target[:definition.start()] + initials + target[definition.start()+len(stem):]
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


TABLE_CELL_GUIDE = ('This is one entry in a narrow printed table. Read it as a short label or answer '
    'in its column, using the shortest faithful English. Preserve meaningful qualifiers, numbers, '
    'and visibly clipped wording. For a relationship or interest, distinguish a named event, place, or '
    'community from a generic activity. Use its established English name. Do not add a subject, an explanation or a final period.')


OUTLINED_FRAGMENT_GUIDE = (
    'The large outlined word was read separately from the finer writing behind it. '
    'It trails off with an ellipsis. Use the depicted activity and nearby dialogue to infer '
    'the interrupted word. Letter that brief action or reaction in English with an ellipsis.')
