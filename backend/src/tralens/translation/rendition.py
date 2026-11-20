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
COMMAND = 'untranslated_command_verb_v1'
LOAN = 'technical_loanword_v1'
GREETING = re.compile(r"(?:welcome(?: back)?|hello|hi|good(?: morning| evening| night))\s*[.!…]*",re.I)
TECHNICAL = re.compile(r'(?:不要|必須|依存|既存|旧式|互換|非対応|標準)(?:な|の)?([ァ-ヺー]{3,})')
SELECTION = {COMMAND:'command_verb', LOAN:'technical_loanword', POLICY:'single_rendition',
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
        bare = source.strip().rstrip('。！!…')
        if (bare.endswith('なさい') and bare not in ('おかえりなさい','お帰りなさい','おやすみなさい','お休みなさい')
                and GREETING.fullmatch(target.strip()) and lexical_hint(source,COMMAND)):
            return COMMAND
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


def candidate_matches(policy, primary, candidate, source=''):
    if ' / ' in candidate:
        return False
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


def payload(original, policy=POLICY, source=''):
    from copy import deepcopy
    result=deepcopy(original)
    messages=result['messages']
    if len(messages)!=1 or messages[0]['role']!='user':
        raise ValueError('single-rendition retry requires one native specialist user message')
    text=messages[0]['content']
    marker='Please translate the following '
    if marker not in text:raise ValueError('unknown specialist prompt structure')
    if policy in (COMMAND,LOAN):
        hint = lexical_hint(source,policy)
        if not hint:
            raise ValueError('missing source dictionary entry')
        if policy==COMMAND:
            guide = f'The inflected verb before なさい has the Japanese dictionary form {hint}. Translate its action as one natural spoken command. Return only the dialogue, with no alternatives or explanation. '
        else:
            guide = (f'Preserve the computing term as “{hint}” in the English dialogue, '
                'including when the speaker uses it as a joke about a person. ')
    else:
        guide = {UNFINISHED:UNFINISHED_INSTRUCTION,WAIT:WAIT_INSTRUCTION,POLICY:INSTRUCTION}[policy]
    messages[0]['content']=text.replace(marker,guide+marker,1)
    return result
