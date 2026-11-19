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


def retry_policy(source, target, style):
    if needs_retry(source,target):
        return POLICY
    if (style.source_language in ('Japanese','ja') and style.target_language in ('English','en')
            and len(source.strip()) >= 5 and source.rstrip().endswith(('だが','ですが','けれど','けれども'))
            and target.strip().endswith('.') and not target.strip().endswith('...')
            and not CONTRAST.search(target)):
        return UNFINISHED
    return None


def candidate_matches(policy, primary, candidate):
    if ' / ' in candidate:
        return False
    if policy != UNFINISHED:
        return True
    # Only take a response that actually restores an open connective and
    # preserves any numeric information from the primary translation.
    return (bool(re.search(r'\b(?:but|though|however|yet)\s*(?:…|\.{3})$',candidate.strip(),re.I))
        and re.findall(r'\d+(?:[.,]\d+)*',primary) == re.findall(r'\d+(?:[.,]\d+)*',candidate))


def payload(original, policy=POLICY):
    from copy import deepcopy
    result=deepcopy(original)
    messages=result['messages']
    if len(messages)!=1 or messages[0]['role']!='user':
        raise ValueError('single-rendition retry requires one native specialist user message')
    text=messages[0]['content']
    marker='Please translate the following '
    if marker not in text:raise ValueError('unknown specialist prompt structure')
    messages[0]['content']=text.replace(marker,(UNFINISHED_INSTRUCTION if policy==UNFINISHED else INSTRUCTION)+marker,1)
    return result
