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


def payload(original):
    from copy import deepcopy
    result=deepcopy(original)
    messages=result['messages']
    if len(messages)!=1 or messages[0]['role']!='user':
        raise ValueError('single-rendition retry requires one native specialist user message')
    text=messages[0]['content']
    marker='Please translate the following '
    if marker not in text:raise ValueError('unknown specialist prompt structure')
    messages[0]['content']=text.replace(marker,INSTRUCTION+marker,1)
    return result
