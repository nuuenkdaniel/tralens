"""Corroborate a repeated sound in two disjoint halves of its source polygon."""
import unicodedata

from . import reader_geometry


def repeat(text):
    text=unicodedata.normalize('NFKC',text).strip()
    if not 4<=len(text)<=512 or not all('ァ'<=c<='ヺ' or c=='ー' for c in text):
        return None
    for length in range(2,5):
        motif=text[:length]
        if len(set(motif))>1 and len(text)>=2*length and motif*(len(text)//length)==text:
            return motif
    return None


def hint(record):
    if (len(record['polygons'])!=1 or record['direction'] not in ('unknown','horizontal')
            or min(record['scores'])>=.92 or len(record['readings'])!=3):
        return None
    votes=[]
    for text in record['readings']:
        if not isinstance(text,str) or len(text)<4:
            continue
        motif=repeat(text) or repeat(text[1:]) or repeat(text[:-1])
        # A damaged opening beat can change both kana. It can trigger the
        # window trial only when at least four intact beats follow it.
        if not motif:
            for length in range(2, 5):
                tail = text[length:]
                core = repeat(tail)
                if core and len(core) == length and len(tail) >= 4 * length:
                    motif = core
                    break
        if motif:
            votes.append(motif)
    return next((motif for motif in votes if votes.count(motif)>=2),None)


def presentations(image,record):
    if not hint(record):
        return None
    try:
        mapping=reader_geometry.plan(image.size,record['polygons'][0])
    except ValueError:
        return None
    width,height=mapping['seed_size']
    if width<1.4*height or height<18:
        return None
    line=reader_geometry.pixels(image,mapping)
    w,h=line.size
    mid=round(w/2)
    return line,line.crop((0,0,mid,h)),line.crop((mid,0,w,h))


def matches(record):
    motif=hint(record)
    if not motif:
        return False
    trial=record['sound_windows']
    if len(trial['readings'])!=3 or len(trial['view_hashes'])!=3:
        return False
    whole,left,right=[unicodedata.normalize('NFKC',t).strip() for t in trial['readings']]
    counts=[]
    for text in (whole,left,right):
        count=len(text)//len(motif)
        if not 1<=count<=16 or motif*count!=text:
            return False
        counts.append(count)
    full,a,b=counts
    return full>=2 and abs(a+b-full)<=max(1,round(full*.2)) and max(a,b)<=2*min(a,b)
