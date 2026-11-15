"""Explicit collection faces and whole-target fallback; never system font lookup."""
import struct

from . import natural as n


def checksum(data):
    padded=data+b'\0'*((-len(data))%4)
    return sum(v[0] for v in struct.iter_unpack('>I',padded)) & 0xffffffff


def standalone(data, face_index=None):
    """Copy one explicitly indexed TTC face to bounded, checksummed sfnt bytes.

    TTC directory offsets address the collection. Internal table offsets remain
    table-relative. The copied tables retain their original glyphs and shaping;
    only directory offsets and head.checkSumAdjustment change.
    """
    if data[:4]!=b'ttcf':
        n.require(face_index is None,'face_index requires a font collection')
        return data
    try:
        version,count=struct.unpack_from('>II',data,4)
        n.require(version in (0x10000,0x20000) and 1<=count<=32,'font collection header/budget')
        n.require(type(face_index) is int and 0<=face_index<count,'explicit valid collection face_index required')
        header_end=12+4*count+(12 if version==0x20000 else 0)
        n.require(header_end<=len(data),'truncated collection header')
        offset=struct.unpack_from('>I',data,12+4*face_index)[0]
        n.require(header_end<=offset and offset+12<=len(data),'invalid collection face offset')
        signature,tables=struct.unpack_from('>4sH',data,offset)
        n.require(signature in (b'\x00\x01\x00\x00',b'OTTO') and 1<=tables<=128,'collection sfnt header/budget')
        n.require(offset+12+16*tables<=len(data),'truncated face directory')
        entries=[];tags=set();total=12+16*tables
        for i in range(tables):
            tag,_,start,length=struct.unpack_from('>4sIII',data,offset+12+16*i)
            n.require(tag not in tags and start>=header_end and start+length<=len(data),'invalid collection table')
            tags.add(tag);entries.append((tag,start,length));total+=(length+3)&~3
        n.budget_check(total<=32*1024*1024,'extracted font byte budget')
        n.require(b'head' in tags and b'fvar' not in tags,'collection needs a static head table')
        result=bytearray(data[offset:offset+12]);result.extend(b'\0'*(16*tables))
        head=None
        for i,(tag,start,length) in enumerate(sorted(entries)):
            content=bytearray(data[start:start+length]);target=len(result)
            if tag==b'head':
                n.require(length>=54,'invalid collection head')
                content[8:12]=b'\0'*4;head=target
            struct.pack_into('>4sIII',result,12+16*i,tag,checksum(content),target,length)
            result.extend(content);result.extend(b'\0'*((-length)%4))
        struct.pack_into('>I',result,head+8,(0xb1b0afba-checksum(result))&0xffffffff)
        return bytes(result)
    except (struct.error,IndexError) as error:
        raise ValueError('malformed font collection') from error


class ApprovedFaces:
    """Use the requested face unless it lacks glyphs; fallback order is explicit."""
    def __init__(self, manifest, requested_id):
        self.fonts=n.load_fonts(manifest)
        n.require(requested_id in self.fonts and not isinstance(self.fonts[requested_id],n.ResourceLimit),
            'unavailable explicit automatic font')
        fallbacks=manifest.get('fallbacks',{})
        n.require(type(fallbacks) is dict and len(fallbacks)<=16,'explicit font fallback map required')
        for key,values in fallbacks.items():
            n.require(key in self.fonts and type(values) is list and len(values)<=15
                and all(type(v) is str and v in self.fonts and v!=key for v in values)
                and len(set(values))==len(values),'invalid explicit font fallback list')
        self.requested=requested_id;self.ids=[requested_id,*fallbacks.get(requested_id,[])]
        self.primary=self.fonts[requested_id];self.cache={}

    def resolve(self,row):
        text=row['binding']['text']
        if text not in self.cache:
            for identity in self.ids:
                font=self.fonts[identity]
                if isinstance(font,n.ResourceLimit):continue
                if all(ord(c) in font[1] for c in text if c!='\n'):
                    n.validate_text(text,font[1])
                    self.cache[text]=identity;break
            n.require(text in self.cache,'missing glyph coverage in requested and explicit fallback faces')
        identity=self.cache[text];font=self.fonts[identity]
        row['font_selection']={'requested_id':self.requested,'selected_id':identity,
            'reason':'requested_complete_coverage' if identity==self.requested else 'explicit_fallback_complete_coverage',
            'font_sha256':n.digest(font[0]),'exact_target_preserved':True}
        return font
