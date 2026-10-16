"""Original synthetic sfnt QA font, constructed offline; not a dialogue-quality font.

ASCII glyphs are rectangles with descenders/sidebearings; A/V have a kern pair.
No installed/system font selection, font download or third-party font bytes.
"""
import struct


def pack(fmt, *values):
    return struct.pack('>'+fmt, *values)


def synthetic_font():
    count = 96  # notdef + printable ASCII
    tables = {}
    tables[b'head'] = pack('IIIIHHqqhhhhHHhhh', 0x10000, 0x10000, 0, 0x5f0f3cf5,
        0, 1000, 0, 0, -50, -150, 550, 700, 0, 8, 2, 1, 0)
    tables[b'hhea'] = pack('IhhhHhhhhhhhhhhhH', 0x10000, 800, -200, 0, 650,
        -50, 0, 600, 1, 0, 0, 0, 0, 0, 0, 0, count)
    tables[b'maxp'] = pack('IH'+'H'*13, 0x10000, count, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0)
    glyf = bytearray()
    offsets = []
    for gid in range(count):
        offsets.append(len(glyf))
        if gid == 1:  # space has advance, no ink
            continue
        # One contour: rectangle, signed coordinate deltas, all on-curve.
        glyf += pack('hhhhhHH', 1, -50, -150, 500, 700, 3, 0)
        glyf += bytes([1]*4)
        glyf += pack('hhhh', -50, 550, 0, -550)
        glyf += pack('hhhh', -150, 0, 850, 0)
        while len(glyf) % 4:
            glyf += b'\0'
    offsets.append(len(glyf))
    tables[b'glyf'] = bytes(glyf)
    tables[b'loca'] = b''.join(pack('I', n) for n in offsets)
    tables[b'hmtx'] = b''.join(pack('Hh', 300 if gid == 1 else 650, 0 if gid == 1 else -50) for gid in range(count))
    # Unicode BMP cmap 4, one ASCII segment plus sentinel.
    sub = pack('HHHHHHH', 4, 32, 0, 4, 4, 1, 0)
    sub += pack('HHH', 126, 65535, 0)+pack('HH', 32, 65535)+pack('hh', -31, 1)+pack('HH', 0, 0)
    tables[b'cmap'] = pack('HHHHI', 0, 1, 3, 1, 12)+sub
    tables[b'name'] = pack('HHH', 0, 0, 6)
    tables[b'post'] = pack('IIhhIIIII', 0x30000, 0, -100, 50, 0, 0, 0, 0, 0)
    pair = pack('HHh', ord('A')-31, ord('V')-31, -150)
    tables[b'kern'] = pack('HH', 0, 1)+pack('HHHHHHH', 0, 20, 1, 1, 6, 0, 0)+pair
    def checksum(data):
        data += b'\0'*((-len(data)) % 4)
        return sum(struct.unpack('>'+'I'*(len(data)//4), data)) & 0xffffffff
    tags = sorted(tables)
    header = pack('IHHHH', 0x10000, len(tags), 128, 3, 16*len(tags)-128)
    directory, body = bytearray(), bytearray()
    head_offset = 0
    for tag in tags:
        data = tables[tag]
        offset = len(header)+16*len(tags)+len(body)
        directory += pack('4sIII', tag, checksum(data), offset, len(data))
        if tag == b'head':
            head_offset = offset
        body += data+b'\0'*((-len(data)) % 4)
    font = bytearray(header+directory+body)
    struct.pack_into('>I', font, head_offset+8, (0xb1b0afba-checksum(bytes(font))) & 0xffffffff)
    return bytes(font)
