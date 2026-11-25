"""Build synthetic .psfxl data files for decoder tests.

The byte layout follows what Spectre writes, as observed across a corpus of
real transient results:

* Type-3 chunk: ``\\x003:<seq>:ffffffffffffffff:<npts>:<csize>:<flag>[:fields]\\n``,
  an alignment pad of ``8 - (pos % 8)`` zero bytes (1-8, never 0), then
  ``csize`` payload bytes; the next record starts right after the payload.
* A flush is seq 0 (time block immediately followed by signal 0's block),
  then seq 1..N-1 with the same point count.
* Seq-0 flag: two hex digits, time then signal 0 (``2`` blosc, ``1`` LZ4,
  ``0`` uncompressed); a flag of ``00`` is written as no flag at all.  The
  fields after it list each compressed block's size.
* Seq >= 1 flag: ``a`` + digit (``2`` blosc, ``1`` LZ4, ``0`` raw or an
  8-byte constant, ``a`` back-reference with a ``<back>.<length>`` field).
* Type-1: ``\\x001:0:0:<npts>:<bsize>\\n`` + pad + grouped doubles.
* Type-2: ``\\x002:0:0:<nt>:<npts>:<bsize>\\n`` + hex index table + pad +
  grouped doubles.
"""

import struct

import numpy as np

try:
    import blosc as _blosc

    def blosc_compress(raw):
        return _blosc.compress(raw, typesize=8)
except ImportError:  # numcodecs backend
    import numcodecs.blosc as _ncblosc

    def blosc_compress(raw):
        return bytes(_ncblosc.compress(raw, b'blosclz', 5, 1))


def lz4_literal_block(raw):
    """A valid raw LZ4 block holding *raw* as literals only."""
    n = len(raw)
    if n < 15:
        return bytes([n << 4]) + raw
    rest, ext = n - 15, bytearray()
    while rest >= 255:
        ext.append(255)
        rest -= 255
    ext.append(rest)
    return bytes([0xF0]) + bytes(ext) + raw


def lz4_repeat_block(a, b):
    """LZ4 block decoding to the doubles [a, a, a, b]: one literal, one
    overlapping match, then a final literal-only sequence."""
    lit = struct.pack('<d', a)
    return (bytes([0x8C]) + lit + struct.pack('<H', 8)     # 8 literals, match len 16
            + bytes([0x80]) + struct.pack('<d', b))         # final 8 literals


def f8(values):
    return np.asarray(values, dtype='<f8').tobytes()


class Builder:
    def __init__(self):
        self.buf = bytearray(b'\x00\x00\x00\x00PSFXL-synthetic\n')
        self.blocks = {}   # 'sig<i>' -> (offset, length) of its latest blosc block

    # -- low level -------------------------------------------------------
    def _pad(self):
        self.buf += b'\x00' * (8 - (len(self.buf) % 8))

    def t3(self, seq, npts, payload, flag, fields=()):
        hdr = ':'.join(['3', f'{seq:x}', 'ffffffffffffffff', f'{npts:x}',
                        f'{len(payload):x}'] + ([flag] if flag else []) + list(fields))
        offset = len(self.buf)
        self.buf += b'\x00' + hdr.encode() + b'\n'
        self._pad()
        start = len(self.buf)
        self.buf += payload
        return offset, start

    # -- blocks ----------------------------------------------------------
    @staticmethod
    def encode(values, enc):
        raw = f8(values)
        if enc == 'blosc':
            return '2', blosc_compress(raw)
        if enc == 'raw':
            return '0', raw
        if enc == 'const':
            return '0', f8(values[:1])
        if enc == 'lz4':
            return '1', lz4_literal_block(raw)
        raise ValueError(enc)

    # -- flushes ---------------------------------------------------------
    def flush(self, time, signals, enc=None):
        """Write one type-3 flush.  *enc* maps signal index -> encoding
        ('blosc', 'raw', 'const', 'lz4', or ('ref', name)); index 't' is the
        time block.  Unlisted blocks are blosc.  Returns chunk offsets."""
        enc = enc or {}
        npts = len(time)
        td, tp = self.encode(time, enc.get('t', 'blosc'))
        sd, sp = self.encode(signals[0], enc.get(0, 'blosc'))
        flag = td + sd
        sizes = [f'{len(p):x}' for d, p in ((td, tp), (sd, sp)) if d != '0']
        offsets = [self.t3(0, npts, tp + sp, '' if flag == '00' else flag, sizes)]
        if sd == '2':
            self.blocks['sig0'] = (offsets[0][1] + len(tp), len(sp))
        for i, sig in enumerate(signals[1:], start=1):
            e = enc.get(i, 'blosc')
            if isinstance(e, tuple) and e[0] == 'ref':
                src, length = self.blocks[e[1]]
                hdr_off = len(self.buf)
                offsets.append(self.t3(i, npts, b'', 'aa',
                                       ['0.0', f'{hdr_off - src:x}.{length:x}']))
                continue
            d, p = self.encode(sig, e)
            off, start = self.t3(i, npts, p, 'a' + d, ['0.0', f'{len(p):x}'])
            if d == '2':
                self.blocks[f'sig{i}'] = (start, len(p))
            offsets.append((off, start))
        return offsets

    def raw_t3(self, seq, npts, payload, flag, fields=()):
        """Write an arbitrary type-3 chunk (for malformed-input tests)."""
        return self.t3(seq, npts, payload, flag, fields)

    def type1(self, time, signals):
        npts = len(time)
        payload = f8(np.vstack([time] + list(signals)).ravel())
        self.buf += f'\x001:0:0:{npts:x}:{len(payload):x}\n'.encode()
        self._pad()
        self.buf += payload

    def type2(self, time, signals):
        npts = len(time)
        payload = f8(np.vstack([time] + list(signals)).ravel())
        self.buf += f'\x002:0:0:{len(signals):x}:{npts:x}:{len(payload):x}\n'.encode()
        self.buf += b'0\n10\n'          # the ASCII index table
        self._pad()
        self.buf += payload

    def end_marker(self):
        """The char(1) end-of-block record Spectre writes between flushes."""
        self.buf += b'\x01:0:0:1:1\n'

    def write(self, path):
        path.write_bytes(bytes(self.buf))
        return path
