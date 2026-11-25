"""PSFXL-specific decoder for .tran/.tran.psfxl file pairs."""

import bisect
import struct
import threading
import warnings

import numpy as np
from packaging import version

_BLOSC_AVAILABLE = False
_NUMCODECS_AVAILABLE = False
_NUMCODECS_VERSION = None
_NUMCODECS_OLD_VERSION = False
_NUMCODECS_BLOSC = None  # direct reference to numcodecs.blosc module
try:
    import blosc
    _BLOSC_AVAILABLE = True
except ImportError:
    try:
        import numcodecs as _numcodecs
        import numcodecs.blosc as _NUMCODECS_BLOSC
        _NUMCODECS_AVAILABLE = True
        _NUMCODECS_VERSION = _numcodecs.__version__
        # Check if using older numcodecs (< 0.16.0) which had less robust decompression
        _NUMCODECS_OLD_VERSION = version.parse(_NUMCODECS_VERSION) < version.parse("0.16.0")
        # For older versions, c-blosc 1.21.0 (bundled with numcodecs 0.9.x) has a known
        # defect in blosc_decompress_ctx() where byte-shuffled float64 data is silently
        # mis-unshuffled when numinternalthreads=1, producing wildly wrong values (~8e188).
        #
        # The ctx path is triggered whenever _get_use_threads() returns False.
        # In old numcodecs, _get_use_threads() starts with:
        #     if not mutex: return False
        # where `mutex` is a module-level multiprocessing.Lock() set at import time.
        # On HPC/container systems where shared memory is restricted,
        # multiprocessing.Lock() raises OSError and mutex is left as None —
        # causing _get_use_threads() to unconditionally return False and always
        # use the buggy ctx path, regardless of set_nthreads() or use_threads.
        #
        # Fix: supply a threading.Lock() fallback when mutex is None so the
        # _get_use_threads() check passes, then force use_threads=True to
        # guarantee blosc_decompress() (the non-ctx, shuffle-correct path).
        if _NUMCODECS_OLD_VERSION:
            try:
                if getattr(_NUMCODECS_BLOSC, 'mutex', True) is None:
                    _NUMCODECS_BLOSC.mutex = threading.Lock()
            except Exception:
                pass
            try:
                _NUMCODECS_BLOSC.use_threads = True
            except Exception:
                pass
            # Limit to 1 internal thread to avoid global-state race conditions
            # while still using the non-ctx blosc_decompress() code path.
            try:
                _NUMCODECS_BLOSC.set_nthreads(1)
            except Exception:
                pass
    except ImportError:
        pass

# Fail immediately if neither blosc nor numcodecs is available
if not _BLOSC_AVAILABLE and not _NUMCODECS_AVAILABLE:
    raise ImportError(
         "Neither 'blosc' nor 'numcodecs' is installed. "
         "Install one of them to use PSFXL decoding: "
         "pip install blosc OR pip install numcodecs"
     )


def _blosc_decompress(buf):
    """Decompress a blosc1 buffer.

    Uses the 'blosc' package if available, otherwise falls back to 'numcodecs'
    which also bundles c-blosc and supports Python 3.6+.
    buf must be the full blosc chunk starting at the blosc magic bytes,
    of exactly cbytes length (bytes 12-15 of the header, little-endian uint32).

    Robustness notes for older numcodecs (< 0.16.0, especially 0.9.x):
    - c-blosc 1.21.0 (bundled with numcodecs 0.9.x) has a defect in
      blosc_decompress_ctx() where byte-shuffled data is incorrectly
      unshuffled when numinternalthreads=1, producing wildly wrong float values
      (~8×10^188). The ctx path is triggered when _get_use_threads() returns
      False — which happens unconditionally when `mutex is None` (common on
      HPC/container systems with restricted shared memory).
    - The fix applied at import time: inject a threading.Lock() into the
      module-level mutex so the _get_use_threads() check passes, then force
      use_threads=True to always take the non-ctx blosc_decompress() path,
      which handles shuffle correctly. set_nthreads(1) limits internal threads.
    - Uses module-level numcodecs.blosc.decompress() which is more stable than
      Blosc().decode() across all versions and avoids ndarray return-type surprises.
    - Passes a bytearray copy of the buffer, which is universally accepted by all
      numcodecs versions and avoids memoryview/slice incompatibilities.
    """
    if _BLOSC_AVAILABLE:
        return blosc.decompress(buf)

    if _NUMCODECS_AVAILABLE:
        # Pre-read the expected decompressed size from the blosc header.
        # Header layout (bytes): version(1) versionlz(1) flags(1) typesize(1)
        #                        nbytes(4LE) blocksize(4LE) cbytes(4LE)
        # nbytes is at offset 4, giving the uncompressed byte count.
        if len(buf) >= 16:
            expected_nbytes = struct.unpack_from('<I', buf, 4)[0]
        else:
            expected_nbytes = None

        # Re-apply the old-numcodecs mutex/use_threads fix (see the module
        # import block for the full explanation) before every call: other code
        # may have reset the module state, and the fix is what keeps
        # decompression off the shuffle-buggy blosc_decompress_ctx() path.
        if _NUMCODECS_OLD_VERSION:
            try:
                if getattr(_NUMCODECS_BLOSC, 'mutex', True) is None:
                    _NUMCODECS_BLOSC.mutex = threading.Lock()
                if _NUMCODECS_BLOSC.use_threads is not True:
                    _NUMCODECS_BLOSC.use_threads = True
            except Exception:
                pass

        # Convert to bytearray: universally accepted by all numcodecs versions
        # and avoids issues with slices returning memoryview in some Python builds.
        buf_ba = bytearray(buf)

        # Use the module-level decompress function which is available in all
        # numcodecs versions and is more robust than Blosc().decode() in 0.9.x.
        try:
            result = _NUMCODECS_BLOSC.decompress(buf_ba)
        except Exception as e:
            if _NUMCODECS_OLD_VERSION:
                warnings.warn(
                    f"numcodecs.blosc.decompress failed with numcodecs "
                    f"{_NUMCODECS_VERSION}: {e}. "
                    f"Retrying via Blosc().decode(). "
                    f"Consider upgrading to numcodecs >= 0.16.0.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                # Fallback: try the codec object path
                try:
                    result = _numcodecs.Blosc().decode(buf_ba)
                except Exception as e2:
                    raise RuntimeError(
                        f"Both numcodecs.blosc.decompress and Blosc().decode failed "
                        f"with numcodecs {_NUMCODECS_VERSION}: first={e}, second={e2}"
                    ) from e2
            else:
                raise RuntimeError(
                    f"Error during blosc decompression with numcodecs "
                    f"{_NUMCODECS_VERSION}: {e}"
                ) from e

        # Normalise return type: older versions may return ndarray or memoryview.
        if isinstance(result, bytes):
            out = result
        elif isinstance(result, (bytearray, memoryview)):
            out = bytes(result)
        else:
            # numpy ndarray or other buffer-protocol object
            out = bytes(memoryview(result).cast('B'))

        # Validate decompressed length against the header's nbytes field.
        # A length mismatch in numcodecs 0.9.x indicates a truncation edge-case
        # in c-blosc 1.21.0, separate from the shuffle bug addressed at import.
        if expected_nbytes is not None and len(out) != expected_nbytes:
            warnings.warn(
                f"Blosc decompressed {len(out)} bytes but header nbytes={expected_nbytes}. "
                f"numcodecs {_NUMCODECS_VERSION} may have a truncation bug. "
                f"Consider upgrading to numcodecs >= 0.16.0.",
                RuntimeWarning,
                stacklevel=3,
            )
            # If we got more bytes than expected, truncate to expected (safe).
            # If we got fewer, the caller will detect the mismatch via num_points check.
            if len(out) > expected_nbytes:
                out = out[:expected_nbytes]

        return out

    raise RuntimeError(
        "No decompression backend available. "
        "Install blosc (pip install blosc) or numcodecs (pip install numcodecs)."
    )


def parse_sig_file(sig_path):
    """Parse a .tran.sig metadata file.

    The .sig file is a small text companion to the .psfxl data file holding
    key/value metadata lines (sweep count, flush count, compression type).

    Args:
        sig_path: Path to the .tran.sig file.

    Returns:
        dict mapping each metadata key to its string value.
    """
    with open(sig_path, 'rb') as f:
        raw = f.read()

    metadata = {}
    for line in raw.split(b'\n'):
        line = line.strip(b'\x01').strip()
        if line.startswith(b'*'):
            parts = line.decode('ascii', errors='replace').split(None, 2)
            if len(parts) >= 3:
                key = parts[0].lstrip('*')
                metadata[key] = parts[2]
    return metadata


try:  # optional fast path; the pure-Python decoder below is the fallback
    import lz4.block as _lz4_block
except ImportError:
    _lz4_block = None


def _lz4_decompress_block(src, out_len):
    """Decompress one raw LZ4 *block* (no frame header) to exactly *out_len* bytes.

    Spectre stores a few short type-3 blocks this way (flag digit ``1``).
    Raises ValueError if the block is malformed or decodes to another length.
    """
    if _lz4_block is not None:
        out = _lz4_block.decompress(bytes(src), uncompressed_size=out_len)
        if len(out) != out_len:
            raise ValueError(f"LZ4 block decoded to {len(out)} bytes, expected {out_len}")
        return out
    return _lz4_pure(bytes(src), out_len)


def _lz4_pure(src, out_len):
    """Pure-Python LZ4 block decoder: exactly *out_len* bytes or ValueError."""
    try:
        return _lz4_pure_unchecked(src, out_len)
    except IndexError:  # ran off the end of the block mid-sequence
        raise ValueError("LZ4 block is truncated") from None


def _lz4_pure_unchecked(src, out_len):
    """Decode loop; never produces more than *out_len* bytes."""
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        if len(out) + lit > out_len or i + lit > n:
            raise ValueError("LZ4 literal run exceeds the block or the output size")
        out += src[i:i + lit]
        i += lit
        if i >= n:
            break  # the last sequence carries literals only
        off = src[i] | (src[i + 1] << 8)
        i += 2
        mlen = token & 15
        if mlen == 15:
            while True:
                b = src[i]
                i += 1
                mlen += b
                if b != 255:
                    break
        mlen += 4
        start = len(out) - off
        if off == 0 or start < 0:
            raise ValueError("LZ4 block has an invalid match offset")
        if len(out) + mlen > out_len:
            raise ValueError("LZ4 match exceeds the output size")
        if off >= mlen:
            out += out[start:start + mlen]
        else:  # overlapping match: each byte may be one just written
            for k in range(mlen):
                out.append(out[start + k])
    if len(out) != out_len:
        raise ValueError(f"LZ4 block decoded to {len(out)} bytes, expected {out_len}")
    return bytes(out)


class _ChunkError(Exception):
    """A type-3 block that cannot be decoded; the message says why."""


class _Chunk:
    """One parsed type-3 chunk header (``\\x003:seq:?:npts:csize:flag:...``)."""
    __slots__ = ('offset', 'data_start', 'payload', 'seq', 'num_points',
                 'chunk_size', 'flag', 'fields')

    def __init__(self, offset, data_start, seq, num_points, chunk_size, flag, fields):
        self.offset = offset            # position of the leading \x00
        self.data_start = data_start    # first byte after the header's \n
        # Payload follows an alignment pad of 8 - (data_start % 8) bytes:
        # always 1-8, never 0, emitted even when data_start is aligned.
        self.payload = data_start + 8 - (data_start % 8)
        self.seq = seq
        self.num_points = num_points
        self.chunk_size = chunk_size
        self.flag = flag
        self.fields = fields            # header fields after the flag, as text

    @property
    def ref(self):
        """The ``<back_offset>.<length>`` field of a back-reference chunk."""
        return self.fields[1] if len(self.fields) > 1 else ''

    def compressed_sizes(self):
        """Sizes listed after the flag of a seq-0 chunk, one per compressed block."""
        sizes = []
        for f in self.fields:
            if '.' in f:
                break
            try:
                sizes.append(int(f, 16))
            except ValueError:
                break
        return sizes


def _scan_chunk_headers(data):
    """Find every type-3 chunk header, walking the file from chunk to chunk.

    A header is ``\\x003:seq:?:num_points:chunk_size:flag[:field...]\\n``
    followed by an alignment pad and ``chunk_size`` payload bytes; the next
    header (of any type) starts after that payload.  The search for the next
    header therefore resumes at the end of the current payload, so a
    ``\\x003:`` byte sequence that happens to occur inside compressed or raw
    data is never mistaken for a header.

    Returns a list of :class:`_Chunk`, in file order.
    """
    chunks = []
    pos = 0
    while True:
        idx = data.find(b'\x003:', pos)
        if idx == -1:
            break
        try:
            nl = data.index(b'\n', idx, idx + 300)
            parts = data[idx + 1:nl].decode('ascii').split(':')
            seq = int(parts[1], 16)
            num_points = int(parts[3], 16)
            chunk_size = int(parts[4].split('.')[0], 16) if len(parts) > 4 else 0
        except (ValueError, IndexError, UnicodeDecodeError):
            pos = idx + 1
            continue
        chunk = _Chunk(idx, nl + 1, seq, num_points, chunk_size,
                       parts[5] if len(parts) > 5 else '', parts[6:])
        end = chunk.payload + chunk_size if chunk_size else chunk.data_start
        if end > len(data):
            # Declared payload runs past EOF: a truncated final chunk, or a
            # false match.  Keep it (decoding will report it) but do not jump.
            chunks.append(chunk)
            pos = idx + 1
            continue
        chunks.append(chunk)
        pos = max(end, idx + 1)
    return chunks


def _decode_blosc(data, start, size, num_points):
    """Decode a blosc block of exactly *size* bytes at *start*."""
    if start + 16 > len(data) or start + size > len(data):
        raise _ChunkError("blosc block runs past end of file")
    if data[start] != 2:
        raise _ChunkError(f"no blosc header (version byte 0x{data[start]:02x})")
    nbytes, _blocksize, cbytes = struct.unpack_from('<3I', data, start + 4)
    if nbytes != num_points * 8:
        raise _ChunkError(f"blosc nbytes {nbytes} != {num_points} points * 8")
    if cbytes != size:
        raise _ChunkError(f"blosc cbytes {cbytes} != block size {size}")
    try:
        out = _blosc_decompress(data[start:start + size])
    except Exception as exc:
        raise _ChunkError(f"blosc decompression failed: {exc}") from exc
    if len(out) != num_points * 8:
        raise _ChunkError(f"blosc produced {len(out)} bytes, expected {num_points * 8}")
    return np.frombuffer(out, dtype='<f8')


def _decode_block(data, start, size, digit, num_points):
    """Decode one block of *size* bytes at *start* encoded per flag *digit*.

    ``2`` = blosc, ``1`` = raw LZ4 block, ``0`` = uncompressed: either an
    array of ``num_points`` little-endian doubles or, when the block is
    8 bytes, one double repeated (a constant signal).
    """
    if start + size > len(data):
        raise _ChunkError("block runs past end of file")
    if digit == '2':
        return _decode_blosc(data, start, size, num_points)
    if digit == '1':
        try:
            out = _lz4_decompress_block(data[start:start + size], num_points * 8)
        except Exception as exc:
            raise _ChunkError(f"LZ4 decompression failed: {exc}") from exc
        return np.frombuffer(out, dtype='<f8')
    if digit == '0':
        if size == num_points * 8:
            return np.frombuffer(data[start:start + size], dtype='<f8')
        if size == 8:
            return np.full(num_points, struct.unpack_from('<d', data, start)[0])
        raise _ChunkError(f"uncompressed block of {size} bytes is neither "
                          f"{num_points} doubles nor one constant")
    raise _ChunkError(f"unknown encoding digit {digit!r}")


def _decode_seq0(data, chunk):
    """Decode a seq-0 chunk into ``(time, signal_0)``.

    The payload is the time block immediately followed (no pad) by signal
    0's block.  The flag's two hex digits give their encodings, time first;
    a missing flag means both are uncompressed.  The fields after the flag
    list the size of each compressed block in order.  Raises _ChunkError if
    the time block cannot be decoded; a signal-0 failure is returned as the
    exception in place of the array.
    """
    npts = chunk.num_points
    flag = (chunk.flag or '0').zfill(2)
    if len(flag) != 2:
        raise _ChunkError(f"unknown seq-0 flag {chunk.flag!r}")
    t_digit, s_digit = flag
    sizes = chunk.compressed_sizes()
    if chunk.chunk_size == 0:
        raise _ChunkError("seq-0 chunk has zero size")

    if t_digit == '0':
        t_size = npts * 8
    elif t_digit == '2' and chunk.payload + 16 <= len(data):
        t_size = struct.unpack_from('<I', data, chunk.payload + 12)[0]
    elif t_digit in '12' and sizes:
        t_size = sizes[0]
    else:
        raise _ChunkError(f"cannot size time block (flag {chunk.flag!r})")
    if t_digit in '12' and sizes and sizes[0] != t_size:
        raise _ChunkError(f"time block is {t_size} bytes but the header lists {sizes[0]}")
    s_size = chunk.chunk_size - t_size
    if s_size <= 0:
        raise _ChunkError(f"time block ({t_size} bytes) fills the {chunk.chunk_size}-byte chunk")

    time = _decode_block(data, chunk.payload, t_size, t_digit, npts)
    if npts > 1 and np.any(np.diff(time) < 0):
        raise _ChunkError("decoded time block is not increasing")
    try:
        sig0 = _decode_block(data, chunk.payload + t_size, s_size, s_digit, npts)
    except _ChunkError as exc:
        sig0 = exc
    return time, sig0


def _resolve_back_reference(data, chunk):
    """Decode the block a zero-size chunk points at.

    A zero-size chunk is NOT a run of zeros. Spectre writes it when this
    signal's data for this flush is byte-identical to a block already in the
    file (two clock phases with the same duty, a bus whose bits carry the
    same code) and points at that block with a trailing
    ``<back_offset>.<length>`` field: the block starts ``back_offset`` bytes
    before this chunk header and its compressed size is ``length``.
    """
    try:
        back_s, length_s = chunk.ref.split('.')[:2]
        back, length = int(back_s, 16), int(length_s, 16)
    except ValueError:
        raise _ChunkError(f"zero-size chunk with unreadable reference {chunk.ref!r}")
    src = chunk.offset - back
    if src < 0:
        raise _ChunkError(f"back-reference {chunk.ref!r} points before the file start")
    return _decode_blosc(data, src, length, chunk.num_points)


def _decode_seqn(data, chunk):
    """Decode a seq >= 1 chunk (one signal) per the low digit of its flag."""
    digit = chunk.flag[-1:] or '0'
    if chunk.chunk_size == 0:
        if digit not in ('a', ''):
            raise _ChunkError(f"zero-size chunk with flag {chunk.flag!r}")
        return _resolve_back_reference(data, chunk)
    return _decode_block(data, chunk.payload, chunk.chunk_size, digit, chunk.num_points)


class _Problems:
    """Collects decode problems so they can be reported in one warning."""

    def __init__(self):
        self.items = []   # (offset, seq, reason)

    def add(self, offset, seq, reason):
        self.items.append((offset, seq, str(reason)))

    def warn(self, data_path, what):
        if not self.items:
            return
        shown = '; '.join(f"seq {s} at offset {o}: {r}" for o, s, r in self.items[:5])
        more = f"; and {len(self.items) - 5} more" if len(self.items) > 5 else ''
        warnings.warn(f"psfxl: {len(self.items)} {what} in {data_path} ({shown}{more})",
                      RuntimeWarning, stacklevel=3)


def _decode_type3_flushes(data, num_traces, chunks, bad_chunks, dropped):
    """Decode type-3 chunks into flushes.

    A flush is a seq-0 chunk (time + signal 0) followed by seq 1..N-1, all
    with the same number of points.  Returns a list of
    ``(offset, 't3', time, signals)`` with one array per signal index; a
    signal whose block is missing or undecodable is NaN for that flush so
    that every signal stays aligned with the time axis.
    """
    flushes = []
    current = None  # (chunk0, time, {seq: array})
    in_dropped_flush = False

    def finish():
        if current is None:
            return
        c0, time, sigs = current
        for i in range(num_traces):
            if i not in sigs:
                bad_chunks.add(c0.offset, i, "block missing from flush")
                sigs[i] = np.full(len(time), np.nan)
        flushes.append((c0.offset, 't3', time, sigs))

    for chunk in chunks:
        if chunk.num_points == 0:
            continue
        if chunk.seq == 0:
            finish()
            current = None
            in_dropped_flush = False
            try:
                time, sig0 = _decode_seq0(data, chunk)
            except _ChunkError as exc:
                dropped.add(chunk.offset, 0, exc)
                in_dropped_flush = True
                continue
            sigs = {}
            if isinstance(sig0, Exception):
                bad_chunks.add(chunk.offset, 0, sig0)
                sigs[0] = np.full(len(time), np.nan)
            else:
                sigs[0] = sig0
            current = (chunk, time, sigs)
            continue
        if current is None:
            # Belongs to a flush whose seq-0 chunk failed (already reported)
            # or that has no seq-0 chunk at all.
            if not in_dropped_flush:
                dropped.add(chunk.offset, chunk.seq, "chunk outside any flush")
            continue
        c0, time, sigs = current
        if chunk.seq >= num_traces:
            bad_chunks.add(chunk.offset, chunk.seq,
                           f"signal index beyond the {num_traces} traces in the header")
            continue
        if chunk.num_points != len(time):
            bad_chunks.add(chunk.offset, chunk.seq,
                           f"{chunk.num_points} points in a {len(time)}-point flush")
            continue
        if chunk.seq in sigs:
            bad_chunks.add(chunk.offset, chunk.seq, "duplicate block in flush")
            continue
        try:
            sigs[chunk.seq] = _decode_seqn(data, chunk)
        except _ChunkError as exc:
            bad_chunks.add(chunk.offset, chunk.seq, exc)
            sigs[chunk.seq] = np.full(len(time), np.nan)
    finish()
    return flushes


def _merge_intervals(intervals):
    """Sort and merge ``(start, end)`` intervals so they do not overlap."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _inside(intervals, pos):
    """True if *pos* lies inside one of the merged ``(start, end)`` intervals."""
    i = bisect.bisect_right(intervals, (pos, float('inf'))) - 1
    return i >= 0 and intervals[i][0] <= pos < intervals[i][1]


def _find_type2_chunks(data):
    """Find type-2 chunk headers (uncompressed flush chunks).

    Type-2 format: \\x002:seq:?:num_traces:num_points:binary_size\\n
    Followed by an ASCII hex index table, an alignment pad, then raw
    little-endian IEEE 754 doubles laid out as (num_traces+1) groups of
    num_points values (time + each signal).

    The pad after the index table is ``8 - (pos % 8)`` bytes -- 1-8, never
    0 -- like the type-1 pad.  It must be computed, not found by skipping
    null bytes: a first time sample of 0.0 is eight null bytes and would be
    consumed as padding, shifting every signal by one sample.

    Returns ``(doubles_start, num_traces, num_points, binary_size)`` tuples.
    """
    chunks = []
    pos = 0
    while True:
        idx = data.find(b'\x002:', pos)
        if idx == -1:
            break
        try:
            nl = data.index(b'\n', idx, idx + 300)
            text = data[idx+1:nl].decode('ascii', errors='replace')
            parts = text.split(':')
            if len(parts) >= 6:
                num_traces_c = int(parts[3], 16)
                num_points = int(parts[4], 16)
                binary_size = int(parts[5], 16)
                expected = (num_traces_c + 1) * num_points * 8
                if binary_size == expected and num_points > 0:
                    # The index table is printable text and newlines.
                    scan_pos = nl + 1
                    while scan_pos < len(data):
                        b = data[scan_pos]
                        if (b < 0x20 and b != 0x0a) or b > 0x7e:
                            break
                        scan_pos += 1
                    doubles_start = scan_pos + 8 - (scan_pos % 8)
                    if doubles_start + binary_size <= len(data):
                        chunks.append((doubles_start, num_traces_c, num_points,
                                       binary_size))
            pos = nl + 1 if nl > idx else idx + 1
        except (ValueError, IndexError):
            pos = idx + 1
    return chunks


def _find_type1_chunks(data):
    """Find type-1 chunk headers (uncompressed, no index table).

    Type-1 format: \\x001:seq:?:num_points:binary_size\\n
    Followed by an 8-byte alignment pad, then raw little-endian IEEE 754
    doubles laid out as (num_traces+1) groups of num_points values
    (time + each signal) -- the same grouped payload as type-2.

    Spectre writes a type-1 chunk for a flush that compresses badly enough
    that it does not use the per-signal blosc-compressed type-3 blocks, so a
    file can mix them: in many runs the first flush is type-1 and every
    later one type-3.

    Two differences from type-2, both of which corrupt a decode if missed:

    1. There is no ``num_traces`` field -- the header has five fields, not
       six -- so it is derived: ``binary_size / 8 / num_points - 1``.

    2. There is no ASCII index table, but there IS an alignment pad of
       ``8 - (offset % 8)`` bytes: always 1-8, never 0, emitted even when the
       offset is already 8-byte aligned.  Do *not* locate the payload by
       scanning past null bytes: the first payload value is the first time
       point, which is legitimately ``0.0`` and therefore eight null bytes
       indistinguishable from padding.

    Returns ``(doubles_start, num_traces, num_points, binary_size)`` tuples
    shaped to match :func:`_find_type2_chunks`.
    """
    chunks = []
    pos = 0
    while True:
        idx = data.find(b'\x001:', pos)
        if idx == -1:
            break
        try:
            nl = data.index(b'\n', idx, idx + 300)
            parts = data[idx+1:nl].decode('ascii', errors='replace').split(':')
            if len(parts) >= 5:
                num_points = int(parts[3], 16)
                binary_size = int(parts[4], 16)
                stride = 8 * num_points
                if num_points > 0 and stride and binary_size % stride == 0:
                    groups = binary_size // stride
                    after_nl = nl + 1
                    align_skip = 8 - (after_nl % 8)   # always 1-8, never 0
                    doubles_start = after_nl + align_skip
                    if groups >= 2 and doubles_start + binary_size <= len(data):
                        # Structural check: the first group is the time axis.
                        # Require it to be non-decreasing AND to have no
                        # duplicated leading sample -- an off-by-one in the
                        # pad shows up exactly as t[0] == t[1], and a plain
                        # monotonicity test does not catch it.
                        t = np.frombuffer(data[doubles_start:doubles_start + stride],
                                          dtype='<f8')
                        if (t.size == num_points and np.all(np.diff(t) >= 0)
                                and (num_points < 2 or t[0] != t[1])):
                            chunks.append((doubles_start, groups - 1,
                                           num_points, binary_size))
            pos = nl + 1 if nl > idx else idx + 1
        except (ValueError, IndexError):
            pos = idx + 1
    return chunks


def _grouped_flushes(data, found, kind, num_traces, bad_chunks):
    """Turn type-1/type-2 chunk tuples into ``(offset, kind, time, signals)``."""
    flushes = []
    for doubles_start, nt, npts, bsize in found:
        groups = np.frombuffer(data[doubles_start:doubles_start + bsize],
                               dtype='<f8').reshape(nt + 1, npts)
        sigs = {}
        for i in range(num_traces):
            if i < nt:
                sigs[i] = groups[1 + i]
            else:
                sigs[i] = np.full(npts, np.nan)
        if nt < num_traces:
            bad_chunks.add(doubles_start, nt,
                           f"type-{kind[1]} flush holds {nt} of {num_traces} signals")
        flushes.append((doubles_start, kind, groups[0], sigs))
    return flushes


def decode_psfxl(data_path, num_traces):
    """Decode the .psfxl waveform data file.

    Args:
        data_path: Path to the .tran.psfxl file
        num_traces: Number of trace signals (from header parsing)

    Returns:
        (time_array, signal_arrays, flush_count) where signal_arrays is a dict
        mapping signal index (0..num_traces-1) to numpy arrays, each the same
        length as time_array.  Samples that could not be decoded are NaN, and
        a RuntimeWarning says which chunks were affected.

    The file is a sequence of flushes, each stored in one of three
    containers: type-3 (one block per signal, each blosc, LZ4, uncompressed
    or a back-reference, as the chunk's flag says), type-1 or type-2 (all
    signals uncompressed in one block).  A file can mix them.  Flushes are
    assembled by start time, which in real files is also file order; a flush
    whose time range is already covered by earlier data (a stale checkpoint)
    is skipped, and one that overlaps is trimmed to its new samples.
    """
    with open(data_path, 'rb') as f:
        data = f.read()

    bad_chunks = _Problems()   # signal blocks replaced by NaN
    dropped = _Problems()      # whole flushes lost (time block undecodable)

    # Each container's marker bytes can also occur by chance inside another
    # container's payload.  Type-3 payloads (that end inside the file) mask
    # type-1/type-2 candidates; the type-1/type-2 flushes that survive then
    # mask type-3 headers inside their payloads.
    chunks = _scan_chunk_headers(data)
    t3_payloads = _merge_intervals(
        (c.payload, c.payload + c.chunk_size) for c in chunks
        if c.chunk_size and c.payload + c.chunk_size <= len(data))
    grouped = []
    masked = 0
    for kind, found in (('t2', _find_type2_chunks(data)), ('t1', _find_type1_chunks(data))):
        keep = []
        for chunk in found:
            if _inside(t3_payloads, chunk[0]):
                masked += 1
            else:
                keep.append(chunk)
        grouped.append((kind, keep))
    grouped_payloads = _merge_intervals(
        (c[0], c[0] + c[3]) for _, found in grouped for c in found)
    chunks = [c for c in chunks if not _inside(grouped_payloads, c.offset)]

    flushes = _decode_type3_flushes(data, num_traces, chunks, bad_chunks, dropped)
    for kind, found in grouped:
        flushes += _grouped_flushes(data, found, kind, num_traces, bad_chunks)
    # Assemble in time order, which in every file examined is also file order;
    # on equal start times prefer a type-3 flush, then the longer one.  Sorting
    # by time (not offset) keeps a checkpoint record that appears out of order
    # from displacing the flush that covers it.
    flushes.sort(key=lambda fl: (fl[2][0] if len(fl[2]) else np.inf,
                                 fl[1] != 't3', -len(fl[2]), fl[0]))

    all_time = []
    all_signals = {i: [] for i in range(num_traces)}
    end = -np.inf
    covered = []    # (kind, first time) of flushes skipped as already covered
    trimmed = 0
    for offset, kind, time, sigs in flushes:
        if len(time) == 0:
            continue
        if time[-1] <= end:
            covered.append((kind, time[0]))
            continue
        keep = slice(None)
        if time[0] <= end:
            first = int(np.searchsorted(time, end, side='right'))
            keep = slice(first, None)
            trimmed += 1
        all_time.append(time[keep])
        for i in range(num_traces):
            all_signals[i].append(sigs[i][keep])
        end = time[-1]

    bad_chunks.warn(data_path, "signal block(s) could not be decoded and are NaN")
    dropped.warn(data_path, "flush(es) were dropped because their time block "
                            "could not be decoded")
    if covered or trimmed:
        kinds = ', '.join(f"{sum(1 for k, _ in covered if k == kind)} {kind}"
                          for kind in ('t1', 't2', 't3')
                          if any(k == kind for k, _ in covered))
        warnings.warn(
            f"psfxl: {len(covered)} flush(es) ({kinds or 'none'}) were skipped "
            f"and {trimmed} trimmed in {data_path} because their simulation "
            f"time was already covered by earlier data in the file (first "
            f"skipped times: {', '.join(f'{t:.4e} s' for _, t in covered[:5])}). "
            f"Checkpoint records of an interrupted simulation look like this.",
            UserWarning, stacklevel=2)
    if masked:
        warnings.warn(
            f"psfxl: {masked} type-1/type-2 record header(s) in {data_path} lie "
            f"inside type-3 payloads and were ignored as chance byte matches.",
            UserWarning, stacklevel=2)

    if not all_time:
        # Returning an empty array quietly is the worst outcome: the caller
        # sees a well-formed PSF with correct signal names and every ordinate
        # empty, and computes on it.
        warnings.warn(
            "psfxl: no decodable waveform chunks found in {} ({} bytes). "
            "The file was parsed but contained no type-1, type-2 or type-3 "
            "data block this decoder recognises, so the result is empty "
            "rather than wrong. Check the chunk markers before trusting "
            "any measurement taken from it.".format(data_path, len(data)),
            RuntimeWarning, stacklevel=2)
        return np.array([]), {}, 0

    time_full = np.concatenate(all_time)
    signals = {i: np.concatenate(all_signals[i]) for i in range(num_traces)}
    return time_full, signals, len(all_time)
