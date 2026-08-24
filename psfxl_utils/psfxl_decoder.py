"""PSFXL-specific decoder for .tran/.tran.psfxl file pairs."""

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


BLOSC_MAGIC = b'\x02\x01\x01\x08'

# Maximum expected blosc overhead (128 bytes generously covers all versions)
_BLOSC_MAX_OVERHEAD = 128


def _find_blosc_header(data, start, end, expected_nbytes=None):
    """Locate a valid blosc1 header in data[start:end].

    This is more robust than searching for the fixed BLOSC_MAGIC pattern
    (b'\\x02\\x01\\x01\\x08') because that pattern encodes specific flag and
    versionlz bytes that can legitimately differ between files:

    - Byte 0: blosc version, always 0x02 for blosc1
    - Byte 1: versionlz, depends on compressor (blosclz=0x01, but lz4=0x02 etc.)
    - Byte 2: flags (0x00=NOSHUFFLE, 0x01=SHUFFLE, 0x02=BITSHUFFLE, etc.)
    - Byte 3: typesize (itemsize of the uncompressed data, 1/2/4/8)
    - Bytes 4-7:  nbytes   (uncompressed size, LE uint32)
    - Bytes 8-11: blocksize (LE uint32)
    - Bytes 12-15: cbytes  (compressed size including 16-byte header, LE uint32)

    If expected_nbytes is given, only a header whose nbytes field matches is
    accepted. This acts as a strong fingerprint: for float64 PSFXL data,
    expected_nbytes = num_points * 8, which uniquely identifies the correct
    block even if multiple blosc headers appear nearby.

    Returns the byte offset of the header within ``data``, or -1.
    """
    pos = start
    while pos < end:
        # Blosc1 always starts with version byte 0x02.
        # We accept any versionlz (byte 1) since it varies by compressor.
        idx = data.find(b'\x02', pos, end)
        if idx == -1 or idx + 16 > len(data):
            break
        try:
            # versionlz must be non-zero; typesize must be a power-of-two <= 16
            versionlz = data[idx + 1]
            typesize   = data[idx + 3]
            if versionlz == 0 or typesize == 0 or typesize > 16:
                pos = idx + 1
                continue

            nbytes, blocksize, cbytes = struct.unpack_from('<3I', data, idx + 4)

            # Structural sanity checks on the header fields:
            #   cbytes  >= 16           (at minimum the header itself)
            #   nbytes  >= blocksize    (block cannot exceed total)
            #   blocksize > 0           (always set, even for single-block data)
            #   cbytes  <= nbytes + _BLOSC_MAX_OVERHEAD  (compressed >= uncompressed rare but ok)
            #   idx + cbytes fits inside the file
            if not (cbytes >= 16
                    and nbytes > 0
                    and blocksize > 0
                    and blocksize <= nbytes
                    and cbytes <= nbytes + _BLOSC_MAX_OVERHEAD
                    and idx + cbytes <= len(data)):
                pos = idx + 1
                continue

            # If caller supplied the expected uncompressed size, use it as
            # the decisive fingerprint.  This avoids false positives caused
            # by 0x02 bytes that happen to sit in the scan window.
            if expected_nbytes is not None and nbytes != expected_nbytes:
                pos = idx + 1
                continue

            return idx
        except Exception:
            pass
        pos = idx + 1
    return -1


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


def _scan_chunk_headers(data):
    """Pre-scan the PSFXL file for all valid chunk headers.

    Returns list of (offset, data_start, seq, num_points, chunk_size, flag, ref)
    tuples, sorted by offset. `flag` is the encoding byte Spectre writes for the
    chunk and `ref` is the trailing `<back_offset>.<length>` field that
    accompanies a back-reference; both are needed to decode correctly.
    """
    chunks = []
    pos = 0
    while True:
        idx = data.find(b'\x003:', pos)
        if idx == -1:
            break
        try:
            nl = data.index(b'\n', idx, idx + 300)
            text = data[idx+1:nl].decode('ascii', errors='replace')
            parts = text.split(':')
            seq = int(parts[1], 16)
            num_points = int(parts[3], 16)
            size_field = parts[4] if len(parts) > 4 else '0'
            if '.' in size_field:
                size_field = size_field.split('.')[0]
            chunk_size = int(size_field, 16)
            flag = parts[5] if len(parts) > 5 else ''
            ref = parts[7] if len(parts) > 7 else ''
            chunks.append((idx, nl + 1, seq, num_points, chunk_size, flag, ref))
        except (ValueError, IndexError):
            pass
        pos = idx + 1
    return chunks


def _find_type2_chunks(data):
    """Find type-2 chunk headers (uncompressed flush chunks).

    Type-2 format: \x002:seq:?:num_traces:num_points:binary_size\n
    Followed by ASCII hex index table, null padding to 8-byte alignment,
    then raw little-endian IEEE 754 doubles laid out as
    (num_traces+1) groups of num_points values (time + each signal).

    Scans the entire file and returns every structurally valid type-2 chunk.
    Callers are responsible for deciding which chunks represent genuine
    end-of-simulation flushes versus intermediate in-flight checkpoints;
    use :func:`_filter_type2_chunks` for that classification.
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
                    # Skip ASCII index table: scan forward while bytes are
                    # printable ASCII or newlines (the index table is text).
                    scan_pos = nl + 1
                    while scan_pos < len(data):
                        b = data[scan_pos]
                        if b < 0x20 and b != 0x0a:
                            break
                        if b > 0x7e:
                            break
                        scan_pos += 1
                    # Skip ALL null padding bytes (amount is variable and can
                    # exceed 8 bytes; just aligning to the next 8-byte boundary
                    # is not sufficient and can leave scan_pos in the padding).
                    while scan_pos < len(data) and data[scan_pos] == 0x00:
                        scan_pos += 1
                    chunks.append((scan_pos, num_traces_c, num_points, binary_size))
            pos = nl + 1 if nl > idx else idx + 1
        except (ValueError, IndexError):
            pos = idx + 1
    return chunks


def _filter_type2_chunks(type2_chunks, data, last_t3_time):
    """Classify type-2 chunks as final flushes or stale intermediate checkpoints.

    During a normal simulation Spectre compresses signal data into type-3
    (blosc-compressed) blocks.  It also periodically writes small uncompressed
    type-2 checkpoint records.  When those checkpoints are later covered by
    type-3 blocks the checkpoint data is redundant — including it would
    duplicate or back-track the time axis.  This happens in particular when a
    simulation is interrupted before it finishes: the last type-2 checkpoint
    written was at some earlier simulation time, while all the type-3 data
    extends to a later time.

    The rule is simple: a type-2 chunk whose first time value is less than or
    equal to ``last_t3_time`` has already been captured by the type-3 decoder
    and must be skipped.  Only a chunk whose first time value is strictly
    greater than ``last_t3_time`` contains genuinely new data (a true
    end-of-simulation flush).

    The classification is based entirely on the simulation-time content of
    each chunk, not on file offsets, so it remains correct regardless of how
    Spectre orders sections within the file.

    Parameters
    ----------
    type2_chunks : list
        List of ``(doubles_start, nt2, npts2, bsize2)`` tuples as returned by
        :func:`_find_type2_chunks`.
    data : bytes
        Raw file contents (used to peek at each chunk's first time value).
    last_t3_time : float
        Last time value decoded from type-3 (compressed) chunks, or
        ``-numpy.inf`` if no type-3 data was found.

    Returns
    -------
    valid : list
        Chunks to include (first time > last_t3_time).
    skipped : list of float
        First-time value of each skipped intermediate checkpoint, for use in
        warning messages.
    """
    valid = []
    skipped = []
    for doubles_start, nt2, npts2, bsize2 in type2_chunks:
        if doubles_start + bsize2 > len(data):
            continue
        # Peek at the first 8 bytes of the data region: in the grouped layout
        # (num_traces+1, npts2) the first float64 is always the earliest time
        # value in the chunk.
        first_time = struct.unpack_from('<d', data, doubles_start)[0]
        if first_time > last_t3_time:
            valid.append((doubles_start, nt2, npts2, bsize2))
        else:
            skipped.append(first_time)
    return valid, skipped


def decode_psfxl(data_path, num_traces):
    """Decode the .psfxl waveform data file.

    Args:
        data_path: Path to the .tran.psfxl file
        num_traces: Number of trace signals (from header parsing)

    Returns:
        (time_array, signal_arrays, flush_count) where signal_arrays is a dict
        mapping signal index (0..num_traces-1) to numpy arrays.
    """
    with open(data_path, 'rb') as f:
        data = f.read()

    # Pass 1: find all chunk headers
    chunks = _scan_chunk_headers(data)

    all_time = []
    all_signals = {i: [] for i in range(num_traces)}
    flush_count = 0

    # Pass 2: process each chunk in order
    for offset, data_start, seq, num_points, chunk_size, flag, ref in chunks:
        if num_points == 0:
            continue

        # A zero-size chunk is NOT a run of zeros. Spectre writes it when this
        # signal's data for this flush is byte-identical to a block already in
        # the file -- two clock phases with the same duty, a bus whose bits
        # carry the same code -- and points at that block with a trailing
        # `<back_offset>.<length>` field: the block starts `back_offset` bytes
        # before this chunk header and its compressed size is `length`.
        # Filling zeros here silently replaces a real waveform with a flat
        # line, and only for signals that happen to duplicate another one.
        if chunk_size == 0:
            if 0 <= seq < num_traces and ref:
                try:
                    back = int(ref.split('.')[0], 16)
                    src = offset - back
                    if src >= 0:
                        cbytes = struct.unpack_from('<I', data, src + 12)[0]
                        vals = np.frombuffer(
                            _blosc_decompress(data[src:src + cbytes]),
                            dtype='<f8')
                        if len(vals) == num_points:
                            all_signals[seq].append(vals)
                            continue
                except Exception:
                    pass
            warnings.warn(
                "psfxl chunk for signal {} at offset {} (flag {!r}, ref {!r}) "
                "has zero size and its back-reference could not be resolved; "
                "the signal is left short rather than filled with zeros"
                .format(seq, offset, flag, ref),
                RuntimeWarning)
            continue

        # Handle 8-byte constant chunks: DC value stored as a single raw
        # little-endian double, preceded by alignment null padding.
        # The padding length is exactly (8 - data_start % 8) % 8 bytes
        # (1–8 bytes to reach the next 8-byte file offset boundary).
        # We must NOT skip all null bytes because:
        #   (a) signals with a DC value of 0.0 have all-zero payload bytes
        #       that are indistinguishable from padding, and
        #   (b) the next chunk header starts with a \x00 byte that would
        #       also be consumed, causing us to read ASCII header text as
        #       a double and produce garbage (~1.9×10^185).
        # Using alignment arithmetic (1–8 bytes) avoids both pitfalls.
        # NOTE: align_skip is always 8 - (data_start % 8), i.e. 1–8 bytes,
        # NEVER 0.  When data_start is already 8-byte aligned, Spectre still
        # emits a full 8-byte null pad before the value; the outer % 8 in
        # (8 - x) % 8 would collapse this to 0 and read the padding as 0.0.
        # Must be caught before the blosc search because the nearest blosc
        # block (a different signal) may fall within the 128-byte search
        # window, causing data to be mis-assigned to this signal.
        if chunk_size == 8 and seq < num_traces:
            align_skip = 8 - (data_start % 8)  # always 1–8, never 0
            val_start = data_start + align_skip
            if val_start + 8 <= len(data):
                val = struct.unpack('<d', data[val_start:val_start + 8])[0]
                all_signals[seq].append(np.full(num_points, val, dtype='<f8'))
            continue

        # Find blosc header after text header.
        # Pass expected_nbytes = num_points * 8 so the search fingerprints the
        # correct block by its uncompressed size rather than relying on the
        # BLOSC_MAGIC flag/versionlz bytes, which can differ across compressors
        # and shuffle modes and were what caused silent skips on numcodecs 0.9.1.
        expected_nbytes = num_points * 8
        blosc_pos = _find_blosc_header(data, data_start, data_start + 1280, expected_nbytes)
        if blosc_pos == -1:
            continue

        try:
            if seq == 0:
                # First chunk of flush: time sweep + first trace signal
                h0 = struct.unpack('<4B3I', data[blosc_pos:blosc_pos+16])
                try:
                    dec0 = _blosc_decompress(data[blosc_pos:blosc_pos+h0[6]])
                    time_vals = np.frombuffer(dec0, dtype='<f8')
                    # Validate that we got the expected number of points
                    if len(time_vals) == num_points:
                        all_time.append(time_vals)
                    elif _NUMCODECS_OLD_VERSION:
                        warnings.warn(
                            f"Blosc decompression produced {len(time_vals)} points "
                            f"but expected {num_points}. Numcodecs {_NUMCODECS_VERSION} "
                            f"may have truncation issues.",
                            RuntimeWarning
                        )
                        all_time.append(time_vals)
                    else:
                        continue
                except Exception as e:
                    if _NUMCODECS_OLD_VERSION:
                        warnings.warn(
                            f"Failed to decompress time data: {e}. "
                            f"Numcodecs {_NUMCODECS_VERSION} may be unstable. "
                            f"Consider upgrading to >= 0.16.0",
                            RuntimeWarning
                        )
                    continue

                # Locate the second blosc block (signal 0).
                # It nominally sits at blosc_pos + h0[6], but there may be
                # padding bytes between the two blocks (alignment to 8 or 16
                # bytes, null terminators, etc.).  Search forward from the
                # nominal position within a small window rather than requiring
                # an exact offset.  Use the expected uncompressed size as the
                # fingerprint so we don't mis-pick an unrelated block.
                sig_expected_nbytes = num_points * 8
                sig_start_nominal = blosc_pos + h0[6]
                # Trace 0 rides in this chunk after the time block, encoded the
                # same way any signal is. When it is constant Spectre stores a
                # single inline double rather than a blosc block, and the
                # declared chunk size exceeds the time block by exactly 8
                # bytes. Searching only for a blosc header misses that and
                # drops trace 0 with no diagnostic.
                if chunk_size - h0[6] == 8:
                    val = struct.unpack_from('<d', data, sig_start_nominal)[0]
                    all_signals[0].append(np.full(num_points, val, dtype='<f8'))
                    flush_count += 1
                    continue
                sig_start = _find_blosc_header(
                    data, sig_start_nominal, sig_start_nominal + 32, sig_expected_nbytes
                )
                if sig_start != -1:
                    try:
                        h1 = struct.unpack('<4B3I', data[sig_start:sig_start+16])
                        dec1 = _blosc_decompress(data[sig_start:sig_start+h1[6]])
                        sig_vals = np.frombuffer(dec1, dtype='<f8')
                        if len(sig_vals) == num_points:
                            all_signals[0].append(sig_vals)
                        elif _NUMCODECS_OLD_VERSION:
                            warnings.warn(
                                f"Blosc decompression produced {len(sig_vals)} points "
                                f"but expected {num_points} for signal 0.",
                                RuntimeWarning
                            )
                            all_signals[0].append(sig_vals)
                    except Exception as e:
                        if _NUMCODECS_OLD_VERSION:
                            warnings.warn(
                                f"Failed to decompress signal 0: {e}",
                                RuntimeWarning
                            )

                flush_count += 1
            else:
                # Single signal frame
                h = struct.unpack('<4B3I', data[blosc_pos:blosc_pos+16])
                try:
                    dec = _blosc_decompress(data[blosc_pos:blosc_pos+h[6]])
                    sig_vals = np.frombuffer(dec, dtype='<f8')
                    sig_idx = seq
                    # Validate expected points
                    if len(sig_vals) == num_points and sig_idx < num_traces:
                        all_signals[sig_idx].append(sig_vals)
                    elif sig_idx < num_traces and _NUMCODECS_OLD_VERSION:
                        warnings.warn(
                            f"Blosc decompression produced {len(sig_vals)} points "
                            f"but expected {num_points} for signal {sig_idx}.",
                            RuntimeWarning
                        )
                        all_signals[sig_idx].append(sig_vals)
                except Exception as e:
                    if _NUMCODECS_OLD_VERSION:
                        warnings.warn(
                            f"Failed to decompress signal {seq}: {e}",
                            RuntimeWarning
                        )
                    pass
        except Exception as e:
            if _NUMCODECS_OLD_VERSION:
                warnings.warn(
                    f"Error processing chunk at seq={seq}: {e}. "
                    f"Numcodecs {_NUMCODECS_VERSION} may be unstable.",
                    RuntimeWarning
                )
            continue

    # Pass 3: find and process type-2 chunks (uncompressed flush chunks).
    #
    # type-2 chunks come in two flavours:
    #   Final flush  — written after all type-3 blocks, contains data not yet
    #                  compressed; first time value > last type-3 time.
    #   Intermediate — periodic in-flight checkpoint written while the
    #                  simulation was still running; the same data was
    #                  subsequently compressed into type-3 blocks, so the
    #                  chunk is redundant; first time value <= last type-3 time.
    #
    # _filter_type2_chunks classifies each candidate by peeking at its first
    # time value and comparing it to the decoded type-3 range.  This is a
    # purely data-driven check — it makes no assumptions about file layout or
    # byte offsets and is therefore robust to any ordering Spectre may use.
    type2_candidates = _find_type2_chunks(data)
    last_t3_time = all_time[-1][-1] if all_time else -np.inf
    type2_chunks, skipped_times = _filter_type2_chunks(type2_candidates, data, last_t3_time)

    if skipped_times:
        warnings.warn(
            f"{len(skipped_times)} type-2 intermediate checkpoint chunk(s) were "
            f"skipped because their simulation time(s) "
            f"({', '.join(f'{t:.4e} s' for t in skipped_times)}) fall within the "
            f"type-3 compressed data range (last type-3 time: {last_t3_time:.4e} s). "
            f"This is expected when a simulation was interrupted before completing — "
            f"the checkpoint data is already captured in the compressed type-3 blocks.",
            UserWarning,
            stacklevel=3,
        )

    for doubles_start, nt2, npts2, bsize2 in type2_chunks:
        doubles = np.frombuffer(data[doubles_start:doubles_start+bsize2], dtype='<f8')
        groups = doubles.reshape(nt2 + 1, npts2)
        all_time.append(groups[0])
        for sig_idx in range(min(nt2, num_traces)):
            all_signals[sig_idx].append(groups[1 + sig_idx])
        flush_count += 1

    # Concatenate arrays
    time_full = np.concatenate(all_time) if all_time else np.array([])
    signals = {}
    for idx in range(num_traces):
        if all_signals[idx]:
            signals[idx] = np.concatenate(all_signals[idx])

    return time_full, signals, flush_count
