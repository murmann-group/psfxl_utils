"""Flat PSF decoder for .ac, .noise, .dc files."""

import struct
import numpy as np
import mmap


def decode_psf_flat(header_path, header):
    """Decode sweep and trace data from a flat PSF file (.ac, .noise, .dc).

    Uses a vectorized numpy approach: the VALUE section has a regular record
    layout, so we compute the stride and extract all values with array slicing
    instead of scanning for TAG_ENTRY markers in a Python loop.  Falls back to
    a sparse marker-scanning path when the layout is irregular.

    Args:
        header_path: Path to the flat PSF file.
        header: Parsed header dict from parse_psf_header().

    Returns:
        (sweep_array, signal_arrays) where signal_arrays maps trace index
        (position in header['traces']) to a 1-D numpy array — complex for
        AC/stability traces, float64 otherwise.
    """
    with open(header_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            file_size = len(data)
            header_end = struct.unpack('>I', data[8:12])[0]

            # Walk sections to find the VALUE section (always last)
            pos = header_end
            value_start = value_end = None
            while pos < file_size - 8:
                sec_end = struct.unpack('>I', data[pos + 4:pos + 8])[0]
                if sec_end <= pos or sec_end > file_size:
                    break
                value_start = pos + 8
                value_end = sec_end
                pos = sec_end

            if value_start is None or value_end is None:
                return np.array([]), {}

            traces = header['traces']
            sweep_eidx = header['sweep'].get('entry_idx')

            # Build entry_idx -> (seq_idx, is_complex) map — skip virtual inline
            # sub-traces (they share an entry_idx with their parent and have no
            # TAG_ENTRY of their own in the VALUE section).
            trace_map = {}
            for i, t in enumerate(traces):
                eidx = t.get('entry_idx')
                if eidx is not None and 'inline_index' not in t:
                    trace_map[eidx] = (i, t.get('complex', False))

            # Try dense (stride-based) format first
            result = _try_dense_format(data, value_start, value_end, sweep_eidx, trace_map)
            if result is not None:
                sweep_arr, signal_arrays = result
            else:
                # Fall back to sparse (marker-scanning) format
                sweep_arr, signal_arrays = _try_sparse_format(
                    data, value_start, value_end, sweep_eidx, trace_map
                )

            # Extract inline per-component signals if the header has any virtual
            # sub-trace entries (added by parse_psf_header for GROUP-typed devices).
            has_inline = any('inline_index' in t for t in traces)
            if has_inline:
                _extract_inline_signals(
                    data, value_start, value_end, sweep_eidx,
                    traces, signal_arrays,
                )

            return sweep_arr, signal_arrays


def _try_dense_format(data, value_start, value_end, sweep_eidx, trace_map):
    """Try to decode as a dense stride-based format (like tran-000_noise.noise).
    
    Returns (sweep_arr, signal_arrays) on success, or None on failure.
    """
    # Parse the first sweep point to learn record order and sizes.
    # Each record: TAG_ENTRY(4) + entry_idx(4) + value(8 real, 16 complex)
    record_layout = []  # (offset_in_stride, 'sweep'|seq_idx, is_complex)
    p = value_start
    offset = 0
    first_eid = None
    
    while p + 8 <= value_end:
        tag = struct.unpack('>I', data[p:p+4])[0]
        if tag != 0x10:
            return None  # Not dense format
        eid = struct.unpack('>I', data[p+4:p+8])[0]

        # Stop if we've looped back to the first EID (next sweep point)
        if first_eid is None:
            first_eid = eid
        elif eid == first_eid:
            break

        # Auto-detect record size: check if next TAG_ENTRY is at +16 or +24
        is_complex_data = False
        if p + 24 + 4 <= value_end:
            tag_at_16 = struct.unpack('>I', data[p+16:p+20])[0]
            tag_at_24 = struct.unpack('>I', data[p+24:p+28])[0]
            if tag_at_16 == 0x10:
                is_complex_data = False  # 16-byte real record
            elif tag_at_24 == 0x10:
                is_complex_data = True   # 24-byte complex record

        if eid == sweep_eidx:
            record_layout.append((offset, 'sweep', False))
            p += 16
            offset += 16
        elif eid in trace_map:
            seq_idx, _ = trace_map[eid]
            record_layout.append((offset, seq_idx, is_complex_data))
            rec_size = 24 if is_complex_data else 16
            p += rec_size
            offset += rec_size
        else:
            # Unknown EID - use detected size, skip
            rec_size = 24 if is_complex_data else 16
            record_layout.append((offset, None, is_complex_data))
            p += rec_size
            offset += rec_size

    stride = offset
    if stride == 0:
        return None

    value_size = value_end - value_start
    n_points = value_size // stride
    
    # Sanity check: make sure we have a reasonable number of points
    if n_points == 0 or n_points > value_size:
        return None

    # Read the entire VALUE section as a flat byte buffer
    raw = np.frombuffer(
        data[value_start:value_start + n_points * stride],
        dtype=np.uint8,
    )

    # Extract sweep and signal arrays using numpy stride tricks
    sweep_arr = np.empty(0, dtype=np.float64)
    signal_arrays = {}

    for rec_offset, target, is_complex in record_layout:
        if target is None:
            continue

        # Value data starts at rec_offset + 8 (skip TAG + EID bytes)
        val_offset = rec_offset + 8

        if target == 'sweep':
            sweep_arr = np.ndarray(
                n_points, dtype='>f8',
                buffer=raw, offset=val_offset, strides=(stride,),
            ).astype(np.float64)
        elif is_complex:
            real_part = np.ndarray(
                n_points, dtype='>f8',
                buffer=raw, offset=val_offset, strides=(stride,),
            )
            imag_part = np.ndarray(
                n_points, dtype='>f8',
                buffer=raw, offset=val_offset + 8, strides=(stride,),
            )
            signal_arrays[target] = (
                real_part.astype(np.float64)
                + 1j * imag_part.astype(np.float64)
            )
        else:
            signal_arrays[target] = np.ndarray(
                n_points, dtype='>f8',
                buffer=raw, offset=val_offset, strides=(stride,),
            ).astype(np.float64)

    return sweep_arr, signal_arrays


def _try_sparse_format(data, value_start, value_end, sweep_eidx, trace_map):
    """Decode as a sparse format with padding between records.

    Locates TAG_ENTRY markers with bytes.find (C speed) instead of scanning
    byte-by-byte in Python.
    """
    value_size = value_end - value_start
    if value_size < 16:
        return np.array([]), {}
    
    # Phase 1: Find all marker positions using C-optimized string search
    value_bytes = bytes(data[value_start:value_end])
    marker_byte = b'\x00\x00\x00\x10'
    marker_offsets = []
    start = 0
    
    # Step size = 4 to skip impossible positions (records are >= 16 bytes)
    while True:
        pos = value_bytes.find(marker_byte, start)
        if pos == -1:
            break
        marker_offsets.append(pos)
        start = pos + 4
    
    if not marker_offsets:
        return np.array([]), {}
    
    # Phase 2: Extract records and group by sweep point
    sweep_values = []
    eid_to_records = {}  # eid -> list of (point_idx, value)
    current_point = 0  # Track which sweep point we're on
    
    for offset in marker_offsets:
        rec_pos = value_start + offset
        if rec_pos + 16 > value_end:
            break
        
        # Read EID and value using struct (optimized C code)
        eid = struct.unpack('>I', data[rec_pos+4:rec_pos+8])[0]
        value = struct.unpack('>d', data[rec_pos+8:rec_pos+16])[0]
        
        if eid == sweep_eidx:
            sweep_values.append(value)
            current_point = len(sweep_values) - 1  # Updated index after adding sweep
        else:
            point_idx = current_point
            if eid not in eid_to_records:
                eid_to_records[eid] = []
            eid_to_records[eid].append((point_idx, value))
    
    if not sweep_values:
        return np.array([]), {}
    
    n_points = len(sweep_values)
    sweep_arr = np.array(sweep_values, dtype=np.float64)
    signal_arrays = {}
    
    # Phase 3: Build signal arrays from records
    for eid, (sig_idx, _) in trace_map.items():
        if sig_idx is not None and eid in eid_to_records:
            arr = np.zeros(n_points, dtype=np.float64)
            for pt_idx, val in eid_to_records[eid]:
                if pt_idx < n_points:
                    arr[pt_idx] = val
            signal_arrays[sig_idx] = arr
    
    return sweep_arr, signal_arrays


def _extract_inline_signals(data, value_start, value_end, sweep_eidx, traces, signal_arrays):
    """Extract per-component inline values from a variable-record-size PSF VALUE section.

    Spectre noise PSF stores per-component noise (rs, id, fn …) directly after
    each device's total-noise value in the VALUE section without additional
    TAG_ENTRY markers.  The number of inline 8-byte doubles per device depends on
    the device type (stored in each trace's 'n_inline' field).

    The VALUE section uses a fixed *total* stride (all frequency points have the
    same byte layout); only the per-record sizes vary.  We locate each record's
    byte offset within the stride by a sequential walk using the known per-EID
    inline counts, then use numpy stride tricks to extract all frequency points
    at once.

    Results are written in-place into *signal_arrays* using the seq_idx values of
    the virtual sub-trace entries in *traces*.
    """
    # Find the total stride by locating the second occurrence of the sweep EID.
    sweep_pattern = struct.pack('>II', 0x10, sweep_eidx)
    second_occ = data.find(sweep_pattern, value_start + 8, value_end)
    if second_occ == -1:
        return
    stride = second_occ - value_start
    total_bytes = value_end - value_start
    n_points = total_bytes // stride
    if n_points == 0:
        return

    # Build maps from entry_idx:
    #   eid_n_inline[eid]  -> number of inline 8-byte values after the main value
    #   eid_inline_seqs[eid] -> list of (inline_index, seq_idx) for virtual sub-traces
    eid_n_inline = {}    # populated from main-trace 'n_inline' field
    eid_inline_seqs = {}

    for seq_idx, t in enumerate(traces):
        eid = t.get('entry_idx')
        if eid is None:
            continue
        if 'inline_index' not in t:
            # Main trace — record its inline count.
            n = t.get('n_inline', 0)
            if n:
                eid_n_inline[eid] = n
        else:
            # Virtual sub-trace — record its position.
            if eid not in eid_inline_seqs:
                eid_inline_seqs[eid] = []
            eid_inline_seqs[eid].append((t['inline_index'], seq_idx))

    if not eid_inline_seqs:
        return

    # Compute the byte offset of each EID's TAG_ENTRY within the stride.
    # Order: sweep first, then traces in their list order (which matches the
    # entry_idx / VALUE-section order observed in the file).
    eid_stride_offset = {}
    byte_off = 0
    # Sweep record is always first and has no inline values.
    eid_stride_offset[sweep_eidx] = byte_off
    byte_off += 16   # TAG(4) + EID(4) + value(8)

    for t in traces:
        if 'inline_index' in t:
            continue   # skip virtual sub-traces
        eid = t.get('entry_idx')
        if eid is None:
            continue
        eid_stride_offset[eid] = byte_off
        n_inline = t.get('n_inline', 0)
        # Record layout: TAG(4)+EID(4) + n_inline doubles(8 each).
        # For GROUP-type devices n_inline = N_comps + 1 (components + total).
        # For plain devices n_inline = 0 so the record is just TAG+EID+value = 16 bytes.
        if n_inline > 0:
            byte_off += 8 + n_inline * 8   # TAG+EID(8) + n_inline×double(8)
        else:
            byte_off += 16   # TAG(4) + EID(4) + value(8)

    # Read the entire VALUE section as a uint8 buffer for numpy striding.
    raw = np.frombuffer(data[value_start: value_start + n_points * stride], dtype=np.uint8)

    # Extract each inline component array using numpy stride tricks.
    # Record layout: [TAG(4)][EID(4)][comp_0(8)][comp_1(8)]...[comp_N-2(8)][total(8)]
    # inline_index=0 → comp_0 at byte offset base_off+8 (immediately after TAG+EID)
    # The main trace reads 'total' at base_off + 8 + (N-1)*8.
    for eid, comps in eid_inline_seqs.items():
        base_off = eid_stride_offset.get(eid)
        if base_off is None:
            continue
        for inline_idx, seq_idx in comps:
            # comp_0 starts immediately after TAG(4)+EID(4) at base_off+8.
            val_off = base_off + 8 + inline_idx * 8
            if val_off + 8 > stride:
                continue
            arr = np.ndarray(
                n_points, dtype='>f8',
                buffer=raw, offset=val_off, strides=(stride,),
            ).astype(np.float64)
            signal_arrays[seq_idx] = arr

    # Also fix the main trace for each GROUP-type device to point at 'total'
    # (the last double in the record) instead of the first double.
    for seq_idx, t in enumerate(traces):
        if 'inline_total_index' not in t:
            continue
        eid = t.get('entry_idx')
        base_off = eid_stride_offset.get(eid)
        if base_off is None:
            continue
        total_idx = t['inline_total_index']
        val_off = base_off + 8 + total_idx * 8
        if val_off + 8 > stride:
            continue
        arr = np.ndarray(
            n_points, dtype='>f8',
            buffer=raw, offset=val_off, strides=(stride,),
        ).astype(np.float64)
        signal_arrays[seq_idx] = arr

