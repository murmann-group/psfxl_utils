"""Decoder for PSF 'info' format (magic 0x00000500).

Handles two sub-types that share the same magic:
  - DC scalar results  (dc.dc)      : simple "signal_name type_name value" entries
  - DC operating point (dcOpInfo.info, etc.) : struct-typed entries where each
    device maps to a set of named scalar fields (gm, vth, ids, …)

Format overview (differences from flat 0x200 PSF)
--------------------------------------------------
File header:  magic(4) count(4) header_end(4)   — identical to 0x200
Both TYPE and VALUE sections are wrapped in an outer TAG_GROUP_EXT(0x16) container.
Every ENTRY (0x10) in the TYPE section has an extra 'base_type' word after type_ref:
    TAG(4) EID(4) name(len-prefix,padded) type_ref(4) base_type(4) [props]
  base_type codes:
    0x0b  FLOAT64  (8-byte big-endian double)
    0x05  INT32    (4-byte big-endian signed integer)
    0x0c  COMPLEX  (16-byte: two 8-byte doubles, real then imag)
    0x10  STRUCT   (no data bytes; declares a struct type; fields follow)

Struct fields are declared as regular ENTRY records in the TYPE section that
follow immediately after their parent STRUCT entry.  Fields belonging to a
top-level scalar type (dc.dc "V", "I", …) carry a "key" property ("node" /
"branch"); struct fields do not.

VALUE entries mirror what is in TYPE:
  - scalars   : TAG EID name type_ref(scalar_eid) float64_value [props]
  - structs   : TAG EID name type_ref(struct_eid)  <packed field values> [props]
    packed fields: each field is 8 bytes (float64) or 4 bytes (int32) based on
    its base_type declared in the TYPE section.  Trailing string/int/float
    properties follow (e.g. PROP("model" "…")).
"""

import os
import struct
import mmap

import numpy as np

from .psf_parse import check_file_header, section_chain

# Base-type codes found in the 0x500 TYPE section
_INFO_FLOAT  = 0x0b   # 8-byte big-endian double
_INFO_INT    = 0x05   # 4-byte big-endian signed int  (INT LONG)
_INFO_INT8   = 0x01   # 4-byte big-endian signed int  (INT BYTE; same storage as INT LONG)
_INFO_CPLX   = 0x0c   # 16-byte complex (re, im)
_INFO_STRUCT = 0x10   # struct container; size = sum of its fields
_INFO_STRING = 0x0e   # length-prefixed, 4-byte-padded ASCII (e.g. stb_state)

# Property / tag constants (same across all PSF formats)
_TAG_ENTRY     = 0x10
_TAG_GROUP_EXT = 0x16  # group container used in 0x500 sections
_TAG_STR_PROP  = 0x21
_TAG_INT_PROP  = 0x22
_TAG_FLT_PROP  = 0x23

_TAG_ENTRY_BYTES = struct.pack('>I', _TAG_ENTRY)


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------

def _u32(data, pos):
    return struct.unpack('>I', data[pos:pos + 4])[0], pos + 4


def _i32(data, pos):
    return struct.unpack('>i', data[pos:pos + 4])[0], pos + 4


def _f64(data, pos):
    return struct.unpack('>d', data[pos:pos + 8])[0], pos + 8


def _read_str(data, pos):
    """Read a length-prefixed, 4-byte-aligned ASCII string."""
    length, pos = _u32(data, pos)
    s = data[pos:pos + length].decode('ascii', errors='replace').rstrip('\x00')
    pos += length
    if length % 4:
        pos += 4 - (length % 4)
    return s, pos


def _read_props(data, pos, end):
    """Consume and return consecutive property tags (0x21/0x22/0x23) as dict."""
    props = {}
    while pos < end - 4:
        tag = struct.unpack('>I', data[pos:pos + 4])[0]
        if tag == _TAG_STR_PROP:
            pos += 4
            k, pos = _read_str(data, pos)
            v, pos = _read_str(data, pos)
            props[k] = v
        elif tag == _TAG_INT_PROP:
            pos += 4
            k, pos = _read_str(data, pos)
            v, pos = _u32(data, pos)
            props[k] = v
        elif tag == _TAG_FLT_PROP:
            pos += 4
            k, pos = _read_str(data, pos)
            v, pos = _f64(data, pos)
            props[k] = v
        else:
            break
    return props, pos


# ---------------------------------------------------------------------------
# TYPE section parser
# ---------------------------------------------------------------------------

def _parse_type_section(data, sec_start, sec_end):
    """Parse the TYPE section of a PSF 0x500 file.

    Returns:
        struct_types : dict  eid -> {'name': str,
                                     'fields': [{'name', 'units', 'base_type'}]}
        scalar_types : dict  eid -> {'name': str, 'units': str, 'base_type': int}
    """
    pos = sec_start + 8   # skip section container header (sec_tag + sec_end)

    # Unwrap optional outer GROUP_EXT container
    tag = struct.unpack('>I', data[pos:pos + 4])[0]
    scan_end = sec_end
    if tag == _TAG_GROUP_EXT:
        group_end = struct.unpack('>I', data[pos + 4:pos + 8])[0]
        pos += 8
        scan_end = min(group_end, sec_end)

    struct_types = {}   # eid -> {name, fields: [{name, units, base_type}]}
    scalar_types = {}   # eid -> {name, units, base_type}
    current_struct_eid = None

    while pos < scan_end - 12:
        idx = data.find(_TAG_ENTRY_BYTES, pos, scan_end)
        if idx == -1:
            break

        try:
            epos = idx + 4
            eid,       epos = _u32(data, epos)
            name,      epos = _read_str(data, epos)
            _type_ref, epos = _u32(data, epos)
            base_type, epos = _u32(data, epos)

            # Reject garbage hits
            if not name or len(name) > 80 or not all(
                '\x20' <= c < '\x7f' for c in name
            ):
                pos = idx + 4
                continue

            props, epos = _read_props(data, epos, scan_end)
            units = props.get('units', '')

            if base_type == _INFO_STRUCT:
                # Top-level STRUCT type declaration (e.g. "bsim4")
                struct_types[eid] = {'name': name, 'fields': []}
                current_struct_eid = eid

            elif base_type in (_INFO_FLOAT, _INFO_INT, _INFO_INT8, _INFO_CPLX, _INFO_STRING):
                # Decide: struct field vs. standalone scalar signal type.
                # Scalar type entries (dc.dc "V", "I", …) carry a "key" property.
                if props.get('key') is not None:
                    # Standalone scalar type
                    scalar_types[eid] = {
                        'name': name, 'units': units, 'base_type': base_type,
                    }
                    current_struct_eid = None   # safe exit from any struct context
                elif current_struct_eid is not None and current_struct_eid in struct_types:
                    # Field belonging to the most recently declared struct
                    struct_types[current_struct_eid]['fields'].append({
                        'name': name, 'units': units, 'base_type': base_type,
                    })
                else:
                    # No active struct and no "key" property → treat as scalar
                    scalar_types[eid] = {
                        'name': name, 'units': units, 'base_type': base_type,
                    }

            pos = epos

        except (struct.error, IndexError, ValueError, UnicodeDecodeError):
            pos = idx + 4

    return struct_types, scalar_types


# ---------------------------------------------------------------------------
# VALUE section parser
# ---------------------------------------------------------------------------

def _parse_value_section(data, sec_start, sec_end, struct_types, scalar_types,
                         strings=None):
    """Parse the VALUE section of a PSF 0x500 file.

    Returns a list of (signal_name, units, value) tuples where value is a
    float, int cast to float, or complex number.  String values (base type
    0x0e, e.g. ``stb_state``) and string properties of entries (e.g.
    ``gainMargin.gainMarginInfo``) go into *strings* if a dict is given.
    """
    if strings is None:
        strings = {}

    def keep_string_props(entry, props):
        for k, v in props.items():
            if isinstance(v, str):
                strings[f"{entry}.{k}"] = v

    pos = sec_start + 8   # skip section container header

    tag = struct.unpack('>I', data[pos:pos + 4])[0]
    scan_end = sec_end
    if tag == _TAG_GROUP_EXT:
        group_end = struct.unpack('>I', data[pos + 4:pos + 8])[0]
        pos += 8
        scan_end = min(group_end, sec_end)

    signals = []   # [(name, units, value), …]

    while pos < scan_end - 12:
        idx = data.find(_TAG_ENTRY_BYTES, pos, scan_end)
        if idx == -1:
            break

        try:
            epos = idx + 4
            _eid,     epos = _u32(data, epos)
            name,     epos = _read_str(data, epos)
            type_ref, epos = _u32(data, epos)

            if not name or len(name) > 128:
                pos = idx + 4
                continue

            if type_ref in struct_types:
                # ---- Struct-typed entry (bsim4 op-point, etc.) ---------------
                stype = struct_types[type_ref]
                for field in stype['fields']:
                    ft = field['base_type']
                    if ft == _INFO_FLOAT:
                        v, epos = _f64(data, epos)
                        signals.append((f"{name}.{field['name']}", field['units'], v))
                    elif ft in (_INFO_INT, _INFO_INT8):
                        v, epos = _i32(data, epos)
                        signals.append((f"{name}.{field['name']}", field['units'], float(v)))
                    elif ft == _INFO_CPLX:
                        re, epos = _f64(data, epos)
                        im, epos = _f64(data, epos)
                        signals.append((f"{name}.{field['name']}", field['units'],
                                        complex(re, im)))
                    elif ft == _INFO_STRING:
                        v, epos = _read_str(data, epos)
                        strings[f"{name}.{field['name']}"] = v
                _props, epos = _read_props(data, epos, scan_end)
                keep_string_props(name, _props)

            elif type_ref in scalar_types:
                # ---- Scalar-typed entry (node voltage, branch current, etc.) -
                stype = scalar_types[type_ref]
                bt = stype['base_type']
                if bt == _INFO_FLOAT:
                    v, epos = _f64(data, epos)
                    signals.append((name, stype['units'], v))
                elif bt in (_INFO_INT, _INFO_INT8):
                    v, epos = _i32(data, epos)
                    signals.append((name, stype['units'], float(v)))
                elif bt == _INFO_CPLX:
                    re, epos = _f64(data, epos)
                    im, epos = _f64(data, epos)
                    signals.append((name, stype['units'], complex(re, im)))
                elif bt == _INFO_STRING:
                    v, epos = _read_str(data, epos)
                    strings[name] = v
                _props, epos = _read_props(data, epos, scan_end)
                keep_string_props(name, _props)

            else:
                # Unknown type reference — skip gracefully
                pos = idx + 4
                continue

            pos = epos

        except (struct.error, IndexError, ValueError, UnicodeDecodeError):
            pos = idx + 4

    return signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_psf_info(path, strings=None):
    """Decode a PSF 'info' format file (magic 0x00000500).

    Works for both:
      * dc.dc        (plain DC node/branch results, one float per signal)
      * dcOpInfo.info (device operating-point struct data, many floats per device)

    Returns:
        sweep_arr (np.ndarray): length-1 array ``[0.0]`` — snapshot, no sweep axis.
        signals   (dict):       sequential_index -> np.ndarray([value])
        traces    (list):       list of dicts with 'name', 'units', 'entry_idx'

    If *strings* is a dict, string-valued entries (e.g. ``stb_state`` in a
    stability ``.margin.stb``) and string properties of entries (keyed
    ``<entry>.<property>``, e.g. ``gainMargin.gainMarginInfo`` or
    ``M0.model``) are stored in it.  They are not signals: a signal's
    ordinate is always numeric.
    """
    path = str(path)

    with open(path, 'rb') as fh:
        if os.fstat(fh.fileno()).st_size < 12:
            raise ValueError(f"{path}: truncated or corrupt PSF file "
                             f"(shorter than the file header)")
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
            header_end = check_file_header(data, path)
            sections, truncated = section_chain(data, header_end)
            if truncated:
                # An operating point with devices silently missing is worse
                # than no result: Spectre writes these files in one go.
                raise ValueError(
                    f"{path}: truncated PSF file (a section runs past the end "
                    f"of the {len(data)}-byte file)")

            if len(sections) < 2:
                return np.array([0.0]), {}, []

            type_sec_start, type_sec_end = sections[0]
            val_sec_start,  val_sec_end  = sections[1]

            struct_types, scalar_types = _parse_type_section(
                data, type_sec_start, type_sec_end)

            raw_signals = _parse_value_section(
                data, val_sec_start, val_sec_end, struct_types, scalar_types,
                strings)

    sweep_arr = np.array([0.0])
    signals: dict = {}
    traces: list  = []

    for i, (sig_name, sig_units, sig_val) in enumerate(raw_signals):
        if isinstance(sig_val, complex):
            arr = np.array([sig_val], dtype=complex)
        else:
            arr = np.array([float(sig_val)])
        signals[i] = arr
        traces.append({'name': sig_name, 'units': sig_units,
                       'type_id': 0, 'entry_idx': i})

    return sweep_arr, signals, traces
