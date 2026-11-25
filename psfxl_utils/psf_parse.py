"""Shared utilities for PSF file parsing (both PSFXL and flat formats)."""

import os
import struct
import mmap

# PSF magic number constants
PSF_MAGIC      = 0x00000400  # PSFXL format (.tran header)
PSF_MAGIC_FLAT = 0x00000200  # flat PSF format (.ac, .noise, .dc files)
PSF_MAGIC_INFO = 0x00000500  # info/snapshot format (dc.dc, dcOpInfo.info, …)

# 4-byte tags in the binary PSF header (big-endian)
TAG_SECTION = 0x01
TAG_ENTRY = 0x10
TAG_GROUP = 0x11
TAG_STRING_PROP = 0x21
TAG_INT_PROP = 0x22
TAG_FLOAT_PROP = 0x23
TAG_DATA_BLOCK = 0x24  # 48-byte inline statistics block

# Data type codes in flat PSF TYPE section entries
FLAT_TYPE_REAL    = 0x0b  # 8-byte real (big-endian double)
FLAT_TYPE_COMPLEX = 0x0c  # 16-byte complex (2 × big-endian doubles: real then imag)


def _read_u32(data, pos):
    """Read a big-endian 32-bit unsigned integer."""
    return struct.unpack('>I', data[pos:pos+4])[0], pos + 4


def _read_string(data, pos):
    """Read a length-prefixed ASCII string with 4-byte alignment padding."""
    length, pos = _read_u32(data, pos)
    s = data[pos:pos+length].decode('ascii', errors='replace').rstrip('\x00')
    pos += length
    if length % 4:
        pos += 4 - (length % 4)
    return s, pos


def _read_tag(data, pos):
    """Read a big-endian 32-bit tag."""
    return struct.unpack('>I', data[pos:pos+4])[0], pos + 4


def _read_properties(data, pos, end):
    """Read consecutive property tags and return dict + new pos."""
    props = {}
    while pos < end - 4:
        tag, _ = _read_tag(data, pos)
        if tag == TAG_STRING_PROP:
            pos += 4
            name, pos = _read_string(data, pos)
            value, pos = _read_string(data, pos)
            props[name] = value
        elif tag == TAG_INT_PROP:
            pos += 4
            name, pos = _read_string(data, pos)
            value, pos = _read_u32(data, pos)
            props[name] = value
        elif tag == TAG_FLOAT_PROP:
            pos += 4
            name, pos = _read_string(data, pos)
            value = struct.unpack('>d', data[pos:pos+8])[0]
            pos += 8
            props[name] = value
        else:
            break
    return props, pos


def section_chain(data, header_end):
    """Walk the sections that follow the header: ``sec_type(4) sec_end(4) ...``.

    Returns ``(sections, truncated)`` where *sections* is a list of
    ``(start, end)`` and *truncated* is True when the last section's declared
    end lies past the end of the file; that section is then included with
    its end clipped to the file size.  A file whose sections are all intact
    but whose trailer (offset table + ``Clarissa`` marker) is cut is not
    truncated: its data is complete.
    """
    file_end = len(data)
    sections = []
    pos = header_end
    while pos + 8 <= file_end:
        sec_end = struct.unpack('>I', data[pos + 4:pos + 8])[0]
        if sec_end <= pos:
            break
        if sec_end > file_end:
            sections.append((pos, file_end))
            return sections, True
        sections.append((pos, sec_end))
        pos = sec_end
    return sections, False


def check_file_header(data, path):
    """Raise ValueError unless *data* holds a complete 12-byte PSF file header."""
    if len(data) < 12:
        raise ValueError(f"{path}: truncated or corrupt PSF file "
                         f"({len(data)} bytes, shorter than the file header)")
    header_end = struct.unpack('>I', data[8:12])[0]
    if not 12 <= header_end <= len(data):
        raise ValueError(f"{path}: truncated or corrupt PSF file (header ends at "
                         f"{header_end}, file is {len(data)} bytes)")
    return header_end


def parse_psf_header(header_path):
    """Parse the binary PSF header file to extract signal metadata.

    Supports both PSFXL (.tran) and flat PSF (.ac, .noise, .dc) formats.
    
    Scans for ENTRY (0x10) tags to find sweep and trace definitions.
    For PSFXL: Trace entries have a 0x24 data block (48 bytes of inline statistics).
    For flat PSF: No data block; matched by excluding property and data-block tags.
    
    TYPE section entries carry units properties that are mapped by entry_idx.

    Returns:
        dict with keys:
            'header_props': dict of header properties
            'sweep': dict with 'name', 'units', 'entry_idx'
            'traces': ordered list of trace signal dicts
            'format': 'psfxl' or 'flat'
            'value_section': (start, end) of a flat file's VALUE section
            'truncated': True if that section runs past the end of the file

    Raises ValueError for a file too short or too damaged to parse.
    """
    with open(header_path, 'rb') as f:
        if os.fstat(f.fileno()).st_size < 12:
            raise ValueError(f"{header_path}: truncated or corrupt PSF file "
                             f"(shorter than the file header)")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:

            pos = 0
            magic, pos = _read_u32(data, pos)
            if magic == PSF_MAGIC_INFO:
                raise ValueError(
                    f"PSF info format (magic=0x{magic:08x}) is not handled by "
                    "parse_psf_header — use decode_psf_info() instead."
                )
            if magic not in (PSF_MAGIC, PSF_MAGIC_FLAT):
                raise ValueError(f"Not a PSF header file (magic=0x{magic:08x})")
            is_flat = (magic == PSF_MAGIC_FLAT)

            _, pos = _read_u32(data, pos)  # count
            header_end, pos = _read_u32(data, pos)
            check_file_header(data, header_path)

            # --- HEADER section: properties from offset 12 to header_end ---
            header_props, _ = _read_properties(data, pos, header_end)

            # --- Scan for ENTRY tags from header_end onwards ---
            # TYPE  entries: name + spare_u32 + spare_u32 + [props incl. 'units']
            # SWEEP entries: name + type_ref  + [props incl. 'units']
            # TRACE entries: name + type_ref  + TAG_DATA_BLOCK(4) + 48 bytes + [props]
            # Units live in TYPE entries, not TRACE entries.  Build a type_id->units
            # map first (two passes through the same flat scan) and apply it to traces.
            sweep_name = 'time'
            sweep_units = 's'
            sweep_entry_idx = None
            traces = []
            type_units: dict = {}  # entry_idx -> units string
            type_codes: dict = {}  # entry_idx -> data type code (0x0b=real, 0x0c=complex)

            # GROUP-type tracking for flat PSF noise files.
            # Spectre noise PSF stores per-component data (rs, id, fn …) inline
            # immediately after each device's total value in the VALUE section —
            # without separate TAG_ENTRY markers.  The TYPE section encodes this
            # using "GROUP" entries (TYPE entries with no properties) whose
            # consecutive component TYPE entries (with properties) follow them,
            # terminated by a TYPE entry named "total".
            _grp_eid: int = -1          # entry_idx of the currently-open GROUP type
            _grp_comps: list = []       # components accumulated for that group
            group_type_components: dict = {}  # group_type_id -> [{'name','units','noise_type'}]

            # For flat PSF, find the VALUE section boundary so we don't scan
            # the waveform data itself for ENTRY tags (avoids false positives).
            # Sections are: sec_type(4) + sec_end(4), last one is VALUE.
            file_end = len(data)
            metadata_end = file_end  # default: scan everything (PSFXL headers)
            value_section, truncated = None, False
            if is_flat:
                # The last section is VALUE.  In a truncated file the section
                # that runs past EOF is taken as VALUE, which is checked below:
                # the complete sections before it must have yielded traces.
                sections, truncated = section_chain(data, header_end)
                if sections:
                    value_section = sections[-1]
                    metadata_end = value_section[0]
                else:
                    metadata_end = header_end

            pos = header_end
            entry_tag_bytes = struct.pack('>I', TAG_ENTRY)

            while pos < metadata_end - 16:
                # Search for next ENTRY tag
                idx = data.find(entry_tag_bytes, pos, metadata_end)
                if idx == -1:
                    break

                try:
                    epos = idx + 4
                    entry_idx, epos = _read_u32(data, epos)  # entry index
                    name, epos = _read_string(data, epos)       # signal name
                    type_ref, epos = _read_u32(data, epos)       # type reference

                    # Validate: name should be printable ASCII, reasonable length
                    if not name or len(name) > 100 or not all(
                        32 <= ord(c) < 127 for c in name if c != '\x00'
                    ):
                        pos = idx + 4
                        continue

                    # Determine entry kind from the tag that follows type_ref.
                    next_tag, _ = _read_tag(data, epos)
                    if next_tag == TAG_DATA_BLOCK:
                        # PSFXL TRACE entry: skip TAG_DATA_BLOCK(4) + 48 bytes of statistics
                        epos += 4 + 48
                        props, epos = _read_properties(data, epos, metadata_end)
                        units = type_units.get(type_ref, props.get('units', ''))
                        traces.append({
                            'name': name,
                            'type_id': type_ref,
                            'units': units,
                        })
                    elif is_flat and type_ref != 0 and next_tag not in (
                        TAG_STRING_PROP, TAG_INT_PROP, TAG_FLOAT_PROP, TAG_DATA_BLOCK
                    ):
                        # Flat PSF TRACE entry (ac.ac / noise.noise): no stats block.
                        # type_ref indexes the TYPE entry that carries units and data type.
                        # next_tag may be another TAG_ENTRY or a section boundary — we
                        # match on what it is NOT rather than requiring TAG_ENTRY so that
                        # the last trace in a section is not silently dropped.
                        units = type_units.get(type_ref, '')
                        is_complex = (type_codes.get(type_ref, FLAT_TYPE_COMPLEX) == FLAT_TYPE_COMPLEX)
                        # The record layout in the VALUE section is:
                        #   [TAG][EID][comp_0][comp_1]...[comp_N-2][comp_N-1=total]
                        # where comp_0..comp_N-2 are noise mechanisms (rd, rs, fn…)
                        # and the LAST double is the device total contribution.
                        # group_type_components includes all members up to but NOT
                        # including 'total' (the parser stops there), so we use the
                        # total's units for the main trace and treat comp_0 as the
                        # first inline double at byte offset +8 from the TAG+EID header.
                        inline_comps = group_type_components.get(type_ref, [])
                        # units for the main (total) trace come from the 'total' TYPE entry
                        total_units = type_units.get(type_ref, units)
                        # n_inline = number of component doubles + 1 for total;
                        # all N+1 doubles follow the TAG+EID bytes.
                        n_total_doubles = len(inline_comps) + 1  # comps + total
                        traces.append({
                            'name': name,
                            'type_id': type_ref,
                            'entry_idx': entry_idx,
                            'units': total_units,
                            'complex': is_complex,
                            # False if the TYPE entry was not found and
                            # 'complex' is only a default.
                            'type_known': type_ref in type_codes,
                            # n_inline counts all trailing doubles (comps + total).
                            # The main trace reads the LAST one (total).
                            'n_inline': n_total_doubles,
                            'inline_total_index': n_total_doubles - 1,  # index of total double
                        })
                        # Add one virtual trace entry per noise-mechanism component.
                        for _ci, _comp in enumerate(inline_comps):
                            traces.append({
                                'name': f"{name}.{_comp['name']}",
                                'type_id': type_ref,
                                'entry_idx': entry_idx,
                                'units': _comp['units'],
                                'complex': False,
                                'inline_index': _ci,   # marks this as a virtual sub-trace
                            })
                        # epos remains pointing at the next TAG_ENTRY; don't advance
                    elif next_tag in (TAG_STRING_PROP, TAG_INT_PROP, TAG_FLOAT_PROP):
                        # SWEEP entry: properties follow the single type_ref u32.
                        # Time/frequency sweeps carry an explicit units property;
                        # DC parameter sweeps omit it, so fall back to blank
                        # rather than guessing.
                        props, epos = _read_properties(data, epos, metadata_end)
                        sweep_name = name
                        sweep_units = props.get('units', '')
                        sweep_entry_idx = entry_idx
                    else:
                        # TYPE section entry: two spare u32s precede properties.
                        # type_ref already consumed the first; the second encodes the
                        # data type (0x0b=real, 0x0c=complex) in flat PSF format.
                        data_type_code, epos = _read_u32(data, epos)
                        type_codes[entry_idx] = data_type_code
                        next_tag2, _ = _read_tag(data, epos)
                        if next_tag2 in (TAG_STRING_PROP, TAG_INT_PROP, TAG_FLOAT_PROP):
                            props, epos = _read_properties(data, epos, metadata_end)
                            type_units[entry_idx] = props.get('units', '')
                            # Track as a component of the currently-open GROUP type.
                            if is_flat and _grp_eid >= 0:
                                if name == 'total':
                                    # 'total' terminates the component list. Store the
                                    # preceding mechanism components (NOT including total
                                    # itself — the decoder maps total to the main trace).
                                    group_type_components[_grp_eid] = list(_grp_comps)
                                    # Also store total's units keyed under the GROUP eid
                                    # so the main trace can pick them up.
                                    type_units[_grp_eid] = props.get('units', '')
                                    _grp_eid = -1
                                    _grp_comps = []
                                else:
                                    _grp_comps.append({
                                        'name': name,
                                        'units': props.get('units', ''),
                                        'noise_type': props.get('noise_type', ''),
                                    })
                        elif is_flat:
                            # No properties after the data-type code: this is a GROUP
                            # container type.  Close any previously-open group first.
                            if _grp_eid >= 0:
                                group_type_components[_grp_eid] = list(_grp_comps)
                            _grp_eid = entry_idx
                            _grp_comps = []

                    pos = epos
                except (struct.error, IndexError):
                    pos = idx + 4

            # Close any GROUP whose "total" terminator was never seen (edge case).
            if _grp_eid >= 0 and _grp_comps:
                group_type_components[_grp_eid] = list(_grp_comps)

            if is_flat and truncated and not traces:
                raise ValueError(
                    f"{header_path}: truncated PSF file: it ends inside its "
                    f"metadata sections ({file_end} bytes), so no signals can "
                    f"be named")

            return {
                'value_section': value_section,
                'truncated': truncated,
                'header_props': header_props,
                'sweep': {'name': sweep_name, 'units': sweep_units,
                          'entry_idx': sweep_entry_idx},
                'traces': traces,
                'format': 'flat' if is_flat else 'psfxl',
            }
