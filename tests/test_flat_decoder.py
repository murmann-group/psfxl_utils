"""Flat PSF VALUE-section decoding from the header-derived record layout.

The VALUE section holds, per sweep point, one record per entry:
TAG_ENTRY(0x10, 4) + entry index (4) + payload, big-endian.  Payload is one
double (real), two (complex: real, imag) or, for a GROUP noise device, its
component doubles followed by its total.  A single-point file is followed
by the 0x0f end tag instead of a second point, so record sizes cannot be
inferred from where the next record starts; they come from the header.
"""

import struct

import numpy as np
import pytest

from psfxl_utils.flat_psf_decoder import _decode_by_header_layout

SWEEP = 1
TRACES = [
    {'name': 'v_out', 'entry_idx': 10, 'complex': True, 'type_known': True,
     'n_inline': 1, 'inline_total_index': 0},
    {'name': 'M0', 'entry_idx': 11, 'complex': False, 'type_known': True,
     'n_inline': 3, 'inline_total_index': 2},
    {'name': 'M0.rd', 'entry_idx': 11, 'complex': False, 'inline_index': 0},
    {'name': 'M0.id', 'entry_idx': 11, 'complex': False, 'inline_index': 1},
    {'name': 'R1', 'entry_idx': 12, 'complex': False, 'type_known': True,
     'n_inline': 1, 'inline_total_index': 0},
]


def value_section(points):
    """points: list of (freq, v_out complex, (rd, id, total), r1)."""
    buf = b''
    for f, v, (rd, idn, tot), r1 in points:
        buf += struct.pack('>IId', 0x10, SWEEP, f)
        buf += struct.pack('>IIdd', 0x10, 10, v.real, v.imag)
        buf += struct.pack('>IIddd', 0x10, 11, rd, idn, tot)
        buf += struct.pack('>IId', 0x10, 12, r1)
    return buf + struct.pack('>I', 0x0f)


POINTS = [(1e3, 1 - 2j, (1e-18, 3e-18, 4e-18), 5e-19),
          (1e4, 0.5 + 0.25j, (2e-18, 1e-18, 3e-18), 6e-19),
          (1e5, -1e-6 - 2e-12j, (0.0, 7e-18, 7e-18), 0.0)]


@pytest.mark.parametrize('n', [1, 2, 3])
def test_header_layout_decodes_any_point_count(n):
    data = value_section(POINTS[:n])
    sweep, sig = _decode_by_header_layout(data, 0, len(data), SWEEP, TRACES)
    np.testing.assert_array_equal(sweep, [p[0] for p in POINTS[:n]])
    np.testing.assert_array_equal(sig[0], [p[1] for p in POINTS[:n]])
    assert np.iscomplexobj(sig[0])
    np.testing.assert_array_equal(sig[1], [p[2][2] for p in POINTS[:n]])  # total
    np.testing.assert_array_equal(sig[2], [p[2][0] for p in POINTS[:n]])  # rd
    np.testing.assert_array_equal(sig[3], [p[2][1] for p in POINTS[:n]])  # id
    np.testing.assert_array_equal(sig[4], [p[3] for p in POINTS[:n]])


def test_unknown_type_falls_back():
    traces = [dict(t) for t in TRACES]
    traces[0]['type_known'] = False
    data = value_section(POINTS[:1])
    assert _decode_by_header_layout(data, 0, len(data), SWEEP, traces) is None


def test_record_not_in_header_falls_back():
    data = value_section(POINTS[:2]).replace(struct.pack('>II', 0x10, 12),
                                             struct.pack('>II', 0x10, 99))
    assert _decode_by_header_layout(data, 0, len(data), SWEEP, TRACES) is None


def test_inconsistent_later_point_falls_back():
    data = bytearray(value_section(POINTS))
    stride = (len(data) - 4) // 3
    struct.pack_into('>I', data, 2 * stride + 4, 12)   # corrupt point 3's sweep EID
    assert _decode_by_header_layout(bytes(data), 0, len(data), SWEEP, TRACES) is None


# --- Truncated files --------------------------------------------------------

from psfxl_utils.psf_parse import section_chain


def test_truncated_value_section_returns_complete_points_only():
    data = value_section(POINTS)[:-4]            # drop the end tag ...
    cut = data[:len(data) - 10]                   # ... and cut into point 3
    sweep, sig = _decode_by_header_layout(cut, 0, len(cut), SWEEP, TRACES,
                                          truncated=True)
    np.testing.assert_array_equal(sweep, [p[0] for p in POINTS[:2]])
    assert all(len(a) == 2 for a in sig.values())


@pytest.mark.parametrize('cut_at', [20, 40, 60])
def test_truncation_inside_first_point_is_refused(cut_at):
    data = value_section(POINTS)[:cut_at]
    assert _decode_by_header_layout(data, 0, len(data), SWEEP, TRACES,
                                    truncated=True) is None


def test_section_chain_flags_a_section_running_past_eof():
    hdr = struct.pack('>III', 0x200, 0, 12)
    s1 = struct.pack('>II', 0x15, 12 + 16) + b'\x00' * 8
    s2 = struct.pack('>II', 0x15, 28 + 100) + b'\x00' * 20      # cut short
    sections, truncated = section_chain(hdr + s1 + s2, 12)
    assert truncated and sections == [(12, 28), (28, 56)]
    sections, truncated = section_chain(hdr + s1, 12)
    assert not truncated and sections == [(12, 28)]


def test_truncated_info_file_raises(tmp_path):
    from psfxl_utils.oppoint_psf_decoder import decode_psf_info
    p = tmp_path / 'dcOp.dc'
    p.write_bytes(struct.pack('>III', 0x500, 0, 12) + struct.pack('>II', 0x15, 500))
    with pytest.raises(ValueError, match='truncated'):
        decode_psf_info(p)
    p.write_bytes(b'\x00\x00\x05')
    with pytest.raises(ValueError, match='truncated'):
        decode_psf_info(p)


def _minimal_flat(trailer):
    """A flat file with intact sections, no traces, with/without trailer."""
    body = struct.pack('>III', 0x200, 0, 12)
    body += struct.pack('>II', 0x15, 12 + 16) + b'\x00' * 8
    if trailer:
        # offset table (its second word points back, ending the section
        # walk, as in real files), then the marker
        body += struct.pack('>II', 0x15, 12) + b'Clarissa' + struct.pack('>I', 8)
    return body


def test_traceless_file_without_trailer_is_truncated(tmp_path):
    from psfxl_utils.psf_parse import parse_psf_header
    p = tmp_path / 'ac.ac'
    p.write_bytes(_minimal_flat(trailer=False))
    with pytest.raises(ValueError, match='truncated'):
        parse_psf_header(p)
    p.write_bytes(_minimal_flat(trailer=True))      # complete, just no traces
    assert parse_psf_header(p)['traces'] == []


@pytest.mark.parametrize('size', [0, 3])
def test_tiny_file_through_psf_is_a_value_error(tmp_path, monkeypatch, size):
    from psfxl_utils import PSF, psf_class
    monkeypatch.setenv('PSFXL_UTILS_CACHE_DIR', str(tmp_path / 'c'))
    monkeypatch.setattr(psf_class.PSF, '_instance_cache', {})
    p = tmp_path / 'dcOp.dc'
    p.write_bytes(b'\x00\x00\x05'[:size])
    with pytest.raises(ValueError, match='truncated'):
        PSF(str(p))
