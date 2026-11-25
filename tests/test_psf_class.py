"""PSF construction: format dispatch and error reporting."""

import numpy as np
import pytest

from psfxl_utils import PSF, psf_class


@pytest.fixture
def header_file(tmp_path, monkeypatch):
    monkeypatch.setenv('PSFXL_UTILS_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(psf_class.PSF, '_instance_cache', {})
    path = tmp_path / 'r.tran.tran'
    path.write_bytes(b'\x00\x00\x04\x00header')
    return path


def fake_header(monkeypatch, window):
    monkeypatch.setattr(psf_class, 'parse_psf_header', lambda p: {
        'header_props': {'PSF window size': window, 'PSF buffer size': 0},
        'traces': [{'name': 'v', 'units': 'V'}],
        'sweep': {'name': 'time', 'units': 's'}, 'format': 'psfxl'})


def test_windowed_classic_psf_is_reported_as_unsupported(header_file, monkeypatch):
    fake_header(monkeypatch, 4096)
    with pytest.raises(ValueError, match='classic windowed binary PSF'):
        PSF(str(header_file))


def test_missing_psfxl_data_file_is_named(header_file, monkeypatch):
    fake_header(monkeypatch, 0)
    with pytest.raises(FileNotFoundError, match=r'r\.tran\.tran\.psfxl is missing'):
        PSF(str(header_file))


def test_psfxl_header_with_window_size_zero_decodes(header_file, monkeypatch):
    fake_header(monkeypatch, 0)
    (header_file.parent / 'r.tran.tran.psfxl').write_bytes(b'x')
    monkeypatch.setattr(psf_class, 'decode_psfxl',
                        lambda p, n: (np.arange(2.0), {0: np.ones(2)}, 1))
    assert len(PSF(str(header_file)).get_signal('v').ordinate) == 2


def test_info_strings_are_exposed_separately(tmp_path, monkeypatch):
    """stb_state-style strings come back through PSF.strings, never as signals."""
    import struct
    monkeypatch.setenv('PSFXL_UTILS_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(psf_class.PSF, '_instance_cache', {})

    def s(text):
        b = text.encode() + b'\x00' * (-len(text) % 4)
        return struct.pack('>I', len(text)) + b

    def entry(eid, name, *rest):
        return struct.pack('>II', 0x10, eid) + s(name) + b''.join(rest)

    types = (entry(52, 'dB', struct.pack('>II', 0, 0x0b),
                   struct.pack('>I', 0x21), s('key'), s('node'))
             + entry(55, 'str', struct.pack('>II', 0, 0x0e),
                     struct.pack('>I', 0x21), s('key'), s('node')))
    values = (entry(56, 'gainMargin', struct.pack('>Id', 52, 18.5),
                    struct.pack('>I', 0x21), s('gainMarginInfo'), s('Gain margin = 18.5 dB'))
              + entry(60, 'stb_state', struct.pack('>I', 55), s('The circuit is stable.')))

    hdr_end = 12                     # each section: 0x15 end, then 0x16 group
    t_end = hdr_end + 16 + len(types)
    v_end = t_end + 16 + len(values)
    t_body = struct.pack('>II', 0x16, t_end) + types
    v_body = struct.pack('>II', 0x16, v_end) + values
    data = (struct.pack('>III', 0x500, 0, hdr_end)
            + struct.pack('>II', 0x15, t_end) + t_body
            + struct.pack('>II', 0x15, v_end) + v_body)
    p = tmp_path / 'stb.margin.stb'
    p.write_bytes(data)
    psf = PSF(str(p))
    assert [x.name for x in psf.all_signals()] == ['gainMargin']
    assert psf.get_signal('gainMargin').ordinate[0] == 18.5
    assert psf.get_string('stb_state') == 'The circuit is stable.'
    assert psf.strings['gainMargin.gainMarginInfo'] == 'Gain margin = 18.5 dB'
    with pytest.raises(KeyError):
        psf.get_string('nope')
    psf_class.PSF._instance_cache.clear()
    assert PSF(str(p)).get_string('stb_state') == 'The circuit is stable.'   # disk hit
