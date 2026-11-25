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
