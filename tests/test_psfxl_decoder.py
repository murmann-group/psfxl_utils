"""PSFXL decoder tests on synthetic files (see psfxl_builder for the layout).

Each test pins one encoding or failure mode found in real Spectre output.
The decoder's contract under test: every returned signal has exactly the
length of the time axis, values are bit-exact, and anything that could not
be decoded is NaN *and* reported with a warning -- never silently shifted,
dropped or borrowed from another signal.
"""

import struct
import warnings

import numpy as np
import pytest

from psfxl_utils.psfxl_decoder import decode_psfxl

from psfxl_builder import Builder, f8, lz4_literal_block, lz4_repeat_block


def sigs_for(time, n, scale=1.0):
    return [np.sin(time * 1e6 + k) * scale + k for k in range(n)]


def decode(builder, tmp_path, n, **kw):
    path = builder.write(tmp_path / 'tran.tran.tran.psfxl')
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        t, s, nflush = decode_psfxl(path, n)
    for i in range(n):
        assert len(s[i]) == len(t), f"signal {i} misaligned"
    return t, s, nflush, [str(w.message) for w in wl]


def assert_signals(s, expected):
    for i, e in enumerate(expected):
        np.testing.assert_array_equal(s[i], e, err_msg=f"signal {i}")


def two_flushes(enc_first=None, enc_last=None, n=4):
    t1 = np.linspace(0, 9e-9, 10)
    t2 = np.linspace(1e-8, 1.4e-8, 5)
    s1, s2 = sigs_for(t1, n), sigs_for(t2, n, 2.0)
    b = Builder()
    b.flush(t1, s1, enc_first)
    b.end_marker()
    b.flush(t2, s2, enc_last)
    return b, np.concatenate([t1, t2]), [np.concatenate(p) for p in zip(s1, s2)]


def test_all_blosc_roundtrip(tmp_path):
    b, t, s = two_flushes()
    got_t, got_s, nflush, w = decode(b, tmp_path, 4)
    np.testing.assert_array_equal(got_t, t)
    assert_signals(got_s, s)
    assert nflush == 2 and w == []


@pytest.mark.parametrize('enc', [
    {'t': 'raw'},                       # flag "02": time stored uncompressed
    {'t': 'raw', 0: 'raw'},             # no flag: both uncompressed
    {0: 'raw'},                         # flag "20": signal 0 uncompressed
    {'t': 'lz4', 0: 'lz4'},             # flag "11"
], ids=['time-raw', 'both-raw', 'sig0-raw', 'both-lz4'])
def test_seq0_encodings(tmp_path, enc):
    b, t, s = two_flushes(enc_last=enc)
    got_t, got_s, _, w = decode(b, tmp_path, 4)
    np.testing.assert_array_equal(got_t, t)
    assert_signals(got_s, s)
    assert w == []


def test_constant_signal0_and_constant_zero(tmp_path):
    t1 = np.linspace(0, 9e-9, 10)
    s1 = [np.full(10, 1.8), np.zeros(10), np.full(10, -0.5)]
    b = Builder()
    b.flush(t1, s1, {0: 'const', 1: 'const', 2: 'const'})
    got_t, got_s, _, w = decode(b, tmp_path, 3)
    assert_signals(got_s, s1)
    assert w == []


def test_raw_signal_in_last_flush_does_not_borrow_next_block(tmp_path):
    """Regression: an uncompressed seq>=1 block made the old decoder search
    onward and decode the *next* signal's block in its place."""
    b, t, s = two_flushes(enc_last={1: 'raw'})
    _, got_s, _, w = decode(b, tmp_path, 4)
    assert_signals(got_s, s)
    assert not np.array_equal(got_s[1][-5:], got_s[2][-5:])
    assert w == []


def test_lz4_signal_block(tmp_path):
    b, t, s = two_flushes(enc_last={2: 'lz4'})
    _, got_s, _, w = decode(b, tmp_path, 4)
    assert_signals(got_s, s)
    assert w == []


def test_lz4_block_with_overlapping_match():
    from psfxl_utils.psfxl_decoder import _lz4_pure
    out = _lz4_pure(lz4_repeat_block(0.675, 1.0125), 32)
    np.testing.assert_array_equal(np.frombuffer(out, '<f8'),
                                  [0.675, 0.675, 0.675, 1.0125])
    with pytest.raises(ValueError):
        _lz4_pure(lz4_repeat_block(0.675, 1.0125), 40)


def test_back_reference_resolves_to_earlier_block(tmp_path):
    t1 = np.linspace(0, 9e-9, 10)
    s1 = sigs_for(t1, 3)
    s1[2] = s1[1].copy()                  # identical waveform -> back-reference
    b = Builder()
    b.flush(t1, s1, {2: ('ref', 'sig1')})
    _, got_s, _, w = decode(b, tmp_path, 3)
    assert_signals(got_s, s1)
    assert w == []


def test_unresolved_back_reference_is_nan_and_warned(tmp_path):
    t1 = np.linspace(0, 9e-9, 10)
    t2 = np.linspace(1e-8, 1.4e-8, 5)
    b = Builder()
    b.flush(t1, sigs_for(t1, 3))
    off = b.flush(t2, sigs_for(t2, 3))
    # Corrupt signal 2 of the last flush into a back-reference to nowhere.
    b.buf = b.buf[:off[2][0]]
    b.raw_t3(2, 5, b'', 'aa', ['0.0', '7.33'])
    _, got_s, _, w = decode(b, tmp_path, 3)
    assert np.isnan(got_s[2][-5:]).all()
    assert not np.isnan(got_s[2][:10]).any()
    assert any('could not be decoded' in m and 'seq 2' in m for m in w)


def test_leading_type1_flush_is_kept(tmp_path):
    """Regression: a first flush stored as type-1 (t = 0 ...) was discarded
    as a 'checkpoint' because it precedes the type-3 data."""
    t0 = np.linspace(0, 6e-9, 7)
    t1 = np.linspace(7e-9, 1.6e-8, 10)
    s0, s1 = sigs_for(t0, 3), sigs_for(t1, 3)
    b = Builder()
    b.type1(t0, s0)
    b.flush(t1, s1)
    got_t, got_s, nflush, w = decode(b, tmp_path, 3)
    np.testing.assert_array_equal(got_t, np.concatenate([t0, t1]))
    assert_signals(got_s, [np.concatenate(p) for p in zip(s0, s1)])
    assert nflush == 2 and w == []


def test_trailing_type2_flush_starting_at_zero_time_value(tmp_path):
    """Type-2 payload located by pad arithmetic: a first sample of 0.0 must
    not be swallowed as padding."""
    t2 = np.array([0.0, 1e-9, 2e-9])
    s2 = [np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.5, 0.0])]
    b = Builder()
    b.type2(t2, s2)
    got_t, got_s, _, w = decode(b, tmp_path, 2)
    np.testing.assert_array_equal(got_t, t2)
    assert_signals(got_s, s2)


def test_trailing_type2_after_type3(tmp_path):
    t1 = np.linspace(0, 9e-9, 10)
    t2 = np.array([1e-8, 1.1e-8])
    s1, s2 = sigs_for(t1, 3), sigs_for(t2, 3)
    b = Builder()
    b.flush(t1, s1)
    b.type2(t2, s2)
    got_t, got_s, nflush, w = decode(b, tmp_path, 3)
    np.testing.assert_array_equal(got_t, np.concatenate([t1, t2]))
    assert nflush == 2 and w == []


def test_covered_checkpoint_skipped_and_overlap_trimmed(tmp_path):
    t1 = np.linspace(0, 9e-9, 10)
    s1 = sigs_for(t1, 2)
    b = Builder()
    b.flush(t1, s1)
    b.type2(t1[2:5], [x[2:5] for x in s1])        # fully covered checkpoint
    t2 = np.concatenate([t1[8:], [1e-8, 1.1e-8, 1.2e-8]])   # overlaps t1's last 2
    b.type2(t2, sigs_for(t2, 2))
    got_t, got_s, nflush, w = decode(b, tmp_path, 2)
    assert np.all(np.diff(got_t) > 0)
    assert len(got_t) == 10 + 3
    assert any('skipped' in m and 'trimmed' in m for m in w)


def test_truncated_final_flush_is_nan_filled_and_warned(tmp_path):
    b, t, s = two_flushes()
    last_hdr = b.buf.rfind(b'\x003:3:')
    payload = b.buf.index(b'\n', last_hdr) + 1
    payload += 8 - payload % 8
    b.buf = b.buf[:payload + 5]            # cut inside the last block's payload
    _, got_s, _, w = decode(b, tmp_path, 4)
    assert np.isnan(got_s[3][-5:]).all()
    assert_signals(got_s, s[:3])
    assert any('could not be decoded' in m and 'past end of file' in m for m in w)


def test_marker_bytes_inside_raw_payload_are_not_headers(tmp_path):
    """b'\\x003:' inside a raw block must not start a phantom chunk."""
    fake = b'\x003:1:0:5:8:a0\n\x00\x00'              # 16 bytes of 'data'
    t1 = np.linspace(0, 3e-9, 4)
    bogus = np.frombuffer(fake * 2, '<f8')             # 4 doubles
    s1 = [np.ones(4), bogus.copy(), np.full(4, 2.0)]
    b = Builder()
    b.flush(t1, s1, {1: 'raw'})
    _, got_s, _, w = decode(b, tmp_path, 3)
    assert_signals(got_s, s1)
    assert w == []


def test_unknown_flag_is_nan_and_warned(tmp_path):
    b, t, s = two_flushes()
    i = b.buf.rfind(b':a2:')
    b.buf[i + 1:i + 3] = b'a7'
    _, got_s, _, w = decode(b, tmp_path, 4)
    assert np.isnan(got_s[3][-5:]).all()
    assert any("unknown encoding digit '7'" in m for m in w)


def test_type1_with_fewer_signals_than_header_is_nan_padded(tmp_path):
    t0 = np.linspace(0, 6e-9, 7)
    b = Builder()
    b.type1(t0, sigs_for(t0, 2))
    _, got_s, _, w = decode(b, tmp_path, 3)
    assert np.isnan(got_s[2]).all()
    assert any('holds 2 of 3 signals' in m for m in w)


def test_no_data_warns_and_returns_empty(tmp_path):
    b = Builder()
    path = b.write(tmp_path / 'x.psfxl')
    with pytest.warns(RuntimeWarning, match='no decodable waveform'):
        t, s, nflush = decode_psfxl(path, 2)
    assert len(t) == 0 and s == {} and nflush == 0


def test_out_of_order_checkpoint_does_not_displace_covering_flush(tmp_path):
    """A checkpoint for t=13..15 ns appearing in the file before the type-3
    flush for 10..15 ns must not leave a gap or drop that flush."""
    t1 = np.linspace(0, 9e-9, 10)
    t2 = np.linspace(1e-8, 1.5e-8, 6)
    s1, s2 = sigs_for(t1, 2), sigs_for(t2, 2)
    b = Builder()
    b.flush(t1, s1)
    b.type2(t2[3:], [x[3:] for x in s2])
    b.flush(t2, s2)
    got_t, got_s, _, w = decode(b, tmp_path, 2)
    np.testing.assert_array_equal(got_t, np.concatenate([t1, t2]))
    assert_signals(got_s, [np.concatenate(p) for p in zip(s1, s2)])


def test_oversized_stray_header_does_not_hide_later_flush(tmp_path):
    """A header whose declared payload runs past EOF must not mask the
    type-2 flush after it."""
    t1 = np.linspace(0, 9e-9, 10)
    t2 = np.linspace(1e-8, 1.4e-8, 5)
    b = Builder()
    b.flush(t1, sigs_for(t1, 2))
    b.buf += b'\x003:0:ffffffffffffffff:5:fffff:22:10:10\n'
    b.type2(t2, sigs_for(t2, 2))
    got_t, _, _, w = decode(b, tmp_path, 2)
    np.testing.assert_array_equal(got_t, np.concatenate([t1, t2]))


def test_lz4_fallback_rejects_malformed_blocks():
    from psfxl_utils.psfxl_decoder import _lz4_pure
    for bad in (b'\x1f\x00\x01\x00\xff\xff\xff\xff\xff', b'\xf0', b'\x40ab'):
        with pytest.raises(ValueError):
            _lz4_pure(bad, 32)
    with pytest.raises(ValueError):           # would expand far beyond 32 bytes
        _lz4_pure(b'\x1f\x00\x01\x00' + b'\xff' * 100 + b'\x00', 32)
