"""Invariants over a directory of real Spectre results (opt-in).

Set ``PSFXL_TEST_CORPUS`` to a directory; every PSFXL result (``*.psfxl``
beside its header) and flat PSF file (magic 0x200) found under it is
decoded, and the test fails on the first file that breaks an invariant:

* every signal has exactly the length of the sweep;
* a PSFXL time axis is strictly increasing, has no NaN in any signal, and
  matches the sweep count in the ``.sig`` file when there is one;
* decoding emits no RuntimeWarning (each one marks data the decoder could
  not read);
* a flat file has as many points as its header's ``PSF sweep points``;
* an operating-point (0x500) file decodes to at least one signal.

Optional limits: ``PSFXL_TEST_CORPUS_MAX_MB`` (default 300) skips larger
data files.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from psfxl_utils.flat_psf_decoder import decode_psf_flat
from psfxl_utils.oppoint_psf_decoder import decode_psf_info
from psfxl_utils.psf_parse import parse_psf_header
from psfxl_utils.psfxl_decoder import decode_psfxl, parse_sig_file

CORPUS = os.environ.get('PSFXL_TEST_CORPUS', '').strip()
MAX_BYTES = float(os.environ.get('PSFXL_TEST_CORPUS_MAX_MB', '300')) * 1e6

pytestmark = pytest.mark.skipif(not CORPUS, reason='set PSFXL_TEST_CORPUS to run')


def _results():
    for root, _dirs, files in os.walk(CORPUS):
        for name in sorted(files):
            p = Path(root) / name
            if name.endswith('.psfxl'):
                if p.stat().st_size <= MAX_BYTES:
                    yield 'psfxl', Path(str(p)[:-len('.psfxl')])
                continue
            try:
                with open(p, 'rb') as f:
                    magic = f.read(4)
            except OSError:
                continue
            if magic == b'\x00\x00\x02\x00':
                yield 'flat', p
            elif magic == b'\x00\x00\x05\x00':
                yield 'info', p


def _check(kind, path):
    if kind == 'info':
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            _t, sig, _traces = decode_psf_info(path)
        problems = [str(w.message) for w in wl if issubclass(w.category, RuntimeWarning)]
        if not sig:
            problems.append("operating-point file decoded to no signals")
        return problems
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        header = parse_psf_header(path)
        n = len(header['traces'])
        if kind == 'psfxl':
            t, sig, _ = decode_psfxl(Path(str(path) + '.psfxl'), n)
        else:
            t, sig = decode_psf_flat(path, header)
    problems = [str(w.message) for w in wl if issubclass(w.category, RuntimeWarning)]
    bad = {i: len(a) for i, a in sig.items() if len(a) != len(t)}
    if bad:
        problems.append(f"signal lengths {bad} != sweep length {len(t)}")
    if kind == 'flat':
        expected = header['header_props'].get('PSF sweep points', 0)
        if isinstance(expected, int) and expected > 0 and expected != len(t):
            problems.append(f"header says {expected} sweep points, decoded {len(t)}")
    if kind == 'psfxl':
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            problems.append("time axis not strictly increasing")
        nan = [i for i, a in sig.items() if np.isnan(a).any()]
        if nan:
            problems.append(f"NaN in signals {nan[:10]}")
        sig_path = Path(str(path) + '.sig')
        if sig_path.exists():
            count = parse_sig_file(sig_path).get('cdnshsweepcount')
            if count and count.strip().isdigit() and int(count) != len(t):
                problems.append(f".sig sweep count {count} != {len(t)} points")
    return problems


def test_corpus_invariants():
    checked, failures = 0, []
    for kind, path in _results():
        checked += 1
        problems = _check(kind, path)
        if problems:
            failures.append(f"{path}: {'; '.join(problems)}")
    assert checked, f"no results found under {CORPUS}"
    assert not failures, f"{len(failures)} of {checked} files:\n" + '\n'.join(failures[:20])
