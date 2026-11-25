"""Tests for the disk cache location and its PSFXL_UTILS_CACHE_DIR override.

The end-to-end test needs a real flat PSF result file.  Point
``PSFXL_TEST_PSF_FILE`` at one (e.g. an ``ac.ac``) to run it; it is skipped
otherwise.
"""

import os
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import pytest

from psfxl_utils import psf_cache

ENV = psf_cache.CACHE_DIR_ENV


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    """Never let a test touch the real ~/.cache/psfxl_utils, even if the
    override under test is broken."""
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv(ENV, raising=False)
    psf_cache._warned.clear()  # once-per-process warnings: reset per test
    return home


@pytest.fixture
def src_file(tmp_path):
    p = tmp_path / 'src' / 'result.ac'
    p.parent.mkdir()
    p.write_bytes(b'dummy')
    return p


def test_env_var_name():
    assert ENV == 'PSFXL_UTILS_CACHE_DIR'


def test_default_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert psf_cache.get_cache_dir() == tmp_path / '.cache' / 'psfxl_utils'


def test_default_when_empty(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, '')
    monkeypatch.setenv('HOME', str(tmp_path))
    assert psf_cache.get_cache_dir() == tmp_path / '.cache' / 'psfxl_utils'


def test_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    assert psf_cache.get_cache_dir() == tmp_path / 'c'


def test_override_expands_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv(ENV, '~/mycache')
    assert psf_cache.get_cache_dir() == tmp_path / 'mycache'


def test_override_relative_is_made_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV, 'rel')
    d = psf_cache.get_cache_dir()
    assert d.is_absolute() and d == tmp_path / 'rel'


def test_override_read_at_call_time(monkeypatch, tmp_path, src_file):
    monkeypatch.setenv(ENV, str(tmp_path / 'a'))
    assert psf_cache.get_cache_path(str(src_file)).parent == tmp_path / 'a'
    monkeypatch.setenv(ENV, str(tmp_path / 'b'))
    assert psf_cache.get_cache_path(str(src_file)).parent == tmp_path / 'b'


def test_roundtrip_in_override_dir(monkeypatch, tmp_path, src_file):
    cache = tmp_path / 'nested' / 'cache'
    monkeypatch.setenv(ENV, str(cache))
    psf_cache.save_to_cache(str(src_file), {'x': 1})
    assert len(list(cache.glob('*.pkl'))) == 1
    assert psf_cache.load_from_cache(str(src_file)) == {'x': 1}
    assert not (tmp_path / 'home' / '.cache' / 'psfxl_utils').exists()
    monkeypatch.setenv(ENV, str(tmp_path / 'other'))
    assert psf_cache.load_from_cache(str(src_file)) is None


def test_stale_entry_purged_in_override_dir(monkeypatch, tmp_path, src_file):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 1)
    st = src_file.stat()
    os.utime(src_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    psf_cache.save_to_cache(str(src_file), 2)
    entries = list((tmp_path / 'c').glob('*.pkl'))
    assert len(entries) == 1
    assert psf_cache.load_from_cache(str(src_file)) == 2


def test_unwritable_location_is_non_fatal(monkeypatch, tmp_path, src_file):
    blocker = tmp_path / 'file'
    blocker.write_text('not a dir')
    monkeypatch.setenv(ENV, str(blocker / 'cache'))
    assert psf_cache.get_cache_path(str(src_file)) is None
    assert psf_cache.load_from_cache(str(src_file)) is None
    psf_cache.save_to_cache(str(src_file), 1)  # must not raise


def test_clear_all_only_removes_pickles(monkeypatch, tmp_path, src_file):
    shared = tmp_path / 'shared'
    shared.mkdir(mode=0o700)
    (shared / 'keep.txt').write_text('user data')
    (shared / 'my_results.pkl').write_bytes(b'user pickle')
    (shared / 'sub').mkdir()
    (shared / 'sub' / 'deep.pkl').write_bytes(b'x')
    monkeypatch.setenv(ENV, str(shared))
    psf_cache.save_to_cache(str(src_file), 1)
    assert list(shared.glob('*.pkl'))

    psf_cache.clear_all()

    assert shared.is_dir()
    assert sorted(p.name for p in shared.glob('*.pkl')) == ['my_results.pkl']
    assert (shared / 'keep.txt').read_text() == 'user data'
    assert (shared / 'sub' / 'deep.pkl').exists()


def test_clear_all_missing_dir_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / 'absent'))
    psf_cache.clear_all()
    assert not (tmp_path / 'absent').exists()


@pytest.mark.skipif(not os.environ.get('PSFXL_TEST_PSF_FILE'),
                    reason='set PSFXL_TEST_PSF_FILE to a flat PSF result file')
def test_psf_end_to_end_uses_override(monkeypatch, tmp_path):
    from psfxl_utils import PSF

    src = tmp_path / Path(os.environ['PSFXL_TEST_PSF_FILE']).name
    shutil.copy(os.environ['PSFXL_TEST_PSF_FILE'], src)
    cache = tmp_path / 'cache'
    monkeypatch.setenv(ENV, str(cache))
    monkeypatch.setattr(PSF, '_instance_cache', {})

    first = PSF(str(src))
    assert len(list(cache.glob('*.pkl'))) == 1

    # New "session": empty memory cache, so this must come from disk.
    monkeypatch.setattr(PSF, '_instance_cache', {})
    loaded = []
    real_load = psf_cache.load_from_cache
    monkeypatch.setattr(psf_cache, 'load_from_cache',
                        lambda *a, **k: loaded.append(real_load(*a, **k)) or loaded[-1])
    second = PSF(str(src))
    assert loaded and loaded[0] is not None
    assert [s.name for s in second._signals] == [s.name for s in first._signals]
    assert (second._sweep.abscissa == first._sweep.abscissa).all()


def test_default_location_roundtrip(isolated_home, src_file):
    psf_cache.save_to_cache(str(src_file), 7)
    d = isolated_home / '.cache' / 'psfxl_utils'
    assert len(list(d.glob('*.pkl'))) == 1
    assert psf_cache.load_from_cache(str(src_file)) == 7
    psf_cache.clear_all()
    assert not list(d.glob('*.pkl'))


def test_blank_value_means_default(monkeypatch, isolated_home):
    monkeypatch.setenv(ENV, '   ')
    assert psf_cache.get_cache_dir() == isolated_home / '.cache' / 'psfxl_utils'


def test_unknown_tilde_user_falls_back(monkeypatch, isolated_home, src_file):
    monkeypatch.setenv(ENV, '~nosuchuser_psfxl_zz/cache')
    with pytest.warns(RuntimeWarning):
        d = psf_cache.get_cache_dir()
    assert d == isolated_home / '.cache' / 'psfxl_utils'
    with warnings.catch_warnings():
        warnings.simplefilter('error')        # reported once per process
        psf_cache.save_to_cache(str(src_file), 1)
        assert psf_cache.load_from_cache(str(src_file)) == 1
        psf_cache.clear_all()


def test_created_override_dir_is_private(monkeypatch, tmp_path, src_file):
    monkeypatch.setenv(ENV, str(tmp_path / 'new'))
    psf_cache.save_to_cache(str(src_file), 1)
    assert (tmp_path / 'new').stat().st_mode & 0o077 == 0


@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='POSIX only')
def test_group_writable_override_disables_cache(monkeypatch, tmp_path,
                                                src_file):
    shared = tmp_path / 'shared'
    shared.mkdir()
    shared.chmod(0o777)
    monkeypatch.setenv(ENV, str(shared))
    with pytest.warns(psf_cache.PSFCacheWarning,
                      match='not owned by you or is writable'):
        assert psf_cache.get_cache_path(str(src_file)) is None
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # once per directory: no repeat
        psf_cache.save_to_cache(str(src_file), 1)
    assert not list(shared.glob('*.pkl'))


def test_group_writable_default_dir_still_cached(isolated_home, src_file):
    d = isolated_home / '.cache' / 'psfxl_utils'
    d.mkdir(parents=True)
    d.chmod(0o775)
    psf_cache.save_to_cache(str(src_file), 3)
    assert psf_cache.load_from_cache(str(src_file)) == 3


@pytest.mark.parametrize('name', ['run[1].ac', 'a**b.ac'])
def test_glob_metachar_stem_cached_and_purged(monkeypatch, tmp_path, name):
    src = tmp_path / name
    src.write_bytes(b'x')
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src), 1)
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    psf_cache.save_to_cache(str(src), 2)
    assert len(list((tmp_path / 'c').iterdir())) == 1
    assert psf_cache.load_from_cache(str(src)) == 2


def test_cache_dir_attribute_compat(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    assert psf_cache.CACHE_DIR == tmp_path / 'c'


@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='POSIX only')
def test_override_owned_by_other_user_disables_cache(monkeypatch, tmp_path,
                                                     src_file):
    d = tmp_path / 'c'
    d.mkdir(mode=0o700)
    monkeypatch.setenv(ENV, str(d))
    real_uid = os.getuid()
    monkeypatch.setattr(os, 'getuid', lambda: real_uid + 1)
    with pytest.warns(RuntimeWarning, match='not owned by you'):
        assert psf_cache.get_cache_path(str(src_file)) is None


@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='POSIX only')
def test_entry_swapped_after_dir_check_is_not_loaded(monkeypatch, tmp_path,
                                                     src_file):
    """Simulate another user's entry appearing after the directory check:
    the open file is checked by fstat, so it must be refused."""
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 'mine')
    real_fstat = os.fstat

    class Foreign:
        def __init__(self, st):
            self.st_mode, self.st_uid = st.st_mode, st.st_uid + 1

    monkeypatch.setattr(os, 'fstat', lambda fd: Foreign(real_fstat(fd)))
    with pytest.warns(RuntimeWarning, match='ignored'):
        assert psf_cache.load_from_cache(str(src_file)) is None


@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='POSIX only')
def test_group_writable_entry_refused_everywhere(monkeypatch, tmp_path,
                                                 isolated_home, src_file):
    """Entries are written 0600, so a group-writable one was not written by
    this library: refused in the default directory too."""
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 1)
    entry, = (tmp_path / 'c').glob('*.pkl')
    entry.chmod(0o664)
    with pytest.warns(psf_cache.PSFCacheWarning, match='ignored'):
        assert psf_cache.load_from_cache(str(src_file)) is None

    monkeypatch.delenv(ENV)
    psf_cache.save_to_cache(str(src_file), 2)
    entry, = (isolated_home / '.cache' / 'psfxl_utils').glob('*.pkl')
    entry.chmod(0o664)
    with pytest.warns(psf_cache.PSFCacheWarning, match='ignored'):
        assert psf_cache.load_from_cache(str(src_file)) is None


def test_default_dir_spelled_differently_is_exempt(monkeypatch, isolated_home,
                                                   src_file):
    d = isolated_home / '.cache' / 'psfxl_utils'
    d.mkdir(parents=True)
    d.chmod(0o775)
    monkeypatch.setenv(ENV, str(isolated_home / '.cache' / '..' / '.cache'
                               / 'psfxl_utils'))
    psf_cache.save_to_cache(str(src_file), 4)
    assert psf_cache.load_from_cache(str(src_file)) == 4


def test_clear_all_removes_nohash_entries_and_skips_symlinks(monkeypatch,
                                                             tmp_path):
    d = tmp_path / 'c'
    d.mkdir()
    (d / 'a_b_01234567_nohash_5.pkl').write_bytes(b'x')
    target = tmp_path / 'precious.pkl'
    target.write_bytes(b'keep')
    (d / 'x_01234567_89abcdef_6.pkl').symlink_to(target)
    monkeypatch.setenv(ENV, str(d))
    psf_cache.clear_all()
    assert not (d / 'a_b_01234567_nohash_5.pkl').exists()
    assert (d / 'x_01234567_89abcdef_6.pkl').is_symlink()
    assert target.read_bytes() == b'keep'


# --- Entry permissions, atomic writes and temp-file cleanup ----------------

posix_only = pytest.mark.skipif(not hasattr(os, 'getuid'), reason='POSIX only')


@pytest.fixture
def umask():
    """Set the process umask for one test, restoring it afterwards."""
    saved = os.umask(0o022)
    try:
        yield os.umask
    finally:
        os.umask(saved)


def _entry(d):
    entry, = d.glob('*.pkl')
    return entry


def _temp_files(d):
    return sorted(p.name for p in d.iterdir() if p.name.endswith('.tmp'))


@posix_only
@pytest.mark.parametrize('mask', [0o002, 0o007, 0o022, 0o077], ids=oct)
def test_override_cache_hits_under_any_umask(monkeypatch, tmp_path, src_file,
                                             umask, mask):
    """Regression: under umask 002/007 entries came out group-writable and
    were then refused, so the override cache never hit."""
    umask(mask)
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), {'x': 1})
    assert _entry(tmp_path / 'c').stat().st_mode & 0o777 == 0o600
    assert psf_cache.load_from_cache(str(src_file)) == {'x': 1}
    assert _temp_files(tmp_path / 'c') == []


@posix_only
def test_existing_group_writable_entry_is_repaired_by_save(monkeypatch,
                                                           tmp_path, src_file):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 1)
    _entry(tmp_path / 'c').chmod(0o660)
    with pytest.warns(RuntimeWarning, match='mode 0660'):
        assert psf_cache.load_from_cache(str(src_file)) is None
    psf_cache.save_to_cache(str(src_file), 2)
    assert _entry(tmp_path / 'c').stat().st_mode & 0o777 == 0o600
    assert psf_cache.load_from_cache(str(src_file)) == 2


@posix_only
def test_refused_entry_warns_once_per_process(monkeypatch, tmp_path, src_file,
                                              recwarn):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 1)
    _entry(tmp_path / 'c').chmod(0o664)
    for _ in range(3):
        assert psf_cache.load_from_cache(str(src_file)) is None
    assert len([w for w in recwarn if 'ignored' in str(w.message)]) == 1


@posix_only
def test_symlink_at_entry_path_is_replaced_not_followed(monkeypatch, tmp_path,
                                                        src_file):
    d = tmp_path / 'c'
    monkeypatch.setenv(ENV, str(d))
    path = psf_cache.get_cache_path(str(src_file))
    victim = tmp_path / 'victim'
    victim.write_bytes(b'precious')
    path.symlink_to(victim)
    psf_cache.save_to_cache(str(src_file), 5)
    assert victim.read_bytes() == b'precious'
    assert path.is_file() and not path.is_symlink()
    assert psf_cache.load_from_cache(str(src_file)) == 5


def test_failed_write_leaves_no_temp_file_and_warns_once(monkeypatch,
                                                         tmp_path, src_file,
                                                         recwarn):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))

    def disk_full(obj, f):
        f.write(b'partial')
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(psf_cache.pickle, 'dump', disk_full)
    psf_cache.save_to_cache(str(src_file), 1)  # must not raise
    psf_cache.save_to_cache(str(src_file), 1)
    assert list((tmp_path / 'c').iterdir()) == []
    msgs = [str(w.message) for w in recwarn]
    assert len([m for m in msgs if 'could not write' in m]) == 1
    assert 'No space left' in msgs[0]


def test_interrupted_write_leaves_no_temp_file(monkeypatch, tmp_path,
                                               src_file):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))

    def interrupted(obj, f):
        f.write(b'partial')
        raise KeyboardInterrupt

    monkeypatch.setattr(psf_cache.pickle, 'dump', interrupted)
    with pytest.raises(KeyboardInterrupt):
        psf_cache.save_to_cache(str(src_file), 1)
    assert list((tmp_path / 'c').iterdir()) == []


def test_failed_write_keeps_previous_entry_intact(monkeypatch, tmp_path,
                                                  src_file):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 'good')

    def boom(obj, f):
        f.write(b'partial')
        raise OSError('boom')

    real_dump = psf_cache.pickle.dump
    monkeypatch.setattr(psf_cache.pickle, 'dump', boom)
    with pytest.warns(RuntimeWarning):
        psf_cache.save_to_cache(str(src_file), 'new')
    monkeypatch.setattr(psf_cache.pickle, 'dump', real_dump)
    assert psf_cache.load_from_cache(str(src_file)) == 'good'


def test_save_sweeps_only_stale_temp_files(monkeypatch, tmp_path, src_file):
    d = tmp_path / 'c'
    d.mkdir(mode=0o700)
    old = d / '.psfxl_tmp_abc123.tmp'
    fresh = d / '.psfxl_tmp_def456.tmp'
    other = d / '.other_tmp_x.tmp'
    for p in (old, fresh, other):
        p.write_bytes(b'x')
    past = old.stat().st_mtime - psf_cache._TMP_MAX_AGE - 10
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    monkeypatch.setenv(ENV, str(d))
    psf_cache.save_to_cache(str(src_file), 1)
    assert not old.exists()
    assert fresh.exists()  # may belong to a writer still in progress
    assert other.exists()


def test_clear_all_removes_temp_files_but_not_symlinks_or_others(monkeypatch,
                                                                 tmp_path):
    d = tmp_path / 'c'
    d.mkdir()
    tmp = d / '.psfxl_tmp_abc123.tmp'
    tmp.write_bytes(b'x')
    target = tmp_path / 'precious'
    target.write_bytes(b'keep')
    link = d / '.psfxl_tmp_link99.tmp'
    link.symlink_to(target)
    other = d / '.bashrc.tmp'
    other.write_bytes(b'keep')
    monkeypatch.setenv(ENV, str(d))
    psf_cache.clear_all()
    assert not tmp.exists()
    assert link.is_symlink() and target.read_bytes() == b'keep'
    assert other.exists()


def test_temp_file_removed_mid_write_is_not_an_error(monkeypatch, tmp_path,
                                                     src_file, recwarn):
    """clear_all() in another process may delete an in-progress temp file."""
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    real_dump = psf_cache.pickle.dump

    def dump_then_clear(obj, f):
        real_dump(obj, f)
        psf_cache.clear_all()

    monkeypatch.setattr(psf_cache.pickle, 'dump', dump_then_clear)
    psf_cache.save_to_cache(str(src_file), 1)
    assert list((tmp_path / 'c').iterdir()) == []
    assert not [w for w in recwarn if 'could not write' in str(w.message)]


# --- Cache keys, key timing and load hardening ------------------------------

class FakeResult:
    """Stub the header parser and PSFXL decoder so PSF() can be exercised
    end to end on a dummy file pair; counts decodes."""

    def __init__(self, monkeypatch, tmp_path, during_decode=None):
        from psfxl_utils import psf_class
        import numpy as np
        self.np = np
        self.header = tmp_path / 'run' / 'tran.tran.tran'
        self.header.parent.mkdir()
        self.header.write_bytes(b'\x00\x00\x04\x00header')
        self.data = Path(str(self.header) + '.psfxl')
        self.data.write_bytes(b'data-v1')
        self.decodes = 0
        self.during_decode = during_decode
        monkeypatch.setattr(psf_class, 'parse_psf_header', lambda p: {
            'traces': [{'name': 'v', 'units': 'V'}],
            'sweep': {'name': 'time', 'units': 's'}, 'format': 'psfxl'})
        monkeypatch.setattr(psf_class, 'decode_psfxl', self._decode)
        monkeypatch.setattr(psf_class.PSF, '_instance_cache', {})

    def _decode(self, data_path, n):
        self.decodes += 1
        value = float(len(Path(data_path).read_bytes()))
        if self.during_decode:
            self.during_decode()
        return self.np.arange(3.0), {0: self.np.full(3, value)}, 1

    def load(self):
        from psfxl_utils import PSF
        return PSF(str(self.header))

    def new_session(self, monkeypatch):
        from psfxl_utils import psf_class
        monkeypatch.setattr(psf_class.PSF, '_instance_cache', {})


def test_data_file_change_invalidates_both_caches(monkeypatch, tmp_path):
    """Regression: only the header was keyed, so a new .psfxl (a run still
    writing, or replaced data) kept returning the old waveform."""
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    fake = FakeResult(monkeypatch, tmp_path)
    assert fake.load()._signals[0].ordinate[0] == 7.0
    fake.data.write_bytes(b'data-version-2')            # header untouched
    assert fake.load()._signals[0].ordinate[0] == 14.0   # memory cache
    fake.new_session(monkeypatch)
    assert fake.load()._signals[0].ordinate[0] == 14.0   # disk cache
    assert fake.decodes == 2


def test_file_changed_during_decode_is_not_cached(monkeypatch, tmp_path):
    """Regression: the key was taken after decoding, so a result decoded
    from the old file was filed under the new file's key."""
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    fake = FakeResult(monkeypatch, tmp_path)
    fake.during_decode = lambda: fake.data.write_bytes(b'rewritten-mid-decode')
    with pytest.warns(psf_cache.PSFCacheWarning, match='changed while'):
        fake.load()
    assert not list((tmp_path / 'c').glob('*.pkl'))
    fake.during_decode = None
    fake.load()
    assert fake.decodes == 2           # memory cache did not keep it either


def test_cache_warning_points_at_the_caller(monkeypatch, tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    shared.chmod(0o777)
    monkeypatch.setenv(ENV, str(shared))
    fake = FakeResult(monkeypatch, tmp_path)
    from psfxl_utils import PSF
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        PSF(str(fake.header)); line = sys._getframe().f_lineno
        fake.new_session(monkeypatch)
        PSF(str(fake.header))
    msgs = [w for w in wl if issubclass(w.category, psf_cache.PSFCacheWarning)]
    assert len(msgs) == 1                      # once, not twice per call
    assert (msgs[0].filename, msgs[0].lineno) == (__file__, line)


def test_unknown_tilde_user_warns_once_at_caller(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, '~no_such_user_xyz/cache')
    fake = FakeResult(monkeypatch, tmp_path)
    from psfxl_utils import PSF
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        PSF(str(fake.header)); line = sys._getframe().f_lineno
        fake.new_session(monkeypatch)
        PSF(str(fake.header))
    msgs = [w for w in wl if 'Ignoring' in str(w.message)]
    assert len(msgs) == 1
    assert (msgs[0].filename, msgs[0].lineno) == (__file__, line)


def test_data_file_rewrite_keeping_size_and_mtime_is_detected(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    fake = FakeResult(monkeypatch, tmp_path)
    before = psf_cache.source_fingerprint(str(fake.header))
    st = fake.data.stat()
    time.sleep(0.05)                                   # let the ctime clock tick
    fake.data.write_bytes(b'data-v2')                  # same size
    os.utime(fake.data, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert psf_cache.source_fingerprint(str(fake.header)) != before


@posix_only
def test_refused_entry_raises_every_time_under_error_filter(monkeypatch,
                                                            tmp_path, src_file):
    """Under -W error the warning must not be swallowed, nor shown once
    and then suppressed for good."""
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 1)
    _entry(tmp_path / 'c').chmod(0o664)
    with warnings.catch_warnings():
        warnings.simplefilter('error', psf_cache.PSFCacheWarning)
        for _ in range(2):
            with pytest.raises(psf_cache.PSFCacheWarning):
                psf_cache.load_from_cache(str(src_file))


@posix_only
def test_symlink_entry_in_default_dir_is_not_followed(isolated_home, tmp_path,
                                                      src_file):
    d = isolated_home / '.cache' / 'psfxl_utils'
    d.mkdir(parents=True)
    path = psf_cache.get_cache_path(str(src_file))
    planted = tmp_path / 'planted.pkl'
    planted.write_bytes(pickle.dumps('planted'))
    planted.chmod(0o600)
    path.symlink_to(planted)
    assert psf_cache.load_from_cache(str(src_file)) is None


@posix_only
def test_fifo_entry_does_not_hang(isolated_home, src_file):
    d = isolated_home / '.cache' / 'psfxl_utils'
    d.mkdir(parents=True)
    path = psf_cache.get_cache_path(str(src_file))
    os.mkfifo(path, 0o600)
    with pytest.warns(psf_cache.PSFCacheWarning, match='not a regular file'):
        assert psf_cache.load_from_cache(str(src_file)) is None


def test_very_long_source_name_is_cached(monkeypatch, tmp_path):
    src = tmp_path / ('x' * 240 + '.ac')
    src.write_bytes(b'x')
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src), 'long')
    assert psf_cache.load_from_cache(str(src)) == 'long'
    assert len(_entry(tmp_path / 'c').name) < 120


def test_clear_all_removes_old_format_entries(monkeypatch, tmp_path):
    d = tmp_path / 'c'
    d.mkdir()
    old = [d / 'run_0123abcd_89abcdef_1700000000000000000.pkl',
           d / 'run_0123abcd_89abcdef_-5000000000.pkl',
           d / 'run_0123abcd_nohash_42.pkl']
    for p in old:
        p.write_bytes(b'x')
    keep = d / 'notes_0123abcd.pkl'
    keep.write_bytes(b'x')
    monkeypatch.setenv(ENV, str(d))
    psf_cache.clear_all()
    assert not any(p.exists() for p in old)
    assert keep.exists()
