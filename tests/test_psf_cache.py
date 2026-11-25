"""Tests for the disk cache location and its PSFXL_UTILS_CACHE_DIR override.

The end-to-end test needs a real flat PSF result file.  Point
``PSFXL_TEST_PSF_FILE`` at one (e.g. an ``ac.ac``) to run it; it is skipped
otherwise.
"""

import os
import shutil
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
                        lambda p: loaded.append(real_load(p)) or loaded[-1])
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
    with pytest.warns(RuntimeWarning):
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
    with pytest.warns(RuntimeWarning, match='not owned by you or is writable'):
        assert psf_cache.get_cache_path(str(src_file)) is None
    with pytest.warns(RuntimeWarning):
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
    assert psf_cache.load_from_cache(str(src_file)) is None


@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='POSIX only')
def test_group_writable_entry_refused_only_for_override(monkeypatch, tmp_path,
                                                        isolated_home,
                                                        src_file):
    monkeypatch.setenv(ENV, str(tmp_path / 'c'))
    psf_cache.save_to_cache(str(src_file), 1)
    entry, = (tmp_path / 'c').glob('*.pkl')
    entry.chmod(0o664)
    assert psf_cache.load_from_cache(str(src_file)) is None

    monkeypatch.delenv(ENV)
    psf_cache.save_to_cache(str(src_file), 2)
    entry, = (isolated_home / '.cache' / 'psfxl_utils').glob('*.pkl')
    entry.chmod(0o664)
    assert psf_cache.load_from_cache(str(src_file)) == 2


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
