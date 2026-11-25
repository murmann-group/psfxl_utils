"""Disk-based PSF caching for persistent load optimization across sessions.

Transparent integration: PSF class calls these functions automatically.
Users don't need to import this module directly.

Cache files are pickles stored in the directory named by the
``PSFXL_UTILS_CACHE_DIR`` environment variable, or in the per-user
``~/.cache/psfxl_utils/`` when that variable is unset or empty.  The variable
is read on every cache access, so it can be changed after import.  Loading a
pickle executes code, so the cache directory must remain writable only by the
local user; never point it at a shared or untrusted location.

Cache keys include a hash of the library source, so upgrading or editing a
decoder invalidates every entry automatically.  Without it a decoder fix
appears to do nothing: ``decode_psfxl()`` returns the corrected result while
``PSF()`` keeps handing back the cached pre-fix one for the same file.
"""

import glob
import hashlib
import os
import pickle
import re
import stat
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional

#: Environment variable that overrides the disk cache directory.
CACHE_DIR_ENV = 'PSFXL_UTILS_CACHE_DIR'


class PSFCacheWarning(RuntimeWarning):
    """Issued when the disk cache is disabled, refuses an entry or cannot
    write one.  The decoded result is unaffected; only caching is.  Filter
    it separately from other warnings, e.g. ``-W error -W
    ignore::psfxl_utils.PSFCacheWarning``."""


def _md5(data: bytes = b''):
    """md5 for naming cache entries (not security); works on FIPS hosts."""
    try:
        return hashlib.md5(data, usedforsecurity=False)
    except TypeError:  # Python built without the keyword
        return hashlib.md5(data)


def get_cache_dir(*, _warn: bool = True) -> Path:
    """Return the disk cache directory.

    Uses ``$PSFXL_UTILS_CACHE_DIR`` if it is set and not blank (``~`` is
    expanded; a relative path is resolved against the current directory *at
    each call*, so it moves if the process changes directory), otherwise
    ``~/.cache/psfxl_utils``.  A value whose ``~user`` cannot be expanded is
    ignored with a warning.  The directory is not created here.
    """
    override = os.environ.get(CACHE_DIR_ENV, '').strip()
    if override:
        try:
            return Path(override).expanduser().absolute()
        except RuntimeError as exc:  # unknown ~user
            if _warn:
                _warn_once('override', override,
                           f"Ignoring ${CACHE_DIR_ENV}={override!r}: {exc}",
                           stacklevel=5)
    return _default_cache_dir()


def _default_cache_dir() -> Path:
    return Path.home() / '.cache' / 'psfxl_utils'


def _is_default_dir(path: Path) -> bool:
    """True if *path* is the default cache dir, however it is spelled."""
    try:
        return path.resolve() == _default_cache_dir().resolve()
    except (OSError, RuntimeError):
        return False


def __getattr__(name):
    # Backward compatibility: CACHE_DIR used to be a module constant.
    if name == 'CACHE_DIR':
        return get_cache_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_private_dir(path: Path) -> bool:
    """True if *path* is a directory owned by this user and not writable by
    group or others.

    Cache entries are pickles, so anyone who can write into the cache
    directory can run code in this process.  Checked on POSIX only.
    """
    if not hasattr(os, 'getuid'):
        return True
    try:
        st = path.stat()
    except OSError:
        return False
    return st.st_uid == os.getuid() and not (st.st_mode & 0o022)


def _is_trusted_entry(st: os.stat_result) -> bool:
    """True if an opened cache entry may be unpickled.

    Checked on the open file descriptor, not the path, so swapping the
    directory or entry after :func:`_is_private_dir` ran cannot substitute
    another user's pickle.  The entry must be a regular file owned by this
    user and not group/world-writable (entries are written 0600).  On
    non-POSIX systems only the regular-file check applies.
    """
    if not stat.S_ISREG(st.st_mode):
        return False
    if not hasattr(os, 'getuid'):
        return True
    return st.st_uid == os.getuid() and not (st.st_mode & 0o022)


#: Filenames written by :func:`get_cache_path`; ``clear_all`` deletes only these.
#: ``<stem>_<path hash>_<library hash>_<fingerprint>.pkl``; the last field
#: was the source mtime (decimal, possibly negative) in earlier versions.
_ENTRY_RE = re.compile(
    r'.+_[0-9a-f]{8}_(?:[0-9a-f]{8}|nohash)_(?:[0-9a-f]{16}|-?\d+)\.pkl')

#: Longest source-file stem kept in an entry name; the path hash keeps
#: truncated names unique, and the full name must stay under NAME_MAX.
_MAX_STEM = 64


def _compute_lib_hash() -> str:
    """Hash the library source so a decoder change invalidates the cache.

    Every ``.py`` in the package is hashed, in name order, because a decoded
    result depends on the decoders, the header parser and the container
    classes alike.  Hashing the source rather than a declared version number
    also catches uncommitted local edits — the case that actually bit us, when
    a patched decoder kept returning cached empty waveforms.

    Falls back to ``'nohash'`` if the source cannot be read (zipimport, a
    frozen build); the cache then behaves as it did before, keyed on path and
    mtime only.
    """
    digest = _md5()
    try:
        for src_file in sorted(Path(__file__).parent.glob('*.py')):
            digest.update(src_file.name.encode())
            digest.update(src_file.read_bytes())
    except OSError:
        return 'nohash'
    return digest.hexdigest()[:8]


#: Identifies the decoder build that produced a cache entry.  Computed once at
#: import; entries written by any other build are ignored and purged.
LIB_HASH = _compute_lib_hash()


def source_fingerprint(filepath: str) -> Optional[str]:
    """Identify the current contents of a result file and its data file.

    A PSFXL result is two files: the header passed to ``PSF()`` and the
    ``<header>.psfxl`` waveform data beside it.  Either can change on its
    own (a running simulation appends to the data file), so both are
    covered.  Uses ``st_mtime_ns``, ``st_size``, ``st_ino`` and
    ``st_ctime_ns`` of each; ctime catches a copy that preserves mtime
    (``cp -p``, ``rsync -t``).  Metadata-only changes (chmod, rename) also
    change it; that costs one re-decode.  Returns ``None`` if the header
    cannot be stat'ed.

    On NFS, a stat may be served from the client's attribute cache for a few
    seconds, so a file rewritten on another host can briefly look unchanged.
    """
    path = Path(filepath)
    try:
        st = path.stat()
    except OSError:
        return None
    parts = [st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns]
    try:
        dst = Path(str(path) + '.psfxl').stat()
        parts += [dst.st_mtime_ns, dst.st_size, dst.st_ino, dst.st_ctime_ns]
    except OSError:
        parts.append('-')
    return _md5(repr(parts).encode()).hexdigest()[:16]


def get_cache_path(filepath: str, *, _warn: bool = True) -> Optional[Path]:
    """Return the cache entry path for the current state of *filepath*.

    The name holds a hash of the resolved path, :data:`LIB_HASH` (so a
    library change invalidates the cache) and :func:`source_fingerprint`
    (so any change to the result file or its ``.psfxl`` data file does).

    Returns ``None`` (caching disabled for this call) if the source file
    cannot be stat'ed, the cache directory cannot be created, or a
    ``$PSFXL_UTILS_CACHE_DIR`` directory is not private to this user.
    """
    path = Path(filepath).resolve()
    fingerprint = source_fingerprint(str(path))
    if fingerprint is None:
        return None

    cache_dir = get_cache_dir(_warn=_warn)
    try:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        # Unwritable or invalid cache location (e.g. a bad
        # $PSFXL_UTILS_CACHE_DIR): run uncached rather than fail the load.
        return None
    # Only a user-chosen location is checked; the default under ~/.cache keeps
    # its previous behaviour (it may be 0775 on umask-002 systems).  Entries
    # are checked individually on load wherever they live.
    if not _is_default_dir(cache_dir) and not _is_private_dir(cache_dir):
        if _warn:
            _warn_once(
                'disabled', cache_dir,
                f"psfxl_utils disk cache disabled: {cache_dir} is not owned by "
                f"you or is writable by others (set ${CACHE_DIR_ENV} to a "
                f"private directory)")
        return None

    # Hash the full path to avoid collisions with same filenames in different dirs
    path_hash = _md5(str(path).encode()).hexdigest()[:8]
    stem = path.stem[:_MAX_STEM]
    return cache_dir / f"{stem}_{path_hash}_{LIB_HASH}_{fingerprint}.pkl"


#: Flags for opening a cache entry: never follow a symlink, never block on a
#: FIFO, binary mode on Windows.  Flags a platform lacks are simply omitted.
_OPEN_FLAGS = os.O_RDONLY
for _flag in ('O_NOFOLLOW', 'O_NONBLOCK', 'O_CLOEXEC', 'O_BINARY'):
    _OPEN_FLAGS |= getattr(os, _flag, 0)
del _flag


def load_from_cache(filepath: str, cache_path: Optional[Path] = None):
    """Load a decoded PSF object from the disk cache, or return ``None``.

    *cache_path* is the key from :func:`get_cache_path`; ``PSF`` passes the
    one it took before decoding.  An entry is unpickled only if it is a
    regular file owned by this user and not group/world-writable, checked on
    the open descriptor; a symlink or FIFO at the entry path is a miss.
    """
    if cache_path is None:
        cache_path = get_cache_path(filepath)
        if cache_path is None:
            return None

    try:
        fd = os.open(cache_path, _OPEN_FLAGS)
    except OSError:
        return None  # missing, a symlink (ELOOP), name too long, ...
    refused = None
    try:
        st = os.fstat(fd)
        if _is_trusted_entry(st):
            # Opened non-blocking so a FIFO cannot hang us; a regular file
            # reads the same either way, but restore blocking mode for the
            # buffered reader.  Only where O_NONBLOCK exists (POSIX):
            # Windows' set_blocking works on pipes only.
            if _OPEN_FLAGS & getattr(os, 'O_NONBLOCK', 0):
                os.set_blocking(fd, True)
            with os.fdopen(fd, 'rb') as f:
                fd = None
                return pickle.load(f)
        refused = st
    except Exception:
        return None  # Cache corrupted or incompatible, fall through
    finally:
        if fd is not None:
            os.close(fd)
    _warn_once(
        'refused', cache_path.parent,
        f"psfxl_utils disk cache entry {cache_path} ignored: it is not a "
        f"regular file owned by you, or is writable by others (mode "
        f"{refused.st_mode & 0o7777:04o}); it will be replaced on the next save")
    return None


def save_to_cache(filepath: str, psf_object, cache_path: Optional[Path] = None):
    """Save a decoded PSF object to the disk cache for future sessions.

    *cache_path* is the key ``PSF`` took **before** decoding.  If the source
    files have changed since (a simulation still writing, a re-run that
    finished mid-decode), the result may not match either version, so
    nothing is saved.

    Cache entries for other states of the same source file are removed so
    one file never holds more than one entry.

    The entry is written to a private (0600) temporary file in the cache
    directory and then renamed over the entry, so it is never left
    half-written, its mode does not depend on the umask (a group-writable
    entry would be refused by :func:`load_from_cache`), and a symlink at the
    entry path is replaced rather than written through.

    Non-fatal if the write fails (e.g. permission denied, disk full); the
    first failure per cache directory in a process is reported with a
    :class:`PSFCacheWarning`.
    """
    if cache_path is None:
        cache_path = get_cache_path(filepath)
        if cache_path is None:
            return
    else:
        now = get_cache_path(filepath, _warn=False)
        if now is None:
            return  # caching became unavailable (source or cache dir gone)
        if now != cache_path:
            _warn_changed(filepath, cache_path)
            return

    try:
        # Drop stale entries for this source file: any other state, and any
        # entry left behind by a different library build.  The glob stops at
        # the path hash so older LIB_HASH entries are purged too — otherwise
        # every upgrade would strand a full copy of the cache on disk.
        stem_prefix = glob.escape(cache_path.name.rsplit('_', 2)[0])
        for old in cache_path.parent.glob(f"{stem_prefix}_*.pkl"):
            if old != cache_path:
                try:
                    old.unlink()
                except OSError:
                    pass
        _remove_temp_files(cache_path.parent, min_age=_TMP_MAX_AGE)
        _write_entry(cache_path, psf_object)
        return
    except Exception as exc:
        error = exc
    _warn_once(
        'save', cache_path.parent,
        f"psfxl_utils could not write disk cache entry {cache_path}: "
        f"{error!r}; results will be decoded again next session")


def _warn_changed(filepath, cache_path):
    _warn_once(
        'changed', cache_path.parent,
        f"psfxl_utils did not cache {filepath}: it changed while it was "
        f"being decoded, so the result may not match the file now on disk",
        stacklevel=5)


#: Prefix and suffix of the temporary files :func:`_write_entry` creates.
#: The ``.tmp`` suffix keeps them out of every ``*.pkl`` glob in this module.
_TMP_PREFIX = '.psfxl_tmp_'
_TMP_SUFFIX = '.tmp'
_TMP_RE = re.compile(r'\.psfxl_tmp_[A-Za-z0-9_]+\.tmp')

#: Temporary files older than this (seconds) are left over from a writer
#: that was killed, and are removed by the next save.
_TMP_MAX_AGE = 3600

#: ``(kind, cache directory)`` pairs already warned about in this process.
_warned = set()


def _warn_once(kind: str, where, message: str, stacklevel: int = 4) -> None:
    """Warn once per process for each kind of problem and cache directory.

    The pair is recorded only after ``warnings.warn`` returns, so under
    ``-W error`` the warning is raised on every call rather than once and
    then never again.  ``stacklevel=4`` points at the caller of ``PSF()``:
    warn <- _warn_once <- cache function <- PSF.__init__ <- caller.
    """
    key = (kind, str(where))
    if key not in _warned:
        warnings.warn(message, PSFCacheWarning, stacklevel=stacklevel)
        _warned.add(key)


def _write_entry(cache_path: Path, obj) -> None:
    """Atomically write *obj* as a 0600 pickle at *cache_path*."""
    fd, tmp = tempfile.mkstemp(dir=cache_path.parent, prefix=_TMP_PREFIX,
                               suffix=_TMP_SUFFIX)
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f)
        try:
            os.replace(tmp, cache_path)
        except FileNotFoundError:
            return  # clear_all() removed our temp file mid-write: no entry
        tmp = None
    finally:  # also on KeyboardInterrupt, which the caller does not catch
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _remove_temp_files(cache_dir: Path, min_age: float = 0) -> None:
    """Remove temp files left by :func:`_write_entry` that are at least
    *min_age* seconds old.  Only regular files are removed, never symlinks."""
    now = time.time()
    for tmp in cache_dir.glob(f'{_TMP_PREFIX}*{_TMP_SUFFIX}'):
        try:
            st = tmp.lstat()
            if (_TMP_RE.fullmatch(tmp.name) and stat.S_ISREG(st.st_mode)
                    and now - st.st_mtime >= min_age):
                tmp.unlink()
        except OSError:
            pass


def clear_all():
    """Clear all PSF disk caches, for every library build.

    Only top-level files whose names match the cache-entry format
    (``<stem>_<8 hex>_<8 hex|nohash>_<fingerprint>.pkl``) are removed, never the
    directory itself or anything else in it: ``$PSFXL_UTILS_CACHE_DIR`` may
    point at a directory not dedicated to the cache.  Temporary files from
    interrupted writes (``.psfxl_tmp_*.tmp``) are removed too.
    """
    try:
        _remove_temp_files(get_cache_dir())
        for entry in get_cache_dir().glob('*.pkl'):
            try:
                if (_ENTRY_RE.fullmatch(entry.name) and entry.is_file()
                        and not entry.is_symlink()):
                    entry.unlink()
            except OSError:
                pass
    except OSError:
        pass
