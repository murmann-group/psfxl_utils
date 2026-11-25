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
import warnings
from pathlib import Path
from typing import Optional

#: Environment variable that overrides the disk cache directory.
CACHE_DIR_ENV = 'PSFXL_UTILS_CACHE_DIR'


def get_cache_dir() -> Path:
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
            warnings.warn(f"Ignoring ${CACHE_DIR_ENV}={override!r}: {exc}",
                          RuntimeWarning, stacklevel=2)
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


def _is_trusted_entry(st: os.stat_result, strict: bool) -> bool:
    """True if an opened cache entry may be unpickled.

    Checked on the open file descriptor, not the path, so swapping the
    directory or entry after :func:`_is_private_dir` ran cannot substitute
    another user's pickle.  The entry must be owned by this user; with
    *strict* (a ``$PSFXL_UTILS_CACHE_DIR`` location) it must also not be
    group/world-writable.  The default location skips the mode check so
    entries written under umask 002 keep loading as before.
    """
    if not hasattr(os, 'getuid'):
        return True
    if st.st_uid != os.getuid():
        return False
    return not (strict and st.st_mode & 0o022)


#: Filenames written by :func:`get_cache_path`; ``clear_all`` deletes only these.
_ENTRY_RE = re.compile(r'.+_[0-9a-f]{8}_(?:[0-9a-f]{8}|nohash)_\d+\.pkl')


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
    digest = hashlib.md5()
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


def get_cache_path(filepath: str) -> Optional[Path]:
    """Generate cache file path from filename, library hash and mtime.

    The cache filename includes the nanosecond-resolution mtime, so the cache
    auto-invalidates if the source file is modified — even when it is
    rewritten within the same second — and :data:`LIB_HASH`, so it also
    auto-invalidates when the library itself changes.

    Returns ``None`` (caching disabled for this call) if the source file
    cannot be stat'ed, the cache directory cannot be created, or a
    ``$PSFXL_UTILS_CACHE_DIR`` directory is not private to this user.
    """
    path = Path(filepath).resolve()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None

    cache_dir = get_cache_dir()
    try:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        # Unwritable or invalid cache location (e.g. a bad
        # $PSFXL_UTILS_CACHE_DIR): run uncached rather than fail the load.
        return None
    # Only a user-chosen location is checked; the default under ~/.cache keeps
    # its previous behaviour (it may be 0775 on umask-002 systems).
    if not _is_default_dir(cache_dir) and not _is_private_dir(cache_dir):
        warnings.warn(
            f"psfxl_utils disk cache disabled: {cache_dir} is not owned by "
            f"you or is writable by others (set ${CACHE_DIR_ENV} to a "
            f"private directory)", RuntimeWarning, stacklevel=3)
        return None

    # Hash the full path to avoid collisions with same filenames in different dirs
    path_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
    return cache_dir / f"{path.stem}_{path_hash}_{LIB_HASH}_{mtime_ns}.pkl"


def load_from_cache(filepath: str):
    """Load PSF object from disk cache if available and valid.
    
    Returns:
        PSF object if cache hit (mtime matches), or None if cache miss/invalid
    """
    cache_path = get_cache_path(filepath)
    if cache_path is None or not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            strict = not _is_default_dir(cache_path.parent)
            if not _is_trusted_entry(os.fstat(f.fileno()), strict):
                return None
            return pickle.load(f)
    except Exception:
        return None  # Cache corrupted or incompatible, fall through


def save_to_cache(filepath: str, psf_object):
    """Save PSF object to disk cache for future sessions.

    Cache entries for older mtimes of the same source file are removed so
    the cache directory does not grow without bound.
    Non-fatal if write fails (e.g., permission denied).
    """
    cache_path = get_cache_path(filepath)
    if cache_path is None:
        return

    try:
        # Drop stale entries for this source file: any other mtime, and any
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
        with open(cache_path, 'wb') as f:
            pickle.dump(psf_object, f)
    except Exception:
        pass  # Silently fail if we can't write cache


def clear_all():
    """Clear all PSF disk caches, for every library build.

    Only top-level files whose names match the cache-entry format
    (``<stem>_<8 hex>_<8 hex|nohash>_<mtime_ns>.pkl``) are removed, never the
    directory itself or anything else in it: ``$PSFXL_UTILS_CACHE_DIR`` may
    point at a directory not dedicated to the cache.
    """
    try:
        for entry in get_cache_dir().glob('*.pkl'):
            try:
                if (_ENTRY_RE.fullmatch(entry.name) and entry.is_file()
                        and not entry.is_symlink()):
                    entry.unlink()
            except OSError:
                pass
    except OSError:
        pass
