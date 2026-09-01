"""Disk-based PSF caching for persistent load optimization across sessions.

Transparent integration: PSF class calls these functions automatically.
Users don't need to import this module directly.

Cache files are pickles stored in the per-user directory
``~/.cache/psfxl_utils/``.  Loading a pickle executes code, so the cache
directory must remain writable only by the local user; never point it at a
shared or untrusted location.

Cache keys include a hash of the library source, so upgrading or editing a
decoder invalidates every entry automatically.  Without it a decoder fix
appears to do nothing: ``decode_psfxl()`` returns the corrected result while
``PSF()`` keeps handing back the cached pre-fix one for the same file.
"""

import hashlib
import pickle
import shutil
from pathlib import Path

CACHE_DIR = Path.home() / '.cache' / 'psfxl_utils'


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


def get_cache_path(filepath: str) -> Path:
    """Generate cache file path from filename, library hash and mtime.

    The cache filename includes the nanosecond-resolution mtime, so the cache
    auto-invalidates if the source file is modified — even when it is
    rewritten within the same second — and :data:`LIB_HASH`, so it also
    auto-invalidates when the library itself changes.
    """
    path = Path(filepath).resolve()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Hash the full path to avoid collisions with same filenames in different dirs
    path_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
    return CACHE_DIR / f"{path.stem}_{path_hash}_{LIB_HASH}_{mtime_ns}.pkl"


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
        stem_prefix = cache_path.name.rsplit('_', 2)[0]
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
    """Clear all PSF disk caches, for every library build."""
    try:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
