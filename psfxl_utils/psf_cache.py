"""Disk-based PSF caching for persistent load optimization across sessions.

Transparent integration: PSF class calls these functions automatically.
Users don't need to import this module directly.

Cache files are pickles stored in the per-user directory
``~/.cache/psfxl_utils/``.  Loading a pickle executes code, so the cache
directory must remain writable only by the local user; never point it at a
shared or untrusted location.
"""

import hashlib
import pickle
import shutil
from pathlib import Path


def get_cache_path(filepath: str) -> Path:
    """Generate cache file path based on filename and mtime.

    Cache filename includes the nanosecond-resolution mtime so the cache
    auto-invalidates if the source file is modified — even when it is
    rewritten within the same second.
    """
    path = Path(filepath).resolve()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None

    cache_dir = Path.home() / '.cache' / 'psfxl_utils'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Hash the full path to avoid collisions with same filenames in different dirs
    path_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
    return cache_dir / f"{path.stem}_{path_hash}_{mtime_ns}.pkl"


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
        # Drop stale entries for this source file (same stem + path hash,
        # different mtime).
        stem_prefix = cache_path.name.rsplit('_', 1)[0]
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
    """Clear all PSF disk caches."""
    cache_dir = Path.home() / '.cache' / 'psfxl_utils'
    try:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
