# Internals

How psfxl_utils is put together: module responsibilities, the decode
pipeline, and the caching layers. Byte-level file-format details live in
[format.md](format.md).

---

## Module responsibilities

| Module | Role |
|--------|------|
| `__init__.py` | Public API facade. Re-exports `PSF`, `Signal`, `Sweep` and the low-level functions (`parse_psf_header`, `decode_psfxl`, `decode_psf_flat`, `parse_sig_file`). The only module users normally import from. |
| `psf_class.py` | The three public classes. `PSF` selects the correct decoder by magic bytes and manages caching. |
| `psf_parse.py` | Shared binary parsing: tag/string/property readers, magic-number constants, and `parse_psf_header` for PSFXL and flat PSF headers. |
| `psfxl_decoder.py` | PSFXL waveform decoder (`.tran.psfxl`), including blosc decompression and flush reassembly. |
| `flat_psf_decoder.py` | Flat PSF decoder (`.ac`, `.noise`, `.dc`) with dense (stride-based) and sparse (marker-scanning) paths. |
| `oppoint_psf_decoder.py` | Info/op-point decoder for the `0x500` magic format (`dcOp.dc`, `dcOpInfo.info`, other `.info`). |
| `psf_cache.py` | Disk cache: pickle read/write and invalidation. Called transparently by `PSF.__init__`. |

All modules except `psf_class.py` are implementation details; their APIs may
change between releases. The re-exports in `__init__.py` are the stable
surface.

---

## Decode pipeline

`PSF(path)` runs the following sequence:

1. **Memory cache probe** — if the path was loaded before in this process
   and the file's SHA-256 hash is unchanged, reuse the decoded arrays.
2. **Disk cache probe** — if `~/.cache/psfxl_utils/` holds a pickle whose
   embedded mtime matches the source file, load it (and promote it to the
   memory cache).
3. **Full decode** — read the first 4 bytes of the file and dispatch on the
   magic number:

   | Magic | Format | Decoder |
   |-------|--------|---------|
   | `0x00000400` | PSFXL header | `parse_psf_header` + `decode_psfxl` |
   | `0x00000200` | Flat PSF | `parse_psf_header` + `decode_psf_flat` |
   | `0x00000500` | Info / op-point | `decode_psf_info` |

   The decoded sweep and signal arrays are wrapped in `Sweep`/`Signal`
   objects, written to both cache layers, and any warnings raised during
   decoding are captured so they can be re-emitted on future cache hits.

### PSFXL decoding

Transient data is split across three files: the `.tran` header (signal
names, types, units), the `.tran.psfxl` data file, and the `.tran.sig`
metadata file. The data file contains a sequence of *flushes* (time
segments); each flush holds one compressed frame per signal:

- **Type-3 chunks** — blosc-compressed (LZ4 + byte-shuffle) IEEE 754
  doubles. The decoder pre-scans all chunk headers, then locates each
  chunk's blosc block by fingerprinting the expected uncompressed size
  (`num_points × 8`) rather than trusting fixed magic bytes, which vary
  with compressor and shuffle settings.
- **Constant chunks** — signals that are constant within a flush are stored
  as either a zero-length chunk (value 0.0) or a single raw double.
- **Type-2 chunks** — uncompressed flush records: an ASCII index table
  followed by raw doubles grouped as `(num_traces + 1) × num_points`
  (time first, then each signal). The final flush of a simulation is
  usually written this way.

Arrays from all flushes are concatenated per signal at the end.

### Interrupted simulations

While running, Spectre periodically writes small uncompressed type-2
*checkpoint* records. Once the same time range has been compressed into
type-3 blocks, those checkpoints are redundant — decoding them would
duplicate or back-track the time axis. This matters in particular for
simulations that were interrupted before completing.

The decoder classifies each type-2 chunk by its first time value: chunks at
or before the last decoded type-3 time are skipped (with a `UserWarning`
listing the skipped checkpoint times); only chunks strictly beyond it are
treated as genuine end-of-simulation flushes. The classification is purely
data-driven, so it is independent of how Spectre orders sections in the file.

### Flat PSF decoding

The VALUE section of `.ac`/`.noise`/`.dc` files is decoded with two
strategies:

- **Dense path** — when records are contiguous, the layout of the first
  sweep point gives a fixed stride, and all points are extracted at once
  with NumPy stride tricks (no Python-level loop).
- **Sparse path** — fallback when padding sits between records; scans for
  `TAG_ENTRY` markers with `bytes.find` and assembles arrays point by point.

Noise files may store per-mechanism device contributions inline after each
device's total value (GROUP types). The header parser synthesizes virtual
sub-traces (`<device>.<mechanism>`) for these, and the decoder extracts them
with the same stride approach.

### Info / op-point decoding

The `0x500` format holds a snapshot rather than a sweep: scalar entries
(node voltage / branch current) and struct-typed entries (per-device
operating-point parameters). Struct fields are flattened into individual
signals named `<device>.<field>`. The sweep axis is a length-1 placeholder
(`[0.0]`).

---

## Decompression backends (blosc / numcodecs)

`psfxl_decoder` uses the `blosc` package when available and falls back to
`numcodecs` (which bundles c-blosc) otherwise. Importing the module raises
`ImportError` immediately if neither is installed.

For numcodecs older than 0.16.0 a workaround is applied: c-blosc 1.21.0 (as
bundled with numcodecs 0.9.x) mis-unshuffles byte-shuffled float64 data in
`blosc_decompress_ctx()`, producing garbage values (~8×10¹⁸⁸). The ctx path
is triggered whenever numcodecs' internal `_get_use_threads()` returns
False — which happens unconditionally on systems where
`multiprocessing.Lock()` fails at import (some HPC/container environments)
and leaves the module-level `mutex` as `None`. The workaround injects a
`threading.Lock()` and forces `use_threads=True` so decompression always
takes the correct non-ctx path. Decompressed lengths are additionally
validated against the blosc header's `nbytes` field, with a
`RuntimeWarning` on mismatch.

---

## Caching

Two transparent layers, both managed by `PSF.__init__`:

| Layer | Store | Invalidation |
|-------|-------|--------------|
| Memory (per process) | `PSF._instance_cache` dict | SHA-256 hash of the source file, recomputed on every hit |
| Disk (across sessions) | `~/.cache/psfxl_utils/*.pkl` | nanosecond mtime **and** library source hash, both embedded in the cache filename |

**Memory layer.** Keyed by resolved path. On every lookup the source file is
re-hashed; a hash mismatch (file changed on disk) forces a reload. This makes
repeated loads within one process cost only the hash pass (< 1 ms for typical
files) instead of a full decode.

**Disk layer.** Cache filenames have the form
`<stem>_<path-md5-prefix>_<lib-hash>_<mtime_ns>.pkl`. The MD5-of-path prefix
only disambiguates identically named files from different directories (it is
not an integrity check); the nanosecond mtime makes any modification of the
source file — including rewrites within the same second — miss the cache and
trigger a re-decode.

`<lib-hash>` is `psf_cache.LIB_HASH`, an MD5 prefix over the source of every
`.py` in the package, computed once at import. It makes a library change
invalidate the cache too. Without it, a decoder fix silently does nothing on
any file already cached: `decode_psfxl()` returns the corrected waveform while
`PSF()` keeps serving the pre-fix pickle for the same path and mtime, and the
only way out is deleting `~/.cache/psfxl_utils` by hand. Hashing the source
rather than a declared version number also covers uncommitted local edits.

When a new cache entry is written, entries for older mtimes *and* older
library hashes of the same source file are deleted, so neither repeated
simulation nor repeated upgrades let the directory grow without bound.
`clear_all()` wipes the whole cache directory.

Corrupted or unreadable cache files are treated as cache misses; cache write
failures are silently ignored (the decode result is still returned).

**Trust model.** Cache entries are pickles, and unpickling executes code.
The cache lives in the per-user `~/.cache/psfxl_utils/` directory and must
not be redirected to a shared or otherwise untrusted location.

**Warnings and caching.** Warnings emitted during a decode (e.g. skipped
checkpoint chunks of an interrupted simulation) are stored on the PSF object
and re-emitted on every cache hit, so callers see them regardless of whether
the data came from cache.
