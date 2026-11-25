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
| `flat_psf_decoder.py` | Flat PSF decoder (`.ac`, `.noise`, `.dc`): record layout from the header types, with data-inferred and marker-scanning fallbacks. |
| `oppoint_psf_decoder.py` | Info/op-point decoder for the `0x500` magic format (`dcOp.dc`, `dcOpInfo.info`, other `.info`). |
| `psf_cache.py` | Disk cache: pickle read/write and invalidation. Called transparently by `PSF.__init__`. |

All modules except `psf_class.py` are implementation details; their APIs may
change between releases. The re-exports in `__init__.py` are the stable
surface.

---

## Decode pipeline

`PSF(path)` runs the following sequence:

1. **Take the cache keys** — stat the header file and its `.psfxl` data file
   (see [Caching](#caching)). This happens before anything is decoded.
2. **Memory cache probe** — if the path was loaded before in this process
   and the fingerprint is unchanged, reuse the decoded arrays. Otherwise
   probe the **disk cache** for an entry with this key, and promote a hit to
   the memory cache.
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
metadata file. The data file is a sequence of *flushes* (time segments),
each stored in one of three containers:

- **Type-3** — one block per signal. The chunk header's flag says how each
  block is stored: blosc, raw LZ4, uncompressed (an array, or one double for
  a constant signal) or a back-reference to an identical block earlier in
  the file. The decoder walks the headers chunk to chunk (the next header
  starts where the declared payload ends) and decodes each block by its
  flag; nothing is located by searching.
- **Type-1 / type-2** — all signals of a flush in one uncompressed block,
  grouped as `(num_traces + 1) × num_points` (time first).

Flushes are assembled by start time, which is file order (see
[format](format.md#flush-order-and-checkpoints)), and each signal is
concatenated across them. A block that cannot be decoded is NaN for that
flush, with a `RuntimeWarning`, so every returned signal has the same length
as the time axis ([format](format.md#undecodable-data)).

### Flat PSF decoding

The VALUE section of `.ac`/`.noise`/`.dc` files is decoded from the
header's types: each record's size follows from its trace (real 16 bytes,
complex 24, a GROUP noise device `8 + 8 × n_inline`), the first sweep point
gives the stride, every record of every point is checked for its tag and
entry index, and all points are then extracted at once with NumPy stride
tricks. Sizes are never inferred from where the next record starts, which
fails for a single-point file (the last record is followed by the end tag,
not another record).

If the header does not determine the layout (a trace's type is missing, or
the data holds an entry the header does not list), the older paths are
used: a dense path that infers record sizes from the data, then a sparse
path that scans for `TAG_ENTRY` markers and reads only the first value of
each record. The sparse path warns, since it loses imaginary parts and
noise components.

**Truncated files.** Sections are walked by one helper
(`psf_parse.section_chain`) shared by the header parser and the decoders.
If a flat file's last section runs past the end of the file, it is taken as
the VALUE section (checked: the complete sections before it must name the
traces), and only the header-derived layout is used: the complete sweep points
are returned with a `RuntimeWarning`, and a file cut inside its first point or
its metadata raises `ValueError`. A truncated operating-point (0x500) file
raises `ValueError`, since a snapshot with devices missing is worse than none.
A cut that only removes the trailer (offset table and `Clarissa` marker)
leaves every section intact and decodes in full.

Noise files may store per-mechanism device contributions inline after each
device's total value (GROUP types). The header parser synthesizes virtual
sub-traces (`<device>.<mechanism>`) for these, and the decoder extracts them
from the same layout.

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
| Memory (per process) | `PSF._instance_cache` dict | source fingerprint |
| Disk (across sessions) | `$PSFXL_UTILS_CACHE_DIR/*.pkl`, default `~/.cache/psfxl_utils/*.pkl` | source fingerprint **and** library source hash, both in the entry name |

**Source fingerprint.** `psf_cache.source_fingerprint()` hashes the header
file's mtime (ns), size, inode and ctime, plus the mtime, size and inode of
the `<header>.psfxl` data file when there is one. A PSFXL result is two files
and either can change alone (a running simulation appends to the data file),
so both are covered. ctime catches copies that preserve mtime and size
(`cp -p`, `rsync -t`); metadata-only changes such as chmod or rename also
change it, which costs one re-decode. On NFS a stat can be served from the
client's attribute cache for a few seconds, so a file rewritten on another
host can briefly look unchanged.

**Keys are taken before decoding.** `PSF.__init__` computes the fingerprint
and the disk key once, before it decodes. After decoding, the result is
saved to disk and registered in memory only if the source files still match
those keys. Otherwise (a simulation still writing, a re-run that finished
mid-decode) the result may not match the file now on disk, so it is not
cached, and a `PSFCacheWarning` says so.

**Disk layer.** Entry names have the form
`<stem>_<path-md5-prefix>_<lib-hash>_<fingerprint>.pkl`, with the stem cut to
64 characters (the path hash keeps names unique). The MD5-of-path prefix only
disambiguates identically named files from different directories (it is not
an integrity check).

`<lib-hash>` is `psf_cache.LIB_HASH`, an MD5 prefix over the source of every
`.py` in the package, computed once at import. It makes a library change
invalidate the cache too. Without it, a decoder fix silently does nothing on
any file already cached: `decode_psfxl()` returns the corrected waveform while
`PSF()` keeps serving the pre-fix pickle, and the only way out is deleting the
cache directory by hand. Hashing the source rather than a declared version
number also covers uncommitted local edits.

When a new entry is written, other entries for the same source file (other
fingerprints, other library hashes) are deleted, so one source file never
holds more than one entry. **The cache as a whole is not size-limited:**
entries for result files that are never loaded again (deleted runs, old
corners) stay until removed. An entry is roughly the size of the decoded
arrays, which can be larger than the `.psfxl` file. `clear_all()` deletes
every entry, and only top-level files whose names match the entry format
(current or older versions), never the directory itself, subdirectories or
other files (including unrelated `.pkl` files).

**Location.** `psf_cache.get_cache_dir()` returns `$PSFXL_UTILS_CACHE_DIR`
when the variable is set and not blank, else `~/.cache/psfxl_utils`. `~` is
expanded (an unknown `~user` is ignored with a warning). A relative path is
resolved against the current directory on every access, so it moves if the
process changes directory; use an absolute path. The variable is read on every
cache access, so changing it after import takes effect immediately. If the
directory cannot be created, the disk layer is skipped and `PSF()` still
decodes. A cache directory the library creates gets mode `0700`.

**Writes.** Entries are written to a `0600` temporary file
(`.psfxl_tmp_*.tmp`) in the cache directory and renamed over the entry, so an
entry is never half-written, its mode does not depend on the umask, and a
symlink at the entry path is replaced rather than written through. Temporary
files left by a killed writer are removed by the next save once they are an
hour old, and by `clear_all()`. Write failures are non-fatal (the decode
result is still returned) and are reported with a `PSFCacheWarning`.

**Trust model.** Cache entries are pickles, and unpickling executes code.
On POSIX systems:

- If `PSFXL_UTILS_CACHE_DIR` points at a directory that is not owned by you
  or is group/world-writable, the disk cache is disabled with a
  `PSFCacheWarning`. (The default `~/.cache/psfxl_utils` directory is not
  checked, to keep its previous behaviour.)
- In every directory, an entry is opened without following symlinks and
  without blocking, and is unpickled only if the open descriptor is a
  regular file owned by you and not group/world-writable. Anything else is a
  miss, with a `PSFCacheWarning`, and is replaced on the next save.

On Windows only the regular-file check applies; keep the cache directory
private by other means.

Corrupted or unreadable entries are treated as cache misses.

**Cache warnings.** Every cache problem is reported as
`psfxl_utils.PSFCacheWarning`, a `RuntimeWarning` subclass, pointing at the
line that called `PSF()`. Each kind of problem is reported once per cache
directory per process. The decoded result never depends on the cache, so
strict test setups can turn warnings into errors and still exempt caching:
`-W error -W ignore::psfxl_utils.PSFCacheWarning`.

**Warnings and caching.** Warnings emitted during a decode (e.g. NaN-filled
chunks) are stored on the PSF object and re-emitted on every cache hit, so
callers see them regardless of whether the data came from cache.
