# Changelog

## 1.1.0

Upgrading: clear the disk cache once (`psfxl_utils.psf_cache.clear_all()`).
Entries written by older versions are never reused, but they stay on disk
until cleared.

### Behaviour changes

- **Signal and sweep arrays are read-only.** Every `PSF()` of the same file
  shares them, so an in-place change used to alter every other result for
  that file. Assigning into one now raises `ValueError: assignment destination
  is read-only`; call `.copy()` first. Each `PSF` instance also has its own
  `Signal`/`Sweep` objects.
- **Undecodable samples are NaN, with a `RuntimeWarning`.** Every signal now
  always has the length of the sweep. Before, a block the decoder could not
  read made that signal shorter than the time axis (later samples shifted in
  time), or was silently filled from a neighbouring signal.
- **Truncated files are reported.** A flat file (`.ac`, `.noise`, `.dc`,
  `.stb`, ...) cut short returns its complete sweep points with a
  `RuntimeWarning`; one cut inside its first point or its metadata, and any
  truncated operating-point file (`dcOp.dc`, `*.info`), raises `ValueError`.
  Before, these came back as a `PSF` with zero signals or raised `struct.error`.
- **Classic windowed binary PSF** (for example a PSS `.td.pss` that Spectre
  writes this way even under `-format psfxl`) raises a `ValueError` saying it
  is not supported, instead of a `FileNotFoundError` for a `.psfxl` file that
  never exists. A PSFXL header whose `.psfxl` file is really missing raises a
  `FileNotFoundError` naming it.
- **Cache problems are `psfxl_utils.PSFCacheWarning`** (a `RuntimeWarning`
  subclass), reported once per kind and cache directory, pointing at the
  caller of `PSF()`.

### Fixed: wrong or missing data

- **PSFXL transient results.** Blocks are now decoded by the encoding their
  chunk header declares (blosc, raw LZ4, uncompressed, an 8-byte constant, or
  a back-reference), instead of by searching for blosc headers. 46 of 2,812
  results in a test corpus decoded wrong before:
  - signals shorter than the time axis
  - one signal's tail copied from another
  - voltages read as the time axis
  - final flushes dropped
- **First flush stored uncompressed.** It was discarded as a "checkpoint", so
  many waveforms started at a few microseconds instead of t = 0.
- **Single-point AC and noise results** (`ac freq=...`, spot noise) lost
  imaginary parts and noise components; record sizes now come from the header.
- **Strings in operating-point and stability files.** These were dropped and
  are now available through `PSF.strings` / `PSF.get_string()`: `stb_state`
  in `.margin.stb` (the stability verdict), `gainMarginInfo` /
  `phaseMarginInfo`, and device `model` names.
- **Disk cache.**
  - Entries were written with the process umask (002/007 made them
    group-writable), refused on load, and re-pickled on every load.
  - The `.psfxl` data file was not part of either cache key, so a changed or
    still-growing data file kept returning the old waveform.
  - A file rewritten during a decode was cached under the new file's key.
  - Loads followed symlinks, could hang on a FIFO, and crashed on long names.
- **Docs:** device-parameter examples (`M0.vgs` is in `dcOpInfo.info`, not
  `dcOp.dc`), noise is real-valued, and the operating-point sweep is `[0.0]`.

Every change to decoded output was checked sample for sample against
Cadence's own reader (OCEAN) on 222 result files, about 180 million samples.
All other files in the corpus decode exactly as before.

### Added

- `PSF.strings`, `PSF.get_string()`.
- `psfxl_utils.PSFCacheWarning`.
- `tests/test_corpus.py`: opt-in invariant checks over a results directory
  (`PSFXL_TEST_CORPUS=<dir>`).

## 1.0.1

- `PSFXL_UTILS_CACHE_DIR` moves the disk cache.
- Back-referenced (zero-size) chunks and constant signals are decoded instead
  of being dropped or zero-filled.
- Type-1 (uncompressed) PSFXL flushes are detected.

## 1.0.0

- Initial release.
