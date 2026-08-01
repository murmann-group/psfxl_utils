# PSF Binary Format Reference

Byte-level notes on the three Cadence Spectre PSF container formats handled
by psfxl_utils. This is reverse-engineered knowledge — Cadence does not
publish the format — so treat it as descriptive of the files observed in
practice rather than a specification.

Unless stated otherwise, multi-byte integers in headers are **big-endian**;
PSFXL waveform payloads are **little-endian** IEEE 754 doubles.

---

## Magic bytes

The first 4 bytes of the header file select the format:

| Magic | Format | Typical files | Decoder module |
|-------|--------|---------------|----------------|
| `0x00000400` | PSFXL header | `tran.tran` | `psfxl_decoder` |
| `0x00000200` | Flat PSF | `ac.ac`, `noise.noise`, `dc.dc`, `stb.stb` | `flat_psf_decoder` |
| `0x00000500` | Info / op-point | `dcOp.dc`, `dcOpInfo.info`, `element.info`, … | `oppoint_psf_decoder` |

Not every file in a Spectre psf directory is binary PSF: some `.info` files
(e.g. `designParamVals.info`, which begins with ASCII `HEAD`) are plain text
and are rejected with a `ValueError`.

---

## Common header structure (0x400 and 0x200)

```
offset 0   magic        (u32 BE)
offset 4   count        (u32 BE)
offset 8   header_end   (u32 BE)  — end offset of the HEADER property block
offset 12  header properties …
```

After `header_end`, the file is a sequence of sections, each introduced by
`sec_type(u32) sec_end(u32)` where `sec_end` is the absolute end offset of
the section. The VALUE section is always last.

### Tags

4-byte big-endian tags structure the metadata sections:

| Tag | Meaning |
|-----|---------|
| `0x01` | SECTION |
| `0x10` | ENTRY (sweep, trace, or type definition; also value records) |
| `0x11` | GROUP |
| `0x16` | GROUP_EXT (outer container in `0x500` sections) |
| `0x21` | string property (`name`, `value` both length-prefixed strings) |
| `0x22` | int property (`name` string, u32 value) |
| `0x23` | float property (`name` string, f64 value) |
| `0x24` | DATA_BLOCK (48-byte inline statistics; follows PSFXL trace entries) |

Strings are length-prefixed ASCII, padded to 4-byte alignment.

### ENTRY kinds

An ENTRY starts `TAG(4) entry_idx(4) name(string) type_ref(4)` and continues
depending on kind:

- **TYPE entry** — one more u32 (the data-type code) then optional
  properties, including `units`. Data-type codes: `0x0b` = 8-byte real,
  `0x0c` = 16-byte complex (real then imaginary).
- **SWEEP entry** — properties follow directly (`units`, …).
- **PSFXL TRACE entry** — `TAG_DATA_BLOCK` + 48 statistics bytes, then
  properties. Units come from the referenced TYPE entry.
- **Flat PSF TRACE entry** — nothing follows (the next tag is another ENTRY
  or a section boundary).

### GROUP types in noise files

Spectre noise PSF stores per-mechanism device contributions (rs, id, fn, …)
inline after each device's total value, without separate value records. In
the TYPE section this is encoded as a GROUP: a TYPE entry with no
properties, followed by one TYPE entry per component, terminated by a TYPE
entry named `total`. psfxl_utils synthesizes a virtual sub-trace per
component, named `<device>.<mechanism>`.

---

## PSFXL (transient)

Three files per analysis:

| File | Content |
|------|---------|
| `*.tran` | Header only: signal names, types, units, per-signal statistics |
| `*.tran.psfxl` | Waveform data, organized in flushes |
| `*.tran.sig` | Text metadata: sweep count, flush count, compression type |

Data is split into *flushes* (time segments). Each flush holds one frame per
signal, in signal order, as a sequence of chunks.

### Type-3 chunk (compressed)

ASCII header of the form `\x003:seq:?:num_points:chunk_size\n` (fields in
hex), followed by a blosc1 frame. `seq` 0 carries the time axis plus signal
0 (two consecutive blosc blocks); `seq` ≥ 1 carries one signal each.

Blosc1 frame header (little-endian):

```
byte 0     version      (0x02 for blosc1)
byte 1     versionlz    (compressor-dependent: blosclz=0x01, lz4=0x02, …)
byte 2     flags        (0x00 none / 0x01 SHUFFLE / 0x02 BITSHUFFLE)
byte 3     typesize     (8 for float64)
bytes 4-7  nbytes       (uncompressed size, u32 LE)
bytes 8-11 blocksize    (u32 LE)
bytes 12-15 cbytes      (compressed size incl. this 16-byte header, u32 LE)
```

Because `versionlz` and `flags` legitimately vary between files, the decoder
identifies the right block by matching `nbytes == num_points × 8` rather
than a fixed magic pattern.

Payload after decompression: `num_points` little-endian float64 values.

### Constant chunks

- `chunk_size == 0` — the signal is 0.0 for the whole flush.
- `chunk_size == 8` — a constant (DC) value stored as one raw double,
  preceded by a 1–8 byte null pad to the next 8-byte file offset. The pad is
  always emitted, even when the position is already aligned, so the skip is
  `8 - (offset % 8)` (never 0). Skipping *all* nulls instead would corrupt
  signals whose constant value is 0.0 and could swallow the next chunk
  header's leading null byte.

### Type-2 chunk (uncompressed)

ASCII header `\x002:seq:?:num_traces:num_points:binary_size\n` (hex fields,
with `binary_size == (num_traces + 1) × num_points × 8`), followed by an
ASCII index table, null padding (variable length), then raw little-endian
doubles grouped as `(num_traces + 1)` blocks of `num_points`: time first,
then each signal.

Two kinds of type-2 chunk exist, distinguished only by their content:

- **Final flush** — data past the end of the type-3 range; decoded normally.
- **Intermediate checkpoint** — written periodically while the simulation
  runs; its time range is later re-compressed into type-3 blocks, making the
  checkpoint redundant. Identified by `first_time <= last type-3 time` and
  skipped (with a warning). Interrupted simulations typically leave one of
  these behind.

---

## Flat PSF (0x200)

Self-contained single file. Metadata sections (TYPE, SWEEP, TRACE) are
followed by a VALUE section containing one record per entry per sweep point:

```
TAG_ENTRY(4) entry_idx(4) value(8 real | 16 complex)
```

Two layouts occur in practice:

- **Dense** — records for all entries of a sweep point are contiguous, and
  every sweep point has the same byte layout (fixed stride). Decoded with
  NumPy stride tricks in one pass.
- **Sparse** — padding between records; decoded by scanning for the
  `\x00\x00\x00\x10` marker.

For noise GROUP types the record is extended:
`TAG EID comp_0(8) comp_1(8) … total(8)` — the per-mechanism doubles come
first and the device total is the **last** double in the record.

Values are big-endian doubles; complex values are two doubles (real,
imaginary).

---

## Info / op-point (0x500)

Same 12-byte file header, then (usually) a TYPE section and a VALUE section,
each wrapped in an outer `0x16` GROUP_EXT container.

TYPE-section entries carry an extra `base_type` word after `type_ref`:

| Code | Type | Value storage |
|------|------|---------------|
| `0x0b` | FLOAT64 | 8-byte BE double |
| `0x05` | INT32 (long) | 4-byte BE signed int |
| `0x01` | INT8 (byte) | stored as 4-byte BE signed int |
| `0x0c` | COMPLEX | 2 × 8-byte BE doubles |
| `0x10` | STRUCT | no data; declares a struct type |

Struct fields are declared as ENTRY records immediately following their
parent STRUCT entry. Standalone scalar types (the `V`/`I` types of `dc.dc`)
carry a `key` property (`node` / `branch`); struct fields do not — that
property is what distinguishes the two cases.

VALUE-section entries mirror the TYPE section:

- **Scalar** — `TAG EID name type_ref value [props]`.
- **Struct** — `TAG EID name type_ref` then the packed field values in
  declaration order (8 or 4 bytes each per the field's base type), then
  optional trailing properties (e.g. `model`).

psfxl_utils flattens struct entries into one signal per field, named
`<device>.<field>`, each holding a length-1 array.