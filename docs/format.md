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

### Type-3 chunk (one block per signal)

ASCII header `\x003:seq:?:num_points:chunk_size:flag[:field...]\n` (numbers
in hex), then an alignment pad of `8 - (offset % 8)` zero bytes -- always 1-8,
never 0, emitted even when the offset is already 8-byte aligned -- then
exactly `chunk_size` payload bytes. The next record starts right after the
payload, so the decoder walks from chunk to chunk instead of searching: a
`\x003:` byte run inside compressed or raw data is never taken for a header.

A flush is a `seq` 0 chunk followed by `seq` 1 ... N-1, all with the same
`num_points`. `seq` 0 carries the time axis and then signal 0, back to back
with no pad between them; every other chunk carries one signal.

The **flag** says how each block is stored, one hex digit per block:

| Digit | Block |
|-------|-------|
| `2` | blosc1 frame |
| `1` | raw LZ4 *block* (no frame header) of `num_points × 8` bytes |
| `0` | uncompressed: `num_points` doubles, or a single double (8 bytes) when the signal is constant for the flush |
| `a` | back-reference (`chunk_size` is 0; see below) |

- **`seq` 0:** two digits, time then signal 0 (`22`, `20`, `02`, `11`, ...).
  A missing flag means both blocks are uncompressed. The fields after the
  flag list the size of each compressed block in order; signal 0's block is
  whatever remains of `chunk_size` after the time block.
- **`seq` ≥ 1:** the low digit applies (`a2`, `a0`, `aa`, `81`, ...); the
  high digit (`a` or `8`) varies between files and does not affect decoding.

Spectre stores short flushes -- typically the last one -- uncompressed or
as LZ4, so a decoder that assumes every block is blosc loses or misplaces
exactly the end of the waveform.

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

The decoder checks `nbytes == num_points × 8` and that `cbytes` matches the
block size the header implies before decompressing. Decompressed payload:
`num_points` little-endian float64 values.

### Back-referenced chunks

`chunk_size == 0` is **not** a run of zeros. Spectre writes it when this
signal's data for this flush is byte-identical to a block already in the
file (two clock phases with the same duty, a bus whose bits carry the same
code) and points at that block with a trailing `<back_offset>.<length>`
field: the block starts `back_offset` bytes before this chunk header and
its compressed size is `length`. Filling zeros here would silently replace a
real waveform with a flat line. A constant signal is instead stored as an
8-byte uncompressed block (flag digit `0`); its value can be `0.0`, which is
why pads are always computed and never found by skipping null bytes.

### Type-1 chunk (uncompressed, no index table)

ASCII header `\x001:seq:?:num_points:binary_size\n` (hex fields), followed by
an alignment pad, then raw little-endian doubles grouped as
`(num_traces + 1)` blocks of `num_points`: time first, then each signal — the
same grouped payload as a type-2 chunk.

Spectre writes a single type-1 chunk covering the whole flush when the data
compresses badly enough that it does not use the per-signal blosc-compressed
type-3 blocks. **The container is chosen by how well the data compresses, not
by how many signals were saved** — adding one save changes the data, and the
data changes the container, so the same circuit can produce type-3 at 13
signals and type-1 at 14, or the reverse.

Two differences from type-2, both of which corrupt a decode if missed:

1. **No `num_traces` field.** The header has five fields, not six. Derive it:

   ```
   num_traces = binary_size / 8 / num_points - 1
   ```

   e.g. `\x001:0:ffffffffffffffff:252:103e0` is `0x252` = 594 points and
   `0x103e0` = 66528 bytes = `(13 + 1) × 594 × 8` — the time axis plus 13
   signals, uncompressed float64.

2. **No ASCII index table, but there IS an alignment pad.** It follows the
   same rule as the type-3 pad above: `8 - (offset % 8)`, always 1–8
   bytes, never 0, emitted even when the offset is already 8-byte aligned.

   Do **not** locate the payload by scanning past null bytes the way the
   type-2 reader does. The first payload value is the first time point, which
   is legitimately `0.0` and therefore eight null bytes indistinguishable from
   padding. Scanning past them consumes the start of the time axis, shifting
   every signal by one sample and dropping the last — and the result still
   looks plausible, because a shifted time axis is still monotonic and every
   value is still a real number. The signature is `t[0] == t[1]`, which a
   monotonicity check does not catch; `_find_type1_chunks()` tests for it
   explicitly.

Identify a file's containers by their chunk markers (a rough count: the same
byte patterns also occur by chance inside compressed data):

```bash
python -c "
import re, sys
from collections import Counter
data = open(sys.argv[1], 'rb').read()
print(Counter(m.group(1).decode() for m in re.finditer(rb'\x00(\d):', data)))
" psf/tran1.tran.tran.psfxl
# {'3': 12}  -> 12 type-3 chunks, one per signal
# {'1': 1}   -> 1 type-1 chunk, the whole flush
```

A file can mix containers from one flush to the next: many runs store the
first flush as type-1 and every later flush as type-3, or end with one or
more type-2 flushes.

### Type-2 chunk (uncompressed)

ASCII header `\x002:seq:?:num_traces:num_points:binary_size\n` (hex fields,
with `binary_size == (num_traces + 1) × num_points × 8`), followed by an
ASCII index table, an alignment pad of `8 - (offset % 8)` bytes after the
table (1-8, never 0), then raw little-endian doubles grouped as
`(num_traces + 1)` blocks of `num_points`: time first, then each signal.

### Flush order and checkpoints

In every file examined, flushes appear in the file in time order, each
starting after the previous one ends, whatever their container. The decoder
assembles them by start time (which is therefore file order), preferring a
type-3 flush when two start together. As a safeguard, a flush whose time range
is already covered by earlier data (such as a checkpoint record left by an
interrupted simulation) is skipped, and one that overlaps is trimmed to its
new samples, both with a `UserWarning`.

Each container's marker bytes can also occur by chance inside another
container's data. A type-1/type-2 record header inside a type-3 payload, or a
type-3 header inside a type-1/type-2 payload, is ignored.

### Undecodable data

A block that cannot be decoded -- an unknown flag, a size that does not add
up, a failed decompression, a back-reference to nothing, a flush cut short
by a truncated file -- becomes NaN for that signal and that flush, and one
`RuntimeWarning` lists the affected chunks. Every signal therefore stays
aligned with the time axis: a sample is never shifted, dropped or taken from
another signal. If a flush's time block itself cannot be decoded, the whole
flush is dropped (with a warning). A file with no decodable flush at all
gives an empty result and a `RuntimeWarning`.

## Flat PSF (0x200)

Self-contained single file. Metadata sections (TYPE, SWEEP, TRACE) are
followed by a VALUE section containing one record per entry per sweep point:

```
TAG_ENTRY(4) entry_idx(4) value(8 real | 16 complex)
```

Every sweep point has the same byte layout. Each record's size follows from
its entry's type in the header (the TYPE entry's data-type code: `0x0b` real,
`0x0c` complex; GROUP noise types as below). After the last point comes the
`0x0f` end tag, so in a single-point file the last record is not followed by
another record and its size cannot be inferred from the data.

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
| `0x0e` | STRING | 4-byte length, ASCII, padded to 4 bytes (e.g. `stb_state` in `.margin.stb`) |
| `0x10` | STRUCT | no data; declares a struct type |

Struct fields are declared as ENTRY records immediately following their
parent STRUCT entry. Standalone scalar types (the `V`/`I` types of `dc.dc`)
carry a `key` property (`node` / `branch`); struct fields do not — that
property is what distinguishes the two cases.

VALUE-section entries mirror the TYPE section:

- **Scalar** — `TAG EID name type_ref value [props]`. A STRING value is not
  a signal: the decoder returns string values, and string properties such as
  `gainMarginInfo` or `model`, separately (`PSF.strings`), keyed by entry
  name or `<entry>.<property>`.
- **Struct** — `TAG EID name type_ref` then the packed field values in
  declaration order (8 or 4 bytes each per the field's base type), then
  optional trailing properties (e.g. `model`).

psfxl_utils flattens struct entries into one signal per field, named
`<device>.<field>`, each holding a length-1 array.