# Usage Examples

All examples assume:

```python
from psfxl_utils import PSF
```

The path passed to `PSF(...)` is always the *header* file of the analysis
(`tran.tran`, `ac.ac`, `dc.dc`, `dcOp.dc`, `dcOpInfo.info`, `noise.noise`,
`stb.stb`). For transient results the companion `tran.tran.psfxl` data file
is located automatically next to the header.

---

## Seeing what a file holds

Every public object reports itself, so probing an unfamiliar result needs no
prior knowledge of the API — useful at a REPL, in a one-off script, or to an
agent with shell access:

```python
>>> psf = PSF('results/psf/tran.tran')
>>> psf
<PSF 'tran.tran': 13 signals, sweep 'time' [s] × 594 points;
 vout, vin, clk, vdd, … (+9 more)>

>>> psf.get_sweep()
<Sweep 'time' [s]: 594 points, 0 … 0.000229 (float64)>

>>> psf.get_signal('vout')
<Signal 'vout' [V]: 594 points, 0 … 1.798 (float64)>

>>> PSF('results/psf/dcOp.dc')
<PSF 'dcOp.dc': 112 signals, no sweep (operating-point snapshot);
 M0:region, M0:trise, M0:dtemp, M0:reversed, … (+108 more)>

>>> PSF('results/psf/dcOp.dc').get_signal('M0:ids')
<Signal 'M0:ids' [A]: 1 point, 1.234e-05>
```

(`dcOp.dc` holds `M0:ids`-style device entries only when operating-point
data is saved to the rawfile; otherwise it holds node voltages and terminal
currents, and device parameters are in `dcOpInfo.info` as `M0.ids`.)

`print()` gives the same text — `__str__` falls back to `__repr__`.

Only names, array lengths and the two endpoint values are read, so a `repr`
stays cheap on a multi-million-point transient.

An array that decoded to nothing is reported as `EMPTY (no data decoded)`
rather than `0 points`, and a `PSF` whose signals are *all* empty says so
outright:

```python
>>> psf
<PSF 'tran.tran': 13 signals, sweep 'time' [s] × 0 points;
 vout, vin, clk, vdd, … (+9 more) — EMPTY: every signal decoded to zero points>
```

That combination — a correct signal count and correct names over no data at
all — is what an unrecognised chunk container looks like, so it is worth
being able to spot at a glance. `decode_psfxl()` also warns in that case; see
[format.md](format.md).

---

## Listing available signals

Works the same for every analysis type:

```python
psf = PSF('results/psf/tran.tran')

for signal in psf.all_signals():
    print(f'{signal.name:<20} {signal.units:<10} {signal.type.kind}')
```

`signal.type.kind` is `'float'` for real data and `'complex'` for AC/stability
results.

Two naming caveats when cross-referencing with the Cadence GUI:

- **Terminal currents** are stored in the PSF file as `instance:index`
  (e.g. `DUT:29`). The GUI renames them to `instance:terminalName`
  (e.g. `DUT:ibias`) using netlist information that is
  not present in the PSF file, so psfxl_utils reports the raw
  `instance:index` name. Use `all_signals()` to discover the stored names.
- **Not every file in a psf directory is binary PSF** — some (e.g.
  `designParamVals.info`) are plain text and raise
  `ValueError: Not a PSF header file` when passed to `PSF()`.

---

## Transient analysis

```python
psf   = PSF('results/psf/tran.tran')
sweep = psf.get_sweep()           # time axis
vout  = psf.get_signal('vout')

print(f"Time: {sweep.abscissa[0]:.3g} to {sweep.abscissa[-1]:.3g} {sweep.units}")
print(f"Vout range: {vout.ordinate.min():.4f} to {vout.ordinate.max():.4f} {vout.units}")
```

If the result file is incomplete (a simulation still running, interrupted,
or a truncated copy), the decoder returns every flush it can decode. Samples
it cannot decode are NaN, with a `RuntimeWarning` naming the affected chunks,
so every signal still lines up with the time axis (see
[format: undecodable data](format.md#undecodable-data)).

---

## Plotting waveforms with matplotlib

```python
import matplotlib.pyplot as plt
from psfxl_utils import PSF

psf = PSF('results/psf/tran.tran')

plot_signals = {'vout', 'vin'}          # names to plot

sweep  = psf.get_sweep()
count  = len(plot_signals)
figure = plt.figure(figsize=(10, 2 * count))

axis_index = 0
for signal in psf.all_signals():
    if signal.name not in plot_signals:
        continue
    axes = figure.add_subplot(count, 1, axis_index + 1)
    if axis_index == 0:
        axes.set_title('Waveforms From Simulation PSFXL')
    axes.plot(sweep.abscissa, signal.ordinate, linewidth=2, label=signal.name)
    axes.set_ylabel(f'{signal.name} ({PSF.units_to_unicode(signal.units)})')
    axes.legend()
    axes.grid()
    axis_index += 1

axes.set_xlabel(f'{sweep.name} ({PSF.units_to_unicode(sweep.units)})')
plt.tight_layout()
plt.savefig('results.png')
plt.show()
```

`matplotlib` is not a dependency of psfxl_utils — install it separately.

---

## DC sweep

```python
psf   = PSF('results/psf/dc.dc')
sweep = psf.get_sweep()           # swept variable (name/units may be blank)
vout  = psf.get_signal('vout')    # one value per sweep point
```

---

## DC operating point — node voltages and currents

```python
psf = PSF('results/psf/dcOp.dc')

# Single value per signal; ordinate is a length-1 array.
vout = psf.get_signal('vout').ordinate[0]
print(f"Vout = {vout:.4f} V")
```

---

## DC operating point — device parameters

`dcOpInfo.info` stores one struct per device instance. Each struct field is
exposed as a separate signal named `<device>.<parameter>`:

```python
psf = PSF('results/psf/dcOpInfo.info')

gm  = psf.get_signal('M1.gm').ordinate[0]
ids = psf.get_signal('M1.ids').ordinate[0]

# Collect one parameter across all devices
gms = {s.name: s.ordinate[0]
       for s in psf.all_signals() if s.name.endswith('.gm')}
```

The same access pattern works for other `.info` results
(`element.info`, `modelParameter.info`, …).

---

## AC analysis (complex signals)

```python
import numpy as np

psf   = PSF('results/psf/ac.ac')
sweep = psf.get_sweep()               # frequency axis (Hz)
vout  = psf.get_signal('vout')        # complex ordinate

gain_db  = 20 * np.log10(np.abs(vout.ordinate))
phase_dg = np.degrees(np.angle(vout.ordinate))
```

---

## Noise analysis

Total output noise plus per-mechanism device contributions. Per-mechanism
components are exposed as sub-signals named `<device>.<mechanism>`:

```python
psf   = PSF('results/psf/noise.noise')
sweep = psf.get_sweep()                    # frequency axis (Hz)

out_total = psf.get_signal('out')          # total output noise density

# Per-device, per-mechanism contributions (when present in the file)
m1_id = psf.get_signal('M1.id')            # channel noise of M1
m1_fn = psf.get_signal('M1.fn')            # flicker noise of M1
```

---

## Monte Carlo post-processing

Collect a signal across many run directories:

```python
import numpy as np
from pathlib import Path
from psfxl_utils import PSF

vgs_vals = []
for run_dir in sorted(Path('mc_results').glob('run_*')):
    psf = PSF(run_dir / 'dcOpInfo.info')
    vgs_vals.append(psf.get_signal('M0.vgs').ordinate[0])

print(f"Vgs  mean={np.mean(vgs_vals):.4f} V  std={np.std(vgs_vals):.4f} V")
```

Repeated loads benefit from the disk cache: only the first pass over the runs
pays the decode cost.

---

## Unicode unit labels

```python
PSF.units_to_unicode('Ohm')  # → 'Ω'
PSF.units_to_unicode('deg')  # → '°'
PSF.units_to_unicode('Hz')   # → 'Hz'  (pass-through)
```

---

## Clearing the disk cache

```python
from psfxl_utils.psf_cache import clear_all
clear_all()
```
