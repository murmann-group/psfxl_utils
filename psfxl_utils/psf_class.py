"""
PSF Data Model

Defines the three public data classes that represent decoded Cadence Spectre
PSF simulation results:

- :class:`PSF`    — top-level file reader; entry point for all PSF access.
- :class:`Signal` — a single decoded trace: real (transient, DC, noise) or
  complex (AC, stability).
- :class:`Sweep`  — the independent axis (time, frequency, DC parameter, etc.).

Supported PSF Formats
---------------------
+------------+------------------------------------------+--------------+
| Extension  | Description                              | Decoder      |
+============+==========================================+==============+
| ``.tran``  | Transient header + ``.tran.psfxl`` data  | PSFXL        |
| ``.ac``    | AC small-signal (flat PSF)               | Flat PSF     |
| ``.noise`` | Noise spectral density (flat PSF)        | Flat PSF     |
| ``.dc``    | DC sweep or op-point snapshot            | Flat / Info  |
+------------+------------------------------------------+--------------+

Caching
-------
PSF decoding is expensive for large files.  Two transparent caching layers
reduce repeated load times:

1. **In-session memory cache** — after the first load the decoded object is
   kept in ``PSF._instance_cache``.  Hash-based invalidation detects file
   changes between loads within the same Python process.

2. **Disk cache** — decoded objects are pickled to
   ``$PSFXL_UTILS_CACHE_DIR`` (default ``~/.cache/psfxl_utils/``).  Subsequent loads from a new Python session
   take ~50 ms instead of ~1.5 s.

Public Interface
----------------
:class:`PSF`
    Load a result file and access signals or the sweep axis.
:class:`Signal`
    Holds ``name``, ``units``, ``ordinate`` (NumPy array), and ``type``
    (``'float'`` or ``'complex'``).
:class:`Sweep`
    Holds ``name``, ``units``, and ``abscissa`` (NumPy array).

Example
-------
.. code-block:: python

    from psfxl_utils import PSF

    # --- DC operating-point (single scalar per signal) ---
    psf = PSF("results/psf0/dcOp.dc")              # node voltages
    for signal in psf.all_signals():
        print(f"{signal.name:<20} {signal.ordinate[0]:.6g} {signal.units}")
    dev = PSF("results/psf0/dcOpInfo.info")        # device parameters
    vgs = dev.get_signal("M0.vgs").ordinate[0]

    # --- Transient sweep ---
    psf   = PSF("results/psf0/tran.tran")
    sweep = psf.get_sweep()               # time axis (seconds)
    vout  = psf.get_signal("vout")        # voltage array

    # --- AC / noise ---
    psf   = PSF("results/psf0/ac.ac")
    sweep = psf.get_sweep()               # frequency axis (Hz)
    gain  = psf.get_signal("vout")        # complex ordinate

    # --- Unicode unit labels ---
    print(PSF.units_to_unicode("Ohm"))    # → Ω
"""

import struct
import warnings
from pathlib import Path

import numpy as np

from .psf_parse import parse_psf_header, PSF_MAGIC_INFO
from .psfxl_decoder import decode_psfxl
from .flat_psf_decoder import decode_psf_flat
from .oppoint_psf_decoder import decode_psf_info
from . import psf_cache


def _fmt_units(units):
    """Render a unit string for a repr: ``' [V]'``, or ``''`` when unitless."""
    return f" [{units}]" if units else ""


def _fmt_value(v):
    """Compact numeric rendering for reprs."""
    try:
        if np.iscomplexobj(v):
            return f"{complex(v).real:.4g}{complex(v).imag:+.4g}j"
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_array(arr):
    """Summarise an array for a repr: length, dtype and endpoints.

    Empty is reported as ``EMPTY`` rather than ``0 points`` because an empty
    array is a decode failure, not a small result, and the point of these
    reprs is that it should be impossible to miss.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return "EMPTY (no data decoded)"
    if arr.size == 1:
        return f"1 point, {_fmt_value(arr.flat[0])}"
    return (f"{arr.size} points, {_fmt_value(arr.flat[0])} … "
            f"{_fmt_value(arr.flat[-1])} ({arr.dtype})")


class _SignalType:
    def __init__(self, kind='float'):
        self.kind = kind


class Signal:
    """
    A single decoded trace from a PSF result file.

    Signals are returned by :meth:`PSF.all_signals` and :meth:`PSF.get_signal`.
    For swept analyses (transient, AC, noise) ``ordinate`` is a 1-D NumPy array
    with one sample per sweep point.  For operating-point snapshots (DC op)
    ``ordinate`` contains a single element.

    Parameters
    ----------
    name : str
        Signal name exactly as it appears in the PSF file
        (e.g., ``'vout'``, ``'DUT.M1:ids'``).
    units : str
        Physical unit string (e.g., ``'V'``, ``'A'``).
    ordinate : numpy.ndarray
        Decoded signal data.  Complex for AC/noise analyses,
        real-valued (``float64``) for DC/transient.

    Attributes
    ----------
    name : str
        Signal identifier.
    units : str
        Physical unit string.
    ordinate : numpy.ndarray
        Signal data array.
    type : _SignalType
        Container whose ``kind`` attribute is ``'float'`` or ``'complex'``.

    Examples
    --------
    .. code-block:: python

        signal = psf.get_signal("M0.vgs")
        print(signal.name)           # 'M0.vgs'
        print(signal.units)          # 'V'
        print(signal.ordinate[0])    # scalar value for a DC op-point
        print(signal.type.kind)      # 'float'
    """

    def __init__(self, name, units, ordinate):
        self.name = name
        self.units = units
        self.ordinate = ordinate
        kind = 'complex' if np.iscomplexobj(ordinate) else 'float'
        self.type = _SignalType(kind)

    def __repr__(self):
        """One-line summary: name, units, length, dtype and endpoints.

        ``__str__`` falls back to this, so ``print(signal)`` is informative
        too.  Reads only the first and last element, so it stays cheap on
        multi-million-point transients.
        """
        return (f"<Signal {self.name!r}{_fmt_units(self.units)}: "
                f"{_fmt_array(self.ordinate)}>")


class Sweep:
    """
    The independent (sweep) axis from a PSF result file.

    Returned by :meth:`PSF.get_sweep`.  For transient analyses the axis is
    time; for AC/noise it is frequency; for DC parameter sweeps it is the
    swept variable value.  For DC operating-point files the sweep is a
    single placeholder point, ``[0.0]``.

    Parameters
    ----------
    name : str
        Axis name as stored in the PSF header (e.g., ``'time'``, ``'freq'``).
    units : str
        Physical unit string (e.g., ``'s'``, ``'Hz'``, ``'V'``).
    abscissa : numpy.ndarray
        1-D array of sweep point values.

    Attributes
    ----------
    name : str
        Axis identifier.
    units : str
        Physical unit string.
    abscissa : numpy.ndarray
        Array of sweep point values.

    Examples
    --------
    .. code-block:: python

        sweep = psf.get_sweep()
        print(f"{sweep.name}: {sweep.abscissa[0]:.3g} "
              f"to {sweep.abscissa[-1]:.3g} {sweep.units}")
        # e.g. 'time: 0 to 2.3e-04 s'
    """

    def __init__(self, name, units, abscissa):
        self.name = name
        self.units = units
        self.abscissa = abscissa

    def __repr__(self):
        """One-line summary: axis name, units, length and endpoints.

        ``__str__`` falls back to this.  An op-point file has no sweep at
        all, which shows as ``<Sweep (none)>``.
        """
        # An op-point file has no sweep: the header names no axis and the
        # decoders hand back a length-1 placeholder, not an empty array.
        if not self.name and np.asarray(self.abscissa).size <= 1:
            return "<Sweep (none) — operating-point snapshot>"
        return (f"<Sweep {self.name!r}{_fmt_units(self.units)}: "
                f"{_fmt_array(self.abscissa)}>")


class PSF:
    """
    High-level reader for Cadence Spectre PSF simulation results.

    Supports both PSFXL format (``.tran`` + ``.tran.psfxl``) and flat PSF
    format (``.ac``, ``.noise``, ``.dc``).  The correct decoder is selected
    automatically by inspecting the file magic bytes.

    Decoded data is transparently cached at two levels (in-session memory
    and on-disk pickle).  First load decodes the raw file (~1.5 s for a
    sparse PSF); subsequent loads within the same session are instant
    (<1 ms); loads from a new session hit the disk cache (~50 ms).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the PSF header file.  Common values:

        - ``"results/psf0/dcOp.dc"``   — DC operating-point snapshot
        - ``"results/psf0/tran.tran"``  — transient analysis
        - ``"results/psf0/ac.ac"``      — AC small-signal analysis
        - ``"results/psf0/ac.noise"``   — noise spectral density

    Attributes
    ----------
    path : pathlib.Path
        Resolved absolute path of the loaded header file.

    Examples
    --------
    **Quick look at an unfamiliar file** — ``repr`` reports the signal count,
    the sweep axis and a sample of signal names, so no prior knowledge of the
    file is needed to see what is in it:

    .. code-block:: python

        >>> PSF("results/psf0/tran.tran")
        <PSF 'tran.tran': 13 signals, sweep 'time' [s] × 594 points;
         vout, vin, clk, vdd, … (+9 more)>

    **DC operating-point — scalar signal access:**

    .. code-block:: python

        from psfxl_utils import PSF

        psf = PSF("results/psf0/dcOpInfo.info")
        vgs = psf.get_signal("M0.vgs").ordinate[0]
        print(f"Vgs = {vgs:.4f} V")

    **Inspect all signals:**

    .. code-block:: python

        for signal in psf.all_signals():
            print(f"{signal.name:<20} {signal.ordinate[0]:<12.6g} "
                  f"{signal.units:<8} {signal.type.kind}")

    **Monte Carlo post-processing — collect a signal across runs:**

    .. code-block:: python

        import numpy as np
        from pathlib import Path
        from psfxl_utils import PSF

        vgs_vals = []
        for run_dir in sorted(Path("mc_results").glob("run_*")):
            psf = PSF(run_dir / "dcOpInfo.info")
            vgs_vals.append(psf.get_signal("M0.vgs").ordinate[0])

        mean = np.mean(vgs_vals)
        std  = np.std(vgs_vals)
        print(f"Vgs  mean={mean:.4f} V  std={std:.4f} V")

    **Transient — sweep + signal:**

    .. code-block:: python

        psf   = PSF("results/psf0/tran.tran")
        sweep = psf.get_sweep()           # time axis
        vout  = psf.get_signal("vout")    # voltage trace
    """

    _UNITS_UNICODE = {
        'V': 'V',
        'A': 'A',
        's': 's',
        'Hz': 'Hz',
        'Ohm': 'Ω',
        'ohm': 'Ω',
        'F': 'F',
        'H': 'H',
        'W': 'W',
        'dB': 'dB',
        'deg': '°',
        'rad': 'rad',
        '%': '%',
    }

    # In-session instance cache: filepath -> {'obj': PSF, 'fingerprint': str}
    # (see psf_cache.source_fingerprint: covers the header and its .psfxl).
    _instance_cache = {}

    def __init__(self, path):
        """
        Load and decode a PSF result file.

        Checks the in-session memory cache first (hash-validated to detect
        file changes), then the on-disk pickle cache, and finally decodes
        the raw PSF file if no valid cache entry exists.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the PSF header file.
        """
        path = Path(path).resolve()
        path_str = str(path)
        # Set before the cache branches below, all three of which return
        # early; `path` names the file in __repr__ and is derived from the
        # constructor argument, never from a cached object.
        self.path = path

        # Both cache keys are taken BEFORE decoding: a key taken afterwards
        # would file a result decoded from the old file under the new file's
        # key if the file changed mid-decode (e.g. a re-run finishing).
        fingerprint = psf_cache.source_fingerprint(path_str)

        # Level 1: in-session memory cache.
        if path_str in PSF._instance_cache:
            cache_entry = PSF._instance_cache[path_str]
            cached = cache_entry['obj']
            if fingerprint is not None and fingerprint == cache_entry['fingerprint']:
                # Cache hit — copy instance data from the cached object.
                self._sweep = cached._sweep
                self._signals = cached._signals
                self._decode_warnings = getattr(cached, '_decode_warnings', [])
                for msg, cat in self._decode_warnings:
                    warnings.warn(msg, cat, stacklevel=2)
                return
            # File changed; fall through to reload.

        # Level 2: on-disk pickle cache (populated by a prior Python session).
        disk_key = psf_cache.get_cache_path(path_str)
        cached_psf = (psf_cache.load_from_cache(path_str, cache_path=disk_key)
                      if disk_key is not None else None)
        if cached_psf is not None:
            self._sweep = cached_psf._sweep
            self._signals = cached_psf._signals
            self._decode_warnings = getattr(cached_psf, '_decode_warnings', [])
            # Promote the disk-cache hit into the memory cache.
            if fingerprint is not None:
                PSF._instance_cache[path_str] = {
                    'obj': self,
                    'fingerprint': fingerprint,
                }
            for msg, cat in self._decode_warnings:
                warnings.warn(msg, cat, stacklevel=2)
            return
        
        # Level 3: full decode from disk.
        # Capture any warnings emitted by the decoders (e.g. skipped
        # intermediate type-2 checkpoint chunks) so they can be stored on
        # the PSF object and re-emitted on future cache hits.
        with open(path, 'rb') as _fh:
            _magic = struct.unpack('>I', _fh.read(4))[0]

        with warnings.catch_warnings(record=True) as _wlist:
            warnings.simplefilter('always')

            if _magic == PSF_MAGIC_INFO:
                # PSF info/snapshot format — DC op-point or plain DC results.
                time_arr, signals, traces = decode_psf_info(path)
                sweep_name   = ''
                sweep_units = ''
            else:
                header = parse_psf_header(path)
                traces = header['traces']
                sweep_meta  = header['sweep']
                sweep_name  = sweep_meta['name']
                sweep_units = sweep_meta['units']

                if header['format'] == 'flat':
                    time_arr, signals = decode_psf_flat(path, header)
                else:
                    # PSFXL: signal data is stored in a companion .psfxl file.
                    data_path = Path(str(path) + '.psfxl')
                    time_arr, signals, _ = decode_psfxl(data_path, len(traces))

        # Store captured warnings as (message_str, category) so they survive
        # pickling into the disk cache and can be re-emitted on cache hits.
        self._decode_warnings = [(str(w.message), w.category) for w in _wlist]

        self._sweep = Sweep(
            name=sweep_name,
            units=sweep_units,
            abscissa=time_arr,
         )

        self._signals = []
        for i, trace in enumerate(traces):
            ordinate = signals.get(i, np.array([]))
            self._signals.append(Signal(
                name=trace['name'],
                units=trace['units'],
                ordinate=ordinate,
             ))
        
        # Persist to disk cache for future sessions, then register in memory
        # cache -- each only if the file is unchanged since the keys were taken.
        if disk_key is not None:
            psf_cache.save_to_cache(path_str, self, cache_path=disk_key)
        if fingerprint is not None and psf_cache.source_fingerprint(path_str) == fingerprint:
            PSF._instance_cache[path_str] = {
                'obj': self,
                'fingerprint': fingerprint,
            }

        # Re-emit captured warnings now that the object is fully built and cached.
        for msg, cat in self._decode_warnings:
            warnings.warn(msg, cat, stacklevel=2)

    #: Number of signal names listed in ``repr(psf)`` before eliding the rest.
    _REPR_SIGNAL_SAMPLE = 4

    def __repr__(self):
        """One-line summary of the file: signal count, sweep axis, sample names.

        ``__str__`` falls back to this, so ``print(psf)`` is informative too.
        The point is that neither an agent nor a person should have to already
        know to call :meth:`all_signals` before it can tell what a file holds::

            <PSF 'tran.tran': 13 signals, sweep 'time' [s] × 594 points;
             vout, vin, clk, vdd, … (+9 more)>

        A file that decoded to nothing says so explicitly rather than looking
        like a normal result with an unremarkable signal list — the failure
        mode that made the type-1 chunk bug expensive to find.

        Only signal names and array lengths are touched, so this stays cheap
        on multi-million-point transients.
        """
        name = getattr(self, 'path', None)
        head = f"<PSF {name.name!r}: " if name is not None else "<PSF: "

        signals = getattr(self, '_signals', [])
        parts = [f"{len(signals)} signal{'' if len(signals) == 1 else 's'}"]

        sweep = getattr(self, '_sweep', None)
        n_sweep = np.asarray(sweep.abscissa).size if sweep is not None else 0
        if sweep is None or (not sweep.name and n_sweep <= 1):
            parts.append("no sweep (operating-point snapshot)")
        else:
            parts.append(f"sweep {sweep.name!r}{_fmt_units(sweep.units)} "
                         f"× {n_sweep} point{'' if n_sweep == 1 else 's'}")

        summary = head + ", ".join(parts)

        if signals:
            shown = [s.name for s in signals[:self._REPR_SIGNAL_SAMPLE]]
            listing = ", ".join(shown)
            remaining = len(signals) - len(shown)
            if remaining:
                listing += f", … (+{remaining} more)"
            summary += "; " + listing

        # An empty decode is a failure, not a small result.  Say so where it
        # cannot be scrolled past.
        if signals and all(np.asarray(s.ordinate).size == 0 for s in signals):
            summary += " — EMPTY: every signal decoded to zero points"

        return summary + ">"

    def all_signals(self):
        """
        Iterate over all decoded trace signals in the PSF file.

        Yields
        ------
        Signal
            Each :class:`Signal` in the order it appears in the file.

        Examples
        --------
        .. code-block:: python

            for signal in psf.all_signals():
                print(f"{signal.name:<20} {signal.ordinate[0]:.6g} {signal.units}")
        """
        return iter(self._signals)

    def get_signal(self, name):
        """
        Return the signal with the given name.

        Parameters
        ----------
        name : str
            Exact signal name as stored in the PSF file
            (e.g., ``'vout'``, ``'DUT.M1.ids'``).

        Returns
        -------
        Signal
            The matching :class:`Signal` object.

        Raises
        ------
        KeyError
            If no signal with that name exists in the file.

        Examples
        --------
        .. code-block:: python

            # Scalar access for DC op-point
            vgs = psf.get_signal("M0.vgs").ordinate[0]

            # Array access for swept analysis
            vout = psf.get_signal("vout").ordinate   # 1-D NumPy array
        """
        for signal in self._signals:
            if signal.name == name:
                return signal
        raise KeyError(f"Signal '{name}' not found")

    def get_sweep(self):
        """
        Return the sweep (independent) axis for this PSF result.

        Returns
        -------
        Sweep
            A :class:`Sweep` object with ``name``, ``units``, and
            ``abscissa`` (NumPy array of sweep point values).

        Examples
        --------
        .. code-block:: python

            sweep = psf.get_sweep()
            print(f"{sweep.name}: {sweep.abscissa[0]:.3g} "
                  f"to {sweep.abscissa[-1]:.3g} {sweep.units}")
        """
        return self._sweep

    @staticmethod
    def units_to_unicode(units):
        """
        Convert a PSF unit string to its Unicode symbol.

        Parameters
        ----------
        units : str
            Unit string as stored in the PSF file (e.g., ``'Ohm'``, ``'deg'``).

        Returns
        -------
        str
            Unicode symbol when a mapping exists; otherwise returns the
            original string unchanged.

        Examples
        --------
        .. code-block:: python

            PSF.units_to_unicode("Ohm")  # → 'Ω'
            PSF.units_to_unicode("deg")  # → '°'
            PSF.units_to_unicode("V")    # → 'V'
            PSF.units_to_unicode("ppm")  # → 'ppm'  (pass-through)
        """
        return PSF._UNITS_UNICODE.get(units, units)
