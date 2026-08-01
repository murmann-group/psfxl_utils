"""
Cadence Spectre PSFXL and Flat PSF Binary File Decoder

Public API for reading PSF simulation results in both formats:
  - PSFXL: .tran (header) + .tran.psfxl (compressed data) + .tran.sig (metadata)
  - Flat PSF: .ac, .noise, .dc (self-contained files)

Example:
    >>> from psfxl_utils import PSF
    >>> psf = PSF('results/tran.tran')
    >>> sweep = psf.get_sweep()
    >>> signal = psf.get_signal('node_name')

Requirements:
    pip install numpy blosc packaging (or numcodecs instead of blosc)
"""

from .psf_class import PSF, Signal, Sweep
from .psf_parse import parse_psf_header
from .psfxl_decoder import decode_psfxl, parse_sig_file
from .flat_psf_decoder import decode_psf_flat

__all__ = [
    'PSF',
    'Signal',
    'Sweep',
    'parse_psf_header',
    'decode_psfxl',
    'decode_psf_flat',
    'parse_sig_file',
]
