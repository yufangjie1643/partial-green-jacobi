"""Editable C3D8R-HG preconditioner boundary.

The surrounding finite-element operator and PCG implementation are frozen for
the Praxist task. Research variants may replace implementation details inside
this package while preserving the public functions exported here.
"""

from .c3d8r_fft import apply_preconditioner, enable_fft_preconditioner

__all__ = ["apply_preconditioner", "enable_fft_preconditioner"]
