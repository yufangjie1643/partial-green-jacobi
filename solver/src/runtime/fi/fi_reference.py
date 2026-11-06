"""HG-style low-precision reference inverse for the unchanged full-integration FE operator.

The FI reference stencil still comes from the frozen eight-Gauss-point builder.
Only its FFT inverse/application can use FP32. Optional Hermitian upper-six
storage reuses the frozen HG spectral-multiply CUDA kernel, not its HG stencil.
"""
from pathlib import Path
from types import SimpleNamespace, MethodType
from weakref import proxy
import importlib.util
import torch
from femlib.batched.stencil_operator import BatchedStencilOperator

_spec = importlib.util.spec_from_file_location(
    'frozen_hg_spectral', Path(__file__).parent/'hg_spectral/c3d8r_hourglass_cuda.py')
_hg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hg)


def _spectral_upper6(inverse, spectrum):
    return _hg.c3d8r_fft_spectral_multiply_cuda(inverse, spectrum, block_size=128)


def _mixed_apply_batch(self, residual, inverse):
    # Identical precision boundary to the existing HG mixed_preconditioner.py.
    context = SimpleNamespace(
        mesh=self.mesh, dtype=torch.float32,
        _fft_spectrum_mode=self._fft_spectrum_mode,
        _fft_inverse_norm_folded=self._fft_inverse_norm_folded,
        _fft_spectral_multiply=self._fft_spectral_multiply)
    out = BatchedStencilOperator._apply_fft_base_precond_batch(
        context, residual.to(torch.float32), inverse)
    return out.to(residual.dtype)


def enable_reference(op, precision='mixed', layout='upper6'):
    if precision not in ('fp64', 'mixed') or layout not in ('full9', 'upper6'):
        raise ValueError((precision, layout))
    assert op.dtype == op.C.dtype == op.S_flat.dtype == torch.float64
    assert op.element_type == 'c3d8' and op.B_mat.shape[0] == 8
    op.__dict__.pop('_apply_fft_base_precond_batch', None)
    op.__dict__.pop('_fft_spectral_multiply', None)
    reference = op._compute_reference_medium('zeman', verbose=False)
    context = SimpleNamespace(
        B=op.B, mesh=op.mesh, device=op.device,
        dtype=torch.float32 if precision == 'mixed' else torch.float64,
        element_type=op.element_type, _B_np=op._B_np, detJ=op.detJ)

    def build_symbol(self, S0_center, **kwargs):
        return BatchedStencilOperator._build_fft_symbol_inv(
            self, S0_center, spectrum_mode='real_half', **kwargs)

    context._build_fft_symbol_inv = MethodType(build_symbol, context)
    # Explicit C0 prevents a second heterogeneous material owner; the FI
    # method constructs the reference using its own 2x2x2 Gauss quadrature.
    BatchedStencilOperator.enable_fft_precond(context, C0_voigt=reference)
    if layout == 'upper6':
        inverse = context._fft_symbol_inv
        context._fft_symbol_inv = torch.stack(
            [inverse[:, i, j] for i, j in ((0, 0), (0, 1), (0, 2),
                                          (1, 1), (1, 2), (2, 2))], dim=1).contiguous()
        del inverse
    for name, value in vars(context).items():
        if name.startswith('_fft_') or name == '_M_inv_cache':
            setattr(op, name, value)
    op._fft_symbol_layout = layout
    if layout == 'upper6':
        op._fft_spectral_multiply = _spectral_upper6
    if precision == 'mixed':
        op._apply_fft_base_precond_batch = MethodType(_mixed_apply_batch, proxy(op))
    assert op.C.dtype == op.S_flat.dtype == torch.float64
    assert op._fft_symbol_inv.dtype == (torch.complex64 if precision == 'mixed' else torch.complex128)

