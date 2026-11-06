"""Lower-precision reference inverse; the physical operator stays FP64.

Uses the frozen builder and spectral multiply with a small dtype-specific
reference context. No second heterogeneous material field is constructed.
"""
from types import SimpleNamespace, MethodType
from weakref import proxy
import torch
from femlib.batched.stencil_operator import BatchedStencilOperator
from femlib.batched.preconditioners.c3d8r_fft import enable_fft_preconditioner


def _mixed_apply_batch(self, residual, inverse):
    context = SimpleNamespace(
        mesh=self.mesh, dtype=torch.float32,
        _fft_spectrum_mode=self._fft_spectrum_mode,
        _fft_inverse_norm_folded=self._fft_inverse_norm_folded,
        _fft_spectral_multiply=self._fft_spectral_multiply)
    out = BatchedStencilOperator._apply_fft_base_precond_batch(
        context, residual.to(torch.float32), inverse)
    return out.to(residual.dtype)


def enable_reference(operator, fft=2, precision='fp64'):
    """Build the same per-RVE reference, then choose its execution precision."""
    if precision not in ('fp64', 'mixed'):
        raise ValueError(precision)
    if operator.dtype != torch.float64:
        raise ValueError('The physical operator must remain FP64')
    operator._pack_real_half_fft_symbol = (fft == 2)
    mode = 'complex_full' if fft == 0 else 'real_half'
    if precision == 'fp64':
        # Remove only this adapter if switching an existing operator back.
        operator.__dict__.pop('_apply_fft_base_precond_batch', None)
        operator.enable_fft_precond(ref_strategy='zeman', spectrum_mode=mode,
            validation_mode='vectorized_hermitian', allow_symbol_sharing=False)
        return
    if operator._C_voigt_gpu is operator.C:
        reference = operator._compute_reference_medium_from_packed('zeman', verbose=False)
    else:
        reference = BatchedStencilOperator._compute_reference_medium(
            operator, 'zeman', verbose=False)
    context = SimpleNamespace(B=operator.B, mesh=operator.mesh,
        device=operator.device, dtype=torch.float32,
        hourglass_coefficient=operator.hourglass_coefficient,
        _pack_real_half_fft_symbol=(fft == 2))
    enable_fft_preconditioner(context, ref_strategy='zeman', C0_voigt=reference,
        spectrum_mode=mode, validation_mode='vectorized_hermitian',
        allow_symbol_sharing=False)
    for name, value in vars(context).items():
        if name.startswith('_fft_') or name == '_M_inv_cache':
            setattr(operator, name, value)
    # Avoid a self-reference cycle: batch cleanup must release the GPU state
    # immediately, with the same lifetime as the unmodified FP64 operator.
    operator._apply_fft_base_precond_batch = MethodType(_mixed_apply_batch, proxy(operator))
    assert operator.C.dtype == torch.float64
    assert operator._fft_symbol_inv.dtype == torch.complex64
