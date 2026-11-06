# -*- coding: utf-8 -*-
"""Batched Jacobi-PCG: (B, n_dofs) right-hand side, all inner products / norms computed independently per batch.

Stopping criterion: the relative residual of every batch is below `tol`, or `max_iter` is reached.
On GPU the kernel is SIMD over the batch dimension, so freezing already-converged batches wouldn't save time;
for simplicity all batches iterate together until max-of-residuals < tol.
"""

from __future__ import annotations

import math
import operator
from functools import lru_cache

import torch
import triton
import triton.language as tl


def solve_pcg_batched(op, rhs, *, tol=1e-6, max_iter=2000, verbose=False,
                      log_every=50, update_mode="eager", x0=None,
                      recycle_callback=None, recycle_stride=25,
                      convergence_check_stride=1,
                      convergence_check_adaptive_factor=None,
                      verify_final_residual=False):
    """
    Parameters
    ----------
    op   : provides apply((B,n)->(B,n)) and apply_precond((B,n)->(B,n))
           (if the latter is missing, falls back to diag() + 1/diag Jacobi)
    rhs  : (B, n) torch tensor
    convergence_check_adaptive_factor : float or None
           With compiled CUDA updates and a stride greater than one, switch
           permanently to per-iteration host checks after every batch reaches
           ``factor * tol``. ``None`` preserves fixed-stride behavior.

    Returns
    -------
    u        : (B, n) best (lowest-residual) solution
    n_iter   : int  iteration count at global stop
    rel_res  : (B,) final relative residual per batch
    """
    convergence_check_stride = _validate_convergence_check_stride(
        convergence_check_stride,
    )

    apply_M_inv = _make_precond(op)
    use_compiled = update_mode == "compiled"
    use_triton = update_mode == "triton"
    use_cuda = update_mode == "cuda"
    if update_mode not in ("eager", "compiled", "triton", "cuda"):
        raise ValueError(f"unknown PCG update mode: {update_mode!r}")
    if (use_triton or use_cuda) and getattr(op, "_fft_enabled", False):
        raise ValueError(
            f"update_mode={update_mode!r} currently supports Jacobi preconditioning only"
        )
    if use_cuda and rhs.device.type != "cuda":
        raise ValueError("update_mode='cuda' requires CUDA solver tensors")

    B, N = rhs.shape
    dtype, device = rhs.dtype, rhs.device
    adaptive_host_checks = convergence_check_adaptive_factor is not None
    if adaptive_host_checks:
        adaptive_factor = _validate_adaptive_factor(
            convergence_check_adaptive_factor,
        )
        if not (
            use_compiled
            and device.type == "cuda"
            and convergence_check_stride > 1
        ):
            raise ValueError(
                "adaptive convergence checks require update_mode='compiled', "
                "CUDA tensors, and convergence_check_stride > 1"
            )
    else:
        adaptive_factor = None

    if x0 is None:
        u = torch.zeros_like(rhs)
        r = rhs.clone()
    else:
        u = x0.clone()
        r = rhs - op.apply(u)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1).clamp(min=1e-30)   # (B,)

    z = apply_M_inv(r)
    p = z.clone()
    rz = (r * z).sum(-1)                           # (B,)

    if use_triton:
        M_inv = _jacobi_m_inv(op)
        return _solve_pcg_batched_triton_jacobi(
            op, rhs, M_inv, u, r, p, rz, rhs_norm,
            tol=tol, max_iter=max_iter, verbose=verbose, log_every=log_every,
            convergence_check_stride=convergence_check_stride,
            verify_final_residual=verify_final_residual,
        )
    if use_cuda:
        M_inv = _jacobi_m_inv(op)
        if M_inv.shape != rhs.shape:
            M_inv = torch.broadcast_to(M_inv, rhs.shape).contiguous()
        elif not M_inv.is_contiguous():
            M_inv = M_inv.contiguous()
        return _solve_pcg_batched_cuda_jacobi(
            op, rhs, M_inv, u, r, p, rz, rhs_norm,
            tol=tol, max_iter=max_iter, verbose=verbose, log_every=log_every,
            recycle_callback=recycle_callback, recycle_stride=recycle_stride,
            convergence_check_stride=convergence_check_stride,
            verify_final_residual=verify_final_residual,
        )

    best_res = torch.full((B,), float('inf'), device=device, dtype=dtype)
    best_u = u.clone()
    eps = torch.tensor(1e-30, device=device, dtype=dtype)
    host_check_stride = convergence_check_stride
    use_compiled_stop_iter = (
        use_compiled
        and convergence_check_stride > 1
        and device.type == "cuda"
    )
    if use_compiled and convergence_check_stride > 1 and device.type != "cuda":
        host_check_stride = 1
    if use_compiled_stop_iter:
        tol_tensor = torch.full((1,), tol, device=device, dtype=dtype)
        if adaptive_host_checks:
            convergence_state = torch.zeros(
                (2,), device=device, dtype=torch.int64,
            )
            stop_iter = convergence_state[:1]
            near_tol_tensor = torch.full(
                (1,), adaptive_factor * tol, device=device, dtype=dtype,
            )
        else:
            stop_iter = torch.zeros((1,), device=device, dtype=torch.int64)
        latch_block = triton.next_power_of_2(B)
    compiled_pre, compiled_post = _compiled_pcg_steps() if use_compiled else (None, None)
    compiled_latched_pre = (
        _compiled_pcg_latched_pre_step() if use_compiled_stop_iter else None
    )

    n_iter_final = max_iter
    last_res = torch.full((B,), float('inf'), device=device, dtype=dtype)

    for it in range(max_iter):
        Ap = op.apply(p)                           # (B, n)
        if recycle_callback is not None and it % max(int(recycle_stride), 1) == 0:
            recycle_callback(it, p, Ap)
        if use_compiled:
            if use_compiled_stop_iter:
                u, r, res = compiled_latched_pre(
                    u, r, p, Ap, rz, rhs_norm, eps,
                )
                _copy_best_latched_kernel[(triton.cdiv(N, 1024), B)](
                    u, best_u, res, best_res, stop_iter, N, BLOCK=1024,
                )
                _update_best_res_latched_kernel[(B,)](
                    res, best_res, stop_iter,
                )
                if adaptive_host_checks:
                    _latch_adaptive_convergence_kernel[(1,)](
                        res,
                        convergence_state,
                        tol_tensor,
                        near_tol_tensor,
                        it + 1,
                        B,
                        BLOCK=latch_block,
                    )
                else:
                    _latch_global_convergence_kernel[(1,)](
                        res, stop_iter, tol_tensor, it + 1, B,
                        BLOCK=latch_block,
                    )
            else:
                u, r, res, best_u, best_res = compiled_pre(
                    u, r, p, Ap, rz, rhs_norm, best_u, best_res, eps,
                )
            last_res = res
        else:
            pAp_raw = (p * Ap).sum(-1)                 # (B,)
            # For batches with non-SPD direction (non-finite or ≤0), freeze this update to avoid wrecking the whole batch
            bad = ~torch.isfinite(pAp_raw) | (pAp_raw <= 0)
            pAp = pAp_raw.clamp(min=eps)               # (B,)
            alpha = (rz / pAp).unsqueeze(-1)           # (B, 1)
            if bad.any():
                alpha = torch.where(bad.unsqueeze(-1),
                                    torch.zeros_like(alpha), alpha)
            u = u + alpha * p
            r = r - alpha * Ap
            res = torch.linalg.vector_norm(r, dim=-1) / rhs_norm     # (B,)
            last_res = res

            # Update best_u per batch
            improved = res < best_res
            if improved.any():
                mask = improved.view(-1, *([1] * (u.ndim - 1)))
                best_u = torch.where(mask, u, best_u)
                best_res = torch.where(improved, res, best_res)

        if verbose and ((it + 1) % log_every == 0 or it == 0):
            print(f"    [PCG-batched] iter {it+1:5d}  "
                  f"max_res={res.max().item():.3e}  "
                  f"min_res={res.min().item():.3e}")

        check_now = (
            it == 0
            or (it + 1) % host_check_stride == 0
            or (it + 1) == max_iter
        )
        logical_stop_iter = 0
        if check_now:
            if use_compiled_stop_iter:
                if adaptive_host_checks:
                    logical_stop_iter, near_latched = (
                        int(value) for value in convergence_state.tolist()
                    )
                    if logical_stop_iter == 0 and near_latched:
                        host_check_stride = 1
                else:
                    logical_stop_iter = int(stop_iter.item())
            elif res.max().item() < tol:
                logical_stop_iter = it + 1
        if logical_stop_iter > 0:
            n_iter_final = logical_stop_iter
            if verbose:
                print(f"    [PCG-batched] iter {n_iter_final:5d}  all batches converged")
            if verify_final_residual:
                return_res = _true_relative_residual(op, rhs, best_u, rhs_norm)
            elif use_compiled_stop_iter:
                return_res = best_res
            else:
                return_res = res
            return best_u, n_iter_final, return_res

        if (it + 1) == max_iter:
            break
        z = apply_M_inv(r)
        if use_compiled:
            p, rz = compiled_post(r, p, z, rz, eps)
        else:
            rz_new = (r * z).sum(-1)
            beta = (rz_new / rz.clamp(min=eps)).unsqueeze(-1)
            p = z + beta * p
            rz = rz_new

    if verbose:
        print(f"    [PCG-batched] max_iter={max_iter} reached, "
              f"max_res={last_res.max().item():.3e}")
    if verify_final_residual:
        best_res = _true_relative_residual(op, rhs, best_u, rhs_norm)
    return best_u, n_iter_final, best_res


def _true_relative_residual(op, rhs, u, rhs_norm):
    residual = rhs - op.apply(u)
    return torch.linalg.vector_norm(residual, dim=-1) / rhs_norm


def _validate_convergence_check_stride(value):
    try:
        stride = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            "convergence_check_stride must be a positive integer"
        ) from exc
    if stride < 1:
        raise ValueError("convergence_check_stride must be at least one")
    return stride


def _validate_adaptive_factor(value):
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "convergence_check_adaptive_factor must be finite and greater than one"
        ) from exc
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError(
            "convergence_check_adaptive_factor must be finite and greater than one"
        )
    return factor


def _make_precond(op):
    """Return a callable r ↦ M^{-1} r. Prefer op.apply_precond, else fall back to Jacobi."""
    if hasattr(op, "apply_precond"):
        return op.apply_precond
    cache = getattr(op, "_M_inv_cache", None)
    if cache is None:
        cache = 1.0 / op.diag()
        op._M_inv_cache = cache
    return lambda r: cache * r


def _jacobi_m_inv(op):
    cache = getattr(op, "_M_inv_cache", None)
    if cache is None:
        cache = 1.0 / op.diag()
        op._M_inv_cache = cache
    return cache


def _solve_pcg_batched_triton_jacobi(op, rhs, M_inv, u, r, p, rz, rhs_norm,
                                     *, tol, max_iter, verbose, log_every,
                                     convergence_check_stride, verify_final_residual):
    """Experimental Jacobi-PCG update path with Triton vector kernels.

    This path keeps the same best-iterate return semantics as the eager path,
    but only supports Jacobi preconditioning.
    """
    B, N = rhs.shape
    dtype, device = rhs.dtype, rhs.device
    block = 1024
    n_blocks = triton.cdiv(N, block)
    partial_pAp = torch.empty((B, n_blocks), device=device, dtype=dtype)
    partial_rr = torch.empty((B, n_blocks), device=device, dtype=dtype)
    partial_rz = torch.empty((B, n_blocks), device=device, dtype=dtype)
    z = torch.empty_like(r)
    res = torch.full((B,), float("inf"), device=device, dtype=dtype)
    best_res = torch.full((B,), float("inf"), device=device, dtype=dtype)
    best_u = u.clone()
    eps = torch.tensor(1e-30, device=device, dtype=dtype)

    n_iter_final = max_iter
    grid = (n_blocks, B)
    for it in range(max_iter):
        Ap = op.apply(p)
        _dot_kernel[grid](p, Ap, partial_pAp, N, BLOCK=block)
        pAp_raw = partial_pAp.sum(dim=1)
        bad = ~torch.isfinite(pAp_raw) | (pAp_raw <= 0)
        alpha = rz / pAp_raw.clamp(min=eps)
        alpha = torch.where(bad, torch.zeros_like(alpha), alpha)

        _update_ur_res_kernel[grid](
            u, r, p, Ap, alpha, partial_rr, N, BLOCK=block,
        )
        rr = partial_rr.sum(dim=1)
        res = torch.sqrt(rr.clamp(min=0.0)) / rhs_norm
        _update_best_kernel[grid](u, best_u, res, best_res, N, BLOCK=block)

        if verbose and ((it + 1) % log_every == 0 or it == 0):
            print(f"    [PCG-batched:Triton] iter {it+1:5d}  "
                  f"max_res={res.max().item():.3e}  "
                  f"min_res={res.min().item():.3e}")

        check_now = (
            it == 0
            or (it + 1) % convergence_check_stride == 0
            or (it + 1) == max_iter
        )
        if check_now and res.max().item() < tol:
            n_iter_final = it + 1
            if verbose:
                print(f"    [PCG-batched:Triton] iter {it+1:5d}  all batches converged")
            if verify_final_residual:
                res = _true_relative_residual(op, rhs, best_u, rhs_norm)
            return best_u, n_iter_final, res

        _jacobi_z_rz_kernel[grid](r, M_inv, z, partial_rz, N, BLOCK=block)
        rz_new = partial_rz.sum(dim=1)
        beta = rz_new / rz.clamp(min=eps)
        _update_p_kernel[grid](p, z, beta, N, BLOCK=block)
        rz = rz_new

    if verbose:
        print(f"    [PCG-batched:Triton] max_iter={max_iter} reached, "
              f"max_res={res.max().item():.3e}")
    if verify_final_residual:
        best_res = _true_relative_residual(op, rhs, best_u, rhs_norm)
    return best_u, n_iter_final, best_res


def _solve_pcg_batched_cuda_jacobi(
    op,
    rhs,
    M_inv,
    u,
    r,
    p,
    rz,
    rhs_norm,
    *,
    tol,
    max_iter,
    verbose,
    log_every,
    recycle_callback,
    recycle_stride,
    convergence_check_stride,
    verify_final_residual,
):
    """Jacobi-PCG path with fused native CUDA vector updates and reductions."""

    from .pcg_cuda import pcg_jacobi_step_cuda

    B, N = rhs.shape
    dtype, device = rhs.dtype, rhs.device
    block = 256
    n_blocks = (N + block - 1) // block
    workspace = torch.empty(
        (3, B, n_blocks), device=device, dtype=dtype,
    )
    state = torch.empty((3, B), device=device, dtype=dtype)
    res = torch.full((B,), float("inf"), device=device, dtype=dtype)
    best_res = torch.full((B,), float("inf"), device=device, dtype=dtype)
    best_u = u.clone()
    recycle_stride = max(int(recycle_stride), 1)

    n_iter_final = max_iter
    for it in range(max_iter):
        Ap = op.apply(p)
        if recycle_callback is not None and it % recycle_stride == 0:
            recycle_callback(it, p, Ap)
        pcg_jacobi_step_cuda(
            u,
            r,
            p,
            Ap,
            M_inv,
            rz,
            rhs_norm,
            best_u,
            best_res,
            res,
            workspace,
            state,
            block_size=block,
        )

        if verbose and ((it + 1) % log_every == 0 or it == 0):
            print(f"    [PCG-batched:CUDA] iter {it+1:5d}  "
                  f"max_res={res.max().item():.3e}  "
                  f"min_res={res.min().item():.3e}")

        check_now = (
            it == 0
            or (it + 1) % convergence_check_stride == 0
            or (it + 1) == max_iter
        )
        if check_now and res.max().item() < tol:
            n_iter_final = it + 1
            if verbose:
                print(f"    [PCG-batched:CUDA] iter {it+1:5d}  all batches converged")
            if verify_final_residual:
                res = _true_relative_residual(op, rhs, best_u, rhs_norm)
            return best_u, n_iter_final, res

    if verbose:
        print(f"    [PCG-batched:CUDA] max_iter={max_iter} reached, "
              f"max_res={res.max().item():.3e}")
    if verify_final_residual:
        best_res = _true_relative_residual(op, rhs, best_u, rhs_norm)
    return best_u, n_iter_final, best_res


@triton.jit
def _dot_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b * N + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    y = tl.load(y_ptr + base, mask=mask, other=0.0)
    acc = tl.sum(x * y, axis=0)
    n_blocks = tl.cdiv(N, BLOCK)
    tl.store(out_ptr + b * n_blocks + pid, acc)


@triton.jit
def _update_ur_res_kernel(u_ptr, r_ptr, p_ptr, Ap_ptr, alpha_ptr, rr_ptr,
                          N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b * N + offs
    alpha = tl.load(alpha_ptr + b)
    u = tl.load(u_ptr + base, mask=mask, other=0.0)
    r = tl.load(r_ptr + base, mask=mask, other=0.0)
    p = tl.load(p_ptr + base, mask=mask, other=0.0)
    Ap = tl.load(Ap_ptr + base, mask=mask, other=0.0)
    u_new = u + alpha * p
    r_new = r - alpha * Ap
    tl.store(u_ptr + base, u_new, mask=mask)
    tl.store(r_ptr + base, r_new, mask=mask)
    acc = tl.sum(r_new * r_new, axis=0)
    n_blocks = tl.cdiv(N, BLOCK)
    tl.store(rr_ptr + b * n_blocks + pid, acc)


@triton.jit
def _jacobi_z_rz_kernel(r_ptr, m_inv_ptr, z_ptr, rz_ptr,
                        N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b * N + offs
    r = tl.load(r_ptr + base, mask=mask, other=0.0)
    m_inv = tl.load(m_inv_ptr + base, mask=mask, other=0.0)
    z = r * m_inv
    tl.store(z_ptr + base, z, mask=mask)
    acc = tl.sum(r * z, axis=0)
    n_blocks = tl.cdiv(N, BLOCK)
    tl.store(rz_ptr + b * n_blocks + pid, acc)


@triton.jit
def _update_best_kernel(u_ptr, best_u_ptr, res_ptr, best_res_ptr,
                        N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    improved = tl.load(res_ptr + b) < tl.load(best_res_ptr + b)
    if improved:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        base = b * N + offs
        u = tl.load(u_ptr + base, mask=mask, other=0.0)
        tl.store(best_u_ptr + base, u, mask=mask)
        if pid == 0:
            tl.store(best_res_ptr + b, tl.load(res_ptr + b))


@triton.jit
def _update_p_kernel(p_ptr, z_ptr, beta_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b * N + offs
    beta = tl.load(beta_ptr + b)
    z = tl.load(z_ptr + base, mask=mask, other=0.0)
    p_old = tl.load(p_ptr + base, mask=mask, other=0.0)
    tl.store(p_ptr + base, z + beta * p_old, mask=mask)


def warmup_compiled_pcg_updates(
    op,
    rhs,
    *,
    tol=1e-6,
    convergence_check_stride=1,
    convergence_check_adaptive_factor=None,
):
    """Compile the optional fused PCG vector-update kernels for this shape."""
    convergence_check_stride = _validate_convergence_check_stride(
        convergence_check_stride,
    )

    apply_M_inv = _make_precond(op)
    B, N = rhs.shape
    dtype, device = rhs.dtype, rhs.device
    adaptive_host_checks = convergence_check_adaptive_factor is not None
    if adaptive_host_checks:
        adaptive_factor = _validate_adaptive_factor(
            convergence_check_adaptive_factor,
        )
        if device.type != "cuda" or convergence_check_stride <= 1:
            raise ValueError(
                "adaptive convergence checks require CUDA tensors and "
                "convergence_check_stride > 1"
            )
    u = torch.zeros_like(rhs)
    r = rhs.clone()
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1).clamp(min=1e-30)
    z = apply_M_inv(r)
    p = z.clone()
    rz = (r * z).sum(-1)
    best_res = torch.full((B,), float('inf'), device=device, dtype=dtype)
    best_u = u.clone()
    eps = torch.tensor(1e-30, device=device, dtype=dtype)
    Ap = op.apply(p)
    compiled_pre, compiled_post = _compiled_pcg_steps()
    if device.type == "cuda" and convergence_check_stride > 1:
        compiled_latched_pre = _compiled_pcg_latched_pre_step()
        u, r, res = compiled_latched_pre(u, r, p, Ap, rz, rhs_norm, eps)
        if adaptive_host_checks:
            convergence_state = torch.zeros(
                (2,), device=device, dtype=torch.int64,
            )
            stop_iter = convergence_state[:1]
            near_tol_tensor = torch.full(
                (1,), adaptive_factor * tol, device=device, dtype=dtype,
            )
        else:
            stop_iter = torch.zeros((1,), device=device, dtype=torch.int64)
        tol_tensor = torch.full((1,), tol, device=device, dtype=dtype)
        _copy_best_latched_kernel[(triton.cdiv(N, 1024), B)](
            u, best_u, res, best_res, stop_iter, N, BLOCK=1024,
        )
        _update_best_res_latched_kernel[(B,)](res, best_res, stop_iter)
        if adaptive_host_checks:
            _latch_adaptive_convergence_kernel[(1,)](
                res,
                convergence_state,
                tol_tensor,
                near_tol_tensor,
                1,
                B,
                BLOCK=triton.next_power_of_2(B),
            )
        else:
            _latch_global_convergence_kernel[(1,)](
                res, stop_iter, tol_tensor, 1, B,
                BLOCK=triton.next_power_of_2(B),
            )
    else:
        u, r, _, best_u, best_res = compiled_pre(
            u, r, p, Ap, rz, rhs_norm, best_u, best_res, eps,
        )
    z = apply_M_inv(r)
    p, rz = compiled_post(r, p, z, rz, eps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return u, p, rz, best_u, best_res


@lru_cache(maxsize=1)
def _compiled_pcg_steps():
    return (
        torch.compile(
            _pcg_pre_precond_step,
            fullgraph=True,
            options={"triton.cudagraphs": False},
        ),
        torch.compile(
            _pcg_post_precond_step,
            fullgraph=True,
            options={"triton.cudagraphs": False},
        ),
    )


def _pcg_pre_precond_step(u, r, p, Ap, rz, rhs_norm, best_u, best_res, eps):
    pAp_raw = (p * Ap).sum(-1)
    bad = ~torch.isfinite(pAp_raw) | (pAp_raw <= 0)
    alpha = (rz / pAp_raw.clamp(min=eps)).unsqueeze(-1)
    alpha = torch.where(bad.unsqueeze(-1), torch.zeros_like(alpha), alpha)

    u_new = u + alpha * p
    r_new = r - alpha * Ap
    res = torch.linalg.vector_norm(r_new, dim=-1) / rhs_norm

    improved = res < best_res
    mask = improved.unsqueeze(-1)
    best_u_new = torch.where(mask, u_new, best_u)
    best_res_new = torch.where(improved, res, best_res)
    return u_new, r_new, res, best_u_new, best_res_new


def _pcg_pre_precond_step_latched(u, r, p, Ap, rz, rhs_norm, eps):
    pAp_raw = (p * Ap).sum(-1)
    bad = ~torch.isfinite(pAp_raw) | (pAp_raw <= 0)
    alpha = (rz / pAp_raw.clamp(min=eps)).unsqueeze(-1)
    alpha = torch.where(bad.unsqueeze(-1), torch.zeros_like(alpha), alpha)

    u_new = u + alpha * p
    r_new = r - alpha * Ap
    res = torch.linalg.vector_norm(r_new, dim=-1) / rhs_norm

    return u_new, r_new, res


@lru_cache(maxsize=1)
def _compiled_pcg_latched_pre_step():
    return torch.compile(
        _pcg_pre_precond_step_latched,
        fullgraph=True,
        options={"triton.cudagraphs": False},
    )


@triton.jit
def _latch_global_convergence_kernel(res_ptr, stop_iter_ptr, tol_ptr, iteration,
                                     B: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    residuals = tl.load(res_ptr + offsets, mask=offsets < B, other=float("inf"))
    tol = tl.load(tol_ptr)
    converged = tl.sum(tl.where(residuals < tol, 1, 0), axis=0) == B
    stop_iter = tl.load(stop_iter_ptr)
    if converged & (stop_iter == 0):
        tl.store(stop_iter_ptr, iteration)


@triton.jit
def _latch_adaptive_convergence_kernel(
    res_ptr,
    state_ptr,
    tol_ptr,
    near_tol_ptr,
    iteration,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    residuals = tl.load(
        res_ptr + offsets, mask=offsets < B, other=float("inf"),
    )
    tol = tl.load(tol_ptr)
    near_tol = tl.load(near_tol_ptr)
    converged = tl.sum(tl.where(residuals < tol, 1, 0), axis=0) == B
    near = tl.sum(tl.where(residuals < near_tol, 1, 0), axis=0) == B
    stop_iter = tl.load(state_ptr)
    near_latched = tl.load(state_ptr + 1)
    if converged & (stop_iter == 0):
        tl.store(state_ptr, iteration)
    if near & (near_latched == 0):
        tl.store(state_ptr + 1, 1)


@triton.jit
def _copy_best_latched_kernel(u_ptr, best_u_ptr, res_ptr, best_res_ptr,
                              stop_iter_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    stop_iter = tl.load(stop_iter_ptr)
    improved = (
        (tl.load(res_ptr + b) < tl.load(best_res_ptr + b))
        & (stop_iter == 0)
    )
    if improved:
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        base = b * N + offsets
        tl.store(best_u_ptr + base, tl.load(u_ptr + base, mask=mask), mask=mask)


@triton.jit
def _update_best_res_latched_kernel(res_ptr, best_res_ptr, stop_iter_ptr):
    b = tl.program_id(0)
    stop_iter = tl.load(stop_iter_ptr)
    res = tl.load(res_ptr + b)
    best_res = tl.load(best_res_ptr + b)
    if (res < best_res) & (stop_iter == 0):
        tl.store(best_res_ptr + b, res)


def _pcg_post_precond_step(r, p, z, rz, eps):
    rz_new = (r * z).sum(-1)
    beta = (rz_new / rz.clamp(min=eps)).unsqueeze(-1)
    p_new = z + beta * p
    return p_new, rz_new
