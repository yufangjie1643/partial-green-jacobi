"""Vectorized validation of nonzero 3x3 Hermitian FFT symbol blocks."""

from __future__ import annotations

import torch


_VALIDATION_CHUNK_SIZE = 65_536
_LEGACY_VALIDATION_CHUNK_SIZE = 512
_THRESHOLD_AMBIGUITY_ULPS = 16.0


def hermitian_eigen_extrema_3x3(
    block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the minimum and maximum-absolute eigenvalues of 3x3 blocks."""
    hermitian = 0.5 * (block + block.mH)
    a00 = hermitian[..., 0, 0].real
    a11 = hermitian[..., 1, 1].real
    a22 = hermitian[..., 2, 2].real
    a01 = hermitian[..., 0, 1]
    a02 = hermitian[..., 0, 2]
    a12 = hermitian[..., 1, 2]

    scale = torch.maximum(
        torch.maximum(torch.maximum(a00.abs(), a11.abs()), a22.abs()),
        torch.maximum(torch.maximum(a01.abs(), a02.abs()), a12.abs()),
    )
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    a00, a11, a22 = a00 / safe_scale, a11 / safe_scale, a22 / safe_scale
    a01, a02, a12 = a01 / safe_scale, a02 / safe_scale, a12 / safe_scale

    q = (a00 + a11 + a22) / 3.0
    p2 = (
        (a00 - q).square()
        + (a11 - q).square()
        + (a22 - q).square()
        + 2.0 * (a01.abs().square() + a02.abs().square() + a12.abs().square())
    )
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=0.0))
    safe_p = torch.where(p > 0.0, p, torch.ones_like(p))
    b00, b11, b22 = (a00 - q) / safe_p, (a11 - q) / safe_p, (a22 - q) / safe_p
    b01, b02, b12 = a01 / safe_p, a02 / safe_p, a12 / safe_p
    det_b = (
        b00 * b11 * b22
        + 2.0 * (b01 * b12 * b02.conj()).real
        - b00 * b12.abs().square()
        - b11 * b02.abs().square()
        - b22 * b01.abs().square()
    )
    phi = torch.acos(torch.clamp(0.5 * det_b, -1.0, 1.0)) / 3.0
    eig_max = q + 2.0 * p * torch.cos(phi)
    eig_min = q + 2.0 * p * torch.cos(phi + 2.0 * torch.pi / 3.0)
    eig_mid = 3.0 * q - eig_min - eig_max
    degenerate = p == 0.0
    eig_min = torch.where(degenerate, q, eig_min)
    eig_max = torch.where(degenerate, q, eig_max)
    eig_mid = torch.where(degenerate, q, eig_mid)
    max_abs = torch.maximum(
        torch.maximum(eig_min.abs(), eig_mid.abs()), eig_max.abs()
    )
    return scale * eig_min, scale * max_abs


_compiled_hermitian_eigen_extrema_3x3 = torch.compile(
    hermitian_eigen_extrema_3x3, fullgraph=True
)


def validate_nonzero_fft_symbol_vectorized(
    symbol: torch.Tensor, real_dtype: torch.dtype
) -> None:
    """Reject singular FFT symbols away from each sample's rigid translation mode."""
    Bsz, _p, _q, Nz, Ny, Nx = symbol.shape
    frequency_count = Nz * Ny * Nx
    sym_flat = symbol.permute(0, 3, 4, 5, 1, 2).reshape(-1, 3, 3)
    zero_idxs = torch.arange(Bsz, device=symbol.device) * frequency_count
    candidate_chunks = []
    ambiguous_chunks = []
    eps = torch.finfo(real_dtype).eps

    for start in range(0, sym_flat.shape[0], _VALIDATION_CHUNK_SIZE):
        end = min(start + _VALIDATION_CHUNK_SIZE, sym_flat.shape[0])
        minimum, max_abs = _compiled_hermitian_eigen_extrema_3x3(sym_flat[start:end])
        threshold = 100.0 * eps * max_abs
        ambiguity = _THRESHOLD_AMBIGUITY_ULPS * eps * max_abs
        nonzero = torch.ones(end - start, dtype=torch.bool, device=symbol.device)
        rigid = zero_idxs[(zero_idxs >= start) & (zero_idxs < end)] - start
        nonzero[rigid] = False
        ambiguous = nonzero & ((minimum - threshold).abs() <= ambiguity)
        clearly_failed = nonzero & (minimum < threshold - ambiguity)
        candidate_chunks.append(clearly_failed | ambiguous)
        ambiguous_chunks.append(ambiguous)

    candidates = torch.nonzero(torch.cat(candidate_chunks), as_tuple=False).flatten()
    if candidates.numel() == 0:
        return

    ambiguous = torch.cat(ambiguous_chunks)
    candidate_is_ambiguous = ambiguous[candidates]
    candidate_failures = ~candidate_is_ambiguous
    ambiguous_indices = candidates[candidate_is_ambiguous]
    if ambiguous_indices.numel() > 0:
        oracle_failures = torch.zeros_like(ambiguous)
        legacy_chunks = torch.unique(
            torch.div(
                ambiguous_indices,
                _LEGACY_VALIDATION_CHUNK_SIZE,
                rounding_mode="floor",
            )
        ).tolist()
        for chunk in legacy_chunks:
            start = chunk * _LEGACY_VALIDATION_CHUNK_SIZE
            end = min(start + _LEGACY_VALIDATION_CHUNK_SIZE, sym_flat.shape[0])
            blocks = sym_flat[start:end]
            hermitian = 0.5 * (blocks + blocks.mH)
            eigenvalues = torch.linalg.eigvalsh(hermitian)
            oracle_threshold = 100.0 * eps * eigenvalues.abs().amax(dim=-1)
            oracle_failures[start:end] = eigenvalues[:, 0] <= oracle_threshold
        candidate_failures[candidate_is_ambiguous] = oracle_failures[
            ambiguous_indices
        ]

    while bool(candidate_failures.any().item()):
        first_position = torch.argmax(candidate_failures.to(torch.int8))
        flat_index = int(candidates[first_position].item())
        start = (flat_index // _LEGACY_VALIDATION_CHUNK_SIZE) * (
            _LEGACY_VALIDATION_CHUNK_SIZE
        )
        end = min(start + _LEGACY_VALIDATION_CHUNK_SIZE, sym_flat.shape[0])
        blocks = sym_flat[start:end]
        hermitian = 0.5 * (blocks + blocks.mH)
        eigenvalues = torch.linalg.eigvalsh(hermitian)
        local_index = flat_index - start
        min_eigenvalue = eigenvalues[local_index, 0]
        oracle_threshold = 100.0 * eps * eigenvalues[local_index].abs().amax()
        if bool((min_eigenvalue <= oracle_threshold).item()):
            batch, frequency = divmod(flat_index, frequency_count)
            kz_index, frequency = divmod(frequency, Ny * Nx)
            ky_index, kx_index = divmod(frequency, Nx)
            raise RuntimeError(
                "FFT reference has a nonzero wavevector singular mode: "
                f"batch={batch}, k=({kz_index},{ky_index},{kx_index}), "
                f"min_eigenvalue={float(min_eigenvalue.item()):.6e}"
            )
        candidate_failures[first_position] = False
