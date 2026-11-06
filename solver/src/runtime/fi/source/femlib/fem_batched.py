# -*- coding: utf-8 -*-
"""Top-level wrapper: FEMHomogenizerBatched

Homogenises B samples (sharing a mesh) in a single call. Requires CUDA + Triton.
"""

from __future__ import annotations

import torch

from .batched import (
    BatchedC3D8RHourglassTwoStageOperator,
    BatchedStencilOperator,
    compute_effective_stiffness_batched,
    compute_effective_stiffness_ir_adaptive_batched,
    compute_effective_stiffness_ir_batched,
)
from .batched.mesh_info import MeshInfo
from .c3d8r import validate_hourglass_coefficient
from .element_types import normalize_element_type


class FEMHomogenizerBatched:
    """
    Parameters
    ----------
    C_voigt_fields : canonical contiguous (B, Nz*Ny*Nx, 6, 6) FP64 torch
                     field, or legacy (B, 6, 6, Nz, Ny, Nx) input
    voxel_size     : (dx, dy, dz)
    grid_shape     : (Nz, Ny, Nx)
    dtype          : torch.float32 (recommended, faster) or torch.float64
    """

    def __init__(self, C_voigt_fields, voxel_size, grid_shape,
                 device=None, dtype=torch.float32, verbose=False,
                 element_type="c3d8", hourglass_coefficient=0.1):
        if device is None:
            device = torch.device('cuda')
        else:
            device = torch.device(device)
        if device.type != 'cuda':
            raise RuntimeError("FEMHomogenizerBatched requires a CUDA device")

        self.mesh = MeshInfo.from_args(voxel_size, grid_shape)
        self.verbose = verbose
        element_key = str(element_type).strip().lower()
        self.element_type = (
            "c3d8r" if element_key == "c3d8r"
            else normalize_element_type(element_type)
        )
        # This is a reference only, not a copy. It lets the mixed-precision
        # path pass the canonical element-major owner directly to its
        # explicitly required FP32 inner operator.
        self._canonical_material_field = (
            C_voigt_fields
            if isinstance(C_voigt_fields, torch.Tensor)
            and C_voigt_fields.ndim == 4
            else None
        )
        if self.element_type == "c3d8r":
            self.hourglass_coefficient = validate_hourglass_coefficient(
                hourglass_coefficient,
            )
            self.op = BatchedC3D8RHourglassTwoStageOperator(
                self.mesh, C_voigt_fields,
                device=device, dtype=dtype, verbose=verbose,
                hourglass_coefficient=self.hourglass_coefficient,
            )
        else:
            self.op = BatchedStencilOperator(
                self.mesh, C_voigt_fields,
                device=device, dtype=dtype, verbose=verbose,
                element_type=self.element_type,
            )
        self._fft_args = None
        self._apply_backend_args = None
        self._ir_inner_apply_backend_args = None
        self._reference_media = {}

    @property
    def B(self) -> int:
        return self.op.B

    def enable_fft_precond(
        self,
        ref_strategy='zeman',
        C0_voigt=None,
        verbose=None,
        *,
        spectrum_mode='complex_full',
        validation_mode='legacy_eig',
        allow_symbol_sharing=True,
    ):
        """Enable the FE Green's-function FFT preconditioner.

        This inverts the Fourier symbol of the homogeneous reference *FE*
        stencil. ``enable_fe_green_precond`` is an explicit alias with the
        same implementation. For C3D8R, the public options also expose the
        half-spectrum cache, symbol validation mode, and exact batch sharing.
        """
        if verbose is None:
            verbose = self.verbose
        C0_voigt = self._resolve_fft_reference(ref_strategy, C0_voigt)
        fft_args = {
            'ref_strategy': ref_strategy,
            'C0_voigt': C0_voigt,
        }
        if self.element_type == 'c3d8r':
            fft_args.update({
                'spectrum_mode': spectrum_mode,
                'validation_mode': validation_mode,
                'allow_symbol_sharing': allow_symbol_sharing,
            })
        elif (
            spectrum_mode != 'complex_full'
            or validation_mode != 'legacy_eig'
            or not bool(allow_symbol_sharing)
        ):
            raise ValueError(
                "spectrum_mode, validation_mode, and allow_symbol_sharing "
                "are currently configurable only for C3D8R"
            )
        self._enable_requested_fft(self.op, fft_args, verbose=verbose)
        self._fft_args = dict(self.op._fft_args)
        self._sync_existing_fp32_fft(verbose=False)

    def defer_fft_precond_for_ir(self, ref_strategy='zeman', C0_voigt=None):
        """Record FFT settings for the fp32 IR inner operator.

        Mixed-precision IR uses the fp64 outer operator for residual and stress
        evaluations; preconditioning happens in the fp32 inner PCG.  This
        avoids building an unused fp64 FFT symbol while still propagating the
        requested reference medium to ``_ensure_op_fp32``.
        """
        C0_voigt = self._resolve_fft_reference(ref_strategy, C0_voigt)
        self._fft_args = {'ref_strategy': ref_strategy, 'C0_voigt': C0_voigt}
        self.op.disable_fft_precond()
        self.op._fft_args = None
        self._sync_existing_fp32_fft(verbose=False)

    def enable_fe_green_precond(
        self,
        ref_strategy='zeman',
        C0_voigt=None,
        verbose=None,
        *,
        spectrum_mode='complex_full',
        validation_mode='legacy_eig',
        allow_symbol_sharing=True,
    ):
        """Explicit alias for the Ladecky-style FE Green FFT preconditioner.

        The batched operator already constructs the inverse Fourier symbol of
        the homogeneous reference FE stencil. This wrapper keeps that
        distinction visible at the public API level; see
        ``BatchedStencilOperator.enable_fe_green_precond`` for details.
        """
        self.enable_fft_precond(
            ref_strategy=ref_strategy,
            C0_voigt=C0_voigt,
            verbose=verbose,
            spectrum_mode=spectrum_mode,
            validation_mode=validation_mode,
            allow_symbol_sharing=allow_symbol_sharing,
        )

    def enable_fft_chebyshev_precond(self, degree=2, lmin=0.2, lmax=1.8,
                                     ref_strategy='zeman', C0_voigt=None,
                                     verbose=None):
        """Enable a fixed-degree Chebyshev wrapper around the FFT preconditioner.

        ``lmin``/``lmax`` bound the spectrum of the FFT-preconditioned operator.
        Higher degree may reduce outer PCG iterations, but each extra degree
        adds one stencil apply and one FFT preconditioner call per PCG step.
        """
        if verbose is None:
            verbose = self.verbose
        C0_voigt = self._resolve_fft_reference(ref_strategy, C0_voigt)
        self._enable_requested_fft(
            self.op,
            {
                'ref_strategy': ref_strategy,
                'C0_voigt': C0_voigt,
                'cheb_degree': degree,
                'cheb_lmin': lmin,
                'cheb_lmax': lmax,
            },
            verbose=verbose,
        )
        self._fft_args = dict(self.op._fft_args)
        self._sync_existing_fp32_fft(verbose=False)

    def disable_fft_precond(self):
        self._fft_args = None
        self.op.disable_fft_precond()
        op_fp32 = getattr(self, '_op_fp32', None)
        if op_fp32 is not None and op_fp32 is not self.op:
            op_fp32.disable_fft_precond()

    def set_fft_batch_chunk_size(self, batch_chunk_size=None):
        """Set an explicit sample-batch chunk for the outer FFT preconditioner.

        ``None`` restores the full-batch transform. This is a hardware- and
        dtype-dependent execution hint, not a change to the linear operator.
        """
        self.op.set_fft_batch_chunk_size(batch_chunk_size)

    def set_apply_backend(
        self,
        backend,
        *,
        cuda_variant=None,
        cuda_block_size=None,
    ):
        """Select and synchronize the outer/inner operator backend.

        The default remains ``"triton"``.  Selecting ``"cuda"`` also applies
        to a lazily-created fp32 iterative-refinement operator.  The
        ``"atomic"`` CUDA variant is never selected implicitly because its
        accumulation order is not bitwise deterministic.
        """
        if not hasattr(self.op, "set_apply_backend"):
            raise NotImplementedError(
                f"the {self.element_type.upper()} operator does not expose "
                "a selectable apply backend"
            )
        self.op.set_apply_backend(
            backend,
            cuda_variant=cuda_variant,
            cuda_block_size=cuda_block_size,
        )
        variant, block_size = self.op.cuda_apply_configuration
        self._apply_backend_args = {
            "backend": self.op.apply_backend,
            "cuda_variant": variant,
            "cuda_block_size": block_size,
        }
        self._sync_existing_fp32_backend()
        return self

    def set_ir_inner_apply_backend(
        self,
        backend,
        *,
        cuda_variant=None,
        cuda_block_size=None,
    ):
        """Override the fp32 IR operator backend independently of the outer.

        Pass ``None`` as ``backend`` to make the inner operator inherit the
        live outer configuration again.  This separate override allows a
        deterministic FP64 ``thread`` outer operator and an explicitly
        requested FP32 ``atomic`` inner operator.
        """
        if self.op.dtype != torch.float64:
            raise RuntimeError(
                "an independent IR inner backend requires a float64 outer operator"
            )
        if backend is None:
            self._ir_inner_apply_backend_args = None
            self._sync_existing_fp32_backend()
            return self
        if backend not in {"triton", "cuda"}:
            raise ValueError(f"unknown IR inner apply backend {backend!r}")
        if backend == "cuda" and self.element_type not in {"c3d8", "c3d8r"}:
            raise NotImplementedError(
                "native CUDA IR is currently implemented only for C3D8 and C3D8R"
            )
        if cuda_variant is not None and cuda_variant not in {"thread", "atomic"}:
            raise ValueError(f"unknown CUDA kernel variant {cuda_variant!r}")
        if cuda_block_size is not None and cuda_block_size not in {64, 128, 256, 512}:
            raise ValueError(
                "cuda_block_size must be one of 64, 128, 256, or 512"
            )
        self._ir_inner_apply_backend_args = {
            "backend": backend,
            "cuda_variant": cuda_variant,
            "cuda_block_size": cuda_block_size,
        }
        self._sync_existing_fp32_backend()
        return self

    def _resolve_fft_reference(self, ref_strategy, C0_voigt):
        if C0_voigt is not None or self.op._C_voigt_gpu is not None:
            return C0_voigt
        try:
            return self._reference_media[ref_strategy]
        except KeyError as exc:
            raise ValueError(f"unknown ref_strategy {ref_strategy!r}") from exc

    def _cache_reference_media(self):
        if self.element_type == "c3d8r" or self._reference_media:
            return
        self._reference_media = {
            strategy: self.op._compute_reference_medium(strategy, verbose=False)
            for strategy in ('mean_eig', 'zeman', 'scalar', 'mean_C')
        }

    def _sync_existing_fp32_fft(self, *, verbose=False):
        op_fp32 = getattr(self, '_op_fp32', None)
        if op_fp32 is None or op_fp32 is self.op:
            return
        if self._fft_args is None:
            op_fp32.disable_fft_precond()
            return
        fft_args = dict(self._fft_args)
        fft_args['C0_voigt'] = self._resolve_fft_reference(
            fft_args['ref_strategy'], fft_args.get('C0_voigt'),
        )
        self._enable_requested_fft(op_fp32, fft_args, verbose=verbose)
        self._fft_args = dict(op_fp32._fft_args)

    def _sync_existing_fp32_backend(self):
        op_fp32 = getattr(self, '_op_fp32', None)
        if op_fp32 is None or op_fp32 is self.op:
            return
        backend_args = (
            self._ir_inner_apply_backend_args
            if self._ir_inner_apply_backend_args is not None
            else self._current_apply_backend_args()
        )
        if backend_args is not None:
            op_fp32.set_apply_backend(**backend_args)

    def _current_apply_backend_args(self):
        """Return the live outer backend configuration, when supported."""
        if hasattr(self.op, "apply_backend"):
            variant, block_size = self.op.cuda_apply_configuration
            return {
                "backend": self.op.apply_backend,
                "cuda_variant": variant,
                "cuda_block_size": block_size,
            }
        return self._apply_backend_args

    @staticmethod
    def _enable_requested_fft(op, fft_args, *, verbose=False):
        fft_args = dict(fft_args)
        cheb_degree = fft_args.pop('cheb_degree', None)
        cheb_lmin = fft_args.pop('cheb_lmin', None)
        cheb_lmax = fft_args.pop('cheb_lmax', None)
        if cheb_degree is None:
            op.enable_fft_precond(**fft_args, verbose=verbose)
        else:
            enable_chebyshev = op.enable_fft_chebyshev_precond
            cheb_degree = int(cheb_degree)
            cheb_lmin = float(cheb_lmin)
            cheb_lmax = float(cheb_lmax)
            if cheb_degree < 1:
                raise ValueError("Chebyshev preconditioner degree must be >= 1")
            if not (0.0 < cheb_lmin < cheb_lmax):
                raise ValueError(
                    f"expected 0 < lmin < lmax, got {cheb_lmin}, {cheb_lmax}"
                )
            op.enable_fft_precond(**fft_args, verbose=verbose)
            fft_args['C0_voigt'] = op._fft_args['C0_voigt']
            enable_chebyshev(
                degree=cheb_degree, lmin=cheb_lmin, lmax=cheb_lmax,
                **fft_args, verbose=verbose,
            )

    def compute_effective_stiffness(
        self,
        *,
        tol=1e-6,
        max_iter=2000,
        pcg_update_mode="eager",
        convergence_check_stride=1,
        convergence_check_adaptive_factor=None,
        verify_final_residual=False,
    ):
        """Returns (C_eff_batch: (B,6,6), info_per_sample: list[list[dict]])."""
        return compute_effective_stiffness_batched(
            self.op, tol=tol, max_iter=max_iter, verbose=self.verbose,
            pcg_update_mode=pcg_update_mode,
            convergence_check_stride=convergence_check_stride,
            convergence_check_adaptive_factor=convergence_check_adaptive_factor,
            verify_final_residual=verify_final_residual,
        )

    # --------------------------------------------------------
    # Mixed-precision IR: fp64 outer + fp32 inner PCG
    # --------------------------------------------------------
    def _ensure_op_fp32(self):
        """Lazily build the fp32 inner operator. Shares the same C field, differing only in dtype."""
        if getattr(self, '_op_fp32', None) is not None:
            return
        if self.op.dtype == torch.float32:
            # Outer is already fp32; inner and outer share the same operator
            self._op_fp32 = self.op
            return
        device = self.op.device
        C_voigt_fp64 = self.op._C_voigt_gpu          # stride-only Voigt view
        C_element_fp64 = self.op.C                # canonical/packed owner when FI
        voxel_size = (self.mesh.dx, self.mesh.dy, self.mesh.dz)
        grid_shape = (self.mesh.Nz, self.mesh.Ny, self.mesh.Nx)
        if self.element_type == "c3d8r":
            self._op_fp32 = BatchedC3D8RHourglassTwoStageOperator(
                MeshInfo.from_args(voxel_size, grid_shape),
                C_voigt_fp64,
                device=device, dtype=torch.float32,
                hourglass_coefficient=self.hourglass_coefficient,
                verbose=False,
            )
        else:
            # The stencil operator does not need outer C while assembling its
            # fp32 counterpart. For canonical and packed owners, keep the one
            # FP64 tensor live because C_element_fp64 is the same storage
            # passed to the inner conversion; for legacy callers retain the
            # historical release/rebuild adapter.
            material_layout = self.op._material_layout
            if material_layout not in {"element_major", "packed_element_major"}:
                self.op.C = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._op_fp32 = BatchedStencilOperator(
                MeshInfo.from_args(voxel_size, grid_shape),
                C_element_fp64
                if self.op._material_layout in {"element_major", "packed_element_major"}
                else C_voigt_fp64,
                device=device, dtype=torch.float32, verbose=False,
                element_type=self.element_type,
            )
        self._sync_existing_fp32_backend()
        # Propagate the wrapper's requested preconditioner to the inner operator.
        if self._fft_args is not None:
            self._enable_requested_fft(
                self._op_fp32, self._fft_args, verbose=False,
            )
            self._fft_args = dict(self._op_fp32._fft_args)
        if self.element_type != "c3d8r":
            self._cache_reference_media()
            # Stencil applies no longer need constitutive layouts after setup.
            self._op_fp32.C = None
            self._op_fp32._C_voigt_gpu = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.op._material_layout == "element_major":
                # The FP64 canonical owner never left the outer operator;
                # restore only its stride-only Voigt adapter after the inner
                # operator has released its construction-time material.
                self.op.C = C_element_fp64
                self.op._C_voigt_gpu = self.op.C.view(
                    self.op.B, self.mesh.Nz, self.mesh.Ny, self.mesh.Nx, 6, 6,
                ).permute(0, 4, 5, 1, 2, 3)
            elif self.op._material_layout == "packed_element_major":
                # Packed FI keeps the sole 21-entry owner; no full tensor
                # adapter may be reconstructed just for the inner backend.
                self.op.C = C_element_fp64
                self.op._C_voigt_gpu = self.op.C
            else:
                # Legacy callers retain the guarded historical adapter.
                self.op.C = C_voigt_fp64.permute(0, 3, 4, 5, 1, 2).reshape(
                    self.op.B, self.mesh.n_elem, 6, 6,
                ).contiguous()
                self.op._C_voigt_gpu = None

    def compute_effective_stiffness_ir(
        self, *,
        outer_tol=1e-8,
        inner_tol=0.1,
        max_outer=30,
        max_inner=80,
        inner_pcg_update_mode="eager",
        inner_convergence_check_stride=1,
        inner_verify_final_residual=False,
    ):
        """Mixed-precision IR solve over the 6 load cases.

        Requires self to be built with fp64 (outer). The first call lazily builds the fp32 inner operator.

        Returns
        -------
        C_eff_batch     : np.ndarray (B, 6, 6)
        info_per_sample : list[list[dict]]
        """
        if self.op.dtype != torch.float64:
            raise RuntimeError(
                "compute_effective_stiffness_ir requires FEMHomogenizerBatched "
                "to be initialised with dtype=torch.float64 (high-precision outer)."
            )
        self._ensure_op_fp32()
        return compute_effective_stiffness_ir_batched(
            self.op, self._op_fp32,
            outer_tol=outer_tol, inner_tol=inner_tol,
            max_outer=max_outer, max_inner=max_inner,
            inner_pcg_update_mode=inner_pcg_update_mode,
            inner_convergence_check_stride=inner_convergence_check_stride,
            inner_verify_final_residual=inner_verify_final_residual,
            verbose=self.verbose,
        )

    def compute_effective_stiffness_ir_adaptive(
        self, *,
        outer_tol=1e-8,
        inner_tols=(0.1, 0.03, 0.01),
        max_outer=30,
        max_inner=80,
        adaptive_max_inner=160,
        stagnation_factor=0.3,
        stall_factor=0.9,
        stall_patience=3,
        inner_pcg_update_mode="eager",
        inner_convergence_check_stride=1,
        inner_verify_final_residual=False,
    ):
        """Adaptive mixed-precision IR over the 6 load cases.

        The outer residual/update are fp64.  The inner fp32 PCG correction
        starts at ``inner_tols[0]`` and tightens only when the previous outer
        correction did not reduce the residual by ``stagnation_factor``.
        """
        if self.op.dtype != torch.float64:
            raise RuntimeError(
                "compute_effective_stiffness_ir_adaptive requires "
                "FEMHomogenizerBatched to use dtype=torch.float64."
            )
        self._ensure_op_fp32()
        return compute_effective_stiffness_ir_adaptive_batched(
            self.op, self._op_fp32,
            outer_tol=outer_tol,
            inner_tols=inner_tols,
            max_outer=max_outer,
            max_inner=max_inner,
            adaptive_max_inner=adaptive_max_inner,
            stagnation_factor=stagnation_factor,
            stall_factor=stall_factor,
            stall_patience=stall_patience,
            inner_pcg_update_mode=inner_pcg_update_mode,
            inner_convergence_check_stride=inner_convergence_check_stride,
            inner_verify_final_residual=inner_verify_final_residual,
            verbose=self.verbose,
        )

    def compute_effective_stiffness_ir_adaptive_fallback(
        self, *,
        outer_tol=1e-8,
        inner_tols=(0.1, 0.03, 0.01),
        max_outer=30,
        max_inner=80,
        adaptive_max_inner=160,
        stagnation_factor=0.3,
        stall_factor=0.9,
        stall_patience=3,
        fallback_max_iter=2000,
        inner_pcg_update_mode="eager",
        inner_convergence_check_stride=1,
        inner_verify_final_residual=False,
        fallback_pcg_update_mode="eager",
    ):
        """Production mixed-precision path with fp64 PCG fallback.

        First tries adaptive IR-PCG.  If any load case misses ``outer_tol`` or
        reports a stall/max-outer stop, the method reruns all load cases with
        the current fp64 operator and preconditioner.  This keeps the fast path
        identical to IR while preserving the reliability of FP64+FFT PCG.
        """
        C_ir, info_ir = self.compute_effective_stiffness_ir_adaptive(
            outer_tol=outer_tol,
            inner_tols=inner_tols,
            max_outer=max_outer,
            max_inner=max_inner,
            adaptive_max_inner=adaptive_max_inner,
            stagnation_factor=stagnation_factor,
            stall_factor=stall_factor,
            stall_patience=stall_patience,
            inner_pcg_update_mode=inner_pcg_update_mode,
            inner_convergence_check_stride=inner_convergence_check_stride,
            inner_verify_final_residual=inner_verify_final_residual,
        )
        needs_fallback = any(
            load.get('rel_res', float('inf')) > outer_tol
            or load.get('stop_reason') != 'converged'
            for sample in info_ir
            for load in sample
        )
        if not needs_fallback:
            for sample in info_ir:
                for load in sample:
                    load['fallback_used'] = False
                    load['primary_method'] = 'ir_adaptive'
            return C_ir, info_ir

        C_pcg, info_pcg = self.compute_effective_stiffness(
            tol=outer_tol, max_iter=fallback_max_iter,
            pcg_update_mode=fallback_pcg_update_mode,
        )
        for b, sample in enumerate(info_pcg):
            for j, load in enumerate(sample):
                ir_load = info_ir[b][j]
                load['fallback_used'] = True
                load['primary_method'] = 'ir_adaptive'
                load['fallback_method'] = 'pcg'
                load['ir_rel_res'] = ir_load.get('rel_res')
                load['ir_outer_iters'] = ir_load.get('outer_iters')
                load['ir_inner_iters'] = ir_load.get('inner_iters')
                load['ir_stop_reason'] = ir_load.get('stop_reason')
                load['ir_final_inner_tol'] = ir_load.get('final_inner_tol')
        return C_pcg, info_pcg
