# muTopOpt / muGrid matched periodic-cell replay

Frozen on 2026-11-04 before formal timing. This adds an external implementation
baseline; it does not change archived optimization trajectories or solver/.

- muTopOpt commit 7290b28fedf6ec9510993d68ddb0692f5720fc3c; muGrid commit
  2faa75b2590d327fbdd846c21cfec959ed381cf5. CUDA 13.1, sm120, Release, FP64,
  single GPU, MPI/NetCDF disabled for this single-rank benchmark. Record build
  logs, binary hashes and any compatibility-only patches.
- Native muTopOpt Homogenization uses Q1 Hex8, eight-point Gauss integration,
  unit periodic domain, lambda=4/3, mu=1, SIMP p=3, eta=1e-4. Spatial storage
  is transposed between our (z,y,x,component) and muGrid (component,x,y,z).
  Verify stiffness actions, three engineering-shear right-hand sides,
  translations and homogenized responses before any performance claim.
- Saved grids 32 and 64; evaluation indices 0,49,62,179; three repeats per
  state, seeded randomized method order. Inputs and previous displacements
  are identical. One untimed warm-up of every route precedes the timed block.
- External rows: muTopOpt-Green, muTopOpt-GJ, and an explicitly labeled
  muTopOpt-partial-0.5 adapter ablation (not an upstream method). The ablation
  changes only the two-sided scale after the native GJ diagonal update. All
  use native muGrid PCG and native kernels, serial loads. Internal rows:
  Green/GJ/frozen-ASG/fixed-0.5 with existing serial and parallel schedules.
- Native solve_rhs(..., warm_start=True) is used because solve_macro resets
  the initial guess. Assemble the RHS with the native macro-load operator.
  Check every returned displacement using the archived general operator:
  norm(b-Ku) <= max(1e-8 norm(b),1e-13/sqrt(n)). Set inner tolerance to 0.95
  of this target; permit at most two restarts and 4000 total iterations.
  Record all failures, exclude them from speedup claims, and do not interpret
  a native best-iterate return as convergence without this certificate.
- Reuse the uniform-reference preconditioner between states. Time native
  density/material/diagonal updates, RHS and layout preparation, native solves,
  independent certification, and common energy response. File loading and
  initial host-to-device input copies are excluded for all methods. Record
  one-time construction separately. Full-field bridge copies stay on GPU;
  the upstream GJ diagonal's host transfer remains part of native update time.
- The adapter partial-0.5 route still pays the upstream diagonal update and
  its additional exponent transform. It is a controlled method ablation,
  not a claim of an optimized upstream implementation. No ASG retraining.
- Timings run under the project's exclusive-GPU Guard. Full muTopOpt outer
  optimization (different target, phase-field regularization and adjoints)
  is NOT measured. Old AmgX timings remain a separate dated batch.
