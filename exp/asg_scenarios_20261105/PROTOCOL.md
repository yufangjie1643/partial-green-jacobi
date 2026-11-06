# Representative periodic-cell application cases

Registered before the new computations, 2026-11-05. The exponent 0.5 and frozen
ASG model are inherited without tuning. All old experiment sources and results
remain unchanged.

## Cases and objective

Engineering-strain order is xx, yy, zz, yz, xz, xy. For load l, Q_l is twice the
mean equilibrium strain energy, q_l = E_l:C_s:E_l, and J=sum_l w_l Q_l/q_l.
All weights are nonnegative and sum to one. Sensitivities use the identical
weighted sum of local solid-material energies. These are prescribed macroscopic
strain cases, not finite-panel force boundary conditions.

| ID | Physical volume limit | Objective / loads | Weights |
|---|---:|---|---|
| S20 | 0.20 | Three unit engineering shears yz, xz, xy | 1/3 each |
| S30 | 0.30 | Three unit engineering shears yz, xz, xy | 1/3 each |
| S45 | 0.45 | Three unit engineering shears yz, xz, xy | 1/3 each |
| B30 | 0.30 | Hydrostatic strain I/3; normalized bulk stiffness K^H/K_s | 1 |
| M30 | 0.30 | Unit zz strain, yz shear, xz shear; normalized weighted stiffness | 1/2, 1/4, 1/4 |

For M30 the axial response is C^H_3333, a constrained-strain stiffness, not the
uniaxial-stress Young modulus. B30 does not impose isotropy. S20/S30/S45 isolate
volume effects; S30/B30/M30 isolate objective/load effects at fixed volume.

## Design and comparisons

- Existing full-integration periodic Q1 mechanics, lambda_s=4/3, mu_s=1,
  eta=1e-4, p=3, sigma/L=0.9/32, OC move=0.1 and 45 bisections.
- beta=1,2,4,8 in 25-evaluation stages, then retain 8. Maximum 500 evaluations.
  Stop after five consecutive final-stage evaluations with max design update
  <0.005 and relative objective change <1e-4, after at least 20 final-stage steps.
  Report actual stop status; this is not a global or KKT optimality certificate.
- New seeds 20261106 and 20261107. Same 12-mode periodic random field for every
  method; adjust its mean to the case's projected physical-volume constraint.
- Full 32^3 comparisons: Green, GJ, fixed 0.5, frozen ASG; two seeds and three
  technical repeats, shuffled within repeat blocks with fixed scheduler seed.
- Three loads execute in parallel. The one-load bulk case uses one PCG workspace.
  Serial and parallel implementations are both validated on three-load cases.
- Additional 64^3 designs: fixed 0.5, seed 20261106, all five cases, one run each.
  They supply resolved geometries and a resolution check, not replicated 64^3
  solver timing comparisons.
- Every solve passes the original generic operator's true residual criterion
  max(1e-8*norm(b),1e-13/sqrt(n)). Saved designs are independently reanalyzed.
- Record setup, filtering, material update, scaling/selection, equilibrium and
  verification, energy, OC and host remainder. Sum numerical step wall times
  plus setup; exclude initial-field generation, compile/warmup, snapshots and I/O.
- Formal timing requires no other CUDA process throughout the run. An externally
  interfered run is rejected, never silently included in timing statistics.

## Verification before formal runs

Homogeneous energies against E:C_s:E; dense constrained small-mesh solves;
directional finite-difference objective/volume gradients; CPU/GPU OC agreement
at all volumes; serial/parallel agreement; legacy S30 equivalence. All checks
must pass before the timing queue starts. New sources and result files are hashed.

## Figure contract

Backend: existing Python workflow. Use actual computed density fields only.
One five-case geometry plate maps material use and load objective to geometry;
one numerical figure compares full workflow time and convergence histories.
Use the same physical-density isovalue 0.5 and camera for all renderings. State
that mechanics uses continuous densities; do not imply binary reanalysis.
Report technical repeat SD separately from differences between initializations.
Select displayed cases before observing performance, retain both seeds in tables.
Export PNG, editable PDF/SVG and CSV source data, 183 mm figure width.
