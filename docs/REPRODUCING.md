# Reproducing the research

Use the `v0.1.0` checkout. The package preserves dated experiment directories to keep source provenance explicit. `examples/` provides portable entry points; `exp/` contains the original research drivers and their registrations. Historical helper modules in `solver/experiments/` are retained because adapters load selected definitions from them; they are not all standalone supported command-line programs.

## 1. Convergence scans, then calibration

```bash
python scripts/reproduce_tables.py --output outputs/reconstructed_tables.json
```

The script reads the archived 17-point alpha grid on 172 development states, applies the two-success eligibility rule and the registered case-weighted loss, and refits `[9/16, 11/16, 13/16]`. The independent replay data are opened only after this fitting stage. This reconstructs the table from measurements instead of treating its entries as unexplained constants.

`exp/asg_alpha_landscape_20261105/study.py` contains collection and scan routines; `exp/asg_density_selector_20261105/fit.py` contains the frozen fitting procedure. The latter's original `main()` guards the original registration paths; call the portable reconstruction script for verification. It does not overwrite the freeze. Broader exploratory scans and grouped cross-validation records are included in the data assets.

## 2. Fresh design and serial/parallel verification

```bash
python examples/shear_design.py --n 16 --steps 10 --methods green gj 0.5 --load-modes serial parallel --output outputs/small
```

All routes use identical initial density, material law, objective, update rule and residual tolerance. This small example verifies functionality; it is not a new timing claim. The three-load solver requires the Study I solid tensor and fixed outer configuration. Different objective/contrast cases use the separate Study II adapter.

For the registered 180-evaluation Study I comparison, first fetch workflow data (the archived warm-up states are required):

```bash
python scripts/fetch_data.py workflow-data
python exp/asg_gpu_resident_20261103/run.py --n 32 --implementations baseline resident --load-mode serial --output-dir outputs/study1-serial
python exp/asg_gpu_resident_20261103/run.py --n 32 --implementations resident --load-mode parallel --output-dir outputs/study1-parallel
```

Each call uses three repeats, two seeds and six methods unless overridden. `baseline` denotes the original outer implementation. The resident driver includes graph capture in its raw `total_s`; the archived paper summary handles common capture costs according to its declared timing boundary. Compare the same implementation and boundary when reproducing a claimed percentage. The 180-evaluation budget does not certify optimization convergence.

## 3. Independent material-state replay

```bash
python scripts/fetch_data.py independent-test-data
python examples/replay_state.py exp/asg_density_selector_20261105/states/S20_e4_A_k001.npz --eta 1e-4 --methods gj eta compact8 --output outputs/replay.json
```

All methods start from the identical archived displacement. The example warms each method, measures one replay, and computes its effective tensor and true-residual certificate. For a serious timing comparison use multiple repetitions and randomized order, as in `exp/asg_density_selector_20261105/run.py`. The reported independent study uses 18 settings, five states per setting and three technical repeats, covering shear, volumetric and an auxetic-surrogate objective.

The archived driver resumes completed output files. To generate a fresh full replay in a separate checkout, preserve downloaded records elsewhere, leave the new `replay_parallel/` directory empty, and invoke `python exp/asg_density_selector_20261105/run.py replay`; its warm-up additionally requires `development-data`. Never replace archived measurements with fresh timings when reconstructing the manuscript tables.

## 4. Native solver ablation

See [NATIVE_SOLVER.md](NATIVE_SOLVER.md). The adapter changes only native two-sided GJ scaling, preserving native muGrid PCG. This experiment uses frozen material states and warm starts, not a comparison of complete muTopOpt outer optimizers.

## Evidence map

| Claim or result | Source tables / entry point |
|---|---|
| Iteration/time dependence on alpha | `data/development/{states,costs}.csv`; `exp/asg_alpha_landscape_20261105/study.py` |
| Three-entry calibration | `exp/asg_density_selector_20261105/fit.py`; `MODELS_FROZEN.json`; CPU reconstruction |
| Independent analysis time and regressions | `data/study2/replay_parallel_attempts.csv`; per-case mean of paired ratios |
| Complete Study I workflows | `data/study1/implementation_summary.csv`; `exp/asg_gpu_resident_20261103/run.py` |
| Larger-grid workflows and memory | `data/study1/full_timing.csv`, `memory_summary.csv`; scaling and resident-memory drivers |
| Native scaling ablation | `data/study1/mutopopt_summary.csv`; native benchmark and its protocol |

`provenance/source_export.json` maps archived sources to their publication copies. Export changes concern build settings, path portability and new entry points; numerical CUDA kernels and the calibration cost function are preserved. The final source manifest hashes the released contents. Verification records in `validation/` distinguish archived measurements from publication-checkout checks.
