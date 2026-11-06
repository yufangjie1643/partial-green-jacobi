# Environment

The archived measurements and publication-checkout smoke test use Linux x86-64, Python 3.12.13, an NVIDIA GeForce RTX 5090 (32 GB), PyTorch 2.12.0+cu130, CUDA toolkit 13.1, NumPy 2.4.5, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.10.9 and Ninja 1.13.0. Both the physical operator and PCG/FFT use FP64.

The requirements files pin the Python packages. CUDA drivers, toolkit and host compiler must be installed separately. CUDA extensions compile into PyTorch's normal cache. `MAX_JOBS` defaults to 2. The export removes the original forced `TORCH_CUDA_ARCH_LIST=12.0`; the installed GPU is now detected by PyTorch. This is a build portability change, not a numerical change. Other architectures have not been benchmarked for this release.

The optional native comparison uses CuPy 14.2.0 and these upstream revisions:

| Project | Commit | License |
|---|---|---|
| muTopOpt | `7290b28fedf6ec9510993d68ddb0692f5720fc3c` | MIT |
| muGrid | `2faa75b2590d327fbdd846c21cfec959ed381cf5` | LGPL-3.0 |

muGrid was built in Release mode with CUDA enabled and MPI/NetCDF disabled. The original experiment preloaded the installed CUDA 13.1 runtime and cuFFT so native and internal solvers used the same libraries. Details and binary hashes are in `exp/asg_mutopopt_20261104/results/runtime_environment.json`; path prefixes in archived metadata are anonymized.

That build needed a cuFFT enum-version guard patch (provided under the upstream license) and a local compatibility overlay for two CUDA `rsqrt` declarations. The overlay must be generated from the user's installed toolkit; NVIDIA headers are not redistributed. See [native solver setup](NATIVE_SOLVER.md).

Timing scripts use an exclusive-GPU guard. If another compute process is present, the run is rejected rather than treated as a valid performance measurement. Compilation and untimed warm-up are separated from formal measurements; the small example does not attempt a benchmark-quality timing protocol.
