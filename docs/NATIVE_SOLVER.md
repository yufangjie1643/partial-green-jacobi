# Optional native muTopOpt/muGrid comparison

This comparison preserves upstream PCG and operator kernels. Our `Native.update()` adapter takes the square root of the native `D^-1/2` scale to obtain the two-sided partial exponent 0.5. Results are frozen-state analysis times, not muTopOpt outer-optimization times.

The external projects are not vendored. Fetch their pinned revisions:

```bash
python scripts/prepare_native.py
python scripts/fetch_data.py workflow-data
```

Use a CUDA-enabled muGrid build. The following settings reproduce the archived single-GPU build configuration; change the architecture for a different GPU:

```bash
python -m pip install cupy-cuda13x==14.2.0
python -m pip install ./exp/asg_mutopopt_20261104/vendor/muGrid \
  --config-settings=cmake.define.MUGRID_ENABLE_CUDA=ON \
  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=120 \
  --config-settings=cmake.define.MUGRID_ENABLE_MPI=OFF \
  --config-settings=cmake.define.MUGRID_ENABLE_NETCDF=OFF
python -m pip install ./exp/asg_mutopopt_20261104/vendor/muTopOpt
```

The preparation script applies the recorded cuFFT-version guard patch. If the CUDA 13.1/system compiler combination rejects the `rsqrt` exception specifications, run `python scripts/prepare_native.py --cuda-overlay /usr/local/cuda` and additionally pass `--config-settings=cmake.define.CMAKE_CUDA_FLAGS=-I<absolute-checkout>/exp/asg_mutopopt_20261104/compat` to the muGrid build. This creates a local overlay from the installed toolkit. No toolkit headers are downloaded from this repository or included in its release.

For the exact archived environment, preload the same CUDA runtime and cuFFT pair for both implementations:

```bash
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so.13:/usr/local/cuda/lib64/libcufft.so.12 \
  python exp/asg_mutopopt_20261104/benchmark.py validate
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so.13:/usr/local/cuda/lib64/libcufft.so.12 \
  python exp/asg_mutopopt_20261104/benchmark.py replay --n 32
```

`validate` checks the native stiffness action, right-hand sides, translations and effective response. Checkpoints 0, 49, 62 and 179 use identical previous displacements and independent residual certification. New outputs go to `outputs/native/`; use `--output-dir` for a different run. The benchmark resumes existing `replay_*.json` outputs in that directory. The portable internal-solver examples do not require these native dependencies.

The archive includes the original native validation and environment records. Publication-checkout GPU smoke checks exercise the internal solver; the external upstream projects are optional builds and their source licenses remain applicable.
