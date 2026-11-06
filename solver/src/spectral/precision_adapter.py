"""Opt-in FP32/FP64 pairing of the same FI native FFT and fused PCG path.

Defaults and source of the production solver remain unchanged. The two CUDA
files mirror the production kernels, with only scalar/complex/cuFFT types
selected by FEAD_FFT_DOUBLE. Physical operators and PCG reductions stay FP64.
"""
from functools import lru_cache
import inspect
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


@lru_cache(None)
def extension(kind, precision):
    from torch.utils.cpp_extension import load
    double = precision == "fp64"
    return load(
        name=f"voxel_fft_precision_{kind}_{precision}_20261024",
        sources=[str(HERE / ("precision_fft.cu" if kind == "native" else "precision_kernels.cu"))],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"-DFEAD_FFT_DOUBLE={int(double)}"],
        extra_ldflags=["-lcufft"] if kind == "native" else [],
        verbose=False,
    )


def install(cs, precision):
    if precision not in ("fp32", "fp64"):
        raise ValueError(precision)
    dtype = torch.float64 if precision == "fp64" else torch.float32
    chain, formula = cs.a, cs.s.f.formula
    native, contest = chain.native, chain.contest
    original_reference = formula.enable_fi_reference

    def reference(op, *args, **kwargs):
        # Rebuild the reference in the requested precision from the same FP64
        # material, rather than promoting already rounded FP32 coefficients.
        return original_reference(op, precision="fp64" if precision == "fp64" else "mixed")

    formula.enable_fi_reference = reference
    contest.extension = lambda: extension("kernels", precision)
    native.extension = lambda: extension("native", precision)
    source = inspect.getsource(contest.attach)
    old = "op._fft_symbol_inv.dtype != torch.float32"
    assert source.count(old) == 1
    source = source.replace(old, "op._fft_symbol_inv.dtype != _fft_dtype")
    context = dict(vars(contest), _fft_dtype=dtype)
    exec(compile(source, "<precision-contest-attach>", "exec"), context)
    paired_attach = context["attach"]

    def attach(op, *args, **kwargs):
        result = paired_attach(op, *args, **kwargs)
        result._contest_metadata["FFT_precision"] = (
            "FP64/complex128" if precision == "fp64" else "FP32/complex64")
        result._contest_metadata["precision_pair"] = "20261024"
        return result

    contest.attach = attach
    # Keep the production recurrence and fusion graph; change its output dtype.
    source = inspect.getsource(chain._pre_core)
    assert source.count("to(torch.float32)") == 1
    source = source.replace("to(torch.float32)", "to(torch.float64)" if precision == "fp64" else "to(torch.float32)")
    context = dict(vars(chain))
    exec(compile(source, "<precision-pcg-boundary>", "exec"), context)
    chain._pre_core = context["_pre_core"]
    chain.compiled.cache_clear()
    return dict(FFT=str(dtype), physics="torch.float64", PCG="torch.float64")
