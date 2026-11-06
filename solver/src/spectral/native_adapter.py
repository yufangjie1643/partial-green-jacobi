"""Own disposable FFT spectrum permits C2R without the framework's clone."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import torch
import contest_adapter as c


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name="fead_fft_contest_native_20261020",sources=[str(Path(__file__).with_name("native_fft.cu"))],
                extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],extra_ldflags=["-lcufft"],verbose=False)


def attach(op,kind,form="cached",release=True):
    if form not in ("cached","dense"):raise ValueError(form)
    c.attach(op,kind,form,release=release)
    grid=op._contest_metadata["grid"];voxel=op._contest_metadata["voxel"]
    coefficient=op._contest_coefficient;tables=op._contest_tables
    plan=extension().NativeFFT(*grid)
    fi=kind=="fi";mode=1 if form=="cached" else 2
    def apply(self,residual):
        field=c.extension().pack(residual,*grid)
        spectrum=plan.forward(field)
        del field
        c.extension().spectral(spectrum,coefficient,*tables,*grid,*voxel,fi,mode)
        physical=plan.inverse_destructive(spectrum)
        del spectrum
        return c.extension().unpack(physical,*grid,False)
    op._contest_plan=plan
    op._contest_metadata.update(native_cufft=True,shared_workspace_bytes=plan.workspace_bytes,
                                spectrum_disposable=True,variant="native_"+form)
    op.apply_precond=MethodType(apply,op)
    return op
