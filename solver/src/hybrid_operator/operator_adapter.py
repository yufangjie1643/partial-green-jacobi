"""Exact operator-only candidates over the frozen hybrid storage and solver."""
from functools import lru_cache
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
HYBRID=Path(os.environ.get("FEAD_HYBRID",str(__import__('project_paths').source_path('materials'))))
if (HERE.parent/"hybrid_storage_20261021").exists():HYBRID=HERE.parent/"hybrid_storage_20261021"
sys.path.insert(0,str(HYBRID))
import hybrid_adapter as prior
from background import uniform_stencil


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ["PATH"]=str(Path(sys.executable).parent)+":/usr/local/cuda/bin:"+os.environ.get("PATH","")
    os.environ.setdefault("MAX_JOBS","2")
    return load(name="fead_hybrid_operator_20261021_v3",sources=[str(HERE/"operator_kernels.cu")],
                extra_cuda_cflags=["-O3","--ptxas-options=-v"],extra_cflags=["-O3"],verbose=False)


def attach(op,data,materials,variant,block=128):
    if op.B!=1 or not op._hybrid_indirect:raise ValueError("Requires B=1 hybrid FI storage")
    if variant=="baseline":
        op._operator_metadata=dict(variant=variant,extra_bytes=0)
        return op
    if variant not in ("isotropic","split","split_delta","split_grouped"):raise ValueError(variant)
    split=variant.startswith("split")
    mf,_=prior.material_modules();labels,tensors,local=mf.phase_table(materials)
    flags=np.array([materials[int(k)]["kind"]=="isotropic" for k in labels],dtype=np.uint8)
    phase=np.asarray(data["phase_id"]).reshape(-1)
    active=np.empty(0,dtype=np.int32);base=-1;S=np.empty(0,dtype=np.float64)
    iso=False
    if split:
        counts=np.array([np.count_nonzero(phase==label) for label in labels])
        counts[local]=0
        if counts.max()==0:
            op._operator_metadata=dict(variant=variant,fallback="no active global-frame shared phase",extra_bytes=0)
            return op
        base=int(np.argmax(counts));active=np.flatnonzero(phase!=labels[base]).astype(np.int32)
        S=uniform_stencil(tensors[base],op._modal_voxel)
        iso=bool(flags[base])
    op._operator_flags=torch.tensor(flags,device=op.device)
    op._operator_active=torch.tensor(active,device=op.device)
    op._operator_stencil=torch.tensor(S,device=op.device)
    op._operator_metadata=dict(variant=variant,block=block,base_row=base,
        base_label=int(labels[base]) if base>=0 else None,base_isotropic=iso,
        active_elements=int(active.size) if split else int(phase.size),
        total_elements=int(phase.size),extra_bytes=int(flags.nbytes+active.nbytes+S.nbytes))
    mode=0 if variant=="isotropic" else 1
    if variant in ("split_delta","split_grouped"):
        # Exact periodic identities: rhs(C)=rhs(C-Cb), and
        # <C:(E+eps(u))>=Cb:E+<(C-Cb):(E+eps(u))> because <eps(u)>=0.
        # Preserve the original physical reference before reusing its owner.
        reference=op._compute_reference_medium("zeman").copy()
        cb=op._hybrid_coeff[base].clone()
        op._hybrid_coeff.sub_(cb)
        del cb
        original_stress=op.compute_avg_stress
        Cb=tensors[base].copy()
        def stress(self,u,macro):return original_stress(u,macro)+(Cb@np.asarray(macro))[None]
        def physical_reference(self,strategy,*,verbose=False):
            if strategy!="zeman":raise ValueError("Only the cached physical zeman reference is supported")
            return reference.copy()
        op.compute_avg_stress=MethodType(stress,op)
        op._compute_reference_medium=MethodType(physical_reference,op)
        op._operator_metadata["constitutive_pool"]="inplace_delta_C_no_duplicate"
        mode=3 if variant=="split_grouped" else 2
        op._operator_metadata["grouped_background"]=variant=="split_grouped" and iso
    def apply(self,u):
        self._validate_apply_input(u)
        return extension().apply(self._hybrid_coeff,self._hybrid_index,self._operator_flags,
            self._operator_active,self._operator_stencil,u,*self.geometry(),base,mode,iso,block)
    op.apply=MethodType(apply,op)
    return op


def install(b,variant,block=128):
    previous=b.build
    if variant!="baseline":extension()
    def build(data,materials,route,precision,stages):
        op=previous(data,materials,route,precision,stages)
        attach(op,data,materials,variant,block)
        b._operator_metadata=op._operator_metadata
        stages.end("operator_optimization",op)
        return op
    b.build=build
