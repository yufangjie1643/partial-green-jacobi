"""Validation-parent GPU memory and convergence preflight, not model selection."""
from pathlib import Path
import gc
import sys
import torch
from project_paths import ROOT
sys.path.insert(0,str(ROOT/'experiments/asg_independent'))
from study import Runner,field
from common import read,save_json
from label import assert_gpu_exclusive

out=ROOT/'runs/asg_independent_20261030'
manifest=read(out/'manifest.json')
parent=next(g for g in manifest['geometries'] if g['family']=='smooth' and g['split']=='validation')
runner=Runner();results=[]
for n in [128,256]:
    assert_gpu_exclusive();torch.cuda.reset_peak_memory_stats()
    rho=field(parent,n,1000);runner.warm(rho)
    for alpha in [0.,.5,1.]:
        value=runner.measure(rho,alpha)
        results.append(dict(n=n,alpha=alpha,**{k:v for k,v in value.items() if k!='alpha'}))
    with torch.inference_mode():features=runner.extract(torch.tensor(rho,device='cuda',dtype=torch.float64)).cpu()
    assert features.shape==(1,92) and torch.isfinite(features).all()
    assert_gpu_exclusive()
    results[-1]['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    results[-1]['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    print(n,results[-1],flush=True)
    del rho,features;gc.collect()
save_json(out/'resolution_preflight.json',dict(validation_parent=parent['geometry_id'],not_used_for_training_or_selection=True,
          purpose='memory and implementation preflight; not a sealed transfer result or a paired timing table',results=results),replace=True)
