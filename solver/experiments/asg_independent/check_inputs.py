"""CPU check that cheap ablation inputs equal the retained full statistics."""
from pathlib import Path
import sys
import numpy as np
import torch
from project_paths import ROOT
sys.path.insert(0,str(ROOT/'experiments/asg_independent'))
from study import load_module,load_rho,FAMILIES
from train import Model,VARIANTS,subset_features
from common import read,save_json

torch.set_num_threads(1)
out=ROOT/'runs/asg_independent_20261030'
manifest=read(out/'manifest.json')
features=load_module('check_full_features',ROOT/'experiments/global_alpha_stats/features.py')
rows=[next(r for r in manifest['samples'] if r['family']==f and r['split']=='validation' and r['contrast']==1000) for f in FAMILIES]
checks=[]
for row in rows:
    rho=torch.tensor(load_rho(out,row),dtype=torch.float64)
    full=features.extract(rho)
    for mode,indices in [('contrast',[90,91]),('one_point',list(range(9))+[90,91])]:
        cheap=subset_features(rho,mode)
        assert cheap.shape==full.shape==(1,92)
        error=float((cheap[:,indices]-full[:,indices]).abs().max())
        assert error<1e-7
        config=VARIANTS['contrast_only' if mode=='contrast' else 'one_point']
        torch.manual_seed(20261030);net=Model(config)
        net.mean.copy_(torch.linspace(-1,1,92));net.std.copy_(torch.linspace(.1,2,92))
        with torch.inference_mode():difference=float((net(full)-net(cheap)).abs().max())
        assert difference<1e-7
        checks.append(dict(sample_id=row['sample_id'],inputs=mode,max_input_difference=error,max_prediction_difference=difference))
save_json(out/'input_verification.json',dict(device='CPU',passed=True,checks=checks,parameter_counts={k:sum(p.numel() for p in Model(v).parameters()) for k,v in VARIANTS.items()}),replace=True)
print('PASS:',len(checks),'ablation input and prediction equivalence checks')
