"""Check FP32 compiled ablation paths against the complete feature extractor."""
import sys
from pathlib import Path
import torch
from project_paths import ROOT
sys.path.insert(0,str(ROOT/'experiments/asg_independent'))
from train import subset_features,load_model
from study import Runner,load_rho,FAMILIES
from common import read,save_json
from label import assert_gpu_exclusive

out=ROOT/'runs/asg_independent_20261030';runner=Runner();manifest=read(out/'manifest.json')
compiled={mode:torch.compile(lambda x,mode=mode:subset_features(x,mode),fullgraph=True) for mode in ['contrast','one_point']}
checks=[]
with torch.inference_mode():
    for family in FAMILIES:
        row=next(r for r in manifest['samples'] if r['family']==family and r['split']=='validation' and r['contrast']==1000)
        rho=torch.tensor(load_rho(out,row),device='cuda',dtype=torch.float64);full=runner.extract(rho)
        for mode,indices,variant in [('contrast',[90,91],'contrast_only'),('one_point',list(range(9))+[90,91],'one_point')]:
            cheap=compiled[mode](rho);assert full.dtype==cheap.dtype==torch.float32
            error=float((full[:,indices]-cheap[:,indices]).abs().max());assert error<1e-6
            net=load_model(out,variant+'_20261030')
            prediction_error=float((net(full)-net(cheap)).abs().max());assert prediction_error<1e-6
            checks.append(dict(sample=row['sample_id'],mode=mode,dtype=str(cheap.dtype),input_error=error,prediction_error=prediction_error))
        assert_gpu_exclusive()
save_json(out/'compiled_input_verification.json',dict(passed=True,checks=checks),replace=True)
print('PASS',len(checks),'compiled FP32 input and prediction checks')
