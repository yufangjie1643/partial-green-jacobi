"""Compare the actual CUDA/graph OC chain with the original SciPy equations."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'asg_topopt_20261103'))
import study as pilot
from outer import ResidentOuter

def run():
    rows=[]
    with pilot.Guard():
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for n in (4,32,64):
                outer=ResidentOuter(n,stream);outer.capture()
                rng=np.random.default_rng(731+n)
                cases=[('random',rng.uniform(.01,.99,(n,n,n)),rng.uniform(.01,2,(n,n,n)))]
                folder=pilot.OUT/f'trajectory_n{n}_s20261104'
                if folder.exists():
                    for k in (0,49,62,179):
                        path=folder/f'state_{k:03d}.npz'
                        if path.exists():
                            state=np.load(path)
                            cases.append((path.name,state['x'],state['energy']))
                for name,x,energy in cases:
                    for beta in (1.,2.,4.,8.):
                        outer.set_initial(x);outer.evaluate(beta)
                        outer.step(torch.as_tensor(energy,device='cuda',dtype=torch.float64))
                        stream.synchronize()
                        ph,dp=pilot.physical(x,beta)
                        g=pilot.filt(dp*3*(1-1e-4)*ph**2*energy);dv=pilot.filt(dp)
                        expected=dict(ph=ph,dp=dp,a=pilot.material(ph),g=g,dv=dv,new=pilot.oc(x,g,dv,beta))
                        errors={key:float(np.max(np.abs(getattr(outer,key).cpu().numpy()-value))) for key,value in expected.items()}
                        # The 45 fixed bisections can take an adjacent final branch.
                        assert max(errors[k] for k in ('ph','dp','a','g','dv'))<2e-12,(n,name,beta,errors)
                        assert errors['new']<2e-8,(n,name,beta,errors)
                        rows.append(dict(n=n,case=name,beta=beta,max_absolute_errors=errors))
                print(f'outer n={n}: passed',flush=True)
    pilot.save(HERE/'results/outer_validation.json',dict(passed=True,cases=rows))

if __name__=='__main__':run()
