"""Frozen-trajectory comparison; all three original true-residual tests apply."""
import argparse
import time
import numpy as np
import torch
from resident import Candidates, Resident, HERE, pilot

METHODS=['green','gj','asg','0.5','0.69','compact8']

def run(grids,load_mode):
    rows=[];startup=[]
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for n in grids:
                begin=pilot.sync();resident=Resident(b,n,load_mode=load_mode);resident.capture()
                startup.append(dict(n=n,build_compile_capture_s=pilot.sync()-begin))
                print('capture ready',startup[-1],flush=True)
                baseline=b.build(np.ones((n,n,n)))
                for k in (0,49,62,179):
                    path=pilot.OUT/f'trajectory_n{n}_s20261104'/f'state_{k:03d}.npz'
                    d=np.load(path);a=d['a'];previous=torch.tensor(d['previous'],device='cuda',dtype=torch.float64)
                    for method in METHODS:
                        old,old_u,old_energy=b.evaluate(baseline,a,method,[v[None] for v in previous])
                        resident.material(torch.tensor(a,device='cuda'));resident.select(method);resident.solutions.copy_(previous)
                        alpha=float(resident.alpha);start=pilot.sync();info,checks=resident.solve();solve_s=pilot.sync()-start
                        moduli,energy=resident.energy();moduli=moduli.cpu().numpy();energy=energy.cpu().numpy()
                        new_u=resident.solutions.cpu().numpy()
                        err_u=np.linalg.norm(new_u-d['solution'])/max(np.linalg.norm(d['solution']),1e-30)
                        err_e=np.linalg.norm(energy-d['energy'])/np.linalg.norm(d['energy'])
                        err_J=abs(float(moduli.mean())-float(d['J']))/abs(float(d['J']))
                        assert abs(alpha-old['alpha'])<1e-10,(method,alpha,old['alpha'])
                        assert err_J<1e-7 and err_e<5e-6 and err_u<5e-5,(n,k,method,err_J,err_e,err_u)
                        assert all(i['converged'] for i in info) and old['converged']
                        row=dict(n=n,step=k,method=method,alpha=alpha,relative_J_error=err_J,relative_energy_error=float(err_e),relative_u_error=float(err_u),
                                 solve_s=solve_s,baseline_solve_s=old['solve_s'],loads=info,host_checks=checks)
                        rows.append(row)
                        print(f'n={n} k={k} {method}: solve {old["solve_s"]:.4f} -> {solve_s:.4f}s; Jerr={err_J:.2g}',flush=True)
                # Exercise exact-zero RHS and zero iterations without 0/0 updates.
                resident.material(torch.ones((n,n,n),device='cuda',dtype=torch.float64));resident.select('green');resident.solutions.zero_()
                info,_=resident.solve();assert all(i['iterations']==0 for i in info),info
                del baseline,resident
    pilot.save(HERE/'results'/('solver_validation_'+load_mode+'_'+'_'.join(map(str,grids))+'.json'),dict(passed=True,load_mode=load_mode,startup=startup,cases=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grids',type=int,nargs='+',default=[32,64])
    p.add_argument('--load-mode',choices=['serial','parallel'],default='serial')
    args=p.parse_args();run(args.grids,args.load_mode)
