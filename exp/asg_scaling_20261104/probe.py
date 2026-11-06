"""Cost pilot only: genuine initial field and interpolated prior design field."""
from pathlib import Path
import sys,time
import numpy as np
import torch
from scipy.ndimage import zoom
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'asg_gpu_resident_20261103'))
from resident import Candidates,Resident,pilot

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=128);args=p.parse_args();n=args.n
    rows=[]
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            start=pilot.sync();r=Resident(b,n,load_mode='parallel');r.capture()
            print('setup',pilot.sync()-start,flush=True)
            old=np.load(pilot.OUT/'trajectory_n64_s20261104/state_179.npz')
            xlate=zoom(old['x'],n/64,order=1,mode='grid-wrap',grid_mode=True)
            for name,x,beta in [('initial',pilot.initial(n,20261104),1.),('interpolated_late',xlate,8.)]:
                r.outer.set_initial(x);r.outer.evaluate(beta);r.material(r.outer.a)
                for method in ['0.5','green','gj','asg']:
                    r.select(method);r.solutions.zero_();start=pilot.sync();info,checks=r.solve();solve=pilot.sync()-start
                    moduli,energy=r.energy();start=pilot.sync();r.outer.step(energy);outer=pilot.sync()-start
                    row=dict(n=n,state=name,method=method,solve_s=solve,outer_s=outer,iterations=sum(i['iterations'] for i in info),J=float(moduli.mean()),loads=info)
                    rows.append(row);print(row,flush=True)
    pilot.save(HERE/'results/probe.json',rows)

if __name__=='__main__':main()
