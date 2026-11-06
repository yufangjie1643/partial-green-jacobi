"""Post-hoc diagnosis of independent-test regressions; never used for refitting."""
from pathlib import Path
import json
import time
import numpy as np
import torch
import run as r
from fit import write,sha

HERE=Path(__file__).resolve().parent
STATES=[('A20_e6_A',1),('A20_e6_A',20),('B20_e6_A',152)]


def main():
    gate=json.loads((HERE/'replay_gate.json').read_text())
    assert not gate['contrast_pass'] and not gate['eta_pass']
    registration=dict(created_unix=time.time(),states=STATES,alpha_grid=[i/16 for i in range(17)],repeats=3,
        status='post-hoc exploratory diagnosis, selected after observing regression, not an independent validation subset',
        reason='worst case A20_e6_A step20; bulk e6_A final regression; same initial state as control',
        models_not_refit=True,model_sha256=sha(HERE/'MODELS_FROZEN.json'),source_sha256=sha(__file__))
    p=HERE/'DIAGNOSTIC_PROTOCOL.json'
    if p.exists():assert json.loads(p.read_text())['source_sha256']==registration['source_sha256']
    else:write(p,registration)
    start=time.perf_counter()
    with r.pilot.Guard() as guard,torch.inference_mode():
        b=r.PaperBackend();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            r.warm(b);solver=r.create(b,r.specs()[0])
            for si,(label,k) in enumerate(STATES):
                out=HERE/'diagnostic'/f'{label}_k{k:03d}.json'
                if out.exists():continue
                d=json.loads((HERE/'reference'/(label+'.json')).read_text());s=next(x for x in d['states'] if x['iteration']==k)
                with np.load(HERE/s['path']) as z:
                    solver.material(torch.tensor(z['a'],device='cuda'));solver.outer.ph.copy_(torch.tensor(z['rho'],device='cuda'));u0=torch.tensor(z['u0'],device='cuda')
                rows=[];methods=['green']+[str(i/16) for i in range(1,17)]
                for rep in range(3):
                    for method in np.random.default_rng(550000+si*10+rep).permutation(methods).tolist():
                        rows.append(r.candidate(solver,u0,method,rep,np.array(s['C_reference']),guard))
                write(out,dict(state=s,records=rows,post_hoc=True));print('DIAGNOSTIC',label,k,flush=True)
        r.check(guard)
    write(HERE/'sessions'/f'diagnostic_{time.time_ns()}.json',dict(seconds=time.perf_counter()-start,complete=True,interference=guard.events,post_hoc=True))


if __name__=='__main__':main()
