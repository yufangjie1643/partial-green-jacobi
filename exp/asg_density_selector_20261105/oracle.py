"""Retrospective alpha grids on preselected fresh endpoint states; not online speed."""
from pathlib import Path
import gc
import json
import time
import numpy as np
import torch
import run as r
from fit import write,sha

HERE=Path(__file__).resolve().parent
CASES=['S20_e4_A','S20_e9_B','B20_e4_A','B20_e9_B','A20_e4_A','A20_e9_B']


def register():
    p=HERE/'ORACLE_FROZEN.json'
    value=dict(created_unix=time.time(),cases=CASES,states='first and last saved states, deduplicated',
        alpha_grid=[i/16 for i in range(17)],repetitions=3,selection='objective x eta endpoints; alternating initialization A/B',
        use='retrospective explanation only; no selector refit',source_sha256=sha(__file__))
    if p.exists():assert json.loads(p.read_text())['source_sha256']==value['source_sha256']
    else:write(p,value)


def main():
    register();r.freeze_test();start=time.perf_counter()
    with r.pilot.Guard() as guard,torch.inference_mode():
        b=r.PaperBackend();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            r.warm(b);solver=r.create(b,r.specs()[0]);states=[]
            for label in CASES:
                d=json.loads((HERE/'reference'/(label+'.json')).read_text());assert d['status']=='complete'
                states.extend([d['states'][0],d['states'][-1]])
            for si,s in enumerate(states):
                out=HERE/'oracle'/(Path(s['path']).stem+'.json')
                if out.exists():continue
                with np.load(HERE/s['path']) as z:
                    solver.material(torch.tensor(z['a'],device='cuda'));solver.outer.ph.copy_(torch.tensor(z['rho'],device='cuda'));u0=torch.tensor(z['u0'],device='cuda')
                methods=['green']+[str(i/16) for i in range(1,17)];rows=[]
                for rep in range(3):
                    for method in np.random.default_rng(540000+si*10+rep).permutation(methods).tolist():
                        rows.append(r.candidate(solver,u0,method,rep,np.array(s['C_reference']),guard))
                write(out,dict(state=s,records=rows));print('ORACLE',si+1,len(states),flush=True)
            del solver;gc.collect();torch.cuda.empty_cache()
        r.check(guard)
    write(HERE/'sessions'/f'oracle_{time.time_ns()}.json',dict(seconds=time.perf_counter()-start,complete=True,interference=guard.events,source_sha256=sha(__file__)))


if __name__=='__main__':main()
