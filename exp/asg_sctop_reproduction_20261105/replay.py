"""Matched frozen-density comparison, with six certified zero-start solves."""
import gc
import hashlib
import importlib.util
import json
import time
import numpy as np
import torch
from reproduction import HERE, PaperBackend, PaperResident, pilot
spec=importlib.util.spec_from_file_location('paper_reproduction_runner',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
warm=runner.warm;write=runner.write


def main():
    cases=['S20_n30_i1','B30_n30_i1','A20_n30_i2']
    methods=['green','gj','0.5'];output=HERE/'results/replay.json'
    assert not output.exists(), 'Preserve existing replay results'
    result=dict(protocol=dict(cases=cases,methods=methods,repetitions=3,
        snapshots='available iterations 1,25,75,150 and accepted final density',
        initial_displacement='zero for every method and repetition',rtol=1e-8,
        timing='preconditioner construction, six solves and independent true residuals; excludes material transfer, graph capture, energy and tensor evaluation'),
        rows=[])
    result['source_sha256']=hashlib.sha256((HERE/'replay.py').read_bytes()).hexdigest()
    with pilot.Guard() as guard,torch.inference_mode():
        b=PaperBackend();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warm(b,30)
            for case in cases:
                path=HERE/'results/full'/f'{case}_0.5_r0_parallel.json'
                d=json.loads(path.read_text());spec=d['spec']
                r=PaperResident(b,30,spec['objective'],spec['volume']);r.capture()
                with np.load(path.with_suffix('.npz')) as data:
                    fields={key:data[key].copy() for key in sorted(data.files) if key.startswith('rho_')}
                    fields['final']=data['physical'].copy()
                for state,rho in fields.items():
                    a=torch.tensor(1e-9+(1-1e-9)*rho**3,device='cuda',dtype=torch.float64)
                    r.material(a)
                    for rep in range(3):
                        # Rotate order across repeats to limit systematic order effects.
                        for method in methods[rep:]+methods[:rep]:
                            if guard.events:raise RuntimeError('GPU interference: '+repr(guard.events))
                            r.solutions.zero_();start=pilot.sync()
                            row=dict(case=case,state=state,repeat=rep,method=method,status='running')
                            try:
                                r.select(method);selected=pilot.sync()
                                certificates,checks=r.solve();end=pilot.sync()
                                row.update(status='complete',preconditioner_s=selected-start,
                                    solve_s=end-selected,total_s=end-start,loads=certificates,host_checks=checks,
                                    pcg_iterations=sum(c['iterations'] for c in certificates))
                                C=r.effective_tensor().cpu().numpy()
                                row['C_effective']=C.tolist()
                            except RuntimeError as exc:
                                row.update(status='failed',error=repr(exc),elapsed_s=pilot.sync()-start)
                            result['rows'].append(row);write(output,result)
                            print(case,state,rep,method,row['status'],round(row.get('total_s',0),4),flush=True)
                del r;gc.collect();torch.cuda.empty_cache()
    write(output,result)


if __name__=='__main__':main()
