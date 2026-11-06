"""Validate the batched Green action and compare captured GPU execution."""
import numpy as np
import torch
from resident import Candidates, Resident, HERE, pilot

def timing(fn,stream):
    for _ in range(3):fn()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(4):fn()
    values=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):graph.replay()
        end.record();end.synchronize();values.append(start.elapsed_time(end)/200)
    return float(np.median(values))

def run():
    rows=[]
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for n in (32,64):
                r=Resident(b,n,load_mode='parallel');r.material(torch.exp(-9*torch.rand((n,n,n),device='cuda',dtype=torch.float64)))
                u=torch.randn_like(r.u)
                for method in ('green','0.5','gj'):
                    r.select(method);old=r.B_serial(u);new=r.B(u)
                    error=float(torch.linalg.vector_norm(new-old)/torch.linalg.vector_norm(old))
                    assert error<2e-13,(n,method,error)
                r.active.fill_(True)
                row=dict(n=n,relative_Green_error=error,serial_green_ms=timing(lambda:r.B_serial(u),stream),
                         batched_green_ms=timing(lambda:r.B(u),stream),
                         original_K_ms=timing(lambda:r.true_K(u),stream),scalar_K_ms=timing(lambda:r.K(u),stream))
                rows.append(row);print(row,flush=True)
    pilot.save(HERE/'results/green_validation.json',dict(passed=True,cases=rows))

if __name__=='__main__':run()
