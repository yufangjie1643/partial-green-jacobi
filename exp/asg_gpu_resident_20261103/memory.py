"""Fresh-process memory measurement. Run separately for each grid/load mode.

Reports PyTorch peak allocated/reserved memory and sampled device-used memory.
The latter includes CUDA/cuFFT allocations outside the PyTorch allocator and
the CUDA context; it is explicitly a sampled peak (not an exact high-water mark).
"""
import argparse
import gc
import threading
import time
import numpy as np
import torch
from resident import Candidates, Resident, HERE, pilot

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=32)
    p.add_argument('--load-mode',choices=['serial','parallel'],required=True)
    p.add_argument('--chunk',type=int,default=4);args=p.parse_args()
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        state=np.load(pilot.OUT/f'trajectory_n{args.n}_s20261104/state_179.npz')
        with torch.cuda.stream(stream):
            # Compile/cache kernels first; release all warm-up field/graph storage.
            r=Resident(b,args.n,args.chunk,args.load_mode);r.capture()
            r.material(torch.tensor(state['a'],device='cuda'));r.select('0.5');r.solve();stream.synchronize()
            del r;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            base_allocated=torch.cuda.memory_allocated();free,total=torch.cuda.mem_get_info();base_used=total-free
            samples=[base_used];stop=threading.Event()
            def sample():
                torch.cuda.set_device(0)
                while not stop.wait(.005):
                    f,t=torch.cuda.mem_get_info();samples.append(t-f)
            r=Resident(b,args.n,args.chunk,args.load_mode);r.capture()
            f,t=torch.cuda.mem_get_info();samples.append(t-f)
            # CUDA APIs from another thread must not interfere with graph capture.
            worker=threading.Thread(target=sample,daemon=True);worker.start()
            try:
                r.outer.set_initial(state['x']);r.solutions.copy_(torch.tensor(state['previous'],device='cuda'))
                for _ in range(5):
                    r.outer.evaluate(float(state['beta']));r.material(r.outer.a);r.select('0.5')
                    r.solve();_,energy=r.energy();r.outer.step(energy);r.outer.advance();stream.synchronize()
                    f,t=torch.cuda.mem_get_info();samples.append(t-f)
                result=dict(n=args.n,load_mode=args.load_mode,chunk=args.chunk,
                    baseline_allocated_bytes=base_allocated,baseline_device_used_bytes=base_used,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    sampled_device_used_peak_bytes=max(samples),sampled_device_increment_bytes=max(samples)-base_used,
                    allocated_increment_bytes=torch.cuda.max_memory_allocated()-base_allocated,
                    persistent_pcg_shape=list(r.p.shape),stored_solutions_shape=list(r.solutions.shape),
                    sample_interval_s=.005,samples=len(samples),gpu=torch.cuda.get_device_name(),
                    scope='fresh process, warmed compilation; allocator peak includes graph capture + five steps; device usage sampled after capture and during five steps; fixed alpha=0.5; isolated GPU')
            finally:stop.set();worker.join()
    pilot.save(HERE/'results'/f'memory_n{args.n}_{args.load_mode}_c{args.chunk}.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
