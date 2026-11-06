"""Independent general-C21 operator comparison for the specialized CUDA kernel."""
import numpy as np
import argparse
import torch
from resident import Candidates, Resident, HERE, pilot

def run(load_mode):
    rows=[]
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            generator=torch.Generator(device='cuda');generator.manual_seed(731)
            for n in (4,32,64):
                r=Resident(b,n,load_mode=load_mode)
                for contrast in (1.,1e4):
                    a=torch.exp(-np.log(contrast)*torch.rand((n,n,n),generator=generator,device='cuda',dtype=torch.float64))
                    r.material(a);r.active.fill_(True)
                    u=torch.randn(r.u.shape,generator=generator,device='cuda',dtype=torch.float64)
                    v=torch.randn(r.u.shape,generator=generator,device='cuda',dtype=torch.float64)
                    actual=r.K(u);expected=r.true_K(u)
                    error=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
                    symmetry=float(((u*r.K(v)).sum()-(v*actual).sum()).abs()/((u*r.K(u)).sum()*(v*r.K(v)).sum()).sqrt())
                    assert error<2e-14 and symmetry<2e-14,(n,contrast,error,symmetry)
                    mute=min(1,r.batch-1);r.active[mute]=False;masked=r.K(u)
                    assert bool((masked[mute]==0).all())
                    keep=[i for i in range(r.batch) if i!=mute]
                    if keep:
                        assert float(torch.linalg.vector_norm(masked[keep]-actual[keep])/torch.linalg.vector_norm(actual[keep]))<2e-14
                    r.active.fill_(True)
                    translated=u.reshape(r.batch,-1,3)+torch.tensor([1.,-2.,.5],device='cuda')
                    shift_error=float(torch.linalg.vector_norm(r.K(translated.reshape_as(u))-actual)/torch.linalg.vector_norm(actual))
                    assert shift_error<2e-14,shift_error
                    rows.append(dict(n=n,contrast=contrast,relative_operator_error=error,symmetry_error=symmetry,translation_error=shift_error))
                print(f'operator n={n}: passed',flush=True)
    pilot.save(HERE/'results'/f'operator_validation_{load_mode}.json',dict(passed=True,load_mode=load_mode,cases=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--load-mode',choices=['serial','parallel'],default='serial')
    run(p.parse_args().load_mode)
