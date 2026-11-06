"""Exact periodic SIMP / OC equations with persistent FP64 GPU buffers."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='asg_resident_outer_20261103',sources=[str(Path(__file__).with_suffix('.cu'))],extra_cuda_cflags=['-O3','--fmad=false'],extra_cflags=['-O3'],verbose=False)

class ResidentOuter:
    def __init__(self,n,stream=None):
        self.n=n;self.stream=stream or torch.cuda.current_stream();self.ext=extension()
        sigma=.9*n/32;radius=int(4*sigma+.5);q=np.arange(-radius,radius+1,dtype=np.float64)
        weights=np.exp(-.5*q*q/(sigma*sigma));weights/=weights.sum()
        self.weights=torch.tensor(weights,device='cuda',dtype=torch.float64)
        self.x=torch.empty((n,n,n),device='cuda',dtype=torch.float64)
        self.beta=torch.ones(1,device='cuda',dtype=torch.float64)
        for name in ['ph','dp','a','energy','new','g','dv','ratio','tmp1','tmp2','t','trial_ph','trial_dp','trial_a']:
            setattr(self,name,torch.empty_like(self.x))
        self.bounds=torch.zeros(2,device='cuda',dtype=torch.float64)
        self.change=torch.zeros(1,device='cuda',dtype=torch.float64)
        self.physical_graph=None;self.update_graph=None

    def gaussian(self,x,out):
        self.ext.filter(x,self.weights,self.tmp1,self.n,0)
        self.ext.filter(self.tmp1,self.weights,self.tmp2,self.n,1)
        self.ext.filter(self.tmp2,self.weights,out,self.n,2)

    def physical(self):
        self.gaussian(self.x,self.t);self.ext.physical(self.t,self.beta,self.ph,self.dp,self.a)

    def update(self):
        self.gaussian(self.dp*3*(1.-1e-4)*self.ph.square()*self.energy,self.g)
        self.gaussian(self.dp,self.dv)
        self.ratio.copy_(self.g.clamp_min(1e-30)/self.dv.clamp_min(1e-30))
        self.bounds[0].zero_();self.bounds[1].copy_(self.ratio.max()*1e4)
        for _ in range(45):
            self.ext.candidate(self.x,self.ratio,self.bounds,self.new)
            self.gaussian(self.new,self.t)
            self.ext.physical(self.t,self.beta,self.trial_ph,self.trial_dp,self.trial_a)
            self.ext.bisect(self.trial_ph.mean().reshape(1),self.bounds)
        self.change.copy_((self.new-self.x).abs().max())

    def capture(self):
        # Capture doesn't change x. It is safe to warm on a positive test density.
        self.x.fill_(.3);self.energy.fill_(1.)
        self.physical();self.update();self.stream.synchronize()
        self.physical_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.physical_graph,stream=self.stream):self.physical()
        self.update_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.update_graph,stream=self.stream):self.update()
        self.stream.synchronize()

    def set_initial(self,x):self.x.copy_(torch.as_tensor(x,device='cuda',dtype=torch.float64))
    def evaluate(self,beta):self.beta.fill_(float(beta));self.physical_graph.replay()
    def step(self,energy):self.energy.copy_(energy);self.update_graph.replay()
    def advance(self):self.x.copy_(self.new)
