"""Additional objectives and volumes on the frozen periodic mechanics backend."""
from pathlib import Path
from functools import lru_cache
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'asg_gpu_resident_20261103'))
from resident import Resident, Candidates, pilot
from outer import ResidentOuter

CASES = {
    'S20': dict(volume=.20, loads=np.eye(6)[3:], weights=[1/3]*3, title='Mean shear, 20%'),
    'S30': dict(volume=.30, loads=np.eye(6)[3:], weights=[1/3]*3, title='Mean shear, 30%'),
    'S45': dict(volume=.45, loads=np.eye(6)[3:], weights=[1/3]*3, title='Mean shear, 45%'),
    'B30': dict(volume=.30, loads=np.array([[1/3,1/3,1/3,0,0,0]]), weights=[1.], title='Bulk stiffness, 30%'),
    'M30': dict(volume=.30, loads=np.eye(6)[[2,3,4]], weights=[.5,.25,.25], title='Axial + shear, 30%'),
}
METHODS = ['green', 'gj', '0.5', 'asg']
SEEDS = [20261106, 20261107]
MAX_STEPS = 500

def initial(n, seed, volume):
    rng = np.random.default_rng(seed)
    z,y,x = np.meshgrid(*([(np.arange(n)+.5)/n]*3), indexing='ij')
    field = np.zeros((n,n,n))
    for _ in range(12):
        k = rng.integers(-2,3,3)
        if not np.any(k): k[0] = 1
        field += rng.normal()*np.cos(2*np.pi*(k[0]*x+k[1]*y+k[2]*z)+rng.uniform(0,2*np.pi))
    field *= .12/field.std()
    lo,hi = -1.,1.
    for _ in range(50):
        shift = (lo+hi)/2
        candidate = np.clip(volume+field+shift, .01, .99)
        if pilot.physical(candidate,1.)[0].mean()>volume: hi=shift
        else: lo=shift
    return candidate

@lru_cache(None)
def volume_extension():
    from torch.utils.cpp_extension import load
    return load(name='asg_scenarios_volume_20261105', sources=[str(HERE/'volume.cu')],
                extra_cuda_cflags=['-O3'], extra_cflags=['-O3'], verbose=False)

@lru_cache(None)
def energy_extension():
    from torch.utils.cpp_extension import load
    return load(name='asg_scenarios_energy_20261105', sources=[str(HERE/'energy.cu')],
                extra_include_paths=[str(ROOT/'solver/src/materials')],
                extra_cuda_cflags=['-O3'], extra_cflags=['-O3'], verbose=False)

class ScenarioOuter(ResidentOuter):
    def __init__(self, n, volume, stream=None):
        super().__init__(n, stream)
        if not 0 < volume < 1: raise ValueError('volume must lie in (0,1)')
        self.target = torch.tensor([volume], device='cuda', dtype=torch.float64)
        self.constraint = volume_extension()

    def update(self):
        self.gaussian(self.dp*3*(1.-1e-4)*self.ph.square()*self.energy, self.g)
        self.gaussian(self.dp, self.dv)
        self.ratio.copy_(self.g.clamp_min(1e-30)/self.dv.clamp_min(1e-30))
        self.bounds[0].zero_(); self.bounds[1].copy_(self.ratio.max()*1e4)
        for _ in range(45):
            self.ext.candidate(self.x,self.ratio,self.bounds,self.new)
            self.gaussian(self.new,self.t)
            self.ext.physical(self.t,self.beta,self.trial_ph,self.trial_dp,self.trial_a)
            self.constraint.bisect(self.trial_ph.mean().reshape(1),self.bounds,self.target)
        self.change.copy_((self.new-self.x).abs().max())

class ScenarioResident(Resident):
    def __init__(self, backend, n, case, load_mode='parallel', chunk=4):
        spec=CASES[case]; backend.loads=np.array(spec['loads'],copy=True)
        # The legacy affine cache is keyed by n, while this study also varies loads.
        backend.affine.cache_clear()
        count=len(backend.loads)
        super().__init__(backend,n,chunk,load_mode if count==3 else 'serial')
        self.case=case; self.load_count=count
        self.outer=ScenarioOuter(n,spec['volume'],self.stream)
        if self.batch==1:
            self.solutions=torch.zeros((count,3*n**3),device='cuda',dtype=torch.float64)
        self.energy_ext=energy_extension()
        self.normalizers=torch.tensor(np.einsum('li,ij,lj->l',backend.loads,backend.C0,backend.loads),device='cuda',dtype=torch.float64)
        self.weights=torch.tensor(spec['weights'],device='cuda',dtype=torch.float64)
        self.energy_weights=self.weights/self.normalizers

    def solve(self, rtol=1e-8):
        if self.batch==3: return self._solve_batch(range(self.load_count),rtol)
        infos=[]; checks=0
        for load in range(self.load_count):
            self.u.copy_(self.solutions[load:load+1])
            info,count=self._solve_batch([load],rtol)
            self.solutions[load:load+1].copy_(self.u)
            infos.extend(info); checks+=count
        return infos,checks

    def energy(self):
        e=self.energy_ext.evaluate(self.solutions,self.b.packed0,self.affine,self.n)
        responses=(e*self.op._material_field.reshape(1,-1)).mean(1)
        local=(e*self.energy_weights[:,None]).sum(0).reshape(self.n,self.n,self.n)
        return responses,local

    def objective(self,responses):
        return (responses*self.energy_weights).sum()
