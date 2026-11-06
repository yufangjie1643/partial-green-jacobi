"""CUDA-resident selector validation; original operator and PCG are unchanged."""
from pathlib import Path
import argparse
import gc
import itertools
import json
import math
import platform
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'asg_sctop_reproduction_20261105'
ARCHIVE=HERE.parent/'asg_alpha_landscape_20261105'
sys.path.insert(0,str(SOURCE))
from reproduction import PaperBackend,PaperResident,pilot
from fit import write,sha

FROZEN=json.loads((HERE/'MODELS_FROZEN.json').read_text())
METHODS=['compact8','compact8q','eta','contrast','gj','green']
if FROZEN['gray_promoted_by_offline']:METHODS.insert(4,'gray')
STAGES=['filter_s','material_s','feature_s','preconditioner_s','solve_certify_s','response_s','outer_s','advance_check_s']


@torch.compile(fullgraph=True)
def contrast_alpha(a,params):
    lo,hi=torch.aminmax(a)
    s=torch.log10(hi/lo).clamp(0.,9.)
    b4,b6,b9=params.unbind()
    value=torch.where(s<4,b4*s/4,torch.where(s<6,b4+(b6-b4)*(s-4)/2,b6+(b9-b6)*(s-6)/3))
    return torch.where(s>1e-12,(value.clamp(0,1)*16+.5).floor()/16,0.)


@torch.compile(fullgraph=True)
def gray_alpha(a,rho,params,gray_params):
    lo,hi=torch.aminmax(a);s=torch.log10(hi/lo).clamp(0.,9.)
    b4,b6,b9=params.unbind()
    base=torch.where(s<4,b4*s/4,torch.where(s<6,b4+(b6-b4)*(s-4)/2,b6+(b9-b6)*(s-6)/3))
    v=rho.mean();g=((v-rho.square().mean())/(v*(1-v)+1e-12)).clamp(0,1)
    value=base+gray_params[0]*(gray_params[1]-g)
    return torch.where(s>1e-12,(value.clamp(0,1)*16+.5).floor()/16,0.)


class SelectorResident(PaperResident):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.parameters={name:torch.tensor(m['anchors'],device='cuda',dtype=torch.float64) for name,m in FROZEN['models'].items()}
        m=FROZEN['models']['gray'];self.gray_parameters=torch.tensor([m['gamma'],m['g_ref']],device='cuda',dtype=torch.float64)
        self.eta_value=FROZEN['models']['eta']['anchors'][int(np.argmin(abs(np.array([4,6,9])+math.log10(self.outer.emin))))]

    def choose(self,method):
        a=self.op._material_field
        if method=='green':self.alpha.zero_()
        elif method=='gj':self.alpha.fill_(1.)
        elif method=='eta':self.alpha.fill_(self.eta_value)
        elif method=='contrast':self.alpha.copy_(contrast_alpha(a,self.parameters['contrast']))
        elif method=='gray':self.alpha.copy_(gray_alpha(a,self.outer.ph,self.parameters['gray'],self.gray_parameters))
        elif method in ['compact8','compact8q']:
            area,jump,contrast,complexity=self.b.law_features(a).to(torch.float64).unbind()
            value=1.-.75*(area*jump).clamp_min(0).sqrt()/contrast.clamp_min(1e-30).sqrt()-(2.5/complexity.clamp_min(1e-30))**8
            value=torch.where((complexity>1e-12)&(contrast>1e-12),value.clamp(0.,1.),0.)
            value=torch.where(value<.375,0.,value)
            if method=='compact8q':value=(16*value+.5).floor()/16
            self.alpha.copy_(value)
        else:self.alpha.fill_(float(method))

    def build_scale(self,method):
        if method=='green':self.scale.fill_(1.);return
        a=self.op._material_field;d=torch.zeros((*a.shape,3),device='cuda',dtype=torch.float64)
        for offset,element in zip(self.b.offsets,self.element_diag):
            d.add_(torch.roll(a,tuple(map(int,offset[::-1])),(0,1,2))[...,None]*element)
        q=d.reshape(1,-1).log();q=q-q.mean()
        self.scale.copy_((-.5*self.alpha*q).exp())

    def select(self,method):self.choose(method);self.build_scale(method)


def specs():
    return [dict(label=f'{name}_e{int(-math.log10(eta))}_{init}',objective=objective,volume=.2,emin=eta,
        init=init,seed=2026110521 if init=='A' else 2026110522,n=32,rmin=3.2,beta=4.,eta=.5)
        for name,objective in [('S20','shear'),('B20','bulk'),('A20','auxetic')]
        for eta in [1e-4,1e-6,1e-9] for init in ['A','B']]


def create(b,spec,mode='parallel'):
    r=SelectorResident(b,spec['n'],spec['objective'],spec['volume'],spec['rmin'],spec['beta'],spec['eta'],spec['emin'],load_mode=mode)
    r.capture();return r


def check(guard):
    if guard.events:raise RuntimeError('GPU interference: '+repr(guard.events))


def warm(b,mode='parallel'):
    spec=specs()[0];r=create(b,spec,mode)
    with np.load(ARCHIVE/'states/S20_e4_i4_k001.npz') as z:
        r.material(torch.tensor(z['a'],device='cuda'));r.outer.ph.copy_(torch.tensor(z['rho'],device='cuda'))
    for method in METHODS+['gray']:r.select(method)
    r.select('gj');r.solutions.zero_();r.solve();pilot.sync()
    del r;gc.collect();torch.cuda.empty_cache()


def candidate(r,u0,method,repeat,Cref,guard):
    check(guard);r.solutions.copy_(u0)
    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    start=pilot.sync();row=dict(method=method,repeat=repeat)
    try:
        events[0].record();r.choose(method);events[1].record();r.build_scale(method);events[2].record()
        certs,checks=r.solve();events[3].record();end=pilot.sync()
        C=r.effective_tensor().cpu().numpy();err=float(np.linalg.norm(C-Cref)/max(np.linalg.norm(Cref),1e-30))
        if err>1e-6:raise RuntimeError(f'effective tensor mismatch {err}')
        row.update(status='complete',total_s=end-start,feature_s=events[0].elapsed_time(events[1])/1000,
            preconditioner_s=events[1].elapsed_time(events[2])/1000,solve_certify_s=events[2].elapsed_time(events[3])/1000,
            alpha=float(r.alpha),loads=certs,checks=checks,pcg_iterations=sum(x['iterations'] for x in certs),relative_C_error=err)
    except RuntimeError as ex:
        row.update(status='failed',elapsed_s=pilot.sync()-start,error=repr(ex))
    check(guard);return row


def development(b,guard):
    selected=[]
    for p in sorted((ARCHIVE/'trajectories').glob('*.json')):
        d=json.loads(p.read_text());states=d['states']
        for s in [states[0],min(states,key=lambda v:abs(v['iteration']-20)),states[-1]]:
            if s['path'] not in [x['path'] for x in selected]:selected.append(s)
    r=create(b,specs()[0]);parity=[]
    for si,s in enumerate(selected):
        target=HERE/'development'/f'{Path(s["path"]).stem}.json'
        with np.load(ARCHIVE/s['path']) as z:
            a=torch.tensor(z['a'],device='cuda');rho=torch.tensor(z['rho'],device='cuda');u0=torch.tensor(z['u0'],device='cuda')
        r.material(a);r.outer.ph.copy_(rho)
        r.eta_value=FROZEN['models']['eta']['anchors'][int(np.argmin(abs(np.array([4,6,9])+math.log10(s['spec']['emin']))))]
        # Check copied formula and scale against the original implementation before timing.
        PaperResident.select(r,'compact8');old_alpha=r.alpha.clone();old_scale=r.scale.clone()
        r.select('compact8')
        assert torch.equal(old_alpha,r.alpha) and torch.equal(old_scale,r.scale)
        r.choose('contrast');contrast=float(r.alpha)
        exact=np.log10(float(a.max()/a.min()));anchors=FROZEN['models']['contrast']['anchors']
        cpu=math.floor(np.interp(np.clip(exact,0,9),[0,4,6,9],[0,*anchors])*16+.5)/16
        assert contrast==cpu,(contrast,cpu)
        parity.append(dict(state=s['path'],formula8_exact=True,contrast=contrast,cpu=cpu))
        if target.exists():continue
        rows=[]
        for rep in range(3):
            for method in np.random.default_rng(512000+si*10+rep).permutation(['compact8','compact8q','eta','contrast','gj']).tolist():
                rows.append(candidate(r,u0,method,rep,np.array(s['C_reference']),guard))
        write(target,dict(state=s,records=rows));print('DEVELOPMENT',si+1,len(selected),flush=True)
    r.material(torch.ones_like(a));r.choose('contrast');assert float(r.alpha)==0
    write(HERE/'implementation_checks.json',dict(formula8_parity=parity,constant_field_alpha=0,
        pcg_recurrence='inherited unchanged',operator='inherited unchanged'))
    del r;gc.collect();torch.cuda.empty_cache()


def freeze_test():
    path=HERE/'TEST_FROZEN.json'
    inputs=[Path(__file__),HERE/'fit.py',HERE/'MODELS_FROZEN.json',SOURCE/'reproduction.py',
        SOURCE/'outer.cu',SOURCE/'scalar_fi.cu',HERE.parent/'asg_gpu_resident_20261103/resident.py']
    value=dict(created_unix=time.time(),cases=specs(),methods=METHODS,repeats=3,steps=[1,5,20,80],always_last=True,
        source_hashes={str(p):sha(p) for p in inputs},
        init_A='continuous family4 centers (linspace(1,32,5)-0.5)/32, radius 3/32, torus distance, base 0.001/0.4; six modes with normal amplitude/uniform phase, multiplicative exp(0.35*unit_variance_noise)',
        init_B='Fourier logit: all integer wave vectors in [-2,2]^3 with first nonzero component positive; independent N(0,1) sine/cosine coefficients divided by 1+|k|^2; normalize analytic variance and multiply by 1.2',
        volume_normalization='one scalar logit shift, 48 bisection steps using original periodic filter/projection; shared initial density across eta/objectives',
        sampling='cell centers (j+0.5)/N, same continuous fields on all grids',
        primary_scope='new seeds and a new generator; only two initial fields, not a population significance test',
        gray_status='excluded by prespecified grouped gate' if not FROZEN['gray_promoted_by_offline'] else 'eligible after net pilot',
        fresh_gate='mean per-case saving >=5%, at least 12/18 wins, worst case ratio<=1.05, no additional failures',
        full_quality='volume abs error<=1e-3; flag objective/C relative difference>1e-3 or nu abs difference>1e-3',
        full_timing='setup + all outer steps including advance and stop checks + final tighter reanalysis; excludes process compilation, serialization and diagnostic snapshot exports')
    if path.exists():
        old=json.loads(path.read_text());assert old['source_hashes']==value['source_hashes'];assert old['cases']==value['cases']
    else:write(path,value)


def raw_logit(spec):
    n=spec['n'];z,y,x=np.meshgrid(*([(np.arange(n)+.5)/n]*3),indexing='ij');rng=np.random.default_rng(spec['seed'])
    noise=np.zeros((n,n,n));variance=0.
    if spec['init']=='A':
        centers=(np.linspace(1,32,5)-.5)/32
        def distance(q):
            d=np.abs(q[...,None]-centers);return np.minimum(d,1-d).min(-1)**2
        base=np.where(distance(z)+distance(y)+distance(x)<(3/32)**2,.001,.4)
        for k in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]:
            amp=rng.normal();phase=rng.uniform(0,2*np.pi)
            noise+=amp*np.cos(2*np.pi*(k[0]*z+k[1]*y+k[2]*x)+phase);variance+=.5*amp**2
        raw=np.clip(base*np.exp(.35*noise/math.sqrt(variance)),.001,.999)
        return np.log(raw/(1-raw))
    for k in itertools.product(range(-2,3),repeat=3):
        if not any(k) or next(t for t in k if t)!=abs(next(t for t in k if t)):continue
        ac,bs=rng.normal(size=2)/(1+sum(t*t for t in k));angle=2*np.pi*(k[0]*z+k[1]*y+k[2]*x)
        noise+=ac*np.cos(angle)+bs*np.sin(angle);variance+=.5*(ac*ac+bs*bs)
    return 1.2*noise/math.sqrt(variance)


def initial(r,spec):
    cache=HERE/'initials'/f'n{spec["n"]}_{spec["init"]}_v{spec["volume"]}_b{spec["beta"]}_r{spec["rmin"]}.npz'
    if cache.exists():
        with np.load(cache) as z:r.outer.set_initial(z['x'])
        return
    freeze_test();logit=torch.tensor(raw_logit(spec),device='cuda',dtype=torch.float64)
    lo,hi=-20.,20.
    for _ in range(48):
        shift=(lo+hi)/2;r.outer.x.copy_(torch.sigmoid(logit+shift));r.outer.evaluate()
        if float(r.outer.ph.mean())>spec['volume']:hi=shift
        else:lo=shift
    r.outer.x.copy_(torch.sigmoid(logit+(lo+hi)/2));r.outer.evaluate()
    assert abs(float(r.outer.ph.mean())-spec['volume'])<1e-10
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,x=r.outer.x.cpu().numpy(),rho=r.outer.ph.cpu().numpy(),shift=(lo+hi)/2)


def optimize(b,spec,method,repeat,guard,collect=False):
    folder='reference' if collect else 'full';stem=spec['label'] if collect else f'{spec["label"]}_{method}_r{repeat}'
    path=HERE/folder/(stem+'.json')
    if path.exists():return
    data=dict(spec=spec,method=method,repeat=repeat,status='running',rows=[],states=[])
    r=None;last=None;start=pilot.sync();stage='setup'
    try:
        check(guard);r=create(b,spec);initial(r,spec);r.solutions.zero_();data['setup_s']=pilot.sync()-start
        for k in range(1,201):
            check(guard);stage=f'iteration_{k}'
            ev=[torch.cuda.Event(enable_timing=True) for _ in range(9)];t=pilot.sync()
            ev[0].record();r.outer.evaluate();ev[1].record();r.material(r.outer.a);ev[2].record()
            if collect:last=dict(a=r.op._material_field.clone(),rho=r.outer.ph.clone(),u0=r.solutions.clone())
            r.choose(method);ev[3].record();r.build_scale(method);ev[4].record();certs,checks=r.solve();ev[5].record()
            f,energy,diagonal=r.energy(k);ev[6].record();r.outer.step(energy);ev[7].record()
            vals=torch.cat((f.reshape(1),r.outer.ph.mean().reshape(1),r.outer.change,r.alpha.reshape(1),r.outer.bounds[2:])).cpu().numpy()
            assert vals[4]==0;change=float(vals[2]);last_step=change<=.01 or k==200
            r.outer.advance();ev[8].record();end=pilot.sync()
            costs={name:ev[j].elapsed_time(ev[j+1])/1000 for j,name in enumerate(STAGES)}
            data['rows'].append(dict(iteration=k,f=float(vals[0]),volume=float(vals[1]),change=change,alpha=float(vals[3]),
                step_s=end-t,host_other_s=end-t-sum(costs.values()),pcg_iterations=sum(c['iterations'] for c in certs),loads=certs,**costs))
            if collect and (k in [1,5,20,80] or last_step):
                sp=HERE/'states'/f'{spec["label"]}_k{k:03d}.npz';sp.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(sp,**{name:v.cpu().numpy() for name,v in last.items()})
                data['states'].append(dict(path=str(sp.relative_to(HERE)),trajectory=spec['label'],iteration=k,spec=spec,
                    C_reference=r.effective_tensor().cpu().numpy().tolist()))
            if k%50==0:print('STEP',stem,k,flush=True)
            if last_step:break
        stage='final_reanalysis';t=pilot.sync();r.outer.evaluate();r.material(r.outer.a);r.select(method)
        certs,_=r.solve(1e-9);f,_,_=r.energy(len(data['rows']));C=r.effective_tensor().cpu().numpy();S=np.linalg.inv(C)
        volume=float(r.outer.ph.mean());end=pilot.sync()
        data.update(status='complete',steps=len(data['rows']),optimization_converged=change<=.01,final_s=end-t,
            total_s=data['setup_s']+sum(x['step_s'] for x in data['rows'])+end-t,
            final=dict(f=float(f),volume=volume,C_effective=C.tolist(),nu=[float(-S[i,j]/S[i,i]) for i in range(3) for j in range(3) if i!=j],loads=certs))
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'),rho=r.outer.ph.cpu().numpy(),x=r.outer.x.cpu().numpy())
    except Exception as ex:
        if 'GPU interference' in str(ex):raise
        data.update(status='failed',failure_stage=stage,error=repr(ex),steps=len(data['rows']))
    data['attempt_wall_s']=pilot.sync()-start;check(guard);write(path,data)
    print('DONE',stem,data['status'],data['steps'],round(data.get('total_s',0),3),data.get('error',''),flush=True)
    del r,last;gc.collect();torch.cuda.empty_cache()


def collect(b,guard):
    freeze_test()
    for spec in specs():optimize(b,spec,'gj',0,guard,True)


def replay(b,guard,mode='parallel'):
    freeze_test();records=[json.loads(p.read_text()) for p in sorted((HERE/'reference').glob('*.json'))]
    assert len(records)==18
    good=[d for d in records if d['status']=='complete']
    if len(good)!=18:raise RuntimeError('Reference trajectories failed; preserve records and investigate before selecting cases')
    states=sorted([s for d in records for s in d['states']],key=lambda s:(s['iteration'],s['trajectory']))
    if mode=='serial':states=[s for s in states if s['iteration']==1 or s==next(d for d in records if d['spec']['label']==s['trajectory'])['states'][-1]]
    r=create(b,specs()[0],mode)
    for si,s in enumerate(states):
        out=HERE/f'replay_{mode}'/(Path(s['path']).stem+'.json')
        if out.exists():continue
        with np.load(HERE/s['path']) as z:
            r.material(torch.tensor(z['a'],device='cuda'));r.outer.ph.copy_(torch.tensor(z['rho'],device='cuda'));u0=torch.tensor(z['u0'],device='cuda')
        r.eta_value=FROZEN['models']['eta']['anchors'][int(np.argmin(abs(np.array([4,6,9])+math.log10(s['spec']['emin']))))]
        rows=[];methods=METHODS if mode=='parallel' else ['compact8','eta','contrast','gj']
        for rep in range(3):
            for method in np.random.default_rng(520000+si*10+rep).permutation(methods).tolist():
                rows.append(candidate(r,u0,method,rep,np.array(s['C_reference']),guard))
        write(out,dict(state=s,mode=mode,records=rows));print('REPLAY',mode,si+1,len(states),flush=True)
    del r;gc.collect();torch.cuda.empty_cache()


def full(b,guard):
    gate=json.loads((HERE/'replay_gate.json').read_text());methods=gate['full_methods']
    assert methods and (gate['contrast_pass'] or gate['eta_pass'])
    for ci,s in enumerate(specs()):
        for rep in range(3):
            for method in np.random.default_rng(530000+ci*10+rep).permutation(methods).tolist():
                optimize(b,s,method,rep,guard)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['development','collect','replay','serial','full']);args=p.parse_args()
    start=time.perf_counter()
    with pilot.Guard() as guard,torch.inference_mode():
        b=PaperBackend();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warm(b,'serial' if args.mode=='serial' else 'parallel')
            if args.mode=='development':development(b,guard)
            elif args.mode=='collect':collect(b,guard)
            elif args.mode=='replay':replay(b,guard)
            elif args.mode=='serial':replay(b,guard,'serial')
            else:full(b,guard)
        check(guard)
    write(HERE/'sessions'/f'{args.mode}_{time.time_ns()}.json',dict(mode=args.mode,seconds=time.perf_counter()-start,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),complete=True,
        interference=guard.events,runner_sha256=sha(__file__)))


if __name__=='__main__':main()
