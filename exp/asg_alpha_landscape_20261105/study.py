"""Frozen-trajectory alpha landscapes. Existing PCG and outer update are reused.

Every alpha receives the same material, load vectors, warm start and tolerance.
Candidate order is shuffled independently per repeat. Scans never update the
reference trajectory. Selection/search overhead is not a deployable speedup.
"""
from pathlib import Path
import argparse
import gc
import json
import math
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'asg_topopt_retrain_20261105'
sys.path.insert(0, str(OLD))
import experiment as e

ALPHAS = [i / 16 for i in range(17)]
SAMPLES = [1, 2, 5, 10, 20, 40, 80, 120, 160, 200]


def specs():
    out = []
    for family, seed in [(4, 2026110505), (3, 2026110506)]:
        for eta in [1e-4, 1e-6, 1e-9]:
            for name, objective in [('S20', 'shear'), ('B20', 'bulk'), ('A20', 'auxetic')]:
                out.append(dict(label=f'{name}_e{int(-math.log10(eta))}_i{family}', objective=objective,
                    volume=.2, emin=eta, field=family, seed=seed, n=32, rmin=3.2, beta=4., eta=.5,
                    full_scan=family == 4 and eta == 1e-4 and objective in ['shear', 'bulk']))
    return out


def register():
    protocol = dict(created_unix=time.time(), cases=specs(), alpha_grid=ALPHAS, repeats=2,
        refine='two half-grid neighbors of coarse timing winner, each repeated twice',
        state_samples=SAMPLES, always_include_last=True, trajectory_policy='GJ, warm previous displacement',
        objectives='existing shear, bulk and iteration-dependent auxetic surrogate; all target volumes 0.20',
        initialization='multiplicative exp(0.35*noise), six low-frequency modes, normalized noise, clipped [0.001,1]',
        scope='exploratory extension; seed/family 4 already seen, no independent predictive-model claim',
        rtol=1e-8, final_rtol=1e-9, max_pcg=4000, max_outer=200, outer_stop=.01,
        precision='FP64 material/operator/PCG, existing FP32 formula features',
        hardware='single GPU, six loads as two batches of three',
        timing='synchronized wall time of selection/scaling and six certified solves; excludes restore, output, tensor audit, scan search',
        failure_policy='record failed attempts and exclude them from eligible winners; never silently time a failed solve',
        cold_start='repeat base grid and compact8 from zero at final states of all nine family-4 cases',
        inference='sampled fastest alpha, 5% near-fastest candidate set, time-regret; no continuous global optimum claim',
        figure_contract=dict(backend='existing Python/Matplotlib', exports=['PNG', 'SVG', 'PDF', 'CSV'],
            claim='Assess whether alpha preferences follow contrast, trajectory stage, morphology and initial residual',
            risks=['near ties','post-selection timing noise','capped outer optimization','two coupled geometry/seed choices','grid-dependent features']),
        inherited_source_hashes={str(p): e.sha(p) for p in [OLD/'experiment.py', e.SOURCE/'reproduction.py',
            HERE.parent/'asg_gpu_resident_20261103/resident.py']})
    path = HERE/'protocol.json'
    if not path.exists(): e.write(path, protocol)
    saved = json.loads(path.read_text())
    assert saved['cases'] == specs() and saved['alpha_grid'] == ALPHAS
    for p, h in saved['inherited_source_hashes'].items(): assert e.sha(p) == h
    return saved


def initial(spec):
    n = spec['n']; x = e.initial(n, spec['field']); rng = np.random.default_rng(spec['seed'])
    z, y, t = np.meshgrid(*([np.arange(n)/n]*3), indexing='ij'); noise = np.zeros_like(x)
    for f in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]:
        noise += rng.normal()*np.cos(2*np.pi*(f[0]*z+f[1]*y+f[2]*t)+rng.uniform(0,2*np.pi))
    noise = (noise-noise.mean())/noise.std()
    return np.clip(x*np.exp(.35*noise), .001, 1.)


def field_features(a, rho):
    ell = np.log(a.astype(np.float32)); span = float(ell.max()-ell.min())
    q = (ell-ell.min())/max(span, 1e-12)
    ds = [np.abs(np.roll(q,-1,j)-q) for j in range(3)]
    A = float(np.mean([np.mean(d>1e-6) for d in ds]))
    J = float(sum(np.mean(d*d) for d in ds)/max(sum(float(np.mean(d)) for d in ds),1e-20))
    lap = sum(np.roll(q,1,j)+np.roll(q,-1,j)-2*q for j in range(3))
    H = float(np.sqrt(np.mean(lap*lap)))
    c = span/math.log(1e4)
    return dict(A=A,J=J,c=c,L=a.shape[0]*A,H=H,AJ=A*J,sqrt_AJ_c=math.sqrt(A*J/max(c,1e-30)),
        H_c=H/max(c,1e-30), log10_contrast=span/math.log(10),
        gray_fraction=float(np.mean((rho>.05)&(rho<.95))), mean_density=float(rho.mean()),
        mean_grayness=float(np.mean(4*rho*(1-rho))))


def residual_features(r, u0):
    norms=[];relative=[]
    for j in range(6):
        rhs=r.hybrid.rhs(r.op._hybrid_coeff,r.op._hybrid_index,r.macros[j],*r.op.geometry(),r.op._hybrid_indirect)
        residual=rhs-r.op.apply(u0[j:j+1])
        norms.append(residual.norm());relative.append(residual.norm()/rhs.norm().clamp_min(1e-30))
    v=torch.stack(norms+relative).cpu().numpy()
    return dict(initial_residual_norms=v[:6].tolist(), initial_relative_residuals=v[6:].tolist(),
                mean_initial_relative_residual=float(v[6:].mean()))


def collect(b, guard):
    for si, spec in enumerate(specs()):
        path=HERE/'trajectories'/(spec['label']+'.json')
        if path.exists(): continue
        r=e.create(b,spec);r.outer.set_initial(initial(spec));r.solutions.zero_()
        data=dict(spec=spec,rows=[],states=[],status='running');previous_change=1.;start=time.perf_counter()
        try:
            for k in range(1,201):
                if guard.events: raise RuntimeError('GPU interference')
                r.outer.evaluate();r.material(r.outer.a);u0=r.solutions.clone()
                r.select('gj');certs,_=r.solve();f,energy,_=r.energy(k);r.outer.step(energy)
                change=float(r.outer.change);last=change<=.01 or k==200
                row=dict(iteration=k,f=float(f),volume=float(r.outer.ph.mean()),change=change,
                    loads=certs,pcg_iterations=sum(x['iterations'] for x in certs))
                data['rows'].append(row)
                if spec['full_scan'] or k in SAMPLES or last:
                    a=r.op._material_field.cpu().numpy();rho=r.outer.ph.cpu().numpy()
                    state_path=HERE/'states'/f'{spec["label"]}_k{k:03d}.npz';state_path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(state_path,a=a,rho=rho,u0=u0.cpu().numpy())
                    state=dict(trajectory=spec['label'],iteration=k,path=str(state_path.relative_to(HERE)),spec=spec,
                        features=field_features(a,rho),previous_change=previous_change,
                        C_reference=r.effective_tensor().cpu().numpy().tolist(),**residual_features(r,u0))
                    data['states'].append(state)
                previous_change=change;r.outer.advance()
                if last: break
            r.outer.evaluate();r.material(r.outer.a);r.select('gj');certs,_=r.solve(1e-9)
            C=r.effective_tensor().cpu().numpy();S=np.linalg.inv(C)
            data.update(status='complete',steps=len(data['rows']),optimization_converged=change<=.01,
                final_C=C.tolist(),final_volume=float(r.outer.ph.mean()),final_loads=certs,
                final_nu=[float(-S[i,j]/S[i,i]) for i in range(3) for j in range(3) if i!=j])
        except RuntimeError as ex:
            data.update(status='failed',error=repr(ex),steps=len(data['rows']))
        data['seconds']=time.perf_counter()-start;e.write(path,data)
        print('TRAJECTORY',si+1,len(specs()),spec['label'],data['status'],data['steps'],len(data['states']),flush=True)
        if guard.events: raise RuntimeError('GPU interference')
        del r,u0;gc.collect();torch.cuda.empty_cache()


def candidate(r, u0, method, repeat, phase, Cref):
    r.solutions.copy_(u0);t=e.pilot.sync();row=dict(method=method,repeat=repeat,phase=phase)
    try:
        r.select('green' if method=='a0' else method[1:] if method.startswith('a') else method)
        selected=e.pilot.sync();certs,checks=r.solve();end=e.pilot.sync()
        C=r.effective_tensor().cpu().numpy();difference=float(np.linalg.norm(C-Cref)/max(np.linalg.norm(Cref),1e-30))
        assert difference<1e-6,(method,difference)
        row.update(status='complete',alpha=float(r.alpha),total_s=end-t,selection_s=selected-t,
            solve_s=end-selected,loads=certs,checks=checks,pcg_iterations=sum(x['iterations'] for x in certs),
            relative_C_error=difference)
    except RuntimeError as ex:
        row.update(status='failed',elapsed_s=e.pilot.sync()-t,error=repr(ex))
    return row


def scan_state(r,state,guard,si,mode):
    out=HERE/'scans'/mode/(Path(state['path']).stem+'.json')
    if out.exists(): return
    with np.load(HERE/state['path']) as z:
        r.material(torch.tensor(z['a'],device='cuda'));r.outer.ph.copy_(torch.tensor(z['rho'],device='cuda'))
        u0=torch.tensor(z['u0'],device='cuda') if mode=='warm' else torch.zeros_like(r.solutions)
    Cref=np.array(state['C_reference']);rows=[];methods=['a'+str(a) for a in ALPHAS]+['compact8']
    # alpha=0 uses the actual Green branch, avoiding unnecessary diagonal assembly.
    methods[0]='a0'
    for rep in range(2):
        for method in np.random.default_rng(71000+si*10+rep+(100000 if mode=='cold' else 0)).permutation(methods).tolist():
            if guard.events: raise RuntimeError('GPU interference')
            rows.append(candidate(r,u0,method,rep,'coarse',Cref))
    eligible=[]
    for method in methods[:-1]:
        good=[x for x in rows if x['method']==method and x['status']=='complete']
        if len(good)==2: eligible.append((float(np.mean([x['total_s'] for x in good])),float(good[0]['alpha'])))
    assert eligible,'No eligible alpha'
    winner=min(eligible)[1]
    refined=[a for a in [winner-1/32,winner+1/32] if 0<=a<=1 and a not in ALPHAS]
    for rep in range(2):
        for a in np.random.default_rng(81000+si*10+rep).permutation(refined).tolist():
            if guard.events: raise RuntimeError('GPU interference')
            rows.append(candidate(r,u0,'a'+str(a),rep,'refine',Cref))
    e.write(out,dict(state=state,initialization=mode,coarse_winner=winner,records=rows))


def scans(b,guard):
    trajectories=[json.loads(p.read_text()) for p in sorted((HERE/'trajectories').glob('*.json'))]
    states=[s for d in trajectories for s in d['states']]
    # Interleave trajectory samples so a partial run does not overrepresent one case.
    states.sort(key=lambda s:(s['iteration'],s['trajectory']))
    r=e.create(b,specs()[0]);start=time.perf_counter()
    for si,state in enumerate(states):
        scan_state(r,state,guard,si,'warm')
        if si%5==0 or si==len(states)-1:
            print('SCAN',si+1,len(states),state['trajectory'],state['iteration'],round(time.perf_counter()-start,1),flush=True)
    cold=[d['states'][-1] for d in trajectories if d['spec']['field']==4 and d['states']]
    for si,state in enumerate(cold):
        scan_state(r,state,guard,si,'cold');print('COLD',si+1,len(cold),state['trajectory'],flush=True)
    del r;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['collect','scan']);args=parser.parse_args()
    register();start=time.perf_counter()
    with e.pilot.Guard() as guard,torch.inference_mode():
        b=e.PaperBackend();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            e.warm(b)
            if args.mode=='collect':collect(b,guard)
            else:scans(b,guard)
        assert not guard.events
    e.write(HERE/'sessions'/f'{args.mode}_{time.time_ns()}.json',dict(mode=args.mode,complete=True,
        seconds=time.perf_counter()-start,interference=guard.events,gpu=torch.cuda.get_device_name(),
        source_sha256=e.sha(__file__)))


if __name__=='__main__':main()
