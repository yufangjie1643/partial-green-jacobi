"""Fresh-parent ASG experiment: exclusive exhaustive labels and complete latency."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from project_paths import ROOT

sys.path.insert(0,str(ROOT/'experiments/global_alpha8_dataset'))
from common import read,save_json,save_npz,digest,array_hash,object_hash,output_path,lock,FAMILIES
from dataset import make_manifest
from label import assert_gpu_exclusive

SEED=20261030
LOAD=np.array([1.,1.,0.,0.,0.,np.sqrt(2)])
GRID=np.linspace(0,1,101)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def raw_shape(g,n):
    p=g['parameters'];family=g['family'];t=np.arange(n,dtype=np.float64)/n
    xyz=(t[None,None,:],t[None,:,None],t[:,None,None])
    waves=[np.sin(2*np.pi*p['frequencies'][i]*v+p['phases'][i]) for i,v in enumerate(xyz)]
    smooth=sum(w*v for w,v in zip(p['weights'],waves))
    if family=='smooth': q=smooth
    elif family=='cosine':q=((.5+.5*waves[0])*(.5+.5*waves[1])*(.5+.5*waves[2]))**p['cosine_power']
    elif family=='ramp':
        coordinate=np.mod(sum(a*v for a,v in zip(p['direction'],xyz))+p['phases'][0]/(2*np.pi),1)**p['coordinate_warp']
        q=(np.minimum(np.floor(coordinate*p['levels']),p['levels']-1)/(p['levels']-1))**p['profile_power']
    elif family in ('sharp','transition'):
        threshold=g.get('threshold',float(np.quantile(smooth,1-p['volume_fraction'])))
        level=smooth-threshold
        q=(level>0).astype(float) if family=='sharp' else .5*(1+np.tanh(level/(2*np.pi*p['transition_width_parameter'])))
    elif family=='mixed':
        product=waves[0]*waves[1]*waves[2]
        threshold=g.get('threshold',float(np.quantile(product,1-p['volume_fraction'])))
        q=p['mixture']*(product>threshold)+(1-p['mixture'])*.5*(smooth+1)
    else:raise ValueError(family)
    return np.broadcast_to(q,(n,n,n)).copy(),smooth,waves

def parent(g):
    g=dict(g);q,smooth,waves=raw_shape(g,128)
    if g['family'] in ('sharp','transition','mixed'):
        values=waves[0]*waves[1]*waves[2] if g['family']=='mixed' else smooth
        g['threshold']=float(np.quantile(values,1-g['parameters']['volume_fraction']))
    g['normalization']=[float(q.min()),float(q.max())]
    return g

def field(g,n,contrast):
    q=raw_shape(g,n)[0];lo,hi=g['normalization'];q=np.clip((q-lo)/(hi-lo),0,1)
    return np.ascontiguousarray(np.exp(np.log(contrast)*q))

def plan(out):
    import dataset
    old=read(ROOT/'runs/global_alpha8_dataset_parallel64_v3/manifest.json')
    dataset.SEED=SEED
    m=make_manifest();geometries=[]
    for family in FAMILIES:
        for split,count in [('train',10),('validation',2),('test',10)]:
            candidates=[g for g in m['geometries'] if g['family']==family and g['split']==split][:count]
            for g in candidates:
                g=dict(g);g['geometry_id']='asg2_'+g['geometry_id'];geometries.append(parent(g))
    assert not ({g['shape_signature'] for g in geometries}&{g['shape_signature'] for g in old['geometries']})
    samples=[]
    for g in geometries:
        for c in (100,1000,10000):
            samples.append(dict(sample_id=f"{g['geometry_id']}_c{c}_n64",geometry_id=g['geometry_id'],split=g['split'],family=g['family'],n=64,contrast=c))
    manifest=dict(seed=SEED,geometries=geometries,samples=samples,load=LOAD.tolist(),reference_grid=128,
                  field_definition='Continuous formulas with phase thresholds and min/max fixed on 128^3; same clipping and normalization at every target grid.',
                  splits=dict(train_parents=60,validation_parents=12,test_parents=60,train_fields=180,validation_fields=36,test_fields=180),
                  old_shape_signatures_disjoint=True,selection_scope='single combined strain load; six basis loads are subsequent transfer tests, not training labels',
                  label_grid=GRID.tolist(),label_repeats=3,tolerance=1e-8,zero_initial_guess=True,
                  planned_transfer='First test parent in each family, all three contrasts, n=32,128,256 and six basis loads at n=64; frozen before testing.',
                  code_sha256=digest(__file__),original_manifest_sha256=digest(ROOT/'runs/global_alpha8_dataset_parallel64_v3/manifest.json'))
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:save_json(out/'manifest.json',manifest)
    for row in tqdm(samples,desc='Generate fresh fields',unit='field',mininterval=3):
        p=out/'fields'/f"{row['sample_id']}.npz"
        if p.exists():continue
        g=next(g for g in geometries if g['geometry_id']==row['geometry_id']);rho=field(g,64,row['contrast'])
        save_npz(p,rho=rho)
        save_json(p.with_suffix('.json'),dict(**row,rho_sha256=array_hash(rho),rho_min=float(rho.min()),rho_max=float(rho.max()),archive_sha256=digest(p)))
    return manifest

class Runner:
    def __init__(self):
        assert_gpu_exclusive();torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.a=load_module('asg_solver',ROOT/'experiments/adaptive_green_scaling/study.py');self.a.s.setup()
        self.features=load_module('asg_features',ROOT/'experiments/global_alpha_stats/features.py')
        self.extract=torch.compile(self.features.extract,fullgraph=True)
        self.models={};self.warmed=set()
        self.source_hashes=self.a.s.old.fingerprint(self.a.s.comp)
        for p in [Path(__file__),Path(self.a.__file__),Path(self.a.s.__file__),Path(self.features.__file__)]:self.source_hashes[str(p)]=digest(p)

    def warm(self,rho):
        n=rho.shape[0]
        if n in self.warmed:return
        for a in (0.,.5,1.):self.measure(rho,a)
        with torch.inference_mode():self.extract(torch.tensor(rho,device='cuda',dtype=torch.float64)).cpu()
        gc.collect();torch.cuda.synchronize();self.warmed.add(n)

    def measure(self,rho,selector,load=LOAD,diagnostics=False):
        gc.collect(0);torch.cuda.synchronize();previous=gc.isenabled();gc.disable()
        try:
            with torch.inference_mode():
                start=time.perf_counter();op=self.a.build(rho);rhs=op.compute_rhs(load)
                norm=float(torch.linalg.vector_norm(rhs));loaded=time.perf_counter()
                if callable(selector):alpha=float(selector(op));torch.cuda.synchronize();chosen=time.perf_counter()
                else:alpha=float(selector);chosen=loaded
                assert 0<=alpha<=1 and np.isfinite(alpha)
                self.a.set_alpha(op,alpha)
                if norm==0:u=torch.zeros_like(rhs);info=dict(iterations=0,restarts=0,relative_residual=0.,converged=True)
                else:u,info=self.a.certified(op,rhs,1e-8,x0=None)
                torch.cuda.synchronize();done=time.perf_counter()
                result=dict(alpha=alpha,time_s=done-start,setup_s=loaded-start,selection_s=chosen-loaded,solve_s=done-chosen,rhs_norm=norm,**info)
                if diagnostics:
                    result['average_stress']=op.compute_avg_stress(u,load).reshape(-1).cpu().tolist()
                    result['displacement_norm']=float(torch.linalg.vector_norm(u))
                assert np.isfinite(result['relative_residual']),result
                return result
        finally:
            if previous:gc.enable()

def load_rho(out,row):
    path=out/'fields'/f"{row['sample_id']}.npz";meta=read(path.with_suffix('.json'))
    assert digest(path)==meta['archive_sha256']
    with np.load(path) as z:rho=z['rho'].copy()
    assert array_hash(rho)==meta['rho_sha256'];return rho

def setup_record(out,runner):
    info=dict(source_hashes=runner.source_hashes,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
              timing='Raw host rho -> fresh H2D/operator/RHS -> selection -> fresh scaling -> zero-start solve -> independent true residual; stress excluded',
              excluded='Generation/disk I/O, compilation/warmup, weights/reference-symbol loading, garbage collection, stress and writing',
              precision='FP64 operator, complex128 FFT, FP32 features/model; TF32 off',cap=4000,check_stride=4)
    if (out/'solver_protocol.json').exists():assert read(out/'solver_protocol.json')==info
    else:
        save_json(out/'solver_protocol.json',info)
        for p in runner.source_hashes:
            dest=out/'source'/Path(p).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

def scan(out,m,runner,split,smoke=False):
    rows=[r for r in m['samples'] if r['split']==split]
    if smoke:rows=[next(r for r in rows if r['family']==f and r['contrast']==1000) for f in FAMILIES]
    directory=out/('smoke' if smoke else 'labels')
    repeats=1 if smoke else 3;alphas=[0,.25,.5,.75,1.] if smoke else GRID.tolist()
    total=len(rows)*len(alphas)*repeats;done=0;start=time.time()
    with tqdm(total=total,desc=f'{split} exclusive scan',unit='solve',mininterval=3) as bar:
        for row in rows:
            sid=row['sample_id'];path=directory/f'{sid}.json'
            if path.exists():done+=len(alphas)*repeats;bar.update(len(alphas)*repeats);continue
            rho=load_rho(out,row);runner.warm(rho);gc.collect();assert_gpu_exclusive()
            feat=runner.extract(torch.tensor(rho,device='cuda',dtype=torch.float64)).cpu().numpy().ravel().tolist()
            records=[];rng=np.random.default_rng(int(hashlib.sha256(sid.encode()).hexdigest()[:8],16))
            for repeat in range(repeats):
                for j in rng.permutation(len(alphas)):
                    result=runner.measure(rho,alphas[j]);records.append(dict(repeat=repeat,alpha_index=int(j),**result))
                    done+=1;bar.update()
                assert_gpu_exclusive();gc.collect()
            costs=np.array([[r['time_s'] for r in sorted([r for r in records if r['repeat']==i],key=lambda r:r['alpha_index'])] for i in range(repeats)])
            eligible=np.array([all(r['converged'] for r in records if r['alpha_index']==j) for j in range(len(alphas))])
            if not eligible.any():raise RuntimeError('No converged exponent: '+sid)
            mean=costs.mean(0);best=int(np.argmin(np.where(eligible,mean,np.inf)))
            save_json(path,dict(**row,features=feat,records=records,alphas=alphas,eligible=eligible.tolist(),mean_time_s=mean.tolist(),
                               std_time_s=(costs.std(0,ddof=1) if repeats>1 else np.zeros(len(alphas))).tolist(),alpha_star=alphas[best],
                               protocol_sha256=digest(out/'solver_protocol.json'),manifest_sha256=digest(out/'manifest.json')))
            save_json(out/'status.json',dict(stage='smoke' if smoke else 'label',split=split,completed=done,total=total,elapsed_s=time.time()-start,last_sample=sid),replace=True)
    save_json(out/('smoke_complete.json' if smoke else f'labels_{split}_complete.json'),dict(fields=len(rows),solves=total,elapsed_s=time.time()-start),replace=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['plan','smoke','label']);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',default='train',choices=['train','validation','test']);args=p.parse_args()
    out=output_path(args.out)
    with lock(out):
        if args.mode=='plan':plan(out);return
        m=read(out/'manifest.json');runner=Runner();setup_record(out,runner)
        if args.split=='test' and not (out/'MODELS_FROZEN.json').exists():raise RuntimeError('Freeze all models before inspecting test labels')
        scan(out,m,runner,args.split,args.mode=='smoke')

if __name__=='__main__':main()
