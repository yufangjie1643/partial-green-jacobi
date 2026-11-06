"""Sealed-parent complete solves, transfer tests, and a full inverse-elasticity workflow."""
import argparse
import gc
from pathlib import Path
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from project_paths import ROOT
sys.path.insert(0,str(ROOT/'experiments/global_alpha8_dataset'))
from common import read,save_json,digest,object_hash,output_path,lock,array_hash
from label import assert_gpu_exclusive
from study import Runner,load_rho,field,load_module,LOAD,FAMILIES
from train import load_model,subset_features

class Evaluation:
    def __init__(self,out):
        self.out=out;self.manifest=read(out/'manifest.json');self.frozen=read(out/'MODELS_FROZEN.json')
        assert self.frozen['manifest_sha256']==digest(out/'manifest.json')
        self.runner=Runner();self.nets={}
        # The production stress routine returns NumPy, whereas the diagnostic
        # wrapper accepts tensors. Normalize only this post-timer return type.
        # Install once before warming; no timed construction or solve is changed.
        probe=self.runner.a.build(np.ones((4,4,4),dtype=np.float64))
        operator_class=type(probe);stress_function=operator_class.compute_avg_stress
        def tensor_stress(op,u,load):return torch.as_tensor(stress_function(op,u,load))
        operator_class.compute_avg_stress=tensor_stress
        del probe
        self.extract={'full':self.runner.extract,'contrast':torch.compile(lambda x:subset_features(x,'contrast'),fullgraph=True),
                      'one_point':torch.compile(lambda x:subset_features(x,'one_point'),fullgraph=True)}
        for name,value in self.frozen['selected'].items():
            assert digest(out/'models'/value['name']/'model.pt')==value['model_sha256']
            self.nets[name]=load_model(out,value['name'])
        old=out/'baselines/development_mlp'
        self.oldmeta=read(old/'FROZEN.json');assert digest(old/'model.pt')==self.oldmeta['model_sha256']
        self.old=self.runner.features.StatsMLP().cuda().eval()
        self.old.load_state_dict(torch.load(old/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
        self.law=load_module('asg_compact_law',ROOT/'experiments/global_alpha_compact_law/law.py')
        self.law_features=torch.compile(self.law.extract_law,fullgraph=True)
        self.selector_hashes=dict(old_checkpoint=digest(old/'model.pt'),old_freeze=digest(old/'FROZEN.json'),
                                 compact_law=digest(Path(self.law.__file__)),training_code=digest(Path(sys.modules['train'].__file__)))
        assert self.selector_hashes['training_code']==self.frozen['selection_code_sha256']
        assert all(self.runner.source_hashes.get(p)==h for p,h in self.frozen['source_hashes'].items()),'Frozen solver source changed'
        self.methods=['alpha_0','alpha_0.25','alpha_0.5','alpha_0.75','alpha_1','validation_fixed','old_mlp',*self.nets,'compact4','compact8','quantized2','quantized5']
        self.warmed=set()

    def selector(self,method,oracle=None):
        if method.startswith('alpha_'):return float(method[6:])
        if method=='validation_fixed':return self.frozen['validation_best_fixed_alpha']
        if method=='scan_reference':return float(oracle)
        def select(op):
            if method in ('compact4','compact8'):
                p=int(method[-1]);raw=self.law.evaluate(self.law_features(op._material_field).cpu(),self.law.CONFIGS[f'law{p}'])
                return 0. if raw<({4:.25,8:.375}[p]) else raw
            if method=='old_mlp':
                pred=self.old(self.extract['full'](op._material_field)).reshape(());t=.2
            else:
                name=self.frozen['primary'] if method.startswith('quantized') else method
                net=self.nets[name];pred=net(self.extract[net.config['inputs']](op._material_field)).reshape(());t=self.frozen['selected'][name]['threshold']
            alpha=float(torch.where(pred<t,torch.zeros_like(pred),pred))
            if method=='quantized2':alpha=float(alpha>=.5)
            if method=='quantized5':alpha=float(np.floor(4*alpha+.5)/4)
            return alpha
        return select

    def warm(self,rho):
        n=rho.shape[0]
        if n in self.warmed:return
        self.runner.warm(rho)
        for method in self.methods:self.runner.measure(rho,self.selector(method))
        gc.collect();torch.cuda.synchronize();self.warmed.add(n)

    def reference(self,rho,load):
        attempts=[]
        with torch.inference_mode():
            op=self.runner.a.build(rho);rhs=op.compute_rhs(load)
            for alpha in [.5,0.,1.,.25,.75]:
                self.runner.a.set_alpha(op,alpha)
                if float(torch.linalg.vector_norm(rhs))==0:u=torch.zeros_like(rhs);info=dict(relative_residual=0.,converged=True)
                else:u,info=self.runner.a.certified(op,rhs,1e-11,x0=None)
                attempts.append(dict(alpha=alpha,**info))
                if info['converged']:break
            stress=op.compute_avg_stress(u,load).reshape(-1).cpu().numpy()
        return dict(stress=stress.tolist(),attempts=attempts,**info)

def cases(ev,mode):
    m=ev.manifest;rows=[r for r in m['samples'] if r['split']=='test']
    if mode=='test':return [dict(**r,load=LOAD.tolist(),group='independent') for r in rows]
    parents=[next(g for g in m['geometries'] if g['split']=='test' and g['family']==f) for f in FAMILIES]
    out=[]
    for g in parents:
        for contrast in [100,1000,10000]:
            if mode=='resolution':
                for n in [32,128,256]:out.append(dict(sample_id=f"{g['geometry_id']}_c{contrast}_n{n}",geometry_id=g['geometry_id'],family=g['family'],n=n,contrast=contrast,load=LOAD.tolist(),group=f'resolution_{n}'))
            if mode=='loads':
                loads=np.vstack([np.eye(6),np.array([1,-.3,.2,.4,.5,-.1])])
                for i,load in enumerate(loads):out.append(dict(sample_id=f"{g['geometry_id']}_c{contrast}_n64_load{i}",geometry_id=g['geometry_id'],family=g['family'],n=64,contrast=contrast,load=load.tolist(),group=f'load_{i}'))
        if mode=='contrast':
            for c in [100000,1000000]:out.append(dict(sample_id=f"{g['geometry_id']}_c{c}_n64",geometry_id=g['geometry_id'],family=g['family'],n=64,contrast=c,load=LOAD.tolist(),group=f'extrapolation_{c}'))
    return out

def benchmark(out,mode):
    ev=Evaluation(out);rows=cases(ev,mode);methods=ev.methods+(['scan_reference'] if mode=='test' else [])
    if mode!='test':methods=['alpha_0','alpha_0.5','alpha_1','validation_fixed','old_mlp',ev.frozen['primary'],'compact8']
    repeats=6 if mode=='test' else 2
    directory=out/f'evaluation_{mode}';directory.mkdir(exist_ok=True)
    protocol=dict(mode=mode,methods=methods,repeats=repeats,rows=rows,model_freeze_sha256=digest(out/'MODELS_FROZEN.json'),
                  selector_hashes=ev.selector_hashes,solver_source_hashes=ev.runner.source_hashes,
                  source_sha256=digest(__file__),solver_protocol_sha256=digest(out/'solver_protocol.json'),
                  setup='fresh operator, RHS and selection each solve; no reused initial displacement',
                  complete_time='raw host coefficient to certified solution; physical stress evaluated after stopping timer',
                  stress_return_adapter='NumPy stress result viewed as a CPU tensor outside the timer; numeric values unchanged',
                  physical_reference='same discrete operator, true residual <=1e-11; try alpha=.5 then 0,1,.25,.75 until certified; record unavailable physical comparison if none succeeds',
                  new_shapes_fixed_before_evaluation=True)
    if (directory/'protocol.json').exists():assert read(directory/'protocol.json')==protocol
    else:save_json(directory/'protocol.json',protocol)
    start=time.time();total=len(rows)*len(methods)*repeats;completed=0
    with tqdm(total=total,desc=mode+' complete solve',unit='solve',mininterval=4) as bar:
        for row in rows:
            sid=row['sample_id'];dest=directory/'records'/f'{sid}.json'
            if dest.exists():completed+=len(methods)*repeats;bar.update(len(methods)*repeats);continue
            if mode=='test':rho=load_rho(out,row);oracle=read(out/'labels'/f'{sid}.json')['alpha_star']
            else:
                g=next(g for g in ev.manifest['geometries'] if g['geometry_id']==row['geometry_id']);rho=field(g,row['n'],row['contrast']);oracle=None
            ev.warm(rho);assert_gpu_exclusive();reference=ev.reference(rho,np.array(row['load']));records=[]
            rng=np.random.default_rng(int.from_bytes(sid.encode()[-10:],'little')%2**32)
            orders=[]
            for _ in range(repeats//2):
                order=list(rng.permutation(methods));orders.extend([order,order[::-1]])
            for repeat,order in enumerate(orders):
                gc.collect();assert_gpu_exclusive()
                for position,method in enumerate(order):
                    v=ev.runner.measure(rho,ev.selector(method,oracle),np.array(row['load']),diagnostics=True)
                    expected=np.asarray(reference['stress']);actual=np.asarray(v['average_stress'])
                    v['relative_stress_error']=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300)) if reference['converged'] else None
                    records.append(dict(method=method,repeat=repeat,position=position,**v));completed+=1;bar.update()
                assert_gpu_exclusive()
            save_json(dest,dict(**row,actual_contrast=float(rho.max()/rho.min()),rho_sha256=array_hash(rho),oracle_alpha=oracle,
                                reference=reference,records=records,protocol_sha256=digest(directory/'protocol.json')))
            save_json(out/'status.json',dict(stage=mode,completed=completed,total=total,elapsed_s=time.time()-start,last_sample=sid),replace=True)
    save_json(directory/'COMPLETE.json',dict(cases=len(rows),solves=total,elapsed_s=time.time()-start),replace=True)

def application(out):
    ev=Evaluation(out);methods=['alpha_0','alpha_0.5','alpha_1','validation_fixed',ev.frozen['primary'],'compact8']
    parents=[next(g for g in ev.manifest['geometries'] if g['split']=='test' and g['family']==f) for f in ['smooth','mixed','sharp']]
    directory=out/'application';directory.mkdir(exist_ok=True);target_log=3.37;load=LOAD
    protocol=dict(problem='Identify coefficient contrast from a synthetic homogenized strain-energy observation on a known spatial profile',
                  selector_hashes=ev.selector_hashes,solver_source_hashes=ev.runner.source_hashes,
                  parents=[g['geometry_id'] for g in parents],target_log10_contrast=target_log,bracket=[2.,4.],
                  stopping='log10 contrast interval <= 1e-4 and relative response error <= 1e-4, max 30 bisections',
                  methods=methods,repeats=2,n=64,load=load.tolist(),solver_tolerance=1e-8,
                  timing='entire bisection: coefficient generation, each complete solve, stress/response integration, control logic; no trajectory reuse',
                  target='synthetic observation generated once at true residual <=1e-11, analogous to supplied measurement; excluded from inversion latency',
                  source_sha256=digest(__file__),model_freeze_sha256=digest(out/'MODELS_FROZEN.json'))
    if (directory/'protocol.json').exists():assert read(directory/'protocol.json')==protocol
    else:save_json(directory/'protocol.json',protocol)
    for g in tqdm(parents,desc='Complete inverse-elasticity workflow',unit='parent'):
        rho=field(g,64,10**target_log);ev.warm(rho);ref=ev.reference(rho,load);assert ref['converged'],'Synthetic observation must be certified';target=float(load@np.array(ref['stress']))
        assert target>0;order=list(np.random.default_rng(20261030).permutation(methods))
        for repeat,ordered in enumerate([order,order[::-1]]):
            for method in ordered:
                dest=directory/'records'/f"{g['geometry_id']}_{method}_{repeat}.json"
                if dest.exists():continue
                gc.collect();assert_gpu_exclusive();torch.cuda.synchronize();start=time.perf_counter();history=[]
                lo,hi=2.,4.
                for iteration in range(30):
                    mid=(lo+hi)/2;rho=field(g,64,10**mid)
                    v=ev.runner.measure(rho,ev.selector(method),load,diagnostics=True)
                    value=float(load@np.array(v['average_stress']));relative=abs(value-target)/target
                    history.append(dict(log10_contrast=mid,response=value,relative_response_error=relative,**v))
                    if value<target:lo=mid
                    else:hi=mid
                    if hi-lo<=1e-4 and relative<=1e-4:break
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;assert_gpu_exclusive()
                result=dict(geometry_id=g['geometry_id'],family=g['family'],method=method,repeat=repeat,
                            elapsed_s=elapsed,iterations=len(history),identified_log10_contrast=mid,
                            identified_contrast=10**mid,relative_contrast_error=abs(10**(mid-target_log)-1),relative_response_error=relative,
                            converged=all(v['converged'] for v in history) and hi-lo<=1e-4 and relative<=1e-4,
                            reference=ref,target_response=target,history=history,protocol_sha256=digest(directory/'protocol.json'))
                save_json(dest,result)
    save_json(directory/'COMPLETE.json',dict(parents=len(parents),methods=len(methods),repeats=2),replace=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['test','resolution','loads','contrast','application']);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=output_path(args.out)
    with lock(out):
        if args.mode=='application':application(out)
        else:benchmark(out,args.mode)

if __name__=='__main__':main()
