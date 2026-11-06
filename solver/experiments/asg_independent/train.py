"""Train only on fresh train/validation labels; freeze through measured validation cost."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from project_paths import ROOT
sys.path.insert(0,str(ROOT/'experiments/global_alpha8_dataset'))
from common import read,save_json,digest,output_path,lock
from label import assert_gpu_exclusive
from study import Runner,load_rho,setup_record

SEEDS=[20261030,20261031,20261032]
THRESHOLDS=[0.,.05,.1,.2,.3,.4]
VARIANTS={'mlp_mse':dict(widths=[64,32],inputs='full',loss='mse'),
          'mlp_set':dict(widths=[64,32],inputs='full',loss='set'),
          'mlp_large':dict(widths=[128,64],inputs='full',loss='mse'),
          'contrast_only':dict(widths=[64,32],inputs='contrast',loss='mse'),
          'one_point':dict(widths=[64,32],inputs='one_point',loss='mse')}

class Model(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.register_buffer('mean',torch.zeros(92));self.register_buffer('std',torch.ones(92));mask=torch.ones(92)
        if config['inputs']=='contrast':mask.zero_();mask[90:]=1
        if config['inputs']=='one_point':mask.zero_();mask[:9]=1;mask[90:]=1
        self.register_buffer('mask',mask);a,b=config['widths']
        self.net=nn.Sequential(nn.Linear(92,a),nn.SiLU(),nn.Linear(a,b),nn.SiLU(),nn.Linear(b,1),nn.Sigmoid())
    def forward(self,x):return self.net((x-self.mean)/self.std*self.mask)

def subset_features(rho,mode):
    ell=rho.float().log();lo,hi=torch.aminmax(ell);span=hi-lo
    if mode=='contrast':
        zero=span*0;v=[zero]*90+[span/math.log(10000.),span.new_tensor(1/rho.shape[0])]
    else:
        q=(ell-lo)/span.clamp_min(1e-12);mean=q.mean();zero=span*0
        v=[mean,(q-mean).square().mean()]+[(q<=t).float().mean() for t in (.01,.05,.25,.5,.75,.95,.99)]
        v += [zero]*81+[span/math.log(10000.),span.new_tensor(1/rho.shape[0])]
    return torch.stack(v).reshape(1,-1).float()

def dataset(out,manifest,split):
    rows=[r for r in manifest['samples'] if r['split']==split]
    labels=[read(out/'labels'/f"{r['sample_id']}.json") for r in rows]
    assert all(len(v['alphas'])==101 and len(v['records'])==303 for v in labels)
    x=torch.tensor([v['features'] for v in labels],dtype=torch.float32)
    y=torch.tensor([[v['alpha_star']] for v in labels],dtype=torch.float32)
    curve=np.array([v['mean_time_s'] for v in labels]);std=np.array([v['std_time_s'] for v in labels]);eligible=np.array([v['eligible'] for v in labels])
    best=np.argmin(np.where(eligible,curve,np.inf),1);minimum=curve[np.arange(len(curve)),best]
    epsilon=np.maximum(.02*minimum,1.96*std[np.arange(len(curve)),best]/np.sqrt(3))
    near=eligible&(curve<=minimum[:,None]+epsilon[:,None])
    return x,y,torch.tensor(near),curve,eligible,labels

def interpolation(curves,pred):
    pos=np.clip(pred.ravel(),0,1)*100;lo=np.floor(pos).astype(int);hi=np.minimum(lo+1,100);w=pos-lo
    return (1-w)*curves[np.arange(len(pos)),lo]+w*curves[np.arange(len(pos)),hi]

def train(out):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    m=read(out/'manifest.json');x,y,near,c,e,labels=dataset(out,m,'train');vx,vy,vnear,vc,ve,vl=dataset(out,m,'validation')
    start=time.time();candidates=[];grid=torch.linspace(0,1,101)
    protocol=dict(variants=VARIANTS,seeds=SEEDS,max_epochs=1000,min_epochs=250,patience=200,batch=60,optimizer='AdamW',lr=.002,weight_decay=1e-4,gradient_clip=5,
                  checkpoint='minimum validation interpolated mean complete time; equal field weights; invalid bracket candidates excluded',
                  near_rule='cost <= minimum + max(2% of minimum, 1.96*sample_std_at_minimum/sqrt(3)); eligible candidates only; preserve disconnected sets',
                  selection='All seeds and cutoffs receive identical isolated validation timing budget',
                  constant_inputs='Mask features with training standard deviation <=1e-8, including grid size when all training fields are 64^3; do not extrapolate untrained input weights',
                  train_fields=len(x),validation_fields=len(vx),test_loaded=False,source_sha256=digest(__file__))
    save_json(out/'training_protocol.json',protocol,replace=True)
    for variant,config in VARIANTS.items():
        for seed in SEEDS:
            name=f'{variant}_{seed}';dest=out/'models'/name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():candidates.append(read(dest/'result.json'));continue
            torch.manual_seed(seed);net=Model(config);net.mean.copy_(x.mean(0));net.std.copy_(x.std(0).clamp_min(.01))
            net.mask.mul_((x.std(0)>1e-8).float())
            opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=40,factor=.5,min_lr=1e-5)
            best=float('inf');stale=0;history=[];started=time.time()
            for epoch in tqdm(range(1,1001),desc=name,unit='epoch',mininterval=4):
                net.train();losses=[]
                for ix in torch.randperm(len(x)).split(60):
                    pred=net(x[ix]);
                    if config['loss']=='mse':loss=(pred-y[ix]).square().mean()
                    else:
                        distance=(pred-grid).abs();distance=torch.where(near[ix],distance,torch.full_like(distance,10.))
                        d=distance.min(1).values;loss=torch.where(d<.05,.5*d*d,.05*(d-.025)).mean()
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()))
                net.eval()
                with torch.inference_mode():pred=net(vx).numpy()
                predicted_cost=interpolation(vc,pred)
                positions=np.clip(pred.ravel(),0,1)*100;low=np.floor(positions).astype(int);high=np.minimum(low+1,100)
                valid=ve[np.arange(len(vx)),low]&ve[np.arange(len(vx)),high]
                # A failed scan bracket cannot win by being stopped early.
                predicted_cost=np.where(valid,predicted_cost,np.maximum(vc.max(1)*100,1.))
                score=float(np.mean(predicted_cost))
                relative=float(np.mean(predicted_cost/np.min(np.where(ve,vc,np.inf),axis=1)))
                scheduler.step(score);history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_time_s=score,validation_relative_cost=relative,validation_mae=float(np.mean(abs(pred-vy.numpy())))))
                if score<best:
                    improvement=best-score;best=score;best_relative=relative;best_epoch=epoch;state={k:v.detach().clone() for k,v in net.state_dict().items()};stale=0 if improvement>1e-7 else stale+1
                else:stale+=1
                if epoch>=250 and stale>=200:break
            net.load_state_dict(state);torch.save(dict(config=config,state_dict=state),dest/'model.pt')
            result=dict(name=name,variant=variant,seed=seed,best_epoch=best_epoch,epochs=epoch,validation_time_s=best,validation_relative_cost=best_relative,parameters=sum(p.numel() for p in net.parameters()),seconds=time.time()-started,model_sha256=digest(dest/'model.pt'))
            save_json(dest/'history.json',history);save_json(dest/'result.json',result);candidates.append(result)
    fixed=int(np.argmin(np.mean(np.where(ve,vc,np.inf),0)))/100
    save_json(out/'TRAINING_COMPLETE.json',dict(candidates=candidates,validation_best_fixed_alpha=fixed,seconds=time.time()-start,test_loaded=False),replace=True)

def load_model(out,name,device='cuda'):
    payload=torch.load(out/'models'/name/'model.pt',map_location='cpu',weights_only=True)
    net=Model(payload['config']);net.load_state_dict(payload['state_dict']);return net.to(device).eval()

def calibrate(out):
    result=read(out/'TRAINING_COMPLETE.json');m=read(out/'manifest.json');rows=[r for r in m['samples'] if r['split']=='validation']
    runner=Runner()
    expected=read(out/'solver_protocol.json')['source_hashes']
    assert all(runner.source_hashes.get(path)==value for path,value in expected.items()),'Frozen solver source changed'
    entry_hashes={p:h for p,h in runner.source_hashes.items() if p not in expected}
    runner.source_hashes=expected
    setup_record(out,runner)
    extraction={'full':runner.extract,'contrast':torch.compile(lambda rho:subset_features(rho,'contrast'),fullgraph=True),
                'one_point':torch.compile(lambda rho:subset_features(rho,'one_point'),fullgraph=True)}
    nets={r['name']:load_model(out,r['name']) for r in result['candidates']}
    choices=[(name,t) for name in nets for t in THRESHOLDS]
    allrows=[];start=time.time()
    for row in tqdm(rows,desc='Calibrate all seeds and cutoffs',unit='field',mininterval=3):
        path=out/'calibration'/f"{row['sample_id']}.json"
        if path.exists():allrows.extend(read(path));continue
        rho=load_rho(out,row);runner.warm(rho)
        # Compile every selected extraction path before timing.
        with torch.inference_mode():
            rho_gpu=torch.tensor(rho,device='cuda',dtype=torch.float64)
            for f in extraction.values():f(rho_gpu).cpu()
            del rho_gpu
        rng=np.random.default_rng(int.from_bytes(row['sample_id'].encode()[-10:],'little')%2**32);order=list(rng.permutation(len(choices)));block=[]
        for repeat,perm in enumerate([order,order[::-1]]):
            gc.collect();assert_gpu_exclusive()
            for j in perm:
                name,t=choices[j];net=nets[name];f=extraction[net.config['inputs']]
                def selector(op):
                    raw=net(f(op._material_field)).reshape(());return float(torch.where(raw<t,torch.zeros_like(raw),raw))
                v=runner.measure(rho,selector);block.append(dict(sample_id=row['sample_id'],name=name,threshold=t,repeat=repeat,**v))
            assert_gpu_exclusive()
        save_json(path,block);allrows.extend(block)
        save_json(out/'status.json',dict(stage='calibration',fields=len({r['sample_id'] for r in allrows}),total=len(rows),elapsed_s=time.time()-start),replace=True)
    selected={}
    for variant in VARIANTS:
        candidates=[]
        for name,t in choices:
            if not any(r['name']==name and r['variant']==variant for r in result['candidates']):continue
            rr=[r for r in allrows if r['name']==name and r['threshold']==t]
            if not all(r['converged'] for r in rr):continue
            candidates.append(dict(name=name,threshold=t,time_s=float(np.mean([r['time_s'] for r in rr])),model_sha256=digest(out/'models'/name/'model.pt')))
        selected[variant]=min(candidates,key=lambda r:(r['time_s'],r['name'],r['threshold']))
    primary=min(['mlp_mse','mlp_set','mlp_large'],key=lambda k:selected[k]['time_s'])
    frozen=dict(selected=selected,primary=primary,validation_best_fixed_alpha=result['validation_best_fixed_alpha'],
                source_hashes=runner.source_hashes,training_code_sha256=read(out/'training_protocol.json')['source_sha256'],
                selection_code_sha256=digest(__file__),selection_entry_hashes=entry_hashes,manifest_sha256=digest(out/'manifest.json'),
                validation_records_sha256={p.name:digest(p) for p in sorted((out/'calibration').glob('*.json'))},test_evaluated=False,
                frozen_unix_time=time.time())
    save_json(out/'MODELS_FROZEN.json',frozen)
    print(json.dumps(frozen['selected'],indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','calibrate']);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=output_path(args.out)
    with lock(out):
        if args.mode=='train':train(out)
        else:calibrate(out)

if __name__=='__main__':main()
