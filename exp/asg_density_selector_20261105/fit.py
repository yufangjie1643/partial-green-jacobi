"""Grouped, cost-sensitive fitting using only archived development labels."""
from pathlib import Path
import hashlib
import itertools
import json
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'asg_alpha_landscape_20261105'
GRID = np.arange(17)/16


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantize(alpha):
    return np.clip(np.floor(np.clip(alpha, 0, 1)*16+.5).astype(int), 0, 16)


def load():
    data = pd.read_csv(SOURCE/'report/states.csv')
    data = data[(data['mode']=='warm') & data.balanced].copy().reset_index(drop=True)
    costs = pd.read_csv(SOURCE/'report/costs.csv')
    costs = costs[(costs['mode']=='warm') & (costs.phase=='coarse') & (costs.method!='compact8')].copy()
    costs.loc[~costs.eligible | (costs.successful!=2), 'mean_ms'] = np.nan
    m = costs.pivot(index=['trajectory','iteration'], columns='alpha', values='mean_ms')
    m = m.reindex(index=pd.MultiIndex.from_frame(data[['trajectory','iteration']]), columns=GRID)
    cost = m.to_numpy()
    assert len(data)==172 and len(data.trajectory.unique())==18
    assert np.isfinite(data.gj_ms).all() and np.isfinite(data.compact_ms).all()
    data['g'] = np.clip(data.mean_grayness/4/(data.mean_density*(1-data.mean_density)+1e-12),0,1)
    data['s'] = np.clip(data.log10_contrast,0,9)
    data['eta_index'] = np.array([np.argmin(abs(np.array([4,6,9])+np.log10(v))) for v in data.eta])
    return data, cost


def fit(data, cost, kind):
    s=data.s.to_numpy(); g=data.g.to_numpy()
    groups=[np.where(data.trajectory.to_numpy()==k)[0] for k in sorted(data.trajectory.unique())]
    gref=float(np.mean([g[ix].mean() for ix in groups]))
    y=cost/data.gj_ms.to_numpy()[:,None]
    y=np.where(np.isfinite(y),y,100.)
    triples=list(itertools.product(range(17),repeat=3)) if kind=='eta' else list(itertools.combinations_with_replacement(range(17),3))
    best=None
    gammas=np.round(np.arange(21)*.025,3) if kind=='gray' else [0.]
    for gamma in gammas:
        for start in range(0,len(triples),256):
            values=np.array(triples[start:start+256])/16
            if kind=='eta':
                alpha=values[:,data.eta_index.to_numpy()]
            else:
                alpha=np.stack([np.interp(s,[0,4,6,9],[0,*v]) for v in values])
                alpha+=gamma*(gref-g)
            ix=quantize(alpha)
            selected=y[np.arange(len(data))[None,:],ix]
            percase=np.stack([selected[:,ids].mean(1) for ids in groups],axis=1)
            loss=.8*percase.mean(1)+.2*percase.max(1)
            order=np.argmin(loss);candidate=(float(loss[order]),float(gamma),*values[order])
            if best is None or candidate<best:
                best=candidate
    return dict(kind=kind,anchors=list(best[2:]),gamma=best[1],g_ref=gref,train_loss=best[0])


def predict(data,model):
    if model['kind']=='eta':alpha=np.array(model['anchors'])[data.eta_index.to_numpy()]
    else:
        alpha=np.interp(data.s.to_numpy(),[0,4,6,9],[0,*model['anchors']])
        alpha+=model['gamma']*(model['g_ref']-data.g.to_numpy())
    return quantize(alpha)


def main():
    started=time.perf_counter();data,cost=load()
    sources=[SOURCE/'report/states.csv',SOURCE/'report/costs.csv',HERE.parent/'asg_density_selector_plan_20261105/plan.json']
    registration=dict(created_unix=time.time(),source_hashes={str(p):sha(p) for p in sources},
        gate='gray requires >=1% mean per-case savings vs contrast in BOTH grouped validation schemes, zero additional failed candidates',
        timing='archived costs screen models; real integrated CUDA timing follows before claiming online savings',
        quantizer='floor(16*alpha+0.5), clipped 0..16; exact half ties round upward',
        loss='.8 mean(case mean cost/GJ) + .2 max(case mean cost/GJ), failed cost/GJ=100',
        figure_contract=dict(backend='existing Python/Matplotlib workflow',archetype='quantitative grid',
            conclusion='Select the simplest method whose gain survives grouped validation and real online timing',
            panels=['grouped validation','paired replay and overhead','full workflow if gate passed'],
            exports=['PNG','SVG','PDF','CSV'],risk='development data are already seen; no random step splits'))
    if not (HERE/'execution_protocol.json').exists():write(HERE/'execution_protocol.json',registration)
    else:assert json.loads((HERE/'execution_protocol.json').read_text())['source_hashes']==registration['source_hashes']
    records=[];models=[]
    for split,col in [('family','family'),('objective','objective')]:
        for held in sorted(data[col].unique()):
            test=(data[col]==held).to_numpy(); train=~test; dev=data.loc[train].reset_index(drop=True)
            for method in ['eta','contrast','gray']:
                model=fit(dev,cost[train],method);models.append(dict(split=split,held=str(held),**model))
                print('FIT',split,held,method,model,flush=True)
                choice=predict(data.loc[test],model)
                for i,j in zip(np.where(test)[0],choice):
                    row=data.iloc[i];valid=bool(np.isfinite(cost[i,j]));value=float(cost[i,j]) if valid else float(100*row.gj_ms)
                    records.append(dict(split=split,held=str(held),method=method,trajectory=row.trajectory,
                        iteration=int(row.iteration),alpha=j/16,cost_ms=value,failed=not valid,
                        ratio_compact8=value/row.compact_ms,ratio_oracle=value/row.coarse_ms,ratio_gj=value/row.gj_ms))
    pred=pd.DataFrame(records);pred.to_csv(HERE/'cv_predictions.csv',index=False)
    case=pred.groupby(['split','method','trajectory']).agg(ratio_compact8=('ratio_compact8','mean'),
        ratio_oracle=('ratio_oracle','mean'),ratio_gj=('ratio_gj','mean'),failures=('failed','sum')).reset_index()
    case.to_csv(HERE/'cv_cases.csv',index=False)
    summary=case.groupby(['split','method']).agg(mean_ratio_compact8=('ratio_compact8','mean'),
        mean_regret=('ratio_oracle','mean'),worst_regret=('ratio_oracle','max'),failures=('failures','sum')).reset_index()
    summary.to_csv(HERE/'cv_summary.csv',index=False)
    gates={}
    for split in ['family','objective']:
        p=pred[pred.split==split].pivot(index=['trajectory','iteration'],columns='method',values='cost_ms')
        ratios=(p.gray/p.contrast).groupby(level='trajectory').mean()
        saving=float(1-ratios.mean());fail=int(pred[(pred.split==split)&(pred.method=='gray')].failed.sum())
        gates[split]=dict(gray_saving_vs_contrast=saving,gray_failures=fail,passed=saving>=.01 and fail==0)
    promote=all(x['passed'] for x in gates.values())
    frozen={m:fit(data,cost,m) for m in ['eta','contrast','gray']}
    result=dict(created_unix=time.time(),data_states=len(data),trajectories=data.trajectory.nunique(),
        grouped_gates=gates,gray_promoted_by_offline=promote,models=frozen,cv_models=models,
        fit_seconds=time.perf_counter()-started,source_hashes=registration['source_hashes'],
        fit_sha256=sha(__file__),test_data_opened=False)
    write(HERE/'MODELS_FROZEN.json',result)
    print(summary.to_string(index=False));print(json.dumps(gates,indent=2));print('FROZEN',frozen,'seconds',result['fit_seconds'])


if __name__=='__main__':main()
