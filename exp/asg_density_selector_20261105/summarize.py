"""Aggregate paired costs by whole trajectory, preserving failed attempts."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from fit import write

HERE=Path(__file__).resolve().parent


def summarize(folder):
    records=[]
    for p in sorted((HERE/folder).glob('*.json')):
        d=json.loads(p.read_text());s=d['state']
        for r in d['records']:
            records.append(dict(trajectory=s['trajectory'],iteration=s['iteration'],objective=s['spec']['objective'],
                eta=s['spec']['emin'],**r))
    data=pd.DataFrame(records)
    if data.empty:raise RuntimeError(f'No records in {folder}')
    if folder=='replay_parallel':
        refs=[json.loads(p.read_text()) for p in (HERE/'reference').glob('*.json')]
        expected={(s['trajectory'],s['iteration']) for d in refs for s in d['states']}
        observed=set(zip(data.trajectory,data.iteration))
        methods=json.loads((HERE/'TEST_FROZEN.json').read_text())['methods']
        assert len(refs)==18 and expected==observed,'Incomplete independent replay'
        assert set(data.method)==set(methods)
        assert (data.groupby(['trajectory','iteration','method']).size()==3).all()
        assert (data[data.method=='compact8'].status=='complete').all(),'Baseline failure invalidates paired speed comparisons'
    data['ok']=data.status=='complete'
    good=data[data.ok].copy()
    key=['trajectory','iteration','repeat']
    baseline=good[good.method=='compact8'].set_index(key).total_s
    good['ratio']=good.total_s.to_numpy()/baseline.reindex(pd.MultiIndex.from_frame(good[key])).to_numpy()
    pairs=good.pivot(index=key,columns='method',values='total_s')
    if 'eta' in pairs and 'contrast' in pairs:
        direct=(pairs.contrast/pairs.eta).groupby(level='trajectory').mean()
    else:direct=None
    cases=good.groupby(['method','trajectory']).agg(ratio_compact8=('ratio','mean'),
        mean_ms=('total_s',lambda x:1000*x.mean()),records=('total_s','size'),
        feature_ms=('feature_s',lambda x:1000*x.mean()),
        preconditioner_ms=('preconditioner_s',lambda x:1000*x.mean()),
        solve_certify_ms=('solve_certify_s',lambda x:1000*x.mean()),
        max_C_error=('relative_C_error','max')).reset_index()
    failures=data.groupby(['method','trajectory']).ok.agg(lambda x:int((~x).sum())).rename('failures').reset_index()
    cases=cases.merge(failures,how='outer').fillna({'records':0})
    summary=cases.groupby('method').agg(mean_ratio_compact8=('ratio_compact8','mean'),
        worst_ratio_compact8=('ratio_compact8','max'),wins=('ratio_compact8',lambda x:int((x<1).sum())),
        failures=('failures','sum'),cases=('trajectory','size'),mean_feature_ms=('feature_ms','mean'),
        mean_preconditioner_ms=('preconditioner_ms','mean'),mean_solve_certify_ms=('solve_certify_ms','mean'),
        max_C_error=('max_C_error','max')).reset_index()
    # A failed method must never receive an aggregate speedup based on its surviving states.
    summary.loc[summary.failures>0,['mean_ratio_compact8','worst_ratio_compact8']]=np.nan
    out=HERE/'report';out.mkdir(exist_ok=True)
    flat=data.drop(columns=['loads'],errors='ignore');flat.to_csv(out/f'{folder}_attempts.csv',index=False)
    good.drop(columns=['loads'],errors='ignore').to_csv(out/f'{folder}_successful.csv',index=False)
    cases.to_csv(out/f'{folder}_cases.csv',index=False);summary.to_csv(out/f'{folder}_summary.csv',index=False)
    payload=dict(folder=folder,states=len(data[key[:2]].drop_duplicates()),attempt_groups=len(data),
        successful_groups=int(data.ok.sum()),failed_groups=int((~data.ok).sum()),certified_load_solves=int(6*data.ok.sum()),
        mean_contrast_over_eta=float(direct.mean()) if direct is not None else None,
        summary=json.loads(summary.to_json(orient='records')),
        max_C_error=float(good.relative_C_error.max()),
        max_absolute_threshold_ratio=float(max(c['absolute_residual']/c['threshold'] for r in records if r['status']=='complete' for c in r['loads'])))
    write(out/f'{folder}_audit.json',payload)
    if folder=='replay_parallel':
        gates={}
        for method in ['contrast','eta']:
            q=summary.set_index('method').loc[method]
            gates[method]=dict(saving=float(1-q.mean_ratio_compact8) if np.isfinite(q.mean_ratio_compact8) else None,
                wins=int(q.wins),worst_ratio=float(q.worst_ratio_compact8) if np.isfinite(q.worst_ratio_compact8) else None,
                failures=int(q.failures),passed=bool(q.failures==0 and q.cases==18 and q.mean_ratio_compact8<=.95 and q.wins>=12 and q.worst_ratio_compact8<=1.05))
        methods=['compact8','eta','contrast','gj'] if any(g['passed'] for g in gates.values()) else []
        write(HERE/'replay_gate.json',dict(criteria='>=5% mean savings, >=12/18 wins, worst<=5% regression, no failure',
            contrast_pass=gates['contrast']['passed'],eta_pass=gates['eta']['passed'],details=gates,full_methods=methods))
        print('GATES',json.dumps(gates,indent=2))
    print(summary.to_string(index=False));print('contrast/eta',payload['mean_contrast_over_eta'])
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();summarize(a.folder)
