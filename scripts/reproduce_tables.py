"""Recompute calibration and the principal timing comparisons on CPU."""
from pathlib import Path
import importlib.util
import json
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def reproduce():
    spec=importlib.util.spec_from_file_location('frozen_fit',ROOT/'exp/asg_density_selector_20261105/fit.py')
    fit=importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)
    states,costs=fit.load()
    fitted=fit.fit(states,costs,'eta')
    frozen=json.loads((ROOT/'exp/asg_density_selector_20261105/MODELS_FROZEN.json').read_text())['models']['eta']
    assert fitted['anchors']==frozen['anchors']
    assert np.isclose(fitted['train_loss'],frozen['train_loss'],rtol=1e-12,atol=1e-14)
    implementations=pd.read_csv(ROOT/'data/study1/implementation_summary.csv').set_index('method')
    attempts=pd.read_csv(ROOT/'data/study2/replay_parallel_attempts.csv')
    good=attempts[attempts.status.eq('complete')]
    paired=good.pivot(index=['trajectory','iteration','repeat'],columns='method',values='total_s')
    ratios_compact=(paired.eta/paired.compact8).groupby(level='trajectory').mean()
    ratios_gj=(paired.eta/paired.gj).groupby(level='trajectory').mean()
    summary=pd.read_csv(ROOT/'data/study2/replay_parallel_summary.csv').set_index('method')
    assert len(ratios_compact)==18 and len(paired)==270
    assert np.isclose(ratios_compact.mean(),summary.loc['eta','mean_ratio_compact8'])
    native=pd.read_csv(ROOT/'data/study1/mutopopt_summary.csv')
    native=native[native.method.str.startswith('mutopopt')].pivot(index='n',columns='method',values='mean_total_s')
    out=dict(calibration=dict(states=len(states),trajectories=int(states.trajectory.nunique()),**fitted),
        study1_baseline_same_budget_savings={m:1-implementations.loc['0.5','baseline_s']/implementations.loc[m,'baseline_s'] for m in ['green','gj']},
        native_scaling_ablation_savings={str(n):1-row['mutopopt-partial-0.5']/row['mutopopt-gj'] for n,row in native.iterrows()},
        study2=dict(cases=len(ratios_compact),mean_saving_vs_gj=1-ratios_gj.mean(),
            mean_saving_vs_compact8=1-ratios_compact.mean(),
            worst_regression_vs_compact8=ratios_compact.max()-1,
            failed_eta_attempts=int(summary.loc['eta','failures']),failed_green_attempts=int(summary.loc['green','failures'])))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=reproduce();text=json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n')
    print(text)
