"""Summarize certified matched replays; generate the manuscript comparison figure.

Figure contract: isolate native partial-scaling benefit from implementation cost.
Panels: mean state time, state dependence, iteration count, native cost breakdown.
Python/Matplotlib; 183 mm width; vector PDF/SVG and 300 dpi PNG. Error bars are
SD of three repeat means over the same four states, not independent-design CIs.
"""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
PAPER=ROOT/'manuscripts/periodic_cell_partial_scaling_20261104'
rows=[]
for n in (32,64):
    for k in (0,49,62,179):
        data=json.loads((OUT/f'replay_n{n}_k{k:03d}.json').read_text());rows.extend(data['rows'])
assert len(rows)==264
assert all(r['converged'] and r['relative_J_error']<1e-7 for r in rows)
assert all(q['converged'] and q['absolute_residual']<=q['threshold'] for r in rows for q in r['loads'])

def csvwrite(path, rs):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

summaries=[];state_summary=[];load_rows=[]
for (n,m,mode) in sorted({(r['n'],r['method'],r['load_mode']) for r in rows}):
    rr=[r for r in rows if (r['n'],r['method'],r['load_mode'])==(n,m,mode)]
    repeats=[np.mean([r['total_s'] for r in rr if r['repeat']==rep]) for rep in range(3)]
    item=dict(n=n,method=m,load_mode=mode,records=len(rr),mean_total_s=np.mean(repeats),
        repeat_mean_std_s=np.std(repeats,ddof=1),mean_iterations=np.mean([r['iterations'] for r in rr]),
        mean_material_precond_s=np.mean([r['material_precond_s'] for r in rr]),
        mean_rhs_bridge_initial_check_s=np.mean([r.get('rhs_bridge_initial_check_s',0) for r in rr]),
        mean_solve_s=np.mean([r['solve_s'] for r in rr]),
        mean_certificate_s=np.mean([r.get('certificate_s',0) for r in rr]),
        mean_response_s=np.mean([r['response_s'] for r in rr]))
    summaries.append(item)
    for k in (0,49,62,179):
        ss=[r for r in rr if r['step']==k]
        state_summary.append(dict(n=n,method=m,load_mode=mode,evaluation=k+1,
            mean_total_s=np.mean([r['total_s'] for r in ss]),std_total_s=np.std([r['total_s'] for r in ss],ddof=1),
            mean_iterations=np.mean([r['iterations'] for r in ss])))
for r in rows:
    for j,q in enumerate(r['loads']):
        load_rows.append(dict(n=r['n'],evaluation=r['step']+1,method=r['method'],load_mode=r['load_mode'],
            repeat=r['repeat'],load=j,iterations=q['iterations'],relative_residual=q['relative_residual'],
            absolute_residual=q['absolute_residual'],threshold=q['threshold'],converged=q['converged']))
csvwrite(OUT/'summary.csv',summaries);csvwrite(OUT/'states.csv',state_summary);csvwrite(OUT/'loads.csv',load_rows)
lookup={(r['n'],r['method'],r['load_mode']):r for r in summaries}
q=lambda n,m,key='mean_total_s':lookup[n,m,'serial'][key]
claims={str(n):dict(native_partial_reduction_vs_gj=1-q(n,'mutopopt-partial-0.5')/q(n,'mutopopt-gj'),
    native_partial_iteration_reduction=1-q(n,'mutopopt-partial-0.5','mean_iterations')/q(n,'mutopopt-gj','mean_iterations'),
    ours_partial_speedup_vs_native_gj=q(n,'mutopopt-gj')/q(n,'0.5'),
    ours_partial_speedup_vs_native_partial=q(n,'mutopopt-partial-0.5')/q(n,'0.5')) for n in (32,64)}
claims.update(records=len(rows),load_cases=len(load_rows),all_passed=True,
    max_relative_J_error=max(r['relative_J_error'] for r in rows),
    max_relative_residual=max(r['relative_residual'] for r in load_rows))
(OUT/'summary.json').write_text(json.dumps(claims,indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
    'axes.labelsize':7,'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300})
methods=['mutopopt-green','mutopopt-gj','mutopopt-partial-0.5','asg','0.5']
labels=['muTopOpt Green','muTopOpt GJ','muTopOpt + partial 0.5','Our frozen ASG','Our partial 0.5']
colors=['#abb6c3','#647b97','#9cbfae','#c8a87f','#307c76']
fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
ax=axs[0,0];x=np.arange(2);width=.15
for i,(m,label,c) in enumerate(zip(methods,labels,colors)):
    ax.bar(x+(i-2)*width,[q(n,m) for n in (32,64)],width,color=c,label=label,
        yerr=[lookup[n,m,'serial']['repeat_mean_std_s'] for n in (32,64)],capsize=2)
ax.set_xticks(x,['32³','64³']);ax.set_ylabel('Three-load state time (s)');ax.set_yscale('log')
ax.set_title('a  Matched native and project implementations',loc='left',fontweight='bold')
ax.legend(loc='upper left',ncol=1,fontsize=5.8)
ax=axs[0,1]
for m,label,c,mark in zip(methods[1:],labels[1:],colors[1:],['o','s','^','D']):
    rr=[r for r in state_summary if r['n']==64 and r['method']==m and r['load_mode']=='serial']
    ax.plot([r['evaluation'] for r in rr],[r['mean_total_s'] for r in rr],marker=mark,color=c,lw=1,label=label,ms=3)
ax.set_xticks([1,50,63,180]);ax.set_xlabel('Saved design evaluation');ax.set_ylabel('Three-load state time (s)')
ax.set_yscale('log');ax.set_title('b  State dependence at 64³',loc='left',fontweight='bold')
ax=axs[1,0]
for i,(m,c) in enumerate(zip(methods,colors)):
    ax.bar(i,q(64,m,'mean_iterations'),color=c)
    ax.text(i,q(64,m,'mean_iterations')+20,f"{q(64,m,'mean_iterations'):.0f}",ha='center',fontsize=6)
ax.set_xticks(range(5),['Native\nGreen','Native\nGJ','Native\npartial','Our\nASG','Our\npartial'])
ax.set_ylabel('Sum of iterations across three loads')
ax.set_ylim(0,max(q(64,m,'mean_iterations') for m in methods)*1.18)
ax.set_title('c  Iterations at 64³',loc='left',fontweight='bold')
ax=axs[1,1];bottom=np.zeros(3)
phases=[('mean_material_precond_s','Material / preconditioner','#9bb6cd'),
    ('mean_rhs_bridge_initial_check_s','RHS / layout / initial check','#dfc6a2'),
    ('mean_solve_s','Native PCG','#647b97'),('mean_certificate_s','Independent certificate','#9cbfae'),
    ('mean_response_s','Response','#b8a9c9')]
for key,label,c in phases:
    v=np.array([q(64,m,key) for m in methods[:3]])
    ax.barh(np.arange(3),v,left=bottom,color=c,label=label);bottom+=v
ax.set_yticks(range(3),['Native Green','Native GJ','Native partial']);ax.invert_yaxis()
ax.set_xlabel('Mean phase time (s)');ax.set_title('d  Native implementation cost at 64³',loc='left',fontweight='bold')
ax.legend(fontsize=5.6,loc='lower right')
for ext in ('png','pdf','svg'):fig.savefig(OUT/f'fig10_mutopopt.{ext}',bbox_inches='tight')
plt.close(fig)
for name in ('summary.csv','states.csv','loads.csv','summary.json','validation.json'):
    shutil.copy2(OUT/name,PAPER/'data'/('mutopopt_'+name))
for ext in ('png','pdf','svg'):shutil.copy2(OUT/f'fig10_mutopopt.{ext}',PAPER/'assets'/f'fig10_mutopopt.{ext}')
print(json.dumps(claims,indent=2))
