"""Audit matched runs and write reusable tables; never estimate missing timings."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
METHODS=['green','gj','asg','0.5','0.69','compact8']
LABELS=dict(zip(METHODS,['Green-PCG','GJ-PCG','ASG（冻结）','固定 α=0.5','固定 α=0.69','紧凑公式 8']))

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def totals(d,implementation,rows=None):
    rows=d['rows'] if rows is None else rows
    stages={k:sum(r[k] for r in rows) for k in ('filter_s','material_s','selection_precond_s','solve_s','response_s')}
    if implementation=='baseline':
        stages['outer_update_s']=sum(r['outer_s']-r['filter_s'] for r in rows)
        stages['host_and_reporting_s']=0.
    else:
        stages['outer_update_s']=sum(r['outer_update_s'] for r in rows)
        stages['host_and_reporting_s']=sum(r['host_and_reporting_s'] for r in rows)
    return stages

def main():
    data={};per_run=[];sources={};errors=[]
    for impl,folder in [('baseline',OUT/'baseline/closed'),('parallel',OUT/'resident'),('serial',OUT/'resident_serial')]:
        files=sorted(folder.glob('n32_*.json'))
        assert len(files)==36,(impl,len(files),'36 complete runs required')
        for path in files:
            d=json.loads(path.read_text());key=(impl,d['method'],d['seed'],d['repeat']);assert key not in data
            data[key]=d
            assert d['steps']==180 and len(d['rows'])==180 and not d['optimization_converged']
            assert all(row['converged'] and all(v['converged'] and v['absolute_residual']<=v['threshold'] for v in row['loads']) for row in d['rows'])
            stages=totals(d,impl)
            assert abs(sum(stages.values())+d['setup_s']-d['total_s'])<1e-8
            per_run.append(dict(implementation=impl,method=d['method'],seed=d['seed'],repeat=d['repeat'],setup_s=d['setup_s'],**stages,total_s=d['total_s']))
            sources[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary=[];per_seed=[];phase=[]
    for method in METHODS:
        means={}
        for impl in ('baseline','parallel','serial'):
            runs=[r for r in per_run if r['implementation']==impl and r['method']==method]
            means[impl]={k:float(np.mean([r[k] for r in runs])) for k in ('setup_s','filter_s','material_s','selection_precond_s','solve_s','response_s','outer_update_s','host_and_reporting_s','total_s')}
            for seed in (20261104,20261105):
                values=[r['total_s'] for r in runs if r['seed']==seed]
                per_seed.append(dict(implementation=impl,method=method,seed=seed,mean_s=float(np.mean(values)),sample_std_s=float(np.std(values,ddof=1))))
        a,b,c=means['baseline'],means['parallel'],means['serial']
        summary.append(dict(method=method,baseline_s=a['total_s'],serial_s=c['total_s'],parallel_s=b['total_s'],
                            serial_speedup=a['total_s']/c['total_s'],parallel_speedup=a['total_s']/b['total_s'],
                            baseline_solve_s=a['solve_s'],serial_solve_s=c['solve_s'],parallel_solve_s=b['solve_s'],
                            baseline_outer_s=a['outer_update_s'],serial_outer_s=c['outer_update_s'],parallel_outer_s=b['outer_update_s']))
        for beta in (1.,2.,4.,8.):
            item=dict(method=method,beta=beta,steps=25 if beta<8 else 105)
            for impl in ('baseline','parallel','serial'):
                values=[totals(d,impl,[r for r in d['rows'] if r['beta']==beta]) for key,d in data.items() if key[:2]==(impl,method)]
                for stage in ('solve_s','outer_update_s'):
                    item[f'{impl}_{stage}']=float(np.mean([v[stage] for v in values]))
            phase.append(item)
    for key,d in data.items():
        if key[0]=='baseline':continue
        old=data[('baseline',*key[1:])]
        errors.append(dict(implementation=key[0],method=key[1],seed=key[2],repeat=key[3],
            relative_final_J=abs(d['final']['J']-old['final']['J'])/abs(old['final']['J']),
            max_relative_path_J=max(abs(r['J']-s['J'])/abs(s['J']) for r,s in zip(d['rows'],old['rows'])),
            absolute_final_volume=abs(d['final']['volume']-old['final']['volume']),
            relative_density_l2=d['validation']['relative_density_l2']))
    worst={k:max(v[k] for v in errors) for k in ('relative_final_J','max_relative_path_J','absolute_final_volume','relative_density_l2')}
    assert worst['max_relative_path_J']<1e-6 and worst['absolute_final_volume']<1e-10 and worst['relative_density_l2']<1e-5,worst
    frozen=json.loads((OUT/'frozen_sources_at_validation.json').read_text())
    root=HERE.parents[1]
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
    checks=['outer_validation.json','green_validation.json']
    checks += [f'operator_validation_{mode}.json' for mode in ('serial','parallel')]
    checks += [f'solver_validation_{mode}_{n}.json' for mode in ('serial','parallel') for n in (32,64)]
    assert all(json.loads((OUT/name).read_text())['passed'] for name in checks)
    old=HERE.parent/'asg_candidates_20261103'
    original_hashes=json.loads((old/'COMPLETION.json').read_text())['sha256']
    assert all(hashlib.sha256((old/p).read_bytes()).hexdigest()==h for p,h in original_hashes.items())
    write_csv(OUT/'per_run.csv',per_run);write_csv(OUT/'per_seed.csv',per_seed)
    write_csv(OUT/'comparison.csv',summary);write_csv(OUT/'by_projection_stage.csv',phase)
    write_csv(OUT/'design_validation.csv',errors)
    result=dict(protocol='32^3; 6 methods x 2 seeds x 3 repetitions x 3 implementations; 180-step budget, not outer convergence',
                summary=summary,worst_validation_errors=worst,load_certificates=180*3*108,
                frozen_sources_verified=len(frozen),original_records_verified=len(original_hashes),validation_files=checks,source_sha256=sources)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    table=['| 方法 | 原实现 / s | GPU 串行 / s | GPU 并行 / s | 串行加速比 | 并行加速比 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f'| {LABELS[r["method"]]} | {r["baseline_s"]:.3f} | {r["serial_s"]:.3f} | {r["parallel_s"]:.3f} | {r["serial_speedup"]:.2f}× | {r["parallel_speedup"]:.2f}× |')
    table.extend(['',f'全部 {result["load_certificates"]:,} 个线性求解均通过原算子的真实残差核验。新旧最终目标最大相对差异 {worst["relative_final_J"]:.3e}，全轨迹目标最大相对差异 {worst["max_relative_path_J"]:.3e}，最终体积分数最大绝对差异 {worst["absolute_final_volume"]:.3e}。',
                  f'最终设计密度相对 L2 差异至多 {worst["relative_density_l2"]:.3e}（与原归档设计比较）。'])
    readme=HERE/'README.md';text=readme.read_text();begin='<!-- RESULTS_START -->';end='<!-- RESULTS_END -->'
    before,rest=text.split(begin,1);_,after=rest.split(end,1)
    readme.write_text(before+begin+'\n'+'\n'.join(table)+'\n'+end+after)
    memory=[]
    for n in (32,64):
        for mode in ('serial','parallel'):
            path=OUT/f'memory_n{n}_{mode}_c4.json'
            if path.exists():memory.append(json.loads(path.read_text()))
    if len(memory)==4:
        write_csv(OUT/'memory_comparison.csv',[dict(n=d['n'],load_mode=d['load_mode'],
            peak_allocated_MiB=d['peak_allocated_bytes']/2**20,peak_reserved_MiB=d['peak_reserved_bytes']/2**20,
            sampled_device_used_peak_MiB=d['sampled_device_used_peak_bytes']/2**20,
            allocated_increment_MiB=d['allocated_increment_bytes']/2**20) for d in memory])
        mt=['| 网格 | 载荷模式 | PyTorch 分配峰值 / MiB | PyTorch 保留峰值 / MiB（含已分配） | 设备占用采样峰值 / MiB |',
            '|---|---|---:|---:|---:|']
        for d in memory:
            mt.append(f'| {d["n"]}³ | {d["load_mode"]} | {d["peak_allocated_bytes"]/2**20:.1f} | {d["peak_reserved_bytes"]/2**20:.1f} | {d["sampled_device_used_peak_bytes"]/2**20:.1f} |')
        text=readme.read_text();before,rest=text.split('<!-- MEMORY_START -->',1);_,after=rest.split('<!-- MEMORY_END -->',1)
        readme.write_text(before+'<!-- MEMORY_START -->\n'+'\n'.join(mt)+'\n<!-- MEMORY_END -->'+after)
    manifest=dict(protocol=result['protocol'],validation_passed=True,full_optimization_runs=len(data),
                  certified_load_solves=result['load_certificates'],worst_validation_errors=worst,
                  frozen_sources_verified=len(frozen),original_records_verified=len(original_hashes),
                  timing_source_notes=dict(parallel='results/parallel_v1_sources matches the paired-run session source hashes; dual-mode parallel path validated again in solver_validation_parallel_*.json',
                                           serial='source hashes recorded in the serial session JSON'),
                  sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.rglob('*')
                          if p.is_file() and p.suffix in ('.py','.cu','.json','.csv','.npz','.md') and p.name!='COMPLETION.json'})
    (HERE/'COMPLETION.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(summary=summary,validation=worst),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
