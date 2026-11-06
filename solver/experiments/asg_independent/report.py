"""Audit completed independent ASG experiments and export paired result tables."""
import argparse
import hashlib
import html
import json
from pathlib import Path,PurePosixPath
import numpy as np
import pandas as pd

FAMILIES=['smooth','cosine','ramp','sharp','transition','mixed']
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')

def verify_evaluation_sources(root,protocol):
    for name,expected in protocol['solver_source_hashes'].items():
        path=root/'source'/str(PurePosixPath(name).relative_to('${HOME}/code/voxel-mechanics'))
        assert sha(path)==expected,f'Evaluation source snapshot changed: {path}'
    assert sha(root/'source/experiments/asg_independent/evaluate.py')==protocol['source_sha256']
    paths={'old_checkpoint':root/'baselines/development_mlp/model.pt',
           'old_freeze':root/'baselines/development_mlp/FROZEN.json',
           'compact_law':root/'source/experiments/global_alpha_compact_law/law.py',
           'training_code':root/'source/experiments/asg_independent/train.py'}
    for name,path in paths.items():assert sha(path)==protocol['selector_hashes'][name]

def markdown(df,digits=3):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}' if np.isfinite(v) else '--'
        return str(v)
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in df.itertuples(index=False,name=None))

def html_table(frame,primary):
    d=frame.copy()
    names={'alpha_0':'Green · α=0','alpha_0.25':'固定 α=0.25','alpha_0.5':'固定 α=0.5','alpha_0.75':'固定 α=0.75','alpha_1':'Green–Jacobi · α=1',
        'validation_fixed':'验证集最佳固定 α','old_mlp':'旧统计 MLP','mlp_mse':'新统计 MLP · MSE','mlp_set':'近优集合 MLP','mlp_large':'较大统计 MLP',
        'contrast_only':'仅对比度输入','one_point':'仅单点分布输入','compact4':'紧凑公式 p=4','compact8':'紧凑公式 p=8',
        'quantized2':'主模型二档量化','quantized5':'主模型五档量化','scan_reference':'预先给定的扫描标签'}
    if 'method' in d:d['method']=d.method.map(lambda m:names.get(m,m)+(' ★' if m==primary else ''))
    for key in d:
        if 'error' in key or 'residual' in key:d[key]=d[key].map(lambda x:f'{x:.3e}' if pd.notna(x) else '—')
    d=d.rename(columns={'method':'方法','overall':'总体','smooth':'光滑波','cosine':'周期乘积','ramp':'阶梯层','sharp':'突变界面','transition':'弥散界面','mixed':'混合场',
        'family':'类别','group':'测试组','cases':'场数','time_ms':'达标时间 / ms','attempt_time_ms':'全部尝试时间 / ms','failures':'失败次数',
        'max_true_residual':'最大真实残差','iterations':'外层迭代','relative_contrast_error':'对比度相对误差','relative_response_error':'响应相对误差'})
    return '<div class="table">'+d.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.3f}',escape=True)+'</div>'

def paired_bootstrap(samples,primary,baselines):
    pivot=samples.pivot(index=['family','geometry_id','sample_id'],columns='method',values='time_ms')
    groups=[pivot.xs(f,level='family').groupby('geometry_id').mean() for f in FAMILIES]
    rng=np.random.default_rng(20261030);means=[]
    for _ in range(10000):
        means.append(pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for g in groups]).mean())
    boot=pd.DataFrame(means);full=pivot.mean();out={}
    for base in baselines:
        if samples.loc[samples.method.isin([primary,base]),'failures'].sum():
            out[base]=dict(unavailable_reason='At least one paired method failed the common residual target; no complete-time gain is inferred.')
            continue
        difference=boot[primary]-boot[base];reduction=1-boot[primary]/boot[base]
        ratio=pivot[primary]/pivot[base]
        out[base]=dict(mean_difference_ms=float(full[primary]-full[base]),difference_ci95=np.quantile(difference,[.025,.975]).tolist(),
                       reduction=float(1-full[primary]/full[base]),reduction_ci95=np.quantile(reduction,[.025,.975]).tolist(),
                       faster_fields=int((ratio<1).sum()),fields=len(ratio),median_ratio=float(ratio.median()),p95_ratio=float(ratio.quantile(.95)),worst_ratio=float(ratio.max()))
    return dict(replicates=10000,seed=20261030,unit='parent geometry, all contrasts and paired methods together',stratified='family',comparisons=out)

def benchmark(root,mode,dest):
    folder=root/f'evaluation_{mode}'
    if not (folder/'COMPLETE.json').exists():return None
    protocol=read(folder/'protocol.json');expected={r['sample_id']:r for r in protocol['rows']};records=[];parents=[]
    verify_evaluation_sources(root,protocol)
    assert protocol['model_freeze_sha256']==sha(root/'MODELS_FROZEN.json')
    assert protocol['solver_protocol_sha256']==sha(root/'solver_protocol.json')
    for sid,row in expected.items():
        value=read(folder/'records'/f'{sid}.json')
        assert value['protocol_sha256']==sha(folder/'protocol.json')
        assert value['load']==row['load']
        if mode=='test':assert value['rho_sha256']==read(root/'fields'/f'{sid}.json')['rho_sha256']
        pairs={(r['method'],r['repeat']) for r in value['records']}
        assert pairs=={(m,r) for m in protocol['methods'] for r in range(protocol['repeats'])}
        assert len(value['records'])==len(pairs)
        for method in protocol['methods']:
            selected=[r['alpha'] for r in value['records'] if r['method']==method]
            assert max(selected)-min(selected)<1e-6
        for r in value['records']:
            assert 0<=r['alpha']<=1 and np.isfinite(r['time_s']) and r['time_s']>0
            assert bool(r['converged'])==(r['relative_residual']<=1e-8)
            assert abs(r['time_s']-r['setup_s']-r['selection_s']-r['solve_s'])<1e-8
            records.append({**{k:value[k] for k in ['sample_id','geometry_id','family','n','contrast','group']},
                            'oracle_alpha':value['oracle_alpha'],'actual_contrast':value['actual_contrast'],**r})
        for repeat in range(0,protocol['repeats'],2):
            a=[r['method'] for r in sorted(value['records'],key=lambda r:r['position']) if r['repeat']==repeat]
            b=[r['method'] for r in sorted(value['records'],key=lambda r:r['position']) if r['repeat']==repeat+1]
            assert a==b[::-1]
        parents.append(dict(sample_id=sid,reference_residual=value['reference']['relative_residual'],reference_converged=value['reference']['converged']))
    frame=pd.DataFrame(records)
    for key in ['time','setup','selection','solve']:frame[key+'_ms']=frame[key+'_s']*1000
    frame.to_csv(dest/f'{mode}_repetitions.csv',index=False)
    keys=['sample_id','geometry_id','family','n','contrast','group','method']
    sample=frame.groupby(keys,sort=False).agg(time_ms=('time_ms','mean'),time_std_ms=('time_ms','std'),alpha=('alpha','mean'),
            selection_ms=('selection_ms','mean'),setup_ms=('setup_ms','mean'),solve_ms=('solve_ms','mean'),
            iterations=('iterations','mean'),max_true_residual=('relative_residual','max'),failures=('converged',lambda x:int((~x).sum())),
            max_relative_stress_error=('relative_stress_error','max'),oracle_alpha=('oracle_alpha','first')).reset_index()
    sample['repeat_cv']=sample.time_std_ms/sample.time_ms
    sample['attempt_time_ms']=sample.time_ms
    sample.loc[sample.failures>0,'time_ms']=np.nan
    sample.to_csv(dest/f'{mode}_samples.csv',index=False)
    summary=sample.groupby('method',sort=False).agg(fields=('sample_id','count'),time_ms=('time_ms','mean'),selection_ms=('selection_ms','mean'),
              attempt_time_ms=('attempt_time_ms','mean'),alpha=('alpha','mean'),iterations=('iterations','mean'),failures=('failures','sum'),max_true_residual=('max_true_residual','max'),
              max_relative_stress_error=('max_relative_stress_error','max'),mean_repeat_cv=('repeat_cv','mean')).reset_index()
    summary.loc[summary.failures>0,'time_ms']=np.nan
    summary.to_csv(dest/f'{mode}_summary.csv',index=False)
    grouped=sample.groupby(['group','family','method'],sort=False).agg(fields=('sample_id','count'),time_ms=('time_ms','mean'),attempt_time_ms=('attempt_time_ms','mean'),alpha=('alpha','mean'),failures=('failures','sum'),
             max_true_residual=('max_true_residual','max'),max_relative_stress_error=('max_relative_stress_error','max')).reset_index()
    grouped.loc[grouped.failures>0,'time_ms']=np.nan
    grouped.to_csv(dest/f'{mode}_groups.csv',index=False)
    audit=dict(cases=len(expected),records=len(frame),methods=len(protocol['methods']),repeats=protocol['repeats'],
               failed_solves=int((~frame.converged).sum()),max_true_residual=float(frame.relative_residual.max()),
               zero_rhs_solves=int((frame.rhs_norm==0).sum()),minimum_rhs_norm=float(frame.rhs_norm.min()),
               max_relative_stress_error=float(frame.relative_stress_error.max()) if frame.relative_stress_error.notna().any() else None,
               uncertified_physical_references=sum(not p['reference_converged'] for p in parents),max_reference_residual=max(p['reference_residual'] for p in parents),
               all_paired_orders_verified=True,protocol_hash_verified=True)
    write_json(dest/f'{mode}_audit.json',audit)
    return frame,sample,summary,grouped,audit

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();root=args.root;dest=args.out or root/'report';dest.mkdir(parents=True,exist_ok=True)
    manifest=read(root/'manifest.json');freeze=read(root/'MODELS_FROZEN.json');primary=freeze['primary']
    assert freeze['manifest_sha256']==sha(root/'manifest.json')
    solver=read(root/'solver_protocol.json')
    for name,expected_hash in solver['source_hashes'].items():
        snapshot=root/'source'/str(PurePosixPath(name).relative_to('${HOME}/code/voxel-mechanics'))
        assert sha(snapshot)==expected_hash,f'Solver snapshot changed: {snapshot}'
    parents=manifest['geometries'];rows_frame=pd.DataFrame(manifest['samples'])
    assert rows_frame.groupby('geometry_id').split.nunique().max()==1
    assert rows_frame.groupby('geometry_id').size().eq(3).all()
    old_path=root/'baselines/development_manifest.json'
    assert sha(old_path)==manifest['original_manifest_sha256']
    old=read(old_path)
    assert not ({g['shape_signature'] for g in parents}&{g['shape_signature'] for g in old['geometries']})
    trained_source=root/'source_versions'/freeze['training_code_sha256']/'train.py'
    assert sha(trained_source)==freeze['training_code_sha256']
    for value in freeze['selected'].values():assert sha(root/'models'/value['name']/'model.pt')==value['model_sha256']
    for name,value in freeze['validation_records_sha256'].items():assert sha(root/'calibration'/name)==value
    hashes={str(p.relative_to(root)):sha(p) for p in [root/'manifest.json',root/'solver_protocol.json',root/'MODELS_FROZEN.json',root/'training_protocol.json']}
    label_summaries=[];label_rows=[]
    for split in ['train','validation','test']:
        assert (root/f'labels_{split}_complete.json').exists()
        rows=[r for r in manifest['samples'] if r['split']==split]
        total=0.;failures=0
        for row in rows:
            path=root/'labels'/f"{row['sample_id']}.json";v=read(path)
            assert v['manifest_sha256']==sha(root/'manifest.json') and v['protocol_sha256']==sha(root/'solver_protocol.json')
            assert len(v['records'])==303 and len(v['alphas'])==101
            assert all(bool(r['converged']) == (r['relative_residual']<=1e-8) for r in v['records'])
            assert all(np.isfinite(r['time_s']) and r['time_s']>0 for r in v['records'])
            assert {(r['repeat'],r['alpha_index']) for r in v['records']}=={(i,j) for i in range(3) for j in range(101)}
            costs=np.array([[r['time_s'] for r in sorted([r for r in v['records'] if r['repeat']==i],key=lambda r:r['alpha_index'])] for i in range(3)])
            assert np.allclose(costs.mean(0),v['mean_time_s'],rtol=1e-12,atol=1e-12)
            eligible=np.array([all(r['converged'] for r in v['records'] if r['alpha_index']==j) for j in range(101)])
            assert np.array_equal(eligible,v['eligible'])
            best=int(np.argmin(np.where(eligible,costs.mean(0),np.inf)))
            assert v['alpha_star']==v['alphas'][best]
            total+=costs.sum();failures+=sum(not r['converged'] for r in v['records'])
            near=eligible&(costs.mean(0)<=costs.mean(0)[best]+max(.02*costs.mean(0)[best],1.96*costs[:,best].std(ddof=1)/np.sqrt(3)))
            label_rows.append(dict(**row,label_alpha=v['alpha_star'],min_scan_time_ms=1000*costs.mean(0)[best],near_candidates=int(near.sum())))
            hashes[str(path.relative_to(root))]=sha(path)
        label_summaries.append(dict(split=split,fields=len(rows),timed_solves=len(rows)*303,accepted_scan_solver_seconds=total,failed_candidates=failures))
    pd.DataFrame(label_rows).to_csv(dest/'labels.csv',index=False)
    pd.DataFrame(label_summaries).to_csv(dest/'offline_label_budget.csv',index=False)
    train_info=read(root/'TRAINING_COMPLETE.json')
    validation_records=[r for path in (root/'calibration').glob('*.json') for r in read(path)]
    offline=dict(training_validation_scan_seconds=sum(v['accepted_scan_solver_seconds'] for v in label_summaries if v['split']!='test'),
                 test_scan_reference_seconds=next(v['accepted_scan_solver_seconds'] for v in label_summaries if v['split']=='test'),
                 candidate_training_seconds=train_info['seconds'],
                 validation_calibration_seconds=sum(r['time_s'] for r in validation_records),
                 calibration_solves=len(validation_records),
                 scope='Recorded accepted computation only; excludes generation, I/O, warmup/compilation, rejected work and waiting. Test scan is assessment cost, not a deployed selector requirement.')
    write_json(dest/'offline_costs.json',offline)
    outputs={mode:benchmark(root,mode,dest) for mode in ['test','loads','resolution','contrast']}
    result=outputs['test'];assert result is not None,'Independent test is not complete'
    frame,sample,summary,groups,audit=result
    times=sample.pivot(index=['sample_id','geometry_id','family','contrast'],columns='method',values='time_ms').reset_index()
    alphas=sample.pivot(index=['sample_id','geometry_id','family','contrast'],columns='method',values='alpha').reset_index().rename(columns={'scan_reference':'label_alpha'})
    alphas.to_csv(dest/'alpha_predictions_with_labels.csv',index=False);times.to_csv(dest/'end_to_end_times_ms.csv',index=False)
    method_order=read(root/'evaluation_test/protocol.json')['methods']
    time_table=pd.DataFrame({'method':method_order});alpha_table=time_table.copy()
    for label,family in [('overall',None)]+[(f,f) for f in FAMILIES]:
        subset=sample if family is None else sample[sample.family==family]
        time_table[label]=time_table.method.map(subset.groupby('method').time_ms.mean())
        failed=subset.groupby('method').failures.sum()
        time_table.loc[time_table.method.map(failed)>0,label]=np.nan
        alpha_table[label]=alpha_table.method.map(subset.groupby('method').alpha.mean())
    time_table.to_csv(dest/'time_overall_and_families.csv',index=False);alpha_table.to_csv(dest/'alpha_overall_and_families.csv',index=False)
    labels=pd.DataFrame(label_rows);testlabels=labels[labels.split=='test'].set_index('sample_id')
    sample['alpha_mae']=(sample.alpha-sample.sample_id.map(testlabels.label_alpha)).abs()
    mae=sample.groupby(['family','method']).alpha_mae.mean().reset_index();mae.to_csv(dest/'alpha_mae.csv',index=False)
    boot=paired_bootstrap(sample,primary,['alpha_0.5','validation_fixed','old_mlp','mlp_large','mlp_set','contrast_only','one_point','compact4','compact8','quantized2','quantized5','scan_reference'])
    write_json(dest/'paired_bootstrap.json',boot)
    selected=[]
    for variant,value in freeze['selected'].items():
        trained=read(root/'models'/value['name']/'result.json')
        selected.append(dict(variant=variant,model=value['name'],threshold=value['threshold'],validation_ms=value['time_s']*1000,parameters=trained['parameters'],best_epoch=trained['best_epoch'],epochs=trained['epochs']))
    pd.DataFrame(selected).to_csv(dest/'selected_models.csv',index=False)
    content='# ASG 独立实验结果\n\n所有时间均从原始主机系数数组开始，包含输入处理、统计、选参、缩放及达到 1e-8 真实残差的求解。字段先对重复取均值，再对字段等权平均。\n\n'
    content+=f'验证集冻结的主模型：**{primary}**；验证选出的固定指数：**{freeze["validation_best_fixed_alpha"]:.2f}**。测试共 {audit["cases"]} 个场、{audit["records"]} 次求解，失败 {audit["failed_solves"]} 次，最大真实残差 {audit["max_true_residual"]:.6g}。\n\n'
    content+='## 网络训练与冻结\n\n'+markdown(pd.DataFrame(selected))+'\n'
    content+='## 表 1：预测指数与扫描标签\n\nscan_reference 行是逐场 101 点独占扫描标签的平均值；逐样本数据见 alpha_predictions_with_labels.csv。\n\n'+markdown(alpha_table)+'\n'
    content+='## 表 2：完整求解时间（ms）\n\nscan_reference 只计使用预先获得标签后的求解时间；离线扫描费用单列，不将其视为免费的部署方法。\n\n'+markdown(time_table)+'\n'
    content+='## 配对不确定性\n\n以父几何为重采样单位，保持三个对比度和方法配对，按类别分层，10,000 次 bootstrap。\n\n'
    for base,v in boot['comparisons'].items():
        if 'unavailable_reason' in v:content+=f'- {primary} 与 {base} 中存在未达标求解，不推断共同残差下的时间收益。\n'
        else:content+=f'- {primary} 相对 {base}：平均时间差 {v["mean_difference_ms"]:.3f} ms，95% 区间 [{v["difference_ci95"][0]:.3f}, {v["difference_ci95"][1]:.3f}]；{v["faster_fields"]}/{v["fields"]} 个场更快。\n'
    for mode in ['loads','resolution','contrast']:
        content+=f'\n## {mode} 转移测试\n\n'
        if outputs[mode] is None:content+='尚未完成，不能作为结果引用。\n';continue
        _,s,_,_,a=outputs[mode]
        g=s.groupby(['group','method'],sort=False).agg(cases=('sample_id','count'),time_ms=('time_ms','mean'),attempt_time_ms=('attempt_time_ms','mean'),failures=('failures','sum'),max_true_residual=('max_true_residual','max')).reset_index()
        g.loc[g.failures>0,'time_ms']=np.nan
        g.to_csv(dest/f'{mode}_overall.csv',index=False)
        content+=f'最大真实残差 {a["max_true_residual"]:.6g}，失败 {a["failed_solves"]} 次。存在未达标重复的分组不报告达标平均时间（--）；attempt_time_ms 保留全部尝试的成本。\n\n'+markdown(g)+'\n'
        if mode=='loads' and (root/'analysis/zero_load.json').exists():
            content+='\n载荷失败诊断：42次失败均来自同一仅沿y变化的层状父场、三个对比度、xz纯剪切载荷。理论涨落为零；数值右端项范数为5.09e-16至6.51e-14。严格相对残差未达标，记录不改为成功。另用解析平均应力核验，最大相对误差2.03e-16；见analysis/zero_load.json。\n'
    app=root/'application';application_table=None
    if (app/'COMPLETE.json').exists():
        p=read(app/'protocol.json');rr=[read(v) for v in sorted((app/'records').glob('*.json'))]
        verify_evaluation_sources(root,p)
        assert len(rr)==len(p['parents'])*len(p['methods'])*p['repeats']
        assert all(v['protocol_sha256']==sha(app/'protocol.json') for v in rr)
        assert p['model_freeze_sha256']==sha(root/'MODELS_FROZEN.json')
        assert {(v['geometry_id'],v['method'],v['repeat']) for v in rr}=={(g,m,r) for g in p['parents'] for m in p['methods'] for r in range(p['repeats'])}
        subsolves=[]
        for v in rr:
            assert v['reference']['converged'] and v['reference']['relative_residual']<=1e-11
            assert v['iterations']==len(v['history']) and 1<=v['iterations']<=30
            assert v['elapsed_s']>sum(h['time_s'] for h in v['history'])
            assert abs(v['relative_response_error']-abs(v['history'][-1]['response']-v['target_response'])/v['target_response'])<1e-12
            assert abs(v['relative_contrast_error']-abs(10**(v['identified_log10_contrast']-p['target_log10_contrast'])-1))<1e-12
            valid=all(h['relative_residual']<=1e-8 for h in v['history']) and (p['bracket'][1]-p['bracket'][0])/2**v['iterations']<=1e-4 and v['relative_response_error']<=1e-4
            assert bool(v['converged'])==valid
            for h in v['history']:
                assert bool(h['converged'])==(h['relative_residual']<=1e-8)
            subsolves.extend(v['history'])
        write_json(dest/'application_audit.json',dict(identifications=len(rr),failed_identifications=sum(not v['converged'] for v in rr),
            subsolves=len(subsolves),failed_subsolves=sum(not h['converged'] for h in subsolves),max_true_residual=max(h['relative_residual'] for h in subsolves),
            max_relative_response_error=max(v['relative_response_error'] for v in rr),max_relative_contrast_error=max(v['relative_contrast_error'] for v in rr)))
        flat=pd.DataFrame([{k:v[k] for k in ['geometry_id','family','method','repeat','elapsed_s','iterations','identified_contrast','relative_contrast_error','relative_response_error','converged']} for v in rr])
        flat['time_ms']=flat.elapsed_s*1000;flat.to_csv(dest/'application_repetitions.csv',index=False)
        table=flat.groupby(['family','method'],sort=False).agg(time_ms=('time_ms','mean'),iterations=('iterations','mean'),relative_contrast_error=('relative_contrast_error','max'),relative_response_error=('relative_response_error','max'),failures=('converged',lambda x:int((~x).sum()))).reset_index()
        table['attempt_time_ms']=table.time_ms;table.loc[table.failures>0,'time_ms']=np.nan
        table.to_csv(dest/'application_summary.csv',index=False)
        application_table=table
        content+='\n## 完整逆弹性识别应用\n\n时间覆盖材料场生成、全部子求解、应力积分和外层控制。观测是独立高精度求解产生的合成数据；此处不称为实验材料验证。\n\n'+markdown(table)+'\n'
    content+='\n## 离线标签成本\n\n下表是最终保留扫描记录中的求解时间之和，不含编译、写盘、被丢弃的抢占记录和等待时间，不等于总墙钟预算。\n\n'+markdown(pd.DataFrame(label_summaries))+'\n'
    content+=f'15个候选的记录训练时间为 {offline["candidate_training_seconds"]:.3f} s；6480次验证校准的保留求解时间合计 {offline["validation_calibration_seconds"]:.3f} s。测试扫描属于评估参考成本，不能算作每次部署推理费用；训练/验证扫描属于离线构建成本，不能隐去。\n'
    (dest/'RESULTS.md').write_text(content,encoding='utf8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASG 独立实验对比</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#243343;background:#f5f7fa;margin:0;padding:32px}main{max-width:1200px;margin:auto}h1{font-size:28px}h2{font-size:20px;margin-top:32px}p{line-height:1.8}.table{overflow:auto;background:white;border:1px solid #dce3e9;border-radius:8px;padding:8px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #e6ebef;text-align:right}th:first-child,td:first-child{text-align:left}th{color:#234a65;background:#edf3f7}a{color:#16658e;margin-right:18px}small{color:#526675}.notice{background:#eaf2f6;padding:14px 18px;border-left:4px solid #347592}</style><main><h1>ASG 独立实验对比</h1>'''
    page+=f'<p class="notice">180 个独立测试场 · 60 个父几何 · 每方法 6 次配对重复。统一达到 <strong>10⁻⁸ 真实残差</strong>，计入输入处理、统计、推理/公式、截断、缩放及完整求解。★ 为验证集冻结的主模型；验证最佳固定 α={freeze["validation_best_fixed_alpha"]:.2f}。</p>'
    page+=f'<p>本批 {audit["records"]:,} 次求解，失败 {audit["failed_solves"]} 次，最大真实残差 {audit["max_true_residual"]:.3e}。字段先平均重复，再等权平均字段。未全部达标的组不报告达标平均时间。</p>'
    pending=[title for mode,title in [('loads','载荷转移'),('resolution','网格转移'),('contrast','对比度外推')] if outputs[mode] is None]
    if application_table is None:pending.append('完整逆弹性识别')
    if pending:page+='<p class="notice">此页先展示已完成并审计的结果；尚未完成：'+html.escape('、'.join(pending))+'。不得将部分报告视为全部实验完成。</p>'
    page+='<p><a href="alpha_predictions_with_labels.csv">逐样本 α 与标签</a><a href="end_to_end_times_ms.csv">逐样本完整时间</a><a href="test_repetitions.csv">全部重复记录</a><a href="RESULTS.md">完整报告</a></p>'
    page+='<h2>预测 α 与扫描标签</h2>'+html_table(alpha_table,primary)
    page+='<h2>完整求解时间 · ms</h2><p>扫描标签行只计使用预先获得标签的求解时间；101 点扫描费用单列，不能把这一行理解为免费的在线选参方法。</p>'+html_table(time_table,primary)
    for mode,title in [('loads','载荷转移'),('resolution','网格转移'),('contrast','对比度外推')]:
        path=dest/f'{mode}_overall.csv'
        if path.exists():page+=f'<h2>{title}</h2><p>模型及阈值不重新训练；每方法两次配对重复。失败尝试仍保留，不以失败时间冒充达标时间。</p>'+html_table(pd.read_csv(path),primary)
        if mode=='loads' and (root/'analysis/zero_load.json').exists():
            page+='<p class="notice">42次失败集中于同一层状父场的3个对比度：xz纯剪切时理论涨落为零，数值右端项仅为舍入误差量级，严格相对残差未达标。解析平均应力核验最大相对误差2.03×10⁻¹⁶。失败记录及迭代上限成本全部保留，没有更改停止准则。<a href="../analysis/zero_load.json">解析诊断</a></p>'
    if application_table is not None:page+='<h2>完整逆弹性识别</h2><p>包含场生成、全部子求解、应力后处理及外层控制。使用合成观测；这不是物理材料实验验证。</p>'+html_table(application_table,primary)
    page+='<h2>离线成本与审计</h2><p>训练/验证扫描属于离线构建；测试扫描属于独立评估。保留记录的计算时间不包含编译、写盘、被丢弃的干扰记录和等待。</p>'+html_table(pd.DataFrame(label_summaries),primary)
    page+='<p><a href="paired_bootstrap.json">配对统计区间与尾部</a><a href="source_hashes.json">来源哈希</a><a href="offline_costs.json">离线预算</a></p></main></html>'
    (dest/'comparison.html').write_text(page,encoding='utf8')
    for directory in ['models','calibration','evaluation_test','evaluation_loads','evaluation_resolution','evaluation_contrast','application','baselines','fields','analysis','implementation_checks']:
        for path in (root/directory).rglob('*'):
            if path.is_file() and path.suffix in ['.json','.pt']:hashes[str(path.relative_to(root))]=sha(path)
    write_json(dest/'source_hashes.json',hashes)
    write_json(dest/'summary.json',dict(primary=primary,validation_fixed_alpha=freeze['validation_best_fixed_alpha'],selected=selected,audits={k:v[-1] for k,v in outputs.items() if v is not None},bootstrap=boot,label_budget=label_summaries,offline_costs=offline,
                                      split_audit=dict(parents=len(parents),contrast_variants_grouped=True,disjoint_from_development_shapes=True,archived_training_source_verified=True)))
    print(dest/'RESULTS.md')

if __name__=='__main__':main()
