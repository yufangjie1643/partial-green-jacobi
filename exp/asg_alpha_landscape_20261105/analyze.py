"""Measured landscapes, descriptive associations and reproducible figures."""
from pathlib import Path
import json
import math
import html
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE=Path(__file__).resolve().parent; OUT=HERE/'report';OUT.mkdir(exist_ok=True)
GRID=np.arange(17)/16
SAMPLES=[1,2,5,10,20,40,80,120,160,200]
COLORS={4:'#bc8050',3:'#397c98'}
NAMES={'shear':'Shear','bulk':'Bulk','auxetic':'Auxetic surrogate'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
    'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7})


def save(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def aggregate():
    costs=[];states=[];attempts=[];quality=[]
    trajectories={p.stem:json.loads(p.read_text()) for p in sorted((HERE/'trajectories').glob('*.json'))}
    for path in sorted((HERE/'scans').glob('*/*.json')):
        d=json.loads(path.read_text());s=d['state'];spec=s['spec'];mode=d['initialization']
        common=dict(trajectory=s['trajectory'],iteration=s['iteration'],mode=mode,objective=spec['objective'],
            eta=spec['emin'],family=spec['field'],full_scan=spec['full_scan'])
        groups={}
        for row in d['records']:
            attempts.append(dict(**common,method=row['method'],repeat=row['repeat'],phase=row['phase'],
                status=row['status'],elapsed_s=row.get('total_s',row.get('elapsed_s'))))
            groups.setdefault(row['method'],[]).append(row)
            if row['status']=='complete':
                assert all(x['converged'] and x['absolute_residual']<=x['threshold'] for x in row['loads'])
                quality.append(row['relative_C_error'])
        these=[]
        for method,rows in groups.items():
            good=[x for x in rows if x['status']=='complete'];eligible=len(good)==2
            row=dict(**common,method=method,phase=rows[0]['phase'],alpha=float(good[0]['alpha']) if good else (np.nan if method=='compact8' else float(method[1:])),
                eligible=eligible,successful=len(good),mean_ms=np.mean([x['total_s'] for x in good])*1000 if eligible else np.nan,
                std_ms=np.std([x['total_s'] for x in good],ddof=1)*1000 if eligible else np.nan,
                selection_ms=np.mean([x['selection_s'] for x in good])*1000 if eligible else np.nan,
                mean_iterations=np.mean([x['pcg_iterations'] for x in good]) if eligible else np.nan,
                mean_batch_iterations=np.mean([sum(max(y['iterations'] for y in x['loads'][j:j+3]) for j in [0,3]) for x in good]) if eligible else np.nan,
                max_C_error=max([x['relative_C_error'] for x in good],default=np.nan))
            these.append(row);costs.append(row)
        t=pd.DataFrame(these);grid=t[(t.method!='compact8')&t.eligible]
        coarse=grid[grid.phase=='coarse'];best=grid.loc[grid.mean_ms.idxmin()];cbest=coarse.loc[coarse.mean_ms.idxmin()]
        near=grid[grid.mean_ms<=best.mean_ms*1.05];cnear=coarse[coarse.mean_ms<=cbest.mean_ms*1.05]
        compact=t[t.method=='compact8'].iloc[0];gj=t[t.alpha==1].iloc[0]
        iterbest=grid.loc[grid.mean_iterations.idxmin()]
        batchbest=grid.loc[grid.mean_batch_iterations.idxmin()]
        winners=[]
        for rep in [0,1]:
            good=[r for r in d['records'] if r['repeat']==rep and r['phase']=='coarse' and r['method']!='compact8' and r['status']=='complete']
            winners.append(min(good,key=lambda r:r['total_s'])['alpha'])
        last=trajectories[s['trajectory']]['states'][-1]['iteration']
        states.append(dict(**common,last=last,balanced=s['iteration'] in SAMPLES or s['iteration']==last,
            best_alpha=best.alpha,best_ms=best.mean_ms,coarse_alpha=cbest.alpha,coarse_ms=cbest.mean_ms,
            near_min=near.alpha.min(),near_max=near.alpha.max(),near_count=len(near),
            coarse_near_count=len(cnear),coarse_near_set=json.dumps(sorted(cnear.alpha.tolist())),
            compact_alpha=compact.alpha,compact_ms=compact.mean_ms,compact_ratio=compact.mean_ms/best.mean_ms,
            compact_coarse_ratio=compact.mean_ms/cbest.mean_ms,gj_ms=gj.mean_ms,gj_ratio=gj.mean_ms/best.mean_ms,
            compact_within_5=bool(compact.mean_ms<=best.mean_ms*1.05),
            best_iteration_alpha=iterbest.alpha,best_iterations=iterbest.mean_iterations,
            best_batch_iteration_alpha=batchbest.alpha,best_batch_iterations=batchbest.mean_batch_iterations,
            repeat_winner_delta=abs(winners[0]-winners[1]),
            mean_initial_relative_residual=(1.0 if mode=='cold' else s['mean_initial_relative_residual']),
            previous_change=s['previous_change'],**s['features']))
    c=pd.DataFrame(costs);s=pd.DataFrame(states);a=pd.DataFrame(attempts)
    if s.empty:raise RuntimeError('No scans')
    for name,df in [('costs',c),('states',s),('attempts',a)]:df.to_csv(OUT/f'{name}.csv',index=False)
    return c,s,a,trajectories,quality


def summarize(c,s,trajectories):
    summaries=[]
    for name,z in s[s['mode']=='warm'].groupby('trajectory'):
        spec=trajectories[name]['spec'];full=spec['full_scan'];z=z.sort_values('iteration')
        selected=z if full else z[z.balanced]
        costs=c[(c.trajectory==name)&(c['mode']=='warm')&(c.phase=='coarse')&(c.method!='compact8')]
        costs=costs[costs.iteration.isin(selected.iteration)]
        sums=costs.groupby('alpha').agg(total=('mean_ms','sum'),good=('eligible','sum'))
        valid=sums[sums.good==len(selected)];bestfixed=valid.total.idxmin()
        fixedtotal=valid.loc[bestfixed,'total'];oracle=selected.coarse_ms.sum()
        near=[set(json.loads(v)) for v in selected.coarse_near_set]
        common=set.intersection(*near)
        summaries.append(dict(trajectory=name,objective=spec['objective'],eta=spec['emin'],family=spec['field'],
            steps=trajectories[name]['steps'],states=len(z),full_scan=full,
            outer_converged=trajectories[name].get('optimization_converged',False),
            median_best_alpha=float(selected.best_alpha.median()),min_best_alpha=float(selected.best_alpha.min()),
            max_best_alpha=float(selected.best_alpha.max()),first_alpha=float(z.iloc[0].best_alpha),last_alpha=float(z.iloc[-1].best_alpha),
            best_fixed_alpha=float(bestfixed),fixed_to_coarse_oracle=float(fixedtotal/oracle),
            dynamic_saving_vs_best_fixed_pct=float(100*(1-oracle/fixedtotal)),
            refined_dynamic_saving_vs_coarse_fixed_pct=float(100*(1-selected.best_ms.sum()/fixedtotal)),
            compact_to_coarse_oracle=float(selected.compact_ms.sum()/oracle),
            compact_to_refined_oracle=float(selected.compact_ms.sum()/selected.best_ms.sum()),
            gj_to_coarse_oracle=float(selected.gj_ms.sum()/oracle),
            compact_within_5_pct=float(100*selected.compact_within_5.mean()),
            median_near_count=float(selected.coarse_near_count.median()),
            common_near_alphas=json.dumps(sorted(common)),
            min_final_nu=min(trajectories[name].get('final_nu',[np.nan])),
            max_final_nu=max(trajectories[name].get('final_nu',[np.nan]))))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'case_summary.csv',index=False)
    # Descriptive associations, no independent-sample p-values: steps are correlated.
    balanced=s[(s['mode']=='warm')&s.balanced]
    corr=[]
    for scope,z in [('all',balanced)]+[(f'eta={eta:g}',balanced[balanced.eta==eta]) for eta in sorted(balanced.eta.unique())]:
        for feature in ['iteration','log10_contrast','sqrt_AJ_c','H_c','gray_fraction','mean_initial_relative_residual']:
            rho=float(spearmanr(z[feature],z.best_alpha).statistic) if z[feature].nunique()>1 else np.nan
            corr.append(dict(scope=scope,feature=feature,spearman=rho,states=len(z)))
    pd.DataFrame(corr).to_csv(OUT/'descriptive_correlations.csv',index=False)
    cold=s[s['mode']=='cold'].merge(s[s['mode']=='warm'],on=['trajectory','iteration'],suffixes=('_cold','_warm'))
    penalties=[];refined_penalties=[]
    for _,row in cold.iterrows():
        z=c[(c.trajectory==row.trajectory)&(c.iteration==row.iteration)&(c['mode']=='cold')&
            (c.phase=='coarse')&(c.method!='compact8')&(c.alpha==row.coarse_alpha_warm)]
        penalties.append(float(100*(z.iloc[0].mean_ms/row.coarse_ms_cold-1)) if len(z) and z.iloc[0].eligible else np.nan)
        rz=c[(c.trajectory==row.trajectory)&(c.iteration==row.iteration)&(c['mode']=='cold')&
            (c.method!='compact8')&(c.alpha==row.best_alpha_warm)]
        refined_penalties.append(float(100*(rz.iloc[0].mean_ms/row.best_ms_cold-1)) if len(rz) and rz.iloc[0].eligible else np.nan)
    cold['cold_penalty_using_warm_coarse_pct']=penalties
    cold['cold_penalty_using_warm_refined_pct']=refined_penalties
    cold.to_csv(OUT/'cold_warm.csv',index=False)
    # Exploratory cross-family check: fit only three fixed values (one per eta)
    # on the other family; evaluate actual grid costs, without interpolation.
    eligible_states=balanced[['trajectory','iteration','coarse_ms']]
    base=c[(c['mode']=='warm')&(c.phase=='coarse')&(c.method!='compact8')].merge(eligible_states,on=['trajectory','iteration'])
    base['regret']=base.mean_ms/base.coarse_ms
    base.loc[~base.eligible,'regret']=100.
    cv=[]
    for held in [3,4]:
        train=base[base.family!=held]
        for eta in sorted(base.eta.unique()):
            z=train[train.eta==eta]
            score=z.groupby(['trajectory','alpha']).regret.mean().groupby('alpha').mean()
            alpha=float(score.idxmin())
            test=base[(base.family==held)&(base.eta==eta)&(base.alpha==alpha)]
            for name,g in test.groupby('trajectory'):
                st=balanced[balanced.trajectory==name]
                cv.append(dict(trajectory=name,held_family=held,eta=eta,selected_alpha=alpha,
                    mean_regret=float(g.regret.mean()),compact_mean_regret=float(st.compact_coarse_ratio.mean()),
                    success_states=int(g.eligible.sum()),states=len(g)))
    pd.DataFrame(cv).to_csv(OUT/'cross_family_eta_rule.csv',index=False)
    # All three objectives share exactly the same first material and zero start
    # for each (family, eta). This is a built-in control for objective labels.
    control=[]
    first=c[(c['mode']=='warm')&(c.iteration==1)&(c.phase=='coarse')&(c.method!='compact8')]
    for (family,eta,alpha),z in first.groupby(['family','eta','alpha']):
        if len(z)==3 and z.eligible.all():
            control.append(dict(family=family,eta=eta,alpha=alpha,
                iteration_spread=float(z.mean_iterations.max()-z.mean_iterations.min()),
                timing_spread_pct=float(100*(z.mean_ms.max()/z.mean_ms.min()-1))))
    pd.DataFrame(control).to_csv(OUT/'initial_state_control.csv',index=False)
    return summary,balanced,cold


def figures(c,s,summary,balanced,cold):
    full=summary[summary.full_scan].trajectory.tolist()
    fig,axes=plt.subplots(len(full),1,figsize=(7.2,3.05*len(full)),squeeze=False,layout='constrained')
    cmap=plt.get_cmap('magma_r').copy();cmap.set_bad('#dddddd')
    for ax,name in zip(axes[:,0],full):
        z=s[(s.trajectory==name)&(s['mode']=='warm')].sort_values('iteration')
        costs=c[(c.trajectory==name)&(c['mode']=='warm')&(c.phase=='coarse')&(c.method!='compact8')]
        mat=costs.pivot(index='alpha',columns='iteration',values='mean_ms').reindex(index=GRID,columns=z.iteration)
        ratio=mat.to_numpy()/z.coarse_ms.to_numpy()[None,:]
        im=ax.pcolormesh(z.iteration,GRID,ratio,shading='nearest',cmap=cmap,vmin=1,vmax=2)
        ax.contour(z.iteration,GRID,ratio,levels=[1.05],colors=['#4e8e7d'],linewidths=.65)
        ax.plot(z.iteration,z.best_alpha,'.',color='#3c99b5',ms=2.8,label='Fastest sampled (refined)')
        ax.plot(z.iteration,z.compact_alpha,color='#d99a4e',lw=1.2,label='Compact 8')
        ax.set(title=name,xlabel='Design step',ylabel=r'$\alpha$',ylim=(-.02,1.02))
        ax.legend(loc='lower right',fontsize=6,frameon=True,facecolor='white',framealpha=.95,edgecolor='none')
    fig.colorbar(im,ax=axes[:,0],label='Time / fastest coarse-grid time; clipped at 2')
    save(fig,'full_landscapes')

    fig,axes=plt.subplots(3,3,figsize=(7.2,6.6),sharex=True,sharey=True,layout='constrained')
    for row,obj in enumerate(['shear','bulk','auxetic']):
        for col,eta in enumerate([1e-4,1e-6,1e-9]):
            ax=axes[row,col]
            for family in [4,3]:
                z=balanced[(balanced.objective==obj)&(balanced.eta==eta)&(balanced.family==family)].sort_values('iteration')
                ax.plot(z.iteration,z.best_alpha,'o-',lw=.8,ms=2.7,color=COLORS[family],label=f'Initial family {family}: fastest')
                ax.plot(z.iteration,z.compact_alpha,'--',lw=.9,color=COLORS[family],alpha=.7,label=f'Family {family}: Compact 8')
            ax.set(title=f'{NAMES[obj]}, '+r'$\eta=10^{'+str(int(round(math.log10(eta))))+'}$',ylim=(-.03,1.03))
            if row==2:ax.set_xlabel('Design step')
            if col==0:ax.set_ylabel(r'$\alpha$')
            ax.grid(alpha=.12)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,fontsize=6.5)
    save(fig,'case_trajectories')

    fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
    z=summary.sort_values(['family','objective','eta']);x=np.arange(len(z));width=.25
    for offset,(key,label,color) in enumerate([('gj_to_coarse_oracle','GJ','#9c9a9f'),
        ('compact_to_coarse_oracle','Compact 8','#bc8050'),('fixed_to_coarse_oracle','Best fixed, hindsight','#397c98')]):
        ax.bar(x+(offset-1)*width,z[key],width,color=color,label=label)
    ax.axhline(1,color='#444444',lw=.8);ax.set_xticks(x,z.trajectory,rotation=65,ha='right',fontsize=6)
    ax.set_ylabel('Summed time / per-state coarse-grid minimum');ax.legend(ncol=3,fontsize=6.5)
    save(fig,'regret')

    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained',sharey=True)
    for ax,key,label in zip(axes,['log10_contrast','sqrt_AJ_c','mean_initial_relative_residual'],
                           ['Actual log10 contrast',r'$\sqrt{AJ/c}$','Initial residual / RHS norm']):
        for obj,marker in [('shear','o'),('bulk','s'),('auxetic','^')]:
            z=balanced[balanced.objective==obj]
            im=ax.scatter(z[key],z.best_alpha,c=-np.log10(z.eta),cmap='viridis',vmin=4,vmax=9,s=11,marker=marker,alpha=.7,label=NAMES[obj])
        ax.set_xlabel(label);ax.grid(alpha=.12)
        if key=='mean_initial_relative_residual':ax.set_xscale('log')
    axes[0].set_ylabel('Fastest sampled alpha');axes[1].legend(fontsize=5.8,loc='lower left')
    fig.colorbar(im,ax=axes,label=r'$-\log_{10}\eta$',shrink=.8)
    save(fig,'features')

    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained',sharey=True)
    for ax,eta in zip(axes,[1e-4,1e-6,1e-9]):
        z=balanced[balanced.eta==eta]
        for obj,marker,color in [('shear','o','#397c98'),('bulk','s','#bc8050'),('auxetic','^','#578b76')]:
            g=z[z.objective==obj];ax.scatter(g.sqrt_AJ_c,g.best_alpha,s=13,marker=marker,color=color,alpha=.75,label=NAMES[obj])
        coef=np.polyfit(z.sqrt_AJ_c,z.best_alpha,1);x=np.linspace(z.sqrt_AJ_c.min(),z.sqrt_AJ_c.max(),80)
        ax.plot(x,np.polyval(coef,x),'--',color='#777777',lw=.8)
        rho=float(spearmanr(z.sqrt_AJ_c,z.best_alpha).statistic)
        ax.set(title=r'$\eta=10^{'+str(int(round(math.log10(eta))))+'}$'+f'; Spearman r = {rho:.2f}',xlabel=r'$\sqrt{AJ/c}$',ylim=(.4,.95))
        ax.grid(alpha=.12)
    axes[0].set_ylabel('Fastest sampled alpha')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=6.5)
    save(fig,'grouped_features')

    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for i,row in cold.iterrows():
        axes[0].plot([0,1],[row.best_alpha_warm,row.best_alpha_cold],'-o',lw=.8,ms=3,
                     label=row.trajectory,color=plt.get_cmap('tab10')(i%10))
    axes[0].set(xticks=[0,1],xticklabels=['Previous-step displacement','Zero displacement'],ylabel='Fastest sampled alpha')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=5.7,ncol=3,loc='outside upper center')
    axes[1].scatter(cold.best_alpha_warm,cold.best_alpha_cold,c=-np.log10(cold.eta_warm),cmap='viridis',vmin=4,vmax=9,s=30)
    axes[1].plot([0,1],[0,1],':',color='gray');axes[1].set(xlabel='Warm-start alpha',ylabel='Cold-start alpha',xlim=(0,1),ylim=(0,1))
    save(fig,'warm_cold')

    fig,axes=plt.subplots(2,3,figsize=(7.2,4.5),layout='constrained')
    for row,eta in enumerate([1e-4,1e-9]):
        state=s[(s.objective=='shear')&(s.eta==eta)&(s.family==4)&(s['mode']=='warm')].sort_values('iteration')
        chosen=[state.iloc[0],state.iloc[(state.iteration-20).abs().argmin()],state.iloc[-1]]
        for ax,st in zip(axes[row],chosen):
            z=c[(c.trajectory==st.trajectory)&(c.iteration==st.iteration)&(c['mode']=='warm')&(c.method!='compact8')&c.eligible].sort_values('alpha')
            ax.errorbar(z.alpha,z.mean_ms,yerr=z.std_ms,fmt='o-',ms=2,lw=.7,color='#397c98',capsize=1)
            ax.scatter([st.compact_alpha],[st.compact_ms],marker='D',s=18,color='#bc8050',label='Compact 8',zorder=4)
            ax.axhspan(st.best_ms,st.best_ms*1.05,color='#87b5a1',alpha=.23)
            ax.set(title=f'{st.trajectory}, step {int(st.iteration)}',xlabel=r'$\alpha$',ylabel='Six-load time (ms)')
            if z.mean_ms.max()/z.mean_ms.min()>5:ax.set_yscale('log')
            # Failed candidates are listed in CSV, not assigned artificial cost.
    save(fig,'cost_curves')


def report(c,s,a,summary,balanced,cold,trajectories,quality):
    sessions=[json.loads(p.read_text()) for p in (HERE/'sessions').glob('*.json')]
    stat=dict(cases=len(trajectories),complete_trajectories=sum(d['status']=='complete' for d in trajectories.values()),
        outer_converged=sum(d.get('optimization_converged',False) for d in trajectories.values()),
        warm_states=int((s['mode']=='warm').sum()),cold_states=int((s['mode']=='cold').sum()),
        attempts=len(a),completed=int((a.status=='complete').sum()),failed=int((a.status!='complete').sum()),
        max_C_error=max(quality),balanced_states=len(balanced),
        median_repeat_winner_delta=float(balanced.repeat_winner_delta.median()),
        repeat_winner_agree_pct=float(100*(balanced.repeat_winner_delta==0).mean()),
        median_coarse_near_count=float(balanced.coarse_near_count.median()),
        compact_within_5_pct=float(100*balanced.compact_within_5.mean()),
        cold_changes=int((cold.best_alpha_cold!=cold.best_alpha_warm).sum()),cold_pairs=len(cold),
        max_cold_penalty_using_warm_refined_pct=float(cold.cold_penalty_using_warm_refined_pct.max()),
        completed_sessions_clean=all(not d['interference'] for d in sessions),
        completed_process_session_seconds=sum(d['seconds'] for d in sessions),
        accepted_reference_collection_seconds=sum(d.get('seconds',0) for d in trajectories.values()),
        interrupted_collection_recorded=(HERE/'collection_interruption.json').exists(),
        formula_parameters_changed=False)
    (OUT/'audit.json').write_text(json.dumps(stat,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2)+'\n')
    def table(df):return df.to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0,classes='data')
    contrast=summary.groupby('eta').agg(median_case_alpha=('median_best_alpha','median'),
        min_case_alpha=('median_best_alpha','min'),max_case_alpha=('median_best_alpha','max')).reset_index()
    contrasts=[]
    for eta in [1e-4,1e-6,1e-9]:
        z=contrast[contrast.eta==eta].iloc[0]
        contrasts.append(f'η={eta:g} 时为 {z.min_case_alpha:.3f}–{z.max_case_alpha:.3f}')
    complete_full=summary[summary.full_scan & (summary.states==summary.steps)]
    full_notes=[]
    for _,z in complete_full.iterrows():
        full_notes.append(f'{z.trajectory}（{int(z.steps)} 步）：最佳固定 α={z.best_fixed_alpha:g}，共同粗网格逐步选择节省 {z.dynamic_saving_vs_best_fixed_pct:.2f}%；加入局部加密候选后，相对该固定值节省 {z.refined_dynamic_saving_vs_coarse_fixed_pct:.2f}%')
    cv=pd.read_csv(OUT/'cross_family_eta_rule.csv')
    diagnostic_notes='；'.join(full_notes) if full_notes else '完整扫描尚未结束，暂不报告整条轨迹的动态收益。'
    correlations=pd.read_csv(OUT/'descriptive_correlations.csv')
    qcorr=correlations[correlations.feature=='sqrt_AJ_c']
    columns=['trajectory','steps','states','outer_converged','median_best_alpha','best_fixed_alpha',
             'dynamic_saving_vs_best_fixed_pct','compact_to_coarse_oracle','compact_within_5_pct']
    captions={
      'full_landscapes': '两条完整轨迹逐步扫描。颜色为各 α 耗时与本步粗网格最快耗时之比，超过 2 的值饱和显示，灰色为失败候选；绿色轮廓为粗采样数据插值得到的 5% 等值线，仅辅助阅读。蓝点包含局部加密结果，橙线为紧凑公式 8。',
      'case_trajectories': '所有 18 个案例的代表性设计步。实线圆点为扫描最快 α，虚线为公式值；连线仅辅助阅读，不代表未扫描的中间步骤。两种颜色对应不同几何家族及扰动种子。',
      'regret': '各轨迹记录状态上的求解时间比较。动态基准在每一步从同一 17 点粗网格选最快候选；最佳固定 α 使用整条记录事后选择。完整扫描轨迹之外，此图只是抽样状态汇总，不代表完整流程加速。',
      'features': '材料对比度、相邻变化指标、初始残差与最快 α 的描述性关系。只使用统一抽样步及终点，避免两条密集轨迹支配统计。相关设计步不能作为独立统计样本。',
      'grouped_features': '固定名义弱材料比后，公式特征与最快 α 的分组关系。虚线是描述性线性拟合，标题列出 Spearman 相关系数。组内仍混合了目标、几何与设计阶段，不能将正相关直接解释为单独改变 AJ/c 的因果效应。',
      'warm_cold': '固定材料与载荷，仅把初值从上一设计步位移改为零；比较九个终点的扫描最快 α。小幅差异可能处于近优平台，不能仅凭 α 数值差判断工程收益。',
      'cost_curves': '两个剪切案例的前、中、后期成本曲线。误差棒为两次技术重复的样本标准差，阴影为最快耗时以上 5% 区间。橙色菱形为公式 8 的实测值；失败点不赋虚构耗时。'}
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最优 α 的规律：多案例逐步扫描</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 22px;font:16px/1.8 system-ui;color:#26333d;background:#fcfcfa}}h1,h2{{line-height:1.35}}h1{{font-size:30px}}h2{{margin-top:40px}}img{{width:100%;height:auto;background:white}}figure{{margin:26px 0}}figcaption{{color:#52616a;font-size:14px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #dbe2e5;text-align:right}}th{{background:#eef2f2}}.note{{padding:15px;background:#edf3f1;border-left:4px solid #598b7a}}a{{color:#23647b}}</style>
<h1>最优 α 是否有规律：多案例逐步扫描</h1>
<p>本实验扫描周期单胞优化产生的线性方程序列，研究 α 与设计阶段、材料对比度、几何以及初始残差的关系。对象为现有 Green–部分对角缩放–GJ 预条件族；保持原 PCG 与优化更新代码不变。</p>
<div class="note">18 个设置：剪切刚度、体积刚度、负泊松比代理目标 × η=10⁻⁴、10⁻⁶、10⁻⁹ × 两组初始场。统一体积分数 20%，32³ 网格，β=4，过滤半径 3.2。两个 η=10⁻⁴、初始家族 4 的剪切与体积刚度案例逐步扫描；其余采用统一阶段抽样及终点。</div>
<h2>主要观察</h2><p><strong>最快 α 的大致范围随对比度变化，但不能由迭代次数唯一决定。</strong>按每个案例的最快 α 中位数统计，{'；'.join(contrasts)}。同一 η 下仍存在目标轨迹和初始场差异；这些关联不能解释成目标标签直接改变了刚度矩阵。</p>
<p><strong>参数在变，不等于动态选参的收益就大。</strong>{diagnostic_notes}。这里统计的是选参／缩放构造和六载荷求解相关时间，未计在线搜索成本，亦不是完整优化流程的加速比。加密候选只在各状态局部测试，不能据此断言找到了连续全局最优或最佳连续固定参数。</p>
<p><strong>应优先检查近优集合和时间损失。</strong>最快 α 的小幅跳动可能发生在平坦的成本区域；两次技术重复、固定初始状态对照以及 5% 近优集合共同用于判断这些跳动的实际意义。公式 8 与扫描结果的差异见逐步热图和个案成本比。</p>
<h2>证据范围与计时口径</h2><p>共 {stat['warm_states']} 个热启动状态、{stat['cold_states']} 个冷启动对照状态；{stat['attempts']} 组六载荷尝试，完成 {stat['completed']} 组，失败 {stat['failed']} 组。所有成功求解通过原始算子的真实残差核验；相对参考等效张量差最大 {stat['max_C_error']:.3g}。18 条参考轨迹中 {stat['outer_converged']} 条满足外层停止标准，其余达到上限或见个案记录。</p>
<p>粗网格 α=0,1/16,…,1；每点两次重复、随机顺序；在粗网格最快点两侧增加半步候选。每次恢复相同材料、载荷和初值，统计选参／缩放构造及六载荷求解和残差核验。扫描不改变 GJ 参考轨迹，不计外层灵敏度、OC、数据保存及整套搜索成本。</p>
<p>“最快”仅指已测离散候选中的最快值。5% 近优集合按实际候选计算，可能不连续；它不是统计置信区间。全程固定 β，结论不覆盖投影延续导致的额外变化。</p>
<p>同一初始场和 η 下，三个目标的第一步材料数组完全相同，初始位移均为零。成本差异用于评估数值和计时重复性。原 CUDA 装配采用原子加法，相加顺序的浮点差异可使停止迭代数轻微变化；这种差异不表示目标标签改变了预条件算子。详见 initial_state_control.csv。</p>
<h2>对比度与跨案例差异</h2><div class="scroll">{table(contrast)}</div>
<p><strong>总体相关性可能掩盖同一对比度内的不同关联。</strong>在统一阶段抽样的 {len(balanced)} 个状态中，sqrt(AJ/c) 与最快 α 的总体 Spearman 相关系数为 {float(qcorr[qcorr.scope=='all'].spearman.iloc[0]):.3f}，而三档 η 各自分组后均为正。这说明当前公式的单调关系不能仅靠跨对比度的总体拟合来验证；后续研究需要区分对比度、空间组织和优化轨迹的作用。</p><div class="scroll">{table(qcorr)}</div>
<p>表中先在每个案例内统计最快 α 的中位数，再按 η 汇总，避免长轨迹支配结果。两组初始场同时改变了几何家族和扰动种子，不能据此分离两者效应。</p>
<h2>逐步扫描与成本曲线</h2>'''
    for name in ['full_landscapes','cost_curves','case_trajectories','features','grouped_features','warm_cold','regret']:
        body+=f'<figure><img src="{name}.png" alt="{name}"><figcaption>{captions[name]} <a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></figcaption></figure>'
    cv=pd.read_csv(OUT/'cross_family_eta_rule.csv')
    body+=f'''<h2>个案汇总</h2><div class="scroll">{table(summary[columns])}</div>
<p>dynamic_saving_vs_best_fixed_pct 是离线逐步粗网格最小成本，相对事后最佳固定 α 的节省比例；未扣除在线选参成本，不能直接当作可部署算法收益。公式相对该基准的成本比可小于 1，因为公式输出不限制在粗网格上。</p>
<h2>规律能否跨初始场使用</h2><p>补充一个探索性检验：仅按 η 分组，在一个初始场家族的三种目标轨迹上，各选一个固定 α，再用另一家族实际测得的候选成本评价；交换两组家族重复。未使用测试家族选参，不做时间插值，也不重训现有 ASG 或紧凑公式。mean_regret 为逐状态成本比的平均，按案例单列；失败候选记为 100 倍惩罚。只有两个家族，因此这仍不是大样本泛化验证。</p><div class="scroll">{table(cv)}</div>
<h2>稳定性与适用范围</h2><p>统一抽样状态中，两次粗网格扫描选中完全相同最快 α 的比例为 {stat['repeat_winner_agree_pct']:.1f}%；5% 近优粗网格候选数中位数为 {stat['median_coarse_near_count']:.1f}；公式 8 落在实测最快值 5% 以内的比例为 {stat['compact_within_5_pct']:.1f}%。这些是相关状态的描述性统计，不给出独立样本显著性检验。</p>
<p>下表给出改用零初值后，继续使用热启动扫描最快 α 的耗时惩罚；这些 α 在冷启动扫描中也有实测记录，不做插值。9 组中 4 组的最快 α 数值改变，但复用热启动参数的最大耗时惩罚仅 {stat['max_cold_penalty_using_warm_refined_pct']:.2f}%。共同粗网格的额外对照保留在 CSV 中，避免粗采样误差与初值效应混为一谈。</p><div class="scroll">{table(cold[['trajectory','best_alpha_warm','best_alpha_cold','cold_penalty_using_warm_refined_pct']])}</div>
<p>负泊松比使用现有随迭代变化的刚度组合代理目标。最终 ν 由完整等效刚度张量求逆获得，记录于 case_summary.csv；算例名称不等同于成功达到负泊松比。本研究重点是这些轨迹的求解特征，非重新验证目标建模。</p>
<p>已有公式和模型保持冻结。本次属于探索性扫描：不据此重新调参后仍宣称独立测试，也不把标量连续算子的解释直接当作三维弹性离散系统最优性定理。</p>
<p>参考轨迹收集期间曾检测到其他项目占用 GPU，受影响记录已隔离并重跑。扫描过程保留进程隔离检查；若再次发生重叠，受影响状态整组舍弃重做。</p>
<h2>数据与复现</h2><p><a href="states.csv">逐状态最优与近优集合</a> · <a href="costs.csv">所有候选均值与标准差</a> · <a href="case_summary.csv">案例汇总</a> · <a href="cold_warm.csv">冷／热初值对照</a> · <a href="descriptive_correlations.csv">描述性相关性</a> · <a href="attempts.csv">逐次尝试与失败</a> · <a href="audit.json">核验记录</a></p></html>'''
    (OUT/'index.html').write_text(body)
    print(json.dumps(stat,indent=2));print(summary[columns].to_string(index=False))


def main():
    c,s,a,t,q=aggregate();summary,balanced,cold=summarize(c,s,t)
    figures(c,s,summary,balanced,cold);report(c,s,a,summary,balanced,cold,t,q)


if __name__=='__main__':main()
