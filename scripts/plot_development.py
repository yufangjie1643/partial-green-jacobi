"""Plot archived convergence measurements and reproduce the calibration table.

Python/matplotlib quantitative grids, 183 mm wide, editable SVG/PDF and PNG.
Evidence: iteration and time sensitivity precede the shared-parameter table.
Technical repeats quantify timing spread; trajectories are not independent steps.
No residual histories or unmeasured exponent costs are reconstructed.
"""
from pathlib import Path
import hashlib
import itertools
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
D = P / 'data/development'
A = P / 'outputs/figures'
A.mkdir(parents=True, exist_ok=True)
GRID = np.arange(17) / 16
COLORS = ['#39758a', '#ba8051', '#71668f']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .65,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})


def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(A / f'{name}.{ext}', dpi=320, facecolor='white')
    plt.close(fig)


def main():
    s = pd.read_csv(D / 'states.csv')
    c = pd.read_csv(D / 'costs.csv')
    assert len(s) == 172 and len(c) == 2924
    assert c.groupby(['trajectory', 'iteration']).size().eq(17).all()
    ok = c.eligible & c.successful.eq(2)
    assert ok.sum() == 2591

    # Figure 1: same shear family, three prescribed stages at each contrast.
    fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 112/25.4))
    fig.subplots_adjust(left=.10, right=.98, bottom=.17, top=.89,
                        hspace=.30, wspace=.32)
    selected = []
    for j, exponent in enumerate([4, 6, 9]):
        name = f'S20_e{exponent}_i4'
        last = int(s.loc[s.trajectory.eq(name), 'iteration'].max())
        for k, step in enumerate([1, 20, last]):
            z = c[c.trajectory.eq(name) & c.iteration.eq(step)].sort_values('alpha')
            valid = z.eligible & z.successful.eq(2)
            it = z.mean_iterations.where(valid)
            time = z.mean_ms.where(valid)
            axes[0, j].plot(z.alpha, it, '.-', color=COLORS[k], lw=1,
                            ms=3, label=['Initial', 'Evaluation 20', 'Last evaluated'][k])
            axes[1, j].plot(z.alpha, time, '.-', color=COLORS[k], lw=1, ms=3)
            axes[1, j].fill_between(z.alpha, time-z.std_ms, time+z.std_ms,
                                    color=COLORS[k], alpha=.14, linewidth=0)
            axes[1, j].scatter(z.loc[~valid, 'alpha'], [0.97]*(~valid).sum(),
                marker='x', s=12, color=COLORS[k], linewidth=.7,
                transform=axes[1, j].get_xaxis_transform())
            best = z.loc[time.idxmin()]
            selected.append({'trajectory': name, 'iteration': step,
                'best_alpha': float(best.alpha), 'mean_ms': float(best.mean_ms),
                'mean_iterations': float(best.mean_iterations)})
        axes[0, j].set_title(rf'$\eta=10^{{-{exponent}}}$; last evaluation {last}')
        for row in range(2):
            ax = axes[row, j]
            ax.set_yscale('log')
            ax.set_xlim(-.03, 1.03)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.grid(axis='y', which='major', color='#dddddd', lw=.45)
            ax.text(-.15, 1.04, 'abcdef'[row*3+j], transform=ax.transAxes,
                    fontsize=9, fontweight='bold')
        axes[1, j].set_xlabel(r'Scaling exponent $\alpha$')
    axes[0, 0].set_ylabel('PCG iterations, six-load sum')
    axes[1, 0].set_ylabel('Analysis time (ms)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.52, .015), ncol=3)
    save(fig, 'development_convergence')

    # Figure 2: every fitted state, coarse candidates only. Lines join samples.
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 72/25.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.27, top=.86, wspace=.12)
    objectives = ['shear', 'bulk', 'auxetic']
    names = ['Shear', 'Volumetric', 'Auxetic surrogate']
    ranges = {}
    for j, exponent in enumerate([4, 6, 9]):
        subset = s[np.isclose(s.eta, 10.**-exponent, rtol=1e-8, atol=0)]
        ranges[str(exponent)] = {'states': len(subset),
            'min': float(subset.coarse_alpha.min()),
            'median': float(subset.coarse_alpha.median()),
            'max': float(subset.coarse_alpha.max())}
        for i, objective in enumerate(objectives):
            for family in [3, 4]:
                z = subset[subset.objective.eq(objective) & subset.family.eq(family)]
                z = z.sort_values('iteration')
                axes[j].plot(z.iteration, z.coarse_alpha,
                    color=COLORS[i], linestyle='-' if family == 3 else '--',
                    marker='o' if family == 3 else '^', ms=2.8, lw=.8, alpha=.85)
        axes[j].set_title(rf'$\eta=10^{{-{exponent}}}$; {len(subset)} states')
        axes[j].set_xlabel('Design evaluation')
        axes[j].set_xticks([1, 50, 100, 150, 200])
        axes[j].set_ylim(.40, 1.01)
        axes[j].set_yticks([.4375, .5625, .6875, .8125, .9375])
        axes[j].grid(axis='y', color='#dddddd', lw=.45)
        axes[j].text(-.10, 1.08, 'abc'[j], transform=axes[j].transAxes,
                     fontsize=9, fontweight='bold')
    axes[0].set_ylabel(r'Fastest measured grid exponent $\alpha$')
    handles = [plt.Line2D([], [], color=color, lw=1.2, label=label)
               for color, label in zip(COLORS, names)]
    handles += [plt.Line2D([], [], color='#666666', linestyle=ls, marker=mark,
                           ms=3, lw=.8, label=f'Initial family {family}')
                for family, ls, mark in [(3, '-', 'o'), (4, '--', '^')]]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(.53, .005), columnspacing=1.2)
    save(fig, 'development_alpha_evolution')

    # Independent reproduction of all 4,913 table candidates from copied costs.
    times = c.assign(cost=c.mean_ms.where(ok)).pivot(
        index=['trajectory', 'iteration'], columns='alpha', values='cost')
    keys = pd.MultiIndex.from_frame(s[['trajectory', 'iteration']])
    costs = times.reindex(keys).to_numpy() / s.gj_ms.to_numpy()[:, None]
    costs[~np.isfinite(costs)] = 100
    etas = [1e-4, 1e-6, 1e-9]
    eta_index = np.argmin(np.abs(np.log10(s.eta.to_numpy()[:, None])-
                                      np.log10(etas)), axis=1)
    groups = [np.flatnonzero(s.trajectory.to_numpy() == tr)
              for tr in sorted(s.trajectory.unique())]
    candidates = []
    for ix in itertools.product(range(17), repeat=3):
        ratios = costs[np.arange(len(s)), np.array(ix)[eta_index]]
        tr = np.array([ratios[g].mean() for g in groups])
        candidates.append((.8*tr.mean()+.2*tr.max(), tuple(GRID[list(ix)])))
    loss, anchors = min(candidates)
    assert anchors == (.5625, .6875, .8125)
    assert abs(loss - .5495816713547617) < 1e-12
    summary = {'date': '2026-11-06', 'new_gpu_measurements': False,
        'claim': 'Measured convergence sensitivity motivates empirical shared exponents.',
        'backend': 'Python/matplotlib', 'figure_width_mm': 183,
        'statistics': 'Two technical repetitions; SD bands, no confidence intervals.',
        'history_scope': 'Terminal iteration counts, final residual certificates and timings; no residual histories.',
        'states': len(s), 'coarse_cost_entries': len(c), 'eligible_entries': int(ok.sum()),
        'candidate_table_combinations': len(candidates), 'table_anchors': anchors,
        'table_loss': loss, 'statewise_ranges': ranges, 'illustrated_states': selected,
        'hashes': {f'data/development/{name}': hashlib.sha256((D/name).read_bytes()).hexdigest()
                   for name in ['states.csv', 'costs.csv']}}
    (A/'evidence_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
