# -*- coding: utf-8 -*-
"""协调因子：实物实测与标称工况仿真预测对照图。

用法：python plot_rho.py real_all.csv ../results/02_nominal_800/validation_M1_M2_Fixed_LC.csv ../results/figures/fig_rho_real_vs_sim.png
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

real = pd.read_csv(sys.argv[1]); sim = pd.read_csv(sys.argv[2]); out = sys.argv[3]
plt.rcParams.update({'font.family': ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'sans-serif'],
                     'font.size': 8, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False})
S = [10, 20, 30, 40, 50, 100, 200, 300]; x = np.arange(len(S))
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.4))
for ax, col, ttl in [(axs[0], 'rho_y_mean', r'(a) 平动协调因子均值 $\rho_y$'),
                     (axs[1], 'rho_a_mean', r'(b) 偏转协调因子均值 $\rho_\alpha$')]:
    rq = real[real.method_id == 'M3_LCBLNN'].groupby('stroke_mm')[col].quantile([.25, .5, .75]).unstack()
    sq = sim[sim.method == 'M3_LCBLNN'].groupby('stroke_mm')[col].quantile([.25, .5, .75]).unstack()
    ax.errorbar(x - 0.08, rq[0.5], yerr=[rq[0.5]-rq[0.25], rq[0.75]-rq[0.5]], fmt='o-', color='#2a78d6', ms=4, lw=1.4, capsize=2, label='实测')
    ax.errorbar(x + 0.08, sq[0.5], yerr=[sq[0.5]-sq[0.25], sq[0.75]-sq[0.5]], fmt='^--', color='#555', ms=4, lw=1.2, capsize=2, label='仿真预测')
    ax.set_xticks(x); ax.set_xticklabels(S); ax.set_xlabel('行程 / mm'); ax.set_title(ttl, fontsize=8.5, loc='left')
    ax.set_ylim(0.5, 1.02); ax.grid(True, color='#e6e6e3', lw=0.5)
axs[0].legend(frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(out, dpi=300)
print('saved', out)
