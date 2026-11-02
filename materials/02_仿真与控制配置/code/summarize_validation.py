import sys, numpy as np, pandas as pd
from scipy import stats
sim = pd.read_csv(sys.argv[1]); real = pd.read_csv(sys.argv[2])
real['overshoot_mm'] = np.maximum(real.reach_max_L_mm, real.reach_max_R_mm) - real.stroke_mm
real['resid_abs_mm'] = real.residual_t20_yG_mm.abs()
real = real.rename(columns={'method_id': 'method', 'track_peak_500h_mm': 'track_peak_mm', 'sync_peak_500h_mm': 'sync_peak_mm', 'sat_L_pct': 'satL_pct'})
K = ['track_rmse_mm', 'sync_rms_mm', 'wy_peak', 'satL_pct', 'rho_y_mean', 'overshoot_mm', 'resid_abs_mm']
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
print('=== medians: real | sim')
for meth in ['M1_BASE', 'M2_BLNN', 'M3_LCBLNN']:
    r = real[real.method == meth].groupby('stroke_mm')[K].median().add_suffix('_R')
    s = sim[sim.method == meth].groupby('stroke_mm')[K].median().add_suffix('_S')
    j = pd.concat([r, s], axis=1)[[c + x for c in K for x in ['_R', '_S']]]
    print(meth); print(j.round(3).to_string())
print('=== LC vs BLNN change of MEAN (%) real | sim, wins sim (LC<BLNN) /10, sim Wilcoxon p')
for s_ in sorted(sim.stroke_mm.unique()):
    o = [int(s_)]
    for c in ['track_rmse_mm', 'sync_rms_mm', 'wy_peak']:
        R = real[real.stroke_mm == s_].groupby('method')[c].mean(); S = sim[sim.stroke_mm == s_].groupby('method')[c].mean()
        pv = sim[sim.stroke_mm == s_].pivot(index='block', columns='method', values=c)
        wins = int((pv['M3_LCBLNN'] < pv['M2_BLNN']).sum())
        try: p = stats.wilcoxon(pv['M3_LCBLNN'], pv['M2_BLNN']).pvalue
        except Exception: p = np.nan
        o.append(f"{c[:5]} {100*(R['M3_LCBLNN']/R['M2_BLNN']-1):+.0f}|{100*(S['M3_LCBLNN']/S['M2_BLNN']-1):+.0f} w{wins} p{p:.3f}")
    F = sim[sim.stroke_mm == s_].groupby('method')[['wy_peak', 'track_rmse_mm']].mean()
    o.append(f"fixedM W {F.loc['Fixed_matched','wy_peak']:.2f} vs LC {F.loc['M3_LCBLNN','wy_peak']:.2f}; tr {F.loc['Fixed_matched','track_rmse_mm']:.3f} vs {F.loc['M3_LCBLNN','track_rmse_mm']:.3f}")
    print(' | '.join(map(str, o)))
m2 = sim[sim.method == 'M2_BLNN']
print('sim M2 spearman(W,track):', {int(s): round(stats.spearmanr(g.wy_peak, g.track_rmse_mm).correlation, 2) for s, g in m2.groupby('stroke_mm')})
print('sim M2 track max:', m2.groupby('stroke_mm').track_rmse_mm.max().round(2).to_dict())
print('real M2 track max:', real[real.method == 'M2_BLNN'].groupby('stroke_mm').track_rmse_mm.max().round(2).to_dict())
