import sys, numpy as np, pandas as pd
from scipy import stats
d = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])
d = d.drop_duplicates(subset=['stroke_mm', 'block', 'method'])
K = ['track_rmse_mm', 'track_peak_mm', 'sync_rms_mm', 'sync_peak_mm', 'wy_peak', 'wa_peak', 'rho_y_mean', 'rho_a_mean', 'resid_abs_mm']
order = [m for m in ['M1_BASE', 'M2_BLNN', 'Fixed_matched', 'Global_scalar', 'Freeze_sat', 'M3_LCBLNN'] if m in d.method.unique()]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
g = d.groupby(['stroke_mm', 'method'])[K].mean().reset_index()
g['method'] = pd.Categorical(g.method, order); g = g.sort_values(['stroke_mm', 'method'])
print(g.round(4).to_string(index=False))
print('\nLC vs comparator: wins (LC better)/10, Wilcoxon p')
for comp in ['Global_scalar', 'Freeze_sat', 'M2_BLNN']:
    if comp not in order: continue
    print('==', comp)
    for s in sorted(d.stroke_mm.unique()):
        o = [int(s)]
        for k in ['track_rmse_mm', 'track_peak_mm', 'sync_rms_mm', 'wy_peak', 'wa_peak']:
            p = d[d.stroke_mm == s].pivot(index='block', columns='method', values=k)
            if 'M3_LCBLNN' not in p or comp not in p: continue
            diff = p['M3_LCBLNN'] - p[comp]
            try: pv = stats.wilcoxon(p['M3_LCBLNN'], p[comp]).pvalue
            except Exception: pv = np.nan
            o.append(f"{k[:8]} {100*diff.mean()/p[comp].mean():+.1f}% w{int((diff<0).sum())} p{pv:.3f}")
        print(' | '.join(map(str, o)))
