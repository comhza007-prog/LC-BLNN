# Calibrate plant parameters on hardware M1_BASE only (no learning in the loop).
# Usage: python calib2.py <real_all.csv> <n_samples> <out.csv> [center_json scale]
import sys, json, time, numpy as np, pandas as pd
from multiprocessing import Pool
from config_e5_final import ControllerConfig
from trajectory import trajectory
from plant_sim import Plant, run, metrics

STROKES = [10, 50, 300]
TGT = ['track_rmse_mm', 'track_peak_mm', 'sync_rms_mm', 'sync_peak_mm', 'satL_pct', 'resid_abs_mm', 'overshoot_mm']
real = pd.read_csv(sys.argv[1])
real['overshoot_mm'] = np.maximum(real.reach_max_L_mm, real.reach_max_R_mm) - real.stroke_mm
real['resid_abs_mm'] = real.residual_t20_yG_mm.abs()
real = real.rename(columns={'track_peak_500h_mm': 'track_peak_mm', 'sync_peak_500h_mm': 'sync_peak_mm', 'sat_L_pct': 'satL_pct'})
R = real[real.method_id == 'M1_BASE'].groupby('stroke_mm')[TGT].median()

RANGES = dict(m_eff=(3, 15, 'lin'), fc_tot=(0.5, 6, 'lin'), fs_ratio=(1.0, 2.5, 'lin'), b_tot=(2, 40, 'log'),
              asym=(1.0, 1.4, 'lin'), k_scale=(0.25, 2.0, 'log'), noise_rpm=(5, 40, 'lin'))

def sample(rng, center=None, scale=1.0):
    p = {}
    for k, (lo, hi, kind) in RANGES.items():
        if kind == 'log':
            lo_, hi_ = np.log(lo), np.log(hi)
        else:
            lo_, hi_ = lo, hi
        if center is None:
            v = rng.uniform(lo_, hi_)
        else:
            c = np.log(center[k]) if kind == 'log' else center[k]
            v = np.clip(c + scale*(hi_-lo_)*rng.normal(0, 0.15), lo_, hi_)
        p[k] = float(np.exp(v)) if kind == 'log' else float(v)
    return p

def evaluate(p):
    out = dict(p); cost = 0.0
    for s in STROKES:
        t, a, b, d = trajectory(s, 0.001)
        m = metrics(run(t, a, b, d, ControllerConfig(enable_blnn=False), Plant(**p), 'VANILLA', seed=7), t, s)
        for k in TGT:
            out[f'{k}_{s}'] = m[k]
            cost += np.log(max(m[k], 0.01)/max(R.loc[s, k], 0.01))**2
    out['cost'] = cost
    return out

if __name__ == '__main__':
    n = int(sys.argv[2]); rng = np.random.default_rng(int(time.time()) % 10000)
    center = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    ps = [sample(rng, center, scale) for _ in range(n)]
    with Pool(2) as pool:
        res = pool.map(evaluate, ps)
    df = pd.DataFrame(res).sort_values('cost')
    df.to_csv(sys.argv[3], index=False)
    print('targets:\n', R.round(3).to_string())
    print(df.head(5)[list(RANGES) + ['cost']].round(3).to_string(index=False))
