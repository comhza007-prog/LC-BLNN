# -*- coding: utf-8 -*-
"""Small parameter-sensitivity benchmark for LC-BLNN.

Keeps the controller topology, trajectory, limits, friction, and sampling fixed.
Only sweeps the physical translational mass m_G and the controller nominal
coefficient M_hat_y. Outputs are isolated from the official benchmark files.
"""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "sensitivity_small"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from config_e5_final import ControllerConfig, PlantConfig
from run_unified_saturation_benchmark import run_single_simulation


def smooth_10mm(Ts):
    t = np.arange(0.0, 20.0, Ts)
    yd = np.zeros_like(t); yd_dot = np.zeros_like(t); yd_ddot = np.zeros_like(t)
    S, T = 0.010, 4.0
    for i, ti in enumerate(t):
        if 4.0 <= ti < 8.0:
            tau = (ti - 4.0) / T; p = 10*tau**3-15*tau**4+6*tau**5
            dp = 30*tau**2-60*tau**3+30*tau**4
            ddp = 60*tau-180*tau**2+120*tau**3
            yd[i], yd_dot[i], yd_ddot[i] = S*p, S/T*dp, S/T**2*ddp
        elif 8.0 <= ti < 10.0:
            yd[i] = S
        elif 10.0 <= ti < 14.0:
            tau = (ti - 10.0) / T; p = 10*tau**3-15*tau**4+6*tau**5
            dp = 30*tau**2-60*tau**3+30*tau**4
            ddp = 60*tau-180*tau**2+120*tau**3
            yd[i], yd_dot[i], yd_ddot[i] = S*(1-p), -S/T*dp, -S/T**2*ddp
    return t, yd, yd_dot, yd_ddot


def constrained_50mm(Ts):
    t = np.arange(0.0, 6.0, Ts); A = 0.025; f = 0.5; w = 2*np.pi*f
    return t, A*(1-np.cos(w*t)), A*w*np.sin(w*t), A*w*w*np.cos(w*t)


def metrics(res):
    ey = res['e_y']*1000.0; es = res['e_sync']*1000.0
    return {
        'track_rmse_mm': float(np.sqrt(np.mean(ey**2))),
        'track_peak_mm': float(np.max(np.abs(ey))),
        'sync_rms_mm': float(np.sqrt(np.mean(es**2))),
        'sync_peak_mm': float(np.max(np.abs(es))),
        'peak_Wy': float(np.max(res['norm_Wy'])),
        'end_Wy': float(res['norm_Wy'][-1]),
        'min_rho_y': float(np.min(res['rho_y'])),
        'sat_rate_pct': float(np.mean(res['sat_alloc'])*100.0),
    }


def run_case(case_name, traj, m_values, mhat_values):
    t, yd, yd_dot, yd_ddot = traj
    rows = []
    base_c = ControllerConfig(Imax_L=800.0, Imax_R=800.0)
    base_p = PlantConfig(m_G=3.0, b_L=63.0, b_R=75.6, f_c_L=14.4, f_c_R=14.4)
    for m in m_values:
        for mhat in mhat_values:
            p = replace(base_p, m_G=float(m))
            c = replace(base_c, M_hat_y=float(mhat))
            for mode, fixed in [('VANILLA',(1.0,1.0)), ('LC',(1.0,1.0))]:
                res = run_single_simulation(t, yd, yd_dot, yd_ddot, c, p, mode=mode, fixed_c=fixed)
                row = {'case': case_name, 'method': 'Vanilla BLNN' if mode=='VANILLA' else 'Dynamic LC-BLNN',
                       'm_G_kg': float(m), 'M_hat_y': float(mhat), 'mismatch_ratio': float(mhat/m)}
                row.update(metrics(res)); rows.append(row)
    return rows


def main():
    # Small, interpretable grid: measured-mass neighborhood and historical nominal coefficient.
    m_values = [2.5, 3.0, 3.5]
    mhat_values = [3.0, 12.445, 15.0]
    rows = run_case('10mm_20s', smooth_10mm(0.001), m_values, mhat_values)
    rows += run_case('50mm_6s_saturated', constrained_50mm(0.001), m_values, mhat_values)
    df = pd.DataFrame(rows)
    csv = OUT / 'LC_BLNN_small_parameter_sensitivity.csv'; df.to_csv(csv, index=False, encoding='utf-8-sig')
    # Add LC-minus-Vanilla paired differences for direct interpretation.
    keys = ['case','m_G_kg','M_hat_y']
    v = df[df.method=='Vanilla BLNN'].set_index(keys)
    l = df[df.method=='Dynamic LC-BLNN'].set_index(keys)
    diff = l[['track_rmse_mm','sync_rms_mm','peak_Wy','sat_rate_pct']].subtract(v[['track_rmse_mm','sync_rms_mm','peak_Wy','sat_rate_pct']])
    diff = diff.reset_index(); diff.columns = keys + ['delta_track_rmse_mm','delta_sync_rms_mm','delta_peak_Wy','delta_sat_rate_pct']
    diff.to_csv(OUT / 'LC_minus_Vanilla_differences.csv', index=False, encoding='utf-8-sig')
    # Heatmaps for the principal B-case metrics.
    for case in ['10mm_20s','50mm_6s_saturated']:
        sub = df[(df.case==case) & (df.method=='Dynamic LC-BLNN')]
        fig, ax = plt.subplots(1,3, figsize=(12,3.5), constrained_layout=True)
        for a, metric, title in zip(ax, ['track_rmse_mm','sync_rms_mm','peak_Wy'], ['Tracking RMSE (mm)','Sync RMS (mm)','Peak ||Wy||']):
            mat = sub.pivot(index='m_G_kg', columns='M_hat_y', values=metric).sort_index()
            im = a.imshow(mat.values, aspect='auto', origin='lower')
            a.set_xticks(range(len(mat.columns)), [str(x) for x in mat.columns]); a.set_yticks(range(len(mat.index)), [str(x) for x in mat.index])
            a.set_xlabel('M_hat_y'); a.set_ylabel('m_G (kg)'); a.set_title(title)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]): a.text(j,i,f'{mat.iloc[i,j]:.3g}',ha='center',va='center',fontsize=8)
            fig.colorbar(im, ax=a, shrink=.8)
        fig.savefig(OUT / f'{case}_LC_sensitivity_heatmaps.png', dpi=220); plt.close(fig)
    summary = {
        'purpose':'Small parameter sensitivity only; no official experiment CSV overwritten.',
        'grid_m_G_kg':m_values, 'grid_M_hat_y':mhat_values,
        'cases':['10mm_20s','50mm_6s_saturated'],
        'methods':['Vanilla BLNN','Dynamic LC-BLNN'],
        'interpretation':'Use trends to assess sensitivity to physical mass and nominal controller coefficient; do not call this absolute parameter identification.'
    }
    import json
    (OUT/'README.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(df.to_string(index=False))
    print(f'Wrote {csv}')

if __name__ == '__main__': main()
