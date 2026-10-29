# -*- coding: utf-8 -*-
"""逐行程仿真：严格复现10–300 mm固件轨迹。
与现有50 mm深饱和算例保持同一对象、控制器、0.5 Hz往返轨迹、800 counts限幅。
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config_e5_final import ControllerConfig, PlantConfig
from simulation_core import run_single_simulation

OUT = HERE / 'multistroke_simulation'
OUT.mkdir(exist_ok=True)


def trajectory(stroke_mm, dt=0.001, duration=20.0):
    """严格复现各行程 motor_task.c 的 20 s 五次平滑轨迹。"""
    t = np.arange(0.0, duration, dt)
    stroke = stroke_mm / 1000.0
    yd = np.zeros_like(t); yd_dot = np.zeros_like(t); yd_ddot = np.zeros_like(t)
    def smooth(u):
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = 30*u**2 - 60*u**3 + 30*u**4
        dds = 60*u - 180*u**2 + 120*u**3
        return s, ds, dds
    # Firmware: pre-hold [0,4), forward [4,8), endpoint hold [8,10),
    # return [10,14), recovery hold [14,20).
    fwd = (t >= 4.0) & (t < 8.0)
    u = (t[fwd]-4.0)/4.0; ss, ds, dds = smooth(u)
    yd[fwd] = stroke*ss; yd_dot[fwd] = stroke*ds/4.0; yd_ddot[fwd] = stroke*dds/16.0
    hold = (t >= 8.0) & (t < 10.0); yd[hold] = stroke
    ret = (t >= 10.0) & (t < 14.0)
    u = (t[ret]-10.0)/4.0; ss, ds, dds = smooth(u)
    yd[ret] = stroke*(1.0-ss); yd_dot[ret] = -stroke*ds/4.0; yd_ddot[ret] = -stroke*dds/16.0
    return t, yd, yd_dot, yd_ddot


def metrics(rec, method, stroke_mm):
    ey = rec['e_y'] * 1000.0
    es = rec['e_sync'] * 1000.0
    return {
        'stroke_mm': stroke_mm,
        'method': method,
        'track_rmse_mm': float(np.sqrt(np.mean(ey**2))),
        'track_max_mm': float(np.max(np.abs(ey))),
        'sync_rms_mm': float(np.sqrt(np.mean(es**2))),
        'sync_max_mm': float(np.max(np.abs(es))),
        'peak_Wy': float(np.max(rec['norm_Wy'])),
        'end_Wy': float(rec['norm_Wy'][-1]),
        'min_rho_y': float(np.min(rec['rho_y'])),
        'sat_rate_pct': float(np.mean(rec['sat_alloc']) * 100.0),
    }


def main():
    c = ControllerConfig(Imax_L=800.0, Imax_R=800.0)
    p = PlantConfig(m_G=3.0, b_L=63.0, b_R=75.6, f_c_L=14.4, f_c_R=14.4)
    rows = []
    strokes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [10, 20, 30, 40, 50, 100, 200, 300]
    for stroke in strokes:
        print(f'Running {stroke} mm ...', flush=True)
        t, yd, yd_dot, yd_ddot = trajectory(stroke, c.Ts)
        cases = [
            ('BLNN', 'VANILLA', (1.0, 1.0)),
            ('Fixed modulation', 'FIXED', (0.5, 0.5)),
            ('LC-BLNN', 'LC', (1.0, 1.0)),
        ]
        for label, mode, fixed in cases:
            rec = run_single_simulation(t, yd, yd_dot, yd_ddot, c, p, mode=mode, fixed_c=fixed)
            rows.append(metrics(rec, label, stroke))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'EXP5_MULTISTROKE_SIMULATION_SUMMARY.csv', index=False, encoding='utf-8-sig')
    # grouped comparison relative to BLNN
    base = df[df.method == 'BLNN'].set_index('stroke_mm')
    comp = []
    for _, r in df.iterrows():
        b = base.loc[r.stroke_mm]
        comp.append({
            'stroke_mm': int(r.stroke_mm), 'method': r.method,
            'track_rmse_change_pct': float((r.track_rmse_mm / b.track_rmse_mm - 1) * 100),
            'sync_rms_change_pct': float((r.sync_rms_mm / b.sync_rms_mm - 1) * 100),
            'peak_Wy_change_pct': float((r.peak_Wy / b.peak_Wy - 1) * 100),
        })
    pd.DataFrame(comp).to_csv(OUT / 'EXP5_MULTISTROKE_SIMULATION_RELATIVE_TO_BLNN.csv', index=False, encoding='utf-8-sig')
    meta = {
        'strokes_mm': [10, 20, 30, 40, 50, 100, 200, 300],
        'methods': ['BLNN', 'Fixed modulation', 'LC-BLNN'],
        'trajectory': '严格复现 motor_task.c：4s预保持+4s五次平滑前进+2s保持+4s五次平滑返回+6s恢复保持，总时长20s',
        'controller_limits_counts': 800,
        'plant_mass_kg': 3.0,
        'controller_nominal_mass_kg': 12.445,
        'note': '严格按各行程固件20 s参考轨迹生成10–300 mm一一对应仿真；不修改实物数据。',
    }
    (OUT / 'README.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

