# Comparison methods: GLOBAL (single scalar from overall saturation) and FREEZE (stop update when saturated)
# Usage: python bench_cmp.py '<plant json>' <methods comma> <out.csv>
import sys, json, pandas as pd
from multiprocessing import Pool
from config_e5_final import ControllerConfig
from trajectory import trajectory
from plant_sim import Plant, run, metrics

P = json.loads(sys.argv[1]); METHODS = sys.argv[2].split(','); OUT = sys.argv[3]
LIM = float(sys.argv[4]) if len(sys.argv) > 4 else 800.0
STROKES = [10, 20, 30, 40, 50, 100, 200, 300]; NBLK = 10
LABEL = {'BASE': 'M1_BASE', 'VANILLA': 'M2_BLNN', 'LC': 'M3_LCBLNN', 'GLOBAL': 'Global_scalar', 'FREEZE': 'Freeze_sat'}

def job(arg):
    s, blk = arg
    t, a, b, d = trajectory(s, 0.001); p = Plant(**P); seed = 100*s + blk; rows = []; m3 = None
    for meth in METHODS:
        if meth == 'FIXED':
            continue
        cfg = ControllerConfig(enable_blnn=False, Imax_L=LIM, Imax_R=LIM) if meth == 'BASE' else ControllerConfig(Imax_L=LIM, Imax_R=LIM)
        mode = 'VANILLA' if meth == 'BASE' else meth
        m = metrics(run(t, a, b, d, cfg, p, mode, seed=seed), t, s, imax=LIM)
        m.update(stroke_mm=s, block=blk, method=LABEL[meth]); rows.append(m)
        if meth == 'LC': m3 = m
    if 'FIXED' in METHODS and m3 is not None:
        m = metrics(run(t, a, b, d, ControllerConfig(Imax_L=LIM, Imax_R=LIM), p, 'FIXED', fixed_c=(m3['rho_y_mean'], m3['rho_a_mean']), seed=seed), t, s, imax=LIM)
        m.update(stroke_mm=s, block=blk, method='Fixed_matched'); rows.append(m)
    return rows

if __name__ == '__main__':
    with Pool(2) as pool:
        res = pool.map(job, [(s, b) for s in STROKES for b in range(1, NBLK+1)])
    pd.DataFrame([r for rs in res for r in rs]).to_csv(OUT, index=False, encoding='utf-8-sig')
    print('done', OUT)
