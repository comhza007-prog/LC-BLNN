# Validation: calibrated plant (fixed, from M1 only) -> predict M1/M2/Fixed-matched/M3 for all strokes.
# Block-to-block difference comes only from the measurement-noise seed (same seed for all methods in a block).
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
from config_e5_final import ControllerConfig
from trajectory import trajectory
from plant_sim import Plant, run, metrics

P = json.loads(sys.argv[1]); NBLK = int(sys.argv[2]); OUT = sys.argv[3]
STROKES = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [10, 20, 30, 40, 50, 100, 200, 300]

def job(arg):
    s, blk = arg
    t, a, b, d = trajectory(s, 0.001); p = Plant(**P); seed = 100*s + blk; rows = []
    def add(rec, lab):
        m = metrics(rec, t, s); m.update(stroke_mm=s, block=blk, method=lab); rows.append(m); return m
    add(run(t, a, b, d, ControllerConfig(enable_blnn=False), p, 'VANILLA', seed=seed), 'M1_BASE')
    add(run(t, a, b, d, ControllerConfig(), p, 'VANILLA', seed=seed), 'M2_BLNN')
    m3 = add(run(t, a, b, d, ControllerConfig(), p, 'LC', seed=seed), 'M3_LCBLNN')
    add(run(t, a, b, d, ControllerConfig(), p, 'FIXED', fixed_c=(m3['rho_y_mean'], m3['rho_a_mean']), seed=seed), 'Fixed_matched')
    return rows

if __name__ == '__main__':
    jobs = [(s, b) for s in STROKES for b in range(1, NBLK+1)]
    with Pool(2) as pool:
        res = pool.map(job, jobs)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(OUT, index=False, encoding='utf-8-sig')
    print('done', OUT)
