# Plant + firmware-faithful measurement chain, with calibratable physical parameters.
#   m_eff     effective translational inertia (kg)  (3 kg cart + reflected motor/gear inertia)
#   fc_tot    total Coulomb friction (N), split L/R by asym
#   fs_ratio  static/Coulomb friction ratio (Stribeck), vs = Stribeck velocity
#   b_tot     total viscous damping (N s/m)
#   asym      right/left friction & damping ratio
#   k_scale   scale on torsional stiffness/inertia/damping of the beam (sync dynamics)
#   noise_rpm Gaussian noise on each motor's speed_rpm feedback (before integer rounding)
import numpy as np
from dataclasses import dataclass
from config_e5_final import ControllerConfig
from controller_e5_final_lc import E5FinalController

R_DRIVE = 0.030; GEAR = 19.0; CPR = 8192.0
M_PER_COUNT = 2*np.pi*R_DRIVE/(GEAR*CPR)
MPS_PER_RPM = (1/60.0)*2*np.pi*R_DRIVE/GEAR

@dataclass
class Plant:
    m_eff: float = 3.0
    fc_tot: float = 2.0
    fs_ratio: float = 1.0
    vs: float = 0.002
    b_tot: float = 10.0
    asym: float = 1.2
    k_scale: float = 1.0
    noise_rpm: float = 25.0
    fext_L: float = 0.0          # external resistance on left side (N), Coulomb-type, opposes motion
    Le: float = 0.6
    J_alpha: float = 0.37335
    K_alpha: float = 2000.0
    B_alpha: float = 22.5


def friction(v, fc, fs, vs, b):
    g = fc + (fs - fc)*np.exp(-(v/vs)**2)
    return b*v + g*np.tanh(v/1e-3)


def make_deriv(p: Plant):
    fcL = p.fc_tot/(1+p.asym); fcR = p.asym*fcL
    bL = p.b_tot/(1+p.asym); bR = p.asym*bL
    J = p.J_alpha; K = p.K_alpha*p.k_scale; B = p.B_alpha*p.k_scale
    def deriv(s, FG, Ta):
        yG, dyG, a, da = s
        vL = dyG - 0.5*p.Le*da; vR = dyG + 0.5*p.Le*da
        FLf = friction(vL, fcL, p.fs_ratio*fcL, p.vs, bL) + p.fext_L*np.tanh(vL/1e-3)
        FRf = friction(vR, fcR, p.fs_ratio*fcR, p.vs, bR)
        ddy = (FG - (FLf + FRf))/p.m_eff
        dda = (Ta - 0.5*p.Le*(FRf - FLf) - K*a - B*da)/J
        return np.array([dyG, ddy, da, dda])
    return deriv


def run(t, yd, ydd, yddd, c: ControllerConfig, p: Plant, mode, fixed_c=(1.0, 1.0), seed=0, substeps=4):
    rng = np.random.default_rng(seed)
    N = len(t); dt = c.Ts; Le = c.Le; h = dt/substeps
    deriv = make_deriv(p)
    ctrl = E5FinalController(cfg=c, mode=mode, fixed_c=fixed_c)
    x = np.zeros(4)
    keys = ['e_y', 'e_sync', 'posL', 'posR', 'yG', 'norm_Wy', 'norm_Wa', 'rho_y', 'rho_a', 'iL', 'iR']
    rec = {k: np.zeros(N) for k in keys}; rec['sat_alloc'] = np.zeros(N, dtype=bool)
    v_prev = np.zeros(2); qdd_f = np.zeros(2)
    for k in range(N):
        yG, dyG, a, da = x
        posL = yG - 0.5*Le*a; posR = yG + 0.5*Le*a
        velL = dyG - 0.5*Le*da; velR = dyG + 0.5*Le*da
        pL = np.round(posL/M_PER_COUNT)*M_PER_COUNT; pR = np.round(posR/M_PER_COUNT)*M_PER_COUNT
        vL = np.round(velL/MPS_PER_RPM + p.noise_rpm*rng.standard_normal())*MPS_PER_RPM
        vR = np.round(velR/MPS_PER_RPM + p.noise_rpm*rng.standard_normal())*MPS_PER_RPM
        q = np.array([0.5*(pL+pR), (pR-pL)/Le]); qd_ = np.array([0.5*(vL+vR), (vR-vL)/Le])
        raw = (qd_ - v_prev)/dt; v_prev = qd_; qdd_f = 0.76*qdd_f + 0.24*raw
        iL, iR, info = ctrl.step(q, qd_, qdd_f.copy(), np.array([yd[k], 0.0]),
                                 np.array([ydd[k], 0.0]), np.array([yddd[k], 0.0]))
        FL = c.Kf_L*iL; FR = -c.Kf_R*iR; FG = FL + FR; Ta = 0.5*Le*(FR - FL)
        rec['e_y'][k] = pL*0.5 + pR*0.5 - yd[k]      # measured (encoder) error, like hardware
        rec['e_sync'][k] = pR - pL                  # firmware e_sync_m
        rec['posL'][k] = pL; rec['posR'][k] = pR; rec['yG'][k] = 0.5*(pL+pR)
        rec['norm_Wy'][k] = info['norm_Wy']; rec['norm_Wa'][k] = info['norm_Wa']
        rec['rho_y'][k] = info['rho_y']; rec['rho_a'][k] = info['rho_alpha']
        rec['iL'][k] = iL; rec['iR'][k] = iR; rec['sat_alloc'][k] = info['sat_alloc']
        for _ in range(substeps):                     # RK4 with substeps (stiff friction)
            k1 = deriv(x, FG, Ta); k2 = deriv(x + 0.5*h*k1, FG, Ta)
            k3 = deriv(x + 0.5*h*k2, FG, Ta); k4 = deriv(x + h*k3, FG, Ta)
            x = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return rec


def metrics(rec, t, stroke_mm, imax=800.0):
    w = t >= 4.0                                   # 16 s analysis window (~8000 frames @500 Hz)
    lim = 0.99*imax
    ey = rec['e_y'][w]*1e3; es = rec['e_sync'][w]*1e3
    return dict(track_rmse_mm=np.sqrt(np.mean(ey**2)), track_peak_mm=np.max(np.abs(ey)),
                sync_rms_mm=np.sqrt(np.mean(es**2)), sync_peak_mm=np.max(np.abs(es)),
                satL_pct=100*np.mean(np.abs(rec['iL'][w]) >= lim), satR_pct=100*np.mean(np.abs(rec['iR'][w]) >= lim),
                sat_alloc_pct=100*np.mean(rec['sat_alloc'][w]),
                resid_abs_mm=abs(rec['yG'][-1]*1e3),
                overshoot_mm=max(rec['posL'].max(), rec['posR'].max())*1e3 - stroke_mm,
                wy_peak=rec['norm_Wy'].max(), wa_peak=rec['norm_Wa'].max(),
                rho_y_mean=np.mean(rec['rho_y'][w]), rho_a_mean=np.mean(rec['rho_a'][w]))
