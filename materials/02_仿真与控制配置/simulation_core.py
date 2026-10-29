# Shared simulation core for exact multistroke benchmark
import numpy as np
from config_e5_final import ControllerConfig, PlantConfig
from controller_e5_final_lc import E5FinalController

def rk4_step(state, FG, Talpha, dt, p_cfg: PlantConfig):
    """4阶 Runge-Kutta 动力学积分"""
    def deriv(s):
        yG, dyG, a, da = s
        Le = p_cfg.Le
        vL = dyG - 0.5 * Le * da
        vR = dyG + 0.5 * Le * da
        eps_v = 0.01
        FL_f = p_cfg.b_L * vL + p_cfg.f_c_L * np.tanh(vL / eps_v)
        FR_f = p_cfg.b_R * vR + p_cfg.f_c_R * np.tanh(vR / eps_v)
        FG_f = FL_f + FR_f
        T_f = 0.5 * Le * (FR_f - FL_f)
        ddy = (FG - FG_f) / p_cfg.m_G
        dda = (Talpha - T_f - p_cfg.K_alpha * a - p_cfg.B_alpha * da) / p_cfg.J_alpha
        return np.array([dyG, ddy, da, dda], dtype=np.float64)

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_single_simulation(t, yd, yd_dot, yd_ddot, c_cfg: ControllerConfig, p_cfg: PlantConfig, mode: str, fixed_c: tuple = (1.0, 1.0)):
    """运行单次闭环仿真"""
    N = len(t)
    dt = c_cfg.Ts
    ctrl = E5FinalController(cfg=c_cfg, mode=mode, fixed_c=fixed_c)
    
    plant_state = np.zeros(4, dtype=np.float64)
    rec = {
        "t": t,
        "yd": yd,
        "yG": np.zeros(N),
        "alpha": np.zeros(N),
        "e_y": np.zeros(N),
        "e_sync": np.zeros(N),
        "iL": np.zeros(N),
        "iR": np.zeros(N),
        "norm_Wy": np.zeros(N),
        "norm_Wa": np.zeros(N),
        "rho_y": np.zeros(N),
        "sat_alloc": np.zeros(N, dtype=bool)
    }
    
    for k in range(N):
        q_meas = np.array([plant_state[0], plant_state[2]])
        qdot_meas = np.array([plant_state[1], plant_state[3]])
        qddot_filt = np.array([yd_ddot[k], 0.0])
        qd_k = np.array([yd[k], 0.0])
        qdot_d_k = np.array([yd_dot[k], 0.0])
        qddot_d_k = np.array([yd_ddot[k], 0.0])
        
        iL, iR, info = ctrl.step(q_meas, qdot_meas, qddot_filt, qd_k, qdot_d_k, qddot_d_k)
        
        FL_real = c_cfg.Kf_L * iL
        FR_real = -c_cfg.Kf_R * iR
        FG_act = FL_real + FR_real
        Talpha_act = 0.5 * c_cfg.Le * (FR_real - FL_real)
        
        rec["yG"][k] = plant_state[0]
        rec["alpha"][k] = plant_state[2]
        rec["e_y"][k] = plant_state[0] - yd[k]
        rec["e_sync"][k] = -c_cfg.Le * plant_state[2]
        rec["iL"][k] = iL
        rec["iR"][k] = iR
        rec["norm_Wy"][k] = info["norm_Wy"]
        rec["norm_Wa"][k] = info["norm_Wa"]
        rec["rho_y"][k] = info["rho_y"]
        rec["sat_alloc"][k] = info["sat_alloc"]
        
        plant_state = rk4_step(plant_state, FG_act, Talpha_act, dt, p_cfg)
        
    return rec


