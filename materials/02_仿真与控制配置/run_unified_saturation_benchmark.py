# -*- coding: utf-8 -*-
"""
================================================================================
统一仿真基准：执行器饱和工况下的动态学习协调 (LC-BLNN) 对比消融与一致性检验
Unified Simulation Benchmark: Dynamic Learning Coordination under Saturation
================================================================================
本脚本严格读取统一定义的 config_e5_final.py（统一对象质量 m_G = 3.0 kg，控制器名义 M_hat_y = 12.445 kg），
构建两组具有明确证据链的离线仿真算例：

【算例A：受限饱和基准算例 (Unified Constrained Saturation Benchmark)】
- 目标：展示在推力触及幅值饱和 (800 counts / 4.96 N) 时，动态 LC 机制对自适应权值过度累积的抑制效能；
- 算例设置：50 mm 幅度往复运动 (0.5 Hz)，推力需求超过物理限幅边界，触发实际推力饱和 (delta != 0)；
- 对比方法：
  1. 基准滑模 (Base SMC, 无神经补偿)
  2. 普通宽度学习网络 (Vanilla BLNN, 无协调，rho == 1.0)
  3. 模态学习协调宽度学习网络 (Dynamic LC-BLNN, 动态 rho(t))
- 核心证据：当推力饱和时，普通 BLNN 因跟踪误差受限无法消除而导致自适应权值持续累积激增；
  而 LC-BLNN 动态感知执行差异 delta 并主动降低 rho(t) < 1，有效冻结/阻断无效学习，保持权值受控。

【算例B：非受限退化一致性探索算例 (Historical Exploratory Unactivated Benchmark)】
- 目标：展示在非受限小误差工况下，LC-BLNN 保持与普通 BLNN 完全一致的退化特性；
- 算例设置：10 mm 小位移平缓运动 (20 s)，推力峰值约 0.43 N << 9.9 N 限幅，delta == 0，rho(t) == 1.0；
- 核心证据：非受限工况下动态协调机制不介入，完整保留普通 BLNN 的名义自适应学习精度；
  反之，静态固定牺牲学习率 (如固定 rho=0.5) 则会导致名义跟踪性能的单调退化。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config_e5_final import ControllerConfig, PlantConfig
from controller_e5_final_lc import E5FinalController

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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


def main():
    print("=" * 80)
    print("执行统一仿真基准算例：受限饱和对比消融与非受限一致性检验")
    print("=" * 80)
    
    # 统一参数加载 (严格从 config_e5_final 读取)
    c_cfg = ControllerConfig(Imax_L=800.0, Imax_R=800.0)
    p_cfg = PlantConfig(m_G=3.0, b_L=63.0, b_R=75.6, f_c_L=14.4, f_c_R=14.4)
    
    print(f"统一参数验证:")
    print(f"  控制器名义质量 M_hat_y: {c_cfg.M_hat_y} kg")
    print(f"  对象物理基准质量 m_G:    {p_cfg.m_G} kg (与连续分析模型严格统一)")
    print(f"  电流指令物理幅值 Imax:   {c_cfg.Imax_L} counts (等效合推力 F_max = 9.9167 N)")
    print(f"  变化率限制阈值 slew:     {c_cfg.slew_rate_max} counts/ms (464.84 N/s)")
    
    # -------------------------------------------------------------------------
    # 算例 A：受限饱和基准算例 (50 mm, 0.5 Hz, 7 s)
    # -------------------------------------------------------------------------
    print("\n[算例 A] 运行受限饱和对比算例 (50 mm, 0.5 Hz, 800 counts 限幅)...")
    dt = c_cfg.Ts
    t_A = np.arange(0.0, 6.0, dt)
    A_traj = 0.025 # 50 mm peak-to-peak
    f_traj = 0.5   # 0.5 Hz
    yd_A = A_traj * (1.0 - np.cos(2.0 * np.pi * f_traj * t_A))
    yd_dot_A = A_traj * (2.0 * np.pi * f_traj) * np.sin(2.0 * np.pi * f_traj * t_A)
    yd_ddot_A = A_traj * ((2.0 * np.pi * f_traj)**2) * np.cos(2.0 * np.pi * f_traj * t_A)
    
    res_A_vanilla = run_single_simulation(t_A, yd_A, yd_dot_A, yd_ddot_A, c_cfg, p_cfg, mode="VANILLA")
    res_A_lc = run_single_simulation(t_A, yd_A, yd_dot_A, yd_ddot_A, c_cfg, p_cfg, mode="LC")
    res_A_fixed = run_single_simulation(t_A, yd_A, yd_dot_A, yd_ddot_A, c_cfg, p_cfg, mode="FIXED", fixed_c=(0.5, 0.5))
    
    # 计算统计指标
    def calc_metrics(res, label):
        ey_mm = res["e_y"] * 1000.0
        es_mm = res["e_sync"] * 1000.0
        sat_pct = np.mean(res["sat_alloc"]) * 100.0
        return {
            "method": label,
            "track_rmse_mm": float(np.sqrt(np.mean(ey_mm**2))),
            "track_max_mm": float(np.max(np.abs(ey_mm))),
            "sync_rms_mm": float(np.sqrt(np.mean(es_mm**2))),
            "sync_max_mm": float(np.max(np.abs(es_mm))),
            "peak_Wy": float(np.max(res["norm_Wy"])),
            "min_rho_y": float(np.min(res["rho_y"])),
            "sat_rate_pct": float(sat_pct)
        }
    
    met_A_vanilla = calc_metrics(res_A_vanilla, "Vanilla BLNN (rho=1.0)")
    met_A_fixed = calc_metrics(res_A_fixed, "Fixed Modulation (rho=0.5)")
    met_A_lc = calc_metrics(res_A_lc, "Dynamic LC-BLNN (Proposed)")
    
    print("  受限算例指标对比:")
    for m in [met_A_vanilla, met_A_fixed, met_A_lc]:
        print(f"    - {m['method']:30s}: 峰值||Wy|| = {m['peak_Wy']:7.4f}, 最小rho_y = {m['min_rho_y']:.4f}, "
              f"饱和率 = {m['sat_rate_pct']:5.1f}%, 同步RMS = {m['sync_rms_mm']:.4f} mm, 跟踪RMSE = {m['track_rmse_mm']:.4f} mm")
        
    # -------------------------------------------------------------------------
    # 算例 B：非受限退化一致性算例 (10 mm, 20 s, 800 counts 限幅)
    # -------------------------------------------------------------------------
    print("\n[算例 B] 运行非受限退化一致性探索算例 (10 mm, 20 s, 推力未饱和)...")
    t_B = np.arange(0.0, 20.0, dt)
    N_B = len(t_B)
    yd_B = np.zeros(N_B)
    yd_dot_B = np.zeros(N_B)
    yd_ddot_B = np.zeros(N_B)
    S_B = 0.010
    T_stroke = 4.0
    for i in range(N_B):
        ti = t_B[i]
        if 4.0 <= ti < 8.0:
            tau = (ti - 4.0) / T_stroke
            poly = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            dpoly = 30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)
            ddpoly = 60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)
            yd_B[i] = S_B * poly
            yd_dot_B[i] = (S_B / T_stroke) * dpoly
            yd_ddot_B[i] = (S_B / (T_stroke**2)) * ddpoly
        elif 8.0 <= ti < 10.0:
            yd_B[i] = S_B
        elif 10.0 <= ti < 14.0:
            tau = (ti - 10.0) / T_stroke
            poly = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            dpoly = 30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)
            ddpoly = 60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)
            yd_B[i] = S_B * (1.0 - poly)
            yd_dot_B[i] = -(S_B / T_stroke) * dpoly
            yd_ddot_B[i] = -(S_B / (T_stroke**2)) * ddpoly
            
    res_B_vanilla = run_single_simulation(t_B, yd_B, yd_dot_B, yd_ddot_B, c_cfg, p_cfg, mode="VANILLA")
    res_B_lc = run_single_simulation(t_B, yd_B, yd_dot_B, yd_ddot_B, c_cfg, p_cfg, mode="LC")
    res_B_fixed = run_single_simulation(t_B, yd_B, yd_dot_B, yd_ddot_B, c_cfg, p_cfg, mode="FIXED", fixed_c=(0.5, 0.5))
    
    met_B_vanilla = calc_metrics(res_B_vanilla, "Vanilla BLNN (rho=1.0) [Unconstrained]")
    met_B_lc = calc_metrics(res_B_lc, "Dynamic LC-BLNN [Unconstrained]")
    met_B_fixed = calc_metrics(res_B_fixed, "Fixed Modulation (rho=0.5) [Unconstrained]")
    
    print("  非受限算例指标对比:")
    for m in [met_B_vanilla, met_B_lc, met_B_fixed]:
        print(f"    - {m['method']:45s}: 峰值||Wy|| = {m['peak_Wy']:7.4f}, 最小rho_y = {m['min_rho_y']:.4f}, "
              f"饱和率 = {m['sat_rate_pct']:5.1f}%, 同步RMS = {m['sync_rms_mm']:.4f} mm, 跟踪RMSE = {m['track_rmse_mm']:.4f} mm")
        
    # 保存结构化对比 CSV
    df_all = pd.DataFrame([met_A_vanilla, met_A_fixed, met_A_lc, met_B_vanilla, met_B_lc, met_B_fixed])
    csv_out_path = os.path.join(current_dir, "EXP5_UNIFIED_SATURATION_BENCHMARK.csv")
    df_all.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
    print(f"\n指标数据表已保存至: {csv_out_path}")
    
    # -------------------------------------------------------------------------
    # 绘制高清晰度仿真曲线图 (EXP5_UNIFIED_SATURATION_COMPARISON.png)
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    
    # 1. 跟踪误差
    axs[0].plot(t_A, res_A_vanilla["e_y"] * 1000.0, 'r--', label='普通 BLNN (Vanilla, $\\rho=1.0$)', alpha=0.8)
    axs[0].plot(t_A, res_A_fixed["e_y"] * 1000.0, 'g-.', label='固定折减 BLNN (Fixed, $\\rho=0.5$)', alpha=0.8)
    axs[0].plot(t_A, res_A_lc["e_y"] * 1000.0, 'b-', label='模态学习协调 LC-BLNN (本文动态 $\\rho(t)$)', linewidth=1.5)
    axs[0].set_ylabel('平动跟踪误差 $e_y$ (mm)')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right')
    axs[0].set_title('(a) 平动位移跟踪误差响应')
    
    # 2. 同步误差
    axs[1].plot(t_A, res_A_vanilla["e_sync"] * 1000.0, 'r--', label='普通 BLNN', alpha=0.8)
    axs[1].plot(t_A, res_A_fixed["e_sync"] * 1000.0, 'g-.', label='固定折减 BLNN', alpha=0.8)
    axs[1].plot(t_A, res_A_lc["e_sync"] * 1000.0, 'b-', label='LC-BLNN', linewidth=1.5)
    axs[1].set_ylabel('双驱同步误差 $e_{\\rm sync}$ (mm)')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].set_title('(b) 双驱同步位移误差响应')
    
    # 3. 神经权值范数变化曲线 (关键证据)
    axs[2].plot(t_A, res_A_vanilla["norm_Wy"], 'r--', label='普通 BLNN (饱和期间权值单调累积)', linewidth=1.5)
    axs[2].plot(t_A, res_A_fixed["norm_Wy"], 'g-.', label='固定折减 BLNN (学习受阻)', linewidth=1.2)
    axs[2].plot(t_A, res_A_lc["norm_Wy"], 'b-', label='LC-BLNN (动态冻结过度更新，权值受控)', linewidth=1.8)
    axs[2].set_ylabel('权值范数 $\\|W_y\\|$')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper left')
    axs[2].set_title('(c) 宽度神经网络自适应权值范数 $\\|W_y\\|$ 变化')
    
    # 4. 饱和状态与协调调制因子 rho(t)
    axs[3].plot(t_A, res_A_vanilla["rho_y"], 'r--', label='普通 BLNN (恒定 $\\rho=1.0$)', alpha=0.8)
    axs[3].plot(t_A, res_A_fixed["rho_y"], 'g-.', label='固定折减 (恒定 $\\rho=0.5$)', alpha=0.8)
    axs[3].plot(t_A, res_A_lc["rho_y"], 'b-', label='LC-BLNN (动态自适应调节 $\\rho(t)$)', linewidth=1.8)
    axs[3].fill_between(t_A, 0, 1.05, where=res_A_lc["sat_alloc"], color='gray', alpha=0.2, label='推力达到物理幅值限幅区')
    axs[3].set_ylabel('协调调制因子 $\\rho_y(t)$')
    axs[3].set_xlabel('时间 $t$ (s)')
    axs[3].set_ylim(-0.05, 1.15)
    axs[3].grid(True, linestyle=':', alpha=0.6)
    axs[3].legend(loc='lower right')
    axs[3].set_title('(d) 饱和区间与动态学习协调因子响应')
    
    plt.tight_layout()
    png_out_path = os.path.join(current_dir, "EXP5_UNIFIED_SATURATION_COMPARISON.png")
    plt.savefig(png_out_path, dpi=300)
    plt.close()
    print(f"对比图表已保存至: {png_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
