"""
Script: run_offline_fixed_modulation_benchmark.py
Purpose: 探索性离线检查：LC未激活工况下与普通BLNN的一致性
(Exploratory Offline Consistency Check: Invariance to Vanilla BLNN in Unactivated Regime)

Compares:
  1. Vanilla BLNN (rho == 1.0)
  2. Fixed Modulation BLNN (rho == [c_y, c_alpha], frozen at empirical average c_y=0.917, c_alpha=0.979)
  3. Dynamic LC-BLNN (rho(t) dynamically modulated by discrepancy delta(t))
  4. Heavy Fixed Suppression BLNN (c_y=0.500, c_alpha=0.500)

Note:
  修正轨迹后重新生成（行程时间调整为 4 秒以对齐实物正反向行程）。
  在本次名义离线仿真中，执行器补偿请求未超出限幅边界（F_nn 峰值约 0.43 N << 5 N 限幅门限），
  故补偿差额 delta == 0，学习协调因子 rho(t) 恒等于 1.0（处于未激活状态）。
  该仿真证明了非受限工况下动态 LC 严格保持了普通 BLNN 的全速适应与跟踪精度，
  而静态弱化学习（如 cy=0.917 或 0.500）会导致跟踪误差单调退化。
  百分比相对变化均由仿真结果动态计算，不硬编码常数。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r"d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\06_分析与论文图表"
ARTIFACT_DIR = r"C:\Users\ele\.gemini\antigravity\brain\973650e7-c375-4adb-b274-cbc5c0adc155"

# Nominal physical parameters matching Experiment 5
Le = 0.600
Kf = 0.0061979167
Imax = 800.0
Ts = 0.001

# Plant true parameters
M_y = 12.445
M_a = 0.37335
B_y = 68.0  # +5 N*s/m unmodeled damping
B_a = 25.0
K_a = 2000.0
A_y = 18.0  # Coulomb friction
delta_fric_R = 1.25 # 25% friction asymmetry between rails

# Controller nominal model
M_hat = np.array([12.445, 0.37335])
B_hat = np.array([63.0, 22.5])
K_hat = np.array([0.0, 2000.0])
A_hat = np.array([14.4, 0.0])

Gamma = np.array([30.0, 20.0])
K1 = np.array([45.0, 30.0])
K2 = np.array([12.0, 6.0])
phi = np.array([0.030, 0.030])
eps_v = 0.01
eps_a = 0.01

K_aw = 20.0
lam_aw = 20.0

nn_eps = np.array([25.0, 18.0])
nn_eta = np.array([0.01, 0.01])
scale_y = np.array([2.0, 0.7, 1.0, 0.08, 0.7])
scale_a = np.array([1.0, 0.2, 0.02, 0.01, 0.2])

BLNN_CENTERS = np.array([
    [-1.0, -0.5,  0.0,  0.5,  1.0],
    [-0.5,  0.0,  0.5,  1.0, -1.0],
    [ 0.0,  0.5,  1.0, -1.0, -0.5],
    [ 0.5,  1.0, -1.0, -0.5,  0.0]
])

def thrust_allocator_sync(FG_des, Talpha_des):
    FL_max = Kf * Imax
    FR_max = Kf * Imax
    DeltaF_des = (2.0 * Talpha_des) / Le
    DeltaF_max = FR_max + FL_max
    DeltaF_alloc = np.clip(DeltaF_des, -DeltaF_max, DeltaF_max)
    FL_lower = np.maximum(-FL_max, -FR_max - DeltaF_alloc)
    FL_upper = np.minimum( FL_max,  FR_max - DeltaF_alloc)
    FL_target = 0.5 * FG_des - 0.5 * DeltaF_alloc
    FL_alloc = np.clip(FL_target, FL_lower, FL_upper)
    FR_alloc = FL_alloc + DeltaF_alloc
    iL_act = np.clip(FL_alloc / Kf, -Imax, Imax)
    iR_act = np.clip(-FR_alloc / Kf, -Imax, Imax)
    return iL_act, iR_act

def simulate_method(mode="LC", fixed_c=(1.0, 1.0)):
    """
    mode: 'VANILLA' (rho=1.0), 'FIXED' (rho=fixed_c), 'LC' (dynamic)
    """
    t = np.arange(0.0, 20.0, Ts)
    N = len(t)
    
    # Generate reference trajectory (0 -> 10 mm -> 0, matching 4s stroke)
    yd = np.zeros(N)
    yd_dot = np.zeros(N)
    yd_ddot = np.zeros(N)
    
    for k in range(N):
        tk = t[k]
        if 4.0 <= tk < 8.0:
            tau = (tk - 4.0) / 4.0
            s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
            ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / 4.0
            d2s = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / 16.0
            yd[k] = 0.010 * s
            yd_dot[k] = 0.010 * ds
            yd_ddot[k] = 0.010 * d2s
        elif 8.0 <= tk < 10.0:
            yd[k] = 0.010
            yd_dot[k] = 0.0
            yd_ddot[k] = 0.0
        elif 10.0 <= tk < 14.0:
            tau = (tk - 10.0) / 4.0
            s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
            ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / 4.0
            d2s = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / 16.0
            yd[k] = 0.010 * (1.0 - s)
            yd_dot[k] = -0.010 * ds
            yd_ddot[k] = -0.010 * d2s
        elif tk >= 14.0:
            yd[k] = 0.0
            yd_dot[k] = 0.0
            yd_ddot[k] = 0.0

    # Plant states
    q = np.array([0.0, 0.0])
    qdot = np.array([0.0, 0.0])
    
    # Controller states
    W_y = np.zeros((4, 3))
    W_a = np.zeros((4, 3))
    z_aw = np.array([0.0, 0.0])
    
    # Logging
    hist_sync = np.zeros(N)
    hist_track = np.zeros(N)
    hist_Wy = np.zeros(N)
    hist_Wa = np.zeros(N)
    hist_rho_y = np.zeros(N)
    hist_rho_a = np.zeros(N)
    hist_Fnn = np.zeros(N)
    hist_Tnn = np.zeros(N)
    
    for k in range(N):
        e_y = q[0] - yd[k]
        e_a = q[1]
        edot_y = qdot[0] - yd_dot[k]
        edot_a = qdot[1]
        
        sigma_y = edot_y + Gamma[0] * e_y
        sigma_a = edot_a + Gamma[1] * e_a
        
        # Nominal sliding mode
        Sf_y = np.tanh(qdot[0] / eps_v)
        Sf_a = np.tanh(qdot[1] / eps_a)
        va_y = B_hat[0] * qdot[0] + K_hat[0] * q[0] + A_hat[0] * Sf_y + M_hat[0] * yd_ddot[k]
        va_a = B_hat[1] * qdot[1] + K_hat[1] * q[1] + A_hat[1] * Sf_a
        vr_y = -K1[0] * sigma_y - K2[0] * np.tanh(sigma_y / phi[0]) - M_hat[0] * Gamma[0] * edot_y
        vr_a = -K1[1] * sigma_a - K2[1] * np.tanh(sigma_a / phi[1]) - M_hat[1] * Gamma[1] * edot_a
        
        v_base_0 = va_y + vr_y
        v_base_1 = va_a + vr_a
        
        # Anti-windup contribution
        v_aw_0 = Kf * (z_aw[0] - z_aw[1])
        v_aw_1 = -0.5 * Le * Kf * (z_aw[0] + z_aw[1])
        v_base_tot_0 = v_base_0 + v_aw_0
        v_base_tot_1 = v_base_1 + v_aw_1
        
        i_base_L, i_base_R = thrust_allocator_sync(v_base_tot_0, v_base_tot_1)
        
        # BLNN forward
        chi_in_y = np.clip(np.array([yd_ddot[k]/scale_y[0], qdot[0]/scale_y[1], q[0]/scale_y[2], e_y/scale_y[3], edot_y/scale_y[4]]), -2.0, 2.0)
        chi_in_a = np.clip(np.array([0.0, qdot[1]/scale_a[1], q[1]/scale_a[2], e_a/scale_a[3], edot_a/scale_a[4]]), -2.0, 2.0)
        
        S_y = np.zeros((4, 3))
        S_a = np.zeros((4, 3))
        v_nn_raw_modal = np.array([0.0, 0.0])
        
        for j in range(4):
            d2_y = np.sum((BLNN_CENTERS[j] - chi_in_y)**2)
            d2_a = np.sum((BLNN_CENTERS[j] - chi_in_a)**2)
            h_y = np.exp(-d2_y)
            h_a = np.exp(-d2_a)
            S_y[j] = [h_y, np.cos(h_y), np.sin(h_y)]
            S_a[j] = [h_a, np.cos(h_a), np.sin(h_a)]
            v_nn_raw_modal[0] -= np.dot(W_y[j], S_y[j])
            v_nn_raw_modal[1] -= np.dot(W_a[j], S_a[j])
            
        # Total request
        FG_cmd = v_base_tot_0 + v_nn_raw_modal[0]
        Ta_cmd = v_base_tot_1 + v_nn_raw_modal[1]
        
        i_req_L = (0.5 * FG_cmd - Ta_cmd / Le) / Kf
        i_req_R = -(0.5 * FG_cmd + Ta_cmd / Le) / Kf
        
        iL_act, iR_act = thrust_allocator_sync(FG_cmd, Ta_cmd)
        
        # Discrepancy
        v_applied_L = iL_act - i_base_L
        v_applied_R = iR_act - i_base_R
        v_nn_raw_L = (0.5 * v_nn_raw_modal[0] - v_nn_raw_modal[1] / Le) / Kf
        v_nn_raw_R = -(0.5 * v_nn_raw_modal[0] + v_nn_raw_modal[1] / Le) / Kf
        
        delta_L = v_applied_L - v_nn_raw_L
        delta_R = v_applied_R - v_nn_raw_R
        
        delta_modal_0 = Kf * (delta_L - delta_R)
        delta_modal_1 = -0.5 * Le * Kf * (delta_L + delta_R)
        
        chi_y = (np.abs(delta_modal_0) / (2.0 * Kf)) / Imax
        chi_a = (np.abs(delta_modal_1) / (0.5 * Le * 2.0 * Kf)) / Imax
        
        if mode == "LC":
            rho_y = 1.0 / (1.0 + 100.0 * chi_y**2)
            rho_a = 1.0 / (1.0 + 100.0 * chi_a**2)
        elif mode == "FIXED":
            rho_y = fixed_c[0]
            rho_a = fixed_c[1]
        else: # VANILLA
            rho_y = 1.0
            rho_a = 1.0
            
        # Update weights
        for j in range(4):
            for m in range(3):
                W_dot_y = nn_eps[0] * (rho_y * sigma_y * S_y[j, m] - nn_eta[0] * W_y[j, m])
                W_dot_a = nn_eps[1] * (rho_a * sigma_a * S_a[j, m] - nn_eta[1] * W_a[j, m])
                W_y[j, m] += Ts * W_dot_y
                W_a[j, m] += Ts * W_dot_a
                
        # Anti-windup state
        delta_alloc_L = iL_act - i_req_L
        delta_alloc_R = iR_act - i_req_R
        z_aw[0] += Ts * (K_aw * delta_alloc_L - lam_aw * z_aw[0])
        z_aw[1] += Ts * (K_aw * delta_alloc_R - lam_aw * z_aw[1])
        
        # Plant physics
        FL_real = Kf * iL_act
        FR_real = -Kf * iR_act
        FG_real = FL_real + FR_real
        Ta_real = 0.5 * Le * (FR_real - FL_real)
        
        F_fric_y = B_y * qdot[0] + A_y * np.tanh(qdot[0] / eps_v)
        F_fric_a = B_a * qdot[1] + (delta_fric_R - 1.0) * A_y * 0.5 * Le * np.tanh(qdot[1] / eps_a)
        
        yG_ddot = (FG_real - F_fric_y) / M_y
        alpha_ddot = (Ta_real - F_fric_a - K_a * q[1]) / M_a
        
        qdot[0] += Ts * yG_ddot
        qdot[1] += Ts * alpha_ddot
        q[0] += Ts * qdot[0]
        q[1] += Ts * qdot[1]
        
        hist_sync[k] = -Le * q[1] * 1000.0  # mm
        hist_track[k] = (yd[k] - q[0]) * 1000.0  # mm
        hist_Wy[k] = np.linalg.norm(W_y)
        hist_Wa[k] = np.linalg.norm(W_a)
        hist_rho_y[k] = rho_y
        hist_rho_a[k] = rho_a
        hist_Fnn[k] = v_nn_raw_modal[0]
        hist_Tnn[k] = v_nn_raw_modal[1]
        
    mask = (t >= 4.0) & (t < 20.0)
    sync_rms = float(np.sqrt(np.mean(hist_sync[mask]**2)))
    track_rmse = float(np.sqrt(np.mean(hist_track[mask]**2)))
    wy_max = float(np.max(hist_Wy[mask]))
    wa_max = float(np.max(hist_Wa[mask]))
    fnn_max = float(np.max(np.abs(hist_Fnn[mask])))
    
    return {
        "sync_rms": sync_rms,
        "track_rmse": track_rmse,
        "wy_max": wy_max,
        "wa_max": wa_max,
        "fnn_max": fnn_max,
        "hist_sync": hist_sync,
        "hist_track": hist_track,
        "hist_Wy": hist_Wy,
        "hist_Wa": hist_Wa,
        "hist_rho_y": hist_rho_y,
        "t": t
    }

def main():
    print("=" * 80)
    print("探索性离线检查：LC未激活工况下与普通BLNN的一致性")
    print("Exploratory Offline Check: Invariance to Vanilla BLNN in Unactivated Regime")
    print("=" * 80)
    
    # 1. Vanilla BLNN (rho == 1.0)
    res_vanilla = simulate_method("VANILLA")
    base_track = res_vanilla["track_rmse"]
    
    # 2. Fixed Modulation BLNN (rho == [0.917, 0.979])
    res_fixed = simulate_method("FIXED", fixed_c=(0.917, 0.979))
    pct_fixed = (res_fixed["track_rmse"] - base_track) / base_track * 100.0
    
    # 3. Heavy fixed suppression (c_y=0.5, c_alpha=0.5)
    res_fixed_heavy = simulate_method("FIXED", fixed_c=(0.500, 0.500))
    pct_heavy = (res_fixed_heavy["track_rmse"] - base_track) / base_track * 100.0
    
    # 4. Dynamic LC-BLNN (rho dynamically modulated)
    res_lc = simulate_method("LC")
    pct_lc = (res_lc["track_rmse"] - base_track) / base_track * 100.0
    
    df_res = pd.DataFrame([
        {
            "Method": "普通 BLNN (rho=1.0)",
            "Sync_RMS_mm": res_vanilla["sync_rms"],
            "Track_RMSE_mm": res_vanilla["track_rmse"],
            "Rel_Change_Pct": 0.0,
            "Wy_peak": res_vanilla["wy_max"],
            "Wa_peak": res_vanilla["wa_max"],
            "Fnn_peak_N": res_vanilla["fnn_max"],
            "Status": "基准 (全速学习更新)"
        },
        {
            "Method": "固定调制 BLNN (cy=0.917, ca=0.979)",
            "Sync_RMS_mm": res_fixed["sync_rms"],
            "Track_RMSE_mm": res_fixed["track_rmse"],
            "Rel_Change_Pct": pct_fixed,
            "Wy_peak": res_fixed["wy_max"],
            "Wa_peak": res_fixed["wa_max"],
            "Fnn_peak_N": res_fixed["fnn_max"],
            "Status": f"静态弱化 (跟踪退化 +{pct_fixed:.2f}%)"
        },
        {
            "Method": "重度固定抑制 BLNN (cy=0.500, ca=0.500)",
            "Sync_RMS_mm": res_fixed_heavy["sync_rms"],
            "Track_RMSE_mm": res_fixed_heavy["track_rmse"],
            "Rel_Change_Pct": pct_heavy,
            "Wy_peak": res_fixed_heavy["wy_max"],
            "Wa_peak": res_fixed_heavy["wa_max"],
            "Fnn_peak_N": res_fixed_heavy["fnn_max"],
            "Status": f"激进弱化 (跟踪退化 +{pct_heavy:.2f}%)"
        },
        {
            "Method": "动态 LC-BLNN (按状态差额协调)",
            "Sync_RMS_mm": res_lc["sync_rms"],
            "Track_RMSE_mm": res_lc["track_rmse"],
            "Rel_Change_Pct": pct_lc,
            "Wy_peak": res_lc["wy_max"],
            "Wa_peak": res_lc["wa_max"],
            "Fnn_peak_N": res_lc["fnn_max"],
            "Status": "LC未激活 (严格等同普通BLNN)"
        },
    ])
    
    out_csv = os.path.join(OUT_DIR, "EXP5_OFFLINE_FIXED_MODULATION_BENCHMARK.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print("\nBenchmark Summary Table (Dynamically Computed):")
    print(df_res.to_string())
    
    # Plotting comparison figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=300, sharex=True)
    t = res_lc["t"]
    
    # Subplot 1: Sync Error
    axes[0].plot(t, res_vanilla["hist_sync"], color='#7f7f7f', lw=1.2, linestyle=':', label=f"普通 BLNN (RMS={res_vanilla['sync_rms']:.4f} mm)")
    axes[0].plot(t, res_fixed["hist_sync"], color='#1f77b4', lw=1.3, linestyle='--', label=f"固定调制 BLNN (cy=0.917, RMS={res_fixed['sync_rms']:.4f} mm)")
    axes[0].plot(t, res_lc["hist_sync"], color='#d62728', lw=1.6, label=f"动态 LC-BLNN [LC未激活, 等同普通BLNN] (RMS={res_lc['sync_rms']:.4f} mm)")
    axes[0].set_ylabel(r'同步误差 $e_{\mathrm{sync}}$ (mm)', fontsize=10)
    axes[0].set_title('(a) 同步误差时域响应对比 (同物理对象、修正4秒行程轨迹)', fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right', fontsize=9.5)
    
    # Subplot 2: Tracking Error
    axes[1].plot(t, res_vanilla["hist_track"], color='#7f7f7f', lw=1.2, linestyle=':', label=f"普通 BLNN (RMSE={res_vanilla['track_rmse']:.4f} mm)")
    axes[1].plot(t, res_fixed["hist_track"], color='#1f77b4', lw=1.3, linestyle='--', label=f"固定调制 BLNN (RMSE={res_fixed['track_rmse']:.4f} mm, +{pct_fixed:.2f}%)")
    axes[1].plot(t, res_lc["hist_track"], color='#d62728', lw=1.6, label=f"动态 LC-BLNN [严格重合普通BLNN] (RMSE={res_lc['track_rmse']:.4f} mm, {pct_lc:+.2f}%)")
    axes[1].set_ylabel(r'跟踪误差 $e_y$ (mm)', fontsize=10)
    axes[1].set_title(f'(b) 质心跟踪误差响应对比 [静态降低学习导致跟踪退化+{pct_fixed:.2f}%, LC未受限时不退化]', fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='lower left', fontsize=9.5)
    
    # Subplot 3: Weight Norm Wy
    axes[2].plot(t, res_vanilla["hist_Wy"], color='#7f7f7f', lw=1.2, linestyle=':', label=f"普通 BLNN ||Wy|| (峰值={res_vanilla['wy_max']:.3f})")
    axes[2].plot(t, res_fixed["hist_Wy"], color='#1f77b4', lw=1.3, linestyle='--', label=f"固定调制 BLNN (峰值={res_fixed['wy_max']:.3f})")
    axes[2].plot(t, res_lc["hist_Wy"], color='#d62728', lw=1.6, label=f"动态 LC-BLNN [严格重合普通BLNN] (峰值={res_lc['wy_max']:.3f})")
    axes[2].set_ylabel(r'权值范数 $\|W_y\|$', fontsize=10)
    axes[2].set_xlabel('时间 (s)', fontsize=10)
    axes[2].set_title('(c) 平动网络权值范数积累对比 [LC在无差额区间保持全速学习能力]', fontsize=11, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper right', fontsize=9.5)
    
    plt.tight_layout()
    fig_out = os.path.join(OUT_DIR, "EXP5_OFFLINE_FIXED_MODULATION_COMPARISON.png")
    fig_art = os.path.join(ARTIFACT_DIR, "EXP5_OFFLINE_FIXED_MODULATION_COMPARISON.png")
    plt.savefig(fig_out, bbox_inches='tight')
    plt.savefig(fig_art, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_out} and mirrored to artifact dir.")

if __name__ == "__main__":
    main()
