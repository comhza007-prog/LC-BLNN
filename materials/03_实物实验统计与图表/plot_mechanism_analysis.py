"""
Script: plot_mechanism_analysis.py
Purpose: Generates publication-quality mechanism figures dynamically from derived data tables:
  1. EXP5_MECHANISM_DISCREPANCY_TO_RHO_VERIFICATION.png
     - Panel (a): Theoretical design law chi -> rho curve vs empirical points (directly reading reconstructed chi)
       with INSET for prediction error residual distribution
     - Panel (b): Formal firmware allocator modification checks across all 277,662 frames
     - Panel (c): Block 3 motor current command timeline & saturation intervals
     - Panel (d): Block 3 compensation discrepancy Delta v (from reconstructed chi) & learning coordination factor rho
  2. EXP5_MECHANISM_M2_VS_M3_WEIGHT_AND_ERROR_CORRESPONDENCE.png
     - Panel (a): Block 3 synchronization error response comparison (e_sync = y_L - y_R = -Le * alpha)
     - Panel (b): Block 3 tracking error response comparison (e_y = y_d - y_G)
     - Panel (c): Block 3 weight Frobenius norm ||W_y|| dynamics
     - Panel (d): Block 3 raw neural translation force F_nn request
     - Panel (e): Full 35-block 4-quadrant classification scatter (13 / 13 / 7 / 2) with B17 & B26 labeled
     - Panel (f): Deconstruction of 9 degraded blocks (7 tracking improved trade-offs + 2 both degraded)
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r"d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\05_数据采集与归档"
OUT_DIR = r"d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\06_分析与论文图表"
ARTIFACT_DIR = r"C:\Users\ele\.gemini\antigravity\brain\973650e7-c375-4adb-b274-cbc5c0adc155"

def plot_fig1_discrepancy_verification():
    print("Generating Figure 1: Discrepancy -> rho verification reading reconstructed chi...")
    
    # 1. Load Summary Table for formal constraint checks
    sum_csv = os.path.join(OUT_DIR, "EXP5_LC_RECONSTRUCTION_SUMMARY.csv")
    df_sum = pd.read_csv(sum_csv)
    row_overall = df_sum[df_sum["block"] == "OVERALL_35BLOCKS"].iloc[0]
    
    tot_mod_fw_pct = float(row_overall["tot_mod_fw_pct"])
    base_mod_fw_pct = float(row_overall["base_mod_fw_pct"])
    active_lc_pct = float(row_overall["active_lc_pct"])
    both_mod_cnt = int(row_overall["both_mod_fw_cnt"])
    active_tot_cnt = both_mod_cnt + int(row_overall["tot_only_fw_cnt"]) + int(row_overall["base_only_fw_cnt"]) + int(row_overall["neither_fw_cnt"])
    both_pct_in_active = (both_mod_cnt / active_tot_cnt) * 100.0 if active_tot_cnt > 0 else 0.0
    
    # 2. Load Reconstructed Points Table: directly read chi_y and chi_a for Block 3
    pts_csv = os.path.join(OUT_DIR, "EXP5_LC_RECONSTRUCTION_POINTS.csv")
    df_pts = pd.read_csv(pts_csv)
    pts_b3 = df_pts[df_pts["block"] == 3].sort_values("timestamp_ms").reset_index(drop=True)
    
    t_s = (pts_b3["timestamp_ms"].values - 4000) / 1000.0
    chi_y = pts_b3["chi_y"].values
    chi_a = pts_b3["chi_a"].values
    rho_y = pts_b3["rho_rec_y"].values
    rho_a = pts_b3["rho_rec_a"].values
    delta_v_y = chi_y * 800.0
    delta_v_a = chi_a * 800.0
    
    # Residual error arrays across all 277,662 points
    err_y_all = df_pts["err_rho_y"].values
    max_err_y = float(np.max(err_y_all))
    
    # Load raw Block 3 time series for motor command signals
    b3_csv = os.path.join(DATA_DIR, "EXP5_5B_B03_M3_LCBLNN_20261017_152547.csv")
    df3 = pd.read_csv(b3_csv)
    win3 = df3[(df3["timestamp_ms"] >= 4000) & (df3["timestamp_ms"] < 20000)].sort_values("timestamp_ms").reset_index(drop=True)
    
    # Create Figure
    fig = plt.figure(figsize=(15, 11), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.15, 1, 1.05], hspace=0.48, wspace=0.25)
    
    # -------------------------------------------------------------
    # Subplot 1: Theoretical curve vs Reconstructed chi Points + Residual Inset
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    chi_sweep = np.linspace(0, 0.45, 500)
    rho_sweep = 1.0 / (1.0 + 100.0 * chi_sweep**2)
    ax1.plot(chi_sweep, rho_sweep, 'k-', lw=2.5, label=r'理论设计律 $\rho = \frac{1}{1 + 100 \chi^2}$')
    
    ax1.scatter(chi_y[::5], rho_y[::5], color='#d62728', alpha=0.5, s=15, label=r'实测平动 $\chi_y$ 散点 (Block 3 独立重构值)')
    ax1.scatter(chi_a[::5], rho_a[::5], color='#1f77b4', alpha=0.5, s=15, label=r'实测旋转 $\chi_\alpha$ 散点 (Block 3 独立重构值)')
    
    ax1.set_title(r'(a) 补偿差额指标 $\chi$ 与门控因子 $\rho$ 的非线性抑制映射及重构残差', fontsize=11, fontweight='bold')
    ax1.set_xlabel(r'独立重构归一化补偿差额 $\chi = \frac{|\Delta v_{\mathrm{modal}}|}{I_{\max}}$', fontsize=10)
    ax1.set_ylabel(r'学习协调门控因子 $\rho$', fontsize=10)
    ax1.set_xlim(-0.01, 0.45)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9.0)
    
    # Inset Axis: Prediction Error Residual Distribution
    ax_ins = ax1.inset_axes([0.18, 0.16, 0.40, 0.36])
    ax_ins.hist(err_y_all[::10], bins=30, color='#d62728', alpha=0.6, label=r'$e_{\rho_y}$')
    ax_ins.axvline(0.0005, color='black', linestyle=':', lw=1.2, label=r'$\pm 0.0005$ 半字边界')
    ax_ins.set_title(f'重构误差分布 (N=277,662)\n极值 {max_err_y:.5f} <= 0.00051', fontsize=8, fontweight='bold')
    ax_ins.set_xlabel(r'绝对误差 $|\hat{\rho} - \rho_{\mathrm{rec}}|$', fontsize=7.5)
    ax_ins.set_ylabel('频数', fontsize=7.5)
    ax_ins.tick_params(axis='both', which='major', labelsize=7)
    ax_ins.set_xlim(0, 0.0006)
    ax_ins.grid(True, linestyle=':', alpha=0.5)
    
    # -------------------------------------------------------------
    # Subplot 2: Overlapping Constraint Trigger Checks
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    checks = ['总请求受限\n(板端分配修改标志)', '基准请求受限\n(模态差额检查)', '已记录点摆率\n未检出明显差额', 'LC明显激活\nrho < 0.999']
    rates = [tot_mod_fw_pct, base_mod_fw_pct, 0.0, active_lc_pct]
    colors = ['#ff7f0e', '#2ca02c', '#7f7f7f', '#d62728']
    bars = ax2.bar(checks, rates, color=colors, width=0.5, edgecolor='black', lw=1.2)
    
    ax2.text(bars[0].get_x() + bars[0].get_width()/2.0, rates[0] + 1.8, f'{rates[0]:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.text(bars[1].get_x() + bars[1].get_width()/2.0, rates[1] + 1.8, f'{rates[1]:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.text(bars[2].get_x() + bars[2].get_width()/2.0, 1.8, '未检出*', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#555555')
    ax2.text(bars[3].get_x() + bars[3].get_width()/2.0, rates[3] + 1.8, f'{rates[3]:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax2.set_title(f'(b) 35 块全部 277,662 帧约束状态检查 (激活期 {both_pct_in_active:.1f}% 共同受限)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('在全评估窗口中占比 (%)', fontsize=10)
    ax2.set_ylim(0, 52)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # -------------------------------------------------------------
    # Subplot 3: Motor Current Command & Saturation in Block 3
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t_s, win3["calc_cmd_L"], color='#1f77b4', lw=1.4, label='左轮计算指令 iL')
    ax3.plot(t_s, win3["calc_cmd_R"], color='#d62728', lw=1.4, label='右轮计算指令 iR')
    ax3.axhline(800, color='black', linestyle=':', lw=1.2, label='CAN 当前指令限幅门限 +/-800 counts')
    ax3.axhline(-800, color='black', linestyle=':', lw=1.2)
    
    sat_mask = (win3["applied_cmd_L"].abs() >= 799) | (win3["applied_cmd_R"].abs() >= 799)
    ax3.fill_between(t_s, -880, 880, where=sat_mask, color='gray', alpha=0.25, label='指令接近限幅边界区间')
    ax3.set_title('(c) 驱动电流指令时域响应与指令接近限幅边界区间 (Block 3 M3)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('评估窗口时间 (s) [原时间 4.0 ~ 20.0 s]', fontsize=10)
    ax3.set_ylabel('电机指令 (counts)', fontsize=10)
    ax3.set_ylim(-920, 920)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4, fontsize=9.0, framealpha=0.9)
    
    # -------------------------------------------------------------
    # Subplot 4: Discrepancy & Rho Dynamics in Block 3
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, :])
    ax4_twin = ax4.twinx()
    
    l1 = ax4.plot(t_s, delta_v_y, color='#9467bd', lw=1.5, label=r'平动补偿差额 $|\Delta v_y|$ (counts, 独立重构)')
    l2 = ax4.plot(t_s, delta_v_a, color='#8c564b', lw=1.5, label=r'旋转补偿差额 $|\Delta v_\alpha|$ (counts, 独立重构)')
    
    l3 = ax4_twin.plot(t_s, rho_y, color='#d62728', lw=1.8, linestyle='-', label=r'平动协调门控因子 $\rho_y$')
    l4 = ax4_twin.plot(t_s, rho_a, color='#1f77b4', lw=1.8, linestyle='--', label=r'旋转协调门控因子 $\rho_\alpha$')
    
    ax4.set_title(r'(d) 独立重构补偿差额 $\Delta v$ 瞬态冲击与学习协调门控因子 $\rho$ 的实时抑制响应', fontsize=11, fontweight='bold', pad=34)
    ax4.set_xlabel('评估窗口时间 (s) [原时间 4.0 ~ 20.0 s]', fontsize=10)
    ax4.set_ylabel('差额幅值 (counts)', fontsize=10)
    ax4_twin.set_ylabel(r'协调因子 $\rho$', fontsize=10)
    ax4.set_ylim(-10, 310)
    ax4_twin.set_ylim(0.0, 1.05)
    
    lines = [l1[0], l2[0], l3[0], l4[0]]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=9.0, frameon=True)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    out_png = os.path.join(OUT_DIR, "EXP5_MECHANISM_DISCREPANCY_TO_RHO_VERIFICATION.png")
    art_png = os.path.join(ARTIFACT_DIR, "EXP5_MECHANISM_DISCREPANCY_TO_RHO_VERIFICATION.png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(art_png, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png} and mirrored to artifact dir.")

def plot_fig2_weight_and_error_correspondence():
    print("Generating Figure 2: Weight and error correspondence dynamically...")
    
    # 1. Load Block 3 M2 and M3
    b3_m2_csv = os.path.join(DATA_DIR, "EXP5_5B_B03_M2_BLNN_20261017_152421.csv")
    b3_m3_csv = os.path.join(DATA_DIR, "EXP5_5B_B03_M3_LCBLNN_20261017_152547.csv")
    df2 = pd.read_csv(b3_m2_csv)
    df3 = pd.read_csv(b3_m3_csv)
    
    win2 = df2[(df2["timestamp_ms"] >= 4000) & (df2["timestamp_ms"] < 20000)].copy()
    win3 = df3[(df3["timestamp_ms"] >= 4000) & (df3["timestamp_ms"] < 20000)].copy()
    
    t_s2 = (win2["timestamp_ms"] - 4000) / 1000.0
    t_s3 = (win3["timestamp_ms"] - 4000) / 1000.0
    
    # Explicitly invert e_sync_m (CSV field is y_R - y_L, plotting convention is e_sync = y_L - y_R = -Le * alpha)
    e_sync_plot2 = -win2["e_sync_m"].values * 1000.0
    e_sync_plot3 = -win3["e_sync_m"].values * 1000.0
    
    sync_b3_m2 = float(np.sqrt(np.mean(e_sync_plot2**2)))
    sync_b3_m3 = float(np.sqrt(np.mean(e_sync_plot3**2)))
    pct_sync_b3 = (sync_b3_m3 - sync_b3_m2) / sync_b3_m2 * 100.0
    
    track_b3_m2 = float(np.sqrt(np.mean((win2["ref_pos_m"] - win2["yG_m"])**2)) * 1000.0)
    track_b3_m3 = float(np.sqrt(np.mean((win3["ref_pos_m"] - win3["yG_m"])**2)) * 1000.0)
    pct_track_b3 = (track_b3_m3 - track_b3_m2) / track_b3_m2 * 100.0
    
    wy_b3_m2 = float(np.max(win2["norm_W_y"]))
    wy_b3_m3 = float(np.max(win3["norm_W_y"]))
    pct_wy_b3 = (wy_b3_m3 - wy_b3_m2) / wy_b3_m2 * 100.0
    
    fnn_b3_m2 = float(np.max(np.abs(win2["F_nn"])))
    fnn_b3_m3 = float(np.max(np.abs(win3["F_nn"])))
    pct_fnn_b3 = (fnn_b3_m3 - fnn_b3_m2) / fnn_b3_m2 * 100.0
    
    fig = plt.figure(figsize=(15, 11), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.35, wspace=0.25)
    
    # Panel 1: Sync Error Comparison in Block 3 (Explicitly e_sync = y_L - y_R)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_s2, e_sync_plot2, color='#1f77b4', lw=1.2, alpha=0.8, label=f'普通 BLNN (RMS={sync_b3_m2:.4f} mm)')
    ax1.plot(t_s3, e_sync_plot3, color='#d62728', lw=1.5, label=f'LC-BLNN (RMS={sync_b3_m3:.4f} mm, {pct_sync_b3:.1f}%)')
    ax1.axhline(0, color='black', linestyle=':', lw=1.0)
    ax1.set_title(r'(a) 示例 Block 3 同步误差响应对比 ($e_{\mathrm{sync}} = y_L - y_R = -L_e \alpha$)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('评估窗口时间 (s)', fontsize=10)
    ax1.set_ylabel(r'同步误差 $e_{\mathrm{sync}}$ (mm)', fontsize=10)
    ax1.set_ylim(-0.35, 0.35)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9.0)
    
    # Panel 2: Tracking Error Comparison in Block 3
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_s2, (win2["ref_pos_m"] - win2["yG_m"]) * 1000.0, color='#1f77b4', lw=1.2, alpha=0.8, label=f'普通 BLNN (RMSE={track_b3_m2:.4f} mm)')
    ax2.plot(t_s3, (win3["ref_pos_m"] - win3["yG_m"]) * 1000.0, color='#d62728', lw=1.5, label=f'LC-BLNN (RMSE={track_b3_m3:.4f} mm, {pct_track_b3:.1f}%)')
    ax2.axhline(0, color='black', linestyle=':', lw=1.0)
    ax2.set_title(r'(b) 示例 Block 3 质心跟踪误差响应对比 ($e_y = y_d - y_G$)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('评估窗口时间 (s)', fontsize=10)
    ax2.set_ylabel(r'跟踪误差 $e_y$ (mm)', fontsize=10)
    ax2.set_ylim(-2.8, 1.8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=9.0)
    
    # Panel 3: Weight Norm Wy Dynamics in Block 3
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_s2, win2["norm_W_y"], color='#1f77b4', lw=1.8, label=f'普通 BLNN $\|W_y\|$ (峰值={wy_b3_m2:.3f})')
    ax3.plot(t_s3, win3["norm_W_y"], color='#d62728', lw=2.0, label=f'LC-BLNN $\|W_y\|$ (峰值={wy_b3_m3:.3f}, {pct_wy_b3:.1f}%)')
    ax3.set_title(r'(c) 示例 Block 3 平动网络权值范数增长与抑制对比', fontsize=11, fontweight='bold')
    ax3.set_xlabel('评估窗口时间 (s)', fontsize=10)
    ax3.set_ylabel(r'权值 Frobenius 范数 $\|W_y\|$', fontsize=10)
    ax3.set_ylim(0, 1.4)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=9.0)
    
    # Panel 4: Raw Neural Force F_nn Dynamics in Block 3
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_s2, win2["F_nn"], color='#1f77b4', lw=1.5, alpha=0.8, label=f'普通 BLNN $F_{{nn}}$ (峰值 {fnn_b3_m2:.2f} N)')
    ax4.plot(t_s3, win3["F_nn"], color='#d62728', lw=1.8, label=f'LC-BLNN $F_{{nn}}$ (峰值 {fnn_b3_m3:.2f} N, {pct_fnn_b3:.1f}%)')
    ax4.set_title(r'(d) 示例 Block 3 神经网络平动补偿力请求对比', fontsize=11, fontweight='bold')
    ax4.set_xlabel('评估窗口时间 (s)', fontsize=10)
    ax4.set_ylabel(r'平动补偿力 $F_{nn}$ (N)', fontsize=10)
    ax4.set_ylim(-3.0, 3.0)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='lower left', fontsize=9.0)
    
    # 2. Load 35-Block Metrics Table for Panels (e) & (f)
    df_all = pd.read_csv(os.path.join(OUT_DIR, "EXP5_MECHANISM_FULL_35BLOCKS_METRICS.csv"))
    q1 = df_all[(df_all['diff_sync'] < 0) & (df_all['diff_track'] < 0)] # Both improved
    q2 = df_all[(df_all['diff_sync'] < 0) & (df_all['diff_track'] >= 0)] # Sync imp, Track deg
    q3 = df_all[(df_all['diff_sync'] >= 0) & (df_all['diff_track'] < 0)] # Sync deg, Track imp
    q4 = df_all[(df_all['diff_sync'] >= 0) & (df_all['diff_track'] >= 0)] # Both degraded
    
    # Panel 5: 4-Quadrant Classification Scatter of All 35 Blocks
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(q1["diff_track"] * 1000.0, q1["diff_sync"] * 1000.0, color='#2ca02c', s=55, marker='o', alpha=0.9, label=f'两指标共同改善 ({len(q1)} 块, {len(q1)/len(df_all)*100.0:.1f}%)')
    ax5.scatter(q2["diff_track"] * 1000.0, q2["diff_sync"] * 1000.0, color='#1f77b4', s=55, marker='^', alpha=0.9, label=f'同步收益·跟踪代价 ({len(q2)} 块, {len(q2)/len(df_all)*100.0:.1f}%)')
    ax5.scatter(q3["diff_track"] * 1000.0, q3["diff_sync"] * 1000.0, color='#ff7f0e', s=55, marker='s', alpha=0.9, label=f'跟踪收益·同步代价 ({len(q3)} 块, {len(q3)/len(df_all)*100.0:.1f}%)')
    ax5.scatter(q4["diff_track"] * 1000.0, q4["diff_sync"] * 1000.0, color='#d62728', s=70, marker='D', alpha=0.95, label=f'两指标均变差 ({len(q4)} 块, {len(q4)/len(df_all)*100.0:.1f}%)')
    
    # Annotate B17 and B26 explicitly
    for idx, r in q4.iterrows():
        ax5.annotate(f"B{int(r['block'])}", (r['diff_track']*1000.0 + 15, r['diff_sync']*1000.0 + 1.0), fontsize=9, fontweight='bold', color='#d62728')
        
    ax5.axhline(0, color='black', linestyle='--', lw=1.0)
    ax5.axvline(0, color='black', linestyle='--', lw=1.0)
    ax5.set_title(r'(e) 全 35 块四象限响应分布 (同步 26/35 改善, 跟踪 20/35 改善)', fontsize=11, fontweight='bold')
    ax5.set_xlabel(r'跟踪 RMSE 变化量 $\Delta \mathrm{Track}$ (um) [负为改善]', fontsize=10)
    ax5.set_ylabel(r'同步 RMS 变化量 $\Delta \mathrm{Sync}$ (um) [负为改善]', fontsize=10)
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.legend(loc='lower left', fontsize=8.8)
    
    # Panel 6: Degraded 9 Blocks Audit (7 Trade-offs + 2 Both Degraded)
    ax6 = fig.add_subplot(gs[2, 1])
    deg = pd.concat([q3, q4]).sort_values("block")
    blocks_deg = deg["block"].values
    x_idx = np.arange(len(blocks_deg))
    width = 0.35
    
    ax6.bar(x_idx - width/2, deg["diff_sync"] * 1000.0, width, color='#d62728', alpha=0.85, label=r'同步 RMS 退化量 (+um)')
    ax6.bar(x_idx + width/2, deg["diff_track"] * 1000.0, width, color='#1f77b4', alpha=0.85, label=r'跟踪 RMSE 变化量 (负为改善, um)')
    ax6.axhline(0, color='black', linestyle='-', lw=1.0)
    ax6.set_xticks(x_idx)
    ax6.set_xticklabels([f'B{int(b)}' for b in blocks_deg], fontsize=9)
    
    ax6.text(4, 380, 'B17 & B26: 两项指标均退化\n(本次对比不支持性能改进)', fontsize=9, color='#d62728', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', edgecolor='#d62728', lw=1.0))
    ax6.set_title(f'(f) 9 个同步退化轮次解构: {len(q3)} 块伴随跟踪改善, {len(q4)} 块两项均退化', fontsize=11, fontweight='bold')
    ax6.set_xlabel('退化轮次编号 (Block Index)', fontsize=10)
    ax6.set_ylabel('误差变化量 (um)', fontsize=10)
    ax6.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax6.legend(loc='upper right', fontsize=8.8)
    
    out_png = os.path.join(OUT_DIR, "EXP5_MECHANISM_M2_VS_M3_WEIGHT_AND_ERROR_CORRESPONDENCE.png")
    art_png = os.path.join(ARTIFACT_DIR, "EXP5_MECHANISM_M2_VS_M3_WEIGHT_AND_ERROR_CORRESPONDENCE.png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(art_png, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png} and mirrored to artifact dir.")

if __name__ == "__main__":
    plot_fig1_discrepancy_verification()
    plot_fig2_weight_and_error_correspondence()
