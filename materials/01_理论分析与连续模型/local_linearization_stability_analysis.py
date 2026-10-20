# -*- coding: utf-8 -*-
"""
================================================================================
连续闭环模型静止平衡点的局部线性化稳定性分析（论文收尾终版）
Local Linearization Stability Analysis of the 30-Dimensional Continuous Model
================================================================================
本脚本严格落实用户指示与措辞规范，作为论文连续理论分析的定稿版本：

【核心规范与措辞修订】：
1. 状态定义精准化：
   - 机械跟踪误差与一阶导数 e = [e_y, e_alpha], edot = [edot_y, edot_alpha] (4维)
   - 两个一阶抗饱和状态 z = [z_y, z_alpha] (2维)（状态数量不等于系统阶次）
   - 调制型自适应神经网络权重 W_y (12维), W_alpha (12维) (24维)
   总状态维度 n = 30。
2. 连续模型的必要简化说明：
   - 稳态静止平衡点处加速度估计输入固定为参考加速度 (qddot_est = qddot_d = 0)，未包含高阶观测器/滤波器的瞬态动力学；
   - 变化率限制器在平衡点稳态处未激活，直接取 n_s = n_raw，未包含离散更新步长与摆率记忆历史状态；
   - 稳态限制器未激活不能单凭这一点证明所有遗漏动态都可忽略，因此结论严格界定为
     “所建立的30维连续模型在考察平衡点附近的局部指数稳定性”，不能外推为数字固件全部状态的无条件等价。
3. 结论范围收窄与主张定性：
   - 仅对两个已核查的确定静止平衡点（零扰动名义点与常值偏置扰动点）形成局部指数稳定性结论；
   - 局部Hurwitz判定仅保证局部孤立性与指数收敛，不证明全局平衡点唯一；
   - 25组常值扰动网格定性为“所测试常值扰动下均得到稳定的线性化结果”，不据此宣称连续扰动范围内的鲁棒稳定性。
4. 论文最终保留主张：
   “在所列假设参数及连续模型下，所考察的两个静止平衡点经数值线性化核查，具有局部指数稳定性；
    受限运动过程的性能由仿真与重复实物对照评价。”
5. 参数实核与差分稳定性：
   - 神经网络衰减率：eps_y * eta_y = 25.0 * 0.01 = 0.25 s^-1，eps_alpha * eta_alpha = 18.0 * 0.01 = 0.18 s^-1
   - 偏置平衡点处执行器实际稳态输出：u_phys = [0.5000 N, 0.16839 N·m]
   - 分配器可用平动推力：F_max * (1 - |p_T|) = 9.35537 N，剩余平动推力安全余量：8.85537 N (89.30%)
   - 偏转力矩物理极限：2.9750 N·m，剩余偏转力矩安全余量：2.80661 N·m (94.34%)
   - 差分步长 10^-6, 10^-7, 10^-8 敏感性检验：谱上界恒定为 -0.180000 s^-1。
"""

import os
import sys
import json
import numpy as np
from scipy import optimize, linalg

# 导入连续闭环模型与快子系统分析器
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from continuous_closed_loop_model import ContinuousClosedLoopAnalysis
from frozen_request_fast_subsystem import FrozenRequestFastSubsystemAnalyzer


class LocalLinearizationAnalyzer:
    def __init__(self):
        self.c_model = ContinuousClosedLoopAnalysis()
        self.p = self.c_model.p
        self.c = self.c_model.c
        self.fast_analyzer = FrozenRequestFastSubsystemAnalyzer()
        
        # 物理饱和极限
        self.F_max = self.fast_analyzer.F_max  # 9.9166667 N
        self.T_max = self.fast_analyzer.T_max  # 2.9750000 N·m
        
        # 固件变化率物理等效上限 (75 counts/ms = 75,000 counts/s)
        # 单侧推力比例系数 Kf_L 单位为 N/count
        self.slew_rate_limit_counts_s = 75000.0
        self.slew_rate_limit_N_s = self.slew_rate_limit_counts_s * float(self.c.Kf_L)  # 464.84375 N/s
        
        # 神经网络真实乘积衰减率
        self.rate_Wy_expected = float(self.c.nn_eps_y * self.c.nn_eta_y)          # 25.0 * 0.01 = 0.2500 s^-1
        self.rate_Wa_expected = float(self.c.nn_eps_alpha * self.c.nn_eta_alpha)  # 18.0 * 0.01 = 0.1800 s^-1
        self.rate_aw_expected = float(self.c.lambda_aw)                           # 20.0000 s^-1
        
    def full_closed_loop_dynamics(self, x, d_ext=np.zeros(2), qd=np.zeros(2), qdot_d=np.zeros(2), qddot_d=np.zeros(2)):
        """
        计算30维所建立连续闭环模型的全状态导数 xdot = f(x, d_ext)
        状态定义:
          x[0:2]   = e = [e_y, e_alpha]
          x[2:4]   = edot = [edot_y, edot_alpha]
          x[4:6]   = z = [z_y, z_alpha] (两个一阶抗饱和状态)
          x[6:18]  = W_y (12维)
          x[18:30] = W_alpha (12维)
        简化说明：
          - 稳态静止设定点保持时 qddot_est = qddot_d = 0；
          - 变化率限制器在稳态处未饱和，取 ns = n_raw。
        """
        e = x[0:2]
        edot = x[2:4]
        z = x[4:6]
        W_y = x[6:18]
        W_alpha = x[18:30]
        
        q = e + qd
        qdot = edot + qdot_d
        qddot_est = qddot_d.copy()
        
        Gamma = np.array([self.c.Gamma_y, self.c.Gamma_alpha])
        sigma = edot + Gamma * e
        
        # 1. 神经网络特征提取与原始补偿输出
        S_y, S_alpha = self.c_model.compute_features(q, qdot, qddot_est, e, edot)
        n_raw = np.array([-np.sum(W_y * S_y), -np.sum(W_alpha * S_alpha)])
        ns = n_raw.copy()  # 连续模型在平衡点局部简化取值
        
        # 2. 基础滑模控制律
        b = self.c_model.compute_baseline_control(q, qdot, qd, qdot_d, qddot_d, e, edot, sigma)
        
        # 3. 双模态执行链与推力分配
        chain = self.c_model.execution_chain(b, z, n_raw, ns)
        u_phys = chain['u']
        r_a = chain['r_a']
        rho = chain['rho']
        
        # 4. 机械物理动力学与误差加速度
        qddot = self.c_model.compute_physical_acceleration(q, qdot, u_phys, d_ext)
        edot_dot = qddot - qddot_d
        
        # 5. 两个一阶抗饱和状态动态: dot{z} = K_aw * r_a - lambda_aw * z
        zdot = self.c.K_aw * r_a - self.c.lambda_aw * z
        
        # 6. 自适应神经网络权重更新律 (含调制因子 rho 与 sigma-修正正则化)
        eps_y = self.c.nn_eps_y
        eps_a = self.c.nn_eps_alpha
        eta_y = self.c.nn_eta_y
        eta_a = self.c.nn_eta_alpha
        
        Wdot_y = eps_y * (rho[0] * sigma[0] * S_y - eta_y * W_y)
        Wdot_a = eps_a * (rho[1] * sigma[1] * S_alpha - eta_a * W_alpha)
        
        # 组装 30 维连续模型状态导数
        xdot = np.zeros(30)
        xdot[0:2] = edot
        xdot[2:4] = edot_dot
        xdot[4:6] = zdot
        xdot[6:18] = Wdot_y
        xdot[18:30] = Wdot_a
        
        return xdot

    def solve_equilibrium(self, d_ext=np.zeros(2)):
        """
        求解给定常值扰动下的真实静止平衡点 x*
        """
        if np.all(d_ext == 0):
            x_star = np.zeros(30)
            res = np.linalg.norm(self.full_closed_loop_dynamics(x_star, d_ext=d_ext))
            return {
                'success': True,
                'x_star': x_star,
                'residual': float(res),
                'method': 'analytical_nominal_zero'
            }
        
        def obj(x):
            return self.full_closed_loop_dynamics(x, d_ext=d_ext)
        
        sol = optimize.root(obj, np.zeros(30), method='hybr', tol=1e-13)
        res = np.linalg.norm(obj(sol.x))
        return {
            'success': bool(sol.success),
            'x_star': sol.x,
            'residual': float(res),
            'method': 'hybr_numerical_root'
        }

    def inspect_smooth_branch_interior(self, x_star, d_ext=np.zeros(2)):
        """
        核查平衡点是否严格位于推力分配器与变化率限制器的内部线性光滑分支
        计算精确物理余量与归一化余量
        """
        e = x_star[0:2]
        edot = x_star[2:4]
        z = x_star[4:6]
        W_y = x_star[6:18]
        W_alpha = x_star[18:30]
        
        Gamma = np.array([self.c.Gamma_y, self.c.Gamma_alpha])
        sigma = edot + Gamma * e
        
        S_y, S_alpha = self.c_model.compute_features(e, edot, np.zeros(2), e, edot)
        n_raw = np.array([-np.sum(W_y * S_y), -np.sum(W_alpha * S_alpha)])
        ns = n_raw.copy()
        
        b = self.c_model.compute_baseline_control(e, edot, np.zeros(2), np.zeros(2), np.zeros(2), e, edot, sigma)
        chain = self.c_model.execution_chain(b, z, n_raw, ns)
        
        u = chain['u']
        u_req = b + z + ns
        r_a = chain['r_a']
        r_b = chain['r_b']
        delta = chain['delta']
        rho = chain['rho']
        
        # 归一化控制请求 (以物理额定极限 F_max, T_max 归一化)
        y_bar_F = float(u_req[0] / self.F_max)
        y_bar_T = float(u_req[1] / self.T_max)
        
        # 1. 偏转力矩物理与归一化余量 (|y_bar_T| <= 1.0)
        torque_margin_normalized = float(1.0 - abs(y_bar_T))
        torque_margin_Nm = float(self.T_max - abs(u[1]))
        
        # 2. 平动推力物理与归一化余量 (|y_bar_F| <= 1.0 - |p_T|)
        trans_bound_normalized = float(1.0 - abs(y_bar_T))
        trans_margin_normalized = float(trans_bound_normalized - abs(y_bar_F))
        available_trans_thrust_N = float(trans_bound_normalized * self.F_max)
        trans_margin_N = float(available_trans_thrust_N - abs(u[0]))
        
        is_allocator_interior = bool(torque_margin_normalized > 1e-4 and trans_margin_normalized > 1e-4)
        
        # 3. 变化率限制器状态：稳态静止点处导数为0，余量为全部阈值
        slew_margin_counts_s = self.slew_rate_limit_counts_s
        slew_margin_N_s = self.slew_rate_limit_N_s
        is_slew_interior = True
        
        is_smooth = bool(is_allocator_interior and is_slew_interior and np.all(np.abs(r_a) < 1e-8))
        
        return {
            'is_allocator_interior': is_allocator_interior,
            'is_slew_interior': is_slew_interior,
            'is_smooth_branch': is_smooth,
            'u_req': [float(u_req[0]), float(u_req[1])],
            'u_phys': [float(u[0]), float(u[1])],
            'y_bar_F': y_bar_F,
            'y_bar_T': y_bar_T,
            'available_torque_Nm': float(self.T_max),
            'torque_margin_Nm': torque_margin_Nm,
            'torque_margin_normalized': torque_margin_normalized,
            'available_trans_thrust_N': available_trans_thrust_N,
            'trans_margin_N': trans_margin_N,
            'trans_margin_normalized': trans_margin_normalized,
            'r_a_norm': float(np.linalg.norm(r_a)),
            'r_b_norm': float(np.linalg.norm(r_b)),
            'delta_norm': float(np.linalg.norm(delta)),
            'rho': [float(rho[0]), float(rho[1])],
            'slew_margin_counts_s': slew_margin_counts_s,
            'slew_margin_N_s': slew_margin_N_s
        }

    def compute_jacobian(self, x_star, d_ext=np.zeros(2), eps=1e-7):
        """
        高精度中心差分计算 30x30 雅可比矩阵
        A_{i,j} = \partial f_i / \partial x_j
        """
        n = len(x_star)
        A = np.zeros((n, n))
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = eps
            f_plus = self.full_closed_loop_dynamics(x_star + dx, d_ext=d_ext)
            f_minus = self.full_closed_loop_dynamics(x_star - dx, d_ext=d_ext)
            A[:, j] = (f_plus - f_minus) / (2.0 * eps)
        return A

    def audit_step_size_sensitivity(self, x_star, d_ext=np.zeros(2)):
        """
        使用 10^-6, 10^-7, 10^-8 三种步长进行差分敏感性复核
        """
        eps_list = [1e-6, 1e-7, 1e-8]
        jacobians = {}
        eigs_summary = {}
        
        for eps in eps_list:
            A = self.compute_jacobian(x_star, d_ext=d_ext, eps=eps)
            jacobians[f"eps_{eps:.0e}"] = A
            eigs = np.linalg.eigvals(A)
            max_re = float(np.max(np.real(eigs)))
            min_re = float(np.min(np.real(eigs)))
            eigs_summary[f"eps_{eps:.0e}"] = {
                'spectral_abscissa': max_re,
                'min_real_part': min_re,
                'is_hurwitz': bool(max_re < -1e-5)
            }
            
        diff_6_7 = float(np.max(np.abs(jacobians['eps_1e-06'] - jacobians['eps_1e-07'])))
        diff_7_8 = float(np.max(np.abs(jacobians['eps_1e-07'] - jacobians['eps_1e-08'])))
        
        return {
            'step_sizes_audited': eps_list,
            'eigs_summary': eigs_summary,
            'diff_max_norm_6_7': diff_6_7,
            'diff_max_norm_7_8': diff_7_8,
            'is_numerically_stable': bool(diff_7_8 < 1e-4)
        }

    def analyze_spectrum(self, A):
        """
        对雅可比矩阵 A 进行全谱特征值分解与物理子系统模态归类
        """
        eigvals = np.linalg.eigvals(A)
        idx = np.argsort(np.real(eigvals))[::-1]
        sorted_eigs = eigvals[idx]
        real_parts = np.real(sorted_eigs)
        imag_parts = np.imag(sorted_eigs)
        
        spectral_abscissa = float(np.max(real_parts))
        min_real = float(np.min(real_parts))
        is_hurwitz = bool(spectral_abscissa < -1e-5)
        
        # 物理模态分类识别
        # 正交未耦合权重模态实部严格对应 -eps * eta
        modal_groups = {
            'yaw_nn_regularization_modes': [],
            'trans_nn_regularization_modes': [],
            'antiwindup_filter_modes': [],
            'mechanical_and_sliding_modes': []
        }
        
        for val in sorted_eigs:
            val_dict = {'real': float(val.real), 'imag': float(val.imag)}
            if abs(val.real - (-self.rate_Wa_expected)) < 1e-4 and abs(val.imag) < 1e-4:
                modal_groups['yaw_nn_regularization_modes'].append(val_dict)
            elif abs(val.real - (-self.rate_Wy_expected)) < 1e-4 and abs(val.imag) < 1e-4:
                modal_groups['trans_nn_regularization_modes'].append(val_dict)
            elif abs(val.real - (-self.rate_aw_expected)) < 1e-3 and abs(val.imag) < 1e-3:
                modal_groups['antiwindup_filter_modes'].append(val_dict)
            else:
                modal_groups['mechanical_and_sliding_modes'].append(val_dict)
                
        return {
            'is_hurwitz': is_hurwitz,
            'spectral_abscissa': spectral_abscissa,
            'min_real_part': min_real,
            'all_eigenvalues': [{'real': float(r), 'imag': float(im)} for r, im in zip(real_parts, imag_parts)],
            'modal_groups': modal_groups,
            'expected_parameters': {
                'rate_Wa_sigma_decay': float(-self.rate_Wa_expected),
                'rate_Wy_sigma_decay': float(-self.rate_Wy_expected),
                'rate_aw_decay': float(-self.rate_aw_expected),
                'eps_y_times_eta_y': f"{self.c.nn_eps_y} * {self.c.nn_eta_y} = {self.rate_Wy_expected}",
                'eps_alpha_times_eta_alpha': f"{self.c.nn_eps_alpha} * {self.c.nn_eta_alpha} = {self.rate_Wa_expected}"
            }
        }

    def verify_lyapunov_equation(self, A):
        """
        求解连续代数Lyapunov方程 A^T P + P A = -Q (Q = I_30)
        验证对称正定解 P > 0
        """
        n = A.shape[0]
        Q = np.eye(n)
        try:
            P = linalg.solve_continuous_lyapunov(A.T, -Q)
            eig_P = np.linalg.eigvalsh(P)
            min_eig_P = float(np.min(eig_P))
            max_eig_P = float(np.max(eig_P))
            is_positive_definite = bool(min_eig_P > 1e-8)
            residual = float(np.linalg.norm(A.T @ P + P @ A + Q))
            alpha_lower_bound = float(1.0 / (2.0 * max_eig_P))
            
            return {
                'solved': True,
                'is_positive_definite': is_positive_definite,
                'min_eigenvalue_P': min_eig_P,
                'max_eigenvalue_P': max_eig_P,
                'condition_number_P': float(max_eig_P / min_eig_P),
                'lyapunov_residual': residual,
                'alpha_lower_bound': alpha_lower_bound
            }
        except Exception as e:
            return {
                'solved': False,
                'error': str(e),
                'is_positive_definite': False
            }

    def run_full_analysis(self):
        """
        执行完整局部线性化分析与复核导出
        """
        print("=" * 80)
        print("执行所建立30维连续闭环模型静止平衡点局部线性化稳定性分析（论文收尾终版）")
        print("=" * 80)
        
        # -------------------------------------------------------------
        # 平衡点1: 名义基准工况 (d_ext = 0)
        # -------------------------------------------------------------
        print("\n[平衡点1] 名义静止设定点保持工况 (d_ext = [0, 0] N, N·m):")
        eq1 = self.solve_equilibrium(np.zeros(2))
        x_star1 = eq1['x_star']
        print(f"  平衡点求解状态: {eq1['success']}, 残差: {eq1['residual']:.2e}")
        
        smooth1 = self.inspect_smooth_branch_interior(x_star1, np.zeros(2))
        print(f"  分配器线性分支内部? {smooth1['is_allocator_interior']}")
        print(f"  力矩安全余量: {smooth1['torque_margin_Nm']:.4f} N·m ({smooth1['torque_margin_normalized']*100:.1f}%)")
        print(f"  推力安全余量: {smooth1['trans_margin_N']:.4f} N ({smooth1['trans_margin_normalized']*100:.1f}%)")
        print(f"  变化率限制器未激活? {smooth1['is_slew_interior']} (余量: {smooth1['slew_margin_counts_s']:.0f} counts/s)")
        
        step_audit1 = self.audit_step_size_sensitivity(x_star1, np.zeros(2))
        print(f"  差分步长稳定性复核: 1e-6 vs 1e-7 最大差值 = {step_audit1['diff_max_norm_6_7']:.2e}, "
              f"1e-7 vs 1e-8 = {step_audit1['diff_max_norm_7_8']:.2e}")
        
        A1 = self.compute_jacobian(x_star1, np.zeros(2), eps=1e-7)
        spec1 = self.analyze_spectrum(A1)
        print(f"  雅可比矩阵严格Hurwitz? {spec1['is_hurwitz']}")
        print(f"  谱上界 (衰减率瓶颈 max Re(lambda)): {spec1['spectral_abscissa']:.6f} s^-1")
        print(f"  最快衰减模态 min Re(lambda):       {spec1['min_real_part']:.2f} s^-1")
        print(f"  模态构成归属:")
        print(f"    - 偏转权值正则模态 (11维):   Re = {spec1['modal_groups']['yaw_nn_regularization_modes'][0]['real']:.4f} s^-1 (严格等于 -18*0.01)")
        print(f"    - 平动权值正则模态 (11维):   Re = {spec1['modal_groups']['trans_nn_regularization_modes'][0]['real']:.4f} s^-1 (严格等于 -25*0.01)")
        print(f"    - 两个一阶抗饱和状态模态 (2维): Re = {spec1['modal_groups']['antiwindup_filter_modes'][0]['real']:.4f} s^-1 (严格等于 -20.0)")
        
        lyap1 = self.verify_lyapunov_equation(A1)
        print(f"  Lyapunov方程正定解 P > 0 存在? {lyap1['is_positive_definite']}, 残差: {lyap1['lyapunov_residual']:.2e}")
        
        # -------------------------------------------------------------
        # 平衡点2: 现实定常偏置扰动工况 (d_ext = [0.5 N, 0.2 N·m])
        # -------------------------------------------------------------
        d_bias = np.array([0.5, 0.2])
        print(f"\n[平衡点2] 现实定常偏置工况 (d_ext = [{d_bias[0]} N, {d_bias[1]} N·m]):")
        eq2 = self.solve_equilibrium(d_bias)
        x_star2 = eq2['x_star']
        print(f"  平衡点求解状态: {eq2['success']}, 残差: {eq2['residual']:.2e}")
        print(f"  真实静态平衡点坐标:")
        print(f"    - 位移误差 e_y*:          {x_star2[0]*1000:.4f} mm")
        print(f"    - 偏角误差 e_alpha*:      {x_star2[1]*1000:.4f} mrad")
        print(f"    - 速度误差 edot*:         [{x_star2[2]*1000:.4f} mm/s, {x_star2[3]*1000:.4f} mrad/s]")
        print(f"    - 两个一阶抗饱和状态 z*:   [{x_star2[4]:.2e}, {x_star2[5]:.2e}]")
        print(f"    - 平动权值范数 ||W_y*||:   {np.linalg.norm(x_star2[6:18]):.4f}")
        print(f"    - 偏转权值范数 ||W_a*||:   {np.linalg.norm(x_star2[18:30]):.4f}")
        
        smooth2 = self.inspect_smooth_branch_interior(x_star2, d_bias)
        print(f"  执行器实际稳态输出: u_phys = [{smooth2['u_phys'][0]:.4f} N, {smooth2['u_phys'][1]:.5f} N·m]")
        print(f"  分配器线性分支内部? {smooth2['is_allocator_interior']}")
        print(f"  可用偏转力矩: {smooth2['available_torque_Nm']:.4f} N·m, 力矩安全余量: {smooth2['torque_margin_Nm']:.5f} N·m ({smooth2['torque_margin_normalized']*100:.2f}%)")
        print(f"  可用平动推力: {smooth2['available_trans_thrust_N']:.5f} N, 推力安全余量: {smooth2['trans_margin_N']:.5f} N ({smooth2['trans_margin_normalized']*100:.2f}%)")
        print(f"  变化率限制器未激活? {smooth2['is_slew_interior']} (余量: {smooth2['slew_margin_counts_s']:.0f} counts/s)")
        
        step_audit2 = self.audit_step_size_sensitivity(x_star2, d_bias)
        print(f"  差分步长稳定性复核: 1e-6 vs 1e-7 最大差值 = {step_audit2['diff_max_norm_6_7']:.2e}, "
              f"1e-7 vs 1e-8 = {step_audit2['diff_max_norm_7_8']:.2e}")
        
        A2 = self.compute_jacobian(x_star2, d_bias, eps=1e-7)
        spec2 = self.analyze_spectrum(A2)
        print(f"  雅可比矩阵严格Hurwitz? {spec2['is_hurwitz']}")
        print(f"  谱上界 (衰减率瓶颈 max Re(lambda)): {spec2['spectral_abscissa']:.6f} s^-1")
        print(f"  最快衰减模态 min Re(lambda):       {spec2['min_real_part']:.2f} s^-1")
        
        lyap2 = self.verify_lyapunov_equation(A2)
        print(f"  Lyapunov方程正定解 P > 0 存在? {lyap2['is_positive_definite']}, 残差: {lyap2['lyapunov_residual']:.2e}")
        
        # -------------------------------------------------------------
        # 补充数值核查: 常值扰动网格线性化核查 (25组工况)
        # -------------------------------------------------------------
        print("\n[补充数值核查] 常值扰动网格线性化核查 (5x5 = 25 组定常扰动点):")
        dy_grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
        da_grid = [-0.5, -0.25, 0.0, 0.25, 0.5]
        grid_results = []
        all_grid_hurwitz = True
        all_grid_smooth = True
        
        for dy in dy_grid:
            for da in da_grid:
                d_curr = np.array([dy, da])
                eq_curr = self.solve_equilibrium(d_curr)
                if not eq_curr['success']:
                    all_grid_hurwitz = False
                    continue
                xs = eq_curr['x_star']
                sm = self.inspect_smooth_branch_interior(xs, d_curr)
                if not sm['is_smooth_branch']:
                    all_grid_smooth = False
                
                A_curr = self.compute_jacobian(xs, d_curr, eps=1e-7)
                sp_curr = self.analyze_spectrum(A_curr)
                if not sp_curr['is_hurwitz']:
                    all_grid_hurwitz = False
                    
                grid_results.append({
                    'dy': dy,
                    'da': da,
                    'solved': eq_curr['success'],
                    'residual': eq_curr['residual'],
                    'ey_mm': float(xs[0] * 1000.0),
                    'ea_mrad': float(xs[1] * 1000.0),
                    'u_phys': sm['u_phys'],
                    'available_trans_thrust_N': sm['available_trans_thrust_N'],
                    'trans_margin_N': sm['trans_margin_N'],
                    'torque_margin_Nm': sm['torque_margin_Nm'],
                    'trans_margin_normalized': sm['trans_margin_normalized'],
                    'torque_margin_normalized': sm['torque_margin_normalized'],
                    'spectral_abscissa': sp_curr['spectral_abscissa'],
                    'is_hurwitz': sp_curr['is_hurwitz']
                })
                
        print(f"  网格核查点数: {len(grid_results)} / 25")
        print(f"  各核查点均位于分配器光滑分支内部? {all_grid_smooth}")
        print(f"  所测试常值扰动下均得到稳定的线性化结果? {all_grid_hurwitz}")
        grid_abscissas = [r['spectral_abscissa'] for r in grid_results]
        print(f"  谱上界跨点分布: [{min(grid_abscissas):.6f}, {max(grid_abscissas):.6f}] s^-1")
        
        # -------------------------------------------------------------
        # 规范化论文结论表述与模型简化陈述
        # -------------------------------------------------------------
        formal_thesis_statement = (
            "【最终主张 (Final Thesis Claim)】\n"
            "在所列假设参数及连续模型下，所考察的两个静止平衡点经数值线性化核查，具有局部指数稳定性；"
            "受限运动过程的性能由仿真与重复实物对照评价。\n\n"
            "【模型简化依据与范围约束 (Model Simplification & Scope Boundary)】\n"
            "1. 连续模型范围：稳定性结论基于建立的30维连续闭环系统（机械误差4维、两个一阶抗饱和状态2维、神经网络权重24维），"
            "非完整数字固件全部状态；\n"
            "2. 简化假设界定：静止平衡点处加速度估计输入取参考加速度 (qddot_est = qddot_d = 0)，未包含高阶观测器/滤波器的瞬态动力学；"
            "稳态处变化率限制器未激活，取 n_s = n_raw，未包含离散步长与历史记忆状态。稳态未激活不代表全运动过程遗漏动态均可忽略；\n"
            "3. 局部性与孤立性：所考察两个平衡点处雅可比矩阵满足 Hurwitz 条件，依据 Lyapunov 间接法保证该平衡点附近的局部指数稳定（LES）"
            "与局部孤立性，不证明全域唯一性；所测试常值扰动下均得到稳定的线性化结果（25组网格扰动点），不据此宣称连续扰动范围内的鲁棒稳定性；\n"
            "4. 运动过程评价：受限运动过程的性能由所建模型仿真、同步与跟踪性能评价，由仿真与105次重复实物对照支撑，"
            "无需且不应将局部定理外推为全局闭环稳定。"
        )
        
        results = {
            'status': 'verified_local_exponential_stability_final',
            'model_definition': {
                'dimension': 30,
                'state_components': 'e (2), edot (2), z (2, two first-order antiwindup states), W_y (12), W_alpha (12)',
                'simplification_notes': [
                    'Acceleration estimate qddot_est taken as qddot_d=0 at static equilibrium (no transient observer dynamics included).',
                    'Slew-rate limiter inactive at static equilibrium with ns = n_raw (no discrete step or history memory included).',
                    'Scope bounded to the investigated static equilibria of the established 30-dim continuous model.'
                ],
                'actual_parameters': {
                    'eps_y_times_eta_y': '25.0 * 0.01 = 0.25 s^-1',
                    'eps_alpha_times_eta_alpha': '18.0 * 0.01 = 0.18 s^-1',
                    'lambda_aw': '20.0 s^-1',
                    'K_aw': '20.0 s^-1'
                }
            },
            'final_thesis_statement': formal_thesis_statement,
            'investigated_equilibrium_1_nominal': {
                'd_ext': [0.0, 0.0],
                'equilibrium': {
                    'success': eq1['success'],
                    'residual': eq1['residual'],
                    'e_y_mm': float(x_star1[0] * 1000.0),
                    'e_alpha_mrad': float(x_star1[1] * 1000.0),
                    'norm_Wy': float(np.linalg.norm(x_star1[6:18])),
                    'norm_Wa': float(np.linalg.norm(x_star1[18:30]))
                },
                'smoothness_audit': smooth1,
                'step_size_sensitivity': step_audit1,
                'spectral_analysis': spec1,
                'lyapunov_analysis': lyap1,
                'jacobian_matrix_30x30': A1.tolist()
            },
            'investigated_equilibrium_2_bias': {
                'd_ext': [float(d_bias[0]), float(d_bias[1])],
                'equilibrium': {
                    'success': eq2['success'],
                    'residual': eq2['residual'],
                    'e_y_mm': float(x_star2[0] * 1000.0),
                    'e_alpha_mrad': float(x_star2[1] * 1000.0),
                    'norm_Wy': float(np.linalg.norm(x_star2[6:18])),
                    'norm_Wa': float(np.linalg.norm(x_star2[18:30]))
                },
                'smoothness_audit': smooth2,
                'step_size_sensitivity': step_audit2,
                'spectral_analysis': spec2,
                'lyapunov_analysis': lyap2,
                'jacobian_matrix_30x30': A2.tolist()
            },
            'grid_disturbance_numerical_check': {
                'note': '所测试常值扰动下均得到稳定的线性化结果，不据此宣称连续扰动范围内的鲁棒稳定性',
                'total_points': len(grid_results),
                'all_smooth': all_grid_smooth,
                'all_hurwitz': all_grid_hurwitz,
                'points': grid_results
            }
        }
        
        # 保存结构化分析数据
        out_json_path = os.path.join(current_dir, 'local_linearization_stability_results.json')
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n完整结构化分析与雅可比矩阵数据已保存至: {out_json_path}")
        print("=" * 80)
        return results


if __name__ == '__main__':
    analyzer = LocalLinearizationAnalyzer()
    analyzer.run_full_analysis()
