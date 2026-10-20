"""
time_varying_request_fast_subsystem.py

时变请求下的抗饱和快子系统性质分析与数值回归
-------------------------------------------------------------
本脚本研究时变请求 v(t) = b(t) + n_s(t) 驱动下，抗饱和快子系统的
非自治摄动动力学与输入至状态稳定性 (ISS)：
    \dot{z} = K_aw [ P(v(t) + z) - v(t) ] - (K_aw + \lambda_aw) z

核心研究变量与平衡点偏差：
    \eta(t) = \bar{z}(t) - \bar{h}(\bar{v}(t)) = \bar{y}(t) - \bar{y}^*(\bar{v}(t))

本轮关键修正与核算依据：
1. 平衡映射雅可比确界：
   在平动受限、力矩未超限分区中，平动平衡点随力矩变化，雅可比谱范数确界为 ||Dh||_2 = 1 / sqrt(2) ≈ 0.70710678；
   全局 Lipschitz 确界严格满足：||h(v1) - h(v2)|| <= (1 / sqrt(2)) ||v1 - v2||；
2. 绝对连续请求下的 ISS 界：
   D^+ ||\eta|| <= -10 ||\eta|| + (1 / sqrt(2)) ||\dot{\bar{v}}||
   ||\eta(t)|| <= exp(-10(t - t_0)) ||\eta(t_0)|| + 0.0707107 * sup ||\dot{\bar{v}}||；
3. 实际零阶保持 (ZOH) 阶跃跳变关系：
   \eta^+ = \eta^- - [h(\bar{v}^+) - h(\bar{v}^-)]，并在步间平滑收缩；
4. 变化率限制器定义澄清：
   75 counts/控制步 (1 ms 步长对应 75,000 counts/s，单侧折算 464.84 N/s)，
   该限制仅作用于补偿量 n_s，不限制基础控制量 b 或全量请求 v = b + n_s；
   指令通过 CAN 总线发送给电调。

结论边界声明：
本脚本仅证明“抗饱和快子系统在时变请求下相对于动平衡点具有确定的输入至状态稳定性 (ISS) 界”；
完整闭环稳定性和奇异摄动适用条件仍未证明，后续须与慢机械动力学和学习动态连接。
"""

import os
import sys
import json
import numpy as np
from scipy.integrate import solve_ivp

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

from frozen_request_fast_subsystem import FrozenRequestFastSubsystemAnalyzer


class TimeVaryingRequestFastSubsystemAnalyzer:
    def __init__(self):
        self.base_analyzer = FrozenRequestFastSubsystemAnalyzer()
        self.c = self.base_analyzer.c
        self.p = self.base_analyzer.p
        
        self.a = self.base_analyzer.a
        self.Le = self.base_analyzer.Le
        self.F_max = self.base_analyzer.F_max
        self.T_max = self.base_analyzer.T_max
        self.K_aw = self.base_analyzer.K_aw          # 20.0
        self.lambda_aw = self.base_analyzer.lambda_aw  # 20.0
        self.L = self.base_analyzer.L                # 40.0
        
        # 理论雅可比与 Lipschitz 常数确界
        self.inv_sqrt2 = float(1.0 / np.sqrt(2.0))
        self.iss_gain = float(self.inv_sqrt2 / 10.0) # 0.070710678...
        
        # 固件变化率限制参数 (CAN 下发给电调)
        self.slew_rate_counts_per_step = 75.0 # counts / step (1 ms)
        self.dt_ctrl = 0.001 # 1 ms
        self.slew_rate_counts_per_sec = self.slew_rate_counts_per_step / self.dt_ctrl # 75,000 counts/s
        self.Kf_single = float(self.c.Kf_L) # 0.0061979167 N/count
        self.slew_rate_single_force_N_per_s = self.slew_rate_counts_per_sec * self.Kf_single # 464.84375 N/s

    def compute_analytic_equilibrium(self, vbar_F, vbar_T):
        """调用已通过验证的平衡点解析计算"""
        return self.base_analyzer.compute_analytic_equilibrium(vbar_F, vbar_T)

    def normalized_allocator(self, ybar_F, ybar_T):
        """归一化力矩优先分配器"""
        return self.base_analyzer.normalized_allocator(ybar_F, ybar_T)

    def compute_analytic_jacobian(self, vbar_F, vbar_T):
        """
        分区计算平衡映射 \bar{h}(\bar{v}) 的解析雅可比矩阵 Dh(v) = \partial \bar{h} / \partial \bar{v}：
        
        分区 1 (可行域未超限): |\bar{v}_T| <= 1, |\bar{v}_F| <= 1 - |\bar{v}_T|
            h = [0, 0]^T
            Dh = [[0, 0], [0, 0]], ||Dh||_2 = 0
            
        分区 2 (力矩未超限，平动受限): |\bar{v}_T| <= 1, |\bar{v}_F| > 1 - |\bar{v}_T|
            h_T = 0
            h_F = -0.5 * \bar{v}_F + 0.5 * sign(\bar{v}_F) * (1 - |\bar{v}_T|)
            Dh = [[-0.5, -0.5 * sign(\bar{v}_F) * sign(\bar{v}_T)], [0, 0]]
            ||Dh||_2 = sqrt((-0.5)^2 + (-0.5)^2) = 1 / sqrt(2) ≈ 0.70710678
            
        分区 3 (力矩受限，平动在裕度内): |\bar{v}_T| > 1, c_F = 0, |\bar{v}_F| <= 0 (即 \bar{v}_F = 0)
            h_T = -0.5 * \bar{v}_T + 0.5 * sign(\bar{v}_T)
            h_F = -0.5 * \bar{v}_F
            Dh = [[-0.5, 0], [0, -0.5]], ||Dh||_2 = 0.5
            
        分区 4 (力矩受限，平动同时超限): |\bar{v}_T| > 1, c_F = 0, |\bar{v}_F| > 0
            h_T = -0.5 * \bar{v}_T + 0.5 * sign(\bar{v}_T)
            h_F = -0.5 * \bar{v}_F
            Dh = [[-0.5, 0], [0, -0.5]], ||Dh||_2 = 0.5
        """
        abs_vT = np.abs(vbar_T)
        abs_vF = np.abs(vbar_F)
        sign_vT = 1.0 if vbar_T > 0 else (-1.0 if vbar_T < 0 else 0.0)
        sign_vF = 1.0 if vbar_F > 0 else (-1.0 if vbar_F < 0 else 0.0)
        
        if abs_vT <= 1.0:
            c_F = 1.0 - abs_vT
            if abs_vF <= c_F:
                region = "Region_1_unconstrained"
                Dh = np.zeros((2, 2), dtype=np.float64)
            else:
                region = "Region_2_trans_constrained_only"
                Dh = np.array([
                    [-0.5, -0.5 * sign_vF * sign_vT],
                    [ 0.0,  0.0]
                ], dtype=np.float64)
        else:
            region = "Region_3_or_4_torque_constrained"
            Dh = np.array([
                [-0.5,  0.0],
                [ 0.0, -0.5]
            ], dtype=np.float64)
            
        spectral_norm = float(np.linalg.norm(Dh, ord=2))
        return Dh, spectral_norm, region

    def verify_partition_jacobian_and_global_lipschitz(self, n_random_pairs=100000, n_cross_pairs=50000):
        """
        检验分区雅可比与全局 Lipschitz 界：
        1. 验证各分区内部数值雅可比与解析雅可比严格吻合；
        2. 在全状态空间随机采样验证 ||h(v1) - h(v2)|| / ||v1 - v2|| <= 1 / sqrt(2)；
        3. 重点对跨边界对 (如跨越 |v_F| + |v_T| = 1 及 |v_T| = 1) 验证 Lipschitz 连续性。
        """
        # 1. 四大典型分区内部解析雅可比检验
        test_points_regions = [
            (0.3, 0.2, "Region 1 内部"),
            (0.8, 0.3, "Region 2 内部"),
            (-0.7, 0.2, "Region 2 内部负平动"),
            (0.0, 1.5, "Region 3 内部 (v_F=0)"),
            (1.5, 1.5, "Region 4 内部双超限"),
            (-2.0, -1.8, "Region 4 内部双负超限")
        ]
        
        region_check_records = []
        eps = 1e-6
        max_jacobian_diff = 0.0
        
        for v_F, v_T, desc in test_points_regions:
            Dh_analytic, norm_analytic, region = self.compute_analytic_jacobian(v_F, v_T)
            
            # 数值差分
            Dh_numeric = np.zeros((2, 2))
            for j, delta in enumerate([[eps, 0.0], [0.0, eps]]):
                _, h_p, _ = self.compute_analytic_equilibrium(v_F + delta[0], v_T + delta[1])
                _, h_m, _ = self.compute_analytic_equilibrium(v_F - delta[0], v_T - delta[1])
                Dh_numeric[:, j] = (h_p - h_m) / (2.0 * eps)
                
            diff = float(np.max(np.abs(Dh_analytic - Dh_numeric)))
            if diff > max_jacobian_diff:
                max_jacobian_diff = diff
                
            region_check_records.append({
                'desc': desc,
                'vbar': [v_F, v_T],
                'region': region,
                'Dh_analytic': Dh_analytic.tolist(),
                'Dh_numeric': Dh_numeric.tolist(),
                'norm_analytic': norm_analytic,
                'diff': diff
            })
            
        # 2. 全局 100,000 对随机点 Lipschitz 界检验
        np.random.seed(20261020)
        max_lipschitz_ratio_random = 0.0
        worst_pair_random = None
        
        for _ in range(n_random_pairs):
            v1 = np.random.uniform(-4.0, 4.0, 2)
            v2 = np.random.uniform(-4.0, 4.0, 2)
            dist = float(np.linalg.norm(v1 - v2))
            if dist < 1e-6:
                continue
            _, h1, _ = self.compute_analytic_equilibrium(v1[0], v1[1])
            _, h2, _ = self.compute_analytic_equilibrium(v2[0], v2[1])
            ratio = float(np.linalg.norm(h1 - h2) / dist)
            if ratio > max_lipschitz_ratio_random:
                max_lipschitz_ratio_random = ratio
                worst_pair_random = {
                    'v1': v1.tolist(),
                    'v2': v2.tolist(),
                    'dist': dist,
                    'h1': h1.tolist(),
                    'h2': h2.tolist(),
                    'ratio': ratio
                }
                
        # 3. 跨边界 50,000 对密集测试
        max_lipschitz_ratio_cross = 0.0
        for _ in range(n_cross_pairs):
            boundary_type = np.random.choice(['sum_boundary', 'torque_boundary'])
            if boundary_type == 'sum_boundary':
                theta = np.random.uniform(0, 2 * np.pi)
                vb = np.array([np.cos(theta), np.sin(theta)])
                vb = vb / (np.abs(vb[0]) + np.abs(vb[1])) # |v_F| + |v_T| = 1
            else:
                sign = np.random.choice([-1.0, 1.0])
                vb = np.array([np.random.uniform(-1.5, 1.5), sign * 1.0]) # |v_T| = 1
                
            delta = np.random.normal(0, 0.02, 2)
            v1 = vb + delta
            v2 = vb - delta
            dist = float(np.linalg.norm(v1 - v2))
            if dist < 1e-7:
                continue
            _, h1, _ = self.compute_analytic_equilibrium(v1[0], v1[1])
            _, h2, _ = self.compute_analytic_equilibrium(v2[0], v2[1])
            ratio = float(np.linalg.norm(h1 - h2) / dist)
            if ratio > max_lipschitz_ratio_cross:
                max_lipschitz_ratio_cross = ratio
                
        tolerance = 1e-12
        passed = bool(
            max_jacobian_diff < 1e-5 and
            max_lipschitz_ratio_random <= (self.inv_sqrt2 + tolerance) and
            max_lipschitz_ratio_cross <= (self.inv_sqrt2 + tolerance)
        )
        
        return {
            'passed': passed,
            'max_jacobian_diff': max_jacobian_diff,
            'theoretical_bound_inv_sqrt2': self.inv_sqrt2,
            'max_lipschitz_ratio_random': max_lipschitz_ratio_random,
            'max_lipschitz_ratio_cross_boundary': max_lipschitz_ratio_cross,
            'worst_pair_random': worst_pair_random,
            'region_check_records': region_check_records
        }

    def run_smooth_and_cross_branch_trajectories(self):
        """
        测试时变连续请求轨迹下的 ISS 界成立性：
        对 4 种典型时变轨迹进行 ODE 仿真与逐点比对：
        - 轨迹 1：椭圆旋转跨区分支轨迹 v(t) = [1.2 sin(5t), 1.5 cos(5t)]
        - 轨迹 2：高频剧烈跨区分支轨迹 v(t) = [2.0 sin(15t), 2.2 cos(12t)]
        - 轨迹 3：直线穿行全象限斜坡轨迹 v(t) 从 (-2, -2) 快速拉至 (+2, +2)
        - 轨迹 4：大范围饱和复合轨迹 (正弦 + 阶跃平滑连接)
        
        验证指标：
        1. 卷积积分界：||\eta(t)|| <= exp(-10t) ||\eta_0|| + (1/sqrt(2)) \int_0^t exp(-10(t-\tau)) ||\dot{v}(\tau)|| d\tau
        2. 峰值上确界：||\eta(t)|| <= exp(-10t) ||\eta_0|| + 0.0707107 * sup ||\dot{v}||
        """
        trajectories = [
            {
                'name': 'elliptical_cross_branch',
                'desc': '中速椭圆跨区分支轨迹 (循环穿越 1, 2, 4 分区)',
                't_span': (0.0, 1.5),
                'v_func': lambda t: np.array([1.2 * np.sin(5.0 * t), 1.5 * np.cos(5.0 * t)]),
                'vdot_func': lambda t: np.array([6.0 * np.cos(5.0 * t), -7.5 * np.sin(5.0 * t)]),
                'sup_vdot': np.sqrt(6.0**2 + 7.5**2), # 9.6047
                'eta0': np.array([0.6, -0.4])
            },
            {
                'name': 'high_frequency_cross_branch',
                'desc': '高频剧烈振荡轨迹 (深度饱和与快速切换)',
                't_span': (0.0, 1.0),
                'v_func': lambda t: np.array([2.0 * np.sin(15.0 * t), 2.2 * np.cos(12.0 * t)]),
                'vdot_func': lambda t: np.array([30.0 * np.cos(15.0 * t), -26.4 * np.sin(12.0 * t)]),
                'sup_vdot': np.sqrt(30.0**2 + 26.4**2), # 40.0
                'eta0': np.array([-1.0, 0.8])
            },
            {
                'name': 'linear_ramp_diagonal',
                'desc': '对角斜坡贯穿全象限轨迹',
                't_span': (0.0, 1.0),
                'v_func': lambda t: np.array([-2.0 + 4.0 * t, -1.8 + 3.6 * t]),
                'vdot_func': lambda t: np.array([4.0, 3.6]),
                'sup_vdot': np.sqrt(4.0**2 + 3.6**2), # 5.3814
                'eta0': np.array([0.2, 0.2])
            },
            {
                'name': 'smooth_step_transition',
                'desc': '平滑 S 型跃迁轨迹 (从可行域跃至深饱和区)',
                't_span': (0.0, 1.2),
                'v_func': lambda t: np.array([0.2 + 1.8 / (1.0 + np.exp(-10.0 * (t - 0.5))),
                                              0.1 + 2.0 / (1.0 + np.exp(-10.0 * (t - 0.5)))]),
                'vdot_func': lambda t: np.array([18.0 * np.exp(-10.0 * (t - 0.5)) / (1.0 + np.exp(-10.0 * (t - 0.5)))**2,
                                                 20.0 * np.exp(-10.0 * (t - 0.5)) / (1.0 + np.exp(-10.0 * (t - 0.5)))**2]),
                'sup_vdot': np.sqrt((18.0 * 0.25)**2 + (20.0 * 0.25)**2), # 6.7268
                'eta0': np.array([0.0, 0.0])
            }
        ]
        
        eval_results = []
        max_integral_violation = 0.0
        max_sup_violation = 0.0
        
        for traj in trajectories:
            t_span = traj['t_span']
            v_func = traj['v_func']
            vdot_func = traj['vdot_func']
            sup_vdot = traj['sup_vdot']
            eta0 = traj['eta0']
            
            # 计算初值 z0
            v0 = v_func(t_span[0])
            _, h0, _ = self.compute_analytic_equilibrium(v0[0], v0[1])
            z0 = h0 + eta0
            
            def odesys(t, z):
                v = v_func(t)
                y = v + z
                p = self.normalized_allocator(y[0], y[1])
                return self.K_aw * (p - v) - self.L * z
                
            t_eval = np.linspace(t_span[0], t_span[1], 1501)
            sol = solve_ivp(odesys, t_span, z0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
            
            norm_eta0 = float(np.linalg.norm(eta0))
            norm_eta_arr = np.zeros(len(sol.t))
            conv_bound_arr = np.zeros(len(sol.t))
            sup_bound_arr = np.zeros(len(sol.t))
            
            for i, t in enumerate(sol.t):
                vt = v_func(t)
                _, ht, _ = self.compute_analytic_equilibrium(vt[0], vt[1])
                zt = sol.y[:, i]
                etat = zt - ht
                norm_eta_arr[i] = float(np.linalg.norm(etat))
                
                # 数值计算卷积项
                if t > t_span[0]:
                    tau_grid = np.linspace(t_span[0], t, 401)
                    vdot_norms = np.array([np.linalg.norm(vdot_func(tau)) for tau in tau_grid])
                    weights = np.exp(-10.0 * (t - tau_grid))
                    dt_sub = (t - t_span[0]) / 400.0
                    conv_integral = float(np.trapezoid(weights * vdot_norms, dx=dt_sub))
                else:
                    conv_integral = 0.0
                    
                conv_b = float(np.exp(-10.0 * (t - t_span[0])) * norm_eta0 + self.inv_sqrt2 * conv_integral)
                sup_b = float(np.exp(-10.0 * (t - t_span[0])) * norm_eta0 + self.iss_gain * sup_vdot)
                
                conv_bound_arr[i] = conv_b
                sup_bound_arr[i] = sup_b
                
            viol_conv = float(np.max(norm_eta_arr - conv_bound_arr))
            viol_sup = float(np.max(norm_eta_arr - sup_bound_arr))
            if viol_conv > max_integral_violation:
                max_integral_violation = viol_conv
            if viol_sup > max_sup_violation:
                max_sup_violation = viol_sup
                
            eval_results.append({
                'name': traj['name'],
                'desc': traj['desc'],
                'sup_vdot': float(sup_vdot),
                'norm_eta0': norm_eta0,
                'peak_actual_norm_eta': float(np.max(norm_eta_arr)),
                'peak_conv_bound': float(np.max(conv_bound_arr)),
                'peak_sup_bound': float(np.max(sup_bound_arr)),
                'max_violation_conv': viol_conv,
                'max_violation_sup': viol_sup,
                'strictly_bounded_conv': bool(viol_conv <= 1e-12),
                'strictly_bounded_sup': bool(viol_sup <= 1e-12)
            })
            
        all_passed = bool(max_integral_violation <= 1e-12 and max_sup_violation <= 1e-12)
        return {
            'all_trajectories_passed': all_passed,
            'max_integral_violation': max_integral_violation,
            'max_sup_violation': max_sup_violation,
            'trajectories': eval_results
        }

    def verify_sampled_data_step_and_hold_jumps(self, n_steps=1000):
        """
        验证实际控制中零阶保持 (ZOH, Ts = 1 ms) 请求下的跳变与衰减关系：
        1. 在采样时刻 t_k，由于实际物理状态 z 连续：z^+ = z^-，
           而平衡映射随离散请求产生阶跃跳变，偏差准确满足跳变恒等式：
           \eta^+ = \eta^- - [h(\bar{v}^+) - h(\bar{v}^-)]；
        2. 跳变幅值受 Lipschitz 常数严格控制：
           ||\eta^+|| <= ||\eta^-|| + (1 / sqrt(2)) ||\bar{v}^+ - \bar{v}^-||；
        3. 在采样间隔 [t_k, t_{k+1}) 内，请求冻结，衰减严格满足：
           ||\eta(t_{k+1}^-)|| <= exp(-10 Ts) ||\eta(t_k^+)||；
        4. 综合得出混合离散步进收敛界：
           ||\eta_{k+1}^-|| <= exp(-10 Ts) ||\eta_k^-|| + (exp(-10 Ts) / sqrt(2)) ||\Delta \bar{v}_k||。
        """
        np.random.seed(20261020)
        Ts = self.dt_ctrl # 0.001 s
        
        # 对应固件 75 counts/步的单侧折算归一化增量步长
        max_delta_norm_F = (self.slew_rate_counts_per_step * self.Kf_single * 2.0) / self.F_max # 75 * 0.0062 * 2 / 9.9167 ≈ 0.09375
        max_delta_norm_T = (self.slew_rate_counts_per_step * self.Kf_single * self.Le) / self.T_max # 75 * 0.0062 * 0.6 / 2.9750 ≈ 0.09375
        max_step_bound = np.array([max_delta_norm_F, max_delta_norm_T])
        
        v_k = np.array([0.0, 0.0])
        z = np.array([0.0, 0.0])
        _, h_k, _ = self.compute_analytic_equilibrium(v_k[0], v_k[1])
        eta = z - h_k
        
        max_identity_err = 0.0
        max_jump_bound_slack = 0.0
        max_inter_sample_decay_violation = 0.0
        
        exp_decay_step = np.exp(-10.0 * Ts)
        
        jump_records = []
        
        for k in range(n_steps):
            # 模拟受变化率限制的离散目标更新
            target_v = np.array([
                1.5 * np.sin(8.0 * k * Ts) + 0.3 * np.cos(25.0 * k * Ts),
                1.8 * np.cos(6.0 * k * Ts) + 0.4 * np.sin(20.0 * k * Ts)
            ])
            delta_v = np.clip(target_v - v_k, -max_step_bound, max_step_bound)
            v_next = v_k + delta_v
            
            # 跳变前状态
            _, h_minus, _ = self.compute_analytic_equilibrium(v_k[0], v_k[1])
            _, h_plus, _ = self.compute_analytic_equilibrium(v_next[0], v_next[1])
            eta_minus = z - h_minus
            
            # 跳变代数恒等核查: \eta^+ = \eta^- - (h(v^+) - h(v^-))
            eta_plus_formula = eta_minus - (h_plus - h_minus)
            eta_plus_physical = z - h_plus # z 连续
            
            ident_err = float(np.max(np.abs(eta_plus_formula - eta_plus_physical)))
            if ident_err > max_identity_err:
                max_identity_err = ident_err
                
            # 跳变模长三角不等式核查
            norm_minus = float(np.linalg.norm(eta_minus))
            norm_plus = float(np.linalg.norm(eta_plus_physical))
            delta_v_norm = float(np.linalg.norm(delta_v))
            jump_upper_bound = norm_minus + self.inv_sqrt2 * delta_v_norm
            jump_slack = jump_upper_bound - norm_plus
            if jump_slack < max_jump_bound_slack:
                max_jump_bound_slack = jump_slack
                
            # 步间连续动态流演化 (dt_sub = 0.1 ms)
            z_curr = z.copy()
            dt_sub = 0.0001
            for _ in range(10):
                y = v_next + z_curr
                p = self.normalized_allocator(y[0], y[1])
                zdot = self.K_aw * (p - v_next) - self.L * z_curr
                z_curr += dt_sub * zdot
            z = z_curr
            
            # 步末偏差与衰减验证
            eta_end = z - h_plus
            norm_end = float(np.linalg.norm(eta_end))
            expected_decay_bound = norm_plus * exp_decay_step
            decay_viol = norm_end - expected_decay_bound
            if decay_viol > max_inter_sample_decay_violation:
                max_inter_sample_decay_violation = decay_viol
                
            v_k = v_next
            
            if k % 200 == 0:
                jump_records.append({
                    'step': k,
                    'time_s': k * Ts,
                    'norm_eta_minus': norm_minus,
                    'norm_eta_plus': norm_plus,
                    'norm_eta_end': norm_end,
                    'delta_v_norm': delta_v_norm
                })
                
        passed = bool(
            max_identity_err < 1e-12 and
            max_jump_bound_slack >= -1e-12 and
            max_inter_sample_decay_violation <= 1e-12
        )
        
        return {
            'passed': passed,
            'n_steps_tested': n_steps,
            'max_identity_residual': max_identity_err,
            'min_jump_bound_slack': max_jump_bound_slack,
            'max_inter_sample_decay_violation': max_inter_sample_decay_violation,
            'exp_decay_per_step': exp_decay_step,
            'sample_step_records': jump_records
        }

    def execute_full_time_varying_audit(self):
        """执行完整时变请求审计并保存 JSON"""
        # 1. 雅可比与全局 Lipschitz 验证
        jac_res = self.verify_partition_jacobian_and_global_lipschitz()
        
        # 2. 连续平滑与跨分支 ISS 轨迹验证
        traj_res = self.run_smooth_and_cross_branch_trajectories()
        
        # 3. 采样保持跳变关系验证
        jump_res = self.verify_sampled_data_step_and_hold_jumps()
        
        # 4. 变化率与执行链参数明确
        rate_limiter_spec = {
            'slew_rate_setting': f"{self.slew_rate_counts_per_step} counts/control_step",
            'control_cycle_ms': self.dt_ctrl * 1000.0,
            'command_rate_limit_counts_per_sec': self.slew_rate_counts_per_sec,
            'thrust_rate_limit_single_N_per_sec': self.slew_rate_single_force_N_per_s,
            'scope_of_limiter': (
                '该限制严格仅作用于神经网络补偿量 n_s 的差分更新，'
                '绝不限制基础滑模/前馈控制量 b，亦不限制全量请求 v = b + n_s。'
                '基础控制量 b 的动态仍取决于机械跟踪速度与外生参考阶跃。'
            ),
            'bus_interface': 'CAN bus to Electronic Speed Controllers (ESC), NOT DAC/PWM'
        }
        
        all_passed = bool(
            jac_res['passed'] and
            traj_res['all_trajectories_passed'] and
            jump_res['passed']
        )
        
        conclusion = {
            'certification_status': 'passed_time_varying_request_fast_subsystem' if all_passed else 'failed',
            'formal_statement': (
                f'在当前对称执行能力配置下，时变请求抗饱和子系统相对于动平衡点 h(v(t)) '
                f'严格满足输入至状态稳定性 (ISS)：'
                f'对任意绝对连续请求 v(t)，偏差范数上界恒满足 '
                f'||eta(t)|| <= exp(-10(t - t_0)) ||eta(t_0)|| + {self.iss_gain:.7f} * sup ||\dot{{v}}||；'
                f'对于 1 ms 采样保持请求，离散跳变严格满足 eta^+ = eta^- - \Delta h 且在步间按 exp(-0.01) 收缩。'
            ),
            'strict_boundaries': (
                '本结论仅闭合了“给定请求变化率下抗饱和跟随误差的有界性输入通道”；'
                '完整闭环系统的 UUB 稳定性和奇异摄动时间尺度解耦条件仍【未证明 (not_proved)】。'
                '下一步必须将该跟随界代入慢子系统，分析动态饱和残差对机械位置误差和神经网络权值自适应的逆向耦合影响。'
            )
        }
        
        full_results = {
            'audit_timestamp': '2026-10-20T08:15:00Z',
            'theoretical_constants': {
                'K_aw': self.K_aw,
                'lambda_aw': self.lambda_aw,
                'L': self.L,
                'inv_sqrt2': self.inv_sqrt2,
                'iss_gain_analytical': self.iss_gain
            },
            'rate_limiter_and_hardware_spec': rate_limiter_spec,
            'partition_jacobian_and_lipschitz_verification': jac_res,
            'time_varying_iss_trajectories_verification': traj_res,
            'sampled_data_step_and_hold_verification': jump_res,
            'conclusion_and_boundaries': conclusion
        }
        
        json_path = os.path.join(cur_dir, "time_varying_request_fast_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
            
        return full_results


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    analyzer = TimeVaryingRequestFastSubsystemAnalyzer()
    res = analyzer.execute_full_time_varying_audit()
    
    print("=" * 80)
    print("时变请求下的抗饱和快子系统性质分析与数值回归完成")
    print("=" * 80)
    
    c_th = res['theoretical_constants']
    print(f"1. 理论常数确界:")
    print(f"   平衡映射全局 Lipschitz 常数: 1 / sqrt(2) = {c_th['inv_sqrt2']:.8f}")
    print(f"   ISS 渐近增益上限:            1 / (10 * sqrt(2)) = {c_th['iss_gain_analytical']:.8f}")
    
    jac = res['partition_jacobian_and_lipschitz_verification']
    print(f"\n2. 分区雅可比与全局 Lipschitz 验证: {'PASS' if jac['passed'] else 'FAIL'}")
    print(f"   分区内部解析雅可比与数值差分最大误差: {jac['max_jacobian_diff']:.2e}")
    print(f"   100,000 组随机点最大 Lipschitz 比值:  {jac['max_lipschitz_ratio_random']:.8f} (<= {jac['theoretical_bound_inv_sqrt2']:.8f})")
    print(f"   50,000 组跨边界点最大 Lipschitz 比值: {jac['max_lipschitz_ratio_cross_boundary']:.8f}")
    
    traj = res['time_varying_iss_trajectories_verification']
    print(f"\n3. 连续平滑与跨分支时变轨迹 ISS 验证: {'PASS' if traj['all_trajectories_passed'] else 'FAIL'}")
    print(f"   卷积积分界最大违例: {traj['max_integral_violation']:.2e} (<= 0 为通过)")
    print(f"   上确界最大违例:     {traj['max_sup_violation']:.2e} (<= 0 为通过)")
    for tr in traj['trajectories']:
        print(f"   - {tr['name']:28s}: peak actual={tr['peak_actual_norm_eta']:.4f} <= sup_bound={tr['peak_sup_bound']:.4f}")
        
    jump = res['sampled_data_step_and_hold_verification']
    print(f"\n4. 采样保持 (ZOH 1ms) 跳变与衰减验证 (1,000 步): {'PASS' if jump['passed'] else 'FAIL'}")
    print(f"   跳变代数恒等式最大残差:   {jump['max_identity_residual']:.2e}")
    print(f"   跳变模长三角不等式最小余量: {jump['min_jump_bound_slack']:.2e} (>= 0)")
    print(f"   步间连续衰减最大违例:     {jump['max_inter_sample_decay_violation']:.2e} (<= 0)")
    
    rl = res['rate_limiter_and_hardware_spec']
    print(f"\n5. 硬件与变化率限制说明:")
    print(f"   设定值: {rl['slew_rate_setting']} ({rl['thrust_rate_limit_single_N_per_sec']:.2f} N/s 单侧折算)")
    print(f"   总线:   {rl['bus_interface']}")
    print(f"   范围:   {rl['scope_of_limiter']}")
    
    conc = res['conclusion_and_boundaries']
    print(f"\n6. 结论与边界声明:")
    print(f"   认证状态: {conc['certification_status']}")
    print(f"   正式结论: {conc['formal_statement']}")
    print(f"   严格边界: {conc['strict_boundaries']}")
    print(f"\n完整结构化结果已导出至: time_varying_request_fast_results.json")
