"""
frozen_request_fast_subsystem.py

冻结请求下的抗饱和快子系统性质分析与数值回归
-------------------------------------------------------------
本脚本专门研究冻结请求 v = b + n_s 下抗饱和子系统的解析动力学与一致指数收敛性：
    \dot{z} = K_aw [ P(v + z) - v ] - (K_aw + \lambda_aw) z

适用范围与约束前提：
1. 仅适用于左右对称执行能力配置：
   K_{f,L} * I_{max,L} == K_{f,R} * I_{max,R}
   不满足该对称条件时立即中止，严禁推广至非对称执行能力配置；
2. 基础请求量 v 视为常数（冻结）；
3. 严格使用实装参数：K_aw = 20.0, \lambda_aw = 20.0 (L = 40.0)。

结论边界声明：
本分析仅证明“在当前对称执行能力配置、请求量冻结条件下，抗饱和子系统相对于其请求相关平衡点
具有一致指数收敛性质”；
严禁宣称完整闭环稳定已证明，亦暂不宣布奇异摄动条件已满足。
"""

import os
import sys
import json
import numpy as np
from scipy.integrate import solve_ivp

# 引入已有模型类以获取配置参数 (只读，不修改原脚本)
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

from continuous_closed_loop_model import ContinuousClosedLoopAnalysis


class FrozenRequestFastSubsystemAnalyzer:
    def __init__(self):
        self.analyzer = ContinuousClosedLoopAnalysis()
        self.c = self.analyzer.c
        self.p = self.analyzer.p
        
        # 1. 验证对称执行能力前置条件
        self.a_L = float(self.c.Kf_L * self.c.Imax_L)
        self.a_R = float(self.c.Kf_R * self.c.Imax_R)
        self.is_symmetric = bool(np.isclose(self.a_L, self.a_R, rtol=1e-9, atol=1e-12))
        
        if not self.is_symmetric:
            raise ValueError(
                f"Symmetric capability violated: a_L = {self.a_L}, a_R = {self.a_R}. "
                "Simplified normalized allocator cannot be applied!"
            )
            
        self.a = self.a_L
        self.Le = float(self.c.Le)
        self.F_max = float(2.0 * self.a)
        self.T_max = float(self.Le * self.a)
        
        # 控制器参数
        self.K_aw = float(self.c.K_aw)          # 20.0
        self.lambda_aw = float(self.c.lambda_aw)  # 20.0
        self.L = float(self.K_aw + self.lambda_aw) # 40.0

    def raw_thrust_allocator_sync(self, FG_des, Talpha_des):
        """原物理控制器中的力矩优先分配器"""
        Kf_L, Kf_R = self.c.Kf_L, self.c.Kf_R
        Imax_L, Imax_R = self.c.Imax_L, self.c.Imax_R
        Le = self.c.Le
        
        FL_max = Kf_L * Imax_L
        FR_max = Kf_R * Imax_R
        DeltaF_des = (2.0 * Talpha_des) / Le
        DeltaF_max = FR_max + FL_max
        DeltaF_alloc = np.clip(DeltaF_des, -DeltaF_max, DeltaF_max)
        
        FL_lower = max(-FL_max, -FR_max - DeltaF_alloc)
        FL_upper = min( FL_max,  FR_max - DeltaF_alloc)
        FL_target = 0.5 * FG_des - 0.5 * DeltaF_alloc
        FL_alloc = np.clip(FL_target, FL_lower, FL_upper)
        FR_alloc = FL_alloc + DeltaF_alloc
        
        iL_raw = FL_alloc / Kf_L
        iR_raw = -FR_alloc / Kf_R
        iL_act = np.clip(iL_raw, -Imax_L, Imax_L)
        iR_act = np.clip(iR_raw, -Imax_R, Imax_R)
        
        FL_real = Kf_L * iL_act
        FR_real = -Kf_R * iR_act
        FG_alloc = FL_real + FR_real
        Talpha_alloc = 0.5 * Le * (FR_real - FL_real)
        return np.array([FG_alloc, Talpha_alloc], dtype=np.float64)

    def normalized_allocator(self, ybar_F, ybar_T):
        """
        归一化分配器：
            p_T = clip(\bar{y}_T, -1, 1)
            p_F = clip(\bar{y}_F, -[1 - |p_T|], 1 - |p_T|)
        """
        p_T = np.clip(ybar_T, -1.0, 1.0)
        c_F = 1.0 - np.abs(p_T)
        p_F = np.clip(ybar_F, -c_F, c_F)
        return np.array([p_F, p_T], dtype=np.float64)

    def verify_allocator_equivalence(self):
        """
        验证归一化分配器与原物理分配器严格代数一致性：
        覆盖：未饱和、仅平动受限、力矩受限、两者同时受限、正负方向与各分支分界线。
        """
        test_points = [
            # 1. 未饱和 (可行域内部)
            (0.0, 0.0, "zero"),
            (0.3, 0.2, "interior_positive"),
            (-0.4, -0.3, "interior_negative"),
            (0.5, -0.3, "interior_mixed_1"),
            (-0.2, 0.6, "interior_mixed_2"),
            # 2. 仅平动受限 (|y_T| < 1, |y_F| > 1 - |y_T|)
            (0.8, 0.4, "trans_saturated_pos"),
            (-0.9, 0.3, "trans_saturated_neg_F"),
            (1.5, 0.0, "pure_trans_saturated"),
            (-1.8, 0.0, "pure_trans_saturated_neg"),
            # 3. 力矩受限 (|y_T| > 1, |y_F| <= 0)
            (0.0, 1.5, "pure_torque_saturated_pos"),
            (0.0, -1.8, "pure_torque_saturated_neg"),
            (0.1, 1.2, "torque_saturated_small_F"),
            (-0.1, -1.4, "torque_saturated_neg_small_F"),
            # 4. 两者同时受限 (|y_T| > 1, |y_F| > 0)
            (1.5, 1.5, "both_saturated_pp"),
            (-1.5, 1.5, "both_saturated_np"),
            (2.0, -2.0, "both_saturated_pn"),
            (-3.0, -2.5, "both_saturated_nn"),
            # 5. 分支分界线
            (0.7, 0.3, "boundary_sum_1_pos"),
            (-0.6, 0.4, "boundary_sum_1_neg_F"),
            (0.5, -0.5, "boundary_sum_1_neg_T"),
            (-0.5, -0.5, "boundary_sum_1_neg_both"),
            (0.0, 1.0, "boundary_pure_torque_pos"),
            (0.0, -1.0, "boundary_pure_torque_neg"),
            (1.0, 0.0, "boundary_pure_trans_pos"),
            (-1.0, 0.0, "boundary_pure_trans_neg"),
            (0.2, 1.0, "boundary_torque_1_with_F"),
            (-0.3, -1.0, "boundary_torque_neg1_with_F")
        ]
        
        records = []
        max_abs_err_F = 0.0
        max_abs_err_T = 0.0
        
        for yb_F, yb_T, desc in test_points:
            # 物理请求量
            y_phys = np.array([yb_F * self.F_max, yb_T * self.T_max])
            u_phys = self.raw_thrust_allocator_sync(y_phys[0], y_phys[1])
            u_phys_norm = np.array([u_phys[0] / self.F_max, u_phys[1] / self.T_max])
            
            p_norm = self.normalized_allocator(yb_F, yb_T)
            
            err_F = float(np.abs(u_phys_norm[0] - p_norm[0]))
            err_T = float(np.abs(u_phys_norm[1] - p_norm[1]))
            if err_F > max_abs_err_F:
                max_abs_err_F = err_F
            if err_T > max_abs_err_T:
                max_abs_err_T = err_T
                
            records.append({
                'desc': desc,
                'ybar_F': yb_F,
                'ybar_T': yb_T,
                'p_norm_F': float(p_norm[0]),
                'p_norm_T': float(p_norm[1]),
                'u_phys_norm_F': float(u_phys_norm[0]),
                'u_phys_norm_T': float(u_phys_norm[1]),
                'abs_err_F': err_F,
                'abs_err_T': err_T
            })
            
        is_consistent = bool(max_abs_err_F < 1e-12 and max_abs_err_T < 1e-12)
        return {
            'is_consistent': is_consistent,
            'max_abs_err_F': max_abs_err_F,
            'max_abs_err_T': max_abs_err_T,
            'test_points_count': len(records),
            'records': records
        }

    def compute_analytic_equilibrium(self, vbar_F, vbar_T):
        """
        解析计算冻结请求下的平衡点：
        1. 力矩通道：
           若 |\bar{v}_T| <= 1: \bar{y}_T^* = \bar{v}_T
           若 |\bar{v}_T| > 1:  \bar{y}_T^* = (\lambda_aw * \bar{v}_T + K_aw * sign(\bar{v}_T)) / L
        2. 平动通道：
           c_F = 1 - |clip(\bar{y}_T^*, -1, 1)|
           若 |\bar{v}_F| <= c_F: \bar{y}_F^* = \bar{v}_F
           若 |\bar{v}_F| > c_F:  \bar{y}_F^* = (\lambda_aw * \bar{v}_F + K_aw * c_F * sign(\bar{v}_F)) / L
        3. 抗饱和平衡位置：
           \bar{h}(\bar{v}) = \bar{y}^* - \bar{v}
        """
        # 力矩通道
        if np.abs(vbar_T) <= 1.0:
            ybar_T_star = float(vbar_T)
        else:
            sign_T = 1.0 if vbar_T > 0 else -1.0
            ybar_T_star = float((self.lambda_aw * vbar_T + self.K_aw * sign_T) / self.L)
            
        p_T_star = float(np.clip(ybar_T_star, -1.0, 1.0))
        c_F = float(1.0 - np.abs(p_T_star))
        
        # 平动通道
        if np.abs(vbar_F) <= c_F:
            ybar_F_star = float(vbar_F)
        else:
            sign_F = 1.0 if vbar_F > 0 else -1.0
            ybar_F_star = float((self.lambda_aw * vbar_F + self.K_aw * c_F * sign_F) / self.L)
            
        ybar_star = np.array([ybar_F_star, ybar_T_star], dtype=np.float64)
        vbar = np.array([vbar_F, vbar_T], dtype=np.float64)
        hbar = ybar_star - vbar
        return ybar_star, hbar, c_F

    def verify_analytic_equilibrium(self):
        """
        代回原微分方程检验平衡点残差：
        \dot{\bar{y}} = K_aw * p(\bar{y}^*) - L * \bar{y}^* + \lambda_aw * \bar{v} == 0
        \dot{\bar{z}} = K_aw * [p(\bar{v} + \bar{h}) - \bar{v}] - L * \bar{h} == 0
        """
        test_v = [
            # 未受限请求 -> 应得 h == 0
            (0.0, 0.0, "zero_request"),
            (0.4, 0.3, "feasible_interior_pos"),
            (-0.3, -0.4, "feasible_interior_neg"),
            (0.7, 0.3, "feasible_boundary_sum1"),
            # 受限请求 -> 允许 h != 0
            (0.8, 0.5, "trans_constrained"),
            (-0.9, 0.2, "trans_constrained_neg"),
            (0.2, 1.5, "torque_constrained"),
            (-0.5, -2.0, "torque_constrained_neg"),
            (2.0, 2.0, "both_constrained_pp"),
            (-3.0, -1.5, "both_constrained_nn"),
            (0.5, 1.0, "boundary_torque_exact_1"),
            (1.0, 0.0, "boundary_trans_exact_1"),
            (0.0, -1.0, "boundary_torque_exact_neg1"),
            (10.0, 10.0, "heavy_saturation")
        ]
        
        records = []
        max_ode_residual = 0.0
        
        for v_F, v_T, desc in test_v:
            vbar = np.array([v_F, v_T], dtype=np.float64)
            ybar_star, hbar, c_F = self.compute_analytic_equilibrium(v_F, v_T)
            
            p_star = self.normalized_allocator(ybar_star[0], ybar_star[1])
            ydot_res = self.K_aw * p_star - self.L * ybar_star + self.lambda_aw * vbar
            
            # 以 z = h 表示的等价方程
            zdot_res = self.K_aw * (p_star - vbar) - self.L * hbar
            
            res_val = float(np.max(np.abs(ydot_res)))
            if res_val > max_ode_residual:
                max_ode_residual = res_val
                
            is_unconstrained = bool(np.abs(v_T) <= 1.0 and np.abs(v_F) <= (1.0 - np.abs(v_T)))
            h_is_zero = bool(np.allclose(hbar, 0.0, atol=1e-12))
            
            records.append({
                'desc': desc,
                'vbar': [v_F, v_T],
                'is_unconstrained': is_unconstrained,
                'ybar_star': ybar_star.tolist(),
                'hbar': hbar.tolist(),
                'h_is_zero': h_is_zero,
                'c_F': c_F,
                'p_star': p_star.tolist(),
                'ode_residual': res_val
            })
            
        all_passed = bool(max_ode_residual < 1e-12)
        return {
            'all_passed': all_passed,
            'max_ode_residual': max_ode_residual,
            'records': records
        }

    def verify_convergence_inequalities(self, n_samples=50000, seed=20261020):
        """
        验证平衡点偏差的收敛不等式：
        定义 \eta = \bar{z} - \bar{h}(\bar{v}) = \bar{y} - \bar{y}^*
        
        1. 力矩通道：
           \eta_T * \dot{\eta}_T <= -20 * \eta_T^2
        2. 平动通道：
           \eta_F * \dot{\eta}_F <= -20 * \eta_F^2 + 20 * |\eta_F * \eta_T|
        3. 复合候选函数 W_f = 0.5 * (\eta_F^2 + \eta_T^2):
           \dot{W}_f <= -20 (\eta_F^2 + \eta_T^2) + 20 |\eta_F \eta_T| <= -20 W_f
        """
        np.random.seed(seed)
        
        min_slack_T = float('inf')
        min_slack_F = float('inf')
        min_slack_Wf_1 = float('inf')
        min_slack_Wf_2 = float('inf')
        
        worst_pt = None
        
        for _ in range(n_samples):
            v_F = np.random.uniform(-4.0, 4.0)
            v_T = np.random.uniform(-4.0, 4.0)
            ybar_star, _, _ = self.compute_analytic_equilibrium(v_F, v_T)
            
            eta_F = np.random.uniform(-3.0, 3.0)
            eta_T = np.random.uniform(-3.0, 3.0)
            eta = np.array([eta_F, eta_T], dtype=np.float64)
            
            ybar = ybar_star + eta
            p = self.normalized_allocator(ybar[0], ybar[1])
            p_star = self.normalized_allocator(ybar_star[0], ybar_star[1])
            
            # \dot{\eta} = -L * \eta + K_aw * (p(y) - p(y^*))
            etadot = -self.L * eta + self.K_aw * (p - p_star)
            
            # 不等式 1: \eta_T * \dot{\eta}_T <= -20 * \eta_T^2
            # 余量 = 右端 - 左端 >= 0
            slack_T = (-20.0 * eta_T**2) - (eta_T * etadot[1])
            if slack_T < min_slack_T:
                min_slack_T = float(slack_T)
                
            # 不等式 2: \eta_F * \dot{\eta}_F <= -20 * \eta_F^2 + 20 * |\eta_F \eta_T|
            slack_F = (-20.0 * eta_F**2 + 20.0 * np.abs(eta_F * eta_T)) - (eta_F * etadot[0])
            if slack_F < min_slack_F:
                min_slack_F = float(slack_F)
                
            # 不等式 3: \dot{W}_f <= -20 (\eta_F^2 + \eta_T^2) + 20 |\eta_F \eta_T|
            Wf = 0.5 * (eta_F**2 + eta_T**2)
            Wf_dot = eta_F * etadot[0] + eta_T * etadot[1]
            bound_1 = -20.0 * (eta_F**2 + eta_T**2) + 20.0 * np.abs(eta_F * eta_T)
            slack_Wf_1 = bound_1 - Wf_dot
            if slack_Wf_1 < min_slack_Wf_1:
                min_slack_Wf_1 = float(slack_Wf_1)
                
            # 不等式 4: \dot{W}_f <= -20 W_f
            bound_2 = -20.0 * Wf
            slack_Wf_2 = bound_2 - Wf_dot
            if slack_Wf_2 < min_slack_Wf_2:
                min_slack_Wf_2 = float(slack_Wf_2)
                worst_pt = {
                    'v': [float(v_F), float(v_T)],
                    'eta': [float(eta_F), float(eta_T)],
                    'Wf': float(Wf),
                    'Wf_dot': float(Wf_dot),
                    'bound_neg20_Wf': float(bound_2),
                    'slack': float(slack_Wf_2)
                }
                
        passed = bool(
            min_slack_T >= -1e-11 and
            min_slack_F >= -1e-11 and
            min_slack_Wf_1 >= -1e-11 and
            min_slack_Wf_2 >= -1e-11
        )
        
        return {
            'passed': passed,
            'n_samples_tested': n_samples,
            'min_slack_T_channel': min_slack_T,
            'min_slack_F_channel': min_slack_F,
            'min_slack_Wf_bound_1': min_slack_Wf_1,
            'min_slack_Wf_exponential': min_slack_Wf_2,
            'worst_case_point': worst_pt
        }

    def run_ode_convergence_regression(self):
        """
        ODE 数值回归检验：
        针对不同典型场景，检验真实轨迹是否恒满足指数收敛上界：
        \|\eta(t)\| <= exp(-10 * (t - t_0)) * \|\eta(t_0)\|
        """
        scenarios = [
            {
                'name': 'feasible_interior',
                'v': [0.4, 0.3],
                'eta0': [1.0, -0.8],
                'desc': '请求在可行域内'
            },
            {
                'name': 'trans_saturated_only',
                'v': [0.9, 0.2],
                'eta0': [-0.5, 0.5],
                'desc': '平动请求超限'
            },
            {
                'name': 'torque_saturated_only',
                'v': [0.1, 1.5],
                'eta0': [0.8, -1.2],
                'desc': '力矩请求超限'
            },
            {
                'name': 'both_saturated',
                'v': [2.0, 2.0],
                'eta0': [-1.5, 1.0],
                'desc': '平动与力矩两者同时超限'
            },
            {
                'name': 'boundary_sum1',
                'v': [0.7, 0.3],
                'eta0': [0.5, -0.5],
                'desc': '分配器分支边界点'
            }
        ]
        
        results = []
        max_overall_violation = 0.0
        
        for sc in scenarios:
            v_F, v_T = sc['v']
            vbar = np.array([v_F, v_T], dtype=np.float64)
            ybar_star, _, _ = self.compute_analytic_equilibrium(v_F, v_T)
            eta0 = np.array(sc['eta0'], dtype=np.float64)
            
            def rhs(t, eta):
                y = ybar_star + eta
                p = self.normalized_allocator(y[0], y[1])
                p_star = self.normalized_allocator(ybar_star[0], ybar_star[1])
                return -self.L * eta + self.K_aw * (p - p_star)
                
            t_span = (0.0, 0.5)
            t_eval = np.linspace(0.0, 0.5, 201)
            sol = solve_ivp(rhs, t_span, eta0, t_eval=t_eval, rtol=1e-10, atol=1e-12)
            
            norm_eta0 = float(np.linalg.norm(eta0))
            norm_eta_t = np.linalg.norm(sol.y, axis=0)
            bound_t = norm_eta0 * np.exp(-10.0 * sol.t)
            
            diff = norm_eta_t - bound_t
            max_viol = float(np.max(diff))
            if max_viol > max_overall_violation:
                max_overall_violation = max_viol
                
            results.append({
                'name': sc['name'],
                'desc': sc['desc'],
                'v': sc['v'],
                'eta0': sc['eta0'],
                'norm_eta0': norm_eta0,
                'final_time_s': float(sol.t[-1]),
                'final_norm_eta': float(norm_eta_t[-1]),
                'final_bound': float(bound_t[-1]),
                'max_violation': max_viol,
                'strictly_bounded': bool(max_viol <= 1e-12)
            })
            
        all_ode_passed = bool(max_overall_violation <= 1e-12)
        return {
            'all_ode_passed': all_ode_passed,
            'max_overall_violation': max_overall_violation,
            'scenarios': results
        }

    def execute_full_audit(self):
        """执行完整冻结请求抗饱和子系统审计并导出 JSON"""
        # 1. 对称配置检查
        sym_check = {
            'Kf_L': float(self.c.Kf_L),
            'Kf_R': float(self.c.Kf_R),
            'Imax_L': float(self.c.Imax_L),
            'Imax_R': float(self.c.Imax_R),
            'a_L': self.a_L,
            'a_R': self.a_R,
            'is_symmetric': self.is_symmetric,
            'F_max_N': self.F_max,
            'T_max_Nm': self.T_max
        }
        
        # 2. 归一化分配器一致性验证
        alloc_res = self.verify_allocator_equivalence()
        
        # 3. 解析平衡点满足原方程核查
        eq_res = self.verify_analytic_equilibrium()
        
        # 4. 收敛不等式核查
        ineq_res = self.verify_convergence_inequalities(n_samples=50000)
        
        # 5. ODE 仿真检验
        ode_res = self.run_ode_convergence_regression()
        
        # 6. 结论边界与判定
        all_passed = bool(
            sym_check['is_symmetric'] and
            alloc_res['is_consistent'] and
            eq_res['all_passed'] and
            ineq_res['passed'] and
            ode_res['all_ode_passed']
        )
        
        conclusion = {
            'certification_status': 'passed_frozen_request_fast_subsystem' if all_passed else 'failed',
            'formal_statement': (
                '在当前对称执行能力配置、请求量冻结的条件下，'
                '抗饱和子系统相对于其请求相关平衡点具有一致指数收敛性质。'
            ),
            'strict_boundaries': (
                '严禁宣称完整闭环稳定已证明，亦暂不宣布奇异摄动条件已满足。'
                '下一步须研究请求时变 v(t) 产生动平衡移动对降阶系统的持续扰动与持续饱和损失。'
            )
        }
        
        full_results = {
            'audit_timestamp': '2026-10-20T08:05:00Z',
            'symmetric_capability_precondition': sym_check,
            'allocator_equivalence_verification': alloc_res,
            'analytic_equilibrium_verification': eq_res,
            'convergence_inequality_verification': ineq_res,
            'ode_trajectory_regression': ode_res,
            'conclusion_and_boundaries': conclusion
        }
        
        # 保存 JSON
        json_path = os.path.join(cur_dir, "frozen_request_fast_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
            
        return full_results


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    analyzer = FrozenRequestFastSubsystemAnalyzer()
    res = analyzer.execute_full_audit()
    
    print("=" * 80)
    print("冻结请求下的抗饱和快子系统性质分析完成")
    print("=" * 80)
    
    sym = res['symmetric_capability_precondition']
    print(f"1. 对称执行能力前置检查: {'PASS' if sym['is_symmetric'] else 'FAIL'}")
    print(f"   单侧能力 a_L = {sym['a_L']:.4f} N, a_R = {sym['a_R']:.4f} N | F_max = {sym['F_max_N']:.4f} N, T_max = {sym['T_max_Nm']:.4f} Nm")
    
    alloc = res['allocator_equivalence_verification']
    print(f"\n2. 归一化分配器与原物理分配器一致性核验: {'PASS' if alloc['is_consistent'] else 'FAIL'}")
    print(f"   测试点数量: {alloc['test_points_count']} | 平动残差: {alloc['max_abs_err_F']:.2e} | 力矩残差: {alloc['max_abs_err_T']:.2e}")
    
    eq = res['analytic_equilibrium_verification']
    print(f"\n3. 解析求平衡点满足原方程检验: {'PASS' if eq['all_passed'] else 'FAIL'}")
    print(f"   最大方程代数残差: {eq['max_ode_residual']:.2e}")
    
    ineq = res['convergence_inequality_verification']
    print(f"\n4. 平衡点偏差收敛不等式验证 (50,000 组工况): {'PASS' if ineq['passed'] else 'FAIL'}")
    print(f"   力矩通道不等式最小余量:   {ineq['min_slack_T_channel']:.4e} (>= -1e-11)")
    print(f"   平动通道不等式最小余量:   {ineq['min_slack_F_channel']:.4e} (>= -1e-11)")
    print(f"   W_f 不等式 (含交叉项) 余量: {ineq['min_slack_Wf_bound_1']:.4e} (>= -1e-11)")
    print(f"   W_f 指数衰减不等式最小余量: {ineq['min_slack_Wf_exponential']:.4e} (>= -1e-11)")
    
    ode = res['ode_trajectory_regression']
    print(f"\n5. ODE 真实收敛轨迹检验 (5 大典型场景): {'PASS' if ode['all_ode_passed'] else 'FAIL'}")
    print(f"   最大违例程度: {ode['max_overall_violation']:.2e}")
    for sc in ode['scenarios']:
        print(f"   - {sc['name']:22s} ({sc['desc']}): ||eta(0)||={sc['norm_eta0']:.2f} -> ||eta(0.5s)||={sc['final_norm_eta']:.2e} (界限: {sc['final_bound']:.2e})")
        
    conc = res['conclusion_and_boundaries']
    print(f"\n6. 结论与边界声明:")
    print(f"   认证状态: {conc['certification_status']}")
    print(f"   正式结论: {conc['formal_statement']}")
    print(f"   严格边界: {conc['strict_boundaries']}")
    print(f"\n完整结构化结果已导出至: frozen_request_fast_results.json")
