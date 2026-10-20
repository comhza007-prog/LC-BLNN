"""
controller_e5_final_lc.py
=========================
最终对齐控制器：与嵌入式 C 源码 controller_e5_blnn_lc.c 严格逐行对应。
包含：
1. 模态与电机空间解析正逆映射；
2. 推力空间力矩优先分配器 (thrust_allocator_sync_c)；
3. 同状态基准分配解算与抗饱和前向递推；
4. 4节点高斯-三角增强特征宽度学习网络；
5. 电机空间 75 counts/ms 完整摆率限幅器；
6. 模态总补偿实现差额解算与非线性 LC 调制律；
7. 仅调制误差驱动项、保留泄漏项的权值自适应律；
8. 动态抗饱和状态变量积分。
"""

import numpy as np
from config_e5_final import ControllerConfig

class E5FinalController:
    def __init__(self, cfg: ControllerConfig = None, mode: str = "LC", fixed_c: tuple = (1.0, 1.0)):
        """
        mode:
          - 'LC': 动态学习协调 (rho dynamically driven by delta_v_total)
          - 'FIXED': 固定调制 (rho fixed to fixed_c=(c_y, c_alpha))
          - 'VANILLA': 普通 BLNN (rho == (1.0, 1.0))
        """
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self.mode = mode.upper()
        self.fixed_c = fixed_c
        self.reset()
        
    def reset(self):
        """重置控制器内部可变状态"""
        self.z_aw = np.zeros(2, dtype=np.float64)       # 抗饱和状态 [z_L, z_R] (counts)
        self.v_prev = np.zeros(2, dtype=np.float64)     # 前一步电机空间神经补偿 [vL_prev, vR_prev]
        self.W_y = np.zeros((4, 3), dtype=np.float64)   # 平动权值矩阵 (4节点 x 3特征)
        self.W_alpha = np.zeros((4, 3), dtype=np.float64)# 旋转权值矩阵 (4节点 x 3特征)
        self.step_count = 0
        
    def map_motor_to_modal(self, iL: float, iR: float):
        """电机指令 (counts) -> 广义模态力 [FG (N), Talpha (N*m)]"""
        FL = self.cfg.Kf_L * iL
        FR = -self.cfg.Kf_R * iR
        FG = FL + FR
        Talpha = 0.5 * self.cfg.Le * (FR - FL)
        return FG, Talpha

    def map_modal_to_motor(self, FG: float, Talpha: float):
        """广义模态力 -> 理想电机电流 (counts)"""
        FL = 0.5 * FG - (Talpha / self.cfg.Le)
        FR = 0.5 * FG + (Talpha / self.cfg.Le)
        iL = FL / self.cfg.Kf_L
        iR = -FR / self.cfg.Kf_R
        return iL, iR

    def thrust_allocator_sync(self, FG_des: float, Talpha_des: float):
        """
        推力空间力矩优先分配器：与 C 函数 thrust_allocator_sync_c 严格一致。
        优先保障偏转纠偏力矩，截断平动合力；严格限制在 [-Imax, Imax]。
        """
        FL_max = self.cfg.Kf_L * self.cfg.Imax_L
        FR_max = self.cfg.Kf_R * self.cfg.Imax_R
        DeltaF_des = (2.0 * Talpha_des) / self.cfg.Le
        DeltaF_max = FR_max + FL_max
        
        # 优先级 1: 纠偏推力差额限幅
        DeltaF_alloc = np.clip(DeltaF_des, -DeltaF_max, DeltaF_max)
        
        # 优先级 2: 平动推力可行域
        FL_lower = max(-FL_max, -FR_max - DeltaF_alloc)
        FL_upper = min( FL_max,  FR_max - DeltaF_alloc)
        
        FL_target = 0.5 * FG_des - 0.5 * DeltaF_alloc
        FL_alloc = np.clip(FL_target, FL_lower, FL_upper)
        FR_alloc = FL_alloc + DeltaF_alloc
        
        iL_act = np.clip(FL_alloc / self.cfg.Kf_L, -self.cfg.Imax_L, self.cfg.Imax_L)
        iR_act = np.clip(-FR_alloc / self.cfg.Kf_R, -self.cfg.Imax_R, self.cfg.Imax_R)
        
        FL_real = self.cfg.Kf_L * iL_act
        FR_real = -self.cfg.Kf_R * iR_act
        FG_alloc = FL_real + FR_real
        Talpha_alloc = 0.5 * self.cfg.Le * (FR_real - FL_real)
        
        is_sat_thrust = abs(FG_alloc - FG_des) > 1e-5
        is_sat_torque = abs(Talpha_alloc - Talpha_des) > 1e-5
        return iL_act, iR_act, is_sat_thrust, is_sat_torque

    def step(self, q_meas, qdot_meas, qddot_est, qd, qdot_d, qddot_d):
        """
        单步控制求解：与 e5_controller_step() 逐行一致。
        q_meas: [yG, alpha]
        qdot_meas: [yG_dot, alpha_dot]
        qddot_est: [yG_ddot, alpha_ddot]
        """
        cfg = self.cfg
        self.step_count += 1
        
        # 1. 误差与广义滑模面
        e_y = q_meas[0] - qd[0]
        e_alpha = q_meas[1] - qd[1]
        edot_y = qdot_meas[0] - qdot_d[0]
        edot_alpha = qdot_meas[1] - qdot_d[1]
        
        sigma_y = edot_y + cfg.Gamma_y * e_y
        sigma_alpha = edot_alpha + cfg.Gamma_alpha * e_alpha
        
        # 2. C2b 基线等效力与鲁棒反馈项
        Sf_y = np.tanh(qdot_meas[0] / cfg.eps_v)
        Sf_alpha = np.tanh(qdot_meas[1] / cfg.eps_alpha)
        
        va_y = cfg.B_hat_y * qdot_meas[0] + cfg.A_hat_y * Sf_y + cfg.M_hat_y * qddot_d[0]
        va_alpha = cfg.B_hat_alpha * qdot_meas[1] + cfg.K_hat_alpha * q_meas[1] + cfg.M_hat_alpha * qddot_d[1]
        
        vr_y = -cfg.K1_y * sigma_y - cfg.K2_y * np.tanh(sigma_y / cfg.phi_y) - cfg.M_hat_y * cfg.Gamma_y * edot_y
        vr_alpha = -cfg.K1_alpha * sigma_alpha - cfg.K2_alpha * np.tanh(sigma_alpha / cfg.phi_alpha) - cfg.M_hat_alpha * cfg.Gamma_alpha * edot_alpha
        
        v_base_y = va_y + vr_y
        v_base_alpha = va_alpha + vr_alpha
        
        # 3. 基准同状态抗饱和占用与理想分配
        v_aw_y, v_aw_alpha = self.map_motor_to_modal(self.z_aw[0], self.z_aw[1])
        v_base_tot_y = v_base_y + v_aw_y
        v_base_tot_alpha = v_base_alpha + v_aw_alpha
        
        i_base_act_L, i_base_act_R, _, _ = self.thrust_allocator_sync(v_base_tot_y, v_base_tot_alpha)
        
        # 4. BLNN 4节点高斯+三角增强特征提取
        v_nn_raw_modal = np.array([0.0, 0.0], dtype=np.float64)
        S_y = np.zeros((4, 3), dtype=np.float64)
        S_a = np.zeros((4, 3), dtype=np.float64)
        
        if cfg.enable_blnn:
            # 输入归一化与 [-2, 2] 截断
            chi_in_y = np.clip([
                qddot_est[0] / cfg.scale_y[0],
                qdot_meas[0] / cfg.scale_y[1],
                q_meas[0]    / cfg.scale_y[2],
                e_y          / cfg.scale_y[3],
                edot_y       / cfg.scale_y[4]
            ], -2.0, 2.0)
            
            chi_in_a = np.clip([
                qddot_est[1] / cfg.scale_alpha[0],
                qdot_meas[1] / cfg.scale_alpha[1],
                q_meas[1]    / cfg.scale_alpha[2],
                e_alpha      / cfg.scale_alpha[3],
                edot_alpha   / cfg.scale_alpha[4]
            ], -2.0, 2.0)
            
            for j in range(4):
                d2_y = np.sum((cfg.centers[j] - chi_in_y) ** 2)
                h_y = np.exp(-d2_y)
                S_y[j, 0] = h_y
                S_y[j, 1] = np.cos(h_y)
                S_y[j, 2] = np.sin(h_y)
                
                d2_a = np.sum((cfg.centers[j] - chi_in_a) ** 2)
                h_a = np.exp(-d2_a)
                S_a[j, 0] = h_a
                S_a[j, 1] = np.cos(h_a)
                S_a[j, 2] = np.sin(h_a)
                
                v_nn_raw_modal[0] -= np.sum(self.W_y[j] * S_y[j])
                v_nn_raw_modal[1] -= np.sum(self.W_alpha[j] * S_a[j])
                
        # 模态网络请求映射到电机空间
        v_nn_raw_motor_L, v_nn_raw_motor_R = self.map_modal_to_motor(v_nn_raw_modal[0], v_nn_raw_modal[1])
        
        # 5. 电机空间 75 counts/ms 摆率限幅器 (严格对齐 C 代码)
        if cfg.slew_rate_max > 0.0:
            dv_L = v_nn_raw_motor_L - self.v_prev[0]
            dv_R = v_nn_raw_motor_R - self.v_prev[1]
            v_slew_L = self.v_prev[0] + np.clip(dv_L, -cfg.slew_rate_max, cfg.slew_rate_max)
            v_slew_R = self.v_prev[1] + np.clip(dv_R, -cfg.slew_rate_max, cfg.slew_rate_max)
        else:
            v_slew_L = v_nn_raw_motor_L
            v_slew_R = v_nn_raw_motor_R
            
        self.v_prev[0] = v_slew_L
        self.v_prev[1] = v_slew_R
        
        # 6. 总控制力合成与分配
        v_applied_modal_y, v_applied_modal_a = self.map_motor_to_modal(v_slew_L, v_slew_R)
        FG_cmd = v_base_tot_y + v_applied_modal_y
        Talpha_cmd = v_base_tot_alpha + v_applied_modal_a
        
        i_ideal_L, i_ideal_R = self.map_modal_to_motor(FG_cmd, Talpha_cmd)
        iL_act, iR_act, is_sat_t, is_sat_q = self.thrust_allocator_sync(FG_cmd, Talpha_cmd)
        
        # 7. 增量与总补偿实现差额解算
        v_applied_motor_L = iL_act - i_base_act_L
        v_applied_motor_R = iR_act - i_base_act_R
        
        delta_v_tot_L = v_applied_motor_L - v_nn_raw_motor_L
        delta_v_tot_R = v_applied_motor_R - v_nn_raw_motor_R
        
        delta_FG_tot, delta_Ta_tot = self.map_motor_to_modal(delta_v_tot_L, delta_v_tot_R)
        
        # 8. 学习协调律 (LC) 模式分支
        if not cfg.enable_blnn:
            rho_y = 1.0
            rho_alpha = 1.0
        elif self.mode == "VANILLA":
            rho_y = 1.0
            rho_alpha = 1.0
        elif self.mode == "FIXED":
            rho_y = float(self.fixed_c[0])
            rho_alpha = float(self.fixed_c[1])
        elif self.mode == "LC":
            max_Imax = max(cfg.Imax_L, cfg.Imax_R)
            chi_y = (abs(delta_FG_tot) / (cfg.Kf_L + cfg.Kf_R)) / max_Imax
            rho_y = 1.0 / (1.0 + cfg.gamma_lc * (chi_y ** 2))
            
            chi_a = (abs(delta_Ta_tot) / (0.5 * cfg.Le * (cfg.Kf_L + cfg.Kf_R))) / max_Imax
            rho_alpha = 1.0 / (1.0 + cfg.gamma_lc * (chi_a ** 2))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        # 权值自适应更新 (仅调制误差项，保留泄漏项)
        if cfg.enable_blnn:
            for j in range(4):
                for m in range(3):
                    W_dot_y = cfg.nn_eps_y * (rho_y * sigma_y * S_y[j, m] - cfg.nn_eta_y * self.W_y[j, m])
                    W_dot_a = cfg.nn_eps_alpha * (rho_alpha * sigma_alpha * S_a[j, m] - cfg.nn_eta_alpha * self.W_alpha[j, m])
                    self.W_y[j, m] += cfg.Ts * W_dot_y
                    self.W_alpha[j, m] += cfg.Ts * W_dot_a
                    
        # 9. 动态抗饱和状态更新 (counts 空间)
        Delta_i_alloc_L = iL_act - i_ideal_L
        Delta_i_alloc_R = iR_act - i_ideal_R
        
        self.z_aw[0] += cfg.Ts * (cfg.K_aw * Delta_i_alloc_L - cfg.lambda_aw * self.z_aw[0])
        self.z_aw[1] += cfg.Ts * (cfg.K_aw * Delta_i_alloc_R - cfg.lambda_aw * self.z_aw[1])
        
        # 10. 诊断信息封装
        info = {
            "iL_cmd": iL_act,
            "iR_cmd": iR_act,
            "e_y": e_y,
            "e_sync": cfg.Le * q_meas[1],
            "sigma_y": sigma_y,
            "sigma_alpha": sigma_alpha,
            "F_nn": v_nn_raw_modal[0],
            "T_nn": v_nn_raw_modal[1],
            "v_slew_L": v_slew_L,
            "v_slew_R": v_slew_R,
            "v_nn_raw_motor_L": v_nn_raw_motor_L,
            "v_nn_raw_motor_R": v_nn_raw_motor_R,
            "delta_v_tot_L": delta_v_tot_L,
            "delta_v_tot_R": delta_v_tot_R,
            "rho_y": rho_y,
            "rho_alpha": rho_alpha,
            "norm_Wy": float(np.linalg.norm(self.W_y)),
            "norm_Wa": float(np.linalg.norm(self.W_alpha)),
            "F_cmd": FG_cmd,
            "T_cmd": Talpha_cmd,
            "is_sat_thrust": is_sat_t,
            "is_sat_torque": is_sat_q,
            "sat_alloc": is_sat_t or is_sat_q,
            "z_aw_L": self.z_aw[0],
            "z_aw_R": self.z_aw[1]
        }
        return iL_act, iR_act, info
