"""
config_e5_final.py
==================
统一配置模块：严格分离控制器参数（描述算法本身）与被控对象参数（描述仿真假设环境）。
参数完全对齐嵌入式固件 experiment5_config.h 与 controller_e5_blnn_lc.c。
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class ControllerConfig:
    """
    控制器算法参数：与嵌入式固件 controller_e5_blnn_lc.c 严格逐项对齐。
    """
    # 采样与几何基准
    Ts: float = 0.001          # 控制步长 1 ms (1 kHz)
    Le: float = 0.600          # 龙门有效跨距 600 mm
    Kf_L: float = 0.0061979167 # 左电机推力转换常数 (N/count)
    Kf_R: float = 0.0061979167 # 右电机推力转换常数 (N/count)
    
    # 执行器限幅与保护
    Imax_L: float = 800.0      # 左电机指令电流限幅 (counts)
    Imax_R: float = 800.0      # 右电机指令电流限幅 (counts)
    slew_rate_max: float = 75.0# 补偿指令单步变化率上限 (counts/step, 即 75 counts/ms)
    
    # C2b 滑模控制器名义增益 (对齐 experiment5_config.h)
    Gamma_y: float = 30.0      # 平动滑模面比例系数
    Gamma_alpha: float = 20.0  # 旋转滑模面比例系数
    K1_y: float = 45.0         # 平动线性趋近增益
    K1_alpha: float = 30.0     # 旋转线性趋近增益
    K2_y: float = 12.0         # 平动开关鲁棒增益
    K2_alpha: float = 6.0      # 旋转开关鲁棒增益
    phi_y: float = 0.030       # 平动边界层厚度
    phi_alpha: float = 0.030   # 旋转边界层厚度
    eps_v: float = 0.01        # 摩擦双曲正切平滑系数
    eps_alpha: float = 0.01    # 旋转摩擦双曲正切平滑系数
    
    # 控制器内部名义动力学模型参数 (Hat)
    M_hat_y: float = 12.445    # 平动名义质量 (kg)
    M_hat_alpha: float = 0.37335 # 偏转名义转动惯量 (kg*m^2)
    B_hat_y: float = 63.0      # 平动名义黏性阻尼 (N*s/m)
    B_hat_alpha: float = 22.5  # 偏转名义旋转阻尼 (N*m*s/rad)
    K_hat_alpha: float = 2000.0# 偏转名义扭转刚度 (N*m/rad)
    A_hat_y: float = 14.4      # 平动名义库仑摩擦力 (N)
    
    # 动态抗饱和参数
    K_aw: float = 20.0         # 抗饱和跟踪增益 (1/s)
    lambda_aw: float = 20.0    # 抗饱和状态指数衰减率 (1/s)
    
    # 宽度学习网络 (BLNN) 参数
    enable_blnn: bool = True   # 是否启用神经网络补偿
    nn_eps_y: float = 25.0     # 平动权值学习率
    nn_eps_alpha: float = 18.0 # 旋转权值学习率
    nn_eta_y: float = 0.01     # 平动泄漏阻尼系数
    nn_eta_alpha: float = 0.01 # 旋转泄漏阻尼系数
    
    # 输入归一化比例尺度 (5维输入: [qddot, qdot, q, e, edot])
    scale_y: np.ndarray = field(default_factory=lambda: np.array([2.0, 0.7, 1.0, 0.08, 0.7], dtype=np.float32))
    scale_alpha: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.2, 0.02, 0.01, 0.2], dtype=np.float32))
    
    # 4 节点中心矩阵 (4 x 5)
    centers: np.ndarray = field(default_factory=lambda: np.array([
        [-1.0, -0.5,  0.0,  0.5,  1.0],
        [-0.5,  0.0,  0.5,  1.0, -1.0],
        [ 0.0,  0.5,  1.0, -1.0, -0.5],
        [ 0.5,  1.0, -1.0, -0.5,  0.0]
    ], dtype=np.float32))
    
    # 学习协调 (LC) 参数
    enable_lc: bool = True     # 是否启用动态学习协调
    gamma_lc: float = 100.0    # LC 灵敏度增益系数


@dataclass
class PlantConfig:
    """
    被控对象物理假设参数：描述仿真中的受控物理环境（与控制器参数明确解耦）。
    """
    # 物理质量与几何
    m_G: float = 3.0           # 统一物理对象基准质量：3.0 kg 实测机械台车秤重质量
    J_alpha: float = 0.37335   # 沿用的假设惯量，未实测；不由4kg直接确定
    Le: float = 0.600          # 配置跨距 (m)，不是本次测量结果
    
    # 导轨物理摩擦模型 (允许设置非对称与外部阻力)
    b_L: float = 63.0          # 左侧导轨黏性摩擦系数 (N*s/m)
    b_R: float = 63.0          # 右侧导轨黏性摩擦系数 (N*s/m)
    f_c_L: float = 14.4        # 左侧导轨库仑摩擦力 (N)
    f_c_R: float = 14.4        # 右侧导轨库仑摩擦力 (N)
    
    # 机构刚度与阻尼
    K_alpha: float = 2000.0    # 横梁扭转刚度 (N*m/rad)
    B_alpha: float = 22.5      # 横梁扭转阻尼 (N*m*s/rad)
    
    # 附加偏载配置 (默认无偏载空载)
    delta_m: float = 0.0       # 附加载荷质量 (kg)
    d_load: float = 0.0        # 附加偏心距 (m)
