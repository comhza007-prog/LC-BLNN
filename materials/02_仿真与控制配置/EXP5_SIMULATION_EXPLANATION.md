# 仿真算例体系与证据链说明文档 (Simulation Benchmarks & Evidence Chain) - 终版规范

本文件作为论文稿件第 5 章“仿真分析”部分的唯一技术说明底册，对归档的仿真算例进行明确分工与定位，消除参数不一致与解释错位。

---

## 一、统一基准算例与历史探索算例的明确划分

在归档目录 `02_仿真与控制配置` 中，明确区分以下三类算例的证据职责：

| 算例编号 | 算例名称 | 运动指令与工况 | 饱和激活状态 | 在论文中的证据链定位 | 生成脚本与产物 |
|---|---|---|---|---|---|
| **算例 A (核心)** | **50 mm 深饱和压力测试** (Deep Saturation Stress Test) | 50 mm 幅度、0.5 Hz 往复机动，单侧电机限幅 800 counts (4.96 N) | **深度触发物理推力饱和** (饱和率 87.0% ~ 99.6%, 动态 LC 最低 $\rho_y = 0.0072$) | **深饱和压力测试：权值大幅受控，但误差付出代价**。<br>展示在严重受限工况下，普通 BLNN 因持续饱和积分导致权值攀升至 76.49；动态 LC-BLNN 通过动态降低 $\rho_y(t)$ 将权值峰值显著抑制至 5.84 (-92.4%)，但自适应增益的削减使得跟踪 RMSE 从 23.82 mm 略增至 25.04 mm，同步 RMS 从 0.00098 mm 略增至 0.00124 mm。 | 脚本：[`run_unified_saturation_benchmark.py`](run_unified_saturation_benchmark.py)<br>数据：[`EXP5_UNIFIED_SATURATION_BENCHMARK.csv`](EXP5_UNIFIED_SATURATION_BENCHMARK.csv)<br>插图：[`EXP5_UNIFIED_SATURATION_COMPARISON.png`](EXP5_UNIFIED_SATURATION_COMPARISON.png) |
| **算例 B (对照)** | **10 mm 受限工况对照** (10 mm Constrained Operation Benchmark) | 10 mm 平缓多项式轨迹 (20 s)，单侧电机限幅 800 counts (4.96 N) | **部分触发推力饱和** (饱和率约 22.8% ~ 23.8%, 动态 LC 最低 $\rho_y = 0.1893$) | **受限工况下的综合性能对照**。<br>在轻中度饱和工况下，动态 LC 明显介入（$\min \rho_y = 0.1893$），将权值峰值从 3.065 抑制到 1.043 (-66.0%)，且跟踪 RMSE 从 0.602 mm 略改善至 0.573 mm，同步 RMS 从 0.00069 mm 保持在 0.00068 mm，实现了权值抑制与轨迹精度的兼顾。 | 脚本：[`run_unified_saturation_benchmark.py`](run_unified_saturation_benchmark.py)<br>数据：[`EXP5_UNIFIED_SATURATION_BENCHMARK.csv`](EXP5_UNIFIED_SATURATION_BENCHMARK.csv) |
| **历史算例** | **固定调制初步消融探索** (Historical Fixed-Modulation Exploration) | 历史 10 mm 轨迹离线消融记录 | 历史探索工况 | **仅保留作为算法设计初期的离线消融探索记录**，说明早期测试参数设定，不作为论文正文论证“受限时动态协调更优”的正式证据。 | 脚本：[`run_offline_fixed_modulation_benchmark.py`](run_offline_fixed_modulation_benchmark.py)<br>数据：[`EXP5_OFFLINE_FIXED_MODULATION_BENCHMARK.csv`](EXP5_OFFLINE_FIXED_MODULATION_BENCHMARK.csv)<br>插图：[`EXP5_OFFLINE_FIXED_MODULATION_COMPARISON.png`](EXP5_OFFLINE_FIXED_MODULATION_COMPARISON.png) |

---

## 二、仿真与理论连续模型的参数对应与扰动说明

算例 A 与算例 B 均通过统一配置脚本 [`config_e5_final.py`](config_e5_final.py) 调用执行：
1. **被控对象基准质量**：统一取实测机械秤重质量 **$m_G = 3.0\,\text{kg}$**（与连续理论分析模型 `PlantPhysicalParameters.m_G_kg = 3.0` 严格一致）；
2. **控制器内部名义模型系数**：统一取固件历史沿用的模型设定值 **$\hat{M}_y = 12.445\,\text{kg}$**（追溯自前期第 3 次实验配置 `experiment3_config.h` 中的设定值 `#define EXP3_C2_M_Y 12.445f`，存在 4.15 倍已知模型失配）；
3. **两轴刚度与阻尼及故意扰动说明**：
   - 扭转刚度与阻尼：$K_\alpha = 2000.0\,\text{N}\cdot\text{m/rad},\;B_\alpha = 22.5\,\text{N}\cdot\text{m}\cdot\text{s/rad}$；
   - **摩擦阻尼不对称扰动**：仿真模型中故意引入了双侧摩擦阻尼不对称设计扰动——左轨 $b_L = 63.0\,\text{N}\cdot\text{s/m}$，右轨 $b_R = 75.6\,\text{N}\cdot\text{s/m}$（相比名义值增加了 20% 的摩擦阻尼不对称），以此考核控制器在非对称摩擦阻尼摄动下的解耦与同步性能。因此仿真环境包含了故意的物理摩擦不对称设计摄动，不可写为“与连续理论标称模型参数 100% 绝对同一”；
4. **控制与限幅边界**：$I_{\max} = 800\,\text{counts},\;F_{\max} = 9.9167\,\text{N},\;T_{\max} = 2.9750\,\text{N}\cdot\text{m}$，单侧推力变化率限幅 $75\,\text{counts/ms}$。

---

## 三、受限工况下仿真性能指标汇总 (引自 CSV 实测值)

以下指标均直接源自 [`EXP5_UNIFIED_SATURATION_BENCHMARK.csv`](EXP5_UNIFIED_SATURATION_BENCHMARK.csv)，严禁手工篡改：

### 3.1 算例 A：50 mm 深饱和压力测试指标对比

| 控制方法 | 跟踪 RMSE (mm) | 跟踪峰值 (mm) | 同步 RMS (mm) | 同步峰值 (mm) | 权值峰值 $\|W_y\|$ | 最小协调因子 $\min(\rho_y)$ | 推力饱和时间占比 | 机制效能定性 |
|---|---|---|---|---|---|---|---|---|
| **普通 BLNN** ($\rho \equiv 1.0$) | **23.8216** | 46.6578 | **0.000977** | **0.001517** | 76.4925 | 1.0000 | 99.62% | 持续饱和深积分，权值严重累积 |
| **固定折减 BLNN** ($\rho \equiv 0.5$) | 23.8203 | **46.6574** | 0.001229 | 0.001578 | 38.2435 | 0.5000 | 99.62% | 线性减缓权值累积，但未完全阻断 |
| **动态 LC-BLNN (本文方法)** | 25.0363 | 46.6578 | 0.001243 | 0.002130 | **5.8356** | **0.00718** | **86.97%** | **权值大幅抑制 (-92.4%)，但跟踪误差付出代价 (+5.1%)** |

> **定性结论**：本算例中，LC降低了观测权值峰值，同时跟踪与同步误差有所增加。

### 3.2 算例 B：10 mm 受限工况对照指标对比

| 控制方法 | 跟踪 RMSE (mm) | 跟踪峰值 (mm) | 同步 RMS (mm) | 同步峰值 (mm) | 权值峰值 $\|W_y\|$ | 最小协调因子 $\min(\rho_y)$ | 推力饱和时间占比 | 机制效能定性 |
|---|---|---|---|---|---|---|---|---|
| **普通 BLNN** ($\rho \equiv 1.0$) | 0.6024 | 1.7155 | 0.000689 | **0.001538** | 3.0650 | 1.0000 | 23.84% | 中轻度饱和积分，权值温和累积 |
| **固定折减 BLNN** ($\rho \equiv 0.5$) | 0.5912 | 1.7065 | 0.000770 | 0.001663 | 1.5510 | 0.5000 | 22.98% | 固定折减自适应速率 |
| **动态 LC-BLNN (本文方法)** | **0.5728** | **1.6547** | **0.000677** | 0.001540 | **1.0426** | **0.1893** | **22.76%** | **兼顾权值抑制 (-66.0%) 与跟踪精度改善 (-4.9%)** |

> **定性结论**：在算例 B 中，推力饱和介入适度（饱和率约 22.8%），动态协调因子最低降至 0.1893。动态 LC-BLNN 不仅成功将权重峰值从 3.065 压制到 1.043，而且由于在饱和瞬间有效抑制了虚假过冲积分，跟踪 RMSE 反而从 0.602 mm 改善为 0.573 mm，同步 RMS 保持在 0.00068 mm 极高水准。
