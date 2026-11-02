# 仿真：实测辨识对象 + 固件测量链 + 多工况对比

本目录替换了旧版 10–300 mm 仿真。旧版仿真对象每侧库仑摩擦 14.4 N、黏性阻尼 63/75.6 N·s/m（两侧合计 28.8 N，超过两侧最大推力约 9.9 N），且直接把真实速度和参考加速度送入控制器。小车在 20 mm 以上行程基本跟不上轨迹（300 mm 跟踪 RMSE 约 143 mm），与实物（约 1–4 mm）不在同一量级，因此旧版结果已删除。

仿真的定位是**机理分析与消融**：解释 LC 为什么有效、与其他饱和处理策略有何不同。跟踪与同步的定量收益以实物实验为准。

## 1. 模型

`code/plant_sim.py`

- **对象**：平动 + 偏转两自由度；等效平动惯量、两侧库仑摩擦（含 Stribeck 静摩擦）、黏性阻尼、左右不对称、横梁扭转刚度与阻尼；RK4，每个 1 ms 控制周期内 4 个子步。
- **测量链**（与 `motor_task.c` 一致）：两侧编码器位置量化；两侧电机 `speed_rpm` 整数反馈（含高斯噪声）；模态速度差分后按 0.76/0.24 一阶滤波得到加速度估计。
- **控制器**：`code/controller_e5_final_lc.py`，与固件 `controller_e5_blnn_lc.c` 逐段对应，限幅、增益、学习率、γ_lc=100 与 `experiment5_config.h` 一致。

## 2. 参数辨识（只用基础控制器 M1）

`code/calibrate_plant.py`，结果见 `results/01_calibration/calibration_search.csv` 和 `results/plant_identified.json`。

- 辨识数据：实物 M1_BASE（不含网络）在 10、50、300 mm 的中位数，指标为跟踪 RMSE、跟踪峰值、同步 RMS、同步峰值、饱和占比、终点残差、超调。
- 方法：120 组全局随机搜索，代价为各指标对数比的平方和。
- M2、M3 的实测数据**不参与辨识**，只用于验证。

| 参数 | 辨识值 |
|---|---|
| 等效平动惯量 m_eff | 9.80 kg（可辨识性弱，3.5–14 kg 代价相近，不作测量值使用） |
| 两侧库仑摩擦合计 fc_tot | 1.15 N |
| 静摩擦/库仑摩擦比 | 2.13 |
| 两侧黏性阻尼合计 b_tot | 9.74 N·s/m |
| 右/左不对称比 | 1.19 |
| 扭转刚度、阻尼缩放 | 0.585 |
| 速度反馈噪声 | 23.0 rpm |

## 3. 工况与对比方法

每个工况：8 个行程（10–300 mm）× 10 个噪声种子；同一种子下各方法使用相同噪声序列（配对设计）。

| 工况 | 设置 | 目的 | 结果文件 |
|---|---|---|---|
| 标称 | 辨识参数，限幅 800 counts | 与实物对照；对比各方法 | `results/02_nominal_800/` |
| 强约束 | 限幅 600 counts | 执行余量更小时的劣化与抑制 | `results/03_strong_600/` |
| 超限 | 限幅 400 counts | 极端情况（长行程超出平台能力） | `results/04_overlimit_400/` |
| 鲁棒性 | 摩擦×2；左侧外部阻力 15%/30% 单侧最大推力 | 补充工况 | `results/05_robustness/` |

| 方法 | 说明 |
|---|---|
| M1_BASE | 基础控制，不含网络 |
| M2_BLNN | 普通 BLNN（ρ_y = ρ_α ≡ 1） |
| M3_LCBLNN | 本文方法 |
| Fixed_matched | 固定调制，ρ 取同一种子下 LC 运行的平均 ρ_y、ρ_α（平均学习强度相同） |
| Global_scalar | 全局饱和度调制：按两侧电流截断量的最大值归一化后算单一标量 ρ，两通道共用（γ_lc 相同） |
| Freeze_sat | 饱和冻结：分配器饱和时整个权值更新（含泄漏项）置零 |

各工况全部均值见 `results/all_conditions_means.csv`。

## 4. 主要结果

**(1) 协调因子与实物一致（标称工况，图 `results/figures/fig_rho_real_vs_sim.png`）**

| 行程 (mm) | 10 | 20 | 30 | 40 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| ρ_y 均值 实测 | 0.934 | 0.877 | 0.861 | 0.818 | 0.794 | 0.719 | 0.661 | 0.605 |
| ρ_y 均值 仿真 | 0.944 | 0.865 | 0.811 | 0.778 | 0.759 | 0.711 | 0.664 | 0.626 |

实测 ρ_α 均值 0.975–0.987，仿真约 0.999。200 mm 下三组方法的跟踪 RMSE 中位数与权值峰值和实测相差约 10% 以内。

**(2) 强约束下普通 BLNN 出现劣化，LC 抑制（600 counts，300 mm）**

| 方法 | 跟踪 RMSE (mm) | 平动权值峰值 均值 / 最大 |
|---|---|---|
| M1_BASE | 2.28 | — |
| M2_BLNN | 3.11 | 10.49 / 22.24 |
| Fixed_matched | 2.86 | 6.63 / 13.94 |
| Global_scalar | 2.17 | 2.09 / 2.31 |
| Freeze_sat | 2.20 | 2.02 / 2.21 |
| M3_LCBLNN | 2.37 | 1.71 / 1.87 |

LC 相对 BLNN：跟踪 −23.9%、权值 −83.7%（10/10 种子）。平均学习强度相同的固定调制未能阻止劣化。

**(3) 按模态协调保留偏转学习**

各工况中 LC 的 ρ_α 均值 ≥ 0.97。800 与 600 counts 工况中，全局调制与饱和冻结的 ρ_α 被压低到 0.47–0.83，偏转权值峰值比 LC 低约 25–55%（400 counts 下更低）。

**(4) 与全局调制、饱和冻结的比较（600 counts，10–200 mm）**

- 对全局调制：LC 跟踪 RMSE 低 1.8–3.0%、跟踪峰值低 3–8%（10/10）；30–200 mm 平动权值低 5–20%；同步 RMS 多数行程低 0.5–1.2%（显著）。
- 对饱和冻结：LC 跟踪 RMSE 低 4.5–10%、跟踪峰值低 4–18%（10/10）。

## 5. 局限（写作时须如实说明）

- 标称工况仿真未复现实物中普通 BLNN 的偶发劣化运行，因此标称工况下 LC 的跟踪 RMSE 略高于 BLNN（10 mm 约 0%，300 mm 约 +27%），实物中 LC 跟踪更好。跟踪与同步收益以实物为准。
- 仿真同步误差约 0.028 mm，约为实测 0.08 mm 的三分之一，推测与同步带柔性等未建模因素有关；各方法间同步差异仅 0.5–3%。
- 最深约束（600 counts 的 300 mm、400 counts）下，全局调制与饱和冻结的平均跟踪略优于 LC（例如 600 counts、300 mm：2.17/2.20 对 2.37 mm），但同时压低了偏转学习；LC 同步 RMS 仍略好（约 1.5%）。
- 400 counts 时 100 mm 以上行程所有方法均无法跟踪轨迹，仅作极端情况展示。
- 摩擦×2 工况跟踪误差反而减小（控制器名义摩擦前馈 14.4 N 远大于辨识值），不构成"强约束"，仅作鲁棒性补充；单侧外部阻力工况下普通 BLNN 也未劣化。
- 等效惯量可辨识性弱；速度噪声为由饱和占比等反推的等效值，可用原始遥测静止段 `vel_L_mmps` 标准差复核。

## 6. 复现

在 `code/` 目录下运行（需要 numpy、pandas、scipy、matplotlib）：

```bash
python build_hardware_table.py ../../.. real_all.csv          # 汇总实物主表
python calibrate_plant.py real_all.csv 120 calib.csv          # 参数辨识（约 15 分钟）
P='{"m_eff":9.799,"fc_tot":1.146,"fs_ratio":2.127,"b_tot":9.736,"asym":1.193,"k_scale":0.585,"noise_rpm":22.974}'
python validate_calibrated.py "$P" 10 nominal.csv             # M1/M2/Fixed/LC，限幅 800
python compare_methods.py "$P" GLOBAL,FREEZE nominal_cmp.csv 800
python compare_methods.py "$P" BASE,VANILLA,LC,FIXED,GLOBAL,FREEZE strong600.csv 600
python compare_methods.py "$P" BASE,VANILLA,LC,GLOBAL,FREEZE overlimit400.csv 400
python summarize_validation.py nominal.csv real_all.csv       # 仿真-实物对照
python summarize_comparison.py strong600.csv                  # 方法对比与配对检验
python plot_rho.py real_all.csv nominal.csv fig_rho.png
```

鲁棒性工况：摩擦×2 将 `fc_tot` 改为 2.292；单侧外部阻力在 `P` 中加入 `"fext_L":0.744` 或 `1.487`，用 `validate_calibrated.py` 运行。

随机搜索的种子取自系统时间，重新辨识的结果会与表中略有差别；固定参数后的验证与对比结果可逐位复现。

`code/config_e5_final.py` 中的 `PlantConfig` 是旧版仿真对象的参数，新仿真不使用，保留仅为兼容理论目录中的历史脚本。
