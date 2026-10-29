# 105 次主实验原始数据、实物补测与固件来源全量索引 (Raw Experiment Data & Firmware Index) - 规范终版

本文件建立物理实验平台 105 次正式主实验原始数据、实物固定调制对照补测数据、正式固件源码、平台诊断试运行数据及离线仿真消融算例的权威来源索引与可核验路径。

---

## 一、嵌入式正式固件源码索引 (Firmware Source)

正式物理实验（第 5 次实验）及历史沿用配置已完整同步至本地目录 [`D:\lunwen2\all\all\05_正式固件与原始数据索引\firmware\`](file:///D:/lunwen2/all/all/05_正式固件与原始数据索引/firmware/)：

| 文件名称 | 原始源工程绝对路径 | 在论文中的证据与溯源职责 |
|---|---|---|
| **`experiment5_config.h`** | `d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\03_固件与代码准备\eassy_exp5_formal_firmware\Task\experiment5_config.h` | 论文第 5 次实物实验所加载的**正式参数配置头文件**。控制任务按 $1\,\text{ms}$ 周期配置（在 `motor_task` 中通过任务延时调度执行）；包含单侧限幅 $I_{\max} = 800\,\text{counts}$ (4.96 N)、单侧推力变化率限幅 $75\,\text{counts/ms}$、C2b 滑模面增益、学习协调增益 $\gamma_{\rm lc} = 100.0$ 及名义质量宏定义 `#define EXP5_M_Y 12.445f`。 |
| **`controller_e5_blnn_lc.c`** | `d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\03_固件与代码准备\eassy_exp5_formal_firmware\Task\controller_e5_blnn_lc.c` | STM32 嵌入式实际运行的**核心控制算法 C 源码**。包含双模态推力分配函数 `thrust_allocator_sync_c`、BLNN 宽度网络更新律、学习协调调制因子 $\rho$ 计算与两个一阶抗饱和状态滤波更新。 |
| **`experiment3_config.h`** | `d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\03_固件与代码准备\eassy_exp5_formal_firmware\Task\experiment3_config.h` | 控制器名义模型系数 $\hat{M}_y = 12.445\,\text{kg}$ 的**客观历史来源出处**。宏定义 `#define EXP3_C2_M_Y 12.445f` 证实该参数系前期第 3 次实验沿用留存，而非滚珠丝杠折算或临时构造。 |

---

## 二、多行程物理实验数据索引 (Multi-Stroke Trials)

最终公开统计包位于 [`../08_多行程实验_20_300mm/README.md`](../08_多行程实验_20_300mm/README.md)，覆盖 20、30、40、50、100、200 和 300 mm 七个行程。每个行程包含 10 个配对块，每个配对块包含 M1_BASE、M2_BLNN、M3_LCBLNN 三次独立运行，共 30 次运行。

公开包包含每个行程的 `FINAL_MASTER.csv`、`FINAL_SUMMARY.csv`、固件配置、HEX 和完整度审计。原始 BIN、逐帧遥测 CSV 和 LOG 保存在本地目录：

`d:\lunwen2\起重机\output\experiment5_{stroke}mm_10blocks\data_{stroke}mm_10blocks\`

本仓库不上传原始高频采集文件。统计表不能替代原始文件，论文中应按本地归档规则说明数据保存和质量审计方法。
## 三、实物固定调制补测数据索引 (Real Fixed-Modulation Hardware Trials)

除 105 次主实验外，在物理实验平台上实际执行了真实的固定调制对照补测实验。相关数据与配对统计已建立直接溯源链接：

- **补测配对统计结果**：[补测配对结果](../06_实物补测_探索性/paired_comparisons.json)
- **单次运行尝试记录底册**：[`all_attempts.csv`](../06_实物补测_探索性/all_attempts.csv)
- **补测实验协议与配置**：[`supplement_protocol.json`](../06_实物补测_探索性/supplement_protocol.json)
- **参数配置**：
  * 固定调制系数设定为：$\rho_{\rm fixed} = [0.917,\, 0.979]$（平动通道 $\rho_y = 0.917$，偏转通道 $\rho_\alpha = 0.979$）；
  * 控制器按 $1\,\text{ms}$ 周期配置，遥测按 $250\,\text{Hz}$ 采样配置；
- **对应运行记录（2 个独立配对 Pilot Block，共 6 次实机运行）**：
  * **Block P01**：
    - Vanilla 模式：`P01_VANILLA_A02_20261019_134201_248397`（同步 RMS $0.1059\,\text{mm}$，跟踪 RMSE $0.4015\,\text{mm}$，权值峰值 $0.650$）
    - Fixed 模式：`P01_FIXED_A01_20261019_140430_397352`（同步 RMS $0.0643\,\text{mm}$，跟踪 RMSE $0.4148\,\text{mm}$，权值峰值 $0.622$）
    - LC 模式：`P01_LC_A01_20261019_142919_136101`（同步 RMS $0.0640\,\text{mm}$，跟踪 RMSE $0.3103\,\text{mm}$，权值峰值 $0.358$）
  * **Block P02**：
    - LC 模式：`P02_LC_A01_20261019_194321_612054`（同步 RMS $0.0995\,\text{mm}$，跟踪 RMSE $0.2514\,\text{mm}$，权值峰值 $0.359$）
    - Fixed 模式：`P02_FIXED_A01_20261019_162704_174233`（同步 RMS $0.0621\,\text{mm}$，跟踪 RMSE $0.3014\,\text{mm}$，权值峰值 $0.337$）
    - Vanilla 模式：`P02_VANILLA_A01_20261019_164112_119007`（同步 RMS $0.1046\,\text{mm}$，跟踪 RMSE $0.2611\,\text{mm}$，权值峰值 $0.295$）
- **实物配对结果汇总 (`paired_comparisons.json`)**：
  * **LC 对比 Fixed**：跟踪 RMSE 均值差为 $-0.0772\,\text{mm}$（2/2 优势，LC 均值 $0.2809\,\text{mm}$ vs Fixed $0.3581\,\text{mm}$）；权值峰值均值差为 $-0.1210$（LC $0.3585$ vs Fixed $0.4795$）；同步 RMS 均值差为 $+0.0185\,\text{mm}$（1/2 优势）；
  * **LC 对比 Vanilla**：跟踪 RMSE 均值差为 $-0.0504\,\text{mm}$（2/2 优势）；权值峰值均值差为 $-0.1140$；同步 RMS 均值差为 $-0.0235\,\text{mm}$（2/2 优势）。

---

## 四、实机诊断数据与离线消融算例的明确区分

为避免实验来源混淆，对以下两类历史辅助数据进行严格定性与物理区分：

### 4.1 物理实验平台实机诊断运行数据 (Platform Diagnostic Trials)
- **数据存放路径**：`d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\05_数据采集与归档\`
- **主要数据文件**：
  * `EXP5_DIAG_BASE_20261017_065845.bin/.csv`: 实机基准滑模功能链路与通信吞吐诊断
  * `EXP5_DIAG_BASE_B1_20261017_103559.bin/.csv`: 实机首轮动力学响应预试运行
  * `EXP5_DIAG_BLNN_20261017_094808.bin/.csv`: 实机普通 BLNN 网络自适应在线更新诊断
  * `EXP5_DIAG_LCBLNN_20261017_100439.bin/.csv`: 实机动态 LC 协调机制与抗饱和状态诊断
- **定位定性**：系正式 35 块主实验开始前在物理实验台上开展的**真实嵌入式固件与传感器通信诊断试运行**，证实了 DMA 遥测队列无丢帧（`enqueue_drop = 0`）、控制执行耗时正常（单步平均约 $45\,\mu\text{s}$）。不混入正式 105 次主实验配对统计中。

### 4.2 逐行程对应离线仿真结果
- **结果目录**：[`D:\lunwen2\all\all\02_仿真与控制配置\multistroke_simulation\`](../02_仿真与控制配置/multistroke_simulation/)
- **主结果表**：[`EXP5_MULTISTROKE_SIMULATION_SUMMARY.csv`](../02_仿真与控制配置/multistroke_simulation/EXP5_MULTISTROKE_SIMULATION_SUMMARY.csv)
- **相对 BLNN 表**：[`EXP5_MULTISTROKE_SIMULATION_RELATIVE_TO_BLNN.csv`](../02_仿真与控制配置/multistroke_simulation/EXP5_MULTISTROKE_SIMULATION_RELATIVE_TO_BLNN.csv)
- **生成脚本**：[`run_multistroke_benchmark.py`](../02_仿真与控制配置/run_multistroke_benchmark.py)
- **定位定性**：纯 Python 离线数值仿真，严格复现 10、20、30、40、50、100、200、300 mm 各行程固件的 20 s 参考轨迹，统一比较 BLNN、固定调制与 LC-BLNN；不修改或替代实物原始记录。
