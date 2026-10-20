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

## 二、105 次物理实验原始采集数据索引 (Raw Experimental Trials)

- **采集目录绝对路径**：`d:\lunwen2\起重机\output\experiment5_sync_coordinated_blnn\05_数据采集与归档\`
- **控制时序与遥测机制**：
  - **控制时序**：正式固件核心控制算法在 `motor_task` 中通过任务延时调度执行，**控制任务按 1 ms 周期配置**，运动全程内部控制器调用累计约 20,000 步（固件摘要字段 `ctrl_call_count = 20000`）；
  - **遥测机制**：上位机通过 DMA 高速遥测接收记录，**遥测标称 500 Hz，存在丢帧导致的较大时间间隔**；
  - **数据规模**：每次时域机动记录时长约 22 秒，产生约 11,000 行遥测记录（通常为 10,997 ~ 11,000 点，字段 `data_generated = 11000`）。
- **实验设计与配对规模**：
  - 共包含 **35 个独立配对块 (Block 01 ~ Block 35)**；
  - 每个 Block 包含 3 种控制模式完全独立的时域运行：
    * `M1_BASE`: 基准滑模控制模式 (35 次)
    * `M2_BLNN`: 普通宽度学习网络模式 (35 次)
    * `M3_LCBLNN`: 模态学习协调宽度学习网络模式 (35 次)
  - 总计 **105 次完整时域运行**，每运行均包含原始二进制采集文件 (`.bin`, 约 1 MB) 与工程转换 CSV 数据 (`.csv`, 约 4 MB)，共计 **210 个高频采集文件**加 105 个单次指标摘要表 (`EXP5_SUMMARY_*.csv`)。

### 2.1 代表性数据文件索引示例（前 5 个 Block 与第 35 个 Block）

| Block 编号 | 控制模式 | 原始二进制文件 (.bin) | 完整时域 CSV 数据 (.csv, 遥测标称 500 Hz) | 单次统计摘要 CSV |
|---|---|---|---|---|
| **Block 01** | M1_BASE | `EXP5_5B_B01_M1_BASE_20261017_151739.bin` | `EXP5_5B_B01_M1_BASE_20261017_151739.csv` | `EXP5_SUMMARY_5B_B01_M1_BASE_*.csv` |
| | M2_BLNN | `EXP5_5B_B01_M2_BLNN_20261017_151906.bin` | `EXP5_5B_B01_M2_BLNN_20261017_151906.csv` | `EXP5_SUMMARY_5B_B01_M2_BLNN_*.csv` |
| | M3_LCBLNN | `EXP5_5B_B01_M3_LCBLNN_20261017_151950.bin` | `EXP5_5B_B01_M3_LCBLNN_20261017_151950.csv` | `EXP5_SUMMARY_5B_B01_M3_LCBLNN_*.csv` |
| **Block 02** | M1_BASE | `EXP5_5B_B02_M1_BASE_20261017_152035.bin` | `EXP5_5B_B02_M1_BASE_20261017_152035.csv` | `EXP5_SUMMARY_5B_B02_M1_BASE_*.csv` |
| | M2_BLNN | `EXP5_5B_B02_M2_BLNN_20261017_152120.bin` | `EXP5_5B_B02_M2_BLNN_20261017_152120.csv` | `EXP5_SUMMARY_5B_B02_M2_BLNN_*.csv` |
| | M3_LCBLNN | `EXP5_5B_B02_M3_LCBLNN_20261017_152247.bin` | `EXP5_5B_B02_M3_LCBLNN_20261017_152247.csv` | `EXP5_SUMMARY_5B_B02_M3_LCBLNN_*.csv` |
| **Block 03** | M1_BASE | `EXP5_5B_B03_M1_BASE_20261017_152334.bin` | `EXP5_5B_B03_M1_BASE_20261017_152334.csv` | `EXP5_SUMMARY_5B_B03_M1_BASE_*.csv` |
| | M2_BLNN | `EXP5_5B_B03_M2_BLNN_20261017_152421.bin` | `EXP5_5B_B03_M2_BLNN_20261017_152421.csv` | `EXP5_SUMMARY_5B_B03_M2_BLNN_*.csv` |
| | M3_LCBLNN | `EXP5_5B_B03_M3_LCBLNN_20261017_152547.bin` | `EXP5_5B_B03_M3_LCBLNN_20261017_152547.csv` | `EXP5_SUMMARY_5B_B03_M3_LCBLNN_*.csv` |
| ... | ... | ... (严格连续递增至 Block 35) | ... (全量 105 组配对完整保留) | ... |
| **Block 35** | M1_BASE | `EXP5_5B_B35_M1_BASE_20261017_175706.bin` | `EXP5_5B_B35_M1_BASE_20261017_175706.csv` | `EXP5_SUMMARY_5B_B35_M1_BASE_*.csv` |
| | M2_BLNN | `EXP5_5B_B35_M2_BLNN_20261017_175833.bin` | `EXP5_5B_B35_M2_BLNN_20261017_175833.csv` | `EXP5_SUMMARY_5B_B35_M2_BLNN_*.csv` |
| | M3_LCBLNN | `EXP5_5B_B35_M3_LCBLNN_20261017_175918.bin` | `EXP5_5B_B35_M3_LCBLNN_20261017_175918.csv` | `EXP5_SUMMARY_5B_B35_M3_LCBLNN_*.csv` |

---

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

### 4.2 离线固定调制消融仿真算例 (Offline Fixed-Modulation Simulation Benchmark)
- **数据存放路径**：[`D:\lunwen2\all\all\02_仿真与控制配置\EXP5_OFFLINE_FIXED_MODULATION_BENCHMARK.csv`](../02_仿真与控制配置/EXP5_OFFLINE_FIXED_MODULATION_BENCHMARK.csv)
- **对应图表文件**：[`D:\lunwen2\all\all\02_仿真与控制配置\EXP5_OFFLINE_FIXED_MODULATION_COMPARISON.png`](../02_仿真与控制配置/EXP5_OFFLINE_FIXED_MODULATION_COMPARISON.png)
- **生成脚本路径**：[`D:\lunwen2\all\all\02_仿真与控制配置\run_offline_fixed_modulation_benchmark.py`](../02_仿真与控制配置/run_offline_fixed_modulation_benchmark.py)
- **定位定性**：系纯 Python **离线数值仿真消融算例**，用于算法设计初期探索固定参数折减（如 $\rho=0.5$）与动态调制的对比。明确区分于物理实验台采集数据，不作为论文实物效能论据，仅作为算法演进的仿真参考。
