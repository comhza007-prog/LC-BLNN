# 10–300 mm 一一对应仿真说明

## 仿真定位

本仿真与实物多行程实验采用相同的行程集合、控制器限幅和参考轨迹，用于比较：

- **BLNN**：普通 BLNN；
- **固定调制**：消融对照；
- **LC-BLNN**：学习协调方法。

仿真用于解释方法趋势，实物实验用于验证真实平台性能。仿真结果不替代实物统计。

## 轨迹和参数

所有行程均严格复现对应 motor_task.c 的参考轨迹：

- 4 s 预保持；
- 4 s 五次平滑前进；
- 2 s 端点保持；
- 4 s 五次平滑返回；
- 6 s 恢复保持；
- 总时长 20 s，控制步长 1 ms。

行程为 10、20、30、40、50、100、200、300 mm。控制器限幅为 800 counts，仿真对象质量为 3 kg，控制器名义平动参数为 12.445 kg。

## 结果文件

- multistroke_simulation/EXP5_MULTISTROKE_SIMULATION_SUMMARY.csv：24 组完整结果；
- multistroke_simulation/EXP5_MULTISTROKE_SIMULATION_RELATIVE_TO_BLNN.csv：相对 BLNN 的变化率；
- run_multistroke_benchmark.py：可复算脚本；
- multistroke_simulation/README.json：参数和轨迹记录。

## 结果概述

LC-BLNN 在全部行程中显著压低权值峰值，且跟踪 RMSE 在本仿真设置下低于 BLNN。同步 RMS 在短行程基本持平或略有改善，长行程下可能出现小幅代价。固定调制用于说明单纯降低学习驱动与 LC-BLNN 的差异。

10、20、30、40、50 mm 构成短行程段，100、200、300 mm 构成长行程段。正文可重点展示 10、50、300 mm，完整 8 行程结果放入汇总表或附录。
