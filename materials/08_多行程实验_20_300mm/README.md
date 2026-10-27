# 20–300 mm 多行程实验上传包

本目录上传论文复现和审计所需的最终主表、统计表、三套固件配置、HEX 文件和 firmware manifest。

## 目录

每个行程目录包含：

- `*_FINAL_MASTER.csv`：30 次运行主数据；
- `*_FINAL_SUMMARY.csv`：三种方法的汇总统计；
- `compiled_firmwares/`：M1_BASE、M2_BLNN、M3_LCBLNN 的 HEX、配置头文件和 manifest。

## 原始数据边界

原始 BIN、逐帧遥测 CSV 和 LOG 文件总量较大，未上传到 GitHub；它们保留在本地实验归档目录。`MULTISTROKE_AUDIT.csv` 记录每个行程的行数、方法配对、故障字、完成阶段、完整度和原始文件数量。

原始数据不能由本目录中的汇总表替代。论文中应同时说明原始数据保存位置和完整度审计规则。

## 方法标识

- M1_BASE：无 BLNN、无 LC；
- M2_BLNN：普通 BLNN；
- M3_LCBLNN：BLNN + 学习协调 LC。

## 注意

20 mm 的本地历史目录名称为 `experiment5_20mm_5blocks`，但其主表实际包含 10 个块；本上传包统一按实际内容标为 20 mm、10 blocks。
