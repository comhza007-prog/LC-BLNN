# -*- coding: utf-8 -*-
"""汇总 10–300 mm 实物主表为一个 CSV，供参数辨识与仿真-实物对照使用。

用法（在本目录运行）：python build_hardware_table.py ../../.. real_all.csv
第一个参数为仓库根目录。
"""
import sys, glob
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else '../../..'
out = sys.argv[2] if len(sys.argv) > 2 else 'real_all.csv'
fs = glob.glob(root + '/materials/0[89]*/**/*MASTER.csv', recursive=True)
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d.to_csv(out, index=False)
print(f'{len(fs)} master tables, {len(d)} runs -> {out}')
