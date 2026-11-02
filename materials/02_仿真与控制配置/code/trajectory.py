# -*- coding: utf-8 -*-
"""与各行程固件 motor_task.c 一致的 20 s 参考轨迹。

0-4 s 预保持；4-8 s 五次多项式平滑前进；8-10 s 端点保持；
10-14 s 平滑返回；14-20 s 恢复保持；控制步长 1 ms。
"""
import numpy as np


def trajectory(stroke_mm, dt=0.001, duration=20.0):
    t = np.arange(0.0, duration, dt)
    stroke = stroke_mm / 1000.0
    yd = np.zeros_like(t); yd_dot = np.zeros_like(t); yd_ddot = np.zeros_like(t)

    def smooth(u):
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = 30*u**2 - 60*u**3 + 30*u**4
        dds = 60*u - 180*u**2 + 120*u**3
        return s, ds, dds

    fwd = (t >= 4.0) & (t < 8.0)
    u = (t[fwd] - 4.0) / 4.0; ss, ds, dds = smooth(u)
    yd[fwd] = stroke*ss; yd_dot[fwd] = stroke*ds/4.0; yd_ddot[fwd] = stroke*dds/16.0
    hold = (t >= 8.0) & (t < 10.0); yd[hold] = stroke
    ret = (t >= 10.0) & (t < 14.0)
    u = (t[ret] - 10.0) / 4.0; ss, ds, dds = smooth(u)
    yd[ret] = stroke*(1.0 - ss); yd_dot[ret] = -stroke*ds/4.0; yd_ddot[ret] = -stroke*dds/16.0
    return t, yd, yd_dot, yd_ddot
