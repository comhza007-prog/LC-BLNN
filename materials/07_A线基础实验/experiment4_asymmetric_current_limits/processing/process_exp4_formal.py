from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_DIR = Path(r"C:\Users\ele\Desktop\lunwen\第四次实验\01_原始数据")
PLAN_FILE = Path(r"C:\Users\ele\Desktop\lunwen\第四次实验\00_实验说明\实验4_正式采集随机顺序.csv")
OUT_DIR = Path(r"C:\Users\ele\Desktop\lunwen\起重机\output\exp4_archive_build")
PATTERN = re.compile(
    r"^E4_(C2B(?:_SA)?)_L(800|1200|1600)_R(800|1200|1600)_M2000g_S10mm_run(0[1-5])\.csv$"
)

CHANNELS = {
    "I0": "time_s",
    "I1": "sample_seq",
    "I2": "phase",
    "I3": "reference_m",
    "I4": "reference_velocity_mps",
    "I5": "pos_left_m",
    "I6": "pos_right_m",
    "I7": "velocity_left_mps",
    "I8": "velocity_right_mps",
    "I9": "current_request_left",
    "I10": "current_request_right",
    "I11": "current_alloc_left",
    "I12": "current_alloc_right",
    "I13": "current_applied_left",
    "I14": "current_applied_right",
    "I15": "current_feedback_left",
    "I16": "current_feedback_right",
    "I17": "saturation_left",
    "I18": "saturation_right",
    "I19": "current_limit_left",
    "I20": "current_limit_right",
    "I21": "controller_id",
    "I22": "fault_flags",
    "I23": "z_aw_left",
    "I24": "z_aw_right",
    "I25": "delta_alloc_left",
    "I26": "delta_alloc_right",
    "I27": "battery_V",
    "I28": "added_load_kg",
    "I29": "temperature_left_C",
    "I30": "temperature_right_C",
}

CONTROLLERS = {
    "C2B": ("C2b", 3),
    "C2B_SA": ("C2b-SyncAlloc", 4),
}
LIMIT_PAIRS = [(1200, 1200), (800, 1600), (1600, 800)]


def compressed(values: pd.Series) -> list[int]:
    result: list[int] = []
    previous = None
    for value in values.round().astype(int):
        if value != previous:
            result.append(int(value))
            previous = value
    return result


def rms(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array)))) if len(array) else math.nan


def ci95(std: float, n: int) -> float:
    return float(2.776445105 * std / math.sqrt(n)) if n > 1 else math.nan


def recovery_time(df: pd.DataFrame, active_mask: pd.Series) -> float | None:
    sat = ((df["I17"].abs() > 0.5) | (df["I18"].abs() > 0.5)) & active_mask
    indexes = np.flatnonzero(sat.to_numpy())
    if len(indexes) == 0:
        return None
    last = int(indexes[-1])
    mean_pos = 0.5 * (df["I5"] + df["I6"])
    tracking_mm = 1000.0 * (mean_pos - df["I3"])
    sync_mm = 1000.0 * (df["I5"] - df["I6"])
    in_band = (tracking_mm.abs() <= 1.0) & (sync_mm.abs() <= 1.0)
    times = df["I0"].to_numpy(dtype=float)
    flags = in_band.to_numpy(dtype=bool)
    for start in range(last + 1, len(df)):
        if not flags[start]:
            continue
        end = start
        while end + 1 < len(df) and flags[end + 1]:
            end += 1
        if times[end] - times[start] >= 0.25:
            return float(times[start] - times[last])
    return None


def expected_names() -> set[str]:
    return {
        f"E4_{code}_L{left}_R{right}_M2000g_S10mm_run{run:02d}.csv"
        for code in CONTROLLERS
        for left, right in LIMIT_PAIRS
        for run in range(1, 6)
    }


def main() -> int:
    for subdir in ["02_处理后数据", "03_统计汇总", "04_论文图", "00_说明文档", "05_实验代码"]:
        (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    raw_files = sorted(p for p in RAW_DIR.glob("E4_*.csv") if PATTERN.match(p.name))
    expected = expected_names()
    actual = {p.name for p in raw_files}
    metrics_rows: list[dict] = []
    quality_rows: list[dict] = []
    cleaned_frames: dict[str, pd.DataFrame] = {}

    for path in raw_files:
        match = PATTERN.match(path.name)
        assert match is not None
        controller_code, left_text, right_text, run_text = match.groups()
        controller, expected_id = CONTROLLERS[controller_code]
        left_limit, right_limit, run = int(left_text), int(right_text), int(run_text)
        raw = pd.read_csv(path)
        issues: list[str] = []
        notes: list[str] = []

        if list(raw.columns) != list(CHANNELS):
            quality_rows.append({
                "file": path.name,
                "valid": False,
                "issues": "通道列不是I0至I30",
                "source_rows": len(raw),
                "columns": len(raw.columns),
            })
            continue
        raw = raw.apply(pd.to_numeric, errors="coerce")
        if raw.isna().any().any():
            issues.append("存在非数值或缺失值")

        reset = (
            raw["I0"].diff().lt(-0.05)
            | raw["I1"].diff().le(0)
            | (raw["I2"].shift(1).round().isin([6, 9]) & raw["I2"].round().eq(0))
        ).fillna(False)
        candidates: list[tuple[pd.DataFrame, int, int]] = []
        for _, segment in raw.groupby(reset.cumsum(), sort=True):
            stable = (
                segment["I19"].round().eq(left_limit)
                & segment["I20"].round().eq(right_limit)
                & segment["I21"].round().eq(expected_id)
                & segment["I28"].sub(2.0).abs().le(1e-4)
            ).to_numpy(dtype=bool)
            if len(stable) < 10:
                continue
            windows = np.convolve(stable.astype(int), np.ones(10, dtype=int), mode="valid")
            starts = np.flatnonzero(windows == 10)
            if len(starts) == 0:
                continue
            offset = int(starts[0])
            trimmed = segment.iloc[offset:].copy()
            if compressed(trimmed["I2"]) == [0, 1, 2, 3, 4, 5, 6]:
                candidates.append((trimmed, int(segment.index[offset]), int(segment.index[-1])))

        if not candidates:
            quality_rows.append({
                "file": path.name,
                "valid": False,
                "issues": "; ".join(issues + ["未找到配置匹配且阶段0至6完整的运行片段"]),
                "source_rows": len(raw),
                "columns": len(raw.columns),
            })
            continue

        selected, start_idx, end_idx = candidates[-1]
        df = selected.reset_index(drop=True)
        dt = df["I0"].diff().dropna()
        ds = df["I1"].diff().dropna()
        phase_path = compressed(df["I2"])
        configured = df[df["I2"].round().ge(1)]
        active_mask = df["I2"].round().between(2, 5)
        active = df.loc[active_mask]
        mean_pos = 0.5 * (df["I5"] + df["I6"])
        tracking_mm = 1000.0 * (mean_pos - df["I3"])
        sync_mm = 1000.0 * (df["I5"] - df["I6"])
        any_sat = (df["I17"].abs() > 0.5) | (df["I18"].abs() > 0.5)
        any_alloc = (df["I25"].abs() > 0.5) | (df["I26"].abs() > 0.5)
        active_dt = df.loc[active_mask, "I0"].diff().fillna(float(dt.median())).clip(lower=0, upper=0.05)
        phase6 = df[df["I2"].round().eq(6)]
        final = phase6.iloc[0]
        final_tracking_mm = 1000.0 * (0.5 * (final["I5"] + final["I6"]) - final["I3"])
        final_sync_mm = 1000.0 * (final["I5"] - final["I6"])
        seq_pct = 100.0 * float((ds == 1).mean()) if len(ds) else 0.0
        median_dt_ms = 1000.0 * float(dt.median()) if len(dt) else math.nan
        max_dt_ms = 1000.0 * float(dt.max()) if len(dt) else math.nan
        active_max_dt_ms = 1000.0 * float(active["I0"].diff().dropna().max()) if len(active) > 1 else math.nan

        if len(df) < 1000:
            issues.append(f"有效记录行数不足1000（{len(df)}）")
        if (dt <= 0).any():
            issues.append("MCU时间非严格递增")
        if (ds <= 0).any():
            issues.append("样本序号非严格递增")
        if seq_pct < 99.0:
            issues.append(f"I1连续率仅{seq_pct:.2f}%")
        if not np.isfinite(median_dt_ms) or median_dt_ms > 25.0:
            issues.append(f"中位采样间隔异常（{median_dt_ms:.1f} ms）")
        if np.isfinite(active_max_dt_ms) and active_max_dt_ms > 50.0:
            issues.append(f"主动阶段最大采样间隔异常（{active_max_dt_ms:.1f} ms）")
        if phase_path != [0, 1, 2, 3, 4, 5, 6]:
            issues.append(f"阶段路径异常（{phase_path}）")
        if int(round(df["I22"].abs().max())) != 0:
            issues.append("故障码非0")
        if set(configured["I19"].round().astype(int).unique()) != {left_limit}:
            issues.append("左轴限流与文件名不一致")
        if set(configured["I20"].round().astype(int).unique()) != {right_limit}:
            issues.append("右轴限流与文件名不一致")
        if set(configured["I21"].round().astype(int).unique()) != {expected_id}:
            issues.append("控制器编号与文件名不一致")
        if not np.allclose(configured["I28"], 2.0, atol=1e-4):
            issues.append("载荷不是2 kg")
        if int(round(df["I2"].iloc[-1])) != 6:
            issues.append("最终未到phase 6")
        if float(sync_mm.loc[active_mask].abs().max()) >= 5.0:
            issues.append("主动阶段同步误差达到5 mm保护阈值")
        if max(float(df["I29"].max()), float(df["I30"].max())) >= 80.0:
            issues.append("温度达到80 ℃保护阈值")
        if abs(final_tracking_mm) >= 1.0:
            issues.append(f"最终位置误差超出1 mm恢复带（{final_tracking_mm:.3f} mm）")
        if abs(final_sync_mm) >= 1.0:
            issues.append(f"最终同步误差超出1 mm恢复带（{final_sync_mm:.3f} mm）")
        if start_idx > 0:
            notes.append(f"舍弃开头{start_idx}帧VOFA历史数据")

        saturation_duration_s = float(np.sum(active_dt.to_numpy() * any_sat.loc[active_mask].to_numpy(dtype=float)))
        allocation_duration_s = float(np.sum(active_dt.to_numpy() * any_alloc.loc[active_mask].to_numpy(dtype=float)))
        applied_tv = float(active["I13"].diff().abs().sum() + active["I14"].diff().abs().sum())
        current_effort = float(
            np.sum((active["I13"].abs() + active["I14"].abs()).to_numpy() * active_dt.to_numpy())
        )
        condition_code = f"{controller_code}_L{left_limit}_R{right_limit}"
        metrics = {
            "file": path.name,
            "condition_code": condition_code,
            "controller_code": controller_code,
            "controller": controller,
            "controller_id": expected_id,
            "sync_alloc": int(controller_code.endswith("_SA")),
            "left_limit": left_limit,
            "right_limit": right_limit,
            "asymmetry_code": f"L{left_limit}_R{right_limit}",
            "run": run,
            "source_rows": len(raw),
            "rows": len(df),
            "discarded_prefix_rows": start_idx,
            "discarded_suffix_rows": len(raw) - end_idx - 1,
            "duration_s": float(df["I0"].iloc[-1] - df["I0"].iloc[0]),
            "median_dt_ms": median_dt_ms,
            "max_dt_ms": max_dt_ms,
            "active_max_dt_ms": active_max_dt_ms,
            "seq_consecutive_pct": seq_pct,
            "tracking_rmse_mm": rms(tracking_mm.loc[active_mask]),
            "tracking_max_abs_mm": float(tracking_mm.loc[active_mask].abs().max()),
            "sync_rms_mm": rms(sync_mm.loc[active_mask]),
            "sync_max_abs_mm": float(sync_mm.loc[active_mask].abs().max()),
            "saturation_fraction_pct": 100.0 * float(any_sat.loc[active_mask].mean()),
            "saturation_duration_s": saturation_duration_s,
            "allocator_fraction_pct": 100.0 * float(any_alloc.loc[active_mask].mean()),
            "allocator_duration_s": allocation_duration_s,
            "recovery_time_s": recovery_time(df, active_mask),
            "control_total_variation_count": applied_tv,
            "current_effort_count_s": current_effort,
            "request_peak_count": float(active[["I9", "I10"]].abs().to_numpy().max()),
            "applied_left_peak_count": float(active["I13"].abs().max()),
            "applied_right_peak_count": float(active["I14"].abs().max()),
            "feedback_peak_count": float(active[["I15", "I16"]].abs().to_numpy().max()),
            "z_aw_peak_count": float(active[["I23", "I24"]].abs().to_numpy().max()),
            "delta_alloc_peak_count": float(active[["I25", "I26"]].abs().to_numpy().max()),
            "battery_mean_V": float(df["I27"].mean()),
            "battery_min_V": float(df["I27"].min()),
            "battery_max_V": float(df["I27"].max()),
            "temperature_left_max_C": float(df["I29"].max()),
            "temperature_right_max_C": float(df["I30"].max()),
            "final_tracking_error_mm": float(final_tracking_mm),
            "final_sync_error_mm": float(final_sync_mm),
            "fault_max": int(round(df["I22"].abs().max())),
            "valid": len(issues) == 0,
            "issues": "; ".join(issues),
            "notes": "; ".join(notes),
            "mtime": path.stat().st_mtime,
        }
        metrics_rows.append(metrics)
        quality_rows.append({
            "file": path.name,
            "valid": metrics["valid"],
            "issues": metrics["issues"],
            "notes": metrics["notes"],
            "source_rows": len(raw),
            "rows": len(df),
            "discarded_prefix_rows": start_idx,
            "discarded_suffix_rows": len(raw) - end_idx - 1,
            "columns": len(df.columns),
            "seq_consecutive_pct": seq_pct,
            "median_dt_ms": median_dt_ms,
            "active_max_dt_ms": active_max_dt_ms,
            "phase_path": "→".join(str(v) for v in phase_path),
            "fault_max": metrics["fault_max"],
            "controller_id": expected_id,
            "left_limit": left_limit,
            "right_limit": right_limit,
            "load_kg": 2.0,
            "sync_max_abs_mm": metrics["sync_max_abs_mm"],
            "temperature_max_C": max(metrics["temperature_left_max_C"], metrics["temperature_right_max_C"]),
            "final_tracking_error_mm": final_tracking_mm,
            "final_sync_error_mm": final_sync_mm,
        })

        clean = df.rename(columns=CHANNELS).copy()
        t0 = float(df.loc[df["I2"].round().eq(2), "I0"].iloc[0])
        clean.insert(0, "time_rel_s", df["I0"] - t0)
        clean["mean_position_m"] = mean_pos
        clean["tracking_error_mm"] = tracking_mm
        clean["sync_error_mm"] = sync_mm
        clean["saturation_any"] = any_sat.astype(int)
        clean["allocator_active"] = any_alloc.astype(int)
        cleaned_frames[path.name] = clean

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["controller_id", "left_limit", "right_limit", "run"])
    quality_df = pd.DataFrame(quality_rows).sort_values("file")
    all_valid = bool(actual == expected and len(metrics_df) == 30 and metrics_df["valid"].all())

    plan = pd.read_csv(PLAN_FILE)
    expected_order = plan["保存文件名"].tolist()
    actual_order_df = metrics_df.sort_values("mtime").copy()
    actual_order_df.insert(0, "actual_sequence", range(1, len(actual_order_df) + 1))
    actual_order_df["expected_file"] = expected_order[: len(actual_order_df)]
    actual_order_df["order_match"] = actual_order_df["file"] == actual_order_df["expected_file"]
    random_order_followed = bool(actual_order_df["order_match"].all())

    metric_columns = [
        "tracking_rmse_mm", "tracking_max_abs_mm", "sync_rms_mm", "sync_max_abs_mm",
        "saturation_fraction_pct", "saturation_duration_s", "allocator_fraction_pct",
        "allocator_duration_s", "recovery_time_s", "control_total_variation_count",
        "current_effort_count_s", "request_peak_count", "applied_left_peak_count",
        "applied_right_peak_count", "feedback_peak_count", "z_aw_peak_count",
        "delta_alloc_peak_count", "battery_mean_V", "battery_min_V",
        "temperature_left_max_C", "temperature_right_max_C",
        "final_tracking_error_mm", "final_sync_error_mm",
    ]
    summary_rows: list[dict] = []
    for (code, left, right), group in metrics_df.groupby(["controller_code", "left_limit", "right_limit"], sort=False):
        out = {
            "condition_code": f"{code}_L{left}_R{right}",
            "controller_code": code,
            "controller": CONTROLLERS[code][0],
            "left_limit": int(left),
            "right_limit": int(right),
            "n": int(len(group)),
            "valid_n": int(group["valid"].sum()),
        }
        for col in metric_columns:
            values = group[col].dropna()
            mean = float(values.mean()) if len(values) else math.nan
            std = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            out[f"{col}_mean"] = mean
            out[f"{col}_std"] = std
            out[f"{col}_ci95"] = ci95(std, len(values)) if len(values) > 1 else math.nan
        summary_rows.append(out)
    summary_df = pd.DataFrame(summary_rows).sort_values(["controller_code", "left_limit", "right_limit"])

    effect_rows: list[dict] = []
    effects_metrics = [
        "tracking_rmse_mm", "sync_rms_mm", "sync_max_abs_mm",
        "saturation_fraction_pct", "recovery_time_s", "control_total_variation_count",
    ]
    for left, right in LIMIT_PAIRS:
        base = summary_df[(summary_df.controller_code == "C2B") & (summary_df.left_limit == left) & (summary_df.right_limit == right)].iloc[0]
        sa = summary_df[(summary_df.controller_code == "C2B_SA") & (summary_df.left_limit == left) & (summary_df.right_limit == right)].iloc[0]
        for metric in effects_metrics:
            b = float(base[f"{metric}_mean"])
            s = float(sa[f"{metric}_mean"])
            effect_rows.append({
                "comparison": "C2b-SyncAlloc vs C2b",
                "left_limit": left,
                "right_limit": right,
                "metric": metric,
                "baseline_mean": b,
                "modified_mean": s,
                "absolute_change": s - b,
                "relative_change_pct": 100.0 * (s - b) / b if b != 0 else math.nan,
            })
    for code in CONTROLLERS:
        balanced = summary_df[(summary_df.controller_code == code) & (summary_df.left_limit == 1200)].iloc[0]
        for left, right in [(800, 1600), (1600, 800)]:
            altered = summary_df[(summary_df.controller_code == code) & (summary_df.left_limit == left) & (summary_df.right_limit == right)].iloc[0]
            for metric in effects_metrics:
                b = float(balanced[f"{metric}_mean"])
                a = float(altered[f"{metric}_mean"])
                effect_rows.append({
                    "comparison": f"{code} L{left}/R{right} vs balanced",
                    "left_limit": left,
                    "right_limit": right,
                    "metric": metric,
                    "baseline_mean": b,
                    "modified_mean": a,
                    "absolute_change": a - b,
                    "relative_change_pct": 100.0 * (a - b) / b if b != 0 else math.nan,
                })
    effects_df = pd.DataFrame(effect_rows)

    actual_order_df["acquisition_index"] = np.arange(1, len(actual_order_df) + 1)
    battery_range = float(metrics_df["battery_mean_V"].max() - metrics_df["battery_mean_V"].min())
    battery_corr = float(actual_order_df["acquisition_index"].corr(actual_order_df["battery_mean_V"], method="spearman"))
    protocol = {
        "expected_file_count": 30,
        "actual_file_count": len(raw_files),
        "missing_files": sorted(expected - actual),
        "unexpected_files": sorted(actual - expected),
        "all_files_valid": all_valid,
        "planned_random_order_followed": random_order_followed,
        "actual_collection_pattern": "condition-blocked (five repeats per condition)" if not random_order_followed else "planned randomized block order",
        "battery_mean_range_V": battery_range,
        "battery_spearman_vs_order": battery_corr,
        "archive_decision": "eligible_with_protocol_deviation" if all_valid else "not_eligible",
    }

    metrics_out = metrics_df.drop(columns=["mtime"])
    actual_order_out = actual_order_df.drop(columns=["mtime"])
    tables = {
        "实验4逐次指标": metrics_out,
        "实验4逐文件质量检查": quality_df,
        "实验4分组统计": summary_df,
        "实验4效应对比": effects_df,
        "实验4实际采集顺序": actual_order_out,
    }
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / "03_统计汇总" / f"{name}.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "03_统计汇总" / "实验4协议与质量结论.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    workbook_json = {
        name: json.loads(frame.to_json(orient="records", force_ascii=False))
        for name, frame in tables.items()
    }
    workbook_json["实验4协议结论"] = [protocol]
    (OUT_DIR / "03_统计汇总" / "实验4工作簿数据.json").write_text(
        json.dumps(workbook_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, clean in cleaned_frames.items():
        clean.to_csv(OUT_DIR / "02_处理后数据" / name.replace(".csv", "_clean.csv"), index=False, encoding="utf-8-sig")

    if len(summary_df):
        make_figures(metrics_df, summary_df, cleaned_frames)

    result = {
        "files": len(raw_files),
        "valid_files": int(metrics_df["valid"].sum()) if len(metrics_df) else 0,
        "all_valid": all_valid,
        "planned_random_order_followed": random_order_followed,
        "battery_mean_range_V": battery_range,
        "battery_spearman_vs_order": battery_corr,
        "invalid": quality_df.loc[~quality_df["valid"], ["file", "issues"]].to_dict("records") if len(quality_df) else [],
        "group_summary": summary_df[[
            "condition_code", "tracking_rmse_mm_mean", "sync_rms_mm_mean",
            "sync_max_abs_mm_mean", "saturation_fraction_pct_mean", "recovery_time_s_mean",
        ]].to_dict("records"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_valid else 2


def make_figures(metrics: pd.DataFrame, summary: pd.DataFrame, cleaned: dict[str, pd.DataFrame]) -> None:
    out = OUT_DIR / "04_论文图"
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    order = [
        "C2B_L1200_R1200", "C2B_L800_R1600", "C2B_L1600_R800",
        "C2B_SA_L1200_R1200", "C2B_SA_L800_R1600", "C2B_SA_L1600_R800",
    ]
    labels = ["C2b\n1200/1200", "C2b\n800/1600", "C2b\n1600/800", "C2b-SA\n1200/1200", "C2b-SA\n800/1600", "C2b-SA\n1600/800"]
    colors = ["#4C78A8"] * 3 + ["#F58518"] * 3

    def bar_panel(ax, metric: str, ylabel: str, title: str) -> None:
        rows = summary.set_index("condition_code").loc[order]
        values = rows[f"{metric}_mean"].to_numpy()
        errors = rows[f"{metric}_std"].to_numpy()
        ax.bar(np.arange(6), values, yerr=errors, capsize=3, color=colors, width=0.72)
        ax.set_xticks(np.arange(6), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    bar_panel(axes[0], "tracking_rmse_mm", "RMSE (mm)", "Tracking error")
    bar_panel(axes[1], "sync_rms_mm", "RMS (mm)", "Synchronization error")
    fig.savefig(out / "图1_六种条件跟踪与同步性能.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    bar_panel(axes[0], "saturation_fraction_pct", "Active samples (%)", "Saturation activity")
    bar_panel(axes[1], "control_total_variation_count", "Total variation (count)", "Control variation")
    fig.savefig(out / "图2_饱和占比与控制量总变差.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True, constrained_layout=True)
    representative = [
        ("C2B", 1600, 800, "C2b 1600/800"),
        ("C2B_SA", 1600, 800, "C2b-SA 1600/800"),
    ]
    for col, (code, left, right, title) in enumerate(representative):
        group = metrics[(metrics.controller_code == code) & (metrics.left_limit == left) & (metrics.right_limit == right)].copy()
        target = float(group["sync_rms_mm"].median())
        row = group.iloc[(group["sync_rms_mm"] - target).abs().argmin()]
        frame = cleaned[str(row["file"])]
        # Phase 6 disables motor output; exclude post-stop load drift from the
        # controlled-trajectory comparison.
        mask = frame["phase"].between(1, 5)
        axes[0, col].plot(frame.loc[mask, "time_rel_s"], frame.loc[mask, "tracking_error_mm"], label="tracking", lw=1.0)
        axes[0, col].plot(frame.loc[mask, "time_rel_s"], frame.loc[mask, "sync_error_mm"], label="sync", lw=1.0)
        axes[0, col].axhline(1.0, color="#999999", ls="--", lw=0.7)
        axes[0, col].axhline(-1.0, color="#999999", ls="--", lw=0.7)
        axes[0, col].set_title(f"{title}, run {int(row['run']):02d}")
        axes[0, col].grid(alpha=0.2)
        axes[1, col].plot(frame.loc[mask, "time_rel_s"], frame.loc[mask, "current_applied_left"], label="left", lw=0.9)
        axes[1, col].plot(frame.loc[mask, "time_rel_s"], frame.loc[mask, "current_applied_right"], label="right", lw=0.9)
        axes[1, col].grid(alpha=0.2)
        axes[1, col].set_xlabel("Time from forward start (s)")
    axes[0, 0].set_ylabel("Error (mm)")
    axes[1, 0].set_ylabel("Applied current (count)")
    axes[0, 1].legend(frameon=False, ncol=2)
    axes[1, 1].legend(frameon=False, ncol=2)
    fig.savefig(out / "图3_代表运行误差与控制电流.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    bar_panel(axes[0], "delta_alloc_peak_count", "Peak (count)", "Allocation deficit")
    bar_panel(axes[1], "z_aw_peak_count", "Peak (count)", "Dynamic anti-windup state")
    fig.savefig(out / "图4_分配缺口与抗饱和状态.png", bbox_inches="tight")
    plt.close(fig)

    actual = metrics.sort_values("mtime")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), constrained_layout=True)
    idx = np.arange(1, len(actual) + 1)
    axes[0].plot(idx, actual["battery_mean_V"], marker="o", ms=2.7, lw=0.8)
    axes[0].set_xlabel("Acquisition index")
    axes[0].set_ylabel("Mean battery voltage (V)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(idx, actual["temperature_left_max_C"], marker="o", ms=2.5, lw=0.8, label="left")
    axes[1].plot(idx, actual["temperature_right_max_C"], marker="o", ms=2.5, lw=0.8, label="right")
    axes[1].set_xlabel("Acquisition index")
    axes[1].set_ylabel("Maximum temperature (C)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(out / "图5_采集顺序中的电压与温度.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
