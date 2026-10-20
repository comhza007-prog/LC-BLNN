from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_DIR = Path(r"C:\Users\ele\Desktop\lunwen\第三次实验\01_原始数据")
OUT_DIR = Path(r"C:\Users\ele\Desktop\lunwen\起重机\output\experiment3_archive_staging")
PATTERN = re.compile(
    r"^E3_(C2A(?:_SA)?|C2B(?:_SA)?)_I(2400|800)_M2000g_S10mm_run(0[1-5])\.csv$"
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
    "I19": "current_limit",
    "I20": "controller_id",
    "I21": "fault_flags",
    "I22": "tracking_in_band",
    "I23": "z_aw_left",
    "I24": "z_aw_right",
    "I25": "delta_alloc_left",
    "I26": "delta_alloc_right",
    "I27": "battery_V",
    "I28": "added_load_kg",
    "I29": "saturation_latched",
}

CONTROLLERS = {
    "C2A": ("C2a", 1),
    "C2A_SA": ("C2a-SyncAlloc", 2),
    "C2B": ("C2b", 3),
    "C2B_SA": ("C2b-SyncAlloc", 4),
}

EXPECTED_ORDER = [
    "C2A_SA_I2400", "C2B_SA_I2400", "C2A_SA_I800", "C2A_I2400",
    "C2B_I800", "C2B_I2400", "C2B_SA_I800", "C2A_I800",
    "C2A_SA_I2400", "C2B_SA_I2400", "C2B_I2400", "C2B_I800",
    "C2A_SA_I800", "C2A_I800", "C2A_I2400", "C2B_SA_I800",
    "C2A_SA_I800", "C2B_I800", "C2A_I2400", "C2B_SA_I800",
    "C2A_I800", "C2B_SA_I2400", "C2B_I2400", "C2A_SA_I2400",
    "C2A_SA_I800", "C2A_I800", "C2A_I2400", "C2B_SA_I800",
    "C2B_SA_I2400", "C2B_I800", "C2B_I2400", "C2A_SA_I2400",
    "C2B_I800", "C2A_I2400", "C2A_SA_I800", "C2A_SA_I2400",
    "C2B_I2400", "C2B_SA_I2400", "C2B_SA_I800", "C2A_I800",
]


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
    return float(np.sqrt(np.mean(np.square(array))))


def recovery_time(df: pd.DataFrame, active_mask: pd.Series) -> float | None:
    sat = ((df["I17"].abs() > 0.5) | (df["I18"].abs() > 0.5)) & active_mask
    indexes = np.flatnonzero(sat.to_numpy())
    if len(indexes) == 0:
        return None
    last = int(indexes[-1])
    t_last = float(df.loc[last, "I0"])
    in_band = df["I22"].to_numpy(dtype=float) > 0.5
    times = df["I0"].to_numpy(dtype=float)
    for start in range(last + 1, len(df)):
        if not in_band[start]:
            continue
        end = start
        while end + 1 < len(df) and in_band[end + 1]:
            end += 1
        if times[end] - times[start] >= 0.25:
            return float(times[start] - t_last)
    return None


def ci95(std: float, n: int) -> float:
    return float(2.776445105 * std / math.sqrt(n)) if n > 1 else math.nan


def main() -> int:
    raw_files = sorted(p for p in RAW_DIR.glob("E3_*.csv") if PATTERN.match(p.name))
    expected_names = {
        f"E3_{code}_I{limit}_M2000g_S10mm_run{run:02d}.csv"
        for code in CONTROLLERS
        for limit in (2400, 800)
        for run in range(1, 6)
    }
    actual_names = {p.name for p in raw_files}

    for subdir in ["00_说明文档", "01_原始数据", "02_处理后数据", "03_统计汇总", "04_论文图", "05_实验代码"]:
        (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    quality: list[dict] = []
    cleaned_frames: dict[str, pd.DataFrame] = {}

    for path in raw_files:
        match = PATTERN.match(path.name)
        assert match is not None
        code, limit_text, run_text = match.groups()
        controller, expected_id = CONTROLLERS[code]
        current_limit = int(limit_text)
        run = int(run_text)
        raw_df = pd.read_csv(path)
        structural_errors: list[str] = []
        expected_columns = list(CHANNELS)
        if list(raw_df.columns) != expected_columns:
            structural_errors.append("通道列不是I0至I29")
            quality.append({
                "file": path.name, "valid": False,
                "issues": "; ".join(structural_errors), "rows": len(raw_df), "columns": len(raw_df.columns),
            })
            continue

        raw_df = raw_df.apply(pd.to_numeric, errors="coerce")
        if raw_df.isna().any().any():
            structural_errors.append("存在非数值或缺失值")

        # VOFA exports its fixed history buffer, so a file can begin with the
        # preceding run. Split on MCU/sequence reset and on done/fault -> arm,
        # then select the last complete segment matching the filename config.
        reset = (
            raw_df["I0"].diff().lt(-0.05)
            | raw_df["I1"].diff().le(0)
            | (raw_df["I2"].shift(1).round().isin([6, 9]) & raw_df["I2"].round().eq(0))
        ).fillna(False)
        segment_id = reset.cumsum()
        candidates: list[pd.DataFrame] = []
        for _, segment in raw_df.groupby(segment_id, sort=True):
            phase_candidate = compressed(segment["I2"])
            configured = segment[segment["I2"].round().ge(1)]
            config_ok = (
                len(configured) > 0
                and set(configured["I19"].round().astype(int).unique()) == {current_limit}
                and set(configured["I20"].round().astype(int).unique()) == {expected_id}
                and np.allclose(configured["I28"], 2.0, atol=1e-4)
            )
            if phase_candidate == [0, 1, 2, 3, 4, 5, 6] and config_ok:
                candidates.append(segment)
        if not candidates:
            structural_errors.append("未找到配置匹配且阶段0至6完整的运行片段")
            quality.append({
                "file": path.name, "valid": False,
                "issues": "; ".join(structural_errors), "rows": len(raw_df), "columns": len(raw_df.columns),
            })
            continue

        selected = candidates[-1]
        selected_start_row = int(selected.index[0])
        selected_end_row = int(selected.index[-1])
        stable_config = (
            selected["I19"].round().eq(current_limit)
            & selected["I20"].round().eq(expected_id)
            & selected["I28"].sub(2.0).abs().le(1e-4)
        ).to_numpy(dtype=bool)
        stable_run = np.convolve(stable_config.astype(int), np.ones(10, dtype=int), mode="valid")
        stable_starts = np.flatnonzero(stable_run == 10)
        if len(stable_starts) == 0:
            structural_errors.append("未找到连续10帧稳定配置数据")
            quality.append({
                "file": path.name, "valid": False,
                "issues": "; ".join(structural_errors), "rows": len(raw_df), "columns": len(raw_df.columns),
            })
            continue
        trim_offset = int(stable_starts[0])
        selected_start_row += trim_offset
        selected = selected.iloc[trim_offset:]
        discarded_prefix_rows = selected_start_row
        discarded_suffix_rows = len(raw_df) - selected_end_row - 1
        df = selected.reset_index(drop=True)

        dt = df["I0"].diff().dropna()
        ds = df["I1"].diff().dropna()
        seq_pct = float((ds == 1).mean()) if len(ds) else 0.0
        median_dt = float(dt.median()) if len(dt) else math.nan
        max_dt = float(dt.max()) if len(dt) else math.nan
        phase_path = compressed(df["I2"])

        if len(df) < 1000:
            structural_errors.append("记录行数不足1000")
        if (dt <= 0).any():
            structural_errors.append("MCU时间非严格递增")
        if (ds <= 0).any():
            structural_errors.append("样本序号非严格递增")
        if seq_pct < 0.99:
            structural_errors.append(f"I1连续率仅{seq_pct:.2%}")
        if not np.isfinite(median_dt) or median_dt > 0.025:
            structural_errors.append(f"中位采样间隔{median_dt * 1000:.1f}ms")
        if phase_path != [0, 1, 2, 3, 4, 5, 6]:
            structural_errors.append(f"阶段路径为{phase_path}")
        if int(round(df["I21"].abs().max())) != 0:
            structural_errors.append("故障码非0")
        configured_df = df[df["I2"].round().ge(1)]
        if set(configured_df["I19"].round().astype(int).unique()) != {current_limit}:
            structural_errors.append("电流档与文件名不一致")
        if set(configured_df["I20"].round().astype(int).unique()) != {expected_id}:
            structural_errors.append("控制器编号与文件名不一致")
        if not np.allclose(configured_df["I28"], 2.0, atol=1e-4):
            structural_errors.append("载荷不是2kg")
        if int(round(df["I2"].iloc[-1])) != 6:
            structural_errors.append("最终未到phase6")

        mean_pos = 0.5 * (df["I5"] + df["I6"])
        tracking_mm = 1000.0 * (mean_pos - df["I3"])
        sync_mm = 1000.0 * (df["I5"] - df["I6"])
        active_mask = df["I2"].between(2, 5)
        active = df.loc[active_mask]
        any_sat = (df["I17"].abs() > 0.5) | (df["I18"].abs() > 0.5)
        any_alloc = (df["I25"].abs() > 0.5) | (df["I26"].abs() > 0.5)
        active_sat = any_sat.loc[active_mask]
        active_alloc = any_alloc.loc[active_mask]
        active_dt = df.loc[active_mask, "I0"].diff().fillna(median_dt).clip(lower=0, upper=0.05)
        applied_tv = float(active["I13"].diff().abs().sum() + active["I14"].diff().abs().sum())
        current_effort = float(
            np.sum((active["I13"].abs() + active["I14"].abs()).to_numpy() * active_dt.to_numpy())
        )
        phase6 = df[df["I2"].round().eq(6)]
        final_row = phase6.iloc[0]
        final_mean = 0.5 * (final_row["I5"] + final_row["I6"])
        final_tracking_mm = 1000.0 * (final_mean - final_row["I3"])
        final_sync_mm = 1000.0 * (final_row["I5"] - final_row["I6"])

        metrics = {
            "file": path.name,
            "condition_code": f"{code}_I{current_limit}",
            "controller_code": code,
            "controller": controller,
            "controller_id": expected_id,
            "sync_alloc": int("_SA" in code),
            "dynamic_aw": int(code.startswith("C2B")),
            "current_limit": current_limit,
            "run": run,
            "source_rows": len(raw_df),
            "rows": len(df),
            "selected_start_row_zero_based": selected_start_row,
            "selected_end_row_zero_based": selected_end_row,
            "discarded_prefix_rows": discarded_prefix_rows,
            "discarded_suffix_rows": discarded_suffix_rows,
            "duration_s": float(df["I0"].iloc[-1] - df["I0"].iloc[0]),
            "median_dt_ms": 1000.0 * median_dt,
            "max_dt_ms": 1000.0 * max_dt,
            "seq_consecutive_pct": 100.0 * seq_pct,
            "tracking_rmse_mm": rms(tracking_mm.loc[active_mask]),
            "tracking_max_abs_mm": float(tracking_mm.loc[active_mask].abs().max()),
            "sync_rms_mm": rms(sync_mm.loc[active_mask]),
            "sync_max_abs_mm": float(sync_mm.loc[active_mask].abs().max()),
            "saturation_fraction_pct": 100.0 * float(active_sat.mean()),
            "saturation_duration_s": float(np.sum(active_dt.to_numpy() * active_sat.to_numpy(dtype=float))),
            "allocator_fraction_pct": 100.0 * float(active_alloc.mean()),
            "recovery_time_s": recovery_time(df, active_mask),
            "control_total_variation_count": applied_tv,
            "current_effort_count_s": current_effort,
            "request_peak_count": float(active[["I9", "I10"]].abs().to_numpy().max()),
            "applied_peak_count": float(active[["I13", "I14"]].abs().to_numpy().max()),
            "feedback_peak_count": float(active[["I15", "I16"]].abs().to_numpy().max()),
            "z_aw_peak_count": float(active[["I23", "I24"]].abs().to_numpy().max()),
            "delta_alloc_peak_count": float(active[["I25", "I26"]].abs().to_numpy().max()),
            "battery_mean_V": float(df["I27"].mean()),
            "battery_min_V": float(df["I27"].min()),
            "battery_max_V": float(df["I27"].max()),
            "final_tracking_error_mm": float(final_tracking_mm),
            "final_sync_error_mm": float(final_sync_mm),
            "fault_max": int(round(df["I21"].abs().max())),
            "valid": len(structural_errors) == 0,
            "issues": "; ".join(structural_errors),
            "mtime": path.stat().st_mtime,
        }
        rows.append(metrics)
        quality.append({
            "file": path.name,
            "valid": metrics["valid"],
            "issues": metrics["issues"],
            "source_rows": len(raw_df),
            "rows": len(df),
            "discarded_prefix_rows": discarded_prefix_rows,
            "discarded_suffix_rows": discarded_suffix_rows,
            "columns": len(df.columns),
            "seq_consecutive_pct": metrics["seq_consecutive_pct"],
            "median_dt_ms": metrics["median_dt_ms"],
            "max_dt_ms": metrics["max_dt_ms"],
            "phase_path": "→".join(str(x) for x in phase_path),
            "fault_max": metrics["fault_max"],
            "controller_id": expected_id,
            "current_limit": current_limit,
            "load_kg": 2.0,
            "final_tracking_error_mm": metrics["final_tracking_error_mm"],
            "final_sync_error_mm": metrics["final_sync_error_mm"],
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

    metrics_df = pd.DataFrame(rows).sort_values(["controller_id", "current_limit", "run"])
    quality_df = pd.DataFrame(quality).sort_values("file")

    name_complete = actual_names == expected_names
    all_valid = bool(name_complete and len(metrics_df) == 40 and metrics_df["valid"].all())

    if len(metrics_df):
        actual_order_df = metrics_df.sort_values("mtime").copy()
        actual_order = actual_order_df["condition_code"].tolist()
        random_order_followed = actual_order == EXPECTED_ORDER
        actual_order_df.insert(0, "actual_sequence", range(1, len(actual_order_df) + 1))
        actual_order_df["expected_condition"] = EXPECTED_ORDER[: len(actual_order_df)]
        actual_order_df["order_match"] = actual_order_df["condition_code"] == actual_order_df["expected_condition"]
    else:
        actual_order_df = pd.DataFrame()
        random_order_followed = False

    metric_columns = [
        "tracking_rmse_mm", "tracking_max_abs_mm", "sync_rms_mm", "sync_max_abs_mm",
        "saturation_fraction_pct", "saturation_duration_s", "allocator_fraction_pct",
        "recovery_time_s", "control_total_variation_count", "current_effort_count_s",
        "request_peak_count", "applied_peak_count", "feedback_peak_count", "z_aw_peak_count",
        "delta_alloc_peak_count", "battery_mean_V", "battery_min_V",
        "final_tracking_error_mm", "final_sync_error_mm",
    ]
    summary_rows: list[dict] = []
    for (code, limit), group in metrics_df.groupby(["controller_code", "current_limit"], sort=False):
        controller = CONTROLLERS[code][0]
        out = {
            "controller_code": code,
            "controller": controller,
            "current_limit": int(limit),
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
    summary_df = pd.DataFrame(summary_rows).sort_values(["controller_code", "current_limit"], ascending=[True, False])

    effect_rows: list[dict] = []
    for base in ["C2A", "C2B"]:
        for limit in [2400, 800]:
            plain = summary_df[(summary_df.controller_code == base) & (summary_df.current_limit == limit)].iloc[0]
            alloc = summary_df[(summary_df.controller_code == f"{base}_SA") & (summary_df.current_limit == limit)].iloc[0]
            for metric in ["tracking_rmse_mm", "sync_rms_mm", "sync_max_abs_mm", "saturation_fraction_pct", "control_total_variation_count"]:
                p = float(plain[f"{metric}_mean"])
                a = float(alloc[f"{metric}_mean"])
                effect_rows.append({
                    "comparison": f"{base}: SyncAlloc vs independent",
                    "current_limit": limit,
                    "metric": metric,
                    "baseline_mean": p,
                    "modified_mean": a,
                    "absolute_change": a - p,
                    "relative_change_pct": 100.0 * (a - p) / p if p != 0 else math.nan,
                })
    effects_df = pd.DataFrame(effect_rows)

    # Protocol-order and drift checks. The timestamps show the actual acquisition order.
    if len(actual_order_df):
        actual_order_df["acquisition_index"] = np.arange(1, len(actual_order_df) + 1)
        battery_range = float(metrics_df["battery_mean_V"].max() - metrics_df["battery_mean_V"].min())
        battery_corr = float(actual_order_df["acquisition_index"].corr(actual_order_df["battery_mean_V"], method="spearman"))
    else:
        battery_range = math.nan
        battery_corr = math.nan

    protocol = {
        "expected_file_count": 40,
        "actual_file_count": len(raw_files),
        "missing_files": sorted(expected_names - actual_names),
        "unexpected_files": sorted(actual_names - expected_names),
        "all_files_valid": all_valid,
        "randomized_order_followed": random_order_followed,
        "actual_collection_pattern": "condition-blocked (five repeats per condition)" if not random_order_followed else "pre-randomized block order",
        "battery_mean_range_V": battery_range,
        "battery_spearman_vs_order": battery_corr,
        "archive_decision": "eligible_with_protocol_deviation" if all_valid else "not_eligible",
    }

    # Save analysis artifacts.
    metrics_df.drop(columns=["mtime"]).to_csv(OUT_DIR / "03_统计汇总" / "实验3逐次指标.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(OUT_DIR / "03_统计汇总" / "实验3逐文件质量检查.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "03_统计汇总" / "实验3分组统计.csv", index=False, encoding="utf-8-sig")
    effects_df.to_csv(OUT_DIR / "03_统计汇总" / "实验3效应对比.csv", index=False, encoding="utf-8-sig")
    actual_order_df.drop(columns=["mtime"], errors="ignore").to_csv(
        OUT_DIR / "03_统计汇总" / "实验3实际采集顺序.csv", index=False, encoding="utf-8-sig"
    )
    (OUT_DIR / "03_统计汇总" / "实验3协议与质量结论.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for name, clean in cleaned_frames.items():
        clean.to_csv(OUT_DIR / "02_处理后数据" / name.replace(".csv", "_clean.csv"), index=False, encoding="utf-8-sig")

    for path in raw_files:
        shutil.copy2(path, OUT_DIR / "01_原始数据" / path.name)

    if len(summary_df):
        make_figures(metrics_df, summary_df, cleaned_frames)

    print(json.dumps({
        "files": len(raw_files),
        "valid_files": int(metrics_df["valid"].sum()) if len(metrics_df) else 0,
        "all_valid": all_valid,
        "randomized_order_followed": random_order_followed,
        "battery_mean_range_V": battery_range,
        "battery_spearman_vs_order": battery_corr,
        "invalid": quality_df.loc[~quality_df["valid"], ["file", "issues"]].to_dict("records") if len(quality_df) else [],
    }, ensure_ascii=False, indent=2))
    return 0 if all_valid else 2


def make_figures(metrics: pd.DataFrame, summary: pd.DataFrame, cleaned_frames: dict[str, pd.DataFrame]) -> None:
    out = OUT_DIR / "04_论文图"
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    order = ["C2A", "C2A_SA", "C2B", "C2B_SA"]
    labels = ["C2a", "C2a + SA", "C2b", "C2b + SA"]
    colors = {2400: "#4C78A8", 800: "#E45756"}
    x = np.arange(len(order))
    width = 0.34

    def grouped_bars(ax, metric: str, ylabel: str, title: str) -> None:
        for idx, limit in enumerate([2400, 800]):
            means = []
            stds = []
            for code in order:
                row = summary[(summary.controller_code == code) & (summary.current_limit == limit)].iloc[0]
                means.append(row[f"{metric}_mean"])
                stds.append(row[f"{metric}_std"])
            shift = (-0.5 if idx == 0 else 0.5) * width
            ax.bar(x + shift, means, width, yerr=stds, capsize=3, color=colors[limit], label=str(limit))
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    grouped_bars(axes[0], "tracking_rmse_mm", "RMSE (mm)", "Tracking error")
    grouped_bars(axes[1], "sync_rms_mm", "RMS (mm)", "Synchronization error")
    axes[1].legend(title="Current limit", frameon=False, loc="upper right")
    fig.savefig(out / "图1_跟踪与同步性能.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    grouped_bars(axes[0], "saturation_fraction_pct", "Active samples (%)", "Saturation or allocation activity")
    grouped_bars(axes[1], "control_total_variation_count", "Total variation (count)", "Control variation")
    axes[1].legend(title="Current limit", frameon=False, loc="upper right")
    fig.savefig(out / "图2_饱和与控制量总变差.png", bbox_inches="tight")
    plt.close(fig)

    # Representative 800-count run: nearest sync RMS to each group median.
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex=True, constrained_layout=True)
    for ax, code, label in zip(axes.flat, order, labels):
        group = metrics[(metrics.controller_code == code) & (metrics.current_limit == 800)].copy()
        target = float(group["sync_rms_mm"].median())
        row = group.iloc[(group["sync_rms_mm"] - target).abs().argmin()]
        clean = cleaned_frames[str(row["file"])]
        mask = clean["phase"].between(1, 6)
        ax.plot(clean.loc[mask, "time_rel_s"], clean.loc[mask, "tracking_error_mm"], label="tracking", lw=1.0)
        ax.plot(clean.loc[mask, "time_rel_s"], clean.loc[mask, "sync_error_mm"], label="sync", lw=1.0)
        ax.axhline(1.0, color="#999999", ls="--", lw=0.7)
        ax.axhline(-1.0, color="#999999", ls="--", lw=0.7)
        ax.set_title(f"{label}, run {int(row['run']):02d}")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Error (mm)")
    axes[1, 0].set_ylabel("Error (mm)")
    axes[1, 0].set_xlabel("Time from forward start (s)")
    axes[1, 1].set_xlabel("Time from forward start (s)")
    axes[0, 1].legend(frameon=False, ncol=2)
    fig.savefig(out / "图3_低限流代表时序.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    grouped_bars(axes[0], "delta_alloc_peak_count", "Peak (count)", "Allocation deficit")
    grouped_bars(axes[1], "z_aw_peak_count", "Peak (count)", "Dynamic anti-windup state")
    axes[1].legend(title="Current limit", frameon=False, loc="upper right")
    fig.savefig(out / "图4_分配与动态抗饱和状态.png", bbox_inches="tight")
    plt.close(fig)

    actual = metrics.sort_values("mtime").copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)
    axes[0].plot(np.arange(1, len(actual) + 1), actual["battery_mean_V"], marker="o", ms=2.5, lw=0.8)
    axes[0].set_xlabel("Acquisition index")
    axes[0].set_ylabel("Mean battery voltage (V)")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(actual["median_dt_ms"], actual["seq_consecutive_pct"], s=16, alpha=0.75)
    axes[1].set_xlabel("Median sample interval (ms)")
    axes[1].set_ylabel("Consecutive I1 samples (%)")
    axes[1].grid(alpha=0.25)
    fig.savefig(out / "图5_采集质量与电压漂移.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
