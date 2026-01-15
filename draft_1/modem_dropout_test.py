"""Modem dropout/switching sanity test.

Example:
    python modem_dropout_test.py --seed 0 --duration-sec 180 --trajectory spiral \
        --targets usv1 usv2 usv3 usv4 --dropout "usv2:30-60;usv4:90-140"
"""
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import matplotlib
if os.environ.get("MPLBACKEND") is None:
    if os.environ.get("DISPLAY"):
        # Use default interactive backend when a display is present
        pass
    else:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from current_acoustic_EKF_patched import run_single_trial


COLORS = {
    "usv1": "#1f77b4",
    "usv2": "#ff7f0e",
    "usv3": "#2ca02c",
    "usv4": "#d62728",
}

DISPLAY_NAMES = {
    "usv1": "Acoustic 1",
    "usv2": "Acoustic 2",
    "usv3": "Acoustic 3",
    "usv4": "Acoustic 4",
}


def pretty_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def parse_dropout(spec: str) -> Dict[str, List[Tuple[float, float]]]:
    """Parse dropout string like "usv2:30-60,120-150;usv3:80-110"."""
    if not spec:
        return {}
    out: Dict[str, List[Tuple[float, float]]] = {}
    for block in spec.split(";"):
        block = block.strip()
        if not block or ":" not in block:
            continue
        name, spans = block.split(":", 1)
        name = name.strip()
        if not name:
            continue
        intervals: List[Tuple[float, float]] = []
        for span in spans.split(","):
            span = span.strip()
            if not span or "-" not in span:
                continue
            a_str, b_str = span.split("-", 1)
            try:
                a, b = float(a_str), float(b_str)
            except ValueError:
                continue
            if b < a:
                a, b = b, a
            intervals.append((a, b))
        if intervals:
            out[name] = intervals
    return out


def merge_intervals(intervals: List[Tuple[float, float]]):
    """Merge overlapping intervals; input/output in seconds."""
    if not intervals:
        return []
    spans = sorted([(float(a), float(b)) if float(a) <= float(b) else (float(b), float(a)) for a, b in intervals], key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_a, cur_b = spans[0]
    for a, b in spans[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return merged


def normalize_dropout_intervals(dropout: Dict[str, List[Tuple[float, float]]]):
    return {k: merge_intervals(v) for k, v in (dropout or {}).items()}


def default_dropout_schedule(duration_sec: float) -> Dict[str, List[Tuple[float, float]]]:
    """Build a canned dropout schedule that hits all modems with overlaps.

    The schedule has three phases:
    1) Staggered pair overlaps early in the run.
    2) A rolling triple-overlap window in the middle.
    3) A short interval where *all* modems are down simultaneously.
    """

    d = float(duration_sec)

    def span(a: float, b: float) -> Tuple[float, float]:
        lo = max(0.0, min(d, a))
        hi = max(0.0, min(d, b))
        return (lo, hi) if lo <= hi else (hi, lo)

    early = {
        # Pairwise overlaps
        "usv1": [span(0.10 * d, 0.30 * d)],
        "usv2": [span(0.20 * d, 0.40 * d)],
        "usv3": [span(0.28 * d, 0.45 * d)],
        "usv4": [span(0.36 * d, 0.52 * d)],
    }

    mid = {
        # Rolling triple overlap
        "usv1": [span(0.48 * d, 0.60 * d)],
        "usv2": [span(0.52 * d, 0.64 * d)],
        "usv3": [span(0.56 * d, 0.68 * d)],
        "usv4": [span(0.60 * d, 0.72 * d)],
    }

    all_out = {
        # Everyone down together for a short outage
        "usv1": [span(0.78 * d, 0.83 * d)],
        "usv2": [span(0.78 * d, 0.83 * d)],
        "usv3": [span(0.78 * d, 0.83 * d)],
        "usv4": [span(0.78 * d, 0.83 * d)],
    }

    merged: Dict[str, List[Tuple[float, float]]] = {k: [] for k in COLORS.keys()}
    for block in (early, mid, all_out):
        for name, spans in block.items():
            merged[name].extend(spans)

    return normalize_dropout_intervals(merged)


def mask_from_intervals(t: np.ndarray, intervals: List[Tuple[float, float]]):
    mask = np.zeros_like(t, dtype=bool)
    for a, b in intervals:
        mask |= (t >= a) & (t <= b)
    return mask


def build_phase_masks(t: np.ndarray, d2: List[Tuple[float, float]], d4: List[Tuple[float, float]]):
    m1 = np.zeros_like(t, dtype=bool)  # placeholder for dropout1
    m3 = np.zeros_like(t, dtype=bool)  # placeholder for dropout3
    m2 = mask_from_intervals(t, d2)
    m4 = mask_from_intervals(t, d4)
    overlap = m1 | m2 | m3 | m4
    overlap = (m1 & m2) | (m1 & m3) | (m1 & m4) | (m2 & m3) | (m2 & m4) | (m3 & m4)
    d2_only = m2 & (~(m1 | m3 | m4))
    d4_only = m4 & (~(m1 | m2 | m3))
    baseline = ~(m1 | m2 | m3 | m4)
    return {
        "overall": np.ones_like(t, dtype=bool),
        "baseline": baseline,
        "dropout1_only": m1,
        "dropout2_only": d2_only,
        "dropout3_only": m3,
        "dropout4_only": d4_only,
        "overlap": overlap,
    }


def downsample_for_plot(t: np.ndarray, y: np.ndarray, target_hz: float = 5.0):
    if t.size == 0:
        return t, y
    dt = np.median(np.diff(t)) if t.size > 1 else 0.0
    if dt <= 0:
        return t, y
    step = max(1, int(round((1.0 / target_hz) / dt)))
    return t[::step], y[::step]


def rolling_median_time(t: np.ndarray, y: np.ndarray, window_sec: float = 10.0):
    """Time-based rolling median (trailing window). Uses pandas if available, else two-heaps."""
    try:
        import pandas as pd  # type: ignore

        idx = pd.to_timedelta(t, unit="s")
        ser = pd.Series(y, index=idx)
        med = ser.rolling(f"{float(window_sec)}s", min_periods=1).median()
        return med.to_numpy()
    except Exception:
        pass

    # Fallback: two-heap sliding window (O(n log n))
    import heapq

    low: List[Tuple[float, int]] = []   # max-heap via -value
    high: List[Tuple[float, int]] = []  # min-heap
    loc = {}
    invalid_low = defaultdict(int)
    invalid_high = defaultdict(int)
    size_low = size_high = 0

    def prune(heap, invalid):
        while heap:
            _, idx = heap[0]
            if invalid[idx] > 0:
                invalid[idx] -= 1
                heapq.heappop(heap)
            else:
                break

    def rebalance():
        nonlocal size_low, size_high
        # ensure size_low >= size_high and diff <= 1
        while size_low > size_high + 1:
            val, idx = heapq.heappop(low)
            size_low -= 1
            heapq.heappush(high, (-val, idx))
            size_high += 1
            loc[idx] = "high"
        while size_high > size_low:
            val, idx = heapq.heappop(high)
            size_high -= 1
            heapq.heappush(low, (-val, idx))
            size_low += 1
            loc[idx] = "low"

    def add(val: float, idx: int):
        nonlocal size_low, size_high
        if not low or val <= -low[0][0]:
            heapq.heappush(low, (-val, idx))
            size_low += 1
            loc[idx] = "low"
        else:
            heapq.heappush(high, (val, idx))
            size_high += 1
            loc[idx] = "high"
        rebalance()

    def remove(idx: int):
        nonlocal size_low, size_high
        side = loc.pop(idx, None)
        if side is None:
            return
        if side == "low":
            invalid_low[idx] += 1
            size_low -= 1
        else:
            invalid_high[idx] += 1
            size_high -= 1
        prune(low, invalid_low)
        prune(high, invalid_high)
        rebalance()
        prune(low, invalid_low)
        prune(high, invalid_high)

    def current_median():
        prune(low, invalid_low)
        prune(high, invalid_high)
        if size_low >= size_high:
            return -low[0][0]
        return high[0][0]

    med_out = np.empty_like(y, dtype=float)
    start = 0
    for i in range(t.size):
        add(float(y[i]), i)
        # evict old samples
        while t[i] - t[start] > window_sec and start <= i:
            remove(start)
            start += 1
        med_out[i] = current_median()
    return med_out


def compute_cdf(values: np.ndarray):
    vals = np.sort(values)
    probs = np.linspace(1.0 / vals.size, 1.0, vals.size)
    return vals, probs


def make_error_plots(t_sec: np.ndarray, pos_err: np.ndarray, dropout_2: List[Tuple[float, float]], dropout_4: List[Tuple[float, float]], out_dir: str = "."):
    """Generate position error plots with dropout shading and phase medians."""
    t = np.asarray(t_sec, dtype=float)
    err = np.asarray(pos_err, dtype=float)

    if t.shape != err.shape:
        raise ValueError("t_sec and pos_err must have the same shape")
    if t.size == 0:
        raise ValueError("Empty time series")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(err)):
        raise ValueError("Non-finite values in input")

    d2 = merge_intervals(dropout_2)
    d4 = merge_intervals(dropout_4)

    phases = build_phase_masks(t, d2, d4)

    # Rolling median on full-rate data
    med = rolling_median_time(t, err, window_sec=10.0)

    # Downsample only for raw visualization
    ds_t, ds_err = downsample_for_plot(t, err, target_hz=5.0)

    # Main plot with smoothing
    fig1 = plt.figure(figsize=(8, 4.5))
    ax1 = fig1.gca()
    ax1.plot(ds_t, ds_err, color="tab:blue", alpha=0.5, lw=1.2, label="pos err (raw, downsampled)")
    ax1.plot(t, med, color="tab:red", lw=1.4, label="rolling median (10 s)")

    def shade(ax, intervals, label, color):
        seen = set()
        for (a, b) in intervals:
            lbl = label if label not in seen else None
            ax.axvspan(a, b, color=color, alpha=0.12, label=lbl)
            seen.add(label)

    shade(ax1, d2, "dropout 2", COLORS.get("usv2", "gray"))
    shade(ax1, d4, "dropout 4", COLORS.get("usv4", "silver"))

    # Phase medians (horizontal lines across full width)
    median_lines = {}
    x_min, x_max = float(t.min()), float(t.max())
    for phase_name, mask in phases.items():
        vals = err[mask]
        if vals.size == 0:
            continue
        median_lines[phase_name] = float(np.median(vals))
        ax1.hlines(median_lines[phase_name], x_min, x_max, linestyle="--", linewidth=1.1, label=f"median {phase_name}")

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("position error [m]")
    ax1.set_title("Position error with rolling median")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(Path(out_dir) / "position_error_main.png", dpi=300)

    # Raw plot (downsampled only)
    fig2 = plt.figure(figsize=(8, 4.0))
    ax2 = fig2.gca()
    ax2.plot(ds_t, ds_err, color="tab:blue", lw=1.2, label="pos err (raw, downsampled)")
    shade(ax2, d2, "dropout 2", COLORS.get("usv2", "gray"))
    shade(ax2, d4, "dropout 4", COLORS.get("usv4", "silver"))
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position error [m]")
    ax2.set_title("Position error (raw)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(Path(out_dir) / "position_error_raw.png", dpi=300)

    # CDF plot
    fig3 = plt.figure(figsize=(6, 4.5))
    ax3 = fig3.gca()

    def add_cdf(name: str, mask: np.ndarray, color: str):
        vals = err[mask]
        if vals.size < 5:
            print(f"[WARN] Phase '{name}' has too few samples ({vals.size}); skipping CDF")
            return
        xs, ps = compute_cdf(vals)
        ax3.plot(xs, ps, label=f"{name} (n={vals.size})", lw=1.3, color=color)

    colors = {
        "overall": "black",
        "baseline": "tab:green",
        "dropout1_only": "tab:cyan",
        "dropout2_only": "tab:orange",
        "dropout3_only": "tab:gray",
        "dropout4_only": "tab:purple",
        "overlap": "tab:red",
    }
    for phase, mask in phases.items():
        add_cdf(phase, mask, colors.get(phase, None) or None)

    ax3.set_xlabel("position error [m]")
    ax3.set_ylabel("empirical CDF")
    ax3.set_title("Position error CDF by phase")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(Path(out_dir) / "position_error_cdf.png", dpi=300)

    # Print stats
    def print_stats(label: str, mask: np.ndarray):
        vals = err[mask]
        if vals.size == 0:
            print(f"[WARN] No samples for phase '{label}'")
            return
        med = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p95 = np.percentile(vals, 95)
        print(f"{label}: median={med:.3f} m, p90={p90:.3f} m, p95={p95:.3f} m (n={vals.size})")

    for label, mask in phases.items():
        print_stats(label, mask)

    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def is_enabled(dropout: Dict[str, List[Tuple[float, float]]], name: str, t_sec: float) -> bool:
    spans = dropout.get(name)
    if not spans:
        return True
    for a, b in spans:
        if a <= t_sec <= b:
            return False
    return True


def _set_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.grid": True,
        "axes.facecolor": "#f8f8f8",
        "grid.alpha": 0.4,
        "font.size": 10,
    })


def write_timeseries_csv(path: Path, ts: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]]) -> None:
    t = np.asarray(ts.get("t", []), dtype=float)
    gt = np.asarray(ts.get("true_pos", []), dtype=float)
    est = np.asarray(ts.get("est_pos", []), dtype=float)
    err = np.asarray(ts.get("err_norm", []), dtype=float)
    Ppos = np.asarray(ts.get("Ppos", []), dtype=float)
    nees_pos = np.asarray(ts.get("nees_pos", []), dtype=float)
    nees_full = np.asarray(ts.get("nees_full", []), dtype=float)
    nis_logs = ts.get("nis_logs", {}) or {}
    acoustic_used_t = np.asarray(nis_logs.get("acoustic", {}).get("t", []), dtype=float)
    acoustic_used_v = np.asarray(nis_logs.get("acoustic", {}).get("values", []), dtype=float)
    acoustic_skip_t = np.asarray(nis_logs.get("acoustic_skipped", {}).get("t", []), dtype=float)
    acoustic_skip_v = np.asarray(nis_logs.get("acoustic_skipped", {}).get("values", []), dtype=float)

    def nis_at(time_array, val_array):
        out = np.full_like(t, np.nan, dtype=float)
        for tt, vv in zip(time_array, val_array):
            idx = int(np.searchsorted(t, tt))
            if 0 <= idx < out.size:
                out[idx] = vv
        return out

    nis_used = nis_at(acoustic_used_t, acoustic_used_v)
    nis_skip = nis_at(acoustic_skip_t, acoustic_skip_v)

    usv_names = ["usv1", "usv2", "usv3", "usv4"]
    enabled_flags = {n: np.array([is_enabled(dropout, n, float(tt)) for tt in t], dtype=bool) for n in usv_names}

    fields = [
        "t", "true_x", "true_y", "true_z", "est_x", "est_y", "est_z", "pos_err_norm",
        "P_xx", "P_yy", "P_zz", "nees_pos", "nees_full", "nis_used", "nis_skipped",
    ] + [f"{n}_enabled" for n in usv_names]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(t.size):
            row = {
                "t": float(t[i]),
                "true_x": float(gt[i, 0]) if gt.size else np.nan,
                "true_y": float(gt[i, 1]) if gt.size else np.nan,
                "true_z": float(gt[i, 2]) if gt.size else np.nan,
                "est_x": float(est[i, 0]) if est.size else np.nan,
                "est_y": float(est[i, 1]) if est.size else np.nan,
                "est_z": float(est[i, 2]) if est.size else np.nan,
                "pos_err_norm": float(err[i]) if err.size else np.nan,
                "P_xx": float(Ppos[i, 0, 0]) if Ppos.size else np.nan,
                "P_yy": float(Ppos[i, 1, 1]) if Ppos.size else np.nan,
                "P_zz": float(Ppos[i, 2, 2]) if Ppos.size else np.nan,
                "nees_pos": float(nees_pos[i]) if nees_pos.size else np.nan,
                "nees_full": float(nees_full[i]) if nees_full.size else np.nan,
                "nis_used": float(nis_used[i]) if np.isfinite(nis_used[i]) else np.nan,
                "nis_skipped": float(nis_skip[i]) if np.isfinite(nis_skip[i]) else np.nan,
            }
            for n in usv_names:
                row[f"{n}_enabled"] = bool(enabled_flags[n][i]) if enabled_flags[n].size else True
            writer.writerow(row)


def _shade(ax, dropout: Dict[str, List[Tuple[float, float]]]):
    for name, spans in dropout.items():
        color = COLORS.get(name, "gray")
        for a, b in spans:
            ax.axvspan(a, b, color=color, alpha=0.15, label=f"{pretty_name(name)} dropout")


def plot_pos_err(path: Path, ts: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]]):
    t = ts["t"]
    err = ts["err_norm"]
    plt.figure(figsize=(7, 4))
    plt.plot(t, err, lw=1.5, label="||pos error||")
    _shade(plt.gca(), dropout)
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title("Position error vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "pos_err_vs_time.png")
    plt.close()


def plot_uncertainty(path: Path, ts: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]]):
    t = ts["t"]
    Ppos = ts["Ppos"]
    traceP = np.einsum("nii->n", Ppos)
    sig = np.sqrt(np.maximum(traceP, 0.0))
    plt.figure(figsize=(7, 4))
    plt.plot(t, sig, lw=1.5, label="sqrt(trace(Ppos)")
    _shade(plt.gca(), dropout)
    plt.xlabel("time [s]")
    plt.ylabel("uncertainty [m]")
    plt.title("Position uncertainty proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "uncertainty_vs_time.png")
    plt.close()


def plot_nis(path: Path, ts: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]]):
    nis_logs = ts.get("nis_logs", {}) or {}
    used_t = nis_logs.get("acoustic", {}).get("t", [])
    used_v = nis_logs.get("acoustic", {}).get("values", [])
    skip_t = nis_logs.get("acoustic_skipped", {}).get("t", [])
    skip_v = nis_logs.get("acoustic_skipped", {}).get("values", [])

    plt.figure(figsize=(7, 4))
    if used_t:
        plt.plot(used_t, used_v, lw=1.2, label="NIS acoustic (used)")
    if skip_t:
        plt.scatter(skip_t, skip_v, s=12, alpha=0.5, label="NIS acoustic (skipped)", color="gray")
    lo = chi2.ppf(0.025, 1)
    hi = chi2.ppf(0.975, 1)
    plt.axhline(lo, color="gray", linestyle="--", linewidth=1, label="95% bounds")
    plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
    _shade(plt.gca(), dropout)
    plt.xlabel("time [s]")
    plt.ylabel("NIS")
    plt.title("Acoustic NIS vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "nis_acoustic_vs_time.png")
    plt.close()


def plot_xy(path: Path, ts: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]]):
    gt = ts["true_pos"]
    est = ts["est_pos"]
    plt.figure(figsize=(6, 6))
    plt.plot(gt[:, 0], gt[:, 1], label="Ground truth", lw=2)
    plt.plot(est[:, 0], est[:, 1], label="Estimate", lw=1.5)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.title("XY trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "xy_traj.png")
    plt.close()


def save_summary(path: Path, trial: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]], duration: float):
    dur = float(duration)
    disabled_seconds = {k: sum(max(0.0, min(dur, b) - max(0.0, a)) for a, b in v) for k, v in dropout.items()}
    summary = {
        "seed": trial.get("seed"),
        "pos_rmse_total": trial.get("pos_rmse_total"),
        "vel_rmse_total": trial.get("vel_rmse_total"),
        "final_position_error": trial.get("final_position_error"),
        "runtime_seconds": trial.get("runtime_seconds"),
        "dropout_intervals": dropout,
        "disabled_seconds": disabled_seconds,
        "config": trial.get("config"),
        "nees": trial.get("nees_full"),
        "nis_acoustic": trial.get("nis", {}).get("acoustic"),
    }
    with (path / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def summarize_results(ts: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]], duration_sec: float, targets: List[str]):
    """Print key metrics for paper-ready reporting."""

    t = np.asarray(ts.get("t", []), dtype=float)
    err = np.asarray(ts.get("err_norm", []), dtype=float)
    Ppos = np.asarray(ts.get("Ppos", []), dtype=float)
    nees_pos = np.asarray(ts.get("nees_pos", []), dtype=float)
    nees_full = np.asarray(ts.get("nees_full", []), dtype=float)
    nis_logs = ts.get("nis_logs", {}) or {}
    used_nis = np.asarray(nis_logs.get("acoustic", {}).get("values", []), dtype=float)
    skipped_nis = np.asarray(nis_logs.get("acoustic_skipped", {}).get("values", []), dtype=float)

    print("\n=== Key results ===")

    if err.size:
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        stats = np.nanpercentile(err, [50, 90, 95])
        max_err = float(np.nanmax(err)) if np.isfinite(err).any() else float("nan")
        print(f"Position error: rmse={rmse:.3f} m, median={stats[0]:.3f} m, p90={stats[1]:.3f} m, p95={stats[2]:.3f} m, max={max_err:.3f} m")

    if Ppos.size:
        traceP = np.einsum("nii->n", Ppos)
        sig = np.sqrt(np.maximum(traceP, 0.0))
        sig_stats = np.nanpercentile(sig, [50, 90, 95])
        print(f"Pos uncertainty sqrt(trace(Ppos)): median={sig_stats[0]:.3f} m, p90={sig_stats[1]:.3f} m, p95={sig_stats[2]:.3f} m")

    if nees_pos.size:
        nees_pos_stats = np.nanpercentile(nees_pos, [50, 90, 95])
        mean_nees = float(np.nanmean(nees_pos))
        expected = 3.0  # pos DOF
        print(f"NEES(pos, dof~3): mean={mean_nees:.3f}, median={nees_pos_stats[0]:.3f}, p90={nees_pos_stats[1]:.3f}, p95={nees_pos_stats[2]:.3f}, mean/expected={mean_nees/expected:.3f}")

    if nees_full.size:
        nf_stats = np.nanpercentile(nees_full, [50, 90, 95])
        mean_nf = float(np.nanmean(nees_full))
        print(f"NEES(full): mean={mean_nf:.3f}, median={nf_stats[0]:.3f}, p90={nf_stats[1]:.3f}, p95={nf_stats[2]:.3f}")

    total_meas = used_nis.size + skipped_nis.size
    if total_meas:
        used_frac = used_nis.size / total_meas
        print(f"Acoustic gating: used {used_nis.size}, skipped {skipped_nis.size}, used_frac={used_frac:.3f}")
        if used_nis.size:
            u_stats = np.nanpercentile(used_nis, [50, 90, 95])
            print(f"NIS(acoustic used): median={u_stats[0]:.3f}, p90={u_stats[1]:.3f}, p95={u_stats[2]:.3f}")

    dur = float(duration_sec)
    for name in targets or []:
        spans = dropout.get(name, [])
        off = sum(max(0.0, min(dur, b) - max(0.0, a)) for a, b in spans)
        uptime = max(0.0, dur - off)
        print(f"{pretty_name(name)} uptime={uptime:.1f} s ({uptime/dur*100.0:.1f}%), downtime={off:.1f} s")

def main():
    parser = argparse.ArgumentParser(description="Modem dropout switching test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=180.0)
    parser.add_argument("--trajectory", choices=["lawnmower", "spiral", "concentric", "figure8"], default="spiral")
    parser.add_argument("--targets", nargs="*", default=["usv1", "usv2", "usv3", "usv4"], choices=["usv1", "usv2", "usv3", "usv4"], help="Subset of modems to include")
    parser.add_argument("--dropout", type=str, default="", help="Dropout schedule e.g. 'usv2:30-60;usv3:80-110'")
    parser.add_argument("--use-default-overlaps", action="store_true", help="Use built-in overlap schedule covering all modems (includes full outage)")
    args = parser.parse_args()

    _set_style()
    if args.use_default_overlaps and args.dropout:
        parser.error("Use either --dropout or --use-default-overlaps, not both")

    if args.use_default_overlaps:
        dropout = default_dropout_schedule(args.duration_sec)
    else:
        dropout = normalize_dropout_intervals(parse_dropout(args.dropout))

    out_dir = Path("results_modem_dropout") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "enable_dvl": True,
        "enable_depth": True,
        "enable_acoustic": True,
        "use_all_acoustic": False,
        "use_currents": False,
        "duration_sec": float(args.duration_sec),
        "trajectory": args.trajectory,
        "modem_dropout_intervals": dropout,
        "modem_dropout_mode": "ignore_updates",
    }

    trial = run_single_trial(
        seed=args.seed,
        config_overrides=overrides,
        return_timeseries=True,
        target_names=args.targets,
    )

    ts = trial.get("timeseries", {})
    write_timeseries_csv(out_dir / "timeseries.csv", ts, dropout)
    make_error_plots(
        t_sec=np.asarray(ts.get("t", []), dtype=float),
        pos_err=np.asarray(ts.get("err_norm", []), dtype=float),
        dropout_2=dropout.get("usv2", []),
        dropout_4=dropout.get("usv4", []),
        out_dir=plots_dir,
    )
    plot_uncertainty(plots_dir, ts, dropout)
    plot_xy(plots_dir, ts, dropout)
    save_summary(out_dir, trial, dropout, args.duration_sec)
    summarize_results(ts, dropout, args.duration_sec, args.targets)

    print(f"pos_rmse_total: {trial.get('pos_rmse_total'):.3f} m")
    print(f"final_position_error: {trial.get('final_position_error'):.3f} m")
    for name, spans in dropout.items():
        total_off = sum(max(0.0, min(args.duration_sec, b) - max(0.0, a)) for a, b in spans)
        print(f"{pretty_name(name)} disabled for {total_off:.1f} s")


if __name__ == "__main__":
    main()
