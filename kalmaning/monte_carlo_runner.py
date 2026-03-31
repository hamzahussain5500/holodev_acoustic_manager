"""Monte Carlo harness for the HoloOcean EKF.

Definitions:
- NEES (Normalized Estimation Error Squared): consistency of state covariance vs error.
- NIS  (Normalized Innovation Squared): consistency of measurement covariance vs innovations.
- Run-level consistency (this repo): a trial is counted consistent if (a) >=X% of per-sample
    NEES are inside the chi-square gate for dof, and optionally (b) a downsampled mean-NEES
    chi-square test passes. This avoids overly-strict mean-NEES tests on long, correlated runs.
- We avoid using the full-length mean-NEES chi-square alone because sample correlation makes
    the nominal bounds too tight for long trajectories, under-reporting practical consistency.
"""

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from current_acoustic_EKF_patched import run_single_trial
from adaptive_modem_manager_v2 import AdaptiveModemManagerV2
from modem_switching_validation_fixed import (
    DEFAULT_PHASE_WEIGHTS,
    GeometryPolicySelector,
    PolicyParams,
    WeightedParams,
    WeightedPolicySelector,
    parse_phase_schedule,
)
from validation_metrics import downsampled_mean_nees_test  # noqa: F401

DEFAULT_SEEDS = [
    3, 7, 11, 19, 23, 37, 41, 53, 67, 79,
    97, 101, 131, 173, 211, 257, 311, 419, 509, 631,
]

CONFIG_PRESETS = {
    "imu_dvl_depth": {
        "enable_dvl": True,
        "enable_depth": True,
        "enable_acoustic": False,
        "use_currents": False,
        "use_all_acoustic": False,
    },
    "imu_dvl_depth_acoustic_all": {
        "enable_dvl": True,
        "enable_depth": True,
        "enable_acoustic": True,
        "use_currents": False,
        "use_all_acoustic": True,
    },
}


DEFAULT_MC_CONFIG: Dict[str, Any] = {
    "algorithms": ["imu_dvl_depth", "imu_dvl_depth_all4", "adaptive"],
    "adaptive_policy": "v2",
    "targets": ["usv1", "usv2", "usv3", "usv4"],
    "seeds": None,
    "x0_pos_std": 1.0,
    "x0_vel_std": 0.1,
    "imu_accel_extra_std": 0.0,
    "imu_bias_rw_std": 0.0,
    "dvl_extra_std": 0.0,
    "depth_extra_std": 0.0,
    "range_extra_std": 0.0,
    "max_workers": 2,
    "sigma_r": 0.5,
    "gdop_xy": 15.0,
    "gdop_3d": 15.0,
    "switch_margin": 0.0,
    "min_dwell_sec": 1.0,
    "size_penalty": 0.0,
    "score_margin": 0.0,
    "power_save_tol": 0.02,
    "rank_req_xy": 2,
    "rank_req_3d": 3,
    "min_beacons_xy": 2,
    "min_beacons_3d": 3,
    "allow_zero_beacons": False,
    "phase_schedule": None,
    "mission_phase": "cruise",
    "churn_interval_sec": 0.0,
    "churn_score_eps": 0.02,
    "battery_wh": 100.0,
    "base_drain_w": 0.0,
    "beacon_drain_w": 1.0,
    "drain_scale": 1.0,
    "soc_init": 1.0,
    "soc_min": 0.0,
    "low_power_soc": 0.1,
    "target_unc_xy": 5.0,
    "target_unc_3d": 8.0,
    "off_unc_mult": 1.5,
    "energy_weight": 0.0,
    "v2_low_power_soc": 0.2,
    "v2_energy_mult": 2.0,
    "v2_size_penalty_mult": 2.0,
    "v2_rank_deficit_mult": 1.0,
    "v2_rank_deficit_penalty": 10.0,
    "ticks_per_sec": 100.0,
}


def _load_mc_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    if not path:
        raise RuntimeError("Config path is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")
    return data


@dataclass
class RandomStreamsConfig:
    x0_pos_std: float = 1.0
    x0_vel_std: float = 0.1


class RandomStreams:
    def __init__(self, seed: int, cfg: RandomStreamsConfig) -> None:
        self.seed = int(seed)
        self.cfg = cfg

    def build(self, n_steps: int, dt: float) -> Dict[str, Any]:
        ss = np.random.SeedSequence(self.seed)
        rng_pos, rng_vel, rng_imu, rng_bias, rng_dvl, rng_depth, rng_range = [
            np.random.default_rng(s) for s in ss.spawn(7)
        ]

        x0_perturb = {
            "pos": rng_pos.normal(0.0, self.cfg.x0_pos_std, size=3).tolist(),
            "vel": rng_vel.normal(0.0, self.cfg.x0_vel_std, size=3).tolist(),
        }

        noise_streams = {
            "imu_accel": rng_imu.normal(0.0, 1.0, size=(n_steps, 3)),
            "imu_bias_rw": rng_bias.normal(0.0, 1.0, size=(n_steps, 3)),
            "dvl": rng_dvl.normal(0.0, 1.0, size=(n_steps, 3)),
            "depth": rng_depth.normal(0.0, 1.0, size=(n_steps,)),
            "range": rng_range.normal(0.0, 1.0, size=(n_steps,)),
        }

        return {
            "x0_perturb": x0_perturb,
            "noise_streams": noise_streams,
        }


def mean_std_ci(values: List[float], alpha: float = 0.05, bootstrap: Optional[int] = None) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "ci": (np.nan, np.nan)}

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    if bootstrap and arr.size > 1:
        rng = np.random.default_rng(12345)
        boot_means = []
        for _ in range(int(bootstrap)):
            sample = rng.choice(arr, size=arr.size, replace=True)
            boot_means.append(np.mean(sample))
        lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci = (float(lo), float(hi))
    else:
        z = 1.96  # normal 95%
        margin = z * std / math.sqrt(arr.size) if arr.size else np.nan
        ci = (float(mean - margin), float(mean + margin))

    return {"mean": mean, "std": std, "ci": ci}


def final_error_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": np.nan, "p90": np.nan, "p95": np.nan}
    return {
        "median": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def run_consistency_flag(run: Dict[str, Any], percent_threshold: float, avg_band: Optional[Tuple[float, float]], require_ds: bool) -> bool:
    nees = run.get("nees_full", {})
    pct = float(nees.get("percent_inside_bounds", 0.0))
    pct_ok = pct >= percent_threshold

    avg_ok = True
    if avg_band:
        expected = float(nees.get("expected_nees", 0.0) or 0.0)
        avg = float(nees.get("avg_nees", np.inf))
        lo = avg_band[0] * expected
        hi = avg_band[1] * expected
        avg_ok = (avg >= lo) and (avg <= hi)

    ds_ok = True
    ds = run.get("nees_full_ds_mean", None)
    if require_ds:
        ds_ok = bool(ds and ds.get("is_consistent", False))

    return pct_ok and avg_ok and ds_ok


def _set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "axes.facecolor": "#ffffff",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d9d9d9",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "lines.linewidth": 1.2,
    })


ALGO_COLORS: Dict[str, str] = {
    "imu_dvl_depth": "#1f77b4",
    "imu_dvl_depth_all4": "#ff7f0e",
    "imu_dvl_depth_acoustic_all": "#ff7f0e",
    "adaptive": "#2ca02c",
}


def _algo_color(name: str) -> str:
    key = str(name).lower()
    if key in ALGO_COLORS:
        return ALGO_COLORS[key]
    if key.startswith("adaptive"):
        return ALGO_COLORS["adaptive"]
    return "#1f77b4"


def _style_axis(ax: Any, legend_loc: str = "best", show_legend: bool = True) -> None:
    try:
        ax.spines[["top", "right"]].set_visible(False)
    except Exception:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d9d9d9", alpha=0.3, linewidth=0.5)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc=legend_loc, frameon=False, fontsize=10)


def _styled_boxplot(ax: Any, data: List[List[float]], labels: List[str]) -> None:
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops={"color": "#202020", "linewidth": 1.2},
        whiskerprops={"color": "#4d4d4d", "linewidth": 1.0},
        capprops={"color": "#4d4d4d", "linewidth": 1.0},
        boxprops={"color": "#4d4d4d", "linewidth": 1.0},
    )
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(_algo_color(label))
        patch.set_alpha(0.6)


def _save_figure(path_hint: Path) -> None:
    out_pdf = path_hint.with_suffix(".pdf")
    out_svg = path_hint.with_suffix(".svg")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_svg, format="svg", bbox_inches="tight")


def _median_dt(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    return float(np.median(np.diff(times)))


def _energy_from_active_count(times: np.ndarray, active_count: np.ndarray, base_w: float, beacon_w: float) -> float:
    if times.size < 2 or active_count.size == 0:
        return 0.0
    dt = _median_dt(times)
    if dt <= 0:
        return 0.0
    power = base_w + beacon_w * np.clip(active_count.astype(float), 0.0, None)
    return float(np.nansum(power) * dt / 3600.0)


def _percent_time_by_active_count(active_count: np.ndarray) -> Dict[int, float]:
    counts: Dict[int, float] = {k: 0.0 for k in range(5)}
    if active_count.size == 0:
        return counts
    vals = np.asarray(active_count)
    total = float(vals.size)
    for k in range(5):
        counts[k] = float(np.sum(vals == k) / total)
    return counts


def _count_switches(active_sets: Sequence[Any]) -> int:
    if not active_sets:
        return 0
    last = None
    switches = 0
    for cur in active_sets:
        cur_key = tuple(cur) if isinstance(cur, (list, tuple)) else cur
        if last is None:
            last = cur_key
            continue
        if cur_key != last:
            switches += 1
            last = cur_key
    return switches


def _mean_ci(stack: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stack.size == 0:
        return stack, stack, stack
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    n = np.sum(np.isfinite(stack), axis=0)
    n = np.where(n <= 0, np.nan, n)
    se = std / np.sqrt(n)
    lo = mean - z * se
    hi = mean + z * se
    return mean, lo, hi


def _stack_series(series_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    series_list = [s for s in series_list if s is not None and len(s) > 0]
    if not series_list:
        return np.asarray([]), np.asarray([])
    min_len = min(len(s) for s in series_list)
    stack = np.stack([s[:min_len] for s in series_list])
    return stack, np.asarray(range(min_len))


def compute_crlb_efficiency(trial: Dict[str, Any]) -> float:
    ts = trial.get("timeseries", {}) or {}
    Ppos = ts.get("Ppos")
    err = np.asarray(ts.get("err_norm", []), dtype=float)
    if Ppos is None or len(err) == 0:
        return float("nan")
    Ppos_arr = np.asarray(Ppos, dtype=float)
    if Ppos_arr.ndim != 3 or Ppos_arr.shape[1] < 3:
        return float("nan")
    tr = np.trace(Ppos_arr[:, :3, :3], axis1=1, axis2=2)
    crlb = np.sqrt(np.clip(tr, 0.0, None))
    if not np.isfinite(crlb).any():
        return float("nan")
    mean_err = float(np.nanmean(err))
    mean_crlb = float(np.nanmean(crlb))
    if mean_crlb <= 0 or not np.isfinite(mean_crlb):
        return float("nan")
    return mean_err / mean_crlb


def compute_avg_beacons(active_count: np.ndarray) -> float:
    if active_count.size == 0:
        return float("nan")
    return float(np.nanmean(active_count.astype(float)))


def compute_soc_energy_series(times: np.ndarray, active_count: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return np.asarray([]), np.asarray([])
    dt = np.diff(times, prepend=times[0])
    dt[0] = dt[1] if dt.size > 1 else 0.0
    power = args.base_drain_w + args.beacon_drain_w * np.clip(active_count.astype(float), 0.0, None)
    energy_wh = np.cumsum(power * dt) / 3600.0
    soc = np.empty_like(energy_wh, dtype=float)
    soc[0] = max(0.0, min(1.0, float(args.soc_init)))
    if args.battery_wh <= 0:
        return np.full_like(energy_wh, soc[0], dtype=float), energy_wh
    for i in range(1, energy_wh.size):
        dsoc = (power[i - 1] * dt[i - 1]) / (args.battery_wh * 3600.0)
        soc[i] = max(float(args.soc_min), min(1.0, soc[i - 1] - args.drain_scale * dsoc))
    return soc, energy_wh


def series_from_selector_meta(times: np.ndarray, selector_meta: List[Dict[str, Any]], key: str) -> np.ndarray:
    if times.size == 0 or not selector_meta:
        return np.asarray([])
    log = {
        "t": [float(m.get("t", 0.0)) for m in selector_meta if key in m],
        "values": [float(m.get(key)) for m in selector_meta if key in m],
    }
    return fill_sparse_series(times, log)


def parse_seed_list(seed_arg: str) -> List[int]:
    if not seed_arg:
        return list(DEFAULT_SEEDS)
    parts = [p.strip() for p in seed_arg.split(",") if p.strip()]
    return [int(p) for p in parts]


def fill_sparse_series(times: np.ndarray, log: Dict[str, Any]) -> np.ndarray:
    series = np.full_like(times, np.nan, dtype=float)
    if not log:
        return series
    t_log = np.asarray(log.get("t", []), dtype=float)
    vals = np.asarray(log.get("values", []), dtype=float)
    for t_val, v in zip(t_log, vals):
        idx = int(np.searchsorted(times, t_val))
        if 0 <= idx < series.size:
            series[idx] = v
    return series


def write_timeseries_csv(trial: Dict[str, Any], path: Path) -> None:
    ts = trial.get("timeseries", {})
    if not ts:
        return

    times = np.asarray(ts.get("t", []), dtype=float)
    gt = np.asarray(ts.get("true_pos", []), dtype=float)
    est = np.asarray(ts.get("est_pos", []), dtype=float)
    err_norm = np.asarray(ts.get("err_norm", []), dtype=float)
    Ppos = np.asarray(ts.get("Ppos", []), dtype=float)
    nees_pos = np.asarray(ts.get("nees_pos", []), dtype=float)
    nees_full = np.asarray(ts.get("nees_full", []), dtype=float)
    nis_logs = ts.get("nis_logs", {}) or {}

    nis_dvl = fill_sparse_series(times, nis_logs.get("dvl", {}))
    nis_depth = fill_sparse_series(times, nis_logs.get("depth", {}))
    nis_ac = fill_sparse_series(times, nis_logs.get("acoustic", {}))

    fields = [
        "t", "gt_x", "gt_y", "gt_z", "est_x", "est_y", "est_z", "err_norm",
        "P_xx", "P_yy", "P_zz", "nees_pos", "nees_full", "nis_dvl", "nis_depth", "nis_acoustic",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(times.size):
            row = {
                "t": float(times[i]),
                "gt_x": float(gt[i, 0]) if gt.size else np.nan,
                "gt_y": float(gt[i, 1]) if gt.size else np.nan,
                "gt_z": float(gt[i, 2]) if gt.size else np.nan,
                "est_x": float(est[i, 0]) if est.size else np.nan,
                "est_y": float(est[i, 1]) if est.size else np.nan,
                "est_z": float(est[i, 2]) if est.size else np.nan,
                "err_norm": float(err_norm[i]) if err_norm.size else np.nan,
                "P_xx": float(Ppos[i, 0, 0]) if Ppos.size else np.nan,
                "P_yy": float(Ppos[i, 1, 1]) if Ppos.size else np.nan,
                "P_zz": float(Ppos[i, 2, 2]) if Ppos.size else np.nan,
                "nees_pos": float(nees_pos[i]) if nees_pos.size else np.nan,
                "nees_full": float(nees_full[i]) if nees_full.size else np.nan,
                "nis_dvl": float(nis_dvl[i]) if np.isfinite(nis_dvl[i]) else np.nan,
                "nis_depth": float(nis_depth[i]) if np.isfinite(nis_depth[i]) else np.nan,
                "nis_acoustic": float(nis_ac[i]) if np.isfinite(nis_ac[i]) else np.nan,
            }
            writer.writerow(row)


def plot_per_seed(trial: Dict[str, Any], plots_dir: Path) -> None:
    ts = trial.get("timeseries", {})
    if not ts:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    t = ts["t"]
    gt = ts["true_pos"]
    est = ts["est_pos"]
    err_norm = ts["err_norm"]
    nees_pos = ts.get("nees_pos")
    nis_logs = ts.get("nis_logs", {}) or {}

    plt.figure(figsize=(6, 4))
    plt.plot(gt[:, 0], gt[:, 1], label="Ground truth", lw=1.2, color="#4d4d4d")
    plt.plot(est[:, 0], est[:, 1], label="Estimate", lw=1.2, color="#1f77b4")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.title("XY trajectory")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(plots_dir / "traj_xy.png")
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(t, err_norm, label="||pos error||", lw=1.2, color="#1f77b4")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.title("Position error vs time")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(plots_dir / "pos_err_vs_time.png")
    plt.close()

    if nees_pos is not None:
        plt.figure(figsize=(6, 3.5))
        plt.plot(t, nees_pos, lw=1.2, label="NEES pos")
        dof = 3
        lo = chi2.ppf(0.025, dof)
        hi = chi2.ppf(0.975, dof)
        plt.axhline(lo, color="gray", linestyle="--", linewidth=1, label="95% bounds")
        plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Time [s]")
        plt.ylabel("NEES (pos)")
        plt.title("NEES vs time")
        _style_axis(plt.gca(), legend_loc="best")
        plt.tight_layout()
        _save_figure(plots_dir / "nees_vs_time.png")
        plt.close()

    for key, fname in [("dvl", "nis_dvl_vs_time.png"), ("depth", "nis_depth_vs_time.png"), ("acoustic", "nis_acoustic_vs_time.png")]:
        log = nis_logs.get(key)
        if not log or len(log.get("t", [])) == 0:
            continue
        plt.figure(figsize=(6, 3.5))
        plt.plot(log["t"], log["values"], lw=1.2, label=f"NIS {key}")
        dof = log.get("dof", 1)
        lo = chi2.ppf(0.025, dof)
        hi = chi2.ppf(0.975, dof)
        plt.axhline(lo, color="gray", linestyle="--", linewidth=1, label="95% bounds")
        plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Time [s]")
        plt.ylabel("NIS")
        plt.title(f"NIS {key} vs time")
        _style_axis(plt.gca(), legend_loc="best")
        plt.tight_layout()
        _save_figure(plots_dir / fname)
        plt.close()



def plot_seed_trajectory(trial: Dict[str, Any], out_path: Path) -> None:
    ts = trial.get("timeseries", {})
    if not ts:
        return
    gt = np.asarray(ts.get("true_pos", []), dtype=float)
    if gt.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(gt[:, 0], gt[:, 1], label="Ground truth", lw=1.2, color="#1f77b4")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.title("XY trajectory")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_path)
    plt.close()


def metrics_from_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "seed": trial.get("seed"),
        "runtime_seconds": trial.get("runtime_seconds"),
        "pos_rmse": trial.get("pos_rmse_total"),
        "vel_rmse": trial.get("vel_rmse_total"),
        "final_error": trial.get("final_position_error"),
        "nees_pct": trial.get("nees_full", {}).get("percent_inside_bounds"),
        "nis_dvl_pct": (trial.get("nis", {}).get("dvl") or {}).get("percent_inside_bounds"),
        "nis_depth_pct": (trial.get("nis", {}).get("depth") or {}).get("percent_inside_bounds"),
        "nis_acoustic_pct": (trial.get("nis", {}).get("acoustic") or {}).get("percent_inside_bounds"),
        "failed": trial.get("failed", False),
        "error": trial.get("error"),
    }


def aggregate_results(config_name: str, runs: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_runs = [r for r in runs if not r.get("failed")]
    if not ok_runs:
        return

    pos_rmse = np.array([r["pos_rmse_total"] for r in ok_runs], dtype=float)
    vel_rmse = np.array([r["vel_rmse_total"] for r in ok_runs], dtype=float)
    final_err = np.array([r["final_position_error"] for r in ok_runs], dtype=float)
    nees_pct = np.array([r["nees_full"]["percent_inside_bounds"] for r in ok_runs], dtype=float)
    nis_dvl = np.array([r["nis"]["dvl"]["percent_inside_bounds"] for r in ok_runs if r["nis"].get("dvl")], dtype=float)
    nis_depth = np.array([r["nis"]["depth"]["percent_inside_bounds"] for r in ok_runs if r["nis"].get("depth")], dtype=float)
    nis_ac = np.array([r["nis"]["acoustic"]["percent_inside_bounds"] for r in ok_runs if r["nis"].get("acoustic")], dtype=float)

    def s(arr):
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    summary_rows = [{
        "config": config_name,
        "n_runs": len(ok_runs),
        "failures": len(runs) - len(ok_runs),
        "pos_rmse_mean": s(pos_rmse)[0],
        "pos_rmse_std": s(pos_rmse)[1],
        "vel_rmse_mean": s(vel_rmse)[0],
        "vel_rmse_std": s(vel_rmse)[1],
        "final_err_median": float(np.nanmedian(final_err)),
        "final_err_p90": float(np.nanpercentile(final_err, 90)),
        "final_err_p95": float(np.nanpercentile(final_err, 95)),
        "nees_pct_in_bounds": s(nees_pct)[0],
        "nis_dvl_pct": s(nis_dvl)[0] if nis_dvl.size else np.nan,
        "nis_depth_pct": s(nis_depth)[0] if nis_depth.size else np.nan,
        "nis_acoustic_pct": s(nis_ac)[0] if nis_ac.size else np.nan,
    }]

    csv_path = out_dir / "summary_table.csv"
    md_path = out_dir / "summary_table.md"
    fieldnames = list(summary_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with md_path.open("w") as f:
        header = " | ".join(fieldnames)
        f.write(header + "\n")
        f.write(" | ".join(["---"] * len(fieldnames)) + "\n")
        for row in summary_rows:
            f.write(" | ".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k]) for k in fieldnames) + "\n")

    plt.figure(figsize=(6, 3.5))
    sorted_err = np.sort(final_err)
    cdf = np.linspace(0, 1, sorted_err.size)
    plt.plot(sorted_err, cdf, lw=1.2, label=config_name, color=_algo_color(config_name))
    plt.xlabel("Final position error [m]")
    plt.ylabel("CDF [1]")
    plt.title("Final error CDF")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "final_error_cdf.png")
    plt.close()

    plt.figure(figsize=(4, 4))
    _styled_boxplot(plt.gca(), [pos_rmse.tolist()], [config_name])
    plt.ylabel("RMSE [m]")
    plt.title("Position RMSE")
    _style_axis(plt.gca(), show_legend=False)
    plt.tight_layout()
    _save_figure(out_dir / "rmse_boxplot.png")
    plt.close()

    plt.figure(figsize=(4, 4))
    _styled_boxplot(plt.gca(), [nees_pct.tolist()], [config_name])
    plt.ylabel("Percent inside gate [%]")
    plt.title("NEES coverage")
    _style_axis(plt.gca(), show_legend=False)
    plt.tight_layout()
    _save_figure(out_dir / "nees_boxplot.png")
    plt.close()

    # Mean NEES over time (optional)
    if ok_runs and ok_runs[0].get("timeseries"):
        min_len = min(r["timeseries"]["nees_pos"].shape[0] for r in ok_runs if r.get("timeseries"))
        if min_len > 0:
            nees_stack = np.stack([r["timeseries"]["nees_pos"][:min_len] for r in ok_runs])
            t_ref = ok_runs[0]["timeseries"]["t"][:min_len]
            mean_nees = np.nanmean(nees_stack, axis=0)
            plt.figure(figsize=(6, 3.5))
            plt.plot(t_ref, mean_nees, lw=1.2, label="Mean NEES pos", color="#1f77b4")
            lo = chi2.ppf(0.025, 3)
            hi = chi2.ppf(0.975, 3)
            plt.axhline(lo, color="gray", linestyle="--", linewidth=1, label="95% bounds")
            plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
            plt.xlabel("Time [s]")
            plt.ylabel("NEES (pos)")
            plt.title("Mean NEES over time")
            _style_axis(plt.gca(), legend_loc="best")
            plt.tight_layout()
            _save_figure(out_dir / "mean_nees_over_time.png")
            plt.close()


def compare_configs(results_by_config: Dict[str, List[Dict[str, Any]]], out_root: Path) -> None:
    if len(results_by_config) < 2:
        return
    out_dir = out_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_config.items():
        ok_runs = [r for r in runs if not r.get("failed")]
        if not ok_runs:
            continue
        final_err = np.sort([r["final_position_error"] for r in ok_runs])
        cdf = np.linspace(0, 1, final_err.size)
        plt.plot(final_err, cdf, lw=1.2, label=name, color=_algo_color(name))
    plt.xlabel("Final position error [m]")
    plt.ylabel("CDF [1]")
    plt.title("Final error CDF (configs)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "final_error_cdf_overlay.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    data = []
    labels = []
    for name, runs in results_by_config.items():
        ok_runs = [r for r in runs if not r.get("failed")]
        if not ok_runs:
            continue
        data.append([r["pos_rmse_total"] for r in ok_runs])
        labels.append(name)
    if data:
        _styled_boxplot(plt.gca(), data, labels)
        plt.ylabel("Position RMSE [m]")
        plt.title("RMSE by config")
        _style_axis(plt.gca(), show_legend=False)
        plt.tight_layout()
        _save_figure(out_dir / "rmse_boxplot_overlay.png")
        plt.close()


def build_selector(policy_type: str, args: argparse.Namespace) -> Tuple[Any, List[Dict[str, Any]]]:
    selector_meta: List[Dict[str, Any]] = []
    phase_schedule = parse_phase_schedule(args.phase_schedule)
    dropout_windows: Dict[str, List[Tuple[float, float]]] = {}
    if getattr(args, "self_test_dropout", False):
        dropout_windows = {
            "usv3": [(5.0, 25.0)],
            "usv4": [(10.0, 25.0)],
        }

    def _apply_dropouts(t: float, targets: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
        if not dropout_windows:
            return targets
        kept: List[Tuple[str, np.ndarray]] = []
        for name, pos in targets:
            spans = dropout_windows.get(name, [])
            if any(a <= t <= b for a, b in spans):
                continue
            kept.append((name, pos))
        return kept

    if policy_type == "weighted":
        weight_scale_energy = 1.0 if args.energy_weight is None else float(args.energy_weight)
        w_params = WeightedParams(
            sigma_r=args.sigma_r,
            min_beacons_xy=args.min_beacons_xy,
            min_beacons_3d=args.min_beacons_3d,
            max_beacons=4,
            rank_req_xy=args.rank_req_xy,
            rank_req_3d=args.rank_req_3d,
            gdop_thresh_xy=args.gdop_xy,
            gdop_thresh_3d=args.gdop_3d,
            min_dwell_sec=args.min_dwell_sec,
            score_margin=args.score_margin,
            power_save_tol=args.power_save_tol,
            soc_init=args.soc_init,
            soc_min=args.soc_min,
            low_power_soc=args.low_power_soc,
            base_drain_w=args.base_drain_w,
            beacon_drain_w=args.beacon_drain_w,
            drain_scale=args.drain_scale,
            battery_wh=args.battery_wh,
            target_unc_xy=args.target_unc_xy,
            target_unc_3d=args.target_unc_3d,
            off_unc_mult=args.off_unc_mult,
            mission_phase=args.mission_phase,
            phase_schedule=phase_schedule,
            allow_zero=bool(args.allow_zero_beacons),
            churn_interval_sec=float(args.churn_interval_sec),
            churn_score_eps=float(args.churn_score_eps),
            weight_scale_energy=weight_scale_energy,
        )
        selector = WeightedPolicySelector(w_params, DEFAULT_PHASE_WEIGHTS)

        def selector_fn(*_args, **kwargs):
            t = kwargs.get("t_current", kwargs.get("t", 0.0))
            xhat = kwargs.get("ekf_state", kwargs.get("xhat", np.zeros(6)))
            depth_available = kwargs.get("depth_available", True)
            target_info = _apply_dropouts(t, kwargs.get("target_info", []))
            covariance = kwargs.get("covariance", None)
            chosen, meta = selector.decide(t, xhat, depth_available, target_info, covariance=covariance)
            selector_meta.append({"t": t, "selected": chosen, "n_active": len(chosen), **meta})
            return {
                "active_names": list(chosen),
                "active_set": list(chosen),
                "n_active": len(chosen),
                "soc": meta.get("soc", None),
                "rank_xy": meta.get("rank_xy", 0),
                "gdop_xy": meta.get("gdop_xy", float("nan")),
                "rank_3d": meta.get("rank_3d", 0),
                "gdop_3d": meta.get("gdop_3d", float("nan")),
                "fim_logdet": meta.get("fim_logdet", float("-inf")),
                "mode": meta.get("mode", "xy"),
            }

        return selector_fn, selector_meta

    if policy_type == "v2":
        manager: Optional[AdaptiveModemManagerV2] = None
        dwell_steps = max(1, int(round(args.min_dwell_sec * args.ticks_per_sec))) if args.min_dwell_sec > 0 else 1
        min_subset_size = 0 if bool(args.allow_zero_beacons) else max(1, int(args.min_beacons_xy))

        def selector_fn(*_args, **kwargs):
            nonlocal manager
            t = kwargs.get("t_current", kwargs.get("t", 0.0))
            xhat = kwargs.get("ekf_state", kwargs.get("xhat", np.zeros(6)))
            depth_available = kwargs.get("depth_available", True)
            target_info = _apply_dropouts(t, kwargs.get("target_info", []))
            covariance = kwargs.get("covariance", None)
            if manager is None:
                beacon_map = {n: p for n, p in target_info}
                manager = AdaptiveModemManagerV2(
                    beacon_positions=beacon_map,
                    sigma_r=args.sigma_r,
                    size_penalty=args.size_penalty,
                    rank_deficit_penalty=float(args.v2_rank_deficit_penalty),
                    min_dwell_steps=dwell_steps,
                    switch_margin=args.switch_margin,
                    target_unc_xy=args.target_unc_xy,
                    target_unc_3d=args.target_unc_3d,
                    off_unc_mult=args.off_unc_mult,
                    gate_min_dwell_steps=dwell_steps,
                    min_subset_size=min_subset_size,
                    max_subset_size=4,
                    prefer_smaller=True,
                    battery_wh=args.battery_wh,
                    base_drain_w=args.base_drain_w,
                    beacon_drain_w=args.beacon_drain_w,
                    drain_scale=args.drain_scale,
                    soc_init=args.soc_init,
                    soc_min=args.soc_min,
                    energy_weight=float(args.energy_weight) if args.energy_weight is not None else 0.0,
                    low_power_soc=float(args.v2_low_power_soc),
                    energy_weight_low_power_mult=float(args.v2_energy_mult),
                    size_penalty_low_power_mult=float(args.v2_size_penalty_mult),
                    rank_deficit_penalty_low_power_mult=float(args.v2_rank_deficit_mult),
                )

            P_use = covariance
            if P_use is None or P_use.size < 9:
                target_unc = args.target_unc_xy if depth_available else args.target_unc_3d
                P_use = np.eye(3) * (target_unc ** 2)

            chosen, metrics = manager.select(
                t=t,
                a_pos=np.asarray(xhat[:3], dtype=float),
                P=np.asarray(P_use, dtype=float),
                available_ids=[n for n, _ in target_info],
                depth_available=depth_available,
            )
            md = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
            if "fim_logdet" not in md:
                md["fim_logdet"] = md.get("fim_logdet_xy", md.get("fim_logdet_3d", float("-inf")))
            selector_meta.append({"t": t, "selected": chosen, "n_active": len(chosen), **md})
            return {
                "active_names": list(chosen),
                "active_set": list(chosen),
                "n_active": len(chosen),
                "soc": md.get("soc", None),
                "rank_xy": md.get("rank_xy", 0),
                "gdop_xy": md.get("gdop_xy", float("nan")),
                "rank_3d": md.get("rank_3d", 0),
                "gdop_3d": md.get("gdop_3d", float("nan")),
                "fim_logdet": md.get("fim_logdet", float("-inf")),
                "mode": md.get("reason", "v2"),
            }

        return selector_fn, selector_meta

    if policy_type == "gdop":
        min_beacons_xy = 0 if bool(args.allow_zero_beacons) else int(args.min_beacons_xy)
        min_beacons_3d = 0 if bool(args.allow_zero_beacons) else int(args.min_beacons_3d)
        p = PolicyParams(
            sigma_r=args.sigma_r,
            gdop_thresh_xy=args.gdop_xy,
            gdop_thresh_3d=args.gdop_3d,
            switch_margin=args.switch_margin,
            min_dwell_sec=args.min_dwell_sec,
            size_penalty=args.size_penalty,
            min_beacons_xy=min_beacons_xy,
            min_beacons_3d=min_beacons_3d,
            max_beacons=4,
        )
        selector = GeometryPolicySelector(p)

        def selector_fn(*_args, **kwargs):
            t = kwargs.get("t_current", kwargs.get("t", 0.0))
            xhat = kwargs.get("ekf_state", kwargs.get("xhat", np.zeros(6)))
            depth_available = kwargs.get("depth_available", True)
            target_info = _apply_dropouts(t, kwargs.get("target_info", []))
            chosen, meta = selector.decide(t, xhat, depth_available, target_info)
            selector_meta.append({"t": t, "selected": chosen, "n_active": len(chosen), **meta})
            return {
                "active_names": list(chosen),
                "active_set": list(chosen),
                "n_active": len(chosen),
                "soc": None,
                "rank_xy": meta.get("rank_xy", 0),
                "gdop_xy": meta.get("gdop_xy", float("nan")),
                "rank_3d": meta.get("rank_3d", 0),
                "gdop_3d": meta.get("gdop_3d", float("nan")),
                "fim_logdet": meta.get("fim_logdet", float("-inf")),
                "mode": meta.get("reason", "gdop"),
            }

        return selector_fn, selector_meta

    raise ValueError(f"Unknown adaptive policy: {policy_type}")


def run_trial(algorithm: str, seed: int, args: argparse.Namespace, random_streams: RandomStreams) -> Dict[str, Any]:
    selector_meta: List[Dict[str, Any]] = []
    selector_fn = None

    config_overrides = {
        "duration_sec": float(args.duration),
        "trajectory": "spiral",
        "use_currents": False,
        "imu_accel_extra_std": float(args.imu_accel_extra_std),
        "imu_bias_rw_std": float(args.imu_bias_rw_std),
        "dvl_extra_std": float(args.dvl_extra_std),
        "depth_extra_std": float(args.depth_extra_std),
        "range_extra_std": float(args.range_extra_std),
    }

    if algorithm == "imu_dvl_depth":
        config_overrides.update({
            "enable_dvl": True,
            "enable_depth": True,
            "enable_acoustic": False,
            "use_acoustic_updates": False,
            "use_all_acoustic": False,
        })
    elif algorithm == "imu_dvl_depth_all4":
        config_overrides.update({
            "enable_dvl": True,
            "enable_depth": True,
            "enable_acoustic": True,
            "use_acoustic_updates": True,
            "use_all_acoustic": True,
        })
    elif algorithm in ("adaptive", "adaptive_gdop", "adaptive_weighted", "adaptive_v2"):
        config_overrides.update({
            "enable_dvl": True,
            "enable_depth": True,
            "enable_acoustic": True,
            "use_acoustic_updates": True,
            "use_all_acoustic": False,
        })
        if algorithm == "adaptive":
            policy_type = args.adaptive_policy
        else:
            policy_type = algorithm.split("_", 1)[1]
        selector_fn, selector_meta = build_selector(policy_type, args)
        config_overrides["modem_selector_fn"] = selector_fn
        if getattr(args, "self_test_dropout", False):
            config_overrides["modem_dropout_mode"] = "ignore_updates"
            config_overrides["modem_dropout_intervals"] = {
                "usv3": [(5.0, 25.0)],
                "usv4": [(10.0, 25.0)],
            }
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")

    trial = run_single_trial(
        seed=int(seed),
        config_overrides=config_overrides,
        return_timeseries=True,
        target_names=list(args.targets),
        make_plots=False,
        random_streams=random_streams,
    )

    if "x0_perturb" not in trial:
        ts = trial.get("timeseries", {}) or {}
        n_steps = len(ts.get("t", [])) if ts else 0
        if n_steps > 0:
            built = random_streams.build(n_steps=n_steps, dt=_median_dt(np.asarray(ts.get("t", []), dtype=float)))
            trial["x0_perturb"] = built.get("x0_perturb")

    if selector_meta:
        trial["selector_meta"] = selector_meta

    return trial


def _run_trial_worker(payload: Tuple[str, int, Dict[str, Any]]) -> Tuple[str, int, Dict[str, Any]]:
    """Process-pool safe wrapper to run a single trial.

    payload: (algorithm, seed, args_dict)
    Returns (algorithm, seed, trial_dict).
    """
    algorithm, seed, args_dict = payload
    args_ns = argparse.Namespace(**args_dict)
    rs_cfg = RandomStreamsConfig(args_ns.x0_pos_std, args_ns.x0_vel_std)
    rs = RandomStreams(seed, rs_cfg)
    trial = run_trial(algorithm, seed, args_ns, rs)
    return algorithm, seed, trial


def compute_energy_metrics(trial: Dict[str, Any], args: argparse.Namespace, algorithm: str) -> float:
    ts = trial.get("timeseries", {}) or {}
    times = np.asarray(ts.get("t", []), dtype=float)

    if algorithm == "imu_dvl_depth":
        return 0.0
    if algorithm == "imu_dvl_depth_all4":
        if times.size < 2:
            return 0.0
        dt = _median_dt(times)
        return float((args.base_drain_w + args.beacon_drain_w * 4.0) * times.size * dt / 3600.0)

    active_count = np.asarray(ts.get("active_count", []), dtype=float)
    if active_count.size == 0 and "selector_meta" in trial:
        meta = trial.get("selector_meta", [])
        active_count = np.asarray([m.get("n_active", 0) for m in meta], dtype=float)
        times = np.asarray([m.get("t", 0.0) for m in meta], dtype=float)

    return _energy_from_active_count(times, active_count, args.base_drain_w, args.beacon_drain_w)


def write_per_run_metrics(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    path = out_dir / "per_run_metrics.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_metrics(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    path = out_dir / "summary_metrics.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_results_readme(out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# Monte Carlo Results

- trajectory: spiral
- currents: disabled
- duration: {args.duration}s
- runs: {args.runs}
- seeds: {','.join(str(s) for s in args.seeds)}
- algorithms: {', '.join(args.algorithms)}

## Figures
- final_error_cdf_overlay.pdf: CDF of final position error across algorithms.
- rmse_boxplot_overlay.pdf: RMSE distribution across algorithms.
- energy_boxplot.pdf: Acoustic energy distribution across algorithms.
- energy_vs_error.pdf: Energy vs final error scatter.
- mean_nees_pos.pdf: Mean NEES over time with 95% bounds.
- mean_nis_acoustic.pdf: Mean acoustic NIS over time (when applicable).
- pos_error_vs_time_ci.pdf: Position error vs time with 95% CI.
- nees_vs_time_ci.pdf: NEES vs time with 95% CI and bounds.
- nis_acoustic_vs_time_ci.pdf: Acoustic NIS vs time with 95% CI and bounds.
- active_count_vs_time_ci.pdf: Active beacon count vs time with 95% CI.
- soc_vs_time_ci.pdf: SOC vs time with 95% CI.
- energy_vs_time_ci.pdf: Energy vs time with 95% CI.
- gdop_vs_time_ci.pdf: GDOP vs time (adaptive policies).
- fim_logdet_vs_time_ci.pdf: FIM logdet vs time (adaptive policies).
- rmse_vs_crlb.pdf: RMSE vs CRLB efficiency scatter.
- active_count_hist_adaptive.pdf: % time in each active-beacon count.
- switches_hist_adaptive.pdf: Switching events per run.

## Metrics to cite
- mean/median final position error
- position RMSE distribution
- energy_Wh savings vs baselines
- switching statistics (mean switches/run, % time at k beacons)
- policy_summary.csv: RMSE/CRLB/consistency/energy summary per policy.
"""
    (out_dir / "README_results.md").write_text(content)


def plot_cdf_overlay(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        vals = np.asarray([r["final_error"] for r in runs if np.isfinite(r["final_error"])], dtype=float)
        if vals.size == 0:
            continue
        vals = np.sort(vals)
        cdf = np.linspace(0, 1, vals.size)
        plt.plot(vals, cdf, lw=1.2, label=name, color=_algo_color(name))
    plt.xlabel("Final position error [m]")
    plt.ylabel("CDF [1]")
    plt.title("Final error CDF")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "final_error_cdf_overlay.png")
    plt.close()


def plot_rmse_boxplot(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    data = []
    labels = []
    for name, runs in results_by_algo.items():
        vals = [r["pos_rmse"] for r in runs if np.isfinite(r["pos_rmse"]) ]
        if vals:
            data.append(vals)
            labels.append(name)
    if not data:
        return
    plt.figure(figsize=(5, 4))
    _styled_boxplot(plt.gca(), data, labels)
    plt.ylabel("Position RMSE [m]")
    plt.title("RMSE by algorithm")
    _style_axis(plt.gca(), show_legend=False)
    plt.tight_layout()
    _save_figure(out_dir / "rmse_boxplot_overlay.png")
    plt.close()


def plot_energy_boxplot(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    data = []
    labels = []
    for name, runs in results_by_algo.items():
        vals = [r["energy_Wh"] for r in runs if np.isfinite(r["energy_Wh"]) ]
        if vals:
            data.append(vals)
            labels.append(name)
    if not data:
        return
    plt.figure(figsize=(5, 4))
    _styled_boxplot(plt.gca(), data, labels)
    plt.ylabel("Energy [Wh]")
    plt.title("Acoustic energy by algorithm")
    _style_axis(plt.gca(), show_legend=False)
    plt.tight_layout()
    _save_figure(out_dir / "energy_boxplot.png")
    plt.close()


def plot_energy_vs_error(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(5, 4))
    for name, runs in results_by_algo.items():
        x = [r["energy_Wh"] for r in runs]
        y = [r["final_error"] for r in runs]
        if x and y:
            plt.scatter(x, y, label=name, alpha=0.7, color=_algo_color(name), edgecolor="white", linewidth=0.5)
    plt.xlabel("Energy [Wh]")
    plt.ylabel("Final error [m]")
    plt.title("Energy vs accuracy")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "energy_vs_error.png")
    plt.close()


def plot_mean_nees(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        series = [r["nees_pos_series"] for r in runs if r.get("nees_pos_series") is not None]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        mean_nees = np.nanmean(stack, axis=0)
        t = runs[0]["t_series"][:min_len]
        plt.plot(t, mean_nees, label=name, lw=1.2, color=_algo_color(name))
    lo = chi2.ppf(0.025, 3)
    hi = chi2.ppf(0.975, 3)
    plt.axhline(lo, color="gray", linestyle="--", linewidth=1)
    plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("NEES (pos)")
    plt.title("Mean NEES (pos)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "mean_nees_pos.png")
    plt.close()


def plot_mean_nis_acoustic(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        series = [r["nis_ac_series"] for r in runs if r.get("nis_ac_series") is not None]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        mean_nis = np.nanmean(stack, axis=0)
        t = runs[0]["t_series"][:min_len]
        plt.plot(t, mean_nis, label=name, lw=1.2, color=_algo_color(name))
    plt.xlabel("Time [s]")
    plt.ylabel("NIS (acoustic)")
    plt.title("Mean acoustic NIS")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "mean_nis_acoustic.png")
    plt.close()


def plot_error_vs_time_ci(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        series = [r["err_series"] for r in runs if r.get("err_series") is not None]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(stack)
        plt.plot(t, mean, label=name, lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("Position error vs time (mean ± 95% CI)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "pos_error_vs_time_ci.png")
    plt.close()


def plot_nees_vs_time_ci(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        series = [r["nees_pos_series"] for r in runs if r.get("nees_pos_series") is not None]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(stack)
        plt.plot(t, mean, label=name, lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
    dof = 3
    lo = chi2.ppf(0.025, dof)
    hi = chi2.ppf(0.975, dof)
    plt.axhline(lo, color="gray", linestyle="--", linewidth=1)
    plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("NEES (pos)")
    plt.title("NEES vs time (mean ± 95% CI)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "nees_vs_time_ci.png")
    plt.close()


def plot_nis_acoustic_vs_time_ci(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    plotted = False
    for name, runs in results_by_algo.items():
        series = [r["nis_ac_series"] for r in runs if r.get("nis_ac_series") is not None]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(stack)
        plt.plot(t, mean, label=name, lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
        plotted = True
    if not plotted:
        return
    dof = None
    for runs in results_by_algo.values():
        for r in runs:
            if r.get("nis_ac_dof") is not None:
                dof = int(r.get("nis_ac_dof"))
                break
        if dof is not None:
            break
    if dof is None:
        dof = 1
    lo = chi2.ppf(0.025, dof)
    hi = chi2.ppf(0.975, dof)
    plt.axhline(lo, color="gray", linestyle="--", linewidth=1)
    plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("NIS (acoustic)")
    plt.title("Acoustic NIS vs time (mean ± 95% CI)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "nis_acoustic_vs_time_ci.png")
    plt.close()


def plot_active_count_vs_time(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        series = [r["active_count"] for r in runs if r.get("active_count") is not None and len(r["active_count"]) > 0]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(stack)
        plt.plot(t, mean, label=name, lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Active beacons")
    plt.title("Active beacon count vs time (mean ± 95% CI)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "active_count_vs_time_ci.png")
    plt.close()


def plot_soc_energy_vs_time(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path, args: argparse.Namespace) -> None:
    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        soc_series = []
        energy_series = []
        for r in runs:
            t = r.get("t_series", np.asarray([]))
            active = r.get("active_count", np.asarray([]))
            if t.size == 0 or active.size == 0:
                continue
            soc, energy = compute_soc_energy_series(t, active, args)
            soc_series.append(soc)
            energy_series.append(energy)
        if not soc_series:
            continue
        min_len = min(len(s) for s in soc_series)
        soc_stack = np.stack([s[:min_len] for s in soc_series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(soc_stack)
        plt.plot(t, mean, label=f"{name} SOC", lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
    plt.xlabel("Time [s]")
    plt.ylabel("SOC [1]")
    plt.title("SOC vs time (mean ± 95% CI)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "soc_vs_time_ci.png")
    plt.close()

    plt.figure(figsize=(6, 3.5))
    for name, runs in results_by_algo.items():
        energy_series = []
        for r in runs:
            t = r.get("t_series", np.asarray([]))
            active = r.get("active_count", np.asarray([]))
            if t.size == 0 or active.size == 0:
                continue
            _, energy = compute_soc_energy_series(t, active, args)
            energy_series.append(energy)
        if not energy_series:
            continue
        min_len = min(len(s) for s in energy_series)
        energy_stack = np.stack([s[:min_len] for s in energy_series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(energy_stack)
        plt.plot(t, mean, label=name, lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [Wh]")
    plt.title("Energy vs time (mean ± 95% CI)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "energy_vs_time_ci.png")
    plt.close()


def plot_gdop_vs_time(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    plotted = False
    for name, runs in results_by_algo.items():
        if not name.startswith("adaptive"):
            continue
        series = [r["gdop_xy_series"] for r in runs if r.get("gdop_xy_series") is not None]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(stack)
        plt.plot(t, mean, label=name, lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
        plotted = True
    if not plotted:
        return
    plt.xlabel("Time [s]")
    plt.ylabel("GDOP (xy)")
    plt.title("GDOP vs time (adaptive policies)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "gdop_vs_time_ci.png")
    plt.close()


def plot_fim_logdet_vs_time(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    plotted = False
    for name, runs in results_by_algo.items():
        if not name.startswith("adaptive"):
            continue
        series = []
        for r in runs:
            t = r.get("t_series", np.asarray([]))
            meta = r.get("selector_meta", [])
            if t.size == 0 or not meta:
                continue
            series.append(series_from_selector_meta(t, meta, "fim_logdet"))
        if not series:
            continue
        min_len = min(len(s) for s in series)
        stack = np.stack([s[:min_len] for s in series])
        t = runs[0]["t_series"][:min_len]
        mean, lo, hi = _mean_ci(stack)
        plt.plot(t, mean, label=name, lw=1.2, color=_algo_color(name))
        plt.fill_between(t, lo, hi, alpha=0.2)
        plotted = True
    if not plotted:
        return
    plt.xlabel("Time [s]")
    plt.ylabel("FIM logdet")
    plt.title("FIM logdet vs time (adaptive policies)")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "fim_logdet_vs_time_ci.png")
    plt.close()


def plot_rmse_vs_crlb(results_by_algo: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    plt.figure(figsize=(5, 4))
    plotted = False
    for name, runs in results_by_algo.items():
        xs = []
        ys = []
        for r in runs:
            if not np.isfinite(r.get("crlb_eff", np.nan)):
                continue
            xs.append(float(r.get("crlb_eff")))
            ys.append(float(r.get("pos_rmse")))
        if xs and ys:
            plt.scatter(xs, ys, label=name, alpha=0.7, color=_algo_color(name), edgecolor="white", linewidth=0.5)
            plotted = True
    if not plotted:
        return
    plt.xlabel(r"$RMSE/CRLB$ [1]")
    plt.ylabel("Position RMSE [m]")
    plt.title("RMSE vs CRLB efficiency")
    _style_axis(plt.gca(), legend_loc="best")
    plt.tight_layout()
    _save_figure(out_dir / "rmse_vs_crlb.png")
    plt.close()


def plot_adaptive_telemetry(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    pct_cols = ["pct_active_0", "pct_active_1", "pct_active_2", "pct_active_3", "pct_active_4"]
    avg_pct = [float(np.mean([r[c] for r in rows])) for c in pct_cols]
    plt.figure(figsize=(5, 3.5))
    plt.bar([0, 1, 2, 3, 4], avg_pct, color=_algo_color("adaptive"), alpha=0.6)
    plt.xlabel("Active beacons")
    plt.ylabel("Time share [%]")
    plt.title("Adaptive: % time by active beacons")
    _style_axis(plt.gca(), show_legend=False)
    plt.tight_layout()
    _save_figure(out_dir / "active_count_hist_adaptive.png")
    plt.close()

    switches = [r["switch_count"] for r in rows]
    plt.figure(figsize=(5, 3.5))
    plt.hist(switches, bins=max(3, min(10, len(switches))), color=_algo_color("adaptive"), alpha=0.6)
    plt.xlabel("Switches per run")
    plt.ylabel("Count [1]")
    plt.title("Adaptive switching count")
    _style_axis(plt.gca(), show_legend=False)
    plt.tight_layout()
    _save_figure(out_dir / "switches_hist_adaptive.png")
    plt.close()


def run_mc_new(args: argparse.Namespace) -> None:
    _set_style()
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "configs").mkdir(parents=True, exist_ok=True)
    (out_root / "trials").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)

    with (out_root / "configs" / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    results_by_algo: Dict[str, List[Dict[str, Any]]] = {a: [] for a in args.algorithms}
    first_trial_with_ts: Optional[Dict[str, Any]] = None
    per_run_rows: List[Dict[str, Any]] = []

    total_runs = len(args.seeds) * len(args.algorithms)
    run_idx = 0
    max_workers = max(1, int(getattr(args, "max_workers", 1)))
    args_dict = vars(args).copy()

    def _process_trial(trial: Dict[str, Any], seed_val: int, algo_val: str) -> None:
        nonlocal first_trial_with_ts
        ts = trial.get("timeseries", {}) or {}
        t_series = np.asarray(ts.get("t", []), dtype=float)
        nees_pos_series = np.asarray(ts.get("nees_pos", []), dtype=float) if ts else None
        err_series = np.asarray(ts.get("err_norm", []), dtype=float) if ts else None
        active_count = np.asarray(ts.get("active_count", []), dtype=float)
        active_set = ts.get("active_set", [])
        gdop_xy_series = np.asarray(ts.get("gdop_xy", []), dtype=float) if ts else None
        gdop_3d_series = np.asarray(ts.get("gdop_3d", []), dtype=float) if ts else None

        nis_ac_series = None
        nis_ac_dof = None
        if ts and ts.get("nis_logs") and ts["nis_logs"].get("acoustic"):
            log = ts["nis_logs"]["acoustic"]
            if log.get("values"):
                nis_ac_dof = log.get("dof", None)
                nis_ac_series = fill_sparse_series(t_series, log)

        energy_wh = compute_energy_metrics(trial, args, algo_val)

        pct = _percent_time_by_active_count(active_count)
        switch_count = _count_switches(active_set if isinstance(active_set, list) else list(active_set))
        avg_beacons = compute_avg_beacons(active_count) if active_count.size else np.nan
        nees_pct = float(trial.get("nees_full", {}).get("percent_inside_bounds", np.nan))
        nis_ac_pct = float((trial.get("nis", {}).get("acoustic") or {}).get("percent_inside_bounds", np.nan))
        crlb_eff = compute_crlb_efficiency(trial)

        row = {
            "seed": seed_val,
            "algorithm": algo_val,
            "pos_rmse": float(trial.get("pos_rmse_total", np.nan)),
            "final_error": float(trial.get("final_position_error", np.nan)),
            "energy_Wh": float(energy_wh),
            "switch_count": float(switch_count),
            "avg_beacons": float(avg_beacons),
            "nees_pct": float(nees_pct),
            "nis_acoustic_pct": float(nis_ac_pct),
            "crlb_eff": float(crlb_eff),
            **{f"pct_active_{k}": pct[k] for k in range(5)},
        }
        per_run_rows.append(row)

        results_by_algo[algo_val].append({
            **row,
            "t_series": t_series,
            "nees_pos_series": nees_pos_series,
            "err_series": err_series,
            "nis_ac_series": nis_ac_series,
            "nis_ac_dof": nis_ac_dof,
            "active_count": active_count,
            "gdop_xy_series": gdop_xy_series,
            "gdop_3d_series": gdop_3d_series,
            "selector_meta": trial.get("selector_meta", []),
        })

        seed_dir = out_root / "trials" / f"seed_{seed_val:04d}" / algo_val
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "trial_summary.json").open("w") as f:
            json.dump({"seed": seed_val, **row}, f, indent=2)
        write_timeseries_csv(trial, seed_dir / "timeseries.csv")
        if "selector_meta" in trial:
            with (seed_dir / "selector_meta.json").open("w") as f:
                json.dump(trial["selector_meta"], f, indent=2)

        if first_trial_with_ts is None and trial.get("timeseries"):
            first_trial_with_ts = trial

    if max_workers <= 1:
        for seed_idx, seed in enumerate(args.seeds, start=1):
            rs_cfg = RandomStreamsConfig(args.x0_pos_std, args.x0_vel_std)
            rs = RandomStreams(seed, rs_cfg)

            for algo_idx, algo in enumerate(args.algorithms, start=1):
                run_idx += 1
                print(
                    f"[MC] run {run_idx}/{total_runs} | seed {seed_idx}/{len(args.seeds)}={seed} | "
                    f"algo {algo_idx}/{len(args.algorithms)}={algo}"
                )
                trial = run_trial(algo, seed, args, rs)
                _process_trial(trial, seed, algo)
    else:
        print(f"[MC] parallel mode: max_workers={max_workers}, total_runs={total_runs}")
        payloads = [(algo, seed, args_dict) for seed in args.seeds for algo in args.algorithms]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_run_trial_worker, payload): payload for payload in payloads}
            for idx, fut in enumerate(as_completed(future_map), start=1):
                algo_val, seed_val, trial = fut.result()
                print(f"[MC] completed {idx}/{total_runs} | seed={seed_val} | algo={algo_val}")
                _process_trial(trial, seed_val, algo_val)

    write_per_run_metrics(per_run_rows, out_root / "tables")

    summary_rows = []
    for algo, rows in results_by_algo.items():
        if not rows:
            continue
        pos_rmse = np.asarray([r["pos_rmse"] for r in rows], dtype=float)
        final_err = np.asarray([r["final_error"] for r in rows], dtype=float)
        energy = np.asarray([r["energy_Wh"] for r in rows], dtype=float)
        switch_count = np.asarray([r["switch_count"] for r in rows], dtype=float)
        pct_cols = [f"pct_active_{k}" for k in range(5)]

        def mean_std(arr: np.ndarray) -> Tuple[float, float]:
            return float(np.nanmean(arr)), float(np.nanstd(arr))

        def mae_max(arr: np.ndarray) -> Tuple[float, float]:
            return float(np.nanmean(np.abs(arr))), float(np.nanmax(arr))

        summary = {
            "algorithm": algo,
            "n_runs": len(rows),
            "pos_rmse_mean": mean_std(pos_rmse)[0],
            "pos_rmse_std": mean_std(pos_rmse)[1],
            "pos_rmse_mae": mae_max(pos_rmse)[0],
            "pos_rmse_max": mae_max(pos_rmse)[1],
            "final_error_mean": mean_std(final_err)[0],
            "final_error_std": mean_std(final_err)[1],
            "final_error_mae": mae_max(final_err)[0],
            "final_error_max": mae_max(final_err)[1],
            "energy_mean": mean_std(energy)[0],
            "energy_std": mean_std(energy)[1],
            "energy_mae": mae_max(energy)[0],
            "energy_max": mae_max(energy)[1],
            "switch_count_mean": mean_std(switch_count)[0],
            "switch_count_std": mean_std(switch_count)[1],
        }
        for col in pct_cols:
            summary[col] = float(np.mean([r[col] for r in rows]))
        summary_rows.append(summary)

    write_summary_metrics(summary_rows, out_root / "tables")

    plot_dir = out_root / "figures"
    # Single trajectory plot (first available run)
    if first_trial_with_ts:
        plot_seed_trajectory(first_trial_with_ts, plot_dir / "traj_xy.png")
    else:
        print("[WARN] No trajectory data available for traj_xy.png")

    plot_cdf_overlay(results_by_algo, plot_dir)
    plot_rmse_boxplot(results_by_algo, plot_dir)
    plot_energy_boxplot(results_by_algo, plot_dir)
    plot_energy_vs_error(results_by_algo, plot_dir)
    plot_mean_nees(results_by_algo, plot_dir)
    plot_mean_nis_acoustic(results_by_algo, plot_dir)
    plot_error_vs_time_ci(results_by_algo, plot_dir)
    plot_nees_vs_time_ci(results_by_algo, plot_dir)
    plot_nis_acoustic_vs_time_ci(results_by_algo, plot_dir)
    plot_active_count_vs_time(results_by_algo, plot_dir)
    plot_soc_energy_vs_time(results_by_algo, plot_dir, args)
    plot_gdop_vs_time(results_by_algo, plot_dir)
    plot_fim_logdet_vs_time(results_by_algo, plot_dir)
    plot_rmse_vs_crlb(results_by_algo, plot_dir)

    for name, rows in results_by_algo.items():
        if name.startswith("adaptive"):
            plot_adaptive_telemetry(rows, plot_dir)

    make_results_readme(out_root, args)

    # Policy summary table and terminal findings
    baseline_energy = np.nan
    if "imu_dvl_depth_all4" in results_by_algo and results_by_algo["imu_dvl_depth_all4"]:
        baseline_energy = float(np.nanmean([r["energy_Wh"] for r in results_by_algo["imu_dvl_depth_all4"]]))

    policy_rows = []
    for algo, rows in results_by_algo.items():
        if not rows:
            continue
        pos_rmse_vals = [r["pos_rmse"] for r in rows]
        rmse_stats = mean_std_ci([float(v) for v in pos_rmse_vals if np.isfinite(v)])
        nees_vals = [r.get("nees_pct", np.nan) for r in rows]
        nis_vals = [r.get("nis_acoustic_pct", np.nan) for r in rows]
        avg_beacons = [r.get("avg_beacons", np.nan) for r in rows]
        crlb_eff = [r.get("crlb_eff", np.nan) for r in rows]
        energy_vals = [r.get("energy_Wh", np.nan) for r in rows]

        energy_mean = float(np.nanmean(energy_vals)) if energy_vals else np.nan
        energy_savings = np.nan
        if np.isfinite(baseline_energy) and baseline_energy > 0:
            energy_savings = 100.0 * (1.0 - (energy_mean / baseline_energy))

        policy_rows.append({
            "policy": algo,
            "rmse_mean": rmse_stats["mean"],
            "rmse_std": rmse_stats["std"],
            "rmse_ci_lo": rmse_stats["ci"][0],
            "rmse_ci_hi": rmse_stats["ci"][1],
            "crlb_eff_mean": float(np.nanmean(crlb_eff)) if crlb_eff else np.nan,
            "nees_pct_mean": float(np.nanmean(nees_vals)) if nees_vals else np.nan,
            "nis_pct_mean": float(np.nanmean(nis_vals)) if nis_vals else np.nan,
            "avg_beacons": float(np.nanmean(avg_beacons)) if avg_beacons else np.nan,
            "energy_mean_Wh": energy_mean,
            "energy_savings_pct": energy_savings,
            "switch_count_mean": float(np.nanmean([r.get("switch_count", np.nan) for r in rows])),
        })

    if policy_rows:
        out_tables = out_root / "tables"
        out_tables.mkdir(parents=True, exist_ok=True)
        csv_path = out_tables / "policy_summary.csv"
        md_path = out_tables / "policy_summary.md"
        fieldnames = list(policy_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(policy_rows)
        with md_path.open("w") as f:
            f.write(" | ".join(fieldnames) + "\n")
            f.write(" | ".join(["---"] * len(fieldnames)) + "\n")
            for row in policy_rows:
                f.write(" | ".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k]) for k in fieldnames) + "\n")

        print("\n=== Main findings ===")
        for row in policy_rows:
            nees_ok = np.isfinite(row["nees_pct_mean"]) and row["nees_pct_mean"] >= 95.0
            nis_ok = np.isfinite(row["nis_pct_mean"]) and row["nis_pct_mean"] >= 95.0
            consistency = "The filter remains 95% consistent over the trajectory." if (nees_ok and nis_ok) else "Consistency below 95% on at least one metric."
            print(
                f"[policy={row['policy']}] RMSE={row['rmse_mean']:.3f}±{row['rmse_std']:.3f} "
                f"(95% CI [{row['rmse_ci_lo']:.3f}, {row['rmse_ci_hi']:.3f}]) | "
                f"NEES%={row['nees_pct_mean']:.1f} | NIS%={row['nis_pct_mean']:.1f} | "
                f"avg_beacons={row['avg_beacons']:.2f} | energy_savings={row['energy_savings_pct']:.1f}%"
            )
            print(f"  {consistency}")


def run_self_test(args: argparse.Namespace) -> None:
    args.duration = 30.0
    args.seeds = [0, 1]
    args.algorithms = ["imu_dvl_depth", "imu_dvl_depth_all4", "adaptive"]
    args.self_test_dropout = True

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        rs_cfg = RandomStreamsConfig(args.x0_pos_std, args.x0_vel_std)
        rs = RandomStreams(seed, rs_cfg)

        trials = {}
        for algo in args.algorithms:
            trials[algo] = run_trial(algo, seed, args, rs)

        x0s = []
        for algo in args.algorithms:
            x0s.append(trials[algo].get("x0_perturb"))
        if len({json.dumps(x0, sort_keys=True) for x0 in x0s}) != 1:
            raise RuntimeError("Self-test failed: x0 perturbations are not identical across algorithms.")

        adaptive = trials["adaptive"]
        baseline = trials["imu_dvl_depth_all4"]
        adapt_active = np.asarray(adaptive.get("timeseries", {}).get("active_count", []), dtype=float)
        base_nis = baseline.get("timeseries", {}).get("nis_logs", {}).get("acoustic", {}).get("values", [])
        adapt_nis = adaptive.get("timeseries", {}).get("nis_logs", {}).get("acoustic", {}).get("values", [])

        selector_meta = adaptive.get("selector_meta", []) or []
        selected_sets = []
        for row in selector_meta:
            sel = row.get("selected", [])
            if isinstance(sel, (list, tuple)):
                selected_sets.append(tuple(sel))
        unique_sets = set(selected_sets)

        changed_active_count = adapt_active.size > 1 and not np.all(adapt_active == adapt_active[0])
        changed_set = len(unique_sets) > 1
        reduced_updates = len(adapt_nis) < len(base_nis)

        if not (changed_active_count or changed_set or reduced_updates):
            raise RuntimeError("Self-test failed: adaptive did not change active set or reduce updates.")


def build_config_overrides(config_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    base = CONFIG_PRESETS.get(config_name, {}).copy()
    base.update({
        "duration_sec": float(args.duration_sec),
        "use_currents": False if args.disable_currents else base.get("use_currents", False),
        "trajectory": args.trajectory,
    })
    return base


def run_one_seed(config_name: str, seed: int, args: argparse.Namespace, out_root: Path) -> Dict[str, Any]:
    seed_dir = out_root / config_name / f"seed_{seed:04d}"
    plots_dir = seed_dir / "plots"
    seed_dir.mkdir(parents=True, exist_ok=True)

    overrides = build_config_overrides(config_name, args)
    try:
        trial = run_single_trial(
            seed=seed,
            config_overrides=overrides,
            return_timeseries=True,
            target_names=None,
        )
    except Exception as e:
        err_trial = {"seed": seed, "failed": True, "error": str(e)}
        with (seed_dir / "metrics.json").open("w") as f:
            json.dump(err_trial, f, indent=2)
        return err_trial

    write_timeseries_csv(trial, seed_dir / "timeseries.csv")
    with (seed_dir / "metrics.json").open("w") as f:
        json.dump(metrics_from_trial(trial), f, indent=2)
    plot_per_seed(trial, plots_dir)
    return trial


def run_config(config_name: str, seeds: List[int], args: argparse.Namespace, out_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"[{config_name}] run {i+1}/{len(seeds)} seed={seed}")
        trial = run_one_seed(config_name, seed, args, out_root)
        runs.append(trial)
    aggregate_results(config_name, runs, out_root / config_name / "aggregate")
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo runner (spiral, no currents)")

    # New pipeline args (minimal CLI; load the rest from YAML)
    parser.add_argument("--outdir", type=str, default=None, help="Output directory root")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration (s)")
    parser.add_argument("--runs", type=int, default=None, help="Number of Monte Carlo runs")
    parser.add_argument("--config", type=str, default="mc_config.yaml", help="YAML config for MC settings")
    parser.add_argument("--self-test", action="store_true")

    # Legacy args (kept for compatibility)
    parser.add_argument("--out", type=str, default="results_mc", help="(legacy) Output root directory")
    parser.add_argument(
        "--config-name",
        choices=["imu_dvl_depth", "imu_dvl_depth_acoustic_all", "all"],
        default="all",
        help="(legacy) Which sensor config to run",
    )
    parser.add_argument("--duration-sec", type=float, default=180.0, help="(legacy) Simulation duration in seconds")
    parser.add_argument("--disable-currents", dest="disable_currents", action="store_true", help="(legacy) Disable currents (default)")
    parser.add_argument("--enable-currents", dest="disable_currents", action="store_false", help="(legacy) Enable currents")
    parser.set_defaults(disable_currents=True)
    parser.add_argument("--trajectory", choices=["lawnmower", "spiral", "concentric", "figure8"], default="spiral",
                        help="(legacy) Trajectory to use")
    parser.add_argument("--legacy", action="store_true", help="Use legacy two-config runner")

    # Parallel control
    parser.add_argument("--max-workers", type=int, default=None, help="Process pool workers for MC runs (default from YAML or 1)")

    return parser.parse_args()


def run_mc_legacy(args: argparse.Namespace):
    _set_style()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds if isinstance(args.seeds, str) else ",".join(args.seeds or []))

    configs = [c for c in CONFIG_PRESETS.keys()] if args.config_name == "all" else [args.config_name]
    results_by_config: Dict[str, List[Dict[str, Any]]] = {}
    for cfg in configs:
        runs = run_config(cfg, seeds, args, out_root)
        results_by_config[cfg] = runs

    compare_configs(results_by_config, out_root)


if __name__ == "__main__":
    args = parse_args()
    if args.legacy:
        run_mc_legacy(args)
    else:
        cfg_data = _load_mc_config(args.config)
        merged = dict(DEFAULT_MC_CONFIG)
        merged.update(cfg_data)
        for key, value in merged.items():
            # Respect CLI overrides when provided (e.g., max_workers)
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

        if args.outdir is None:
            args.outdir = cfg_data.get("outdir", "results_monte_carlo/spiral_T180_N20")
        if args.duration is None:
            args.duration = float(cfg_data.get("duration", 180.0))
        if args.runs is None:
            args.runs = int(cfg_data.get("runs", 20))
        if args.algorithms is None:
            args.algorithms = list(DEFAULT_MC_CONFIG["algorithms"])

        seeds_cfg = args.seeds
        if seeds_cfg is None or len(seeds_cfg) == 0:
            args.seeds = list(range(int(args.runs)))
        else:
            if isinstance(seeds_cfg, str):
                seeds_list = [s.strip() for s in seeds_cfg.split(",") if s.strip()]
            else:
                seeds_list = [str(s) for s in seeds_cfg]
            args.seeds = [int(s) for s in seeds_list]
            if args.runs is not None and len(args.seeds) > int(args.runs):
                args.seeds = args.seeds[:int(args.runs)]
            args.runs = len(args.seeds)

        if args.self_test:
            run_self_test(args)
        else:
            run_mc_new(args)
