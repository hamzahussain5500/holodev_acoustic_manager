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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2

from current_acoustic_EKF_patched import (
    run_single_trial,
    DVL_VEL_STD,
    DEPTH_STD,
    ACOUSTIC_RANGE_STD,
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
        "figure.dpi": 300,
        "axes.grid": True,
        "axes.facecolor": "#f8f8f8",
        "grid.alpha": 0.4,
        "font.size": 10,
    })


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
    plt.plot(gt[:, 0], gt[:, 1], label="Ground truth", lw=2)
    plt.plot(est[:, 0], est[:, 1], label="Estimate", lw=1.5)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.title("XY trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "traj_xy.png")
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(t, err_norm, label="||pos error||", lw=1.5)
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title("Position error vs time")
    plt.tight_layout()
    plt.savefig(plots_dir / "pos_err_vs_time.png")
    plt.close()

    if nees_pos is not None:
        plt.figure(figsize=(6, 3.5))
        plt.plot(t, nees_pos, lw=1.2, label="NEES pos")
        dof = 3
        lo = chi2.ppf(0.025, dof)
        hi = chi2.ppf(0.975, dof)
        plt.axhline(lo, color="gray", linestyle="--", linewidth=1, label="95% bounds")
        plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("time [s]")
        plt.ylabel("NEES (pos)")
        plt.title("NEES vs time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "nees_vs_time.png")
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
        plt.xlabel("time [s]")
        plt.ylabel("NIS")
        plt.title(f"NIS {key} vs time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / fname)
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
    plt.plot(sorted_err, cdf, lw=1.5, label=config_name)
    plt.xlabel("Final position error [m]")
    plt.ylabel("CDF")
    plt.title("Final error CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "final_error_cdf.png")
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.boxplot([pos_rmse], labels=["pos_rmse"], patch_artist=True)
    plt.ylabel("RMSE [m]")
    plt.title("Position RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_boxplot.png")
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.boxplot([nees_pct], labels=["NEES % inside"], patch_artist=True)
    plt.ylabel("Percent inside gate [%]")
    plt.title("NEES coverage")
    plt.tight_layout()
    plt.savefig(out_dir / "nees_boxplot.png")
    plt.close()

    # Mean NEES over time (optional)
    if ok_runs and ok_runs[0].get("timeseries"):
        min_len = min(r["timeseries"]["nees_pos"].shape[0] for r in ok_runs if r.get("timeseries"))
        if min_len > 0:
            nees_stack = np.stack([r["timeseries"]["nees_pos"][:min_len] for r in ok_runs])
            t_ref = ok_runs[0]["timeseries"]["t"][:min_len]
            mean_nees = np.nanmean(nees_stack, axis=0)
            plt.figure(figsize=(6, 3.5))
            plt.plot(t_ref, mean_nees, lw=1.5, label="Mean NEES pos")
            lo = chi2.ppf(0.025, 3)
            hi = chi2.ppf(0.975, 3)
            plt.axhline(lo, color="gray", linestyle="--", linewidth=1, label="95% bounds")
            plt.axhline(hi, color="gray", linestyle="--", linewidth=1)
            plt.xlabel("time [s]")
            plt.ylabel("NEES (pos)")
            plt.title("Mean NEES over time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "mean_nees_over_time.png")
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
        plt.plot(final_err, cdf, lw=1.5, label=name)
    plt.xlabel("Final position error [m]")
    plt.ylabel("CDF")
    plt.title("Final error CDF (configs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "final_error_cdf_overlay.png")
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
        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.ylabel("Position RMSE [m]")
        plt.title("RMSE by config")
        plt.tight_layout()
        plt.savefig(out_dir / "rmse_boxplot_overlay.png")
        plt.close()


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
    parser = argparse.ArgumentParser(description="Monte Carlo harness (short runs, two configs)")
    parser.add_argument("--out", type=str, default="results_mc", help="Output root directory")
    parser.add_argument("--config-name", choices=["imu_dvl_depth", "imu_dvl_depth_acoustic_all", "all"], default="all",
                        help="Which sensor config to run")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated seed list (default fixed set)")
    parser.add_argument("--duration-sec", type=float, default=180.0, help="Simulation duration in seconds")
    parser.add_argument("--disable-currents", dest="disable_currents", action="store_true", help="Disable currents (default)")
    parser.add_argument("--enable-currents", dest="disable_currents", action="store_false", help="Enable currents")
    parser.set_defaults(disable_currents=True)
    parser.add_argument("--trajectory", choices=["lawnmower", "spiral", "concentric", "figure8"], default="spiral",
                        help="Trajectory to use")
    return parser.parse_args()


def run_mc(args: argparse.Namespace):
    _set_style()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seeds)

    configs = [c for c in CONFIG_PRESETS.keys()] if args.config_name == "all" else [args.config_name]
    results_by_config: Dict[str, List[Dict[str, Any]]] = {}
    for cfg in configs:
        runs = run_config(cfg, seeds, args, out_root)
        results_by_config[cfg] = runs

    compare_configs(results_by_config, out_root)


if __name__ == "__main__":
    args = parse_args()
    run_mc(args)
