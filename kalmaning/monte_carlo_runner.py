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

from current_acoustic_EKF_patched import (
    run_single_trial,
    DVL_VEL_STD,
    DEPTH_STD,
    ACOUSTIC_RANGE_STD,
)
from validation_metrics import downsampled_mean_nees_test


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


def save_plots(out_dir: Path, runs: List[Dict[str, Any]]):
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_rmse = [r["pos_rmse_total"] for r in runs]
    final_err = [r["final_position_error"] for r in runs]
    nees_full = [r["nees_full"]["avg_nees"] for r in runs]
    nis_dvl = [r["nis"]["dvl"]["avg_nis"] for r in runs if r["nis"].get("dvl")]
    nis_depth = [r["nis"]["depth"]["avg_nis"] for r in runs if r["nis"].get("depth")]
    nis_ac = [r["nis"]["acoustic"]["avg_nis"] for r in runs if r["nis"].get("acoustic")]

    plt.figure()
    plt.hist(pos_rmse, bins=20, color="purple", edgecolor="black", alpha=0.8)
    plt.xlabel("Position RMSE [m]")
    plt.ylabel("Count")
    plt.title("MC histogram: position RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_pos_rmse.png", dpi=200)
    plt.close()

    plt.figure()
    sorted_err = np.sort(final_err)
    cdf = np.linspace(0, 1, len(sorted_err))
    plt.plot(sorted_err, cdf, color="teal")
    plt.xlabel("Final position error [m]")
    plt.ylabel("CDF")
    plt.title("MC CDF: final error")
    plt.tight_layout()
    plt.savefig(out_dir / "cdf_final_error.png", dpi=200)
    plt.close()

    plt.figure()
    data = [nees_full]
    labels = ["NEES avg"]
    if nis_dvl:
        data.append(nis_dvl)
        labels.append("NIS DVL")
    if nis_depth:
        data.append(nis_depth)
        labels.append("NIS Depth")
    if nis_ac:
        data.append(nis_ac)
        labels.append("NIS Acoustic")
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.ylabel("Metric value")
    plt.title("MC boxplot: NEES/NIS averages (run means)")
    plt.tight_layout()
    plt.savefig(out_dir / "box_nees_nis.png", dpi=200)
    plt.close()


def decimate_timeseries(ts: Dict[str, np.ndarray], target_len: int = 500) -> Dict[str, np.ndarray]:
    t = ts["t"]
    if t.size == 0:
        return ts
    step = max(1, int(math.ceil(t.size / target_len)))
    sl = slice(0, None, step)
    return {k: v[sl] for k, v in ts.items()}


def save_timeseries_bundle(out_dir: Path, runs: List[Dict[str, Any]], mode: str):
    if mode == "none":
        return
    ts_dir = out_dir / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    for idx, run in enumerate(runs):
        ts = run.get("timeseries")
        if ts is None:
            continue
        decim = decimate_timeseries(ts)
        np.savez(ts_dir / f"run_{idx:03d}.npz", **decim)
        collected.append(decim)

    if mode == "mean" and collected:
        min_len = min(ts["t"].shape[0] for ts in collected)
        t_ref = collected[0]["t"][:min_len]
        est_pos = np.stack([ts["est_pos"][:min_len] for ts in collected])
        true_pos = np.stack([ts["true_pos"][:min_len] for ts in collected])
        mean_bundle = {
            "t": t_ref,
            "est_pos_mean": np.mean(est_pos, axis=0),
            "est_pos_std": np.std(est_pos, axis=0, ddof=1),
            "true_pos_mean": np.mean(true_pos, axis=0),
        }
        np.savez(ts_dir / "timeseries_mean.npz", **mean_bundle)


def write_per_run(out_dir: Path, runs: List[Dict[str, Any]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_run.csv"
    jsonl_path = out_dir / "per_run.jsonl"

    fieldnames = [
        "seed",
        "runtime_seconds",
        "pos_rmse_total",
        "pos_rmse_x",
        "pos_rmse_y",
        "pos_rmse_z",
        "vel_rmse_total",
        "vel_rmse_x",
        "vel_rmse_y",
        "vel_rmse_z",
        "final_position_error",
        "nees_full_avg",
        "nees_full_lower",
        "nees_full_upper",
        "nees_full_percent_inside",
        "nees_full_is_consistent",
        "nees_pos_avg",
        "nees_pos_lower",
        "nees_pos_upper",
        "nees_pos_percent_inside",
        "nis_dvl_avg",
        "nis_dvl_lower",
        "nis_dvl_upper",
        "nis_dvl_percent_inside",
        "nis_depth_avg",
        "nis_depth_lower",
        "nis_depth_upper",
        "nis_depth_percent_inside",
        "nis_acoustic_avg",
        "nis_acoustic_lower",
        "nis_acoustic_upper",
        "nis_acoustic_percent_inside",
        "run_consistent",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            row = {
                "seed": r["seed"],
                "runtime_seconds": r["runtime_seconds"],
                "pos_rmse_total": r["pos_rmse_total"],
                "pos_rmse_x": r["pos_rmse_xyz"][0],
                "pos_rmse_y": r["pos_rmse_xyz"][1],
                "pos_rmse_z": r["pos_rmse_xyz"][2],
                "vel_rmse_total": r["vel_rmse_total"],
                "vel_rmse_x": r["vel_rmse_xyz"][0],
                "vel_rmse_y": r["vel_rmse_xyz"][1],
                "vel_rmse_z": r["vel_rmse_xyz"][2],
                "final_position_error": r["final_position_error"],
                "nees_full_avg": r["nees_full"]["avg_nees"],
                "nees_full_lower": r["nees_full"]["lower_bound"],
                "nees_full_upper": r["nees_full"]["upper_bound"],
                "nees_full_percent_inside": r["nees_full"]["percent_inside_bounds"],
                "nees_full_is_consistent": r["nees_full"]["is_consistent"],
                "nees_pos_avg": r["nees_pos"]["avg_nees"],
                "nees_pos_lower": r["nees_pos"]["lower_bound"],
                "nees_pos_upper": r["nees_pos"]["upper_bound"],
                "nees_pos_percent_inside": r["nees_pos"]["percent_inside_bounds"],
            }

            nis = r["nis"]
            for key, prefix in [("dvl", "nis_dvl"), ("depth", "nis_depth"), ("acoustic", "nis_acoustic")]:
                res = nis.get(key)
                if res is None:
                    row[f"{prefix}_avg"] = ""
                    row[f"{prefix}_lower"] = ""
                    row[f"{prefix}_upper"] = ""
                    row[f"{prefix}_percent_inside"] = ""
                else:
                    row[f"{prefix}_avg"] = res["avg_nis"]
                    row[f"{prefix}_lower"] = res["lower_bound"]
                    row[f"{prefix}_upper"] = res["upper_bound"]
                    row[f"{prefix}_percent_inside"] = res["percent_inside_bounds"]

            row["run_consistent"] = r.get("run_consistent", "")

            writer.writerow(row)

    with jsonl_path.open("w") as f:
        for r in runs:
            clean = {k: v for k, v in r.items() if k != "timeseries"}
            f.write(json.dumps(clean) + "\n")


def run_mc(args: argparse.Namespace):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.runs)]

    avg_band = tuple(args.consistency_avg_band) if args.consistency_avg_band else None

    config_overrides = {
        "use_currents": args.currents == "on",
        "dvl_extra_std": args.dvl_extra_std,
        "depth_extra_std": args.depth_extra_std,
        "range_extra_std": args.range_extra_std,
        "imu_accel_extra_std": args.imu_accel_extra_std,
        "imu_bias_rw_std": args.imu_bias_rw_std,
        "nees_ds_stride": args.nees_ds_stride,
        "nees_ds_alpha": args.nees_ds_alpha,
    }

    if args.trajectory:
        config_overrides["trajectory"] = args.trajectory

    if args.q_vel_std is not None:
        config_overrides["q_vel_std"] = args.q_vel_std

    if args.meas_scale != 1.0:
        s = args.meas_scale
        config_overrides.update({
            "dvl_measurement_std": DVL_VEL_STD * s,
            "depth_measurement_std": DEPTH_STD * s,
            "acoustic_measurement_std": ACOUSTIC_RANGE_STD * s,
        })

    runs: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"[RUN {i+1}/{len(seeds)}] seed={seed}")
        trial = run_single_trial(
            seed=seed,
            config_overrides=config_overrides,
            return_timeseries=args.save_timeseries != "none",
            target_names=args.target_name,
        )
        trial["run_consistent"] = run_consistency_flag(
            trial,
            percent_threshold=args.consistency_percent,
            avg_band=avg_band,
            require_ds=args.consistency_require_ds,
        )
        runs.append(trial)

    # Aggregate metrics
    pos_rmse_vals = [r["pos_rmse_total"] for r in runs]
    final_err_vals = [r["final_position_error"] for r in runs]
    vel_rmse_vals = [r["vel_rmse_total"] for r in runs]
    nees_consistent = [1 if r.get("run_consistent", False) else 0 for r in runs]

    summary = {
        "runs": len(runs),
        "seeds": seeds,
        "config": runs[0]["config"] if runs else config_overrides,
        "pos_rmse_total": mean_std_ci(pos_rmse_vals, bootstrap=args.bootstrap),
        "vel_rmse_total": mean_std_ci(vel_rmse_vals, bootstrap=args.bootstrap),
        "final_position_error": mean_std_ci(final_err_vals, bootstrap=args.bootstrap),
        "final_position_error_percentiles": final_error_stats(final_err_vals),
        "nees_consistency_rate": float(np.mean(nees_consistent) * 100.0) if runs else np.nan,
        "nees_full_avg": mean_std_ci([r["nees_full"]["avg_nees"] for r in runs]),
        "nees_pos_avg": mean_std_ci([r["nees_pos"]["avg_nees"] for r in runs]),
        "nis_dvl_avg": mean_std_ci([r["nis"]["dvl"]["avg_nis"] for r in runs if r["nis"].get("dvl")]),
        "nis_depth_avg": mean_std_ci([r["nis"]["depth"]["avg_nis"] for r in runs if r["nis"].get("depth")]),
        "nis_acoustic_avg": mean_std_ci([r["nis"]["acoustic"]["avg_nis"] for r in runs if r["nis"].get("acoustic")]),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    write_per_run(out_dir, runs)
    save_plots(out_dir, runs)
    save_timeseries_bundle(out_dir, runs, args.save_timeseries)

    pos_stats = summary["pos_rmse_total"]
    final_stats = summary["final_position_error"]
    rate = summary["nees_consistency_rate"]
    fe_stats = summary["final_position_error_percentiles"]
    print(
        f"MC({len(runs)}): pos_rmse={pos_stats['mean']:.3f}±{pos_stats['std']:.3f} m, "
        f"final_err={final_stats['mean']:.3f}±{final_stats['std']:.3f} m (med={fe_stats['median']:.3f}, p90={fe_stats['p90']:.3f}, p95={fe_stats['p95']:.3f}), "
        f"run-consistency={rate:.1f}%"
        + (f" | q_vel_std={args.q_vel_std}" if args.q_vel_std is not None else "")
        + (f" | meas_scale={args.meas_scale}" if args.meas_scale != 1.0 else "")
        + (f" | traj={args.trajectory}" if args.trajectory else "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation harness for HoloOcean EKF")
    parser.add_argument("--runs", type=int, default=50, help="Number of Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--out", type=str, default="results_mc", help="Output directory")
    parser.add_argument("--currents", choices=["on", "off"], default="on", help="Enable currents")
    parser.add_argument("--save-timeseries", choices=["none", "decimated", "mean"], default="none")
    parser.add_argument("--target-name", action="append", help="Target agent name (repeatable)")

    parser.add_argument("--dvl-extra-std", type=float, default=0.0, help="Extra DVL noise std (m/s)")
    parser.add_argument("--depth-extra-std", type=float, default=0.0, help="Extra depth noise std (m)")
    parser.add_argument("--range-extra-std", type=float, default=0.0, help="Extra acoustic range noise std (m)")
    parser.add_argument("--imu-accel-extra-std", type=float, default=0.0, help="Extra IMU accel noise std (m/s^2)")
    parser.add_argument("--imu-bias-rw-std", type=float, default=0.0, help="IMU accel bias RW std (m/s^2/sqrt(s))")

    parser.add_argument("--q-vel-std", type=float, default=None, help="Override process accel std (q_vel_std)")
    parser.add_argument("--meas-scale", type=float, default=1.0, help="Scale factor applied to DVL/depth/range measurement stds")

    parser.add_argument("--trajectory", choices=["lawnmower", "spiral", "concentric", "figure8"], default=None,
                        help="Trajectory for waypoint generation (default=lawnmower)")

    parser.add_argument("--consistency-percent", type=float, default=90.0,
                        help="Minimum percent of NEES samples inside per-sample chi2 gate to mark run consistent")
    parser.add_argument("--consistency-avg-band", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Optional band on avg NEES as multiples of dof (e.g., 0.7 1.3)")
    parser.add_argument("--consistency-require-ds", action="store_true",
                        help="Require downsampled mean-NEES chi-square test to pass")
    parser.add_argument("--nees-ds-stride", type=int, default=0,
                        help="Downsample stride for optional mean-NEES chi2 test (0 disables)")
    parser.add_argument("--nees-ds-alpha", type=float, default=0.05,
                        help="Alpha for downsampled mean-NEES chi2 test")

    parser.add_argument("--bootstrap", type=int, default=None, help="Bootstrap samples for CI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mc(args)
