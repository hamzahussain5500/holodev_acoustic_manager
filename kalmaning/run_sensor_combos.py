import os
import sys
import numpy as np

# Fix mpl_toolkits namespace conflict: system dist-packages ships a stale
# version that fails to import with the pip-installed matplotlib.  We must
# (1) remove the system path, (2) clear any cached mpl_toolkits refs, and
# (3) force reimport from the pip location.
sys.path = [p for p in sys.path if "/usr/lib/python3/dist-packages" not in p]
for _k in list(sys.modules.keys()):
    if "mpl_toolkits" in _k:
        del sys.modules[_k]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
import csv

from current_acoustic_EKF_patched import run_single_trial
from uncertainty_utils import covariance_ellipse_points

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_sensor_combos")


# Spiral parameters matching the Monte Carlo runs (current_acoustic_EKF_patched.py defaults)
SPIRAL_CFG = {
    "trajectory": "spiral",
    "spiral_center": (200.0, -200.0),
    "spiral_min_radius": 20.0,
    "spiral_max_radius": 50.0,
    "spiral_turns": 6,
    "spiral_points_per_rev": 250,
    "spiral_z_start": -5.0,
    "spiral_z_end": -150.0,
}

# Sensor combinations to evaluate (all with spiral trajectory)
COMBOS = [
    {
        "name": "imu_only",
        "config": {
            **SPIRAL_CFG,
            "use_dvl_update": False,
            "use_depth_update": False,
            "use_acoustic_updates": False,
        },
        "targets": None,
    },
    {
        "name": "imu_dvl",
        "config": {
            **SPIRAL_CFG,
            "use_dvl_update": True,
            "use_depth_update": False,
            "use_acoustic_updates": False,
        },
        "targets": None,
    },
    {
        "name": "imu_dvl_depth",
        "config": {
            **SPIRAL_CFG,
            "use_dvl_update": True,
            "use_depth_update": True,
            "use_acoustic_updates": False,
        },
        "targets": None,
    },
    {
        "name": "imu_dvl_depth_usv1",
        "config": {
            **SPIRAL_CFG,
            "use_dvl_update": True,
            "use_depth_update": True,
            "use_acoustic_updates": True,
        },
        "targets": ["usv1"],
    },
    {
        "name": "imu_dvl_depth_usv1_usv2",
        "config": {
            **SPIRAL_CFG,
            "use_dvl_update": True,
            "use_depth_update": True,
            "use_acoustic_updates": True,
        },
        "targets": ["usv1", "usv2"],
    },
    {
        "name": "imu_dvl_depth_usv1_usv2_usv3",
        "config": {
            **SPIRAL_CFG,
            "use_dvl_update": True,
            "use_depth_update": True,
            "use_acoustic_updates": True,
        },
        "targets": ["usv1", "usv2", "usv3"],
    },

    {
        "name": "imu_dvl_depth_usv1_usv2_usv3_usv4",
        "config": {
            **SPIRAL_CFG,
            "use_dvl_update": True,
            "use_depth_update": True,
            "use_acoustic_updates": True,
        },
        "targets": ["usv1", "usv2", "usv3", "usv4"],
    }
]


def run_combo(name: str, config: dict, targets, seed: int = 123):
    """Run one trial with specified sensor configuration and optional acoustic targets."""
    trial = run_single_trial(
        seed=seed,
        config_overrides=config,
        return_timeseries=True,
        target_names=targets,
    )
    return trial


def collect_results(seed: int = 123):
    results = []
    for combo in COMBOS:
        print(f"\n=== Running combo: {combo['name']} ===")
        trial = run_combo(combo["name"], combo["config"], combo["targets"], seed=seed)
        results.append({"name": combo["name"], "trial": trial})
    return results


def print_metrics_summary(results):
    if not results:
        return

    def fmt(v):
        return "nan" if v is None else f"{v:.3f}"

    def fmt_pct(v):
        return "nan" if v is None else f"{v:.1f}%"

    print("\n=== Metrics Summary ===")
    header = (
        f"{'combo':30s}  {'dist_m':>8s}  {'pos_rmse':>8s}  {'vel_rmse':>8s}  {'final_err':>9s}  "
        f"{'pos95_x':>8s}  {'pos95_y':>8s}  {'pos95_z':>8s}  "
        f"{'NEES_avg':>8s}  {'NEES_ok':>7s}  {'NEES_in':>7s}  "
        f"{'NIS_dvl':>8s}  {'dvl_in':>7s}  {'NIS_depth':>10s}  {'depth_in':>9s}  {'NIS_acoustic':>12s}  {'ac_in':>6s}"
    )
    print(header)
    print("-" * len(header))

    for res in results:
        trial = res["trial"]
        name = res["name"]
        dist_m = trial.get("distance_true_m", float("nan"))
        pos_rmse = trial.get("pos_rmse_total", float("nan"))
        vel_rmse = trial.get("vel_rmse_total", float("nan"))
        final_err = trial.get("final_position_error", float("nan"))

        # Uncertainty (final 95% half-widths per axis using chi2 1 dof @95%)
        pos95_x = pos95_y = pos95_z = None
        ts = trial.get("timeseries", {}) or {}
        Ppos = ts.get("Ppos", None)
        if Ppos is not None and len(Ppos) > 0:
            chi2_1d_95 = 3.841
            diag = np.diag(Ppos[-1])
            half = np.sqrt(diag * chi2_1d_95)
            pos95_x, pos95_y, pos95_z = half.tolist()

        nees = trial.get("nees_full", {}) or {}
        nees_avg = nees.get("avg_nees", float("nan"))
        nees_ok = nees.get("is_consistent", False)
        nees_pct = nees.get("percent_inside_bounds", None)
        nis = trial.get("nis", {}) or {}
        nis_dvl = nis.get("dvl", {}).get("avg_nis") if nis.get("dvl") else None
        nis_dvl_pct = nis.get("dvl", {}).get("percent_inside_bounds") if nis.get("dvl") else None
        nis_depth = nis.get("depth", {}).get("avg_nis") if nis.get("depth") else None
        nis_depth_pct = nis.get("depth", {}).get("percent_inside_bounds") if nis.get("depth") else None
        nis_ac = nis.get("acoustic", {}).get("avg_nis") if nis.get("acoustic") else None
        nis_ac_pct = nis.get("acoustic", {}).get("percent_inside_bounds") if nis.get("acoustic") else None

        print(
            f"{name:30s}  {fmt(dist_m):>8s}  {fmt(pos_rmse):>8s}  {fmt(vel_rmse):>8s}  {fmt(final_err):>9s}  "
            f"{fmt(pos95_x):>8s}  {fmt(pos95_y):>8s}  {fmt(pos95_z):>8s}  "
            f"{fmt(nees_avg):>8s}  {str(nees_ok):>7s}  {fmt_pct(nees_pct):>7s}  "
            f"{fmt(nis_dvl):>8s}  {fmt_pct(nis_dvl_pct):>7s}  {fmt(nis_depth):>10s}  {fmt_pct(nis_depth_pct):>9s}  {fmt(nis_ac):>12s}  {fmt_pct(nis_ac_pct):>6s}"
        )


def write_metrics_csv(results, path=None):
    if not results:
        return
    if path is None:
        path = os.path.join(RESULTS_DIR, "sensor_combo_summary.csv")

    fields = [
        "combo",
        "distance_true_m",
        "distance_est_m",
        "pos_rmse",
        "vel_rmse",
        "final_err",
        "pos95_x",
        "pos95_y",
        "pos95_z",
        "nees_avg",
        "nees_ok",
        "nees_percent_inside",
        "nis_dvl",
        "nis_dvl_percent_inside",
        "nis_depth",
        "nis_depth_percent_inside",
        "nis_acoustic",
        "nis_acoustic_percent_inside",
        "runtime_seconds",
    ]

    def safe(val):
        if val is None:
            return ""
        if isinstance(val, bool):
            return str(val)
        try:
            return float(val)
        except Exception:
            return val

    rows = []
    for res in results:
        trial = res.get("trial", {})
        nees = trial.get("nees_full", {}) or {}
        nis = trial.get("nis", {}) or {}

        pos95_x = pos95_y = pos95_z = None
        ts = trial.get("timeseries", {}) or {}
        Ppos = ts.get("Ppos", None)
        if Ppos is not None and len(Ppos) > 0:
            chi2_1d_95 = 3.841
            diag = np.diag(Ppos[-1])
            half = np.sqrt(diag * chi2_1d_95)
            pos95_x, pos95_y, pos95_z = half.tolist()

        rows.append({
            "combo": res.get("name", ""),
            "distance_true_m": safe(trial.get("distance_true_m")),
            "distance_est_m": safe(trial.get("distance_est_m")),
            "pos_rmse": safe(trial.get("pos_rmse_total")),
            "vel_rmse": safe(trial.get("vel_rmse_total")),
            "final_err": safe(trial.get("final_position_error")),
            "pos95_x": safe(pos95_x),
            "pos95_y": safe(pos95_y),
            "pos95_z": safe(pos95_z),
            "nees_avg": safe(nees.get("avg_nees")),
            "nees_ok": safe(nees.get("is_consistent")),
            "nees_percent_inside": safe(nees.get("percent_inside_bounds")),
            "nis_dvl": safe(nis.get("dvl", {}).get("avg_nis") if nis.get("dvl") else None),
            "nis_dvl_percent_inside": safe(nis.get("dvl", {}).get("percent_inside_bounds") if nis.get("dvl") else None),
            "nis_depth": safe(nis.get("depth", {}).get("avg_nis") if nis.get("depth") else None),
            "nis_depth_percent_inside": safe(nis.get("depth", {}).get("percent_inside_bounds") if nis.get("depth") else None),
            "nis_acoustic": safe(nis.get("acoustic", {}).get("avg_nis") if nis.get("acoustic") else None),
            "nis_acoustic_percent_inside": safe(nis.get("acoustic", {}).get("percent_inside_bounds") if nis.get("acoustic") else None),
            "runtime_seconds": safe(trial.get("runtime_seconds")),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {path}")


def plot_comparisons(results, out_dir=None):
    if not results:
        print("No results to plot.")
        return

    if out_dir is None:
        out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    def _save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                        dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {name}")

    true_ref = results[0]["trial"]["timeseries"]["true_pos"]
    t_ref = results[0]["trial"]["timeseries"]["t"]

    # Semantic colors
    colors = {
        "truth": "black",
        "imu_dvl": "#1f77b4",  # blue
        "imu_dvl_depth": "#2ca02c",  # green
        "imu_dvl_depth_usv1_usv2_usv3_usv4": "#d62728",  # red
    }

    # Helper to fetch trials by name
    def get_trial(name):
        for res in results:
            if res["name"] == name:
                return res["trial"]
        return None

    primary_names = [
        "imu_only",
        "imu_dvl",
        "imu_dvl_depth",
        "imu_dvl_depth_usv1_usv2_usv3_usv4",
    ]

    # Depth vs time (simplified set)
    fig, ax = plt.subplots()
    ax.plot(t_ref, true_ref[:, 2], color=colors["truth"], linewidth=2.0, label="Ground truth")
    for name in primary_names:
        tr = get_trial(name)
        if tr is None:
            continue
        ts = tr["timeseries"]
        t = ts["t"]
        est = ts["est_pos"]
        label = {
            "imu_dvl": "IMU+DVL",
            "imu_dvl_depth": "IMU+DVL+Depth",
            "imu_dvl_depth_usv1_usv2_usv3_usv4": "IMU+DVL+Depth+Acoustic",
        }.get(name, name)
        ax.plot(t, est[:, 2], color=colors.get(name, "gray"), linewidth=1.6, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [m]")
    ax.set_title("Depth vs time (spiral)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "depth_vs_time")

    # XY trajectory comparison (qualitative)
    fig, ax = plt.subplots()
    ax.plot(true_ref[:, 0], true_ref[:, 1], color=colors["truth"], linewidth=2.0, label="Ground truth")
    for name in primary_names:
        tr = get_trial(name)
        if tr is None:
            continue
        est = tr["timeseries"]["est_pos"]
        label = {
            "imu_dvl": "IMU+DVL",
            "imu_dvl_depth": "IMU+DVL+Depth",
            "imu_dvl_depth_usv1_usv2_usv3_usv4": "IMU+DVL+Depth+Acoustic",
        }.get(name, name)
        ax.plot(est[:, 0], est[:, 1], color=colors.get(name, "gray"), linewidth=1.6, label=label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory comparison (spiral)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    _save(fig, "traj_xy")

    # Position error magnitude vs time (reduced set)
    fig, ax = plt.subplots()
    for name in primary_names:
        tr = get_trial(name)
        if tr is None:
            continue
        ts = tr["timeseries"]
        t = ts["t"]
        est = ts["est_pos"]
        n = min(len(t_ref), len(t), len(est), len(true_ref))
        err = np.linalg.norm(est[:n] - true_ref[:n], axis=1)
        label = {
            "imu_dvl": "IMU+DVL",
            "imu_dvl_depth": "IMU+DVL+Depth",
            "imu_dvl_depth_usv1_usv2_usv3_usv4": "IMU+DVL+Depth+Acoustic",
        }.get(name, name)
        ax.plot(t[:n], err, color=colors.get(name, "gray"), linewidth=1.6, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||position error|| [m]")
    ax.set_title("Position error vs time")
    ax.legend()
    fig.tight_layout()
    _save(fig, "pos_error_vs_time")

    # 3D trajectory (qualitative) showing acoustic localization
    best = get_trial("imu_dvl_depth_usv1_usv2_usv3_usv4")
    if best is not None:
        ts = best["timeseries"]
        est = ts["est_pos"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(true_ref[:, 0], true_ref[:, 1], true_ref[:, 2], color=colors["truth"], linewidth=2.0, label="Ground truth")
        ax.plot(est[:, 0], est[:, 1], est[:, 2], color=colors["imu_dvl_depth_usv1_usv2_usv3_usv4"], linewidth=1.8, label="IMU+DVL+Depth+Acoustic")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("3D spiral trajectory (acoustic localization)")
        ax.legend()
        _save(fig, "traj_3d")

    # 2D uncertainty ellipses for best configuration (acoustic)
    best = get_trial("imu_dvl_depth_usv1_usv2_usv3_usv4")
    if best is not None:
        ts = best["timeseries"]
        est = ts["est_pos"]
        Ppos = ts.get("Ppos")
        if Ppos is not None and len(Ppos) > 0:
            fig, ax = plt.subplots()
            ax.plot(est[:, 0], est[:, 1], color=colors["imu_dvl_depth_usv1_usv2_usv3_usv4"], linewidth=1.6, label="IMU+DVL+Depth+Acoustic")
            ax.plot(true_ref[:, 0], true_ref[:, 1], color=colors["truth"], linewidth=2.0, label="Ground truth")
            num_ellipses = 6
            idxs = np.linspace(0, len(est) - 1, num_ellipses, dtype=int)
            for idx in idxs:
                P_xy = Ppos[idx][0:2, 0:2]
                mean_xy = est[idx, 0:2]
                ex, ey = covariance_ellipse_points(P_xy, mean_xy, chi2_val=5.991, num_points=80)
                ax.plot(ex, ey, color=colors["imu_dvl_depth_usv1_usv2_usv3_usv4"], alpha=0.5, linewidth=1.0)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title("Uncertainty ellipses (95%) — best config")
            ax.set_aspect("equal")
            ax.legend()
            fig.tight_layout()
            _save(fig, "uncertainty_ellipses")

    print(f"\nAll figures saved to {out_dir}/")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = collect_results(seed=123)
    print_metrics_summary(results)
    write_metrics_csv(results, path=os.path.join(RESULTS_DIR, "sensor_combo_summary.csv"))
    plot_comparisons(results, out_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()
