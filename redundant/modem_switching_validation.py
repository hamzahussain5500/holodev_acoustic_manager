"""Modem switching validation (manual dropout vs policy-based geometry gating).

Examples:
  # Manual dropout windows (disconnect/reconnect specific beacons)
  python modem_switching_validation.py --mode manual --dropout "usv2:30-60;usv4:90-140" \
      --targets usv1 usv2 usv3 usv4 --trajectory spiral --seed 0

  # Policy-driven switching using GDOP thresholds
  python modem_switching_validation.py --mode policy --gdop-xy-thresh 8 --gdop-3d-thresh 12 \
      --targets usv1 usv2 usv3 usv4 --trajectory spiral --seed 1
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_modem_manager import AdaptiveModemManager
from current_acoustic_EKF_patched import run_single_trial
from modem_dropout_test import parse_dropout, is_enabled, pretty_name


def _set_style():
    plt.rcParams.update({
        "figure.dpi": 250,
        "axes.grid": True,
        "axes.facecolor": "#f8f8f8",
        "grid.alpha": 0.35,
        "font.size": 10,
        "lines.linewidth": 1.4,
    })


def make_manual_selector(dropout: Dict[str, List[Tuple[float, float]]]):
    def selector(t_current: float, ekf_state: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]]):
        active = [name for name, _ in target_info if is_enabled(dropout, name, float(t_current))]
        return {
            "active_names": active,
            "mode": "manual",
            "rank_xy": np.nan,
            "gdop_xy": np.nan,
            "rank_3d": np.nan,
            "gdop_3d": np.nan,
            "reason": "manual_dropout",
        }

    return selector


def make_policy_selector(
    sigma_r: float,
    gdop_xy_thresh: float,
    gdop_3d_thresh: float,
    min_dwell: int,
    switch_margin: float,
    dropout: Dict[str, List[Tuple[float, float]]],
    size_penalty: float,
):
    manager = AdaptiveModemManager(
        modem_positions=np.zeros((4, 3)),
        sigma_r=sigma_r,
        gdop_xy_thresh=gdop_xy_thresh,
        gdop_3d_thresh=gdop_3d_thresh,
        min_dwell_steps=min_dwell,
        switch_margin=switch_margin,
        size_penalty=size_penalty,
    )

    def selector(t_current: float, ekf_state: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]]):
        enabled = [(name, pos) for name, pos in target_info if is_enabled(dropout, name, float(t_current))]
        if not enabled:
            return {
                "active_names": [],
                "mode": "policy",
                "rank_xy": np.nan,
                "gdop_xy": np.nan,
                "rank_3d": np.nan,
                "gdop_3d": np.nan,
                "reason": "no_targets",
            }

        manager.update_positions(np.vstack([pos for _, pos in enabled]))
        decision = manager.decide(ekf_state[0:3], depth_available=depth_available)
        active_idx = decision.get("active_indices", [])
        active_names = [enabled[i][0] for i in active_idx if 0 <= i < len(enabled)]
        decision["active_names"] = active_names
        return decision

    return selector


def write_timeseries_csv(path: Path, ts: Dict[str, Any]):
    t = np.asarray(ts.get("t", []), dtype=float)
    err = np.asarray(ts.get("err_norm", []), dtype=float)
    Ppos = np.asarray(ts.get("Ppos", []), dtype=float)
    traceP = np.einsum("nii->n", Ppos) if Ppos.size else np.array([])
    active_count = np.asarray(ts.get("active_count", []), dtype=float)
    active_set = ts.get("active_set", [])
    rank_xy = np.asarray(ts.get("rank_xy", []), dtype=float)
    gdop_xy = np.asarray(ts.get("gdop_xy", []), dtype=float)
    rank_3d = np.asarray(ts.get("rank_3d", []), dtype=float)
    gdop_3d = np.asarray(ts.get("gdop_3d", []), dtype=float)
    mode = ts.get("mode", [])

    fields = [
        "t", "err_norm", "trace_Ppos", "active_count", "active_set", "rank_xy", "gdop_xy", "rank_3d", "gdop_3d", "mode",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(t)):
            row = {
                "t": float(t[i]),
                "err_norm": float(err[i]) if err.size else np.nan,
                "trace_Ppos": float(traceP[i]) if traceP.size else np.nan,
                "active_count": float(active_count[i]) if active_count.size else np.nan,
                "active_set": str(active_set[i]) if len(active_set) > i else "",
                "rank_xy": float(rank_xy[i]) if rank_xy.size else np.nan,
                "gdop_xy": float(gdop_xy[i]) if gdop_xy.size else np.nan,
                "rank_3d": float(rank_3d[i]) if rank_3d.size else np.nan,
                "gdop_3d": float(gdop_3d[i]) if gdop_3d.size else np.nan,
                "mode": str(mode[i]) if len(mode) > i else "",
            }
            writer.writerow(row)


def _shade_dropout(ax, dropout: Dict[str, List[Tuple[float, float]]]):
    for name, spans in dropout.items():
        for a, b in spans:
            ax.axvspan(a, b, color="#bbbbbb", alpha=0.25, label=f"{pretty_name(name)} dropout")


def _mark_switches(ax, t: np.ndarray, active_set: List[str]):
    if len(t) == 0 or len(active_set) == 0:
        return
    prev = None
    for tt, aset in zip(t, active_set):
        if prev is None:
            prev = aset
            continue
        if aset != prev:
            ax.axvline(tt, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
            prev = aset


def plot_series(out_dir: Path, ts: Dict[str, Any], dropout: Dict[str, List[Tuple[float, float]]], mode: str):
    t = np.asarray(ts.get("t", []), dtype=float)
    err = np.asarray(ts.get("err_norm", []), dtype=float)
    traceP = np.einsum("nii->n", np.asarray(ts.get("Ppos", []), dtype=float)) if ts.get("Ppos") is not None else None
    active_count = np.asarray(ts.get("active_count", []), dtype=float)
    active_set = list(ts.get("active_set", []))
    gdop_xy = np.asarray(ts.get("gdop_xy", []), dtype=float)
    gdop_3d = np.asarray(ts.get("gdop_3d", []), dtype=float)

    # Error norm
    plt.figure(figsize=(7, 3.2))
    plt.plot(t, err, label="||pos error||")
    if mode == "manual":
        _shade_dropout(plt.gca(), dropout)
    else:
        _mark_switches(plt.gca(), t, active_set)
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title("Position error vs time")
    plt.tight_layout()
    plt.savefig(out_dir / "err_norm_vs_time.png")
    plt.close()

    # trace(P)
    if traceP is not None:
        plt.figure(figsize=(7, 3.2))
        plt.plot(t, traceP, label="trace(Ppos)")
        if mode == "manual":
            _shade_dropout(plt.gca(), dropout)
        else:
            _mark_switches(plt.gca(), t, active_set)
        plt.xlabel("time [s]")
        plt.ylabel("trace(Ppos) [m^2]")
        plt.title("Position covariance trace")
        plt.tight_layout()
        plt.savefig(out_dir / "traceP_vs_time.png")
        plt.close()

    # active count
    plt.figure(figsize=(7, 3.0))
    plt.plot(t, active_count, label="# active")
    if mode == "manual":
        _shade_dropout(plt.gca(), dropout)
    else:
        _mark_switches(plt.gca(), t, active_set)
    plt.xlabel("time [s]")
    plt.ylabel("active beacons")
    plt.title("Active acoustic beacons vs time")
    plt.tight_layout()
    plt.savefig(out_dir / "active_count_vs_time.png")
    plt.close()

    # GDOP xy
    plt.figure(figsize=(7, 3.0))
    plt.plot(t, gdop_xy, label="GDOP_xy")
    if mode == "manual":
        _shade_dropout(plt.gca(), dropout)
    else:
        _mark_switches(plt.gca(), t, active_set)
    plt.xlabel("time [s]")
    plt.ylabel("GDOP XY")
    plt.title("GDOP_xy vs time")
    plt.tight_layout()
    plt.savefig(out_dir / "gdop_xy_vs_time.png")
    plt.close()

    # Optional GDOP 3D
    if gdop_3d.size:
        plt.figure(figsize=(7, 3.0))
        plt.plot(t, gdop_3d, label="GDOP_3D")
        if mode == "manual":
            _shade_dropout(plt.gca(), dropout)
        else:
            _mark_switches(plt.gca(), t, active_set)
        plt.xlabel("time [s]")
        plt.ylabel("GDOP 3D")
        plt.title("GDOP_3D vs time")
        plt.tight_layout()
        plt.savefig(out_dir / "gdop_3d_vs_time.png")
        plt.close()


def save_summary(path: Path, trial: Dict[str, Any], mode: str, dropout: Dict[str, List[Tuple[float, float]]], policy_cfg: Dict[str, Any]):
    config = trial.get("config")
    jsonable_config = None
    if isinstance(config, dict):
        jsonable_config = dict(config)
        selector = jsonable_config.get("modem_selector_fn")
        if callable(selector):
            jsonable_config["modem_selector_fn"] = getattr(selector, "__name__", "callable")

    summary = {
        "seed": trial.get("seed"),
        "pos_rmse_total": trial.get("pos_rmse_total"),
        "vel_rmse_total": trial.get("vel_rmse_total"),
        "final_position_error": trial.get("final_position_error"),
        "runtime_seconds": trial.get("runtime_seconds"),
        "mode": mode,
        "dropout_intervals": dropout,
        "policy": policy_cfg,
        "config": jsonable_config,
    }
    with (path / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Validate modem switching (manual dropout vs policy GDOP)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=180.0)
    parser.add_argument("--trajectory", choices=["lawnmower", "spiral", "concentric", "figure8"], default="spiral")
    parser.add_argument("--targets", nargs="*", default=["usv1", "usv2", "usv3", "usv4"],
                        choices=["usv1", "usv2", "usv3", "usv4"], help="Acoustic beacons to include")
    parser.add_argument("--mode", choices=["manual", "policy"], default="manual")
    parser.add_argument("--dropout", type=str, default="", help="Dropout schedule e.g. 'usv2:30-60;usv4:90-140' (manual mode)")
    parser.add_argument("--depth-mode", choices=["on", "off"], default="on", help="If off, enforce 3D constraints")
    parser.add_argument("--gdop-xy-thresh", type=float, default=8.0)
    parser.add_argument("--gdop-3d-thresh", type=float, default=12.0)
    parser.add_argument("--min-dwell", type=int, default=50)
    parser.add_argument("--switch-margin", type=float, default=0.25)
    parser.add_argument("--sigma-r", type=float, default=0.1, help="Range noise std used for GDOP scoring")
    parser.add_argument("--size-penalty", type=float, default=0.0, help="Penalty weight per beacon to prefer smaller active sets")
    parser.add_argument("--acoustic-uncertainty-trace-thresh", type=float, default=None, help="If trace(Ppos) <= this, gate acoustic updates to save energy")
    args = parser.parse_args()

    _set_style()
    dropout = parse_dropout(args.dropout)
    out_dir = Path("results_modem_switching") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "manual":
        selector_fn = make_manual_selector(dropout)
        policy_cfg = None
    else:
        selector_fn = make_policy_selector(
            sigma_r=args.sigma_r,
            gdop_xy_thresh=args.gdop_xy_thresh,
            gdop_3d_thresh=args.gdop_3d_thresh,
            min_dwell=args.min_dwell,
            switch_margin=args.switch_margin,
            dropout=dropout,
            size_penalty=args.size_penalty,
        )
        policy_cfg = {
            "sigma_r": args.sigma_r,
            "gdop_xy_thresh": args.gdop_xy_thresh,
            "gdop_3d_thresh": args.gdop_3d_thresh,
            "min_dwell": args.min_dwell,
            "switch_margin": args.switch_margin,
            "dropout": dropout,
            "size_penalty": args.size_penalty,
        }

    depth_enabled = args.depth_mode == "on"
    overrides = {
        "enable_dvl": True,
        "enable_depth": depth_enabled,
        "use_depth_update": depth_enabled,
        "enable_acoustic": True,
        "use_all_acoustic": args.mode != "policy",
        "acoustic_uncertainty_trace_thresh": args.acoustic_uncertainty_trace_thresh,
        "use_currents": False,
        "duration_sec": float(args.duration_sec),
        "trajectory": args.trajectory,
        "modem_selector_fn": selector_fn,
    }

    if dropout:
        overrides["modem_dropout_intervals"] = dropout
        overrides["modem_dropout_mode"] = "ignore_updates"

    trial = run_single_trial(
        seed=args.seed,
        config_overrides=overrides,
        return_timeseries=True,
        target_names=args.targets,
    )

    ts = trial.get("timeseries", {})
    write_timeseries_csv(out_dir / "timeseries.csv", ts)
    plot_series(plots_dir, ts, dropout, args.mode)
    save_summary(out_dir, trial, args.mode, dropout, policy_cfg)

    print(f"pos_rmse_total: {trial.get('pos_rmse_total'):.3f} m")
    print(f"final_position_error: {trial.get('final_position_error'):.3f} m")
    print(f"mode: {args.mode}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
