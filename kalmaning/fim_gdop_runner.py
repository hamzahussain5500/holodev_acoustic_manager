"""
FIM/GDOP/CRLB geometry analysis for acoustic beacon configurations.

Example:
    python fim_gdop_runner.py --out results_fim_gdop --sigma-r 0.1 --decimate 10 --mode logdet
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np

from sbl_geometry import SBLConfigurationAnalyzer
from trajectory import build_trajectory


# ---------- Data helpers ----------

def default_static_beacons(n: int = 4, radius: float = 50.0, z: float = 0.0) -> np.ndarray:
    # Quick hack: fixed 4-beacon layout supplied by user (static, in meters).
    beacons = np.array([
        [0.0,  -660.0,   0.0],
        [15.0, -660.0,   0.0],
        [10.0, -650.0,   0.0],
        [0.0,  -650.0, -10.0],
    ])
    if n != 4:
        # If caller asks for fewer/more than 4, fall back to subset or repeat first as needed.
        if n < 4:
            return beacons[:n]
        reps = int(math.ceil(n / 4))

        return np.vstack([beacons for _ in range(reps)])[:n]
    return beacons


def default_trajectory(num_points: int = 600) -> np.ndarray:
    cfg = {
        "trajectory": "spiral",
        "spiral_center": (200.0, -200.0),
        "spiral_min_radius": 20.0,
        "spiral_max_radius": 50.0,
        "spiral_turns": 6.0,
        "spiral_points_per_rev": 250, #max(50, num_points // 4),
        "spiral_z_start": -5.0,
        "spiral_z_end": -150.0,
    }
    traj = build_trajectory("spiral", cfg)
    if traj.shape[0] > num_points:
        traj = traj[:num_points]
    return traj


def load_npz_or_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectory and beacons from npz/csv.

    Expected NPZ keys: trajectory_xyz (T,3), beacon_positions (N,3) or (T,N,3)
    CSV fallback: first three columns = x,y,z trajectory; beacons become default static.
    """
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        traj = data["trajectory_xyz"]
        beacons = data["beacon_positions"]
        return traj, beacons
    # CSV
    traj = np.loadtxt(path, delimiter=",", ndmin=2)
    if traj.shape[1] < 3:
        raise ValueError("CSV must have at least x,y,z columns")
    traj_xyz = traj[:, :3]
    beacons = default_static_beacons()
    return traj_xyz, beacons


def load_beacons_csv(path: Path) -> np.ndarray:
    """Load beacon positions from CSV with columns x,y,z."""
    try:
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        # Likely has a header; retry skipping the first line
        arr = np.loadtxt(path, delimiter=",", ndmin=2, skiprows=1)
    if arr.size == 0 or arr.shape[1] < 3:
        raise ValueError(f"Beacon CSV must have at least 3 numeric columns (x,y,z): {path}")
    return arr[:, :3]


def prepare_inputs(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        traj, beacons = load_npz_or_csv(data_path)
    else:
        # Fallback: synthetic trajectory + static beacons
        traj = default_trajectory()
        beacons = default_static_beacons()

    if args.beacons_csv:
        b_path = Path(args.beacons_csv)
        if not b_path.exists():
            raise FileNotFoundError(f"Beacon CSV not found: {b_path}")
        beacons = load_beacons_csv(b_path)

    return traj, beacons


# ---------- Output helpers ----------

def format_cfg(cfg: Optional[Tuple[int, ...]]) -> str:
    if cfg is None:
        return ""
    return "-".join(str(i) for i in cfg)


def save_npz(result: Dict[str, any], out_path: Path, save_all_configs: bool) -> None:
    payload = {
        "t_idx": result["t_idx"],
        "all_rank": result["all_beacons"]["rank"],
        "all_logdet": result["all_beacons"]["logdet"],
        "all_gdop": result["all_beacons"]["gdop"],
        "all_crlb_diag": result["all_beacons"]["crlb_diag"],
        "all_rank_xy": result["all_beacons_xy"]["rank"],
        "all_logdet_xy": result["all_beacons_xy"]["logdet"],
        "all_gdop_xy": result["all_beacons_xy"]["gdop"],
        "all_crlb_diag_xy": result["all_beacons_xy"]["crlb_diag"],
        "best2_idx": result["best_2"]["idx"],
        "best2_gdop": result["best_2"]["gdop"],
        "best2_logdet": result["best_2"]["logdet"],
        "best2_rank": result["best_2"]["rank"],
        "best2_idx_xy": result["best_2_xy"]["idx"],
        "best2_gdop_xy": result["best_2_xy"]["gdop"],
        "best2_logdet_xy": result["best_2_xy"]["logdet"],
        "best2_rank_xy": result["best_2_xy"]["rank"],
        "best3_idx": result["best_3"]["idx"],
        "best3_gdop": result["best_3"]["gdop"],
        "best3_logdet": result["best_3"]["logdet"],
        "best3_rank": result["best_3"]["rank"],
        "best3_idx_xy": result["best_3_xy"]["idx"],
        "best3_gdop_xy": result["best_3_xy"]["gdop"],
        "best3_logdet_xy": result["best_3_xy"]["logdet"],
        "best3_rank_xy": result["best_3_xy"]["rank"],
        "mode": result["mode"],
        "sigma_r": result["sigma_r"],
        "decimate": result["decimate"],
    }

    if save_all_configs:
        cfgs = []
        ranks = []
        gdops = []
        logdets = []
        invs = []
        crlbs = []
        ranks_xy = []
        gdops_xy = []
        logdets_xy = []
        invs_xy = []
        crlbs_xy = []
        for rec in result["config_records"].values():
            cfgs.append(rec.cfg)
            ranks.append(rec.ranks)
            gdops.append(rec.gdops)
            logdets.append(rec.logdets)
            invs.append(rec.invertible)
            crlbs.append(rec.crlb_diags)
            ranks_xy.append(rec.ranks_xy)
            gdops_xy.append(rec.gdops_xy)
            logdets_xy.append(rec.logdets_xy)
            invs_xy.append(rec.invertible_xy)
            crlbs_xy.append(rec.crlb_diags_xy)
        payload.update({
            "cfg_list": np.array(cfgs, dtype=object),
            "cfg_ranks": np.array(ranks, dtype=object),
            "cfg_gdops": np.array(gdops, dtype=object),
            "cfg_logdets": np.array(logdets, dtype=object),
            "cfg_invertible": np.array(invs, dtype=object),
            "cfg_crlb_diag": np.array(crlbs, dtype=object),
            "cfg_ranks_xy": np.array(ranks_xy, dtype=object),
            "cfg_gdops_xy": np.array(gdops_xy, dtype=object),
            "cfg_logdets_xy": np.array(logdets_xy, dtype=object),
            "cfg_invertible_xy": np.array(invs_xy, dtype=object),
            "cfg_crlb_diag_xy": np.array(crlbs_xy, dtype=object),
        })

    np.savez_compressed(out_path, **payload)


def write_summary_csv(result: Dict[str, any], out_path: Path) -> None:
    rows = []
    for cfg, rec in result["config_records"].items():
        gd = np.asarray(rec.gdops)
        rk = np.asarray(rec.ranks)
        inv = np.asarray(rec.invertible)
        rows.append({
            "config": format_cfg(cfg),
            "min_gdop": np.nanmin(gd),
            "median_gdop": np.nanmedian(gd),
            "max_gdop": np.nanmax(gd),
            "mean_rank": float(np.nanmean(rk)),
            "pct_invertible": float(np.nanmean(inv) * 100.0),
        })
    fieldnames = ["config", "min_gdop", "median_gdop", "max_gdop", "mean_rank", "pct_invertible"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_subset_csv(result: Dict[str, any], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k",
            "best2_cfg",
            "best2_gdop",
            "best2_logdet",
            "best2_rank",
            "best3_cfg",
            "best3_gdop",
            "best3_logdet",
            "best3_rank",
        ])
        for i, k in enumerate(result["t_idx"]):
            writer.writerow([
                int(k),
                format_cfg(result["best_2"]["idx"][i]),
                result["best_2"]["gdop"][i],
                result["best_2"]["logdet"][i],
                int(result["best_2"]["rank"][i]),
                format_cfg(result["best_3"]["idx"][i]),
                result["best_3"]["gdop"][i],
                result["best_3"]["logdet"][i],
                int(result["best_3"]["rank"][i]),
            ])


def write_summary_csv_2d(result: Dict[str, any], out_path: Path) -> None:
    rows = []

    def stats_from_series(series):
        arr = np.asarray(series, dtype=float)
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan
        return {
            "median": float(np.nanmedian(arr)) if arr.size else np.nan,
            "p95": float(np.nanpercentile(arr, 95)) if arr.size else np.nan,
            "min": float(np.nanmin(arr)) if arr.size else np.nan,
            "max": float(np.nanmax(arr)) if arr.size else np.nan,
        }

    # Per configuration
    for cfg, rec in result["config_records"].items():
        gd = stats_from_series(rec.gdops_xy)
        ld = stats_from_series(rec.logdets_xy)
        rk = np.asarray(rec.ranks_xy, dtype=float)
        inv_xy = np.asarray(rec.invertible_xy, dtype=bool)
        obs_frac = float(np.mean(rk >= 2)) if rk.size else np.nan
        crlb = np.asarray(rec.crlb_diags_xy, dtype=float)
        mean_sig_x = float(np.nanmean(crlb[:, 0])) if crlb.ndim == 2 else np.nan
        mean_sig_y = float(np.nanmean(crlb[:, 1])) if crlb.ndim == 2 else np.nan
        rows.append({
            "config": format_cfg(cfg),
            "observable_frac": obs_frac,
            "median_gdop": gd["median"],
            "p95_gdop": gd["p95"],
            "median_logdet": ld["median"],
            "mean_sigma_x": mean_sig_x,
            "mean_sigma_y": mean_sig_y,
            "pct_invertible": float(np.nanmean(inv_xy) * 100.0) if inv_xy.size else np.nan,
        })

    # All-beacons aggregate
    all_xy = result["all_beacons_xy"]
    gd = stats_from_series(all_xy["gdop"])
    ld = stats_from_series(all_xy["logdet"])
    rk = np.asarray(all_xy["rank"], dtype=float)
    crlb = np.asarray(all_xy["crlb_diag"], dtype=float)
    rows.append({
        "config": "all_beacons",
        "observable_frac": float(np.mean(rk >= 2)) if rk.size else np.nan,
        "median_gdop": gd["median"],
        "p95_gdop": gd["p95"],
        "median_logdet": ld["median"],
        "mean_sigma_x": float(np.nanmean(crlb[:, 0])) if crlb.ndim == 2 else np.nan,
        "mean_sigma_y": float(np.nanmean(crlb[:, 1])) if crlb.ndim == 2 else np.nan,
        "pct_invertible": float(np.mean(np.isfinite(all_xy["gdop"])) * 100.0) if rk.size else np.nan,
    })

    fieldnames = [
        "config",
        "observable_frac",
        "median_gdop",
        "p95_gdop",
        "median_logdet",
        "mean_sigma_x",
        "mean_sigma_y",
        "pct_invertible",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_subset_csv_2d(result: Dict[str, any], traj: np.ndarray, out_path: Path) -> None:
    t = result["t_idx"]
    cfg_map = result["config_records"]
    all_xy = result["all_beacons_xy"]

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k",
            "x",
            "y",
            "z",
            "logdet_all_xy",
            "gdop_all_xy",
            "rank_all_xy",
            "sigma_x_all",
            "sigma_y_all",
            "best3_cfg",
            "logdet_best3_xy",
            "gdop_best3_xy",
            "rank_best3_xy",
            "sigma_x_best3",
            "sigma_y_best3",
            "best2_cfg",
            "logdet_best2_xy",
            "gdop_best2_xy",
            "rank_best2_xy",
            "sigma_x_best2",
            "sigma_y_best2",
        ])

        for idx, k in enumerate(t):
            px = float(traj[k][0]) if traj is not None and len(traj) > k else math.nan
            py = float(traj[k][1]) if traj is not None and len(traj) > k else math.nan
            pz = float(traj[k][2]) if traj is not None and len(traj) > k else math.nan

            row = [
                int(k),
                px,
                py,
                pz,
                all_xy["logdet"][idx],
                all_xy["gdop"][idx],
                int(all_xy["rank"][idx]),
                all_xy["crlb_diag"][idx, 0],
                all_xy["crlb_diag"][idx, 1],
            ]

            def append_cfg_metrics(cfg_idx):
                if cfg_idx is None:
                    return ["", math.nan, math.nan, 0, math.nan, math.nan]
                if isinstance(cfg_idx, np.ndarray):
                    cfg_idx = tuple(cfg_idx.tolist())
                rec = cfg_map[cfg_idx]
                crlb_xy_row = np.asarray(rec.crlb_diags_xy[idx])
                return [
                    format_cfg(cfg_idx),
                    rec.logdets_xy[idx],
                    rec.gdops_xy[idx],
                    int(rec.ranks_xy[idx]),
                    float(crlb_xy_row[0]) if crlb_xy_row.size >= 1 else math.nan,
                    float(crlb_xy_row[1]) if crlb_xy_row.size >= 2 else math.nan,
                ]

            row.extend(append_cfg_metrics(result["best_3_xy"]["idx"][idx]))
            row.extend(append_cfg_metrics(result["best_2_xy"]["idx"][idx]))

            writer.writerow(row)


# ---------- Plot helpers ----------

def _mask_inf(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def plot_timeseries(result: Dict[str, any], out_dir: Path) -> None:
    t = result["t_idx"]

    # Log-det FIM
    plt.figure(figsize=(7, 4))
    plt.plot(t, result["all_beacons"]["logdet"], label="All beacons", linewidth=2)
    plt.plot(t, result["best_3"]["logdet"], label="Best 3", linestyle="--")
    plt.plot(t, result["best_2"]["logdet"], label="Best 2", linestyle=":")
    plt.xlabel("index")
    plt.ylabel("log det F")
    plt.title("FIM log-det vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_fim_logdet_vs_time.png", dpi=300)
    plt.close()

    # GDOP
    plt.figure(figsize=(7, 4))
    plt.plot(t, _mask_inf(result["all_beacons"]["gdop"]), label="All beacons", linewidth=2)
    plt.plot(t, _mask_inf(result["best_3"]["gdop"]), label="Best 3", linestyle="--")
    plt.plot(t, _mask_inf(result["best_2"]["gdop"]), label="Best 2", linestyle=":")
    plt.xlabel("index")
    plt.ylabel("GDOP")
    plt.title("GDOP vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_gdop_vs_time.png", dpi=300)
    plt.close()

    # Rank
    plt.figure(figsize=(7, 4))
    plt.plot(t, result["all_beacons"]["rank"], label="All beacons", linewidth=2)
    plt.plot(t, result["best_3"]["rank"], label="Best 3", linestyle="--")
    plt.plot(t, result["best_2"]["rank"], label="Best 2", linestyle=":")
    plt.xlabel("index")
    plt.ylabel("rank(H)")
    plt.title("Rank vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_rank_vs_time.png", dpi=300)
    plt.close()

    # CRLB position stds (all-beacon only)
    crlb_diag = result["all_beacons"]["crlb_diag"]
    stds = np.sqrt(_mask_inf(crlb_diag))
    plt.figure(figsize=(7, 4))
    plt.plot(t, stds[:, 0], label="sigma_x")
    plt.plot(t, stds[:, 1], label="sigma_y")
    plt.plot(t, stds[:, 2], label="sigma_z")
    plt.xlabel("index")
    plt.ylabel("CRLB std [m]")
    plt.title("CRLB position std vs time (all beacons)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_crlb_pos_std_vs_time.png", dpi=300)
    plt.close()


def plot_timeseries_xy(result: Dict[str, any], out_dir: Path) -> None:
    t = result["t_idx"]
    xy = result["all_beacons_xy"]

    plt.figure(figsize=(7, 4))
    plt.plot(t, xy["logdet"], label="All beacons", linewidth=2)
    plt.plot(t, result["best_3_xy"]["logdet"], label="Best 3", linestyle="--")
    plt.plot(t, result["best_2_xy"]["logdet"], label="Best 2", linestyle=":")
    plt.xlabel("index")
    plt.ylabel("log det F_xy")
    plt.title("FIM log-det (xy) vs time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_fim_logdet_xy_vs_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, _mask_inf(xy["gdop"]), label="All beacons", linewidth=2)
    plt.plot(t, _mask_inf(result["best_3_xy"]["gdop"]), label="Best 3", linestyle="--")
    plt.plot(t, _mask_inf(result["best_2_xy"]["gdop"]), label="Best 2", linestyle=":")
    plt.xlabel("index")
    plt.ylabel("GDOP (xy)")
    plt.title("GDOP (xy) vs time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_gdop_xy_vs_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, xy["rank"], label="All beacons", linewidth=2)
    plt.plot(t, result["best_3_xy"]["rank"], label="Best 3", linestyle="--")
    plt.plot(t, result["best_2_xy"]["rank"], label="Best 2", linestyle=":")
    plt.xlabel("index")
    plt.ylabel("rank(H_xy)")
    plt.title("Rank (xy) vs time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_rank_xy_vs_time.png", dpi=300)
    plt.close()

    crlb_diag = xy["crlb_diag"]
    stds = np.sqrt(_mask_inf(crlb_diag))
    plt.figure(figsize=(7, 4))
    plt.plot(t, stds[:, 0], label="sigma_x (xy)")
    plt.plot(t, stds[:, 1], label="sigma_y (xy)")
    plt.xlabel("index")
    plt.ylabel("CRLB std [m]")
    plt.title("CRLB position std (xy) vs time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_crlb_xy_std_vs_time.png", dpi=300)
    plt.close()


def write_readme(out_dir: Path) -> None:
    text = """Geometry outputs (3D and 2D horizontal-only)

- 3D metrics: rank(H), log det(F), GDOP, CRLB diag for full xyz.
- 2D metrics: rank(H_xy), log det(F_xy), GDOP_xy, CRLB_xy std (sigma_x, sigma_y).
- Best subsets: chosen independently for 3D and 2D (best 2, best 3 by log-det or GDOP policy).
- Why 2D: when depth is known from a Depth sensor, acoustics mainly constrain x–y, so 2 beacons can still provide good horizontal geometry.
- Observability: 2D observable if rank(H_xy) >= 2; otherwise GDOP_xy=inf and CRLB_xy is NaN.

Files:
- summary.csv (3D), summary_2d.csv (2D)
- best_subset_timeseries.csv (3D), best_subset_timeseries_2d.csv (2D)
- metrics_timeseries.npz (both 3D/2D arrays)
- fig_*png: logdet, GDOP, rank, CRLB for 3D and 2D.
"""
    (out_dir / "README_geometry.md").write_text(text)


def _series_stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=float)
    a = a.copy()
    a[~np.isfinite(a)] = np.nan
    if a.size == 0:
        return {"median": math.nan, "p95": math.nan, "min": math.nan, "max": math.nan}
    return {
        "median": float(np.nanmedian(a)),
        "p95": float(np.nanpercentile(a, 95)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
    }


def _fmt_num(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.3f}"


def print_terminal_summary(result: Dict[str, any]) -> None:
    all3 = result["all_beacons"]
    all2 = result["all_beacons_xy"]

    gd3 = _series_stats(all3["gdop"])
    ld3 = _series_stats(all3["logdet"])
    gd2 = _series_stats(all2["gdop"])
    ld2 = _series_stats(all2["logdet"])

    rank3_full = float(np.mean(np.asarray(all3["rank"]) >= 3)) * 100.0 if all3["rank"] is not None else math.nan
    rank2_full = float(np.mean(np.asarray(all2["rank"]) >= 2)) * 100.0 if all2["rank"] is not None else math.nan

    crlb3 = np.sqrt(np.asarray(all3["crlb_diag"], dtype=float))
    crlb2 = np.sqrt(np.asarray(all2["crlb_diag"], dtype=float))
    mean_sig3 = np.nanmean(crlb3, axis=0) if crlb3.ndim == 2 else np.full(3, math.nan)
    mean_sig2 = np.nanmean(crlb2, axis=0) if crlb2.ndim == 2 else np.full(2, math.nan)

    def best_summary(key: str, req_rank: int) -> Tuple[Dict[str, float], Dict[str, float], float]:
        gd = _series_stats(result[key]["gdop"])
        ld = _series_stats(result[key]["logdet"])
        rank_ok = float(np.mean(np.asarray(result[key]["rank"]) >= req_rank)) * 100.0 if result[key]["rank"] is not None else math.nan
        return gd, ld, rank_ok

    b3_gd, b3_ld, b3_rank = best_summary("best_3", 3)
    b2_gd, b2_ld, b2_rank = best_summary("best_2", 2)
    b3_gd_xy, b3_ld_xy, b3_rank_xy = best_summary("best_3_xy", 2)
    b2_gd_xy, b2_ld_xy, b2_rank_xy = best_summary("best_2_xy", 2)

    print("=== Geometry highlights (for paper) ===")
    print(f"All-beacons 3D: GDOP med/p95 {_fmt_num(gd3['median'])}/{_fmt_num(gd3['p95'])}, logdet med {_fmt_num(ld3['median'])}, full-rank {rank3_full:.1f}%")
    print(f"CRLB mean stds 3D [m]: {_fmt_num(mean_sig3[0])}, {_fmt_num(mean_sig3[1])}, {_fmt_num(mean_sig3[2])}")
    print(f"All-beacons 2D: GDOP med/p95 {_fmt_num(gd2['median'])}/{_fmt_num(gd2['p95'])}, logdet med {_fmt_num(ld2['median'])}, observable {rank2_full:.1f}%")
    print(f"CRLB mean stds 2D [m]: {_fmt_num(mean_sig2[0])}, {_fmt_num(mean_sig2[1])}")
    print(f"Best-3 3D subsets: GDOP med {_fmt_num(b3_gd['median'])}, logdet med {_fmt_num(b3_ld['median'])}, rank>=3 {b3_rank:.1f}%")
    print(f"Best-2 3D subsets: GDOP med {_fmt_num(b2_gd['median'])}, logdet med {_fmt_num(b2_ld['median'])}, rank>=2 {b2_rank:.1f}%")
    print(f"Best-3 2D subsets: GDOP med {_fmt_num(b3_gd_xy['median'])}, logdet med {_fmt_num(b3_ld_xy['median'])}, rank>=2 {b3_rank_xy:.1f}%")
    print(f"Best-2 2D subsets: GDOP med {_fmt_num(b2_gd_xy['median'])}, logdet med {_fmt_num(b2_ld_xy['median'])}, rank>=2 {b2_rank_xy:.1f}%")


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FIM/GDOP/CRLB geometry analyzer")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--sigma-r", type=float, default=0.1, help="Range noise std [m]")
    p.add_argument("--decimate", type=int, default=1, help="Decimate trajectory samples")
    p.add_argument("--trajectory-source", choices=["truth", "ekf"], default="truth", help="Label only (placeholder)")
    p.add_argument("--mode", choices=["logdet", "gdop"], default="logdet", help="Subset selection policy")
    p.add_argument("--configs-max-r", type=int, default=None, help="Max subset size (default N)")
    p.add_argument("--save-all-configs", action="store_true", help="Store per-config timeseries in NPZ")
    p.add_argument("--data", help="Optional npz/csv providing trajectory_xyz and beacon_positions")
    p.add_argument("--beacons-csv", help="Optional CSV with beacon positions (x,y,z per row)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj, beacons = prepare_inputs(args)

    analyzer = SBLConfigurationAnalyzer()
    result = analyzer.analyze_trajectory(
        trajectory_xyz=traj,
        beacon_positions=beacons,
        sigma_r=args.sigma_r,
        mode=args.mode,
        decimate=args.decimate,
        max_r=args.configs_max_r,
        all_configs_metrics=args.save_all_configs,
    )

    print_terminal_summary(result)

    save_npz(result, out_dir / "metrics_timeseries.npz", save_all_configs=args.save_all_configs)
    write_summary_csv(result, out_dir / "summary.csv")
    write_summary_csv_2d(result, out_dir / "summary_2d.csv")
    write_best_subset_csv(result, out_dir / "best_subset_timeseries.csv")
    write_best_subset_csv_2d(result, traj, out_dir / "best_subset_timeseries_2d.csv")
    plot_timeseries(result, out_dir)
    plot_timeseries_xy(result, out_dir)
    write_readme(out_dir)
    files_list = sorted([p.name for p in out_dir.iterdir()])
    print(f"Saved metrics and plots to {out_dir}")
    print("Files:")
    for name in files_list:
        print(f"  {name}")


if __name__ == "__main__":
    main()
