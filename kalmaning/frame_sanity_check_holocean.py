"""Frame sanity checker for HoloOcean + EKF setup.

This script runs a short HoloOcean simulation and compares two candidate
rotations for mapping body-frame DVL velocity to world frame:
    v_world = R_ws @ v_body
    v_world = R_ws.T @ v_body

It also checks IMU gravity alignment and basic axis conventions (x forward,
y left, z up). Results are written to results_frame_check/<timestamp>_seed*/.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import holoocean
import matplotlib.pyplot as plt
import numpy as np


# Keep scenario/agent names aligned with current_acoustic_EKF_patched.py
SCENARIO_NAME = "usv_auv_100_imu"
AUV_NAME = "auv"

# World gravity used by the EKF code (z up, +9.81)
GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def compute_errors(truth: np.ndarray, est: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    truth = np.asarray(truth)
    est = np.asarray(est)
    valid = np.isfinite(truth).all(axis=1) & np.isfinite(est).all(axis=1)
    if not np.any(valid):
        return float("nan"), np.full(3, np.nan), valid
    err = est[valid] - truth[valid]
    mse_axis = np.mean(err * err, axis=0)
    rmse_axis = np.sqrt(mse_axis)
    rmse_total = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    return rmse_total, rmse_axis, valid


def rotation_to_yaw(R: np.ndarray) -> float:
    """Extract yaw (heading about +z) in radians assuming x-forward, y-left, z-up."""
    # yaw = atan2(r21, r11) for RH frames with z-up
    return float(np.arctan2(R[1, 0], R[0, 0]))


def run_frame_check(
    duration_sec: float,
    seed: int,
    show_viewport: bool,
    rest_sec: float,
    output_root: Path,
    forward_speed: float,
    yaw_amp_deg: float,
    yaw_freq_hz: float,
    lateral_amp: float,
) -> Dict[str, float]:
    np.random.seed(seed)
    ensure_dir(output_root)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"{stamp}_seed{seed}"
    ensure_dir(out_dir)

    with holoocean.make(
        SCENARIO_NAME,
        show_viewport=show_viewport,
        frames_per_sec=False,
        verbose=False,
    ) as env:
        ticks_per_sec = getattr(env, "ticks_per_sec", 100.0)
        dt = 1.0 / float(ticks_per_sec)

        total_steps = int(duration_sec * ticks_per_sec)
        rest_steps = int(rest_sec * ticks_per_sec)

        # Grab initial state to anchor relative commands
        state0 = env.tick()
        auv0 = state0.get(AUV_NAME, {})
        start_pos = np.array(auv0.get("LocationSensor", np.zeros(3)), dtype=float)
        start_yaw = 0.0
        if "PoseSensor" in auv0:
            R0 = np.asarray(auv0["PoseSensor"][0:3, 0:3])
            start_yaw = rotation_to_yaw(R0)

        times: List[float] = []
        pos_true: List[np.ndarray] = []
        vel_true: List[np.ndarray] = []
        vel_body: List[np.ndarray] = []
        vel_w1: List[np.ndarray] = []
        vel_w2: List[np.ndarray] = []
        err_norm1: List[float] = []
        err_norm2: List[float] = []
        cos1_list: List[float] = []
        cos2_list: List[float] = []
        accel_body_log: List[np.ndarray] = []
        grav_body_exp_log: List[np.ndarray] = []
        yaw_cmd_log: List[float] = []
        yaw_true_log: List[float] = []

        prev_pos = None

        for k in range(total_steps):
            t = k * dt
            t_move = max(0.0, t - rest_sec)

            if k < rest_steps:
                target = np.array([start_pos[0], start_pos[1], start_pos[2], 0.0, 0.0, np.rad2deg(start_yaw)], dtype=float)
                yaw_cmd = start_yaw
            else:
                dx = forward_speed * t_move
                dy = lateral_amp * np.sin(2 * np.pi * yaw_freq_hz * t_move)
                yaw_cmd = yaw_amp_deg * np.sin(2 * np.pi * yaw_freq_hz * t_move)
                target = np.array([
                    start_pos[0] + dx,
                    start_pos[1] + dy,
                    start_pos[2],
                    0.0,
                    0.0,
                    yaw_cmd,
                ], dtype=float)

            state = env.step(target)

            sensors = state.get(AUV_NAME, {})
            imu = sensors.get("IMUSensor", None)
            pose = sensors.get("PoseSensor", None)
            loc = sensors.get("LocationSensor", None)
            dvl = sensors.get("DVLSensor", None)

            if pose is None or loc is None or dvl is None:
                continue

            loc = np.asarray(loc, dtype=float)
            R_ws = np.asarray(pose[0:3, 0:3], dtype=float)
            v_body = np.asarray(dvl[0:3], dtype=float)

            if prev_pos is None:
                v_true = np.zeros(3)
            else:
                v_true = (loc - prev_pos) / dt
            prev_pos = loc.copy()

            v_w1 = R_ws @ v_body
            v_w2 = R_ws.T @ v_body

            err1 = v_w1 - v_true
            err2 = v_w2 - v_true

            times.append(t)
            pos_true.append(loc.copy())
            vel_true.append(v_true.copy())
            vel_body.append(v_body.copy())
            vel_w1.append(v_w1.copy())
            vel_w2.append(v_w2.copy())
            err_norm1.append(float(np.linalg.norm(err1)))
            err_norm2.append(float(np.linalg.norm(err2)))
            cos1_list.append(cos_sim(v_true, v_w1))
            cos2_list.append(cos_sim(v_true, v_w2))

            yaw_cmd_log.append(np.deg2rad(target[5]))
            yaw_true_log.append(rotation_to_yaw(R_ws))

            if imu is not None and imu.shape[0] >= 3:
                accel_meas = np.asarray(imu[0, :], dtype=float)
                accel_bias = np.asarray(imu[2, :], dtype=float)
                accel_body = accel_meas - accel_bias
                accel_body_log.append(accel_body)
                grav_body_exp = R_ws.T @ GRAVITY_WORLD
                grav_body_exp_log.append(grav_body_exp)

        # Convert to arrays
        times_a = np.asarray(times)
        pos_true_a = np.asarray(pos_true)
        vel_true_a = np.asarray(vel_true)
        vel_body_a = np.asarray(vel_body)
        vel_w1_a = np.asarray(vel_w1)
        vel_w2_a = np.asarray(vel_w2)
        err_norm1_a = np.asarray(err_norm1)
        err_norm2_a = np.asarray(err_norm2)
        cos1_a = np.asarray(cos1_list)
        cos2_a = np.asarray(cos2_list)
        yaw_cmd_a = np.asarray(yaw_cmd_log)
        yaw_true_a = np.asarray(yaw_true_log)
        accel_body_a = np.asarray(accel_body_log)
        grav_body_exp_a = np.asarray(grav_body_exp_log)

        # Metrics
        rmse_w1, rmse_axis_w1, valid_mask = compute_errors(vel_true_a, vel_w1_a)
        rmse_w2, rmse_axis_w2, _ = compute_errors(vel_true_a, vel_w2_a)
        mae_w1 = float(np.nanmean(np.linalg.norm(vel_w1_a[valid_mask] - vel_true_a[valid_mask], axis=1))) if np.any(valid_mask) else float("nan")
        mae_w2 = float(np.nanmean(np.linalg.norm(vel_w2_a[valid_mask] - vel_true_a[valid_mask], axis=1))) if np.any(valid_mask) else float("nan")
        mean_cos1 = float(np.nanmean(cos1_a)) if cos1_a.size else float("nan")
        mean_cos2 = float(np.nanmean(cos2_a)) if cos2_a.size else float("nan")

        winner = "R_ws @ v_body" if rmse_w1 <= rmse_w2 else "R_ws.T @ v_body"

        # IMU gravity alignment (first rest interval if available)
        imu_msg = "insufficient data"
        if accel_body_a.size and grav_body_exp_a.size:
            n_rest = min(rest_steps, accel_body_a.shape[0])
            if n_rest > 0:
                g_meas = accel_body_a[:n_rest]
                g_exp = grav_body_exp_a[:n_rest]
                cos_g = np.array([cos_sim(g_exp[i], g_meas[i]) for i in range(n_rest)])
                mean_cos_g = float(np.nanmean(cos_g))
                sign_flip = np.sign(np.mean(g_meas, axis=0)) * np.sign(np.mean(g_exp, axis=0))
                imu_msg = (
                    f"IMU gravity alignment cos={mean_cos_g:.3f}, sign agreement (x,y,z)="
                    f"({sign_flip[0]:.0f},{sign_flip[1]:.0f},{sign_flip[2]:.0f})"
                )

        # Basic axis checks
        axis_msg = ""
        if pos_true_a.shape[0] >= 2:
            delta = pos_true_a[-1] - pos_true_a[0]
            axis_msg = (
                f"Net displacement: dx={delta[0]:.2f}, dy={delta[1]:.2f}, dz={delta[2]:.2f}"
                f" | heading change (deg)={np.rad2deg(yaw_true_a[-1] - yaw_true_a[0]):.2f}"
            )

        # Save CSV
        csv_path = out_dir / "timeseries.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t",
                "p_true_x",
                "p_true_y",
                "p_true_z",
                "v_true_x",
                "v_true_y",
                "v_true_z",
                "v_body_x",
                "v_body_y",
                "v_body_z",
                "vW1_x",
                "vW1_y",
                "vW1_z",
                "vW2_x",
                "vW2_y",
                "vW2_z",
                "err_norm_W1",
                "err_norm_W2",
                "cos_W1",
                "cos_W2",
            ])
            for i in range(len(times_a)):
                writer.writerow([
                    times_a[i],
                    *pos_true_a[i],
                    *vel_true_a[i],
                    *vel_body_a[i],
                    *vel_w1_a[i],
                    *vel_w2_a[i],
                    err_norm1_a[i],
                    err_norm2_a[i],
                    cos1_a[i],
                    cos2_a[i],
                ])

        # Plots
        plt.figure(figsize=(10, 6))
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(times_a, vel_true_a[:, 0], label="v_true_x", color="k")
        ax1.plot(times_a, vel_w1_a[:, 0], label="vW1_x", linestyle="--")
        ax1.plot(times_a, vel_w2_a[:, 0], label="vW2_x", linestyle=":")
        ax1.set_ylabel("vx [m/s]")
        ax1.legend()
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(times_a, vel_true_a[:, 1], color="k")
        ax2.plot(times_a, vel_w1_a[:, 1], linestyle="--")
        ax2.plot(times_a, vel_w2_a[:, 1], linestyle=":")
        ax2.set_ylabel("vy [m/s]")
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(times_a, vel_true_a[:, 2], color="k")
        ax3.plot(times_a, vel_w1_a[:, 2], linestyle="--")
        ax3.plot(times_a, vel_w2_a[:, 2], linestyle=":")
        ax3.set_ylabel("vz [m/s]")
        ax3.set_xlabel("time [s]")
        plt.tight_layout()
        plt.savefig(out_dir / "vel_compare_components.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(times_a, err_norm1_a, label="||vW1 - v_true||")
        plt.plot(times_a, err_norm2_a, label="||vW2 - v_true||")
        plt.xlabel("time [s]")
        plt.ylabel("velocity error norm [m/s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "vel_error_norms.png", dpi=150)
        plt.close()

        if accel_body_a.size and grav_body_exp_a.size:
            plt.figure(figsize=(8, 4))
            cos_g = [cos_sim(grav_body_exp_a[i], accel_body_a[i]) for i in range(min(len(accel_body_a), len(grav_body_exp_a)))]
            plt.plot(cos_g, label="cos(meas_g, expected_g)")
            plt.axhline(1.0, color="k", linestyle=":", linewidth=1)
            plt.xlabel("sample index")
            plt.ylabel("cosine similarity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "imu_gravity_alignment.png", dpi=150)
            plt.close()

        # Save yaw tracking for left/right sanity
        plt.figure(figsize=(8, 4))
        plt.plot(times_a, np.rad2deg(yaw_cmd_a), label="yaw cmd [deg]")
        plt.plot(times_a, np.rad2deg(yaw_true_a), label="yaw true [deg]")
        plt.xlabel("time [s]")
        plt.ylabel("yaw [deg]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "yaw_tracking.png", dpi=150)
        plt.close()

        summary = {
            "rmse_w1": rmse_w1,
            "rmse_w2": rmse_w2,
            "mae_w1": mae_w1,
            "mae_w2": mae_w2,
            "mean_cos1": mean_cos1,
            "mean_cos2": mean_cos2,
            "winner": winner,
            "imu_msg": imu_msg,
            "axis_msg": axis_msg,
            "output_dir": str(out_dir),
        }

        return summary


def main():
    parser = argparse.ArgumentParser(description="HoloOcean frame sanity check (DVL + IMU)")
    parser.add_argument("--duration", type=float, default=40.0, help="Simulation duration in seconds")
    parser.add_argument("--rest", type=float, default=3.0, help="Rest time at start to check gravity (seconds)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--show-viewport", action="store_true", help="Show HoloOcean viewport")
    parser.add_argument("--output-root", type=str, default="results_frame_check", help="Output root folder")
    parser.add_argument("--forward-speed", type=float, default=0.8, help="Forward speed command in m/s")
    parser.add_argument("--yaw-amp-deg", type=float, default=15.0, help="Yaw oscillation amplitude in degrees")
    parser.add_argument("--yaw-freq-hz", type=float, default=0.05, help="Yaw oscillation frequency in Hz")
    parser.add_argument("--lateral-amp", type=float, default=3.0, help="Lateral sway amplitude in meters")
    args = parser.parse_args()

    summary = run_frame_check(
        duration_sec=args.duration,
        seed=args.seed,
        show_viewport=args.show_viewport,
        rest_sec=args.rest,
        output_root=Path(args.output_root),
        forward_speed=args.forward_speed,
        yaw_amp_deg=args.yaw_amp_deg,
        yaw_freq_hz=args.yaw_freq_hz,
        lateral_amp=args.lateral_amp,
    )

    print("\nFrame sanity check results")
    print(f"Output dir: {summary['output_dir']}")
    print(f"RMSE v_world = R_ws @ v_body : {summary['rmse_w1']:.4f} m/s")
    print(f"RMSE v_world = R_ws.T @ v_body: {summary['rmse_w2']:.4f} m/s")
    print(f"MAE  v_world = R_ws @ v_body : {summary['mae_w1']:.4f} m/s")
    print(f"MAE  v_world = R_ws.T @ v_body: {summary['mae_w2']:.4f} m/s")
    print(f"Mean cos similarity (R_ws): {summary['mean_cos1']:.4f}")
    print(f"Mean cos similarity (R_ws.T): {summary['mean_cos2']:.4f}")
    print(f"Conclusion: Best mapping is {summary['winner']}")
    print(summary.get("imu_msg", ""))
    print(summary.get("axis_msg", ""))

    # Actionable patch suggestion for the EKF DVL transform
    if summary["winner"] == "R_ws.T @ v_body":
        print("Actionable patch suggestion: change DVL transform to v_world_meas = R_ws.T @ v_body in current_acoustic_EKF_patched.py")
    else:
        print("Actionable patch suggestion: existing v_world_meas = R_ws @ v_body appears correct.")


if __name__ == "__main__":
    main()