"""Reusable trajectory generators for EKF simulations.

Includes:
- lawnmower (coverage sweeps)
- spiral (observability-focused, expanding radius)
- concentric circles (fixed-radius loops)
- figure-eight (lemniscate)
"""
import numpy as np
from typing import Iterable, Tuple


SPAWN_POINT = np.array([-50.0, -50.0, -5.0])
SCALE_FACTOR = 3.0


def _add_transit_to_start(path: np.ndarray, start_xyz: np.ndarray = SPAWN_POINT, max_step: float = 25.0) -> np.ndarray:
    """Insert waypoints from spawn to the first trajectory point via a straight-line transit."""
    if path is None:
        return np.asarray(start_xyz, dtype=float).reshape(1, 3)
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr
    if arr.size == 0:
        return np.asarray(start_xyz, dtype=float).reshape(1, 3)

    start = np.asarray(start_xyz, dtype=float).reshape(3,)
    first = arr[0, :3]
    if np.allclose(start, first):
        return arr

    dist = float(np.linalg.norm(first - start))
    if dist <= 1e-6:
        return arr

    steps = max(1, int(np.ceil(dist / max_step)))
    transit = np.linspace(start, first, num=steps + 1, endpoint=False)
    return np.vstack((transit, arr))


def _densify_path(path: np.ndarray, max_step: float = 5.0) -> np.ndarray:
    """Insert intermediate points so each segment length is <= max_step."""
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] < 2:
        return arr

    max_step = float(max_step)
    if max_step <= 0:
        return arr

    pieces = []
    for i in range(arr.shape[0] - 1):
        p0 = arr[i]
        p1 = arr[i + 1]
        dist = float(np.linalg.norm(p1[:3] - p0[:3]))
        steps = max(1, int(np.ceil(dist / max_step)))
        segment = np.linspace(p0, p1, num=steps, endpoint=False)
        pieces.append(segment)

    pieces.append(arr[-1].reshape(1, -1))
    return np.vstack(pieces)


def lawnmower_waypoints(
    start_xy: Tuple[float, float] = (SPAWN_POINT[0], SPAWN_POINT[1]),
    xmin: float = SPAWN_POINT[0] - 100.0,
    xmax: float = SPAWN_POINT[0] + 100.0,
    ymin: float = SPAWN_POINT[1] - 100.0,
    ymax: float = SPAWN_POINT[1] + 100.0,
    spacing: float = 25 * SCALE_FACTOR,
) -> np.ndarray:
    """Classic lawnmower/coverage sweep pattern (scaled x3, anchored at AUV spawn)."""
    start_xy = np.array(start_xy, dtype=float)
    ys = np.arange(ymin, ymax + 1e-9, spacing)

    wps = [start_xy, np.array([xmin, ys[0]])]  # start + entry

    # alternating sweeps with vertical step between lanes
    x_end = xmin
    for i, y in enumerate(ys):
        x_target = xmax if (i % 2 == 0) else xmin
        wps.append(np.array([x_target, y]))
        x_end = x_target
        if i < len(ys) - 1:
            wps.append(np.array([x_end, ys[i + 1]]))

    wps_xy = np.vstack(wps)
    z_col = np.full((wps_xy.shape[0], 1), SPAWN_POINT[2])
    return np.hstack((wps_xy, z_col))


def spiral_waypoints(
    center: Tuple[float, float] = (-35.0, -35.0),
    min_radius: float = 5.0,
    max_radius: float = 30.0,
    turns: float = 3.5,
    points_per_rev: int = 250,
    z_start: float = 0.0,
    z_end: float = -15.0,
) -> np.ndarray:
    """Expanding spiral (outward) maximizing observability per Yu et al.

    Radius grows linearly from min_radius to max_radius over the given number of turns
    while z descends linearly from z_start to z_end.
    """
    cx, cy = center
    total_points = max(points_per_rev, 1) * max(int(np.ceil(turns)), 1)
    theta = np.linspace(0.0, 2 * np.pi * turns, total_points + 1)
    r = np.linspace(min_radius, max_radius, theta.size)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = np.linspace(z_start, z_end, theta.size)
    return np.column_stack((x, y, z))


def concentric_circles_waypoints(
    center: Tuple[float, float] = (SPAWN_POINT[0], SPAWN_POINT[1]),
    radii: Iterable[float] = (10.0 * SCALE_FACTOR, 20.0 * SCALE_FACTOR, 30.0 * SCALE_FACTOR),
    points_per_circle: int = 400,
) -> np.ndarray:
    """Three circles at increasing radii; good for stable GDOP across scales."""
    cx, cy = center
    radii_list = list(radii)
    pts = []
    for r in radii_list:
        theta = np.linspace(0.0, 2 * np.pi, max(points_per_circle, 8), endpoint=True)
        pts.append(np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta))))
    xy = np.vstack(pts)
    z_col = np.full((xy.shape[0], 1), SPAWN_POINT[2])
    return np.hstack((xy, z_col))


def figure_eight_waypoints(
    a: float = 100.0 * SCALE_FACTOR,
    turns: int = 3,
    points_per_turn: int = 600,
) -> np.ndarray:
    """Lemniscate (figure-eight) path with smooth curvature."""
    total_points = max(points_per_turn, 10) * max(turns, 1)
    t = np.linspace(0.0, 2 * np.pi * turns, total_points + 1)
    x = a * np.cos(t) / (1.0 + np.sin(t) ** 2)
    y = a * np.sin(t) * np.cos(t) / (1.0 + np.sin(t) ** 2)
    x += SPAWN_POINT[0]
    y += SPAWN_POINT[1]
    z = np.full_like(x, SPAWN_POINT[2])
    return np.column_stack((x, y, z))


def build_trajectory(name: str, cfg: dict) -> np.ndarray:
    """Factory to build waypoint arrays (XY or XYZ) from a trajectory name and config overrides."""
    key = (name or "lawnmower").lower()
    max_seg = float(cfg.get("max_segment_length", 5.0))

    if key == "lawnmower":
        spacing = float(cfg.get("waypoint_spacing", 12.5 * SCALE_FACTOR))
        # Adjust extents relative to spawn when overridden
        xmin = float(cfg.get("xmin", SPAWN_POINT[0] - 45.0))
        xmax = float(cfg.get("xmax", SPAWN_POINT[0] + 45.0))
        ymin = float(cfg.get("ymin", SPAWN_POINT[1] - 45.0))
        ymax = float(cfg.get("ymax", SPAWN_POINT[1] + 45.0))
        start_xy = tuple(cfg.get("start_xy", (SPAWN_POINT[0], SPAWN_POINT[1])))
        traj = lawnmower_waypoints(
            start_xy=start_xy,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            spacing=spacing,
        )
        return _densify_path(_add_transit_to_start(traj), max_step=max_seg)

    if key == "spiral":
        traj = spiral_waypoints(
            center=tuple(cfg.get("spiral_center", (0.0, 0.0))),
            min_radius=float(cfg.get("spiral_min_radius", 5.0)),
            max_radius=float(cfg.get("spiral_max_radius", 30.0)),
            turns=float(cfg.get("spiral_turns", 3.5)),
            points_per_rev=int(cfg.get("spiral_points_per_rev", 250)),
            z_start=float(cfg.get("spiral_z_start", 0.0)),
            z_end=float(cfg.get("spiral_z_end", -15.0)),
        )
        return _densify_path(_add_transit_to_start(traj), max_step=max_seg)

    if key == "concentric":
        traj = concentric_circles_waypoints(
            center=tuple(cfg.get("concentric_center", (SPAWN_POINT[0], SPAWN_POINT[1]))),
            radii=cfg.get("concentric_radii", (
                10.0 * SCALE_FACTOR,
                20.0 * SCALE_FACTOR,
                30.0 * SCALE_FACTOR,
            )),
            points_per_circle=int(cfg.get("concentric_points_per_circle", 400)),
        )
        return _densify_path(_add_transit_to_start(traj), max_step=max_seg)

    if key in ("figure8", "figure-eight", "fig8"):
        traj = figure_eight_waypoints(
            a=float(cfg.get("figure8_scale", 20.0 * SCALE_FACTOR)),
            turns=int(cfg.get("figure8_turns", 3)),
            points_per_turn=int(cfg.get("figure8_points_per_turn", 300)),
        )
        return _densify_path(_add_transit_to_start(traj), max_step=max_seg)

    raise ValueError(f"Unknown trajectory '{name}'")
