"""Reusable trajectory generators for EKF simulations.

Includes:
- lawnmower (coverage sweeps)
- spiral (observability-focused, expanding radius)
- concentric circles (fixed-radius loops)
- figure-eight (lemniscate)
"""
import numpy as np
from typing import Iterable, Tuple


def lawnmower_waypoints(
    start_xy: Tuple[float, float] = (-15.0, -15.0),
    xmin: float = -15.0,
    xmax: float = 15.0,
    ymin: float = -15.0,
    ymax: float = 15.0,
    spacing: float = 12.5,
) -> np.ndarray:
    """Classic lawnmower/coverage sweep pattern."""
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

    return np.vstack(wps)


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
    center: Tuple[float, float] = (0.0, 0.0),
    radii: Iterable[float] = (10.0, 20.0, 30.0),
    points_per_circle: int = 400,
) -> np.ndarray:
    """Three circles at increasing radii; good for stable GDOP across scales."""
    cx, cy = center
    radii_list = list(radii)
    pts = []
    for r in radii_list:
        theta = np.linspace(0.0, 2 * np.pi, max(points_per_circle, 8), endpoint=True)
        pts.append(np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta))))
    return np.vstack(pts)


def figure_eight_waypoints(
    a: float = 20.0,
    turns: int = 3,
    points_per_turn: int = 300,
) -> np.ndarray:
    """Lemniscate (figure-eight) path with smooth curvature."""
    total_points = max(points_per_turn, 10) * max(turns, 1)
    t = np.linspace(0.0, 2 * np.pi * turns, total_points + 1)
    x = a * np.cos(t) / (1.0 + np.sin(t) ** 2)
    y = a * np.sin(t) * np.cos(t) / (1.0 + np.sin(t) ** 2)
    return np.column_stack((x, y))


def build_trajectory(name: str, cfg: dict) -> np.ndarray:
    """Factory to build waypoint arrays (XY or XYZ) from a trajectory name and config overrides."""
    key = (name or "lawnmower").lower()

    if key == "lawnmower":
        spacing = float(cfg.get("waypoint_spacing", 12.5))
        return lawnmower_waypoints(spacing=spacing)

    if key == "spiral":
        return spiral_waypoints(
            center=tuple(cfg.get("spiral_center", (0.0, 0.0))),
            min_radius=float(cfg.get("spiral_min_radius", 5.0)),
            max_radius=float(cfg.get("spiral_max_radius", 30.0)),
            turns=float(cfg.get("spiral_turns", 3.5)),
            points_per_rev=int(cfg.get("spiral_points_per_rev", 250)),
            z_start=float(cfg.get("spiral_z_start", 0.0)),
            z_end=float(cfg.get("spiral_z_end", -15.0)),
        )

    if key == "concentric":
        return concentric_circles_waypoints(
            center=tuple(cfg.get("concentric_center", (0.0, 0.0))),
            radii=cfg.get("concentric_radii", (10.0, 20.0, 30.0)),
            points_per_circle=int(cfg.get("concentric_points_per_circle", 400)),
        )

    if key in ("figure8", "figure-eight", "fig8"):
        return figure_eight_waypoints(
            a=float(cfg.get("figure8_scale", 20.0)),
            turns=int(cfg.get("figure8_turns", 3)),
            points_per_turn=int(cfg.get("figure8_points_per_turn", 300)),
        )

    raise ValueError(f"Unknown trajectory '{name}'")
