"""Adaptive modem (beacon) selection based on geometry metrics.

Implements lightweight GDOP-based gating for acoustic updates. Supports both 3D and
XY-only observability checks (the latter assumes depth is reliable).
"""
import itertools
from typing import Dict, List, Optional

import numpy as np


class AdaptiveModemManager:
    def __init__(
        self,
        modem_positions: np.ndarray,
        sigma_r: float = 0.1,
        gdop_xy_thresh: float = 8.0,
        gdop_3d_thresh: float = 12.0,
        min_dwell_steps: int = 50,
        switch_margin: float = 0.25,
        size_penalty: float = 0.0,
    ) -> None:
        self.modem_positions = np.asarray(modem_positions, dtype=float)
        self.sigma_r = float(sigma_r)
        self.gdop_xy_thresh = float(gdop_xy_thresh)
        self.gdop_3d_thresh = float(gdop_3d_thresh)
        self.min_dwell_steps = int(min_dwell_steps)
        self.switch_margin = float(switch_margin)
        self.size_penalty = float(size_penalty)

        self.active_indices: List[int] = list(range(self.modem_positions.shape[0]))
        self._dwell_counter = 0

    def update_positions(self, modem_positions: np.ndarray) -> None:
        """Optionally refresh modem positions (for moving beacons)."""
        self.modem_positions = np.asarray(modem_positions, dtype=float)
        n = self.modem_positions.shape[0]
        # Reset active set if size changes or indices are invalid
        if not self.active_indices or any(i >= n for i in self.active_indices):
            self.active_indices = list(range(n))
            self._dwell_counter = 0

    def _rank(self, H: np.ndarray) -> int:
        if H.size == 0:
            return 0
        try:
            s = np.linalg.svd(H, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0
        if s.size == 0:
            return 0
        tol = float(np.max(s)) * 1e-10
        return int(np.sum(s > tol))

    def _gdop_from_H(self, H: np.ndarray) -> float:
        if H.size == 0:
            return float("inf")
        try:
            F = (1.0 / (self.sigma_r * self.sigma_r)) * (H.T @ H)
            Finv = np.linalg.pinv(F)
        except Exception:
            return float("inf")
        tr = float(np.trace(Finv))
        if not np.isfinite(tr) or tr <= 0.0:
            return float("inf")
        return float(np.sqrt(tr))

    def _jacobian(self, p_xyz: np.ndarray, subset_idx: List[int]) -> np.ndarray:
        if len(subset_idx) == 0 or self.modem_positions.size == 0:
            return np.zeros((0, 3))
        p = np.asarray(p_xyz, dtype=float).reshape(3,)
        beacons = self.modem_positions[np.asarray(subset_idx, dtype=int)]
        diffs = p - beacons  # (m,3)
        dists = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-9
        return diffs / dists

    def _evaluate_subset(self, p_xyz: np.ndarray, subset_idx: List[int]) -> Dict[str, float]:
        H = self._jacobian(p_xyz, subset_idx)
        H_xy = H[:, :2] if H.size else np.zeros((0, 2))
        rank3 = self._rank(H)
        rank2 = self._rank(H_xy)
        gdop3 = self._gdop_from_H(H)
        gdop2 = self._gdop_from_H(H_xy) if H_xy.size else float("inf")
        return {
            "rank_xy": rank2,
            "gdop_xy": gdop2,
            "rank_3d": rank3,
            "gdop_3d": gdop3,
        }

    def decide(self, p_hat: np.ndarray, depth_available: bool) -> Dict[str, object]:
        n = self.modem_positions.shape[0]
        mode = "xy" if depth_available else "3d"
        req_rank = 2 if mode == "xy" else 3
        gdop_thresh = self.gdop_xy_thresh if mode == "xy" else self.gdop_3d_thresh
        allowed_sizes = [2, 3, 4] if mode == "xy" else [3, 4]

        candidates = []
        for r in allowed_sizes:
            if r > n:
                continue
            for combo in itertools.combinations(range(n), r):
                metrics = self._evaluate_subset(p_hat, list(combo))
                candidates.append({
                    "idx": list(combo),
                    **metrics,
                    "size": r,
                })

        def is_valid(c):
            rank_ok = (c["rank_xy"] if mode == "xy" else c["rank_3d"]) >= req_rank
            gdop_val = c["gdop_xy"] if mode == "xy" else c["gdop_3d"]
            return rank_ok and np.isfinite(gdop_val) and (gdop_val <= gdop_thresh)

        valid = [c for c in candidates if is_valid(c)]

        def best_of(pool):
            if not pool:
                return None
            def score(c):
                gdop_val = c["gdop_xy"] if mode == "xy" else c["gdop_3d"]
                return self.size_penalty * c["size"] + gdop_val

            pool_sorted = sorted(pool, key=lambda c: (score(c), c["size"]))
            return pool_sorted[0]

        best_valid = best_of(valid)
        # If nothing meets threshold, fall back to overall best by gdop even if above threshold
        fallback = best_of(candidates)
        chosen = best_valid or fallback

        # Evaluate current set
        current = None
        if self.active_indices:
            current = {"idx": list(self.active_indices)}
            current.update(self._evaluate_subset(p_hat, self.active_indices))
            current["size"] = len(self.active_indices)

        reason = "init"
        switch = False

        def meets_req(c):
            if c is None:
                return False
            if c["size"] not in allowed_sizes:
                return False
            rank_val = c["rank_xy"] if mode == "xy" else c["rank_3d"]
            gdop_val = c["gdop_xy"] if mode == "xy" else c["gdop_3d"]
            return rank_val >= req_rank and np.isfinite(gdop_val) and gdop_val <= gdop_thresh

        current_ok = meets_req(current)

        if current_ok:
            self._dwell_counter += 1
        else:
            self._dwell_counter = 0

        if not current_ok:
            switch = True
            reason = "current_invalid"
        elif best_valid is None:
            switch = False
            reason = "no_valid_candidates"
        elif chosen is None:
            switch = False
            reason = "no_candidates"
        elif current and chosen["idx"] == current["idx"]:
            switch = False
            reason = "stable"
        else:
            if self._dwell_counter < self.min_dwell_steps:
                switch = False
                reason = "dwell"
            else:
                current_gdop = current["gdop_xy"] if mode == "xy" else current["gdop_3d"]
                candidate_gdop = chosen["gdop_xy"] if mode == "xy" else chosen["gdop_3d"]
                improvement = current_gdop - candidate_gdop
                if improvement > self.switch_margin:
                    switch = True
                    reason = "better_gdop"
                else:
                    switch = False
                    reason = "within_margin"

        if switch and chosen is not None:
            self.active_indices = chosen["idx"]
            self._dwell_counter = 0
        elif current is None and chosen is not None:
            self.active_indices = chosen["idx"]
            self._dwell_counter = 0

        active = self.active_indices
        metrics = self._evaluate_subset(p_hat, active)

        return {
            "active_indices": list(active),
            "rank_xy": metrics["rank_xy"],
            "gdop_xy": metrics["gdop_xy"],
            "rank_3d": metrics["rank_3d"],
            "gdop_3d": metrics["gdop_3d"],
            "mode": mode,
            "reason": reason,
        }
