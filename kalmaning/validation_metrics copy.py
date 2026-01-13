import numpy as np
from scipy.stats import chi2


def _safe_quadratic_form(e, P):
    """Compute e^T P^{-1} e safely; return np.nan on failure."""
    try:
        # Prefer solve over explicit inverse for stability
        sol = np.linalg.solve(P, e)
        return float(e.T @ sol)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse
        try:
            sol = np.linalg.pinv(P) @ e
            return float(e.T @ sol)
        except Exception:
            return np.nan


def chi2_bounds(dof, alpha=0.05, samples=1):
    """Return lower/upper chi-square bounds for a given dof and sample count."""
    lower = chi2.ppf(alpha / 2.0, dof * samples) / samples
    upper = chi2.ppf(1.0 - alpha / 2.0, dof * samples) / samples
    return lower, upper


def calculate_nees(errors, covariances, indices=None):
    """
    Compute NEES time series.

    errors: (N, n) array of estimation error (est - true)
    covariances: (N, n, n) array of state covariance matrices
    indices: optional iterable of state indices to evaluate (e.g., [0,1,2] for position)

    Returns: (N,) array of NEES values (np.nan where covariance is singular).
    """
    errs = np.asarray(errors)
    covs = np.asarray(covariances)
    if indices is not None:
        idx = np.asarray(indices, dtype=int)
        errs = errs[:, idx]
        covs = covs[:, idx][:, :, idx]
    n_samples = errs.shape[0]
    nees = np.zeros(n_samples)
    for k in range(n_samples):
        e = errs[k].reshape(-1, 1)
        P = covs[k]
        nees[k] = _safe_quadratic_form(e, P)
    return nees


def nees_consistency_test(nees_values, dof, alpha=0.05):
    """
    Consistency test for NEES.

    Returns dict with avg_nees, expected_nees, bounds, flag, and percent inside bounds.
    Bounds are for the average NEES over all samples.
    """
    nees_vals = np.asarray(nees_values)
    valid = np.isfinite(nees_vals)
    count = int(np.sum(valid))
    avg = float(np.nanmean(nees_vals)) if count > 0 else np.nan
    lower, upper = chi2_bounds(dof, alpha=alpha, samples=max(count, 1))
    is_consistent = bool(lower <= avg <= upper) if np.isfinite(avg) else False
    per_sample_lower, per_sample_upper = chi2.ppf(alpha / 2.0, dof), chi2.ppf(1.0 - alpha / 2.0, dof)
    inside = np.sum((nees_vals >= per_sample_lower) & (nees_vals <= per_sample_upper)) if count > 0 else 0
    percent_inside = (inside / count * 100.0) if count > 0 else np.nan
    return {
        "is_consistent": is_consistent,
        "avg_nees": avg,
        "expected_nees": dof,
        "lower_bound": lower,
        "upper_bound": upper,
        "percent_inside_bounds": percent_inside,
        "per_sample_bounds": (per_sample_lower, per_sample_upper),
        "num_samples": count,
    }


def calculate_nis(innovations, innovation_covariances):
    """
    Compute NIS time series for a batch of innovations.

    innovations: (N, m) array
    innovation_covariances: (N, m, m) array
    Returns: (N,) array of NIS values (np.nan on singular S).
    """
    innovs = np.asarray(innovations)
    covs = np.asarray(innovation_covariances)
    n_samples = innovs.shape[0]
    nis = np.zeros(n_samples)
    for k in range(n_samples):
        v = innovs[k].reshape(-1, 1)
        S = covs[k]
        nis[k] = _safe_quadratic_form(v, S)
    return nis


def nis_consistency_test(nis_values, dof, alpha=0.05):
    """
    Consistency test for NIS.
    Returns dict mirroring nees_consistency_test.
    """
    nis_vals = np.asarray(nis_values)
    valid = np.isfinite(nis_vals)
    count = int(np.sum(valid))
    avg = float(np.nanmean(nis_vals)) if count > 0 else np.nan
    lower, upper = chi2_bounds(dof, alpha=alpha, samples=max(count, 1))
    is_consistent = bool(lower <= avg <= upper) if np.isfinite(avg) else False
    per_sample_lower, per_sample_upper = chi2.ppf(alpha / 2.0, dof), chi2.ppf(1.0 - alpha / 2.0, dof)
    inside = np.sum((nis_vals >= per_sample_lower) & (nis_vals <= per_sample_upper)) if count > 0 else 0
    percent_inside = (inside / count * 100.0) if count > 0 else np.nan
    return {
        "is_consistent": is_consistent,
        "avg_nis": avg,
        "expected_nis": dof,
        "lower_bound": lower,
        "upper_bound": upper,
        "percent_inside_bounds": percent_inside,
        "per_sample_bounds": (per_sample_lower, per_sample_upper),
        "num_samples": count,
    }
