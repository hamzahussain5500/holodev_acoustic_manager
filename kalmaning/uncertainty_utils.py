# uncertainty_utils.py
import numpy as np


def covariance_ellipse_points(P_2x2, mean_xy, chi2_val=5.991, num_points=100):
    """
    Compute 2D ellipse points for a given 2x2 covariance and mean.

    Parameters
    ----------
    P_2x2 : (2,2) ndarray
        Covariance matrix for x,y.
    mean_xy : (2,) array_like
        Center of the ellipse [mx, my].
    chi2_val : float
        Chi-square value for desired confidence (2 dof).
        95% conf for 2D -> ~5.991, 99% -> ~9.21
    num_points : int
        Number of points for the ellipse curve.

    Returns
    -------
    xs, ys : (num_points,) ndarrays
        Coordinates of the ellipse.
    """
    P_2x2 = np.asarray(P_2x2)
    mean_xy = np.asarray(mean_xy)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(P_2x2)

    # Radii: sqrt(lambda_i * chi2_val)
    axes = np.sqrt(eigvals * chi2_val)

    # Parametric angle
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, N)

    # Scale and rotate
    ellipse = eigvecs @ (axes[:, None] * circle)  # (2,2) @ (2,N) -> (2,N)

    # Translate to mean
    ellipse[0, :] += mean_xy[0]
    ellipse[1, :] += mean_xy[1]

    return ellipse[0, :], ellipse[1, :]


def covariance_ellipsoid_mesh(P_3x3, mean_xyz, chi2_val=7.815, num_u=20, num_v=20):
    """
    Compute a 3D ellipsoid mesh for a given 3x3 covariance and mean.

    Parameters
    ----------
    P_3x3 : (3,3) ndarray
        Covariance of [x,y,z].
    mean_xyz : (3,) array_like
        Center of the ellipsoid [mx, my, mz].
    chi2_val : float
        Chi-square value for desired confidence (3 dof).
        95% conf for 3D -> ~7.815, 99% -> ~11.345
    num_u, num_v : int
        Resolution of the ellipsoid mesh in parametric angles.

    Returns
    -------
    X, Y, Z : (num_u, num_v) ndarrays
        Coordinates of the ellipsoid surface.
    """
    P_3x3 = np.asarray(P_3x3)
    mean_xyz = np.asarray(mean_xyz)

    eigvals, eigvecs = np.linalg.eigh(P_3x3)
    axes = np.sqrt(eigvals * chi2_val)  # radii along principal axes

    # Parametric angles
    u = np.linspace(0, 2 * np.pi, num_u)
    v = np.linspace(0, np.pi, num_v)
    uu, vv = np.meshgrid(u, v)

    # Unit sphere
    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)
    sphere = np.stack([xs, ys, zs], axis=0)  # (3, num_u, num_v)

    # Scale along principal axes: (3,1,1) * (3,Nu,Nv)
    ellipsoid_local = axes[:, None, None] * sphere

    # Rotate to global frame: eigvecs (3,3) @ (3,Nu,Nv)
    ellipsoid_global = eigvecs @ ellipsoid_local.reshape(3, -1)  # (3, Nu*Nv)
    ellipsoid_global = ellipsoid_global.reshape(3, xs.shape[0], xs.shape[1])

    X = ellipsoid_global[0, :, :] + mean_xyz[0]
    Y = ellipsoid_global[1, :, :] + mean_xyz[1]
    Z = ellipsoid_global[2, :, :] + mean_xyz[2]

    return X, Y, Z


def axis_uncertainty_bounds(P_3x3, mean_xyz, chi2_val=3.841):
    """
    Compute 1D positional bounds (x,y,z) at a chosen chi-square level.

    Parameters
    ----------
    P_3x3 : (3,3) ndarray
        Covariance of [x,y,z].
    mean_xyz : (3,) array_like
        Mean position.
    chi2_val : float
        Chi-square threshold for 1 dof (default 95% ≈ 3.841).

    Returns
    -------
    centers : (3,) ndarray
        Mean position components.
    half_widths : (3,) ndarray
        Half-widths for each axis (sqrt(var * chi2_val)).
    labels : tuple
        Axis labels ("x", "y", "z").
    """
    P_3x3 = np.asarray(P_3x3)
    mean_xyz = np.asarray(mean_xyz)
    variances = np.diag(P_3x3)
    half_widths = np.sqrt(variances * chi2_val)
    return mean_xyz, half_widths, ("x", "y", "z")
