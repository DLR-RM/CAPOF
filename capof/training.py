"""Shared training helper for CAPOF policies.

Provides :func:`train_capof`, which encapsulates the GP fitting boilerplate
used by all examples (subsample the demonstrations, build the RBF kernel, fit
the GP, and wrap it in a :class:`~capof.capof.ContextAdaptivePolicyFramework`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from capof.capof import ContextAdaptivePolicyFramework, Vector
from capof.gp_scikit import GP


def train_capof(
    X: np.ndarray,
    Y: np.ndarray,
    goal_poses: np.ndarray,
    K_sp: Vector,
    K_gap: Vector,
    alpha: float,
    kernel_length_vec: Vector,
    c_dim: int,
    x_dim: int,
    phi_dim: int,
    x_dot_dim: int,
    phi_dot_dim: int,
    mu_add_dim: int,
    n_regression_points: int = 500,
    epi_reg: Optional[float] = None,
) -> ContextAdaptivePolicyFramework:
    """Fit a GP on the demonstrations and build a CAPOF policy.

    The demonstrations are subsampled to at most ``n_regression_points``
    evenly spaced points (by index), a fixed-length-scale RBF kernel is used
    (no kernel hyperparameter optimization), and the fitted GP is wrapped in a
    :class:`ContextAdaptivePolicyFramework`.

    Note: this function does **not** scale the hyperparameters. Callers pass
    already-scaled (absolute) values, e.g. ``alpha * max(var(Y))`` or
    ``kernel_length_vec * (max(X) - min(X))``.

    Args:
        X: (n_samples, n_features) demonstration inputs ``[c, x, phi]``.
        Y: (n_samples, n_targets) demonstration outputs ``[x_dot, phi_dot, mu_add]``.
        goal_poses: (n_poses, n_features) goal poses (last frame of each demo).
        K_sp: Stabilizing policy gains ``[K_sp_x, K_sp_phi]`` (phi entry only
            used when ``phi_dim > 0``).
        K_gap: Goal-attractor gains ``[K_gap_x, K_gap_phi]``.
        alpha: GP noise variance (absolute value).
        kernel_length_vec: RBF length scales (absolute values), one per feature.
        c_dim: Task-parameter dimension.
        x_dim: Position dimension.
        phi_dim: Orientation dimension.
        x_dot_dim: Position-velocity dimension.
        phi_dot_dim: Orientation-velocity dimension.
        mu_add_dim: Additional target dimension (e.g. gripper command).
        n_regression_points: Maximum number of evenly spaced regression points.
        epi_reg: Epistemic uncertainty regularization (default: ``alpha``).

    Returns:
        The fitted :class:`ContextAdaptivePolicyFramework` instance.
    """
    # Sample regression points evenly spaced (by data index) from the demos
    # (at most n_regression_points, or all points if fewer are available)
    n_points = min(n_regression_points, len(X))
    indices = np.round(np.linspace(0, len(X) - 1, n_points)).astype(int)

    # Create kernel and fit the GP (kernel hyperparameters kept fixed)
    kernel = RBF(length_scale=np.asarray(kernel_length_vec, dtype=float))
    model = GP(kernel=kernel, alpha=alpha, epi_reg=epi_reg, optimizer=None)
    model.fit(X[indices], Y[indices])

    # Save the (fixed) kernel length scales on the model
    model.l = model.kernel_.length_scale

    # Build the context-adaptive policy framework
    return ContextAdaptivePolicyFramework(
        model,
        K_sp,
        K_gap,
        model.l,
        c_dim=c_dim,
        x_dim=x_dim,
        phi_dim=phi_dim,
        x_dot_dim=x_dot_dim,
        phi_dot_dim=phi_dot_dim,
        mu_add_dim=mu_add_dim,
        goal_poses=goal_poses,
    )
